"""Offline tests for mtg_recommender.deck_profile + theme_classifier.

The fixtures stand up a miniature oracle_tags hierarchy with:

  removal/
    spot-removal
    mass-removal
  ramp/
    mana-rock
    mana-dork
  card-advantage/
    cantrip
    draw-outlet
  cycle/              ← in default blocklist; discover_themes drops it
    cycle-child-1

Each tag doc carries a packed-float32 `embedding`. Deck cards carry
`tags` arrays referencing those slugs, plus their own `text_embedding`
in the same 4-dim space so the multi-theme tiebreak has something
meaningful to compare.

Test classes:
    TestResolveDeckCards        name lookup, missing, dedup, case-insensitive
    TestCollectTagUniverse      union across deck
    TestDiscoverThemes          blocklist, min_children, coverage window,
                                 missing embeddings handled
    TestClassifyCard            0 / 1 / 2+ candidates; text-embedding
                                 tiebreak; missing embedding falls back
    TestClassifyDeck            per-theme membership + unassigned bucket
    TestBuildDeckProfile        end-to-end on a seeded mongomock cluster
    TestRenderProfile           truncation marker, cluster display shape
    TestMainCLI                 argparse: filters pipe through, --theme-
                                 blocklist override, --file reads decklist
"""
from __future__ import annotations

import io
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory

import mongomock
import numpy as np

from mtg_recommender import deck_profile as dp
from mtg_recommender import embeddings as emb
from mtg_recommender import storage
from mtg_recommender import theme_classifier as tc


def _pack(vec: list[float]) -> bytes:
    """Pack a vector in the storage format `_unpack_embedding` reads."""
    return emb._pack_embedding(np.array(vec, dtype=np.float32))


# ---------------------------------------------------------------------------
# Fixture tag hierarchy
# ---------------------------------------------------------------------------
#
# Three thematic top-levels (removal, ramp, card-advantage), each with
# two children. One blocklisted top-level (`cycle`, in DEFAULT_BLOCKLIST)
# to exercise the filter. Each theme sits near a different basis vector
# so the oracle-text tiebreak behaves predictably.

TAG_HIERARCHY = [
    # removal theme — near [1, 0, 0, 0]
    {"_id": "removal",            "parent_slugs": [], "child_slugs": ["spot-removal", "mass-removal"],
     "embedding": _pack([1.00, 0.00, 0.00, 0.00])},
    {"_id": "spot-removal",       "parent_slugs": ["removal"], "child_slugs": [],
     "embedding": _pack([0.95, 0.05, 0.00, 0.00])},
    {"_id": "mass-removal",       "parent_slugs": ["removal"], "child_slugs": [],
     "embedding": _pack([0.95, -0.05, 0.00, 0.00])},

    # ramp theme — near [0, 1, 0, 0]
    {"_id": "ramp",               "parent_slugs": [], "child_slugs": ["mana-rock", "mana-dork"],
     "embedding": _pack([0.00, 1.00, 0.00, 0.00])},
    {"_id": "mana-rock",          "parent_slugs": ["ramp"], "child_slugs": [],
     "embedding": _pack([0.05, 0.95, 0.00, 0.00])},
    {"_id": "mana-dork",          "parent_slugs": ["ramp"], "child_slugs": [],
     "embedding": _pack([-0.05, 0.95, 0.00, 0.00])},

    # card-advantage theme — near [0, 0, 1, 0]
    {"_id": "card-advantage",     "parent_slugs": [], "child_slugs": ["cantrip", "draw-outlet"],
     "embedding": _pack([0.00, 0.00, 1.00, 0.00])},
    {"_id": "cantrip",            "parent_slugs": ["card-advantage"], "child_slugs": [],
     "embedding": _pack([0.00, 0.05, 0.95, 0.00])},
    {"_id": "draw-outlet",        "parent_slugs": ["card-advantage"], "child_slugs": [],
     "embedding": _pack([0.00, -0.05, 0.95, 0.00])},

    # cycle — top-level BUT in DEFAULT_BLOCKLIST; filter should drop it.
    {"_id": "cycle",              "parent_slugs": [], "child_slugs": ["cycle-child-1", "cycle-child-2"],
     "embedding": _pack([0.00, 0.00, 0.00, 1.00])},
    {"_id": "cycle-child-1",      "parent_slugs": ["cycle"], "child_slugs": [],
     "embedding": _pack([0.00, 0.00, 0.00, 1.00])},
    {"_id": "cycle-child-2",      "parent_slugs": ["cycle"], "child_slugs": [],
     "embedding": _pack([0.00, 0.00, 0.00, 1.00])},
]


class _MongoBackedTestCase(unittest.TestCase):
    """Fresh mongomock per test; indexes created so names lookups work."""

    def setUp(self):
        storage.reset_client(mongomock.MongoClient())
        storage.ensure_indexes()
        self.cards = storage.cards_collection()
        self.tags = storage.tags_collection()
        # The theme filter's default min_coverage=100 wants ≥100 cards
        # per theme — way more than any test fixture builds. All tests
        # that call discover_themes / build_deck_profile override to a
        # permissive filter.
        self.permissive = tc.ThemeFilter(
            min_children=2, min_coverage=1, max_coverage=10_000,
            blocklist=tc.DEFAULT_BLOCKLIST,
        )

    def tearDown(self):
        storage.reset_client(None)

    def _seed_hierarchy(self):
        self.tags.insert_many([dict(t) for t in TAG_HIERARCHY])

    def _seed_card(
        self, sid, name, tags, text_embedding=None,
    ):
        doc = {
            "_id": sid, "name": name, "names": [name.lower()],
            "tags": list(tags),
        }
        if text_embedding is not None:
            doc["text_embedding"] = _pack(text_embedding)
        self.cards.insert_one(doc)


# ---------------------------------------------------------------------------
# resolve_deck_cards
# ---------------------------------------------------------------------------

class TestResolveDeckCards(_MongoBackedTestCase):

    def test_resolves_known_names(self):
        self._seed_card("id-bolt", "Lightning Bolt", ["spot-removal"])
        self._seed_card("id-sol", "Sol Ring", ["mana-rock"])
        docs, missing = dp.resolve_deck_cards(
            ["Lightning Bolt", "Sol Ring"], self.cards
        )
        self.assertEqual([d["_id"] for d in docs], ["id-bolt", "id-sol"])
        self.assertEqual(missing, [])

    def test_missing_names_collected_in_input_order(self):
        self._seed_card("id-bolt", "Lightning Bolt", ["spot-removal"])
        _, missing = dp.resolve_deck_cards(
            ["Nonexistent 1", "Lightning Bolt", "Nonexistent 2"], self.cards
        )
        self.assertEqual(missing, ["Nonexistent 1", "Nonexistent 2"])

    def test_case_insensitive_lookup(self):
        self._seed_card("id-bolt", "Lightning Bolt", ["spot-removal"])
        docs, _ = dp.resolve_deck_cards(["LIGHTNING BOLT"], self.cards)
        self.assertEqual([d["_id"] for d in docs], ["id-bolt"])

    def test_duplicate_input_names_dedupe_by_id(self):
        self._seed_card("id-bolt", "Lightning Bolt", ["spot-removal"])
        docs, _ = dp.resolve_deck_cards(
            ["Lightning Bolt", "Lightning Bolt"], self.cards
        )
        self.assertEqual(len(docs), 1)

    def test_empty_input(self):
        docs, missing = dp.resolve_deck_cards([], self.cards)
        self.assertEqual(docs, [])
        self.assertEqual(missing, [])


# ---------------------------------------------------------------------------
# collect_tag_universe
# ---------------------------------------------------------------------------

class TestCollectTagUniverse(unittest.TestCase):

    def test_unions_and_sorts(self):
        docs = [
            {"tags": ["spot-removal", "burn-any"]},
            {"tags": ["mana-rock", "spot-removal"]},
            {"tags": []},
        ]
        self.assertEqual(
            dp.collect_tag_universe(docs),
            ["burn-any", "mana-rock", "spot-removal"],
        )

    def test_handles_missing_tags_field(self):
        docs = [{"name": "no tags here"}, {"tags": ["a"]}]
        self.assertEqual(dp.collect_tag_universe(docs), ["a"])


# ---------------------------------------------------------------------------
# discover_themes
# ---------------------------------------------------------------------------

class TestDiscoverThemes(_MongoBackedTestCase):

    def test_blocklisted_top_level_is_dropped(self):
        # The `cycle` top-level is in DEFAULT_BLOCKLIST — must not appear
        # in discovered themes even though it has 2 children and would
        # otherwise clear the coverage filter.
        self._seed_hierarchy()
        self._seed_card("c1", "Card 1", ["cycle-child-1"])
        self._seed_card("c2", "Card 2", ["cycle-child-2"])
        themes = tc.discover_themes(
            self.cards, self.tags, theme_filter=self.permissive,
        )
        theme_names = {t.name for t in themes}
        self.assertNotIn("cycle", theme_names)

    def test_min_children_filter_drops_point_tags(self):
        # A top-level with 0 or 1 children isn't a theme umbrella.
        self.tags.insert_one({
            "_id": "lonely-top", "parent_slugs": [], "child_slugs": [],
            "embedding": _pack([1.0, 0.0, 0.0, 0.0]),
        })
        self._seed_card("c1", "C1", ["lonely-top"])
        themes = tc.discover_themes(
            self.cards, self.tags, theme_filter=self.permissive,
        )
        self.assertNotIn("lonely-top", {t.name for t in themes})

    def test_coverage_window_filters_too_few_cards(self):
        # Fixture themes cover 2 cards each (via their subtree); set a
        # min_coverage of 10 and nothing survives.
        self._seed_hierarchy()
        self._seed_card("c1", "C1", ["spot-removal"])
        self._seed_card("c2", "C2", ["mana-rock"])
        filt = tc.ThemeFilter(
            min_children=2, min_coverage=10, max_coverage=10_000,
            blocklist=tc.DEFAULT_BLOCKLIST,
        )
        themes = tc.discover_themes(self.cards, self.tags, theme_filter=filt)
        self.assertEqual(themes, [])

    def test_coverage_window_filters_too_many_cards(self):
        # Dense fixture: a theme that covers 5 cards; cap max at 3.
        self._seed_hierarchy()
        for i in range(5):
            self._seed_card(f"c{i}", f"C{i}", ["spot-removal"])
        filt = tc.ThemeFilter(
            min_children=2, min_coverage=1, max_coverage=3,
            blocklist=tc.DEFAULT_BLOCKLIST,
        )
        themes = tc.discover_themes(self.cards, self.tags, theme_filter=filt)
        self.assertNotIn("removal", {t.name for t in themes})

    def test_surviving_themes_sorted_by_coverage_desc(self):
        self._seed_hierarchy()
        # 3 cards tagged with ramp, 1 with removal, 2 with card-advantage.
        for i in range(3):
            self._seed_card(f"ramp-{i}", f"Ramp {i}", ["mana-rock"])
        self._seed_card("rem-0", "Rem 0", ["spot-removal"])
        for i in range(2):
            self._seed_card(f"draw-{i}", f"Draw {i}", ["cantrip"])
        themes = tc.discover_themes(
            self.cards, self.tags, theme_filter=self.permissive,
        )
        # Expect ramp (3) → card-advantage (2) → removal (1).
        self.assertEqual(
            [t.name for t in themes],
            ["ramp", "card-advantage", "removal"],
        )

    def test_representative_is_unit_length(self):
        self._seed_hierarchy()
        self._seed_card("c1", "C1", ["spot-removal"])
        themes = tc.discover_themes(
            self.cards, self.tags, theme_filter=self.permissive,
        )
        for theme in themes:
            self.assertAlmostEqual(
                float(np.linalg.norm(theme.representative)), 1.0, places=5,
            )

    def test_subtree_contains_descendants(self):
        self._seed_hierarchy()
        self._seed_card("c1", "C1", ["spot-removal"])
        themes = tc.discover_themes(
            self.cards, self.tags, theme_filter=self.permissive,
        )
        removal = next(t for t in themes if t.name == "removal")
        self.assertEqual(
            removal.tag_slugs,
            frozenset({"removal", "spot-removal", "mass-removal"}),
        )

    def test_subtree_without_embeddings_drops_theme(self):
        # A top-level whose children have no `embedding` field → the
        # representative is None → the theme is skipped. Guards against
        # a half-populated cluster producing junk zero-vector themes.
        self.tags.insert_many([
            {"_id": "x", "parent_slugs": [], "child_slugs": ["xa", "xb"]},
            {"_id": "xa", "parent_slugs": ["x"], "child_slugs": []},
            {"_id": "xb", "parent_slugs": ["x"], "child_slugs": []},
        ])
        self._seed_card("c1", "C1", ["xa"])
        themes = tc.discover_themes(
            self.cards, self.tags, theme_filter=self.permissive,
        )
        self.assertNotIn("x", {t.name for t in themes})


# ---------------------------------------------------------------------------
# classify_card
# ---------------------------------------------------------------------------

class TestClassifyCard(unittest.TestCase):

    def _theme(self, name, tag_slugs, representative):
        return tc.Theme(
            name=name,
            tag_slugs=frozenset(tag_slugs),
            representative=np.array(representative, dtype=np.float64),
            card_coverage=1,
        )

    def test_zero_candidates_returns_none(self):
        themes = [self._theme("removal", ["spot-removal"], [1, 0, 0, 0])]
        got = tc.classify_card(["unrelated-tag"], None, themes)
        self.assertIsNone(got)

    def test_single_candidate_returns_it(self):
        themes = [self._theme("removal", ["spot-removal"], [1, 0, 0, 0])]
        got = tc.classify_card(["spot-removal"], None, themes)
        self.assertEqual(got.name, "removal")

    def test_multi_candidate_text_embedding_tiebreak(self):
        # Card has tags from both removal and ramp. Its text embedding
        # points at the ramp axis → ramp should win the tiebreak.
        themes = [
            self._theme("removal", ["spot-removal"], [1, 0, 0, 0]),
            self._theme("ramp",    ["mana-rock"],    [0, 1, 0, 0]),
        ]
        got = tc.classify_card(
            ["spot-removal", "mana-rock"],
            np.array([0.1, 0.9, 0.0, 0.0]),
            themes,
        )
        self.assertEqual(got.name, "ramp")

    def test_multi_candidate_no_text_embedding_falls_back_to_coverage(self):
        # Both themes match; neither card_text is available. Pick the
        # one with higher card_coverage so the degrade is deterministic.
        removal = tc.Theme(
            name="removal", tag_slugs=frozenset({"spot-removal"}),
            representative=np.array([1, 0, 0, 0], dtype=np.float64),
            card_coverage=10,
        )
        ramp = tc.Theme(
            name="ramp", tag_slugs=frozenset({"mana-rock"}),
            representative=np.array([0, 1, 0, 0], dtype=np.float64),
            card_coverage=100,
        )
        got = tc.classify_card(
            ["spot-removal", "mana-rock"], None, [removal, ramp]
        )
        self.assertEqual(got.name, "ramp")

    def test_empty_tags_returns_none(self):
        themes = [self._theme("x", ["y"], [1, 0, 0, 0])]
        self.assertIsNone(tc.classify_card([], None, themes))
        self.assertIsNone(tc.classify_card(None, None, themes))


# ---------------------------------------------------------------------------
# classify_deck
# ---------------------------------------------------------------------------

class TestClassifyDeck(unittest.TestCase):

    def _theme(self, name, tag_slugs, rep):
        return tc.Theme(
            name=name,
            tag_slugs=frozenset(tag_slugs),
            representative=np.array(rep, dtype=np.float64),
            card_coverage=1,
        )

    def test_distributes_cards_across_themes(self):
        themes = [
            self._theme("removal", ["spot-removal"], [1, 0, 0, 0]),
            self._theme("ramp",    ["mana-rock"],    [0, 1, 0, 0]),
        ]
        deck = [
            {"_id": "a", "tags": ["spot-removal"]},
            {"_id": "b", "tags": ["mana-rock"]},
            {"_id": "c", "tags": ["unrelated"]},
        ]
        by_theme, unassigned = tc.classify_deck(deck, themes)
        self.assertEqual(set(by_theme), {"removal", "ramp"})
        self.assertEqual(by_theme["removal"], ["a"])
        self.assertEqual(by_theme["ramp"], ["b"])
        self.assertEqual(unassigned, ["c"])

    def test_empty_themes_everything_unassigned(self):
        deck = [{"_id": "a", "tags": ["x"]}, {"_id": "b", "tags": ["y"]}]
        by_theme, unassigned = tc.classify_deck(deck, [])
        self.assertEqual(by_theme, {})
        self.assertEqual(unassigned, ["a", "b"])


# ---------------------------------------------------------------------------
# build_deck_profile — end-to-end
# ---------------------------------------------------------------------------

class TestBuildDeckProfile(_MongoBackedTestCase):

    def _seed_full_deck(self):
        self._seed_hierarchy()
        # Three single-theme cards + one multi-theme card.
        self._seed_card(
            "id-bolt", "Lightning Bolt", ["spot-removal"],
            text_embedding=[1.0, 0.0, 0.0, 0.0],
        )
        self._seed_card(
            "id-wrath", "Wrath of God", ["mass-removal"],
            text_embedding=[0.9, 0.1, 0.0, 0.0],
        )
        self._seed_card(
            "id-sol", "Sol Ring", ["mana-rock"],
            text_embedding=[0.0, 1.0, 0.0, 0.0],
        )
        self._seed_card(
            "id-brainstorm", "Brainstorm", ["cantrip"],
            text_embedding=[0.0, 0.0, 1.0, 0.0],
        )
        # Multi-theme card: has tags from both removal AND ramp. Its
        # text embedding sits on the ramp axis → ramp wins the tiebreak.
        self._seed_card(
            "id-ambiguous", "Hybrid Card", ["spot-removal", "mana-rock"],
            text_embedding=[0.1, 0.9, 0.0, 0.0],
        )

    def test_profile_reports_resolved_deck_and_missing(self):
        self._seed_full_deck()
        profile = dp.build_deck_profile(
            ["Lightning Bolt", "Wrath of God", "Nothing Here",
             "Sol Ring", "Brainstorm", "Hybrid Card"],
            cards_coll=self.cards, tags_coll=self.tags,
            theme_filter=self.permissive,
        )
        self.assertEqual(
            set(profile.deck_card_ids),
            {"id-bolt", "id-wrath", "id-sol", "id-brainstorm", "id-ambiguous"},
        )
        self.assertEqual(profile.missing_names, ("Nothing Here",))

    def test_three_themes_catch_the_cards(self):
        self._seed_full_deck()
        profile = dp.build_deck_profile(
            ["Lightning Bolt", "Wrath of God", "Sol Ring",
             "Brainstorm", "Hybrid Card"],
            cards_coll=self.cards, tags_coll=self.tags,
            theme_filter=self.permissive,
        )
        self.assertEqual(
            {c.label for c in profile.clusters},
            {"removal", "ramp", "card-advantage"},
        )

    def test_ambiguous_card_goes_to_ramp_via_text_tiebreak(self):
        self._seed_full_deck()
        profile = dp.build_deck_profile(
            ["Lightning Bolt", "Wrath of God", "Sol Ring",
             "Brainstorm", "Hybrid Card"],
            cards_coll=self.cards, tags_coll=self.tags,
            theme_filter=self.permissive,
        )
        by_label = {c.label: c for c in profile.clusters}
        self.assertIn("id-ambiguous", by_label["ramp"].deck_card_ids)
        self.assertNotIn("id-ambiguous", by_label["removal"].deck_card_ids)

    def test_cluster_tags_reflect_what_the_deck_actually_brought(self):
        # Deck only tagged with `spot-removal` from the removal theme —
        # `mass-removal` is in the theme's subtree but no card used it,
        # so the cluster's tags list omits it.
        self._seed_hierarchy()
        self._seed_card("id-bolt", "Lightning Bolt", ["spot-removal"],
                        text_embedding=[1.0, 0.0, 0.0, 0.0])
        profile = dp.build_deck_profile(
            ["Lightning Bolt"],
            cards_coll=self.cards, tags_coll=self.tags,
            theme_filter=self.permissive,
        )
        removal = next(c for c in profile.clusters if c.label == "removal")
        self.assertEqual(removal.tags, ("spot-removal",))

    def test_unassigned_card_captured_separately(self):
        self._seed_hierarchy()
        self._seed_card("id-unknown", "Unknown", ["unrelated-tag"])
        profile = dp.build_deck_profile(
            ["Unknown"],
            cards_coll=self.cards, tags_coll=self.tags,
            theme_filter=self.permissive,
        )
        self.assertEqual(profile.unassigned_card_ids, ("id-unknown",))
        self.assertEqual(profile.clusters, ())

    def test_noise_tags_are_deck_tags_outside_any_theme(self):
        # Fixture doesn't include "random-flavor" in any theme subtree.
        self._seed_hierarchy()
        self._seed_card("id-x", "X", ["spot-removal", "random-flavor"],
                        text_embedding=[1.0, 0.0, 0.0, 0.0])
        profile = dp.build_deck_profile(
            ["X"],
            cards_coll=self.cards, tags_coll=self.tags,
            theme_filter=self.permissive,
        )
        self.assertIn("random-flavor", profile.noise_tags)
        self.assertNotIn("spot-removal", profile.noise_tags)

    def test_empty_deck_returns_empty_profile(self):
        self._seed_hierarchy()
        profile = dp.build_deck_profile(
            [], cards_coll=self.cards, tags_coll=self.tags,
            theme_filter=self.permissive,
        )
        self.assertEqual(profile.deck_card_ids, ())
        self.assertEqual(profile.clusters, ())
        self.assertEqual(profile.unassigned_card_ids, ())


# ---------------------------------------------------------------------------
# _render_profile
# ---------------------------------------------------------------------------

class TestRenderProfile(unittest.TestCase):

    def _make_profile(self, cluster_tags, cluster_card_ids, *, unassigned=()):
        return dp.DeckProfile(
            deck_card_ids=tuple(cluster_card_ids) + tuple(unassigned),
            missing_names=(),
            tag_universe=tuple(sorted(cluster_tags)),
            clusters=(
                dp.DeckCluster(
                    label=cluster_tags[0],
                    tags=tuple(cluster_tags),
                    centroid=np.array([1.0, 0.0, 0.0, 0.0]),
                    deck_card_ids=tuple(cluster_card_ids),
                ),
            ),
            noise_tags=(),
            unassigned_card_ids=tuple(unassigned),
        )

    def test_truncation_marker_appears_when_items_exceed_limit(self):
        profile = self._make_profile(
            cluster_tags=["a", "b", "c", "d", "e"],
            cluster_card_ids=["id-1", "id-2", "id-3"],
        )
        text = dp._render_profile(
            profile,
            name_lookup={f"id-{i}": f"Card {i}" for i in (1, 2, 3)},
            limit=2,
        )
        self.assertIn("(+3 more)", text)
        self.assertIn("(+1 more)", text)

    def test_no_truncation_marker_when_within_limit(self):
        profile = self._make_profile(["a", "b"], ["id-1"])
        text = dp._render_profile(
            profile, name_lookup={"id-1": "Card 1"}, limit=10
        )
        self.assertNotIn("more)", text)

    def test_unassigned_cards_listed(self):
        profile = self._make_profile(
            cluster_tags=["a"],
            cluster_card_ids=["id-1"],
            unassigned=("id-99",),
        )
        text = dp._render_profile(
            profile, name_lookup={"id-1": "Card 1", "id-99": "Mystery"}, limit=10
        )
        self.assertIn("unassigned cards: Mystery", text)
        self.assertIn("1 card(s) unassigned", text)


# ---------------------------------------------------------------------------
# main() CLI
# ---------------------------------------------------------------------------

def _run_cli(argv: list[str]) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        rc = dp.main(argv)
    return rc, out.getvalue(), err.getvalue()


class TestMainCLI(_MongoBackedTestCase):

    def _seed_cli_deck(self):
        self._seed_hierarchy()
        self._seed_card("id-bolt", "Lightning Bolt", ["spot-removal"],
                        text_embedding=[1.0, 0.0, 0.0, 0.0])
        self._seed_card("id-sol", "Sol Ring", ["mana-rock"],
                        text_embedding=[0.0, 1.0, 0.0, 0.0])
        self._seed_card("id-brainstorm", "Brainstorm", ["cantrip"],
                        text_embedding=[0.0, 0.0, 1.0, 0.0])

    def test_prints_cluster_summary(self):
        self._seed_cli_deck()
        rc, out, _ = _run_cli([
            "Lightning Bolt", "Sol Ring", "Brainstorm",
            # Override coverage so fixture themes survive the default
            # min_coverage=100 floor.
            "--min-coverage", "1",
        ])
        self.assertEqual(rc, 0)
        self.assertIn("resolved 3 unique cards", out)
        self.assertIn("Lightning Bolt", out)
        self.assertIn("theme ", out)

    def test_file_input_reads_decklist(self):
        self._seed_cli_deck()
        with TemporaryDirectory() as d:
            deck = Path(d) / "deck.txt"
            deck.write_text(
                "# commander\n"
                "1 Lightning Bolt\n"
                "1 Sol Ring\n"
                "1 Brainstorm\n",
                encoding="utf-8",
            )
            rc, out, _ = _run_cli([
                "--file", str(deck), "--min-coverage", "1",
            ])
        self.assertEqual(rc, 0)
        self.assertIn("resolved 3 unique cards", out)

    def test_theme_blocklist_override_unblocks_cycle(self):
        # Pass --theme-blocklist "" to clear the default blocklist.
        # The fixture's `cycle` top-level then survives filtering.
        self._seed_hierarchy()
        self._seed_card("id-c1", "Cycle Card 1", ["cycle-child-1"],
                        text_embedding=[0.0, 0.0, 0.0, 1.0])
        self._seed_card("id-c2", "Cycle Card 2", ["cycle-child-2"],
                        text_embedding=[0.0, 0.0, 0.0, 1.0])
        rc, out, _ = _run_cli([
            "Cycle Card 1", "Cycle Card 2",
            "--min-coverage", "1", "--theme-blocklist", "",
        ])
        self.assertEqual(rc, 0)
        self.assertIn("theme 'cycle'", out)

    def test_min_children_pipes_through(self):
        self._seed_cli_deck()
        rc, out, _ = _run_cli([
            "Lightning Bolt", "Sol Ring", "Brainstorm",
            "--min-coverage", "1",
            "--min-children", "10",  # no theme has 10 children → zero survive
        ])
        self.assertEqual(rc, 0)
        self.assertIn("0 theme(s) matched", out)

    def test_missing_names_surfaced_in_output(self):
        self._seed_cli_deck()
        rc, out, _ = _run_cli([
            "Lightning Bolt", "No Such Card", "--min-coverage", "1",
        ])
        self.assertEqual(rc, 0)
        self.assertIn("missing: No Such Card", out)

    def test_empty_input_errors(self):
        with self.assertRaises(SystemExit):
            _run_cli([])


if __name__ == "__main__":
    unittest.main()
