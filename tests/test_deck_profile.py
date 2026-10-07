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
    TestMergeSimilarThemes      threshold off/negative is a no-op; similar
                                 themes merge with higher-coverage keeper;
                                 chain merging via union-find; themes
                                 without card_vectors stay singleton
    TestTopLevelLabelling       _top_ancestors memoisation + walk-up;
                                 most-frequent voter dedupes per card;
                                 blocklist respected; fallback label
    TestClusterCardsByVector    HDBSCAN on card_vector; cards without a
                                 vector skipped; degenerate inputs don't
                                 crash
    TestBuildClusterProfile     end-to-end cluster mode: 2 themes labelled
                                 by top-level vote, noise_tags always
                                 empty, missing-vector cards unassigned,
                                 fallback label when all top-levels blocked
    TestBuildDeckProfile        end-to-end on a seeded mongomock cluster
                                 (incl. default-merge and disabled-merge)
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
from unittest.mock import patch

import mongomock
import numpy as np

from mtg_recommender import card_clusterer as cc
from mtg_recommender import deck_profile as dp
from mtg_recommender import edhrec_fetch as edh
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
# merge_similar_themes
# ---------------------------------------------------------------------------

class TestMergeSimilarThemes(unittest.TestCase):
    """The merge pass is a pure function over three plain dicts, so
    these tests inject hand-built inputs and assert on the output
    shape. No Mongo involved.
    """

    def _theme(self, name, coverage):
        # tag_slugs and representative don't matter here — the merge
        # pass only reads coverage for the keeper tie-break. Give the
        # representative something non-zero so it's a valid Theme.
        return tc.Theme(
            name=name,
            tag_slugs=frozenset({name}),
            representative=np.array([1.0, 0.0, 0.0, 0.0]),
            card_coverage=coverage,
        )

    def _packed_card(self, card_vector):
        """Mongo doc shape with a packed card_vector."""
        return {"card_vector": emb._pack_embedding(np.array(card_vector, dtype=np.float32))}

    def test_zero_threshold_is_no_op(self):
        # Even blatantly similar profiles stay separate.
        per_theme = {"a": ["c1"], "b": ["c2"]}
        themes_by_name = {"a": self._theme("a", 10), "b": self._theme("b", 20)}
        cards_by_id = {
            "c1": self._packed_card([1.0, 0.0, 0.0, 0.0]),
            "c2": self._packed_card([1.0, 0.0, 0.0, 0.0]),
        }
        got, merge_map = tc.merge_similar_themes(
            per_theme, themes_by_name, cards_by_id, threshold=0.0,
        )
        self.assertEqual(got, per_theme)
        self.assertEqual(merge_map, {"a": "a", "b": "b"})

    def test_negative_threshold_disables_pass(self):
        per_theme = {"a": ["c1"], "b": ["c2"]}
        themes_by_name = {"a": self._theme("a", 10), "b": self._theme("b", 20)}
        cards_by_id = {
            "c1": self._packed_card([1.0, 0.0, 0.0, 0.0]),
            "c2": self._packed_card([1.0, 0.0, 0.0, 0.0]),
        }
        got, _ = tc.merge_similar_themes(
            per_theme, themes_by_name, cards_by_id, threshold=-1.0,
        )
        self.assertEqual(got, per_theme)

    def test_two_similar_themes_merge_and_higher_coverage_keeps(self):
        # Both themes have card profiles pointing at +x. Threshold 0.9
        # should merge them; the higher-coverage theme's name wins.
        per_theme = {"removal": ["c1"], "spot-removal": ["c2"]}
        themes_by_name = {
            "removal":      self._theme("removal", 500),
            "spot-removal": self._theme("spot-removal", 50),
        }
        cards_by_id = {
            "c1": self._packed_card([1.0, 0.0, 0.0, 0.0]),
            "c2": self._packed_card([0.98, 0.02, 0.0, 0.0]),
        }
        got, merge_map = tc.merge_similar_themes(
            per_theme, themes_by_name, cards_by_id, threshold=0.9,
        )
        self.assertEqual(len(got), 1)
        self.assertIn("removal", got)
        self.assertEqual(sorted(got["removal"]), ["c1", "c2"])
        self.assertEqual(merge_map, {"removal": "removal", "spot-removal": "removal"})

    def test_far_apart_themes_stay_separate(self):
        # Profiles on +x and +z — cosine sim ≈ 0, well below any
        # reasonable threshold. Both themes survive untouched.
        per_theme = {"removal": ["c1"], "ramp": ["c2"]}
        themes_by_name = {
            "removal": self._theme("removal", 500),
            "ramp":    self._theme("ramp", 300),
        }
        cards_by_id = {
            "c1": self._packed_card([1.0, 0.0, 0.0, 0.0]),
            "c2": self._packed_card([0.0, 0.0, 1.0, 0.0]),
        }
        got, merge_map = tc.merge_similar_themes(
            per_theme, themes_by_name, cards_by_id, threshold=0.9,
        )
        self.assertEqual(set(got), {"removal", "ramp"})
        self.assertEqual(merge_map, {"removal": "removal", "ramp": "ramp"})

    def test_chain_merging_via_union_find(self):
        # Three themes a-b-c where a~b and b~c but NOT a~c.
        # Expected: all three collapse into one component (union-find
        # transitively, even when a~c doesn't clear the floor alone).
        per_theme = {"a": ["c1"], "b": ["c2"], "c": ["c3"]}
        themes_by_name = {
            "a": self._theme("a", 100),
            "b": self._theme("b", 300),  # middle has highest coverage → keeper
            "c": self._theme("c", 200),
        }
        # Place vectors so adjacent pairs are ~0.95 cos-sim but the
        # outer pair is ~0.86. With threshold 0.9 the chain merges.
        cards_by_id = {
            "c1": self._packed_card([1.0, 0.0, 0.0, 0.0]),
            "c2": self._packed_card([0.72, 0.70, 0.0, 0.0]),
            "c3": self._packed_card([0.0, 1.0, 0.0, 0.0]),
        }
        got, merge_map = tc.merge_similar_themes(
            per_theme, themes_by_name, cards_by_id, threshold=0.65,
        )
        # One cluster, holding all three cards.
        self.assertEqual(len(got), 1)
        keeper = next(iter(got))
        self.assertEqual(sorted(got[keeper]), ["c1", "c2", "c3"])
        # b has the highest coverage → wins the keeper role.
        self.assertEqual(keeper, "b")
        # Every original theme points at b.
        self.assertEqual(set(merge_map.values()), {"b"})

    def test_theme_without_card_vectors_stays_singleton(self):
        # One theme has a card with a card_vector, the other doesn't.
        # The one without a profile can't participate in merging —
        # must stay a singleton.
        per_theme = {"with-vec": ["c1"], "no-vec": ["c2"]}
        themes_by_name = {
            "with-vec": self._theme("with-vec", 100),
            "no-vec":   self._theme("no-vec", 200),
        }
        cards_by_id = {
            "c1": self._packed_card([1.0, 0.0, 0.0, 0.0]),
            "c2": {},  # no card_vector field
        }
        got, _ = tc.merge_similar_themes(
            per_theme, themes_by_name, cards_by_id, threshold=0.5,
        )
        # Both themes survive untouched since one has no profile.
        self.assertEqual(set(got), {"with-vec", "no-vec"})

    def test_single_theme_is_a_no_op(self):
        per_theme = {"only": ["c1"]}
        themes_by_name = {"only": self._theme("only", 100)}
        cards_by_id = {"c1": self._packed_card([1.0, 0.0, 0.0, 0.0])}
        got, _ = tc.merge_similar_themes(
            per_theme, themes_by_name, cards_by_id, threshold=0.5,
        )
        self.assertEqual(got, per_theme)


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

    def test_default_merge_threshold_does_not_merge_distant_themes(self):
        # Cards in each theme have card_vectors pointing at different
        # basis vectors — merge pass finds no similar-enough pairs.
        self._seed_hierarchy()
        self._seed_card("id-bolt", "Lightning Bolt", ["spot-removal"],
                        text_embedding=[1.0, 0.0, 0.0, 0.0])
        self._seed_card("id-sol", "Sol Ring", ["mana-rock"],
                        text_embedding=[0.0, 1.0, 0.0, 0.0])
        # Give cards a card_vector so the merge pass has profiles to work with.
        self.cards.update_one({"_id": "id-bolt"},
            {"$set": {"card_vector": _pack([1.0, 0.0, 0.0, 0.0])}})
        self.cards.update_one({"_id": "id-sol"},
            {"$set": {"card_vector": _pack([0.0, 1.0, 0.0, 0.0])}})
        profile = dp.build_deck_profile(
            ["Lightning Bolt", "Sol Ring"],
            cards_coll=self.cards, tags_coll=self.tags,
            theme_filter=self.permissive,  # default merge_threshold = 0.9
        )
        # Two themes, each a singleton cluster.
        self.assertEqual(len(profile.clusters), 2)
        for cluster in profile.clusters:
            self.assertEqual(len(cluster.constituent_themes), 1)
            self.assertEqual(cluster.constituent_themes[0], cluster.label)

    def test_similar_themes_merge_via_card_vector_profiles(self):
        # Two themes whose cards' card_vectors all point at +x — the
        # merge pass should collapse them. Keeper is whichever has
        # higher card_coverage; in the fixture both themes have 1
        # card each, so coverage comes from count_documents on tags
        # — removal (1 card tagged) ties with card-advantage (1 card
        # tagged). Alphabetical tiebreak means "card-advantage" wins.
        self._seed_hierarchy()
        self._seed_card("id-a", "Card A", ["spot-removal"],
                        text_embedding=[1.0, 0.0, 0.0, 0.0])
        self._seed_card("id-b", "Card B", ["cantrip"],
                        text_embedding=[0.0, 0.0, 1.0, 0.0])
        self.cards.update_one({"_id": "id-a"},
            {"$set": {"card_vector": _pack([1.0, 0.0, 0.0, 0.0])}})
        self.cards.update_one({"_id": "id-b"},
            {"$set": {"card_vector": _pack([0.98, 0.02, 0.0, 0.0])}})
        profile = dp.build_deck_profile(
            ["Card A", "Card B"],
            cards_coll=self.cards, tags_coll=self.tags,
            theme_filter=self.permissive, merge_threshold=0.9,
        )
        # One merged cluster carrying both cards.
        self.assertEqual(len(profile.clusters), 1)
        cluster = profile.clusters[0]
        self.assertEqual(set(cluster.deck_card_ids), {"id-a", "id-b"})
        self.assertEqual(
            set(cluster.constituent_themes),
            {"removal", "card-advantage"},
        )

    def test_disabled_merge_keeps_themes_separate(self):
        # Same setup as the "similar themes merge" test, but with
        # merge_threshold=-1 the pass is skipped and we see both.
        self._seed_hierarchy()
        self._seed_card("id-a", "Card A", ["spot-removal"],
                        text_embedding=[1.0, 0.0, 0.0, 0.0])
        self._seed_card("id-b", "Card B", ["cantrip"],
                        text_embedding=[0.0, 0.0, 1.0, 0.0])
        self.cards.update_one({"_id": "id-a"},
            {"$set": {"card_vector": _pack([1.0, 0.0, 0.0, 0.0])}})
        self.cards.update_one({"_id": "id-b"},
            {"$set": {"card_vector": _pack([0.98, 0.02, 0.0, 0.0])}})
        profile = dp.build_deck_profile(
            ["Card A", "Card B"],
            cards_coll=self.cards, tags_coll=self.tags,
            theme_filter=self.permissive, merge_threshold=-1.0,
        )
        self.assertEqual(len(profile.clusters), 2)


# ---------------------------------------------------------------------------
# card_clusterer: _top_ancestors + most_frequent_top_level
# ---------------------------------------------------------------------------

class TestTopLevelLabelling(unittest.TestCase):
    """Pure-function tests on the top-level ancestor voting path."""

    BY_SLUG = {
        # removal top-level, two leaves.
        "removal":          {"parent_slugs": []},
        "spot-removal":     {"parent_slugs": ["removal"]},
        "mass-removal":     {"parent_slugs": ["removal"]},
        # ramp top-level, one leaf.
        "ramp":             {"parent_slugs": []},
        "mana-rock":        {"parent_slugs": ["ramp"]},
        # structural top-level (would be blocklisted).
        "triggered-ability":{"parent_slugs": []},
        "etb-trigger":      {"parent_slugs": ["triggered-ability"]},
    }
    BLOCKLIST = frozenset({"triggered-ability"})

    def test_top_ancestors_walks_up_to_root(self):
        memo: dict = {}
        self.assertEqual(
            cc._top_ancestors("spot-removal", self.BY_SLUG, memo),
            frozenset({"removal"}),
        )
        # The same slug on a repeated call hits the memo (correctness
        # check: the memoized value stays correct, not that we observe
        # a speedup).
        self.assertEqual(
            cc._top_ancestors("spot-removal", self.BY_SLUG, memo),
            frozenset({"removal"}),
        )

    def test_top_ancestors_handles_unknown_slug(self):
        memo: dict = {}
        self.assertEqual(
            cc._top_ancestors("not-in-catalog", self.BY_SLUG, memo),
            frozenset(),
        )

    def test_most_frequent_picks_winning_top_level(self):
        # Three cards, two under removal, one under ramp → removal wins.
        cards_by_id = {
            "a": {"tags": ["spot-removal"]},
            "b": {"tags": ["mass-removal"]},
            "c": {"tags": ["mana-rock"]},
        }
        winner = cc.most_frequent_top_level(
            ["a", "b", "c"], cards_by_id, self.BY_SLUG,
            blocklist=self.BLOCKLIST,
        )
        self.assertEqual(winner, "removal")

    def test_most_frequent_dedupes_tags_per_card(self):
        # A single card with ten removal-flavored tags should still
        # count once for removal, so the two single-tag ramp cards
        # tie-or-beat it.
        cards_by_id = {
            "stuffed-removal": {"tags": ["spot-removal"] * 10},
            "r1": {"tags": ["mana-rock"]},
            "r2": {"tags": ["mana-rock"]},
        }
        winner = cc.most_frequent_top_level(
            ["stuffed-removal", "r1", "r2"], cards_by_id, self.BY_SLUG,
            blocklist=self.BLOCKLIST,
        )
        # 1 vote for removal, 2 votes for ramp → ramp wins.
        self.assertEqual(winner, "ramp")

    def test_most_frequent_respects_blocklist(self):
        # Three cards, all under the blocklisted structural top-level.
        # No surviving candidate → None.
        cards_by_id = {
            "a": {"tags": ["etb-trigger"]},
            "b": {"tags": ["etb-trigger"]},
            "c": {"tags": ["etb-trigger"]},
        }
        winner = cc.most_frequent_top_level(
            ["a", "b", "c"], cards_by_id, self.BY_SLUG,
            blocklist=self.BLOCKLIST,
        )
        self.assertIsNone(winner)

    def test_label_cluster_falls_back_when_no_winner(self):
        cards_by_id = {"a": {"tags": ["etb-trigger"]}}
        label = cc.label_cluster(
            ["a"], cards_by_id, self.BY_SLUG,
            blocklist=self.BLOCKLIST, fallback_name="cluster-7",
        )
        self.assertEqual(label, "cluster-7")


# ---------------------------------------------------------------------------
# card_clusterer: structural feature-vector helpers
# ---------------------------------------------------------------------------

class TestParseTypeLine(unittest.TestCase):

    def test_simple_creature_line(self):
        supers, types, subs = cc.parse_type_line("Creature — Human Wizard")
        self.assertEqual(supers, [])
        self.assertEqual(types, ["Creature"])
        self.assertEqual(subs, ["Human", "Wizard"])

    def test_legendary_supertype_picked_up(self):
        supers, types, subs = cc.parse_type_line(
            "Legendary Creature — Human Wizard"
        )
        self.assertEqual(supers, ["Legendary"])
        self.assertEqual(types, ["Creature"])
        self.assertEqual(subs, ["Human", "Wizard"])

    def test_no_subtypes(self):
        # "Instant" has no em-dash section — subs should be empty.
        supers, types, subs = cc.parse_type_line("Instant")
        self.assertEqual((supers, types, subs), ([], ["Instant"], []))

    def test_multi_face_line_merges_both_sides(self):
        # DFC: " // " splits faces; each face is parsed independently and
        # the subs across faces concatenate so "Insect" doesn't vanish.
        supers, types, subs = cc.parse_type_line(
            "Creature — Human Wizard // Creature — Human Insect"
        )
        self.assertEqual(supers, [])
        self.assertEqual(types, ["Creature", "Creature"])
        self.assertEqual(subs, ["Human", "Wizard", "Human", "Insect"])

    def test_empty_or_missing_line(self):
        self.assertEqual(cc.parse_type_line(""), ([], [], []))
        self.assertEqual(cc.parse_type_line(None), ([], [], []))


class TestBuildTypeVector(unittest.TestCase):

    def test_multi_hot_shape_and_hits(self):
        # Vector has fixed supertype + type blocks then the deck-local
        # subtype block. A Legendary Creature — Human Wizard should
        # trigger exactly 4 dims: Legendary, Creature, Human, Wizard.
        doc = {"type_line": "Legendary Creature — Human Wizard"}
        vocab = ("Elf", "Human", "Wizard")
        vec = cc.build_type_vector(doc, vocab)
        expected_dim = len(cc._SUPERTYPES) + len(cc._TYPES) + len(vocab)
        self.assertEqual(vec.shape, (expected_dim,))
        self.assertEqual(vec.dtype, np.float32)
        self.assertEqual(float(vec.sum()), 4.0)
        # Spot-check the right dims lit up.
        super_offset = cc._SUPERTYPES.index("Legendary")
        type_offset = len(cc._SUPERTYPES) + cc._TYPES.index("Creature")
        human_offset = len(cc._SUPERTYPES) + len(cc._TYPES) + vocab.index("Human")
        wizard_offset = len(cc._SUPERTYPES) + len(cc._TYPES) + vocab.index("Wizard")
        self.assertEqual(vec[super_offset], 1.0)
        self.assertEqual(vec[type_offset], 1.0)
        self.assertEqual(vec[human_offset], 1.0)
        self.assertEqual(vec[wizard_offset], 1.0)

    def test_subtype_outside_vocab_silently_drops(self):
        # "Elf" isn't in vocab → that dim doesn't exist and no error raised.
        # Only the vocab subtypes count. Protects against subtypes that
        # appear in one card but not elsewhere in the deck.
        doc = {"type_line": "Creature — Elf Druid"}
        vocab = ("Human", "Wizard")  # neither matches
        vec = cc.build_type_vector(doc, vocab)
        # Only "Creature" dim should fire.
        self.assertEqual(float(vec.sum()), 1.0)

    def test_empty_type_line_returns_zero_vector(self):
        vec = cc.build_type_vector({}, ("Human",))
        self.assertEqual(
            vec.shape,
            (len(cc._SUPERTYPES) + len(cc._TYPES) + 1,),
        )
        self.assertEqual(float(vec.sum()), 0.0)


class TestBuildManaCostVector(unittest.TestCase):

    def test_coloured_pips_counted(self):
        # Birds of Paradise — one green pip, no other symbols.
        vec = cc.build_mana_cost_vector("{G}")
        self.assertEqual(vec.shape, (10,))
        self.assertEqual(vec[5], 1.0)  # G
        self.assertEqual(float(vec.sum()), 1.0)

    def test_generic_plus_coloured(self):
        # {3}{W}{W} — generic bucket gets 3.0, white bucket gets 2.
        vec = cc.build_mana_cost_vector("{3}{W}{W}")
        self.assertEqual(vec[0], 3.0)  # generic
        self.assertEqual(vec[1], 2.0)  # W
        self.assertEqual(float(vec.sum()), 5.0)

    def test_variable_x_counted_separately(self):
        # {X}{R}{R} — X goes in its own bucket, not generic.
        vec = cc.build_mana_cost_vector("{X}{R}{R}")
        self.assertEqual(vec[0], 0.0)
        self.assertEqual(vec[4], 2.0)  # R
        self.assertEqual(vec[6], 1.0)  # X/Y/Z

    def test_hybrid_pip_counted_once_in_hybrid_bucket(self):
        # {W/U} is one hybrid pip — not half-and-half across W and U.
        vec = cc.build_mana_cost_vector("{W/U}")
        self.assertEqual(vec[1], 0.0)  # not W
        self.assertEqual(vec[2], 0.0)  # not U
        self.assertEqual(vec[7], 1.0)  # hybrid

    def test_phyrexian_pip_counted_in_phyrexian_bucket(self):
        # {W/P} — phyrexian takes precedence over hybrid.
        vec = cc.build_mana_cost_vector("{W/P}")
        self.assertEqual(vec[1], 0.0)
        self.assertEqual(vec[7], 0.0)
        self.assertEqual(vec[8], 1.0)  # phyrexian

    def test_snow_pip(self):
        vec = cc.build_mana_cost_vector("{S}{2}")
        self.assertEqual(vec[9], 1.0)
        self.assertEqual(vec[0], 2.0)

    def test_empty_or_none_cost_is_zero_vector(self):
        self.assertEqual(float(cc.build_mana_cost_vector(None).sum()), 0.0)
        self.assertEqual(float(cc.build_mana_cost_vector("").sum()), 0.0)

    def test_multi_face_cost_sums_both_faces(self):
        # MDFC-style " // " join — both faces' pips contribute.
        vec = cc.build_mana_cost_vector("{2}{G} // {3}{G}")
        self.assertEqual(vec[0], 5.0)
        self.assertEqual(vec[5], 2.0)


# ---------------------------------------------------------------------------
# card_clusterer: build_keyword_vector
# ---------------------------------------------------------------------------

class TestBuildKeywordVector(unittest.TestCase):

    def test_multi_hot_against_vocab(self):
        doc = {"keywords": ["Flying", "Lifelink"]}
        vocab = ("flying", "haste", "lifelink", "vigilance")
        vec = cc.build_keyword_vector(doc, vocab)
        self.assertEqual(vec.shape, (4,))
        self.assertEqual(vec.dtype, np.float32)
        self.assertEqual(list(vec), [1.0, 0.0, 1.0, 0.0])

    def test_case_insensitive_match(self):
        # Vocab is lowercase (deck_profile.build_deck_vocab lowercases
        # on collect); keyword list on the card may use mixed case.
        doc = {"keywords": ["FLYING", "haste"]}
        vocab = ("flying", "haste")
        vec = cc.build_keyword_vector(doc, vocab)
        self.assertEqual(list(vec), [1.0, 1.0])

    def test_keyword_outside_vocab_drops(self):
        # Keyword present on card but not in deck vocab doesn't error.
        doc = {"keywords": ["Flying", "Horsemanship"]}
        vocab = ("flying", "haste")
        vec = cc.build_keyword_vector(doc, vocab)
        self.assertEqual(list(vec), [1.0, 0.0])

    def test_missing_keywords_field(self):
        # Older cached doc with no keywords field → zero vector of the
        # expected vocab size.
        vec = cc.build_keyword_vector({}, ("flying",))
        self.assertEqual(vec.shape, (1,))
        self.assertEqual(float(vec.sum()), 0.0)


# ---------------------------------------------------------------------------
# card_clusterer: build_color_vector + build_color_identity_vector
# ---------------------------------------------------------------------------

class TestBuildColorVectors(unittest.TestCase):

    def test_colors_simple_multihot(self):
        vec = cc.build_color_vector({"colors": ["W", "U"]})
        self.assertEqual(vec.shape, (5,))
        self.assertEqual(list(vec), [1.0, 1.0, 0.0, 0.0, 0.0])

    def test_colors_colourless_is_zero_vector(self):
        # Artifacts, lands → empty colors list → all zeros.
        self.assertEqual(float(cc.build_color_vector({"colors": []}).sum()), 0.0)
        self.assertEqual(float(cc.build_color_vector({}).sum()), 0.0)

    def test_color_identity_matches_wubrg_order(self):
        # The WUBRG dim assignment must be stable — spot-check B at
        # index 2 and G at index 4.
        vec = cc.build_color_identity_vector({"color_identity": ["B", "G"]})
        self.assertEqual(vec[2], 1.0)
        self.assertEqual(vec[4], 1.0)
        self.assertEqual(float(vec.sum()), 2.0)

    def test_unknown_colour_token_silently_drops(self):
        # Defensive: a stale cache with an invalid color like "C" doesn't
        # crash — just contributes nothing. Guards against upstream vocab
        # drift.
        vec = cc.build_color_vector({"colors": ["W", "C"]})
        self.assertEqual(float(vec.sum()), 1.0)


# ---------------------------------------------------------------------------
# card_clusterer: build_power_vector + build_toughness_vector
# ---------------------------------------------------------------------------

class TestBuildPowerToughnessVectors(unittest.TestCase):

    def test_integer_power_sets_value_and_flag(self):
        vec = cc.build_power_vector({"power": "3"})
        self.assertEqual(vec.shape, (2,))
        self.assertEqual(vec[0], 3.0)
        self.assertEqual(vec[1], 1.0)

    def test_missing_power_is_all_zero(self):
        # Non-creature → power=None → [0, 0]. The flag dim distinguishes
        # this from an actual 0-power creature below.
        vec = cc.build_power_vector({"power": None})
        self.assertEqual(list(vec), [0.0, 0.0])
        # And a missing field (older cached doc) behaves the same.
        self.assertEqual(list(cc.build_power_vector({})), [0.0, 0.0])

    def test_zero_power_creature_keeps_flag(self):
        # Walking Wall "0/4" — value 0 but it IS a creature, so flag=1.
        # Without the flag, this would collapse into non-creature.
        vec = cc.build_power_vector({"power": "0"})
        self.assertEqual(list(vec), [0.0, 1.0])

    def test_variable_star_power_parses_as_zero_with_flag(self):
        # Tarmogoyf-style "*" has no leading integer → value 0, flag 1.
        vec = cc.build_power_vector({"power": "*"})
        self.assertEqual(list(vec), [0.0, 1.0])

    def test_hybrid_variable_parses_leading_integer(self):
        # "1+*" (e.g. some ability-granted creature) → value 1, flag 1.
        vec = cc.build_power_vector({"power": "1+*"})
        self.assertEqual(list(vec), [1.0, 1.0])
        # "7-*" (negative modifier) → leading integer 7.
        vec2 = cc.build_power_vector({"power": "7-*"})
        self.assertEqual(list(vec2), [7.0, 1.0])

    def test_toughness_follows_same_rules(self):
        self.assertEqual(list(cc.build_toughness_vector({"toughness": "4"})), [4.0, 1.0])
        self.assertEqual(list(cc.build_toughness_vector({})), [0.0, 0.0])
        self.assertEqual(list(cc.build_toughness_vector({"toughness": "*"})), [0.0, 1.0])


# ---------------------------------------------------------------------------
# card_clusterer: cluster_cards_by_vector
# ---------------------------------------------------------------------------

class TestClusterCardsByVector(unittest.TestCase):

    def _card(self, sid, vec, tags=(), type_line=None, mana_cost=None):
        return {
            "_id": sid,
            "tags": list(tags),
            "card_vector": emb._pack_embedding(np.array(vec, dtype=np.float32)),
            "type_line": type_line,
            "mana_cost": mana_cost,
        }

    def test_three_tight_groups_cluster_cleanly(self):
        # Three card groups near orthogonal axes; HDBSCAN should split
        # them into 3 clusters with no noise.
        deck = [
            self._card("a1", [1.0, 0.0, 0.0, 0.0]),
            self._card("a2", [0.95, 0.05, 0.0, 0.0]),
            self._card("a3", [0.9, 0.1, 0.0, 0.0]),
            self._card("b1", [0.0, 1.0, 0.0, 0.0]),
            self._card("b2", [0.05, 0.95, 0.0, 0.0]),
            self._card("b3", [0.0, 0.9, 0.1, 0.0]),
            self._card("c1", [0.0, 0.0, 1.0, 0.0]),
            self._card("c2", [0.0, 0.0, 0.95, 0.05]),
            self._card("c3", [0.0, 0.0, 0.9, 0.1]),
        ]
        clusters, noise = cc.cluster_cards_by_vector(deck, min_cluster_size=2)
        self.assertEqual(len(clusters), 3)
        self.assertEqual(noise, [])
        memberships = {frozenset(sids) for sids, _ in clusters}
        self.assertEqual(memberships, {
            frozenset({"a1", "a2", "a3"}),
            frozenset({"b1", "b2", "b3"}),
            frozenset({"c1", "c2", "c3"}),
        })

    def test_cards_without_card_vector_skipped(self):
        # One card has no card_vector field — doesn't participate in
        # clustering, doesn't appear in the noise bucket either.
        deck = [
            self._card("a", [1.0, 0.0, 0.0, 0.0]),
            self._card("b", [1.0, 0.0, 0.0, 0.0]),
            {"_id": "no-vec", "tags": ["x"]},
        ]
        clusters, noise = cc.cluster_cards_by_vector(deck, min_cluster_size=2)
        all_ids = set(noise) | {sid for sids, _ in clusters for sid in sids}
        self.assertNotIn("no-vec", all_ids)

    def test_degenerate_input_returns_empty(self):
        # One card < default min_cluster_size → short-circuit, no crash.
        deck = [self._card("a", [1.0, 0.0, 0.0, 0.0])]
        clusters, noise = cc.cluster_cards_by_vector(deck, min_cluster_size=3)
        self.assertEqual(clusters, [])
        self.assertEqual(noise, [])

    def test_type_weight_separates_otherwise_similar_vectors(self):
        # Six cards packed tightly on the +x axis (small perturbations so
        # HDBSCAN's mutual-reachability doesn't degenerate). Without
        # feature weighting the cluster shape depends on tiny perturbations
        # alone; adding a dominant --type-weight should force a split
        # along Creature vs. Instant irrespective of those perturbations.
        deck = [
            self._card("c1", [1.0, 0.01, 0.0, 0.0], type_line="Creature — Human"),
            self._card("c2", [1.0, 0.02, 0.0, 0.0], type_line="Creature — Elf"),
            self._card("c3", [1.0, 0.03, 0.0, 0.0], type_line="Creature — Human"),
            self._card("i1", [1.0, 0.04, 0.0, 0.0], type_line="Instant"),
            self._card("i2", [1.0, 0.05, 0.0, 0.0], type_line="Instant"),
            self._card("i3", [1.0, 0.06, 0.0, 0.0], type_line="Instant"),
        ]
        # Dominant type weight: Creature-vs-Instant distance >> within-type
        # distance → two clusters, one per type.
        weighted_clusters, _ = cc.cluster_cards_by_vector(
            deck,
            min_cluster_size=2,
            feature_weights={"types": 10.0},
            deck_vocab={"subtypes": ("Human", "Elf")},
        )
        memberships = {frozenset(sids) for sids, _ in weighted_clusters}
        self.assertEqual(memberships, {
            frozenset({"c1", "c2", "c3"}),
            frozenset({"i1", "i2", "i3"}),
        })

    def test_centroid_stays_in_base_card_vector_space(self):
        # Centroid dim must equal card_vector dim regardless of feature
        # weights — downstream ranking expects base-space centroids.
        # Reuse the 6-card type-split setup so HDBSCAN reliably produces
        # clusters under cosine metric; what we check here is that EACH
        # cluster's centroid is 4-dim (base space), not 4 + augmented dims.
        deck = [
            self._card("c1", [1.0, 0.01, 0.0, 0.0], type_line="Creature — Human"),
            self._card("c2", [1.0, 0.02, 0.0, 0.0], type_line="Creature — Elf"),
            self._card("c3", [1.0, 0.03, 0.0, 0.0], type_line="Creature — Human"),
            self._card("i1", [1.0, 0.04, 0.0, 0.0], type_line="Instant"),
            self._card("i2", [1.0, 0.05, 0.0, 0.0], type_line="Instant"),
            self._card("i3", [1.0, 0.06, 0.0, 0.0], type_line="Instant"),
        ]
        clusters, _ = cc.cluster_cards_by_vector(
            deck,
            min_cluster_size=2,
            feature_weights={"types": 10.0, "mana_cost": 2.0},
            deck_vocab={"subtypes": ("Human", "Elf")},
        )
        self.assertGreater(len(clusters), 0)
        for _sids, centroid in clusters:
            self.assertEqual(centroid.shape, (4,))

    def test_zero_weight_feature_is_a_noop(self):
        # Weight 0.0 shouldn't change cluster membership vs. no feature
        # at all — the sub-vector is simply skipped.
        deck = [
            self._card("a", [1.0, 0.01, 0.0, 0.0], type_line="Creature"),
            self._card("b", [1.0, 0.02, 0.0, 0.0], type_line="Instant"),
            self._card("c", [1.0, 0.03, 0.0, 0.0], type_line="Sorcery"),
        ]
        bare, _ = cc.cluster_cards_by_vector(deck, min_cluster_size=2)
        with_zero, _ = cc.cluster_cards_by_vector(
            deck,
            min_cluster_size=2,
            feature_weights={"types": 0.0},
            deck_vocab={"subtypes": ()},
        )
        self.assertEqual(
            [sorted(sids) for sids, _ in bare],
            [sorted(sids) for sids, _ in with_zero],
        )

    def test_unknown_feature_name_raises(self):
        # Typo in feature name should surface loudly, not silently drop.
        deck = [
            self._card("a", [1.0, 0.0, 0.0, 0.0]),
            self._card("b", [1.0, 0.0, 0.0, 0.0]),
        ]
        with self.assertRaises(KeyError):
            cc.cluster_cards_by_vector(
                deck,
                min_cluster_size=2,
                feature_weights={"bogus": 0.5},
            )

    def test_color_identity_weight_splits_otherwise_identical_vectors(self):
        # Six cards near the +x axis — card_vector alone can't split them.
        # Three carry WU identity, three carry BR. A dominant
        # --color-identity-weight should split by colour wedge.
        def card(sid, offset, ci):
            return {
                "_id": sid,
                "tags": [],
                "card_vector": emb._pack_embedding(
                    np.array([1.0, offset, 0.0, 0.0], dtype=np.float32)
                ),
                "color_identity": list(ci),
            }
        deck = [
            card("wu1", 0.01, ["W", "U"]),
            card("wu2", 0.02, ["W", "U"]),
            card("wu3", 0.03, ["W", "U"]),
            card("br1", 0.04, ["B", "R"]),
            card("br2", 0.05, ["B", "R"]),
            card("br3", 0.06, ["B", "R"]),
        ]
        clusters, _ = cc.cluster_cards_by_vector(
            deck,
            min_cluster_size=2,
            feature_weights={"color_identity": 10.0},
        )
        memberships = {frozenset(sids) for sids, _ in clusters}
        self.assertEqual(memberships, {
            frozenset({"wu1", "wu2", "wu3"}),
            frozenset({"br1", "br2", "br3"}),
        })

    def test_power_weight_splits_creatures_from_noncreatures(self):
        # Half the cards are creatures (power="3"), half are non-creatures
        # (power=None). card_vector is near-identical. A dominant
        # --power-weight should split by the has-power flag dim.
        def card(sid, offset, power):
            doc = {
                "_id": sid,
                "tags": [],
                "card_vector": emb._pack_embedding(
                    np.array([1.0, offset, 0.0, 0.0], dtype=np.float32)
                ),
            }
            if power is not None:
                doc["power"] = power
            return doc
        deck = [
            card("c1", 0.01, "3"),
            card("c2", 0.02, "4"),
            card("c3", 0.03, "2"),
            card("nc1", 0.04, None),
            card("nc2", 0.05, None),
            card("nc3", 0.06, None),
        ]
        clusters, _ = cc.cluster_cards_by_vector(
            deck,
            min_cluster_size=2,
            feature_weights={"power": 10.0},
        )
        memberships = {frozenset(sids) for sids, _ in clusters}
        # Both groups should surface; exact size depends on HDBSCAN but
        # creatures must not mix with non-creatures.
        self.assertEqual(len(clusters), 2)
        creature_set = frozenset({"c1", "c2", "c3"})
        noncreature_set = frozenset({"nc1", "nc2", "nc3"})
        self.assertIn(creature_set, memberships)
        self.assertIn(noncreature_set, memberships)


# ---------------------------------------------------------------------------
# build_cluster_profile — end-to-end
# ---------------------------------------------------------------------------

class TestBuildClusterProfile(_MongoBackedTestCase):

    def _seed_cluster_fixture(self):
        """3 removal cards (card_vector near +x), 3 ramp cards (near +y)."""
        self._seed_hierarchy()
        for i in range(3):
            sid = f"id-rem-{i}"
            self.cards.insert_one({
                "_id": sid, "name": f"Rem {i}", "names": [f"rem {i}"],
                "tags": ["spot-removal"],
                "card_vector": _pack([1.0, 0.0, 0.0, 0.0]),
            })
        for i in range(3):
            sid = f"id-ramp-{i}"
            self.cards.insert_one({
                "_id": sid, "name": f"Ramp {i}", "names": [f"ramp {i}"],
                "tags": ["mana-rock"],
                "card_vector": _pack([0.0, 1.0, 0.0, 0.0]),
            })

    def test_cluster_mode_produces_two_labelled_clusters(self):
        self._seed_cluster_fixture()
        profile = dp.build_cluster_profile(
            ["Rem 0", "Rem 1", "Rem 2", "Ramp 0", "Ramp 1", "Ramp 2"],
            cards_coll=self.cards, tags_coll=self.tags,
            min_cluster_size=2,
        )
        self.assertEqual(len(profile.clusters), 2)
        labels = {c.label for c in profile.clusters}
        # Each cluster's cards all share one top-level ancestor →
        # labels are deterministic.
        self.assertEqual(labels, {"removal", "ramp"})

    def test_cluster_mode_noise_tags_is_empty(self):
        # Cluster mode doesn't partition tags — noise_tags is always ().
        self._seed_cluster_fixture()
        profile = dp.build_cluster_profile(
            ["Rem 0", "Rem 1", "Rem 2"],
            cards_coll=self.cards, tags_coll=self.tags,
            min_cluster_size=2,
        )
        self.assertEqual(profile.noise_tags, ())

    def test_cluster_mode_cards_without_vector_land_in_unassigned(self):
        self._seed_hierarchy()
        self.cards.insert_one({
            "_id": "id-no-vec", "name": "No Vector",
            "names": ["no vector"], "tags": ["spot-removal"],
        })
        profile = dp.build_cluster_profile(
            ["No Vector"], cards_coll=self.cards, tags_coll=self.tags,
            min_cluster_size=2,
        )
        self.assertEqual(profile.unassigned_card_ids, ("id-no-vec",))
        self.assertEqual(profile.clusters, ())

    def test_cluster_mode_label_falls_back_when_blocked(self):
        # Seed a cluster of cards tagged only with a blocklisted
        # top-level — label falls back to cluster-N. 5 cards give
        # HDBSCAN enough density to form one cluster; fewer than that
        # can trip the density check and all land in noise.
        self.tags.insert_many([
            {"_id": "triggered-ability", "parent_slugs": [], "child_slugs": ["etb"],
             "embedding": _pack([1.0, 0.0, 0.0, 0.0])},
            {"_id": "etb", "parent_slugs": ["triggered-ability"], "child_slugs": [],
             "embedding": _pack([1.0, 0.0, 0.0, 0.0])},
        ])
        vecs = [
            [1.0, 0.0, 0.0, 0.0],
            [0.95, 0.05, 0.0, 0.0],
            [0.9, 0.1, 0.0, 0.0],
            [0.98, 0.02, 0.0, 0.0],
            [0.92, 0.08, 0.0, 0.0],
        ]
        for i, vec in enumerate(vecs):
            self.cards.insert_one({
                "_id": f"id-etb-{i}", "name": f"ETB {i}", "names": [f"etb {i}"],
                "tags": ["etb"],
                "card_vector": _pack(vec),
            })
        profile = dp.build_cluster_profile(
            [f"ETB {i}" for i in range(5)],
            cards_coll=self.cards, tags_coll=self.tags,
            min_cluster_size=2,
        )
        # HDBSCAN may form 1 or 2 clusters depending on local density;
        # the invariant we care about is that EVERY cluster falls back
        # to the "cluster-N" label since every cluster's cards only
        # reach a blocklisted top-level.
        self.assertGreater(len(profile.clusters), 0)
        for cluster in profile.clusters:
            self.assertTrue(cluster.label.startswith("cluster-"))


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

    def test_mode_cluster_dispatches_to_build_cluster_profile(self):
        # Seed a 3-card removal cluster + 3-card ramp cluster, run the
        # CLI in cluster mode, and confirm the labels come out via the
        # most-frequent-top-level voting path.
        self._seed_hierarchy()
        for i in range(3):
            self.cards.insert_one({
                "_id": f"rem-{i}", "name": f"Rem {i}", "names": [f"rem {i}"],
                "tags": ["spot-removal"],
                "card_vector": _pack([1.0, 0.0, 0.0, 0.0]),
            })
        for i in range(3):
            self.cards.insert_one({
                "_id": f"ramp-{i}", "name": f"Ramp {i}", "names": [f"ramp {i}"],
                "tags": ["mana-rock"],
                "card_vector": _pack([0.0, 1.0, 0.0, 0.0]),
            })
        rc, out, _ = _run_cli([
            "Rem 0", "Rem 1", "Rem 2", "Ramp 0", "Ramp 1", "Ramp 2",
            "--mode", "cluster", "--cluster-min-size", "2",
        ])
        self.assertEqual(rc, 0)
        self.assertIn("theme 'removal'", out)
        self.assertIn("theme 'ramp'", out)

    def test_type_and_mana_cost_weight_flags_pipe_through(self):
        # Confirms the two new cluster-mode knobs are accepted by argparse
        # and reach the clusterer without crashing. Seeds the same two
        # groups as the prior test with type_line + mana_cost populated so
        # the feature builders have real data to consume.
        self._seed_hierarchy()
        for i in range(3):
            self.cards.insert_one({
                "_id": f"rem-{i}", "name": f"Rem {i}", "names": [f"rem {i}"],
                "tags": ["spot-removal"],
                "card_vector": _pack([1.0, 0.0, 0.0, 0.0]),
                "type_line": "Instant",
                "mana_cost": "{R}",
            })
        for i in range(3):
            self.cards.insert_one({
                "_id": f"ramp-{i}", "name": f"Ramp {i}", "names": [f"ramp {i}"],
                "tags": ["mana-rock"],
                "card_vector": _pack([0.0, 1.0, 0.0, 0.0]),
                "type_line": "Artifact",
                "mana_cost": "{1}",
            })
        rc, out, _ = _run_cli([
            "Rem 0", "Rem 1", "Rem 2", "Ramp 0", "Ramp 1", "Ramp 2",
            "--mode", "cluster", "--cluster-min-size", "2",
            "--type-weight", "0.4", "--mana-cost-weight", "0.3",
        ])
        self.assertEqual(rc, 0)
        # Both groups should still surface as their respective themes —
        # the structural features reinforce the base split rather than
        # scrambling it on this orthogonal fixture.
        self.assertIn("theme 'removal'", out)
        self.assertIn("theme 'ramp'", out)

    def test_remaining_five_weight_flags_pipe_through(self):
        # All of --keyword-weight / --color-weight / --color-identity-weight
        # / --power-weight / --toughness-weight on one CLI call. Seeds the
        # same orthogonal two-group fixture with full structured fields
        # populated so every feature builder has real data to consume.
        self._seed_hierarchy()
        for i in range(3):
            self.cards.insert_one({
                "_id": f"creat-{i}", "name": f"Creat {i}", "names": [f"creat {i}"],
                "tags": ["spot-removal"],
                "card_vector": _pack([1.0, 0.0, 0.0, 0.0]),
                "type_line": "Creature — Human",
                "mana_cost": "{R}",
                "keywords": ["Haste", "Lifelink"],
                "colors": ["R"],
                "color_identity": ["R"],
                "power": "2",
                "toughness": "1",
            })
        for i in range(3):
            self.cards.insert_one({
                "_id": f"inst-{i}", "name": f"Inst {i}", "names": [f"inst {i}"],
                "tags": ["mana-rock"],
                "card_vector": _pack([0.0, 1.0, 0.0, 0.0]),
                "type_line": "Instant",
                "mana_cost": "{U}",
                "keywords": ["Flash"],
                "colors": ["U"],
                "color_identity": ["U"],
                "power": None,
                "toughness": None,
            })
        rc, out, _ = _run_cli([
            "Creat 0", "Creat 1", "Creat 2", "Inst 0", "Inst 1", "Inst 2",
            "--mode", "cluster", "--cluster-min-size", "2",
            "--keyword-weight", "0.3",
            "--color-weight", "0.2",
            "--color-identity-weight", "0.2",
            "--power-weight", "0.3",
            "--toughness-weight", "0.3",
        ])
        self.assertEqual(rc, 0)
        # On the orthogonal fixture both groups still surface — knobs
        # reinforce the natural split rather than scrambling it.
        self.assertIn("theme 'removal'", out)
        self.assertIn("theme 'ramp'", out)

    def test_train_flag_requires_commander(self):
        # --train without --commander should error out via argparse.
        self._seed_hierarchy()
        self._seed_card("id-a", "Card A", ["spot-removal"],
                        text_embedding=[1.0, 0.0, 0.0, 0.0])
        with self.assertRaises(SystemExit):
            _run_cli(["Card A", "--mode", "cluster", "--train"])

    def test_train_flag_pipes_through_with_mocked_edhrec(self):
        # End-to-end --train run: six cards (3 creatures hi-lift, 3
        # instants lo-lift), EDHREC mocked to return those lifts, trainer
        # runs 5 trials with --train-knobs restricted to "types" so the
        # search is deterministic enough to assert on. Verifies: the CLI
        # path reaches get_commander_signals, prints the trained-weights
        # summary, and still produces a cluster render below it.
        self._seed_hierarchy()
        for i in range(3):
            sid = f"creat-{i}"
            self.cards.insert_one({
                "_id": sid, "name": f"Creat {i}", "names": [f"creat {i}"],
                "tags": ["spot-removal"],
                "card_vector": _pack([1.0, 0.0, 0.0, 0.0]),
                "type_line": "Creature — Human",
                "mana_cost": "{R}",
            })
        for i in range(3):
            sid = f"inst-{i}"
            self.cards.insert_one({
                "_id": sid, "name": f"Inst {i}", "names": [f"inst {i}"],
                "tags": ["mana-rock"],
                "card_vector": _pack([1.0, 0.0, 0.0, 0.0]),
                "type_line": "Instant",
                "mana_cost": "{U}",
            })

        fake_signals = [
            edh.CardSignal(name=f"Creat {i}", scryfall_id=f"creat-{i}",
                           lift=5.0, synergy=0.0, num_decks=100,
                           potential_decks=1000)
            for i in range(3)
        ] + [
            edh.CardSignal(name=f"Inst {i}", scryfall_id=f"inst-{i}",
                           lift=1.0, synergy=0.0, num_decks=10,
                           potential_decks=1000)
            for i in range(3)
        ]
        with patch.object(dp.edh, "get_commander_signals",
                          return_value=fake_signals) as mock_edh:
            rc, out, _err = _run_cli([
                "Creat 0", "Creat 1", "Creat 2", "Inst 0", "Inst 1", "Inst 2",
                "--mode", "cluster", "--cluster-min-size", "2",
                "--train",
                "--commander", "Fake Commander",
                "--train-trials", "5",
                "--train-knobs", "types",
                "--train-seed", "0",
            ])
        self.assertEqual(rc, 0)
        mock_edh.assert_called_once_with("Fake Commander")
        # Trainer header shows a score; cluster render follows below.
        self.assertIn("best eta^2", out)
        self.assertIn("best weights:", out)
        self.assertIn("types", out)

    def test_merge_threshold_flag_pipes_through(self):
        # Seed two cards whose themes should merge at threshold 0.9.
        # Confirm the "merged:" annotation appears in the render.
        self._seed_hierarchy()
        self._seed_card("id-a", "Card A", ["spot-removal"],
                        text_embedding=[1.0, 0.0, 0.0, 0.0])
        self._seed_card("id-b", "Card B", ["cantrip"],
                        text_embedding=[0.0, 0.0, 1.0, 0.0])
        self.cards.update_one({"_id": "id-a"},
            {"$set": {"card_vector": _pack([1.0, 0.0, 0.0, 0.0])}})
        self.cards.update_one({"_id": "id-b"},
            {"$set": {"card_vector": _pack([0.98, 0.02, 0.0, 0.0])}})
        rc, out, _ = _run_cli([
            "Card A", "Card B",
            "--min-coverage", "1", "--merge-threshold", "0.9",
        ])
        self.assertEqual(rc, 0)
        # Exactly one cluster, with a "merged: ..." annotation naming
        # the non-keeper constituent.
        self.assertIn("[merged:", out)
        # And the no-merge version keeps them split.
        rc, out_split, _ = _run_cli([
            "Card A", "Card B",
            "--min-coverage", "1", "--merge-threshold", "-1",
        ])
        self.assertEqual(rc, 0)
        self.assertNotIn("[merged:", out_split)


if __name__ == "__main__":
    unittest.main()
