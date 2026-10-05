"""Offline tests for mtg_recommender.deck_profile.

Clustering tests use hand-crafted 4-dim tag embeddings arranged in
tight groups around orthogonal basis vectors, so HDBSCAN's output
is deterministic and we can assert on exact group memberships rather
than fiddling with "about this many clusters" heuristics.

Test classes:
    TestResolveDeckCards        name lookup, missing names, dedup by id,
                                 case-insensitive
    TestCollectTagUniverse      union across deck cards, sort, empty
    TestLoadTagEmbeddings       bulk fetch, skips tags without embedding
    TestClusterTagEmbeddings    3 tight groups -> 3 clusters;
                                 degenerate universe returns empty
    TestReassignNoise           noise-reassignment pass: absorb when near,
                                 skip when far, threshold boundary strict,
                                 negative threshold disables
    TestCentroidAndLabel        picks nearest tag; L2-normalises centroid
    TestBuildDeckProfile        end-to-end on a seeded mongomock cluster
    TestRenderProfile           truncation marker, cluster display shape
    TestMainCLI                 argparse: --file, --min-cluster-size, --limit,
                                 --min-samples / --reassign-threshold piping,
                                 empty-input error
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


def _pack(vec: list[float]) -> bytes:
    """Pack a vector in the storage format `load_tag_embeddings` expects."""
    return emb._pack_embedding(np.array(vec, dtype=np.float32))


# ---------------------------------------------------------------------------
# Fixture: three tight clusters in 4-dim space.
# ---------------------------------------------------------------------------

# Cluster A ("removal"): tag embeddings near [1,0,0,0].
# Cluster B ("ramp"):    tag embeddings near [0,1,0,0].
# Cluster C ("draw"):    tag embeddings near [0,0,1,0].
#
# The exact "nearest tag to centroid" winner within a tight cluster is
# sensitive to tiny floating-point drift (the centroid of asymmetric
# satellites doesn't sit exactly on the axis). Tests therefore assert
# that each cluster's label is drawn from its own tag set rather than
# naming a specific slug — that's the actual contract we care about.

TAG_FIXTURES = {
    # removal cluster
    "spot-removal":      [1.00, 0.00, 0.00, 0.00],
    "burn-any":          [0.95, 0.05, 0.00, 0.00],
    "removal-creature":  [0.90, 0.10, 0.05, 0.00],
    # ramp cluster
    "mana-rock":         [0.00, 1.00, 0.00, 0.00],
    "mana-dork":         [0.05, 0.95, 0.00, 0.00],
    "ramp":              [0.00, 0.90, 0.05, 0.10],
    # draw cluster
    "card-draw":         [0.00, 0.00, 1.00, 0.00],
    "cantrip":           [0.00, 0.00, 0.95, 0.10],
}


class _MongoBackedTestCase(unittest.TestCase):
    """Fresh mongomock per test; indexes created so names lookups work."""

    def setUp(self):
        storage.reset_client(mongomock.MongoClient())
        storage.ensure_indexes()
        self.cards = storage.cards_collection()
        self.tags = storage.tags_collection()

    def tearDown(self):
        storage.reset_client(None)

    def _seed_tags(self, slugs=TAG_FIXTURES):
        self.tags.insert_many([
            {"_id": slug, "label": slug, "embedding": _pack(vec)}
            for slug, vec in slugs.items()
        ])

    def _seed_card(self, sid, name, tags):
        self.cards.insert_one({
            "_id": sid, "name": name, "names": [name.lower()], "tags": list(tags),
        })


# ---------------------------------------------------------------------------
# resolve_deck_cards
# ---------------------------------------------------------------------------

class TestResolveDeckCards(_MongoBackedTestCase):

    def test_resolves_known_names(self):
        self._seed_card("id-bolt", "Lightning Bolt", ["spot-removal", "burn-any"])
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
# load_tag_embeddings
# ---------------------------------------------------------------------------

class TestLoadTagEmbeddings(_MongoBackedTestCase):

    def test_loads_requested_tags_only(self):
        self._seed_tags()
        got = dp.load_tag_embeddings(
            ["spot-removal", "mana-rock"], self.tags
        )
        self.assertEqual(set(got), {"spot-removal", "mana-rock"})
        np.testing.assert_allclose(got["spot-removal"], [1.0, 0.0, 0.0, 0.0])

    def test_skips_tags_without_embedding(self):
        # A tag that exists but has no embedding (new import, pre-mtg-embed-tags)
        # must not land in the result dict.
        self.tags.insert_many([
            {"_id": "spot-removal", "label": "x", "embedding": _pack([1.0, 0.0, 0.0, 0.0])},
            {"_id": "new-tag", "label": "new"},
        ])
        got = dp.load_tag_embeddings(["spot-removal", "new-tag"], self.tags)
        self.assertEqual(set(got), {"spot-removal"})

    def test_empty_input_returns_empty_dict(self):
        self.assertEqual(dp.load_tag_embeddings([], self.tags), {})


# ---------------------------------------------------------------------------
# _cluster_tag_embeddings (HDBSCAN)
# ---------------------------------------------------------------------------

class TestClusterTagEmbeddings(unittest.TestCase):

    def test_three_tight_groups_yield_three_clusters(self):
        embeddings = {slug: np.array(vec) for slug, vec in TAG_FIXTURES.items()}
        groups = dp._cluster_tag_embeddings(
            embeddings, min_cluster_size=2, min_samples=1,
            cluster_selection_epsilon=0.0,
        )
        # Drop the noise bucket if present; three themes should emerge.
        real_clusters = {k: v for k, v in groups.items() if k != dp.NOISE_LABEL}
        self.assertEqual(len(real_clusters), 3)
        # Each cluster should contain exactly the slugs we planted near
        # the same basis vector.
        memberships = {frozenset(v) for v in real_clusters.values()}
        self.assertIn(frozenset({"spot-removal", "burn-any", "removal-creature"}), memberships)
        self.assertIn(frozenset({"mana-rock", "mana-dork", "ramp"}), memberships)
        self.assertIn(frozenset({"card-draw", "cantrip"}), memberships)

    def test_degenerate_universe_returns_empty(self):
        # With fewer tags than min_cluster_size, HDBSCAN can't do anything —
        # we bail early rather than letting sklearn raise.
        got = dp._cluster_tag_embeddings(
            {"only-one": np.array([1.0, 0.0, 0.0, 0.0])},
            min_cluster_size=2, min_samples=1, cluster_selection_epsilon=0.0,
        )
        self.assertEqual(got, {})


# ---------------------------------------------------------------------------
# _reassign_noise_to_nearest_cluster  (Option B)
# ---------------------------------------------------------------------------

class TestReassignNoise(unittest.TestCase):
    """The reassignment pass is a pure function — no Mongo involved —
    so these tests inject hand-built `groups` dicts and embeddings and
    assert on the shape of the output. HDBSCAN output isn't needed.
    """

    def _emb(self, mapping):
        return {k: np.array(v, dtype=np.float64) for k, v in mapping.items()}

    def test_noise_tag_near_cluster_gets_absorbed(self):
        # Two cluster tags on +x axis, one noise tag ALSO near +x. With
        # threshold 0.6 the noise tag should jump into the real cluster.
        embeddings = self._emb({
            "a": [1.0, 0.0, 0.0],
            "b": [0.95, 0.05, 0.0],
            "orphan": [0.9, 0.1, 0.1],
        })
        groups = {0: ["a", "b"], dp.NOISE_LABEL: ["orphan"]}
        got = dp._reassign_noise_to_nearest_cluster(
            groups, embeddings, threshold=0.6
        )
        self.assertIn("orphan", got[0])
        self.assertEqual(got[dp.NOISE_LABEL], [])

    def test_noise_tag_far_from_all_clusters_stays(self):
        # Cluster tags on +x; noise tag on +z. Cosine sim ≈ 0, well
        # below the 0.6 floor → tag stays in noise.
        embeddings = self._emb({
            "a": [1.0, 0.0, 0.0],
            "b": [0.95, 0.05, 0.0],
            "orphan": [0.0, 0.0, 1.0],
        })
        groups = {0: ["a", "b"], dp.NOISE_LABEL: ["orphan"]}
        got = dp._reassign_noise_to_nearest_cluster(
            groups, embeddings, threshold=0.6
        )
        self.assertEqual(got[0], ["a", "b"])
        self.assertEqual(got[dp.NOISE_LABEL], ["orphan"])

    def test_noise_picks_nearest_of_multiple_clusters(self):
        # Two clusters on +x and +z. Orphan is at (0.8, 0, 0.2) —
        # closer to the +x cluster. Must land there, not +z.
        embeddings = self._emb({
            "x-anchor":  [1.0, 0.0, 0.0],
            "x-pair":    [0.95, 0.05, 0.0],
            "z-anchor":  [0.0, 0.0, 1.0],
            "z-pair":    [0.0, 0.05, 0.95],
            "orphan":    [0.8, 0.0, 0.2],
        })
        groups = {
            0: ["x-anchor", "x-pair"],
            1: ["z-anchor", "z-pair"],
            dp.NOISE_LABEL: ["orphan"],
        }
        got = dp._reassign_noise_to_nearest_cluster(
            groups, embeddings, threshold=0.6
        )
        self.assertIn("orphan", got[0])
        self.assertNotIn("orphan", got[1])

    def test_threshold_boundary_strict(self):
        # The threshold is a FLOOR we strictly exceed. If a noise tag's
        # best sim equals the threshold exactly, it stays in noise —
        # ties shouldn't drag borderline tags into clusters.
        embeddings = self._emb({
            "a": [1.0, 0.0, 0.0],
            "b": [1.0, 0.0, 0.0],
            # orphan at the exact boundary (sim = 0.6).
            "orphan": [0.6, 0.8, 0.0],
        })
        groups = {0: ["a", "b"], dp.NOISE_LABEL: ["orphan"]}
        got = dp._reassign_noise_to_nearest_cluster(
            groups, embeddings, threshold=0.6
        )
        self.assertEqual(got[dp.NOISE_LABEL], ["orphan"])

    def test_negative_threshold_disables_pass(self):
        # A −1 threshold is the "feature off" sentinel. Groups must come
        # back untouched even for a tag that would otherwise be absorbed.
        embeddings = self._emb({
            "a": [1.0, 0.0, 0.0],
            "orphan": [0.99, 0.01, 0.0],
        })
        groups = {0: ["a", "a"], dp.NOISE_LABEL: ["orphan"]}
        got = dp._reassign_noise_to_nearest_cluster(
            groups, embeddings, threshold=-1.0
        )
        self.assertEqual(got, groups)

    def test_no_real_clusters_leaves_noise_unchanged(self):
        # Only a noise bucket exists — nothing to reassign TO.
        embeddings = self._emb({"orphan": [1.0, 0.0, 0.0]})
        groups = {dp.NOISE_LABEL: ["orphan"]}
        got = dp._reassign_noise_to_nearest_cluster(
            groups, embeddings, threshold=0.6
        )
        self.assertEqual(got, groups)

    def test_no_noise_bucket_is_a_no_op(self):
        embeddings = self._emb({"a": [1.0, 0.0], "b": [0.9, 0.1]})
        groups = {0: ["a", "b"]}
        got = dp._reassign_noise_to_nearest_cluster(
            groups, embeddings, threshold=0.6
        )
        self.assertEqual(got, groups)


# ---------------------------------------------------------------------------
# _centroid_and_label
# ---------------------------------------------------------------------------

class TestCentroidAndLabel(unittest.TestCase):

    def test_centroid_is_unit_length(self):
        embeddings = {s: np.array(TAG_FIXTURES[s]) for s in TAG_FIXTURES}
        centroid, _ = dp._centroid_and_label(
            ["spot-removal", "burn-any"], embeddings
        )
        self.assertAlmostEqual(float(np.linalg.norm(centroid)), 1.0, places=6)

    def test_label_is_one_of_input_tags(self):
        # The label contract is "closest tag to centroid", which within
        # a tight cluster can land on any member depending on which
        # satellite's offsets happen to point most in the aggregated
        # direction. The invariant we care about is that the label is
        # drawn from the cluster's own tag set — not some string from
        # elsewhere.
        embeddings = {s: np.array(TAG_FIXTURES[s]) for s in TAG_FIXTURES}
        tags = ["spot-removal", "burn-any", "removal-creature"]
        _, label = dp._centroid_and_label(tags, embeddings)
        self.assertIn(label, tags)


# ---------------------------------------------------------------------------
# build_deck_profile — end-to-end
# ---------------------------------------------------------------------------

class TestBuildDeckProfile(_MongoBackedTestCase):

    def _seed_full_deck(self):
        """Seed tag catalog + a small 'deck' spanning all three themes."""
        self._seed_tags()
        # 2 removal cards, 2 ramp cards, 1 draw card.
        self._seed_card("id-bolt",      "Lightning Bolt",   ["spot-removal", "burn-any"])
        self._seed_card("id-wrath",     "Wrath of God",     ["spot-removal", "removal-creature"])
        self._seed_card("id-sol",       "Sol Ring",         ["mana-rock"])
        self._seed_card("id-mystic",    "Birds of Paradise", ["mana-dork", "ramp"])
        self._seed_card("id-brainstorm", "Brainstorm",      ["card-draw", "cantrip"])

    def test_profile_reports_resolved_deck_and_missing(self):
        self._seed_full_deck()
        profile = dp.build_deck_profile(
            ["Lightning Bolt", "Wrath of God", "Nothing Here", "Sol Ring",
             "Birds of Paradise", "Brainstorm"],
            cards_coll=self.cards, tags_coll=self.tags,
        )
        self.assertEqual(
            set(profile.deck_card_ids),
            {"id-bolt", "id-wrath", "id-sol", "id-mystic", "id-brainstorm"},
        )
        self.assertEqual(profile.missing_names, ("Nothing Here",))

    def test_profile_surfaces_three_themes(self):
        self._seed_full_deck()
        profile = dp.build_deck_profile(
            ["Lightning Bolt", "Wrath of God", "Sol Ring",
             "Birds of Paradise", "Brainstorm"],
            cards_coll=self.cards, tags_coll=self.tags,
        )
        self.assertEqual(len(profile.clusters), 3)
        # Each cluster's tag set must match one of the three themes we
        # planted — we don't care which cluster index got which theme,
        # just that the memberships came out right.
        memberships = {frozenset(c.tags) for c in profile.clusters}
        self.assertEqual(memberships, {
            frozenset({"spot-removal", "burn-any", "removal-creature"}),
            frozenset({"mana-rock", "mana-dork", "ramp"}),
            frozenset({"card-draw", "cantrip"}),
        })

    def test_cluster_deck_card_ids_reflect_membership(self):
        self._seed_full_deck()
        profile = dp.build_deck_profile(
            ["Lightning Bolt", "Wrath of God", "Sol Ring",
             "Birds of Paradise", "Brainstorm"],
            cards_coll=self.cards, tags_coll=self.tags,
        )
        # Look clusters up by what's IN them, not by label (the label
        # is a nearest-tag pick that drifts with fixture choice).
        by_tags = {frozenset(c.tags): c for c in profile.clusters}
        removal = by_tags[frozenset({"spot-removal", "burn-any", "removal-creature"})]
        ramp    = by_tags[frozenset({"mana-rock", "mana-dork", "ramp"})]
        draw    = by_tags[frozenset({"card-draw", "cantrip"})]
        self.assertEqual(set(removal.deck_card_ids), {"id-bolt", "id-wrath"})
        self.assertEqual(set(ramp.deck_card_ids),    {"id-sol", "id-mystic"})
        self.assertEqual(set(draw.deck_card_ids),    {"id-brainstorm"})
        # And every cluster's label must come from its own tag set —
        # the pivotal invariant for interpretability downstream.
        for c in profile.clusters:
            self.assertIn(c.label, c.tags)

    def test_tag_without_stored_embedding_lands_in_noise(self):
        self._seed_tags()
        # Add a card whose tag list references a slug we never embedded.
        self._seed_card("id-bolt", "Lightning Bolt", ["spot-removal", "burn-any"])
        self._seed_card("id-sol", "Sol Ring", ["mana-rock", "ramp"])
        self._seed_card("id-weird", "Weird Card", ["spot-removal", "unknown-slug"])
        profile = dp.build_deck_profile(
            ["Lightning Bolt", "Sol Ring", "Weird Card"],
            cards_coll=self.cards, tags_coll=self.tags,
        )
        # unknown-slug has no embedding → degraded to noise. It must
        # appear in noise_tags, not in any cluster.
        self.assertIn("unknown-slug", profile.noise_tags)
        for cluster in profile.clusters:
            self.assertNotIn("unknown-slug", cluster.tags)

    def test_empty_deck_returns_empty_profile(self):
        self._seed_tags()
        profile = dp.build_deck_profile(
            [], cards_coll=self.cards, tags_coll=self.tags
        )
        self.assertEqual(profile.deck_card_ids, ())
        self.assertEqual(profile.clusters, ())
        self.assertEqual(profile.tag_universe, ())


# ---------------------------------------------------------------------------
# _render_profile
# ---------------------------------------------------------------------------

class TestRenderProfile(unittest.TestCase):

    def _make_profile(self, cluster_tags, cluster_card_ids):
        return dp.DeckProfile(
            deck_card_ids=tuple(cluster_card_ids),
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

    def test_missing_names_listed(self):
        profile = dp.DeckProfile(
            deck_card_ids=(),
            missing_names=("Nonexistent Card",),
            tag_universe=(),
            clusters=(),
            noise_tags=(),
        )
        text = dp._render_profile(profile, name_lookup={}, limit=10)
        self.assertIn("1 missing", text)
        self.assertIn("Nonexistent Card", text)


# ---------------------------------------------------------------------------
# main() CLI
# ---------------------------------------------------------------------------

def _run_cli(argv: list[str]) -> tuple[int, str, str]:
    """Invoke the CLI with argv, returning (rc, stdout, stderr)."""
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        rc = dp.main(argv)
    return rc, out.getvalue(), err.getvalue()


class TestMainCLI(_MongoBackedTestCase):

    def _seed_full_deck(self):
        """Same fixture TestBuildDeckProfile uses, lifted here so the CLI
        tests don't depend on test-class inheritance."""
        self._seed_tags()
        self._seed_card("id-bolt",      "Lightning Bolt",   ["spot-removal", "burn-any"])
        self._seed_card("id-wrath",     "Wrath of God",     ["spot-removal", "removal-creature"])
        self._seed_card("id-sol",       "Sol Ring",         ["mana-rock"])
        self._seed_card("id-mystic",    "Birds of Paradise", ["mana-dork", "ramp"])
        self._seed_card("id-brainstorm", "Brainstorm",      ["card-draw", "cantrip"])

    def test_prints_cluster_summary_from_positional_names(self):
        self._seed_full_deck()
        rc, out, _ = _run_cli([
            "Lightning Bolt", "Wrath of God", "Sol Ring",
            "Birds of Paradise", "Brainstorm",
        ])
        self.assertEqual(rc, 0)
        self.assertIn("resolved 5 unique cards", out)
        self.assertIn("theme ", out)
        # Card names appear in the rendered output (not just scryfall ids).
        self.assertIn("Lightning Bolt", out)
        # And the three themes line is correct.
        self.assertIn("3 theme(s) clustered", out)

    def test_file_input_reads_decklist(self):
        self._seed_full_deck()
        with TemporaryDirectory() as d:
            deck = Path(d) / "deck.txt"
            deck.write_text(
                "# commander\n"
                "1 Lightning Bolt\n"
                "1 Wrath of God\n"
                "1 Sol Ring\n"
                "1 Birds of Paradise\n"
                "1 Brainstorm\n",
                encoding="utf-8",
            )
            rc, out, _ = _run_cli(["--file", str(deck)])
        self.assertEqual(rc, 0)
        self.assertIn("resolved 5 unique cards", out)

    def test_limit_flag_truncates(self):
        self._seed_full_deck()
        rc, out, _ = _run_cli([
            "Lightning Bolt", "Wrath of God", "Sol Ring",
            "Birds of Paradise", "Brainstorm", "--limit", "1",
        ])
        self.assertEqual(rc, 0)
        # With only 1 item per cluster shown, truncation markers must appear.
        self.assertIn("more)", out)

    def test_min_cluster_size_pipes_through(self):
        # min_cluster_size=10 is larger than any single theme's membership,
        # so no themes should emerge — everything lands in noise.
        self._seed_full_deck()
        rc, out, _ = _run_cli([
            "Lightning Bolt", "Wrath of God", "Sol Ring",
            "Birds of Paradise", "Brainstorm", "--min-cluster-size", "10",
        ])
        self.assertEqual(rc, 0)
        self.assertIn("0 theme(s) clustered", out)

    def test_missing_names_surfaced_in_output(self):
        self._seed_full_deck()
        rc, out, _ = _run_cli(["Lightning Bolt", "No Such Card"])
        self.assertEqual(rc, 0)
        self.assertIn("missing: No Such Card", out)

    def test_empty_input_errors(self):
        with self.assertRaises(SystemExit):
            _run_cli([])

    def test_reassign_threshold_flag_pipes_through(self):
        # Force every tag into HDBSCAN noise by setting min_samples
        # higher than any single cluster's membership (the universe has
        # 8 tags in groups of 2–3; min_samples=4 means no tag has
        # enough dense neighbours to seed a cluster). Then set the
        # reassignment floor to 0.999 so nothing gets rescued.
        self._seed_full_deck()
        rc, out, _ = _run_cli([
            "Lightning Bolt", "Wrath of God", "Sol Ring",
            "Birds of Paradise", "Brainstorm",
            "--min-samples", "4",
            "--reassign-threshold", "0.999",
        ])
        self.assertEqual(rc, 0)
        self.assertIn("0 theme(s) clustered", out)
        # And at least one embedded tag ended up as noise (otherwise
        # we'd know the flags didn't thread through).
        self.assertNotIn("0 tag(s) in noise", out)


if __name__ == "__main__":
    unittest.main()
