"""Offline tests for weight_trainer.

Covers the scoring function (`score_clustering`) and the random-search
trainer (`train_weights`). Fixtures are hand-constructed so the expected
score is derivable with pen and paper and the training loop sees a
synthetic deck where one knob is unambiguously the right one to raise.
"""
from __future__ import annotations

import unittest

import numpy as np

from mtg_recommender import card_clusterer as cc
from mtg_recommender import embeddings as emb
from mtg_recommender import weight_trainer as wt


def _pack(vec: list[float]) -> bytes:
    """Shared packing helper (same shape as test_deck_profile uses)."""
    return emb._pack_embedding(np.array(vec, dtype=np.float32))


# ---------------------------------------------------------------------------
# score_clustering
# ---------------------------------------------------------------------------

class TestScoreClustering(unittest.TestCase):

    def test_perfect_lift_stratification_scores_one(self):
        # Two clusters: one all-high-lift, the other all-low-lift.
        # Within-cluster variance = 0, between-cluster variance = all of it.
        # eta^2 should be exactly 1.0 (within float wobble).
        clusters = [
            (["a1", "a2", "a3"], np.array([1.0, 0.0, 0.0, 0.0])),
            (["b1", "b2", "b3"], np.array([0.0, 1.0, 0.0, 0.0])),
        ]
        lifts = {
            "a1": 5.0, "a2": 5.0, "a3": 5.0,
            "b1": 1.0, "b2": 1.0, "b3": 1.0,
        }
        self.assertAlmostEqual(wt.score_clustering(clusters, lifts), 1.0, places=6)

    def test_random_lift_scores_near_zero(self):
        # Each cluster's lifts are a mix; the clustering carries no info
        # about lift distribution. eta^2 should be near 0.
        clusters = [
            (["a1", "a2"], np.array([1.0, 0.0, 0.0, 0.0])),
            (["b1", "b2"], np.array([0.0, 1.0, 0.0, 0.0])),
        ]
        # Both clusters have mean=3.0 (same cluster-level mean). The
        # between-cluster variance is 0 by construction, so eta^2 = 0
        # even though within-cluster variance is non-zero.
        lifts = {"a1": 1.0, "a2": 5.0, "b1": 1.0, "b2": 5.0}
        self.assertAlmostEqual(wt.score_clustering(clusters, lifts), 0.0, places=6)

    def test_partial_stratification_between_zero_and_one(self):
        # Mixed case: clusters have different means but non-zero
        # within-cluster variance. eta^2 is somewhere in (0, 1).
        clusters = [
            (["a1", "a2"], np.array([1.0, 0.0, 0.0, 0.0])),
            (["b1", "b2"], np.array([0.0, 1.0, 0.0, 0.0])),
        ]
        lifts = {"a1": 4.0, "a2": 6.0, "b1": 1.0, "b2": 3.0}
        score = wt.score_clustering(clusters, lifts)
        self.assertGreater(score, 0.0)
        self.assertLess(score, 1.0)

    def test_cards_without_lift_are_ignored(self):
        # Lift dict only covers half the cards. Score computed on the
        # covered half; absent entries don't crash and don't contribute.
        clusters = [
            (["a1", "a2", "a3"], np.array([1.0, 0.0, 0.0, 0.0])),
            (["b1", "b2", "b3"], np.array([0.0, 1.0, 0.0, 0.0])),
        ]
        lifts = {"a1": 5.0, "b1": 1.0}  # a2/a3/b2/b3 missing
        # Each cluster degenerates to a single scored card; both have
        # zero within-cluster variance and distinct means — eta^2 = 1.
        self.assertAlmostEqual(wt.score_clustering(clusters, lifts), 1.0, places=6)

    def test_empty_clusters_returns_zero(self):
        self.assertEqual(wt.score_clustering([], {"a": 1.0}), 0.0)

    def test_no_overlap_between_clusters_and_lifts_returns_zero(self):
        clusters = [(["a1"], np.array([1.0]))]
        self.assertEqual(wt.score_clustering(clusters, {"other": 5.0}), 0.0)

    def test_uniform_lift_returns_zero(self):
        # Every lift identical → total variance = 0 → undefined ratio;
        # the function clamps to 0.0 instead of returning NaN.
        clusters = [
            (["a1", "a2"], np.array([1.0, 0.0])),
            (["b1", "b2"], np.array([0.0, 1.0])),
        ]
        lifts = {"a1": 3.0, "a2": 3.0, "b1": 3.0, "b2": 3.0}
        self.assertEqual(wt.score_clustering(clusters, lifts), 0.0)


# ---------------------------------------------------------------------------
# train_weights
# ---------------------------------------------------------------------------

class TestTrainWeights(unittest.TestCase):
    """Synthetic 6-card deck where types IS the discriminating feature.

    card_vector near +x axis for every card (so base clustering can't
    separate them meaningfully). Three Creatures have high lift; three
    Instants have low lift. A good training run should raise the
    types weight — because with a non-trivial types weight HDBSCAN
    separates Creatures from Instants, which perfectly stratifies lift.
    """

    def _deck_and_lifts(self):
        deck = []
        lifts = {}
        for i, sid in enumerate(["c1", "c2", "c3"]):
            deck.append({
                "_id": sid,
                "tags": [],
                "card_vector": _pack([1.0, 0.01 * (i + 1), 0.0, 0.0]),
                "type_line": "Creature — Human",
            })
            lifts[sid] = 5.0
        for i, sid in enumerate(["i1", "i2", "i3"]):
            deck.append({
                "_id": sid,
                "tags": [],
                "card_vector": _pack([1.0, 0.01 * (i + 4), 0.0, 0.0]),
                "type_line": "Instant",
            })
            lifts[sid] = 1.0
        vocab = {"subtypes": ("Human",)}
        return deck, lifts, vocab

    def test_training_selects_discriminating_knob(self):
        deck, lifts, vocab = self._deck_and_lifts()
        result = wt.train_weights(
            deck, lifts,
            knobs=("types", "mana_cost"),
            n_trials=30,
            seed=42,
            weight_bounds=(0.0, 1.0),
            deck_vocab=vocab,
            min_cluster_size=2,
        )
        # Found weights that produce a meaningful lift split.
        self.assertGreater(result.best_score, 0.5)
        # The discriminating knob gets non-trivial weight.
        self.assertGreater(result.best_weights["types"], 0.1)
        # mana_cost is irrelevant to this split; the search may or may
        # not pick a positive value for it (randomness), but the score
        # story is governed by types. Just confirm it's recorded.
        self.assertIn("mana_cost", result.best_weights)

    def test_seed_determinism(self):
        # Same seed → same best weights, same history.
        deck, lifts, vocab = self._deck_and_lifts()
        a = wt.train_weights(deck, lifts, knobs=("types",), n_trials=10,
                             seed=7, deck_vocab=vocab, min_cluster_size=2)
        b = wt.train_weights(deck, lifts, knobs=("types",), n_trials=10,
                             seed=7, deck_vocab=vocab, min_cluster_size=2)
        self.assertEqual(a.best_weights, b.best_weights)
        self.assertEqual(a.best_score, b.best_score)
        self.assertEqual(
            [t.weights for t in a.history],
            [t.weights for t in b.history],
        )

    def test_zero_trials_returns_zero_weight_floor(self):
        # Degenerate: no trials ran, caller still gets a valid
        # TrainResult with every knob at 0.0 and score 0.0.
        deck, lifts, vocab = self._deck_and_lifts()
        result = wt.train_weights(
            deck, lifts, knobs=("types", "mana_cost"),
            n_trials=0, seed=0, deck_vocab=vocab, min_cluster_size=2,
        )
        self.assertEqual(result.best_score, 0.0)
        self.assertEqual(result.best_weights, {"types": 0.0, "mana_cost": 0.0})
        self.assertEqual(result.history, ())

    def test_history_length_matches_trial_count(self):
        deck, lifts, vocab = self._deck_and_lifts()
        result = wt.train_weights(
            deck, lifts, knobs=("types",), n_trials=5,
            seed=0, deck_vocab=vocab, min_cluster_size=2,
        )
        self.assertEqual(len(result.history), 5)


if __name__ == "__main__":
    unittest.main()
