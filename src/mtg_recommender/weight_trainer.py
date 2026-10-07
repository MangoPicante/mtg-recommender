"""Per-deck training of cluster-mode structural feature weights.

`mtg-deck-profile --mode cluster` has seven knobs that reweight how
structural features (types, mana_cost, keywords, colors, color_identity,
power, toughness) contribute to the clustering (see `card_clusterer`
for the mechanics). Hand-picking these per deck is tedious. This module
automates it: given a deck + a commander's EDHREC per-card lift scores,
random-searches over knob weights and returns the configuration that
best stratifies lift across clusters.

Scoring (see `score_clustering`): eta-squared on lift — the ratio of
between-cluster lift variance to total lift variance, in [0, 1]. A score
of 1.0 means the clustering perfectly explains the deck's lift
distribution (each cluster is uniform in lift, clusters have distinct
means); 0.0 means the clustering tells us nothing about lift. This
targets the recommender's end goal from `PLAN.md` Phase 3: "a cluster
whose member cards consistently show high lift is a validated theme for
this commander." High eta-squared = that validation is strong on BOTH
tails (the clustering also pushes low-lift cards out of the high-lift
groups).

Search (see `train_weights`): plain random search over each knob's
weight in a user-supplied bound. Random search is intentional: HDBSCAN
is non-differentiable, our search space is low-dimensional (≤ 7 knobs),
each trial is cheap (<100 ms for a 100-card deck), and random search
avoids the smoothness assumption Bayesian optimisation would need. If
future scale makes trial budget the bottleneck, swap in Optuna behind
the same function signature.

Why not optimise mean lift or a simpler objective:
    - Sum/mean of per-card lift is invariant under clustering — it
      only depends on which cards are in the deck.
    - Max cluster lift is noisy (one lucky small cluster dominates).
    - Eta-squared is standard, bounded, and sensitive to clustering
      structure in a way the above are not.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Optional, Sequence

import numpy as np

from . import card_clusterer as cc


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

# All currently-shipped structural feature names. Default knob set for
# training when the caller doesn't restrict it. Reflects the keys in
# cc._FEATURE_BUILDERS at the time of writing — if a new builder lands,
# add it here too.
DEFAULT_TRAIN_KNOBS: tuple[str, ...] = (
    "types", "mana_cost", "keywords",
    "colors", "color_identity",
    "power", "toughness",
)

# Default per-knob weight sampling range. 0.0 lets the search freely
# disable a feature; 1.0 is a sensible ceiling — once a weight passes
# roughly 1.0 its sub-vector starts to dominate the already-unit
# card_vector and the clustering degenerates toward "only this
# structural feature matters."
DEFAULT_WEIGHT_BOUNDS: tuple[float, float] = (0.0, 1.0)

# Default trial budget. 50 random configs over 7 knobs is enough to find
# a near-optimum in practice on a 100-card deck without the trainer
# feeling slow (sub-second total on commodity hardware).
DEFAULT_TRAIN_TRIALS: int = 50


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_clustering(
    clusters: Sequence[tuple[Sequence[str], np.ndarray]],
    card_lifts: Mapping[str, float],
) -> float:
    """Eta-squared of per-card lift explained by cluster membership.

    Returns a scalar in [0.0, 1.0]:
      1.0 — every cluster has uniform lift internally AND distinct lift
            mean from the other clusters. Perfect lift stratification.
      0.0 — clustering tells us nothing about lift. Either every cluster
            has the same mean, or lift variance within clusters equals
            total variance.

    Cards without an entry in `card_lifts` are dropped from the sum —
    EDHREC doesn't rate every card in a deck (fringe picks, basics,
    recent printings). We only train on what we have signal for.

    Clusters that end up with fewer than two scored cards contribute to
    the overall mean but not to within-cluster variance (a one-card
    cluster has zero variance by definition). This matches the standard
    ANOVA treatment.

    Edge cases that return 0.0 (not NaN):
      - no clusters, or no cards with lift
      - every lift value equal (total variance = 0; the ratio is
        undefined mathematically, but "the data isn't informative"
        maps cleanly to a floor score)
    """
    # Collect (lift, cluster_idx) tuples for every card we have a lift
    # for. A single ndarray + a parallel group label array is the clean
    # shape for the ANOVA math below.
    per_cluster_lifts: list[list[float]] = []
    for sids, _centroid in clusters:
        bucket: list[float] = []
        for sid in sids:
            v = card_lifts.get(sid)
            if v is None:
                continue
            bucket.append(float(v))
        if bucket:
            per_cluster_lifts.append(bucket)

    if not per_cluster_lifts:
        return 0.0
    all_lifts = np.concatenate([np.array(b, dtype=np.float64) for b in per_cluster_lifts])
    n_total = all_lifts.size
    if n_total < 2:
        return 0.0

    grand_mean = all_lifts.mean()
    # Total sum of squares — denominator in eta-squared.
    ss_total = float(((all_lifts - grand_mean) ** 2).sum())
    if ss_total <= 0.0:
        # Every lift identical; clustering can't explain non-existent variance.
        return 0.0

    # Between-cluster sum of squares: cluster_size * (cluster_mean - grand_mean)^2.
    ss_between = 0.0
    for bucket in per_cluster_lifts:
        arr = np.array(bucket, dtype=np.float64)
        ss_between += float(arr.size * (arr.mean() - grand_mean) ** 2)
    # Clamp to [0, 1] against float wobble on near-degenerate inputs.
    return max(0.0, min(1.0, ss_between / ss_total))


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TrialResult:
    """One random-search trial's weights + score. Immutable for safe
    aggregation into history lists."""

    weights: dict[str, float]
    score: float


@dataclass(frozen=True)
class TrainResult:
    """The best configuration the search found + the full search history.

    Attributes:
      best_weights: knob-name → weight; the configuration with the
                    highest score across all trials.
      best_score:   that configuration's eta-squared.
      history:      every trial in chronological order. Useful to
                    diagnose a flat search (every score ~equal → the
                    objective isn't sensitive to these knobs on this
                    deck, worth telling the user to pick weights by
                    hand or train on a different commander).
    """

    best_weights: dict[str, float]
    best_score: float
    history: tuple[TrialResult, ...]


def train_weights(
    deck_cards: Iterable[dict],
    card_lifts: Mapping[str, float],
    *,
    knobs: Sequence[str] = DEFAULT_TRAIN_KNOBS,
    n_trials: int = DEFAULT_TRAIN_TRIALS,
    seed: int = 0,
    weight_bounds: tuple[float, float] = DEFAULT_WEIGHT_BOUNDS,
    deck_vocab: Optional[Mapping[str, tuple[str, ...]]] = None,
    min_cluster_size: int = cc.DEFAULT_CLUSTER_MIN_SIZE,
    cluster_selection_epsilon: float = cc.DEFAULT_CLUSTER_SELECTION_EPSILON,
) -> TrainResult:
    """Random-search knob weights that maximise `score_clustering`.

    Each trial samples one weight per knob uniformly in `weight_bounds`,
    clusters the deck with those weights, scores the result against
    `card_lifts`, and records the trial. The best trial is returned.

    `knobs` restricts which features participate — defaults to all seven
    shipped builders. Pass a subset like `("types", "mana_cost")` to tune
    only those and leave the rest at 0.0; makes the search cheaper and
    the result more interpretable.

    `deck_cards` must be materialised (list, not generator) because the
    inner loop iterates it `n_trials` times. The caller usually already
    has it in a list via `resolve_deck_cards`.

    The clustering kwargs (`min_cluster_size`, `cluster_selection_epsilon`)
    are passed through to each trial unchanged — we tune feature
    weights, not HDBSCAN hyperparameters, so the user's deck-size
    choices stay fixed across the search.

    Deterministic given a seed: numpy's RNG is seeded locally so the
    same (deck, lifts, seed) always produces the same TrainResult.
    """
    deck_list = list(deck_cards)
    # Materialise the knob tuple so the weight sampling order is stable
    # across trials (also protects against the caller passing a set).
    knob_tuple = tuple(knobs)

    rng = np.random.default_rng(seed)
    low, high = weight_bounds

    history: list[TrialResult] = []
    best: Optional[TrialResult] = None
    for _ in range(n_trials):
        # One weight per knob, uniform in bounds. np.random.default_rng
        # gives us independent streams per seed so trials don't leak
        # state between successive runs.
        sample = rng.uniform(low, high, size=len(knob_tuple))
        trial_weights = {k: float(w) for k, w in zip(knob_tuple, sample)}
        clusters, _noise = cc.cluster_cards_by_vector(
            deck_list,
            min_cluster_size=min_cluster_size,
            cluster_selection_epsilon=cluster_selection_epsilon,
            feature_weights=trial_weights,
            deck_vocab=deck_vocab or {},
        )
        score = score_clustering(clusters, card_lifts)
        trial = TrialResult(weights=trial_weights, score=score)
        history.append(trial)
        if best is None or trial.score > best.score:
            best = trial

    # best is None only if n_trials == 0; in that case we return the
    # zero-weight config at score 0.0 so the caller still gets a
    # structurally-valid TrainResult.
    if best is None:
        best = TrialResult(
            weights={k: 0.0 for k in knob_tuple}, score=0.0,
        )
    return TrainResult(
        best_weights=dict(best.weights),
        best_score=best.score,
        history=tuple(history),
    )
