"""Unsupervised card clustering mode for `mtg-deck-profile`.

An alternative to the theme classifier. Where the theme flow starts
from a curated top-level vocabulary and sorts cards into those bins,
this flow goes the other direction: cluster cards directly by their
`card_vector` (the Phase 2 fused oracle_text + aggregated tag blend),
then label each resulting group by the top-level oracle tag its member
cards most frequently share.

Shape:
    cluster_cards_by_vector(deck_cards) -> list[(cluster_card_ids,
                                                  centroid)]
    label_cluster(cluster_card_ids, cards_by_id, by_slug, blocklist,
                  fallback_name) -> str

Why this is useful alongside the theme flow:
    - Discovers deck themes that aren't in the curated top-level set
      (e.g. deck-specific archetypes the hierarchy doesn't name).
    - Fits decks that straddle multiple canonical themes cleanly
      (the clustering shows what the oracle_text + tag blend
      actually looks like).

Why it's not the default:
    - Theme labels are stable across decks ("removal" always means
      "removal"). Cluster labels drift with deck composition.
    - HDBSCAN parameters are more fiddly than a top-level tag vocabulary.
    - EDHREC lift (Phase 3 step 3) works most cleanly against stable
      labels — the recommender's attribution strings are more useful
      when "removal" means the same thing from one deck to the next.

Clustering: `sklearn.cluster.HDBSCAN` with cosine metric, same
machinery we use elsewhere. Noise cards (HDBSCAN label -1) come back
as the second return value for the caller to surface as unassigned.

Labelling: for each cluster, build a Counter over the top-level
ancestors reachable from each member card's tag array, deduplicated
per card so a card with ten tags all under "removal" doesn't drown
out a different card entirely. Blocklisted top-levels (same
DEFAULT_BLOCKLIST that theme_classifier uses) are stripped before
the vote. If nothing wins, the caller gets `fallback_name` back.
"""
from __future__ import annotations

from collections import Counter
from typing import Iterable, Optional

import numpy as np
from sklearn.cluster import HDBSCAN

from . import embeddings as emb

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

# HDBSCAN parameters tuned for ~100-card decks. 3 is big enough that
# noise rarely rescues a one-off; a Commander deck typically has 5+
# cards per real theme so this is still permissive.
DEFAULT_CLUSTER_MIN_SIZE = 3

# cluster_selection_epsilon merges clusters whose boundary cosine
# distance is below this. 0.0 preserves HDBSCAN's natural splits; try
# 0.1-0.2 to lump similar card groups.
DEFAULT_CLUSTER_SELECTION_EPSILON = 0.0

# HDBSCAN marks orphan points as -1. We surface those cards to the
# caller as "unassigned" rather than inventing a cluster for them.
_NOISE_LABEL = -1


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------

def cluster_cards_by_vector(
    deck_cards: Iterable[dict],
    *,
    min_cluster_size: int = DEFAULT_CLUSTER_MIN_SIZE,
    cluster_selection_epsilon: float = DEFAULT_CLUSTER_SELECTION_EPSILON,
) -> tuple[list[tuple[list[str], np.ndarray]], list[str]]:
    """Cluster the deck's card_vectors with HDBSCAN. Return clusters + noise ids.

    Each `deck_cards` entry is expected to carry an unpacked
    `card_vector` (bytes/Binary); cards without one are silently
    excluded from clustering (they have no vector to cluster), which
    means they also don't appear in the noise bucket — a caller that
    cares about surfacing "cards we couldn't cluster at all" tracks
    that separately upstream.

    Return value: `(clusters, noise_card_ids)`.
      clusters: list of (card_ids, unit_centroid). card_ids sorted;
                centroid is L2-normalized mean of the cluster's
                card_vectors.
      noise_card_ids: cards HDBSCAN labelled -1 (density orphans),
                       sorted. The CLI render shows these as
                       "unassigned".

    Degenerate inputs (fewer than `min_cluster_size` cards with
    card_vectors) short-circuit to `([], [])` rather than letting
    sklearn raise — a small deck shouldn't crash the profile.
    """
    with_vec: list[tuple[str, np.ndarray]] = []
    for doc in deck_cards:
        sid = doc.get("_id")
        packed = doc.get("card_vector")
        if sid is None or packed is None:
            continue
        with_vec.append((sid, emb._unpack_embedding(packed)))
    if len(with_vec) < min_cluster_size:
        return [], []

    sids = [sid for sid, _ in with_vec]
    matrix = np.vstack([vec for _, vec in with_vec])
    labels = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=1,  # same inclusivity default we used for tag clustering
        cluster_selection_epsilon=cluster_selection_epsilon,
        metric="cosine",
        copy=True,
    ).fit_predict(matrix)

    groups: dict[int, list[int]] = {}
    for idx, lbl in enumerate(labels):
        groups.setdefault(int(lbl), []).append(idx)

    noise: list[str] = sorted(sids[i] for i in groups.pop(_NOISE_LABEL, []))

    clusters: list[tuple[list[str], np.ndarray]] = []
    for lbl in sorted(groups):
        idxs = groups[lbl]
        member_ids = sorted(sids[i] for i in idxs)
        mean = matrix[idxs].mean(axis=0)
        norm = float(np.linalg.norm(mean))
        centroid = mean / norm if norm > 0 else mean
        clusters.append((member_ids, centroid))
    return clusters, noise


# ---------------------------------------------------------------------------
# Labelling
# ---------------------------------------------------------------------------

def _top_ancestors(
    slug: str,
    by_slug: dict[str, dict],
    memo: dict[str, frozenset[str]],
) -> frozenset[str]:
    """Return the set of top-level tag slugs reachable from `slug`.

    Walks up through `parent_slugs`. A slug with no parents IS a
    top-level and returns `{slug}`. Memoised because many cards share
    the same deep tags and we don't want to re-walk the hierarchy
    for each occurrence.

    Guards against cycles (shouldn't exist in Scryfall data, but we
    don't want a stack overflow if upstream drifts) by seeding `memo`
    with the empty set before recursing.
    """
    if slug in memo:
        return memo[slug]
    # Pre-populate with empty to detect cycles mid-recursion.
    memo[slug] = frozenset()
    doc = by_slug.get(slug)
    if doc is None:
        return memo[slug]
    parents = doc.get("parent_slugs") or []
    if not parents:
        tops = frozenset({slug})
    else:
        acc: set[str] = set()
        for p in parents:
            acc |= _top_ancestors(p, by_slug, memo)
        tops = frozenset(acc)
    memo[slug] = tops
    return tops


def most_frequent_top_level(
    card_ids: Iterable[str],
    cards_by_id: dict[str, dict],
    by_slug: dict[str, dict],
    *,
    blocklist: frozenset[str],
) -> Optional[str]:
    """Pick the most common non-blocklisted top-level ancestor.

    For each card in the cluster, trace every tag up to its top-level
    ancestor(s), dedupe per card (a card with 10 removal-flavored tags
    counts once for `removal`, not ten times), and count across the
    cluster. Blocklisted top-levels are stripped before the count.

    Returns the winning slug, or None if the cluster has no cards with
    any non-blocklisted top-level ancestor. Caller picks a fallback
    label in that case.

    Ties: `Counter.most_common` returns insertion order for ties, and
    tag iteration order through `by_slug` is deterministic in Python
    3.7+, so the output is stable for a given input.
    """
    counts: Counter[str] = Counter()
    memo: dict[str, frozenset[str]] = {}
    for sid in card_ids:
        card = cards_by_id.get(sid)
        if card is None:
            continue
        tops_for_card: set[str] = set()
        for slug in card.get("tags") or []:
            tops_for_card |= _top_ancestors(slug, by_slug, memo)
        tops_for_card -= blocklist
        for top in tops_for_card:
            counts[top] += 1
    if not counts:
        return None
    return counts.most_common(1)[0][0]


def label_cluster(
    card_ids: Iterable[str],
    cards_by_id: dict[str, dict],
    by_slug: dict[str, dict],
    *,
    blocklist: frozenset[str],
    fallback_name: str,
) -> str:
    """Convenience wrapper: return the winning top-level or `fallback_name`."""
    winner = most_frequent_top_level(
        card_ids, cards_by_id, by_slug, blocklist=blocklist,
    )
    return winner or fallback_name
