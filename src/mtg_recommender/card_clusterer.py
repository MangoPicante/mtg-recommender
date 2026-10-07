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

import re
from collections import Counter
from typing import Callable, Iterable, Mapping, Optional

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
# Feature-vector helpers
# ---------------------------------------------------------------------------
#
# These convert a card doc's structured fields (type_line, mana_cost, …) into
# dense numeric vectors that can be concatenated onto `card_vector` so HDBSCAN
# can weight them. Each is a pure function of its input — no Mongo, no I/O.
#
# How weighting composes: `cluster_cards_by_vector` L2-normalizes each
# sub-vector, multiplies by its weight, and concatenates onto `card_vector`
# (also unit-norm from Phase 2's fuse step). Under cosine distance that makes
# the overall similarity a weighted mean of per-feature cosines with weights
# proportional to w_i².

# Canonical Magic supertypes. "Elite", "Host", "Ongoing", "Token" are rare
# but included so an odd card doesn't silently fall through — a type we
# don't recognise would otherwise get mis-classified as a card type below.
_SUPERTYPES: tuple[str, ...] = (
    "Basic", "Legendary", "Snow", "World", "Elite", "Host", "Ongoing", "Token",
)
_SUPERTYPES_SET: frozenset[str] = frozenset(_SUPERTYPES)

# Canonical Magic card types. Covers everything printed in official sets.
# "Tribal" is the legacy spelling of "Kindred"; both are listed so an older
# cached doc isn't lost on a vocabulary rename.
_TYPES: tuple[str, ...] = (
    "Artifact", "Battle", "Conspiracy", "Creature", "Dungeon",
    "Enchantment", "Instant", "Kindred", "Land", "Phenomenon",
    "Plane", "Planeswalker", "Scheme", "Sorcery", "Tribal", "Vanguard",
)
_TYPES_SET: frozenset[str] = frozenset(_TYPES)


def parse_type_line(type_line: Optional[str]) -> tuple[list[str], list[str], list[str]]:
    """Split a type_line into (supertypes, types, subtypes).

    Scryfall uses the em-dash " — " (U+2014, surrounded by spaces) to divide
    the type portion from the subtype portion: "Legendary Creature — Human
    Wizard" → supers=["Legendary"], types=["Creature"], subs=["Human", "Wizard"].
    Multi-face cards carry a " // "-joined combined line (see
    `scryfall_fetch.pick`); each face is parsed independently and the
    results are concatenated — order within a face is preserved, duplicates
    across faces are kept (lets a Human/Human DFC's "Human" count twice if
    the caller cares, though `build_type_vector` dedupes to multi-hot).

    An empty or missing type_line returns three empty lists — defensive
    against the odd test fixture that forgets the field.
    """
    if not type_line:
        return [], [], []
    supers: list[str] = []
    types: list[str] = []
    subs: list[str] = []
    for face in type_line.split(" // "):
        parts = face.split(" — ", 1)
        for word in parts[0].split():
            if word in _SUPERTYPES_SET:
                supers.append(word)
            else:
                # Anything to the left of the em-dash that isn't a known
                # supertype is a card type by elimination. A truly unknown
                # token (new type-line vocab from a future set) still lands
                # here; it won't match _TYPES_SET in build_type_vector, so
                # it contributes zero — safer than failing hard.
                types.append(word)
        if len(parts) == 2:
            subs.extend(parts[1].split())
    return supers, types, subs


def build_type_vector(
    doc: dict, deck_subtype_vocab: tuple[str, ...]
) -> np.ndarray:
    """Multi-hot encoding of a card's supertypes + types + subtypes.

    Dimension layout: fixed `_SUPERTYPES` block, fixed `_TYPES` block, then
    one dim per subtype in `deck_subtype_vocab`. The subtype block is
    deck-local — a vocab of every subtype actually appearing in the deck —
    so a 100-card deck gets a vector around 24 + len(deck_vocab) dims, not
    the thousands of subtypes Scryfall knows about. Caller is responsible
    for building the vocab once per deck and passing the same tuple for
    every card so dims line up.

    Returns a float32 ndarray with 0.0 / 1.0 entries. Unknown types (future
    sets, typos) silently contribute zero — see `parse_type_line`'s note.
    """
    supers, types, subs = parse_type_line(doc.get("type_line") or "")
    super_set = set(supers)
    type_set = set(types)
    sub_set = set(subs)
    dim = len(_SUPERTYPES) + len(_TYPES) + len(deck_subtype_vocab)
    vec = np.zeros(dim, dtype=np.float32)
    for i, s in enumerate(_SUPERTYPES):
        if s in super_set:
            vec[i] = 1.0
    offset = len(_SUPERTYPES)
    for i, t in enumerate(_TYPES):
        if t in type_set:
            vec[offset + i] = 1.0
    offset += len(_TYPES)
    for i, sub in enumerate(deck_subtype_vocab):
        if sub in sub_set:
            vec[offset + i] = 1.0
    return vec


# Mana-cost sub-vector layout — 10 dims total:
#   0:         generic (sum of all plain numeric pips, so {3}{R} → 3.0)
#   1-5:       W, U, B, R, G  (count of each coloured pip)
#   6:         X / Y / Z  (variable costs, count of pips not value)
#   7:         hybrid pips (any {A/B} that isn't phyrexian)
#   8:         phyrexian pips (any symbol containing "P")
#   9:         snow pips  ({S})
#
# A hybrid pip is counted in bucket 7 only — not split half-and-half across
# its two colors — because for clustering intent two hybrid cards feel more
# "alike" than a hybrid and a strictly-mono card of either colour. The same
# argument applies to phyrexian: its payment flexibility is the signal.
_MANA_VEC_DIM = 10
_COLOR_INDEX: dict[str, int] = {"W": 1, "U": 2, "B": 3, "R": 4, "G": 5}
_VARIABLE_SYMBOLS: frozenset[str] = frozenset({"X", "Y", "Z"})
_SYMBOL_RE = re.compile(r"\{([^}]+)\}")


def build_mana_cost_vector(mana_cost: Optional[str]) -> np.ndarray:
    """Convert a mana_cost string into a 10-dim symbol-count vector.

    Multi-face cards store mana_cost joined with " // "; the regex pulls out
    every `{SYMBOL}` across both faces so an MDFC with a land back doesn't
    silently drop its front-face cost. See `_MANA_VEC_DIM` comment above for
    the per-bucket semantics.
    """
    vec = np.zeros(_MANA_VEC_DIM, dtype=np.float32)
    if not mana_cost:
        return vec
    for symbol in _SYMBOL_RE.findall(mana_cost):
        s = symbol.upper()
        if s in _VARIABLE_SYMBOLS:
            vec[6] += 1.0
            continue
        if s == "S":
            vec[9] += 1.0
            continue
        if "/" in s:
            # Hybrid or phyrexian — the "P" branch is checked first because
            # a phyrexian hybrid like {W/U/P} would otherwise be mis-bucketed.
            if "P" in s.split("/"):
                vec[8] += 1.0
            else:
                vec[7] += 1.0
            continue
        if s in _COLOR_INDEX:
            vec[_COLOR_INDEX[s]] += 1.0
            continue
        if s.isdigit():
            vec[0] += float(s)
        # Anything else (ad-hoc symbols from un-sets, half-mana) silently
        # drops — low enough signal to not warrant a bucket.
    return vec


# Feature-name → builder registry. The builder receives the card doc and a
# `deck_vocab` dict so a feature can pull its own deck-local vocabulary
# (currently only subtypes); features without vocab (mana_cost) ignore it.
# Adding a feature in a future PR is: write the builder, add it here, add a
# CLI knob. The rest of the clustering plumbing is unchanged.
_FEATURE_BUILDERS: dict[str, Callable[[dict, Mapping[str, tuple[str, ...]]], np.ndarray]] = {
    "types": lambda doc, vocab: build_type_vector(doc, vocab.get("subtypes", ())),
    "mana_cost": lambda doc, vocab: build_mana_cost_vector(doc.get("mana_cost")),
}


def _augmented_vector(
    base: np.ndarray,
    doc: dict,
    feature_weights: Mapping[str, float],
    deck_vocab: Mapping[str, tuple[str, ...]],
) -> np.ndarray:
    """Concatenate L2-normalized * weighted feature sub-vectors onto `base`.

    `base` is the card's `card_vector` (already unit-norm from Phase 2
    fuse). Each feature with a positive weight contributes an additional
    sub-vector; features at weight 0.0 are skipped so the output
    dimensionality only grows with the active knobs. An all-zero
    sub-vector (card with empty mana_cost, say) is appended at full
    (zero) magnitude so dims line up across the deck — the per-feature
    cosine for it is then undefined but it contributes zero to the
    weighted sum, which is the desired behaviour (no spurious similarity).

    Unknown feature names raise `KeyError` — explicit over silent so a
    typo in the CLI dispatch doesn't quietly disable a knob.
    """
    pieces: list[np.ndarray] = [base]
    for name, weight in feature_weights.items():
        if weight <= 0.0:
            continue
        builder = _FEATURE_BUILDERS[name]
        sub = builder(doc, deck_vocab).astype(np.float32, copy=False)
        norm = float(np.linalg.norm(sub))
        if norm > 0:
            sub = sub / norm
        pieces.append(sub * float(weight))
    if len(pieces) == 1:
        return base
    return np.concatenate(pieces)


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------

def cluster_cards_by_vector(
    deck_cards: Iterable[dict],
    *,
    min_cluster_size: int = DEFAULT_CLUSTER_MIN_SIZE,
    cluster_selection_epsilon: float = DEFAULT_CLUSTER_SELECTION_EPSILON,
    feature_weights: Optional[Mapping[str, float]] = None,
    deck_vocab: Optional[Mapping[str, tuple[str, ...]]] = None,
) -> tuple[list[tuple[list[str], np.ndarray]], list[str]]:
    """Cluster the deck's card_vectors with HDBSCAN. Return clusters + noise ids.

    Each `deck_cards` entry is expected to carry an unpacked
    `card_vector` (bytes/Binary); cards without one are silently
    excluded from clustering (they have no vector to cluster), which
    means they also don't appear in the noise bucket — a caller that
    cares about surfacing "cards we couldn't cluster at all" tracks
    that separately upstream.

    `feature_weights`: optional {feature_name: weight} dict that
    concatenates weighted structural sub-vectors onto each card's
    base `card_vector` before clustering. See `_FEATURE_BUILDERS` for
    the available names (currently "types", "mana_cost") and
    `_augmented_vector` for the composition rules. Omitted or all-zero
    weights preserve the previous "card_vector-only" behaviour.

    `deck_vocab`: deck-local vocabularies some features need (e.g.
    `{"subtypes": (…)}` for the type feature). Caller is responsible
    for building this once per deck.

    Return value: `(clusters, noise_card_ids)`.
      clusters: list of (card_ids, unit_centroid). card_ids sorted;
                centroid is L2-normalized mean of the cluster's
                card_vectors in BASE (card_vector) space — the
                structural sub-vectors used for clustering don't
                propagate into the centroid, because downstream code
                (candidate ranking) works in base space.
      noise_card_ids: cards HDBSCAN labelled -1 (density orphans),
                       sorted. The CLI render shows these as
                       "unassigned".

    Degenerate inputs (fewer than `min_cluster_size` cards with
    card_vectors) short-circuit to `([], [])` rather than letting
    sklearn raise — a small deck shouldn't crash the profile.
    """
    feature_weights = feature_weights or {}
    deck_vocab = deck_vocab or {}

    # Keep base + augmented separately: HDBSCAN clusters on the augmented
    # vectors (so the weighted structural sub-vectors influence grouping),
    # but centroids are computed from the base `card_vector` so downstream
    # ranking (Phase 3 step 4) stays in the same space as the stored
    # per-card vectors.
    with_vec: list[tuple[str, np.ndarray, np.ndarray]] = []
    for doc in deck_cards:
        sid = doc.get("_id")
        packed = doc.get("card_vector")
        if sid is None or packed is None:
            continue
        base = emb._unpack_embedding(packed)
        augmented = _augmented_vector(base, doc, feature_weights, deck_vocab)
        with_vec.append((sid, base, augmented))
    if len(with_vec) < min_cluster_size:
        return [], []

    sids = [sid for sid, _, _ in with_vec]
    base_matrix = np.vstack([base for _, base, _ in with_vec])
    aug_matrix = np.vstack([aug for _, _, aug in with_vec])
    labels = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=1,  # same inclusivity default we used for tag clustering
        cluster_selection_epsilon=cluster_selection_epsilon,
        metric="cosine",
        copy=True,
    ).fit_predict(aug_matrix)

    groups: dict[int, list[int]] = {}
    for idx, lbl in enumerate(labels):
        groups.setdefault(int(lbl), []).append(idx)

    noise: list[str] = sorted(sids[i] for i in groups.pop(_NOISE_LABEL, []))

    clusters: list[tuple[list[str], np.ndarray]] = []
    for lbl in sorted(groups):
        idxs = groups[lbl]
        member_ids = sorted(sids[i] for i in idxs)
        mean = base_matrix[idxs].mean(axis=0)
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
