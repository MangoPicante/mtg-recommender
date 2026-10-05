"""Per-deck tag clustering (Phase 3 step 2).

Given a decklist, this module:
  1. Resolves each card name against the `cards` collection (via the
     multikey `names` index — ambiguous names get the first match,
     fully-missing names are reported but don't fail the build).
  2. Collects the union of tag slugs across the resolved cards.
  3. Loads the tag embeddings from the `tags` collection.
  4. Clusters them into themes using `sklearn.cluster.HDBSCAN` with
     cosine distance — auto-picks k, labels density-sparse tags as
     noise (−1).
  5. Returns a `DeckProfile` carrying per-cluster centroids, member
     tags, and the deck's cards whose tag array intersects each cluster.

What a cluster represents: a semantic group of tags that co-occur in
this deck ("spot-removal + burn-any + removal-creature", say). The
Phase 3 recommender uses each cluster's centroid as a query vector in
`card_vector` space, scored against candidates for similarity, with
EDHREC lift (from `edhrec_fetch`) layered on as a per-commander
quality signal.

Scope boundary: this module just builds the profile. Cluster-level
EDHREC evaluation lives in Phase 3 step 3 (a separate module); final
candidate ranking is step 4. Keeping the pieces separate lets each
step be tested and tuned in isolation.

Minimum deck size: the HDBSCAN defaults assume at least a few tags.
With `min_cluster_size=2` (our default), a deck whose unique-tag count
is under ~5 will likely return all-noise. That's fine — the caller
checks `profile.clusters` and falls back to a tag-free ranking strategy
if the deck's too thin for themes to emerge.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
from pymongo.collection import Collection
from sklearn.cluster import HDBSCAN

from . import embeddings as emb
from . import scryfall_fetch as sf
from . import storage

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

# HDBSCAN's min_cluster_size: the smallest group of tags that counts as
# a theme. 2 is permissive (a pair of co-occurring tags IS a theme in
# a deck context), 3 is stricter. Overridable per call.
DEFAULT_MIN_CLUSTER_SIZE = 2

# HDBSCAN's min_samples: how many points must be in a core point's
# epsilon-neighborhood for it to count as "dense enough" to seed a
# cluster. Lower = more inclusive (fewer orphans flagged as noise);
# sklearn defaults to min_cluster_size when None, which is stricter
# than we want for ~small decks. 1 means every tag can be a core
# point, so tags only land in noise when they're genuinely isolated
# in embedding space.
DEFAULT_MIN_SAMPLES = 1

# HDBSCAN's cluster_selection_epsilon: merges clusters whose boundary
# distance falls below this value. 0.0 (default) keeps HDBSCAN's
# natural cluster boundaries; raise it to collapse near-themes that
# split too eagerly. Users wanting more inclusive clusters can set
# this to 0.1–0.3 (cosine distance) without changing anything else.
DEFAULT_CLUSTER_SELECTION_EPSILON = 0.0

# Cosine-similarity threshold for the noise-reassignment pass. After
# HDBSCAN runs, any tag it flagged as noise gets reassigned to its
# nearest real cluster when sim ≥ this value. 0.6 is empirically
# forgiving enough to catch "almost belonged to a theme" tags without
# gluing unrelated signals together. Set to a negative value to
# disable the pass entirely and preserve the raw HDBSCAN output.
DEFAULT_REASSIGN_THRESHOLD = 0.6

# Noise label HDBSCAN assigns to tags that don't fit any cluster. We
# keep this out of `DeckProfile.clusters` and surface it separately
# as `noise_tags` so the recommender knows which signals got dropped.
NOISE_LABEL = -1


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DeckCluster:
    """One themed group of tags from a deck's clustering pass.

    Frozen so a tuple of these stays safely immutable and hashable
    per-field. The `centroid` is a numpy array (not hashable), so direct
    cluster hashing is unavailable — but no caller needs it.

    Attributes:
      label: human-readable name, picked as the slug of the tag whose
             embedding is nearest the centroid. Debug-friendly ("spot-
             removal" beats "cluster-0") and stable across runs.
      tags: tag slugs that landed in this cluster (sorted for stability).
      centroid: unit-length numpy vector in the tag embedding space;
                the mean of member tag embeddings, L2-normalised.
      deck_card_ids: scryfall_ids of the deck's cards whose `tags` array
                     intersects this cluster. Sorted for stability.
    """

    label: str
    tags: tuple[str, ...]
    centroid: np.ndarray
    deck_card_ids: tuple[str, ...]


@dataclass(frozen=True)
class DeckProfile:
    """Everything the recommender needs about a decklist's shape.

    Attributes:
      deck_card_ids: scryfall_ids that resolved from the input names,
                     in input order (duplicates removed).
      missing_names: input names that didn't resolve against the cards
                     collection; the caller may want to warn the user.
      tag_universe: all unique tag slugs across the resolved deck, sorted.
      clusters: themed groups surfaced by HDBSCAN. Order undefined but
                stable within a single build thanks to internal sorting.
      noise_tags: tags HDBSCAN flagged as noise (sorted). Not used by
                  the main ranker but surfaced so the caller can audit
                  what was dropped.
    """

    deck_card_ids: tuple[str, ...]
    missing_names: tuple[str, ...]
    tag_universe: tuple[str, ...]
    clusters: tuple[DeckCluster, ...]
    noise_tags: tuple[str, ...]


# ---------------------------------------------------------------------------
# Decklist resolution
# ---------------------------------------------------------------------------

def resolve_deck_cards(
    card_names: Iterable[str], cards_coll: Collection
) -> tuple[list[dict], list[str]]:
    """Resolve input names against the cards collection.

    Returns `(resolved_docs, missing_names)`. For each input name, if
    one or more cards match (via the lowered-names multikey index) the
    FIRST match's doc is returned — downstream clustering only needs
    the tags array, so ambiguity between art-card variants is fine.
    Missing names are collected in input order so the caller can warn
    the user ("4 cards in your decklist weren't found in cache").

    Deduplicates by scryfall id so a decklist with "1 Lightning Bolt"
    appearing twice doesn't double-count its tags.
    """
    seen_ids: set[str] = set()
    docs: list[dict] = []
    missing: list[str] = []
    for name in card_names:
        matches = list(cards_coll.find({"names": name.lower()}))
        if not matches:
            missing.append(name)
            continue
        # First match wins; identical oracle_id variants carry identical
        # tags, which is all we care about here.
        doc = matches[0]
        sid = doc.get("_id")
        if sid and sid not in seen_ids:
            seen_ids.add(sid)
            docs.append(doc)
    return docs, missing


# ---------------------------------------------------------------------------
# Tag collection + embedding load
# ---------------------------------------------------------------------------

def collect_tag_universe(deck_cards: Iterable[dict]) -> list[str]:
    """Union of tag slugs across deck cards, sorted for stable output."""
    universe: set[str] = set()
    for doc in deck_cards:
        for slug in doc.get("tags") or []:
            universe.add(slug)
    return sorted(universe)


def load_tag_embeddings(
    tag_slugs: Iterable[str], tags_coll: Collection
) -> dict[str, np.ndarray]:
    """Fetch embedded tag docs by slug and unpack their vectors.

    Tags without a stored embedding are silently skipped — a new tag
    added by Scryfall between embed passes shouldn't crash the deck
    profile. The caller sees fewer tags than it asked for and acts
    accordingly (one fewer dimension of the universe).
    """
    out: dict[str, np.ndarray] = {}
    slugs = list(tag_slugs)
    if not slugs:
        return out
    cursor = tags_coll.find(
        {"_id": {"$in": slugs}, "embedding": {"$exists": True}},
        {"_id": 1, "embedding": 1},
    )
    for doc in cursor:
        out[doc["_id"]] = emb._unpack_embedding(doc["embedding"])
    return out


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------

def _cluster_tag_embeddings(
    embeddings_by_slug: dict[str, np.ndarray],
    *,
    min_cluster_size: int,
    min_samples: int,
    cluster_selection_epsilon: float,
) -> dict[int, list[str]]:
    """Return `{cluster_label: [tag_slug, ...]}` from HDBSCAN.

    `min_samples` and `cluster_selection_epsilon` are passed through
    to HDBSCAN so callers can trade inclusivity for strictness without
    forking the implementation — the defaults in `build_deck_profile`
    err on the inclusive side, which matches what a recommender wants.

    A degenerate universe (0 or 1 tag) can't cluster meaningfully; we
    return the empty labelling and let the caller surface the lone tag
    as noise. HDBSCAN would raise on an empty matrix.
    """
    slugs = sorted(embeddings_by_slug)
    if len(slugs) < min_cluster_size:
        return {}
    matrix = np.vstack([embeddings_by_slug[s] for s in slugs])
    # `copy=True` silences a FutureWarning in sklearn 1.9 — the default
    # flips in 1.10 and we're being explicit about not mutating input.
    labels = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_epsilon=cluster_selection_epsilon,
        metric="cosine",
        copy=True,
    ).fit_predict(matrix)
    groups: dict[int, list[str]] = {}
    for slug, label in zip(slugs, labels):
        groups.setdefault(int(label), []).append(slug)
    return groups


def _reassign_noise_to_nearest_cluster(
    groups: dict[int, list[str]],
    embeddings_by_slug: dict[str, np.ndarray],
    *,
    threshold: float,
) -> dict[int, list[str]]:
    """Move each noise tag to its nearest real cluster if sim ≥ `threshold`.

    HDBSCAN flags a tag as noise when its embedding neighbourhood is
    too sparse to form a core point. Those tags carry real signal that
    the recommender would otherwise discard. For each noise tag we:

      1. Compute the unit vector of its embedding.
      2. For each real cluster, average the member tags' unit vectors
         (an unweighted cheap centroid) and dot-product against the
         noise tag's unit vector — that's cosine similarity.
      3. Reassign the tag to whichever cluster has the highest sim,
         but only if sim ≥ `threshold`. Below the floor, the tag
         stays in the noise bucket (its signal was too far from any
         existing theme to glue in without diluting it).

    The pass runs against the ORIGINAL HDBSCAN groupings — we don't
    iterate (adding a tag to a cluster slightly shifts its centroid,
    which could re-rank other noise tags' nearest-cluster choice).
    For the small matrices this operates on (~hundred-tag decks) a
    single pass is both simpler and good enough; the recommender's
    EDHREC-based cluster evaluation downstream is where the real
    quality filter happens.

    `threshold < 0` disables the pass (returns groups unchanged).
    """
    if threshold < 0 or NOISE_LABEL not in groups:
        return groups
    real_labels = [lbl for lbl in groups if lbl != NOISE_LABEL]
    if not real_labels:
        return groups

    # Precompute each real cluster's unit-norm centroid once.
    centroids: dict[int, np.ndarray] = {}
    for lbl in real_labels:
        vecs = np.vstack([embeddings_by_slug[s] for s in groups[lbl]])
        mean = vecs.mean(axis=0)
        n = float(np.linalg.norm(mean))
        centroids[lbl] = mean / n if n > 0 else mean

    noise_slugs = list(groups[NOISE_LABEL])
    kept_noise: list[str] = []
    # Build the new mapping as a copy so we don't mutate the original
    # groups dict while iterating anything derived from it.
    updated = {lbl: list(members) for lbl, members in groups.items()}
    for slug in noise_slugs:
        vec = embeddings_by_slug[slug]
        v_norm = float(np.linalg.norm(vec))
        if v_norm == 0:
            kept_noise.append(slug)
            continue
        unit = vec / v_norm
        best_label, best_sim = None, threshold
        for lbl, centroid in centroids.items():
            sim = float(np.dot(unit, centroid))
            # Strict > so ties don't drag a tag into a cluster at the
            # threshold boundary. Threshold is initialised to the floor
            # so the first clear hit wins.
            if sim > best_sim:
                best_label, best_sim = lbl, sim
        if best_label is None:
            kept_noise.append(slug)
        else:
            updated[best_label].append(slug)
    updated[NOISE_LABEL] = kept_noise
    return updated


def _centroid_and_label(
    tag_slugs: list[str], embeddings_by_slug: dict[str, np.ndarray]
) -> tuple[np.ndarray, str]:
    """Return (unit centroid, nearest-tag-to-centroid slug) for a cluster.

    Picking a tag label from the centroid gives interpretable names
    ("spot-removal" rather than "cluster-3") at near-zero cost: one
    cosine comparison per cluster member. Falls back to the first slug
    if cosine comparisons somehow all tie (shouldn't happen in practice
    with float32 vectors).
    """
    vecs = np.vstack([embeddings_by_slug[s] for s in tag_slugs])
    mean = vecs.mean(axis=0)
    # L2-normalise so the centroid is in the same unit space as
    # downstream card_vectors (which the fuse step also normalises).
    norm = float(np.linalg.norm(mean))
    centroid = mean / norm if norm > 0 else mean
    # Nearest tag by cosine similarity. For unit-norm centroid and
    # (near-)unit tag embeddings, this is max of dot products.
    best_sim = -2.0
    best_slug = tag_slugs[0]
    for slug in tag_slugs:
        vec = embeddings_by_slug[slug]
        v_norm = float(np.linalg.norm(vec))
        if v_norm == 0:
            continue
        sim = float(np.dot(centroid, vec / v_norm))
        if sim > best_sim:
            best_sim = sim
            best_slug = slug
    return centroid, best_slug


# ---------------------------------------------------------------------------
# Top-level
# ---------------------------------------------------------------------------

def build_deck_profile(
    card_names: Iterable[str],
    *,
    cards_coll: Optional[Collection] = None,
    tags_coll: Optional[Collection] = None,
    min_cluster_size: int = DEFAULT_MIN_CLUSTER_SIZE,
    min_samples: int = DEFAULT_MIN_SAMPLES,
    cluster_selection_epsilon: float = DEFAULT_CLUSTER_SELECTION_EPSILON,
    reassign_threshold: float = DEFAULT_REASSIGN_THRESHOLD,
) -> DeckProfile:
    """Full pipeline: decklist → resolved cards → tags → clusters.

    All Mongo collections default to the project handles; tests pass
    mongomock collections directly.

    Clustering knobs (all defaults err on the inclusive side so the
    recommender has more signal to work with — the EDHREC-based
    cluster evaluation downstream filters weak themes):

      min_cluster_size:
        Minimum tag count for a cluster. Default 2 (any pair counts).
      min_samples:
        HDBSCAN density parameter. Default 1 — every tag can be a
        core point, so tags only land in noise when they're genuinely
        isolated in embedding space.
      cluster_selection_epsilon:
        Merges clusters whose boundary distance (cosine) falls below
        this value. 0.0 preserves HDBSCAN's natural splits; raise to
        collapse near-themes.
      reassign_threshold:
        After HDBSCAN runs, any tag it flagged as noise gets moved
        to its nearest real cluster when cosine-sim ≥ this value.
        Default 0.6. Set negative to disable the pass.
    """
    if cards_coll is None:
        cards_coll = storage.cards_collection()
    if tags_coll is None:
        tags_coll = storage.tags_collection()

    deck_cards, missing = resolve_deck_cards(card_names, cards_coll)
    tag_universe = collect_tag_universe(deck_cards)
    embeddings_by_slug = load_tag_embeddings(tag_universe, tags_coll)

    # Tags present in the deck but not in `embeddings_by_slug` have no
    # embedding — treat them as degenerate noise so the profile stays
    # honest about what the clusterer actually saw. These never get
    # reassigned (no vector to compare against); they're a different
    # category of orphan than HDBSCAN-noise.
    embedded_slugs = set(embeddings_by_slug)
    unembedded = [s for s in tag_universe if s not in embedded_slugs]

    groups = _cluster_tag_embeddings(
        embeddings_by_slug,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_epsilon=cluster_selection_epsilon,
    )
    # Reassign HDBSCAN-noise tags to their nearest real cluster when
    # the fit is close enough. Threshold < 0 skips the pass.
    groups = _reassign_noise_to_nearest_cluster(
        groups, embeddings_by_slug, threshold=reassign_threshold
    )

    clusters: list[DeckCluster] = []
    noise_tags: list[str] = list(unembedded)
    for label_idx, slugs in groups.items():
        if label_idx == NOISE_LABEL:
            noise_tags.extend(slugs)
            continue
        slugs_sorted = sorted(slugs)
        centroid, human_label = _centroid_and_label(slugs_sorted, embeddings_by_slug)
        tag_set = set(slugs_sorted)
        deck_card_ids = sorted(
            doc["_id"] for doc in deck_cards
            if tag_set.intersection(doc.get("tags") or [])
        )
        clusters.append(
            DeckCluster(
                label=human_label,
                tags=tuple(slugs_sorted),
                centroid=centroid,
                deck_card_ids=tuple(deck_card_ids),
            )
        )

    return DeckProfile(
        deck_card_ids=tuple(doc["_id"] for doc in deck_cards),
        missing_names=tuple(missing),
        tag_universe=tuple(tag_universe),
        clusters=tuple(clusters),
        noise_tags=tuple(sorted(noise_tags)),
    )


# ---------------------------------------------------------------------------
# CLI (mtg-deck-profile)
# ---------------------------------------------------------------------------
#
# Human-readable render of a deck profile. Not the recommender yet —
# that's Phase 3 step 4 (a separate CLI). This one is here so you can
# eyeball the clustering output on a real deck without opening Python.

DEFAULT_RENDER_LIMIT = 10


def _render_profile(profile: DeckProfile, *, name_lookup: dict[str, str], limit: int) -> str:
    """Format a `DeckProfile` as a block of printable text.

    `name_lookup` maps scryfall id → display name so cluster output
    shows "Lightning Bolt" rather than a 36-char UUID. `limit` caps
    the per-cluster tag + card lists so a 100-card commander deck's
    output stays scannable; a trailing "(+N more)" marker makes the
    truncation obvious.
    """
    lines: list[str] = []
    lines.append(
        f"resolved {len(profile.deck_card_ids)} unique cards "
        f"({len(profile.missing_names)} missing)"
    )
    if profile.missing_names:
        for name in profile.missing_names:
            lines.append(f"  missing: {name}")
    lines.append(f"tag universe: {len(profile.tag_universe)} unique tags")
    lines.append(
        f"{len(profile.clusters)} theme(s) clustered, "
        f"{len(profile.noise_tags)} tag(s) in noise"
    )
    lines.append("")

    if not profile.clusters:
        lines.append("(no themes — deck too thin for clustering, or no "
                     "tag embeddings available)")
    for cluster in profile.clusters:
        lines.append(
            f"theme '{cluster.label}'  "
            f"({len(cluster.tags)} tags, {len(cluster.deck_card_ids)} cards)"
        )
        lines.append(f"  tags : {_truncate(list(cluster.tags), limit)}")
        card_names = [name_lookup.get(sid, sid) for sid in cluster.deck_card_ids]
        card_names.sort()
        lines.append(f"  cards: {_truncate(card_names, limit)}")
        lines.append("")

    if profile.noise_tags:
        lines.append(
            f"noise: {_truncate(list(profile.noise_tags), limit)}"
        )
    return "\n".join(lines)


def _truncate(items: list[str], limit: int) -> str:
    """Join `items` with ', ', trimming to `limit` with a "(+N more)" tail."""
    if len(items) <= limit:
        return ", ".join(items)
    shown = ", ".join(items[:limit])
    return f"{shown}, (+{len(items) - limit} more)"


def _name_lookup_for(deck_card_ids: Iterable[str], cards_coll: Collection) -> dict[str, str]:
    """One Mongo round trip to fetch display names for the rendered cards."""
    ids = list(deck_card_ids)
    if not ids:
        return {}
    return {
        doc["_id"]: doc.get("name") or doc["_id"]
        for doc in cards_coll.find({"_id": {"$in": ids}}, {"_id": 1, "name": 1})
    }


def main(argv: Optional[Iterable[str]] = None) -> int:
    """`mtg-deck-profile` — build and print a decklist's theme clusters."""
    parser = argparse.ArgumentParser(
        prog="mtg-deck-profile",
        description=(
            "Cluster a decklist's tags into themes and print the result. "
            "Reads cards from the local Mongo cache populated by `mtg-embed`."
        ),
    )
    parser.add_argument(
        "cards", nargs="*",
        help="Card names (quote multi-word names). Combine with --file.",
    )
    parser.add_argument(
        "-f", "--file",
        help="Path to a decklist file (one card per line; '#' for comments).",
    )
    parser.add_argument(
        "--min-cluster-size", type=int, default=DEFAULT_MIN_CLUSTER_SIZE,
        help=(
            f"HDBSCAN min_cluster_size (default: {DEFAULT_MIN_CLUSTER_SIZE}). "
            "Lower for small decks; raise to keep only strong themes."
        ),
    )
    parser.add_argument(
        "--min-samples", type=int, default=DEFAULT_MIN_SAMPLES,
        help=(
            f"HDBSCAN min_samples (default: {DEFAULT_MIN_SAMPLES}). "
            "Lower = more inclusive (fewer orphans flagged as noise). "
            "1 means every tag can be a core point."
        ),
    )
    parser.add_argument(
        "--cluster-selection-epsilon", type=float,
        default=DEFAULT_CLUSTER_SELECTION_EPSILON,
        help=(
            "Cosine-distance threshold for merging near-themes HDBSCAN "
            "would otherwise split (default: "
            f"{DEFAULT_CLUSTER_SELECTION_EPSILON}). Try 0.1–0.3 for a "
            "more lumped output."
        ),
    )
    parser.add_argument(
        "--reassign-threshold", type=float, default=DEFAULT_REASSIGN_THRESHOLD,
        help=(
            "Cosine-similarity threshold for the noise-reassignment "
            "pass (default: {}). A noise tag whose embedding has "
            "sim ≥ this to some cluster's centroid gets absorbed into "
            "that cluster. Set negative to disable the pass."
        ).format(DEFAULT_REASSIGN_THRESHOLD),
    )
    parser.add_argument(
        "--limit", type=int, default=DEFAULT_RENDER_LIMIT,
        help=(
            f"Per-cluster display cap for tags + cards (default: {DEFAULT_RENDER_LIMIT}). "
            "Everything beyond collapses to a '(+N more)' marker."
        ),
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    names = sf.read_names(args)
    if not names:
        parser.error("no card names provided (pass names as args or via --file)")

    cards_coll = storage.cards_collection()
    tags_coll = storage.tags_collection()
    profile = build_deck_profile(
        names,
        cards_coll=cards_coll,
        tags_coll=tags_coll,
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        cluster_selection_epsilon=args.cluster_selection_epsilon,
        reassign_threshold=args.reassign_threshold,
    )
    name_lookup = _name_lookup_for(profile.deck_card_ids, cards_coll)
    print(_render_profile(profile, name_lookup=name_lookup, limit=args.limit))
    return 0


if __name__ == "__main__":
    sys.exit(main())
