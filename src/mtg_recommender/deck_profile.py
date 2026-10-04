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

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
from pymongo.collection import Collection
from sklearn.cluster import HDBSCAN

from . import embeddings as emb
from . import storage

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

# HDBSCAN's min_cluster_size: the smallest group of tags that counts as
# a theme. 2 is permissive (a pair of co-occurring tags IS a theme in
# a deck context), 3 is stricter. Overridable per call.
DEFAULT_MIN_CLUSTER_SIZE = 2

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
) -> dict[int, list[str]]:
    """Return `{cluster_label: [tag_slug, ...]}` from HDBSCAN.

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
        metric="cosine",
        copy=True,
    ).fit_predict(matrix)
    groups: dict[int, list[str]] = {}
    for slug, label in zip(slugs, labels):
        groups.setdefault(int(label), []).append(slug)
    return groups


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
) -> DeckProfile:
    """Full pipeline: decklist → resolved cards → tags → clusters.

    All Mongo collections default to the project handles; tests pass
    mongomock collections directly. `min_cluster_size` controls HDBSCAN
    strictness — lower it (to 2) for small decks; raise it (to 4+) for
    decks where you want only strong themes.
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
    # honest about what the clusterer actually saw.
    embedded_slugs = set(embeddings_by_slug)
    unembedded = [s for s in tag_universe if s not in embedded_slugs]

    groups = _cluster_tag_embeddings(
        embeddings_by_slug, min_cluster_size=min_cluster_size
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
