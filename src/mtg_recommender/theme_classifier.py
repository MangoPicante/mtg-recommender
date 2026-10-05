"""Data-driven theme discovery + per-card classification (Phase 3 step 2 v2).

Replaces the embedding-only HDBSCAN clustering in `deck_profile` with a
hierarchy-aware grouping. The core insight from auditing the Scryfall
oracle_tags bulk: ~900 top-level tags exist, but only ~30–50 of them
read as real play themes — the rest are either structural
("triggered-ability"), catalog ("card-names"), or too-generic umbrellas
("cycle" covers 8k cards). This module programmatically picks the
"real theme" subset via:

    1. a hand-shaped blocklist (hard-coded set of slugs we know are
       structural / flavor / catalog),
    2. a minimum-children floor (a top-level with 0 or 1 children is
       a point concept, not an umbrella worth grouping on),
    3. a card-coverage window (`min`, `max`) — too few covered cards
       is a niche concept; too many is a mechanical signal that
       applies to almost everything.

Each surviving top-level becomes a `Theme` with:
    - name: the top-level tag's slug (human readable)
    - tag_slugs: self + every descendant slug (so a card tagged with
                 the descendant `spot-removal` counts as part of the
                 `removal` theme)
    - representative: unit-length mean of the subtree's tag embeddings
                      (what `classify_card` compares a card's own text
                      embedding against for multi-theme tiebreak)
    - card_coverage: how many distinct cards in the collection fall
                     under this theme (bookkeeping, not used in
                     classification)

Per-card classification:

    1. Collect candidate themes = any theme whose tag_slugs intersect
       the card's `tags` array.
    2. Zero candidates → unassigned (returned as None).
    3. One candidate → that theme.
    4. Two or more → cosine similarity between the card's stored
       `text_embedding` and each candidate theme's representative
       vector; the max wins. This is the "based on oracle text"
       tiebreak the user asked for — stored text_embedding IS the
       card's oracle_text in vector form.

Representative = text embedding (not card_vector), because card_vector
already blends tag embeddings — tiebreaking a tag-ambiguous card using
a vector that includes tag signal would be circular.

Discovery runs per invocation (one aggregation + a handful of
count_documents calls). Cheap enough for a CLI call; if a long-running
service wants to amortise, cache the result at the caller level.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
from pymongo.collection import Collection

from . import embeddings as emb

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

# Top-level slugs we never want to treat as themes. Hand-shaped from
# auditing the top-30 by card coverage in the live oracle_tags bulk:
#
#   triggered-ability / activated-ability : structural — almost every
#       creature has one; grouping on these would absorb half the deck.
#   cycle            : flavor-only ("card cycle" like the Praetors) —
#                      not the Cycling keyword. Covers 8k cards across
#                      1300+ children — a catalog, not a theme.
#   card-names       : purely cataloging shared-name cards.
#   flavors-of-vanilla,
#   type-errata,
#   unique-type-line : flavor / catalog, no gameplay signal.
#   staple-with-set-s-mechanic : set-staples catalog.
#   repeatable-crime : very niche umbrella.
DEFAULT_BLOCKLIST = frozenset({
    "triggered-ability",
    "activated-ability",
    "cycle",
    "card-names",
    "flavors-of-vanilla",
    "type-errata",
    "unique-type-line",
    "staple-with-set-s-mechanic",
    "repeatable-crime",
})

# Default coverage window. Top-levels covering fewer cards are niche
# single-concept umbrellas; top-levels covering more are mechanical
# signals (every card has one). Both numbers tuned by hand against the
# live top-30 — adjust via `ThemeFilter` kwargs if your sense of "theme"
# differs.
DEFAULT_MIN_COVERAGE = 100
DEFAULT_MAX_COVERAGE = 5000

# A top-level with 0 or 1 children is a point concept — the hierarchy
# never collapsed any sibling under it — and acting on it as a group
# gives us a one-element bucket. 2 is the lowest useful threshold.
DEFAULT_MIN_CHILDREN = 2


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ThemeFilter:
    """Knobs that decide which top-level tags become themes.

    All defaults are the ones that survive the audit of the live
    oracle_tags bulk and feel sensible for Commander play. Override
    per call to loosen or tighten.
    """

    min_children: int = DEFAULT_MIN_CHILDREN
    min_coverage: int = DEFAULT_MIN_COVERAGE
    max_coverage: int = DEFAULT_MAX_COVERAGE
    blocklist: frozenset[str] = DEFAULT_BLOCKLIST


DEFAULT_THEME_FILTER = ThemeFilter()


@dataclass(frozen=True)
class Theme:
    """A discovered theme: a top-level tag plus everything under it.

    Attributes:
      name:
        The top-level tag's slug ("removal", "ramp", …).
      tag_slugs:
        Frozenset of every slug in this theme's subtree, including the
        top-level itself. A card belongs to the theme iff its tag
        array intersects this set.
      representative:
        Unit-length numpy vector; mean of embeddings for every tag
        in `tag_slugs` that has an embedding (we skip missing ones
        rather than failing — a mid-import state shouldn't block
        classification).
      card_coverage:
        Count of distinct cards tagged with anything in `tag_slugs`.
        Reported for transparency; not used in classification itself.
    """

    name: str
    tag_slugs: frozenset[str]
    representative: np.ndarray
    card_coverage: int


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def _collect_descendants(
    top_slug: str, by_slug: dict[str, dict]
) -> frozenset[str]:
    """Return `top_slug` ∪ every descendant slug reachable through child_slugs.

    Guards against hierarchy cycles (shouldn't exist in Scryfall data
    but we don't want an infinite loop if upstream drifts) by tracking
    visited slugs. Unknown child slugs (schema drift between tag docs
    and the catalog state we're reading) are silently dropped.
    """
    seen: set[str] = set()
    stack = [top_slug]
    while stack:
        slug = stack.pop()
        if slug in seen or slug not in by_slug:
            continue
        seen.add(slug)
        stack.extend(by_slug[slug].get("child_slugs") or [])
    return frozenset(seen)


def _representative_vector(
    tag_slugs: Iterable[str], by_slug: dict[str, dict]
) -> Optional[np.ndarray]:
    """Mean of embeddings for `tag_slugs`, L2-normalised. None if none exist."""
    vecs: list[np.ndarray] = []
    for slug in tag_slugs:
        doc = by_slug.get(slug)
        if doc is None:
            continue
        packed = doc.get("embedding")
        if packed is None:
            continue
        vecs.append(emb._unpack_embedding(packed))
    if not vecs:
        return None
    mean = np.vstack(vecs).mean(axis=0)
    norm = float(np.linalg.norm(mean))
    return mean / norm if norm > 0 else mean


def discover_themes(
    cards_coll: Collection,
    tags_coll: Collection,
    *,
    theme_filter: ThemeFilter = DEFAULT_THEME_FILTER,
) -> list[Theme]:
    """Walk the tag hierarchy, apply the filter, return surviving themes.

    One Mongo pass loads every tag doc (we need the whole hierarchy to
    compute subtrees). Then one `count_documents` per surviving
    candidate measures card coverage — cheap on a ~30-candidate set.

    Returns themes sorted by `card_coverage` descending, so the first
    few are the biggest / most mainstream themes. Order is deterministic
    for a given input.
    """
    all_tags = list(tags_coll.find({}, {
        "_id": 1, "parent_slugs": 1, "child_slugs": 1, "embedding": 1,
    }))
    by_slug = {doc["_id"]: doc for doc in all_tags}

    top_level = [
        doc for doc in all_tags
        if not (doc.get("parent_slugs") or [])
    ]
    themes: list[Theme] = []
    for doc in top_level:
        slug = doc["_id"]
        if slug in theme_filter.blocklist:
            continue
        children = doc.get("child_slugs") or []
        if len(children) < theme_filter.min_children:
            continue
        subtree = _collect_descendants(slug, by_slug)
        # Card coverage: one count per candidate. On ~30 survivors
        # that's a handful of fast indexed queries (tags has a hash
        # index via mongomock; real Mongo uses the array multikey one
        # on cards.tags that we never created... worth adding if this
        # becomes slow, but 30 counts on 40k cards is sub-second).
        coverage = cards_coll.count_documents({"tags": {"$in": list(subtree)}})
        if coverage < theme_filter.min_coverage:
            continue
        if coverage > theme_filter.max_coverage:
            continue
        representative = _representative_vector(subtree, by_slug)
        if representative is None:
            # No embeddings in the subtree — can't classify multi-theme
            # cards against this one. Skip rather than carry a bogus
            # zero vector that would always lose tiebreaks.
            continue
        themes.append(Theme(
            name=slug,
            tag_slugs=subtree,
            representative=representative,
            card_coverage=coverage,
        ))
    themes.sort(key=lambda t: (-t.card_coverage, t.name))
    return themes


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def classify_card(
    card_tags: Iterable[str],
    card_text_embedding: Optional[np.ndarray],
    themes: Iterable[Theme],
) -> Optional[Theme]:
    """Pick the best theme for one card, or None if nothing matches.

    `card_tags` is the card doc's `tags` array. `card_text_embedding`
    is the stored `text_embedding` already unpacked to a float64 numpy
    vector (callers do the unpack once per card to avoid repeating it
    inside this inner loop).

    Resolution:
      - 0 candidate themes → None (unassigned).
      - 1 candidate → that theme (no tiebreak needed).
      - 2+ candidates → cosine similarity between the card's text
        embedding and each candidate's representative vector; max wins.
        If the card has no stored text_embedding, we fall back to the
        candidate with the highest `card_coverage` — a graceful degrade
        rather than crashing on a half-populated cache.
    """
    card_tag_set = set(card_tags or [])
    if not card_tag_set:
        return None
    candidates = [t for t in themes if card_tag_set & t.tag_slugs]
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    if card_text_embedding is None:
        return max(candidates, key=lambda t: t.card_coverage)
    # L2-normalise once so each dot product is a true cosine similarity.
    norm = float(np.linalg.norm(card_text_embedding))
    unit = card_text_embedding / norm if norm > 0 else card_text_embedding
    return max(candidates, key=lambda t: float(np.dot(unit, t.representative)))


def classify_deck(
    deck_cards: list[dict],
    themes: list[Theme],
) -> tuple[dict[str, list[str]], list[str]]:
    """Classify every card in `deck_cards` into a theme (or unassigned).

    Returns `(per_theme_cards, unassigned_card_ids)`:

      per_theme_cards: {theme_name: [card_id, ...]} with themes that
                       actually caught cards (empty themes dropped).
      unassigned_card_ids: cards that matched zero themes.

    Both lists are sorted for stable output. Themes with no matching
    cards don't appear in `per_theme_cards` — downstream code shouldn't
    pretend empty buckets exist.
    """
    per_theme: dict[str, list[str]] = {}
    unassigned: list[str] = []
    for doc in deck_cards:
        text_vec = None
        packed = doc.get("text_embedding")
        if packed is not None:
            text_vec = emb._unpack_embedding(packed)
        theme = classify_card(doc.get("tags") or [], text_vec, themes)
        sid = doc.get("_id")
        if theme is None:
            if sid:
                unassigned.append(sid)
        else:
            per_theme.setdefault(theme.name, []).append(sid)
    # Sorting here (vs. at insert time) keeps classify_card's inner
    # loop O(1) per card instead of O(log n).
    for name in per_theme:
        per_theme[name].sort()
    unassigned.sort()
    return per_theme, unassigned
