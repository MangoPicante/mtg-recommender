"""Per-deck theme classification (Phase 3 step 2).

Given a decklist, this module:
  1. Resolves each card name against the `cards` collection (via the
     multikey `names` index — ambiguous names get the first match,
     fully-missing names are reported but don't fail the build).
  2. Walks the oracle_tags hierarchy to find a small, hand-shaped set
     of "real theme" top-level tags (see `theme_classifier` for the
     filter: blocklist + min_children + card-coverage window).
  3. Classifies each deck card into one theme by intersecting its tag
     array with each theme's subtree. If a card lands in multiple
     themes, the tiebreak compares the card's stored `text_embedding`
     against each theme's representative vector (mean of the theme's
     tag embeddings).

What a theme represents: a top-level play-pattern umbrella from
EDHREC's Tagger project — "removal", "ramp", "card-advantage",
"sacrifice-outlet" — pulled out of the ~900 top-level tags by filtering
away structural ("triggered-ability"), catalog ("card-names"), and
over-generic ("cycle") entries.

Why themes beat the previous embedding-only clustering:
  - Stable across decks (same theme vocabulary for every deck → the
    recommender's attribution strings don't drift run-to-run).
  - Interpretable without a reader having to look up tag slugs.
  - Handles multi-match cleanly via the oracle-text tiebreak, which
    is exactly what cosine similarity against a theme's rep vector
    computes.

The embedding-based clustering (HDBSCAN + noise reassignment) is gone;
the hierarchy carries the structure we were previously inferring.
Phase 3 step 4 (candidate ranking) will use each theme's representative
vector as the query vector against `card_vector`, with EDHREC lift
layered on as a per-commander quality signal.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
from pymongo.collection import Collection

from . import scryfall_fetch as sf
from . import storage
from . import theme_classifier as tc

# ---------------------------------------------------------------------------
# Dataclasses (output shape)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DeckCluster:
    """One theme's slice of the deck.

    Attributes:
      label: the theme name (top-level tag slug — "removal", "ramp").
      tags: the deck's tag slugs that fall under this theme's subtree.
            Not the full subtree — only slugs actually present on
            deck cards, so the attribution output reflects what the
            CURRENT deck brought in rather than listing every known
            member tag.
      centroid: the theme's representative unit vector in the tag
                embedding space. Phase 3 step 4 queries `card_vector`
                against this to rank candidate cards.
      deck_card_ids: scryfall_ids of deck cards classified into this
                     theme (sorted for stability).
    """

    label: str
    tags: tuple[str, ...]
    centroid: np.ndarray
    deck_card_ids: tuple[str, ...]


@dataclass(frozen=True)
class DeckProfile:
    """Everything the recommender needs about a decklist's shape.

    Attributes:
      deck_card_ids: scryfall_ids that resolved from input names
                     (duplicates removed, input order preserved).
      missing_names: input names that didn't resolve against the cards
                     collection.
      tag_universe: all unique tag slugs across the resolved deck,
                    sorted.
      clusters: one `DeckCluster` per theme that caught ≥1 card from
                this deck.
      noise_tags: tag slugs in the deck's universe that don't belong
                  to ANY surviving theme's subtree — tags we had but
                  the filter rejected (or that no theme covered).
      unassigned_card_ids: deck cards that matched zero themes. If
                           this list is big relative to deck size,
                           consider loosening the theme filter.
    """

    deck_card_ids: tuple[str, ...]
    missing_names: tuple[str, ...]
    tag_universe: tuple[str, ...]
    clusters: tuple[DeckCluster, ...]
    noise_tags: tuple[str, ...]
    unassigned_card_ids: tuple[str, ...]


# ---------------------------------------------------------------------------
# Decklist resolution
# ---------------------------------------------------------------------------

def resolve_deck_cards(
    card_names: Iterable[str], cards_coll: Collection
) -> tuple[list[dict], list[str]]:
    """Resolve input names against the cards collection.

    Returns `(resolved_docs, missing_names)`. For each input name, if
    one or more cards match (via the lowered-names multikey index) the
    FIRST match's doc is returned — downstream classification only
    needs the tags array + text_embedding, so ambiguity between
    art-card variants is fine. Missing names are collected in input
    order so the caller can warn the user.

    Deduplicates by scryfall id so a decklist with "1 Lightning Bolt"
    appearing twice doesn't double-count its tags. Projects just the
    fields classify_deck will read, keeping each doc small.
    """
    seen_ids: set[str] = set()
    docs: list[dict] = []
    missing: list[str] = []
    projection = {"_id": 1, "name": 1, "tags": 1, "text_embedding": 1}
    for name in card_names:
        matches = list(cards_coll.find({"names": name.lower()}, projection))
        if not matches:
            missing.append(name)
            continue
        doc = matches[0]
        sid = doc.get("_id")
        if sid and sid not in seen_ids:
            seen_ids.add(sid)
            docs.append(doc)
    return docs, missing


def collect_tag_universe(deck_cards: Iterable[dict]) -> list[str]:
    """Union of tag slugs across deck cards, sorted for stable output."""
    universe: set[str] = set()
    for doc in deck_cards:
        for slug in doc.get("tags") or []:
            universe.add(slug)
    return sorted(universe)


# ---------------------------------------------------------------------------
# Profile builder
# ---------------------------------------------------------------------------

def build_deck_profile(
    card_names: Iterable[str],
    *,
    cards_coll: Optional[Collection] = None,
    tags_coll: Optional[Collection] = None,
    theme_filter: tc.ThemeFilter = tc.DEFAULT_THEME_FILTER,
) -> DeckProfile:
    """Full pipeline: decklist → resolved cards → themes → classification.

    All Mongo collections default to the project handles; tests pass
    mongomock collections directly. `theme_filter` tunes which top-level
    tags count as "themes" worth classifying against; see
    `theme_classifier.ThemeFilter` for the knobs.
    """
    if cards_coll is None:
        cards_coll = storage.cards_collection()
    if tags_coll is None:
        tags_coll = storage.tags_collection()

    deck_cards, missing = resolve_deck_cards(card_names, cards_coll)
    tag_universe = collect_tag_universe(deck_cards)
    themes = tc.discover_themes(
        cards_coll, tags_coll, theme_filter=theme_filter
    )
    per_theme_cards, unassigned = tc.classify_deck(deck_cards, themes)

    # Build DeckClusters only for themes that caught at least one deck
    # card. Keep theme order from discover_themes (coverage-descending)
    # so output is stable + themes with broader fit lead.
    name_to_theme = {t.name: t for t in themes}
    clusters: list[DeckCluster] = []
    cards_by_id = {doc["_id"]: doc for doc in deck_cards}
    for theme in themes:
        card_ids = per_theme_cards.get(theme.name)
        if not card_ids:
            continue
        # The "tags" field for a cluster is the deck's own tag slugs
        # that fall under this theme — the output reflects what the
        # current deck actually brought, not the full subtree.
        deck_tags_in_theme = set()
        for sid in card_ids:
            for slug in cards_by_id[sid].get("tags") or []:
                if slug in theme.tag_slugs:
                    deck_tags_in_theme.add(slug)
        clusters.append(DeckCluster(
            label=theme.name,
            tags=tuple(sorted(deck_tags_in_theme)),
            centroid=theme.representative,
            deck_card_ids=tuple(card_ids),  # already sorted by classify_deck
        ))

    # Noise: tags in the deck universe that no surviving theme covers.
    covered = set()
    for t in themes:
        covered |= t.tag_slugs
    noise_tags = tuple(s for s in tag_universe if s not in covered)

    # name_to_theme is unused in the current build but might help a
    # future caller correlate names to theme objects without a second
    # discover pass — keep a reference so lint doesn't flag the local.
    _ = name_to_theme

    return DeckProfile(
        deck_card_ids=tuple(doc["_id"] for doc in deck_cards),
        missing_names=tuple(missing),
        tag_universe=tuple(tag_universe),
        clusters=tuple(clusters),
        noise_tags=noise_tags,
        unassigned_card_ids=tuple(unassigned),
    )


# ---------------------------------------------------------------------------
# CLI (mtg-deck-profile)
# ---------------------------------------------------------------------------
#
# Human-readable render of a deck profile. Not the recommender yet —
# that's Phase 3 step 4 (a separate CLI). This one is here so you can
# eyeball the theme classification output on a real deck.

DEFAULT_RENDER_LIMIT = 10


def _render_profile(
    profile: DeckProfile,
    *,
    name_lookup: dict[str, str],
    limit: int,
) -> str:
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
        f"{len(profile.clusters)} theme(s) matched, "
        f"{len(profile.unassigned_card_ids)} card(s) unassigned, "
        f"{len(profile.noise_tags)} tag(s) outside any theme"
    )
    lines.append("")

    if not profile.clusters:
        lines.append("(no themes matched — try loosening the filter with "
                     "--max-coverage / --min-coverage / --min-children)")
    for cluster in profile.clusters:
        lines.append(
            f"theme '{cluster.label}'  "
            f"({len(cluster.tags)} tags from deck, "
            f"{len(cluster.deck_card_ids)} cards)"
        )
        lines.append(f"  tags : {_truncate(list(cluster.tags), limit)}")
        card_names = [name_lookup.get(sid, sid) for sid in cluster.deck_card_ids]
        card_names.sort()
        lines.append(f"  cards: {_truncate(card_names, limit)}")
        lines.append("")

    if profile.unassigned_card_ids:
        unassigned_names = [
            name_lookup.get(sid, sid) for sid in profile.unassigned_card_ids
        ]
        unassigned_names.sort()
        lines.append(f"unassigned cards: {_truncate(unassigned_names, limit)}")
    if profile.noise_tags:
        lines.append(f"noise tags: {_truncate(list(profile.noise_tags), limit)}")
    return "\n".join(lines)


def _truncate(items: list[str], limit: int) -> str:
    """Join `items` with ', ', trimming to `limit` with a "(+N more)" tail."""
    if len(items) <= limit:
        return ", ".join(items)
    shown = ", ".join(items[:limit])
    return f"{shown}, (+{len(items) - limit} more)"


def _name_lookup_for(
    deck_card_ids: Iterable[str], cards_coll: Collection
) -> dict[str, str]:
    """One Mongo round trip to fetch display names for the rendered cards."""
    ids = list(deck_card_ids)
    if not ids:
        return {}
    return {
        doc["_id"]: doc.get("name") or doc["_id"]
        for doc in cards_coll.find({"_id": {"$in": ids}}, {"_id": 1, "name": 1})
    }


def _parse_blocklist(raw: Optional[str]) -> Optional[frozenset[str]]:
    """Parse the --theme-blocklist CLI arg into an extra-blocklist set.

    None (flag omitted) returns None, meaning "use the module default".
    Empty string clears the blocklist entirely so the user can see the
    structural tags if they really want to.
    """
    if raw is None:
        return None
    if not raw.strip():
        return frozenset()
    return frozenset(s.strip() for s in raw.split(",") if s.strip())


def main(argv: Optional[Iterable[str]] = None) -> int:
    """`mtg-deck-profile` — classify a decklist into themes and print."""
    parser = argparse.ArgumentParser(
        prog="mtg-deck-profile",
        description=(
            "Classify a decklist's cards into themes using the "
            "Scryfall oracle_tags hierarchy, with an oracle-text "
            "tiebreak for cards that match multiple themes."
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
        "--min-children", type=int, default=tc.DEFAULT_MIN_CHILDREN,
        help=(
            f"Minimum number of child tags a top-level must have to count "
            f"as a theme (default: {tc.DEFAULT_MIN_CHILDREN}). Raising this "
            "drops smaller point concepts."
        ),
    )
    parser.add_argument(
        "--min-coverage", type=int, default=tc.DEFAULT_MIN_COVERAGE,
        help=(
            f"Minimum distinct cards tagged by a theme (default: "
            f"{tc.DEFAULT_MIN_COVERAGE}). Below this the theme is too "
            "niche to be useful."
        ),
    )
    parser.add_argument(
        "--max-coverage", type=int, default=tc.DEFAULT_MAX_COVERAGE,
        help=(
            f"Maximum distinct cards tagged by a theme (default: "
            f"{tc.DEFAULT_MAX_COVERAGE}). Above this the theme is a "
            "mechanical signal that applies to almost everything."
        ),
    )
    parser.add_argument(
        "--theme-blocklist", default=None,
        help=(
            "Comma-separated top-level slugs to exclude. Overrides the "
            "built-in blocklist (defaults to a hand-shaped set of "
            "structural / catalog tags). Pass an empty string to disable "
            "blocklisting entirely."
        ),
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

    blocklist = _parse_blocklist(args.theme_blocklist)
    theme_filter = tc.ThemeFilter(
        min_children=args.min_children,
        min_coverage=args.min_coverage,
        max_coverage=args.max_coverage,
        blocklist=blocklist if blocklist is not None else tc.DEFAULT_BLOCKLIST,
    )

    cards_coll = storage.cards_collection()
    tags_coll = storage.tags_collection()
    profile = build_deck_profile(
        names,
        cards_coll=cards_coll,
        tags_coll=tags_coll,
        theme_filter=theme_filter,
    )
    name_lookup = _name_lookup_for(profile.deck_card_ids, cards_coll)
    print(_render_profile(profile, name_lookup=name_lookup, limit=args.limit))
    return 0


if __name__ == "__main__":
    sys.exit(main())
