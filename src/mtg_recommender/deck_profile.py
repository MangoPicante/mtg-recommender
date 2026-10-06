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

from . import card_clusterer as cc
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
             For a merged cluster, the keeper's name (highest card
             coverage among the constituents).
      tags: the deck's tag slugs that fall under this theme's subtree.
            Not the full subtree — only slugs actually present on
            deck cards, so the attribution output reflects what the
            CURRENT deck brought in rather than listing every known
            member tag. For a merged cluster, the union of
            constituent themes' deck-visible slugs.
      centroid: the cluster's representative unit vector in the tag
                embedding space. Phase 3 step 4 queries `card_vector`
                against this to rank candidate cards. For a merged
                cluster, the L2-normalized mean of its constituents'
                representatives.
      deck_card_ids: scryfall_ids of deck cards classified into this
                     cluster (sorted for stability).
      constituent_themes: the theme names that got merged into this
                          cluster. Length-1 tuple containing just
                          `label` for an unmerged theme; longer when
                          the post-classification merge pass
                          collapsed similar themes together. Useful
                          in attribution to show which themes
                          collapsed.
    """

    label: str
    tags: tuple[str, ...]
    centroid: np.ndarray
    deck_card_ids: tuple[str, ...]
    constituent_themes: tuple[str, ...] = ()


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
    # Project card_vector too: the optional merge-similar-themes pass
    # needs it to compute per-theme card profiles. Deck cards without
    # a card_vector are harmless — the merge step just excludes them
    # from the profile averages.
    projection = {
        "_id": 1, "name": 1, "tags": 1,
        "text_embedding": 1, "card_vector": 1,
    }
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
    merge_threshold: float = tc.DEFAULT_MERGE_THRESHOLD,
) -> DeckProfile:
    """Full pipeline: decklist → resolved cards → themes → classification.

    All Mongo collections default to the project handles; tests pass
    mongomock collections directly.

    `theme_filter` tunes which top-level tags count as "themes" worth
    classifying against (see `theme_classifier.ThemeFilter`).

    `merge_threshold` controls the post-classification merge pass:
    pairs of themes whose averaged card_vector profiles have cosine
    similarity ≥ this value collapse into one cluster (chains fold
    together via Union-Find). The keeper's name is the component
    member with the highest `card_coverage`. Default 0.9 — cautious.
    Pass a negative value to disable the pass entirely.
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

    themes_by_name = {t.name: t for t in themes}
    cards_by_id = {doc["_id"]: doc for doc in deck_cards}

    # Optional merge pass. Returns a merged per-theme-cards dict keyed
    # by the component keeper's name, plus a merge_map saying what each
    # original theme collapsed into. On a no-op (threshold ≤ 0 or
    # degenerate inputs) the dict is unchanged and merge_map is identity.
    merged_per_theme, merge_map = tc.merge_similar_themes(
        per_theme_cards, themes_by_name, cards_by_id,
        threshold=merge_threshold,
    )

    # Invert merge_map so each keeper knows all its constituents.
    constituents_by_keeper: dict[str, list[str]] = {}
    for original, keeper in merge_map.items():
        constituents_by_keeper.setdefault(keeper, []).append(original)

    # Build DeckClusters in coverage-descending order (same order
    # `discover_themes` returned), using each cluster's keeper as the
    # anchor and aggregating centroid + subtree + deck tags from all
    # its constituents.
    clusters: list[DeckCluster] = []
    seen_keepers: set[str] = set()
    for theme in themes:
        keeper = merge_map.get(theme.name, theme.name)
        if keeper in seen_keepers:
            continue
        card_ids = merged_per_theme.get(keeper)
        if not card_ids:
            continue
        seen_keepers.add(keeper)

        constituent_names = sorted(constituents_by_keeper.get(keeper, [keeper]))
        constituent_themes = [themes_by_name[n] for n in constituent_names
                              if n in themes_by_name]

        # Aggregate centroid: L2-normalized mean of the constituents'
        # representative vectors. For a singleton (unmerged) cluster
        # this is just the one representative.
        reps = np.vstack([t.representative for t in constituent_themes])
        mean = reps.mean(axis=0)
        norm = float(np.linalg.norm(mean))
        centroid = mean / norm if norm > 0 else mean

        # Deck tags attributed to this cluster: union of deck slugs
        # that fell under any constituent theme's subtree.
        combined_subtree: set[str] = set()
        for t in constituent_themes:
            combined_subtree |= t.tag_slugs
        deck_tags_in_cluster: set[str] = set()
        for sid in card_ids:
            for slug in cards_by_id[sid].get("tags") or []:
                if slug in combined_subtree:
                    deck_tags_in_cluster.add(slug)

        clusters.append(DeckCluster(
            label=keeper,
            tags=tuple(sorted(deck_tags_in_cluster)),
            centroid=centroid,
            deck_card_ids=tuple(card_ids),  # already sorted by merge_similar_themes
            constituent_themes=tuple(constituent_names),
        ))

    # Noise: tags in the deck universe that no surviving theme covers
    # (irrespective of merging — a tag outside every theme's subtree
    # stays outside).
    covered: set[str] = set()
    for t in themes:
        covered |= t.tag_slugs
    noise_tags = tuple(s for s in tag_universe if s not in covered)

    return DeckProfile(
        deck_card_ids=tuple(doc["_id"] for doc in deck_cards),
        missing_names=tuple(missing),
        tag_universe=tuple(tag_universe),
        clusters=tuple(clusters),
        noise_tags=noise_tags,
        unassigned_card_ids=tuple(unassigned),
    )


# ---------------------------------------------------------------------------
# Alternative mode: unsupervised card_vector clustering + top-level labels
# ---------------------------------------------------------------------------

def build_cluster_profile(
    card_names: Iterable[str],
    *,
    cards_coll: Optional[Collection] = None,
    tags_coll: Optional[Collection] = None,
    min_cluster_size: int = cc.DEFAULT_CLUSTER_MIN_SIZE,
    cluster_selection_epsilon: float = cc.DEFAULT_CLUSTER_SELECTION_EPSILON,
    blocklist: frozenset[str] = tc.DEFAULT_BLOCKLIST,
) -> DeckProfile:
    """Alternative to `build_deck_profile`: cluster cards by `card_vector`.

    Where `build_deck_profile` sorts cards into pre-defined themes,
    this flow goes the other direction — it clusters cards directly
    by their Phase 2 fused `card_vector` and labels each resulting
    group by the top-level oracle tag its member cards most
    frequently share (dropping tags on the blocklist first).

    Shape of the returned `DeckProfile` matches `build_deck_profile`
    so the CLI render is shared, with these semantic differences:
      - `clusters` are card-vector groups, not theme-tag groups.
      - `constituent_themes` is always a single-element tuple naming
        the cluster's label — no merge pass runs in this mode (the
        clustering itself is the primary knob).
      - `noise_tags` is always empty — this mode doesn't partition
        tags.
      - `unassigned_card_ids` holds cards HDBSCAN labelled as noise
        OR cards that lack a stored `card_vector`.
    """
    if cards_coll is None:
        cards_coll = storage.cards_collection()
    if tags_coll is None:
        tags_coll = storage.tags_collection()

    deck_cards, missing = resolve_deck_cards(card_names, cards_coll)
    tag_universe = collect_tag_universe(deck_cards)
    cards_by_id = {doc["_id"]: doc for doc in deck_cards}

    # Cluster the deck's card_vectors. Cards without a card_vector are
    # excluded by the clusterer — surface them as unassigned so the
    # user sees a complete accounting of their deck.
    raw_clusters, hdbscan_noise = cc.cluster_cards_by_vector(
        deck_cards,
        min_cluster_size=min_cluster_size,
        cluster_selection_epsilon=cluster_selection_epsilon,
    )
    clustered_ids: set[str] = set(hdbscan_noise)
    for sids, _centroid in raw_clusters:
        clustered_ids.update(sids)
    cards_without_vector = [
        doc["_id"] for doc in deck_cards
        if doc.get("_id") not in clustered_ids
    ]
    unassigned = sorted(hdbscan_noise + cards_without_vector)

    # Load the tag hierarchy once; label_cluster walks `parent_slugs`
    # per cluster to find top-level ancestors for the vote.
    by_slug = {
        doc["_id"]: doc for doc in tags_coll.find(
            {}, {"_id": 1, "parent_slugs": 1}
        )
    }

    clusters: list[DeckCluster] = []
    for idx, (sids, centroid) in enumerate(raw_clusters):
        label = cc.label_cluster(
            sids, cards_by_id, by_slug,
            blocklist=blocklist,
            fallback_name=f"cluster-{idx}",
        )
        # Tags attributed to this cluster = union of the member cards'
        # tag slugs. Keeps attribution interpretable; same semantic
        # role as DeckCluster.tags under the theme-mode build.
        deck_tags_in_cluster: set[str] = set()
        for sid in sids:
            for slug in cards_by_id[sid].get("tags") or []:
                deck_tags_in_cluster.add(slug)
        clusters.append(DeckCluster(
            label=label,
            tags=tuple(sorted(deck_tags_in_cluster)),
            centroid=centroid,
            deck_card_ids=tuple(sids),
            constituent_themes=(label,),
        ))

    return DeckProfile(
        deck_card_ids=tuple(doc["_id"] for doc in deck_cards),
        missing_names=tuple(missing),
        tag_universe=tuple(tag_universe),
        clusters=tuple(clusters),
        noise_tags=(),  # not applicable in cluster mode
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
        header = (
            f"theme '{cluster.label}'  "
            f"({len(cluster.tags)} tags from deck, "
            f"{len(cluster.deck_card_ids)} cards)"
        )
        if len(cluster.constituent_themes) > 1:
            merged_in = [n for n in cluster.constituent_themes if n != cluster.label]
            header += f"  [merged: {', '.join(merged_in)}]"
        lines.append(header)
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
    """`mtg-deck-profile` — classify a decklist into themes and print.

    Two modes:
      theme (default) — sort cards into themes discovered from the
                        oracle_tags hierarchy (filter + oracle-text
                        tiebreak + optional merge pass).
      cluster         — cluster cards directly on `card_vector`
                        (HDBSCAN, cosine). Label each cluster by its
                        most frequently shared non-blocklisted
                        top-level tag.
    """
    parser = argparse.ArgumentParser(
        prog="mtg-deck-profile",
        description=(
            "Classify a decklist's cards into themes. Default mode "
            "uses the Scryfall oracle_tags hierarchy with an "
            "oracle-text tiebreak; --mode cluster uses unsupervised "
            "HDBSCAN clustering on card_vector with a top-level-tag "
            "voting label."
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
        "--mode", choices=("theme", "cluster"), default="theme",
        help=(
            "theme (default): sort cards into themes discovered from "
            "the oracle_tags hierarchy. cluster: cluster cards by "
            "card_vector and label each group by its most frequent "
            "top-level tag."
        ),
    )
    # ------- theme-mode flags -------
    parser.add_argument(
        "--min-children", type=int, default=tc.DEFAULT_MIN_CHILDREN,
        help=(
            f"[theme mode] Minimum number of child tags a top-level must "
            f"have to count as a theme (default: {tc.DEFAULT_MIN_CHILDREN})."
        ),
    )
    parser.add_argument(
        "--min-coverage", type=int, default=tc.DEFAULT_MIN_COVERAGE,
        help=(
            f"[theme mode] Minimum distinct cards tagged by a theme "
            f"(default: {tc.DEFAULT_MIN_COVERAGE})."
        ),
    )
    parser.add_argument(
        "--max-coverage", type=int, default=tc.DEFAULT_MAX_COVERAGE,
        help=(
            f"[theme mode] Maximum distinct cards tagged by a theme "
            f"(default: {tc.DEFAULT_MAX_COVERAGE})."
        ),
    )
    parser.add_argument(
        "--merge-threshold", type=float, default=tc.DEFAULT_MERGE_THRESHOLD,
        help=(
            f"[theme mode] Cosine-similarity floor for the post-"
            f"classification merge pass (default: {tc.DEFAULT_MERGE_THRESHOLD}). "
            "Negative disables."
        ),
    )
    # ------- cluster-mode flags -------
    parser.add_argument(
        "--cluster-min-size", type=int, default=cc.DEFAULT_CLUSTER_MIN_SIZE,
        help=(
            f"[cluster mode] HDBSCAN min_cluster_size (default: "
            f"{cc.DEFAULT_CLUSTER_MIN_SIZE}). Smallest card count that "
            "counts as its own cluster."
        ),
    )
    parser.add_argument(
        "--cluster-selection-epsilon", type=float,
        default=cc.DEFAULT_CLUSTER_SELECTION_EPSILON,
        help=(
            f"[cluster mode] HDBSCAN cluster_selection_epsilon (default: "
            f"{cc.DEFAULT_CLUSTER_SELECTION_EPSILON}). Merge near-"
            "clusters below this cosine distance."
        ),
    )
    # ------- shared flags -------
    parser.add_argument(
        "--theme-blocklist", default=None,
        help=(
            "Comma-separated top-level slugs to exclude. Overrides the "
            "built-in blocklist. Applies to both modes: theme mode uses "
            "it to filter theme discovery; cluster mode uses it to "
            "filter label candidates. Pass '' to clear entirely."
        ),
    )
    parser.add_argument(
        "--limit", type=int, default=DEFAULT_RENDER_LIMIT,
        help=(
            f"Per-cluster display cap for tags + cards (default: "
            f"{DEFAULT_RENDER_LIMIT}). Everything beyond collapses to a "
            "'(+N more)' marker."
        ),
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    names = sf.read_names(args)
    if not names:
        parser.error("no card names provided (pass names as args or via --file)")

    parsed_blocklist = _parse_blocklist(args.theme_blocklist)
    effective_blocklist = (
        parsed_blocklist if parsed_blocklist is not None else tc.DEFAULT_BLOCKLIST
    )

    cards_coll = storage.cards_collection()
    tags_coll = storage.tags_collection()
    if args.mode == "cluster":
        profile = build_cluster_profile(
            names,
            cards_coll=cards_coll,
            tags_coll=tags_coll,
            min_cluster_size=args.cluster_min_size,
            cluster_selection_epsilon=args.cluster_selection_epsilon,
            blocklist=effective_blocklist,
        )
    else:
        theme_filter = tc.ThemeFilter(
            min_children=args.min_children,
            min_coverage=args.min_coverage,
            max_coverage=args.max_coverage,
            blocklist=effective_blocklist,
        )
        profile = build_deck_profile(
            names,
            cards_coll=cards_coll,
            tags_coll=tags_coll,
            theme_filter=theme_filter,
            merge_threshold=args.merge_threshold,
        )
    name_lookup = _name_lookup_for(profile.deck_card_ids, cards_coll)
    print(_render_profile(profile, name_lookup=name_lookup, limit=args.limit))
    return 0


if __name__ == "__main__":
    sys.exit(main())
