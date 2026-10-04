"""Ad-hoc inspection CLI for the MongoDB-backed card and tag collections.

Subcommands:
    card   — show a card's document (by name, case-insensitive).
    tag    — show a tag's catalog entry + card count.
    list   — list cards, optionally filtered by tag.
    stats  — collection sizes, tagged %, top tags, snapshot timestamps.

Nothing here mutates the database; this module is purely read-path. CLI
output is human-readable text rather than JSON — use `extract-oracle`
for a machine-readable subset.

Usage (after `pip install -e .`):
    mtg-inspect card "Wrath of God"
    mtg-inspect tag spot-removal
    mtg-inspect list --tag spot-removal --limit 20
    mtg-inspect stats

Equivalently from a source checkout:
    python -m mtg_recommender.explore card "Wrath of God"
"""
from __future__ import annotations

import argparse
import sys
from typing import Iterable, Optional

from . import storage


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _fmt_card(doc: dict) -> str:
    """Render one card document as a block of text."""
    name = doc.get("name") or "(unnamed)"
    sid = doc.get("scryfall_id") or doc.get("_id") or "?"
    mana = doc.get("mana_cost") or ""
    type_line = doc.get("type_line") or ""
    oracle = doc.get("oracle_text") or ""
    tags = doc.get("tags") or []

    lines = [f"{name}  [{sid}]"]
    # Header line format depends on which pieces exist, so build it piecewise.
    header_bits = []
    if mana:
        header_bits.append(mana)
    if type_line:
        header_bits.append(type_line)
    if header_bits:
        lines.append("    " + "  —  ".join(header_bits))
    if oracle:
        # Preserve Scryfall's newlines inside oracle text so the output
        # matches what players see on the card.
        for ol in oracle.splitlines():
            lines.append(f"    {ol}")
    if tags:
        lines.append(f"    tags: {', '.join(tags)}")
    else:
        lines.append("    tags: (none)")
    return "\n".join(lines)


def _fmt_tag(doc: dict, card_count: int) -> str:
    """Render one tag document as a block of text."""
    slug = doc.get("_id") or "?"
    label = doc.get("label") or slug
    description = doc.get("description") or "(no description)"
    parents = doc.get("parent_slugs") or []
    children = doc.get("child_slugs") or []
    aliases = doc.get("aliases") or []
    lines = [
        f"{slug}  ({label})",
        f"    description : {description}",
        f"    parents     : {', '.join(parents) if parents else '(none)'}",
        f"    children    : {', '.join(children) if children else '(none)'}",
        f"    aliases     : {', '.join(aliases) if aliases else '(none)'}",
        f"    cards tagged: {card_count}",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Query helpers (shared across subcommands)
# ---------------------------------------------------------------------------

def _find_cards_by_name(name: str) -> list[dict]:
    """Case-insensitive lookup against the names index."""
    coll = storage.cards_collection()
    return list(coll.find({"names": name.lower()}))


def _top_tags(limit: int = 10) -> list[tuple[str, int]]:
    """Aggregate the top `limit` tags by card count across the cards collection."""
    coll = storage.cards_collection()
    # $unwind flattens the tags array so each tagging becomes its own row;
    # $group then counts per slug. mongomock supports this pipeline shape.
    pipeline = [
        {"$unwind": "$tags"},
        {"$group": {"_id": "$tags", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}},
        {"$limit": limit},
    ]
    return [(row["_id"], row["count"]) for row in coll.aggregate(pipeline)]


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------

def cmd_card(args: argparse.Namespace) -> int:
    """Show the full doc(s) for a given card name."""
    matches = _find_cards_by_name(args.name)
    if not matches:
        print(f"not found: {args.name}", file=sys.stderr)
        return 1
    if len(matches) > 1:
        # Ambiguity is real data (art-card variants, meld pieces, shared face
        # names). Show every match so the user can pick.
        print(f"{len(matches)} matches for '{args.name}':\n")
    for i, doc in enumerate(matches):
        if i:
            print()
        print(_fmt_card(doc))
    return 0


def cmd_tag(args: argparse.Namespace) -> int:
    """Show a tag's catalog entry + card count."""
    tags_coll = storage.tags_collection()
    doc = tags_coll.find_one({"_id": args.slug})
    if not doc:
        print(f"tag not found: {args.slug}", file=sys.stderr)
        return 1
    card_count = storage.cards_collection().count_documents({"tags": args.slug})
    print(_fmt_tag(doc, card_count))
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    """List cards, optionally filtered by tag."""
    coll = storage.cards_collection()
    query: dict = {}
    if args.tag:
        query["tags"] = args.tag
    total = coll.count_documents(query)
    if total == 0:
        if args.tag:
            print(f"no cards tagged '{args.tag}'", file=sys.stderr)
        else:
            print("no cards in collection", file=sys.stderr)
        return 1

    # Sort by name for stable, scannable output.
    cursor = coll.find(query, {"name": 1, "scryfall_id": 1}).sort("name", 1).limit(args.limit)
    rows = list(cursor)
    header_what = f"cards tagged '{args.tag}'" if args.tag else "cards"
    print(f"showing {len(rows)} of {total} {header_what}:")
    for r in rows:
        print(f"  {r.get('name', '?')}")
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    """Show collection sizes, tagged %, top tags, snapshot timestamps."""
    cards = storage.cards_collection()
    tags = storage.tags_collection()

    total_cards = cards.estimated_document_count()
    with_oracle_id = cards.count_documents({"oracle_id": {"$exists": True, "$ne": None}})
    # A card is "tagged" if its tags array exists and is non-empty.
    with_tags = cards.count_documents({"tags": {"$exists": True, "$ne": []}})
    tagged_pct = (100 * with_tags / total_cards) if total_cards else 0

    total_tags = tags.estimated_document_count()
    top = _top_tags(limit=args.top)

    tags_snapshot = storage.get_snapshot_timestamp("oracle_tags")

    print(f"cards.{cards.name}:")
    print(f"  total          : {total_cards}")
    print(f"  with oracle_id : {with_oracle_id}")
    print(f"  with tags      : {with_tags} ({tagged_pct:.1f}%)")
    print(f"  without tags   : {total_cards - with_tags}")
    print()
    print(f"tags.{tags.name}:")
    print(f"  total          : {total_tags}")
    if top:
        print(f"  top {len(top)} by card count:")
        width = max(len(slug) for slug, _ in top)
        for slug, count in top:
            print(f"    {slug:<{width}}  {count}")
    print()
    print("meta:")
    print(f"  oracle_tags snapshot: {tags_snapshot or '(not set)'}")
    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="mtg-inspect",
        description="Ad-hoc read-only inspection of the Mongo-backed card and tag collections.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    card_p = sub.add_parser("card", help="Show a card's document(s) by name.")
    card_p.add_argument("name", help="Card name (quote multi-word names).")
    card_p.set_defaults(func=cmd_card)

    tag_p = sub.add_parser("tag", help="Show a tag's catalog entry.")
    tag_p.add_argument("slug", help="Tag slug (e.g. spot-removal).")
    tag_p.set_defaults(func=cmd_tag)

    list_p = sub.add_parser("list", help="List cards, optionally filtered by tag.")
    list_p.add_argument("--tag", help="Filter to cards carrying this tag slug.")
    list_p.add_argument("--limit", type=int, default=20, help="Max rows to print (default: 20).")
    list_p.set_defaults(func=cmd_list)

    # argparse runs help strings through %-formatting, so a bare `%` would
    # raise at parse time; escape it as `%%`.
    stats_p = sub.add_parser("stats", help="Collection sizes, tagged %%, top tags.")
    stats_p.add_argument("--top", type=int, default=10, help="How many top tags to show (default: 10).")
    stats_p.set_defaults(func=cmd_stats)

    args = parser.parse_args(list(argv) if argv is not None else None)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
