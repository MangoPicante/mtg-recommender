"""Extract oracle text for a specified set of cards from MongoDB.

Reads the cards collection populated by `scryfall-fetch` and writes a
much smaller JSON file containing only the cards the caller asked for.
Handy for shipping a minimal subset alongside a notebook, a model, or a
downstream tool without dragging along the full ~40 k-entry collection
or requiring the consumer to speak Mongo.

Input reuse:

    - `read_names` from scryfall_fetch handles --file + positional args,
      with '#' comments and case-insensitive dedup.
    - `find_cards_by_name` from scryfall_fetch does the Mongo lookup via
      the multikey index on `names`, returning every card that matches
      the lowered input so ambiguous names (art variants, meld pieces,
      shared face names) remain visible.

Output shape (JSON):

    {
      "<the name the caller passed in>": [
        {
          "name": "...",              # canonical Scryfall name
          "mana_cost": "...",
          "type_line": "...",
          "oracle_text": "...",
          "scryfall_id": "..."
        },
        ...
      ],
      ...
    }

    Values are LISTS because a single input name can legitimately match
    more than one cached card. Downstream consumers can pick with
    whatever criterion suits them (canonical-name shape, type line, etc.).

The `scryfall_id` field in the output is renamed from Mongo's `_id` at
export time; storage keeps a single id field (`_id`) rather than
duplicating it. `updated_at`, `names`, `oracle_id`, `oracle_text_sha`,
and `tags` are intentionally dropped — the subset is meant for text
consumers that just want "what does this card say?", not full-fat
records.

Usage (after `pip install -e .`, with MONGODB_URI set in .env):
    extract-oracle "Lightning Bolt" "Counterspell" -o subset.json
    extract-oracle --file cards.txt -o deck_oracle.json
    extract-oracle --file cards.txt        # defaults to ./oracle_subset.json

Equivalently from a source checkout without installing:
    python -m mtg_recommender.extract_oracle --file cards.txt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Intra-package relative imports — pair with the src/mtg_recommender/
# layout declared in pyproject.toml.
from . import scryfall_fetch as sf
from . import storage

# Default output lives next to wherever the user runs the CLI from. CWD-relative
# rather than module-relative so the file lands somewhere the user can see,
# rather than inside the installed package tree. Override with -o.
DEFAULT_OUTPUT = Path.cwd() / "oracle_subset.json"

# Fields to copy per card into the output, read verbatim from the card
# doc. Chosen to be everything a downstream text-based recommender would
# want, minus Mongo bookkeeping (names, updated_at, oracle_text_sha) and
# fields owned by other modules (tags, oracle_id — handy inside Mongo
# but noise to a text consumer).
_PASSTHROUGH_FIELDS = ("name", "mana_cost", "type_line", "oracle_text")


def project_for_output(card: dict) -> dict:
    """Copy the downstream-facing fields from a cached card entry.

    `_id` is renamed to `scryfall_id` in the output: Mongo's primary-key
    name is an implementation detail, and consumers of this JSON have
    always seen `scryfall_id` as the semantic field. The storage layer
    no longer duplicates the id into a `scryfall_id` field, so we do the
    one-line rename here instead.
    """
    out = {k: card.get(k) for k in _PASSTHROUGH_FIELDS}
    out["scryfall_id"] = card.get("_id")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract oracle text for a specified set of cards from MongoDB.",
    )
    parser.add_argument("cards", nargs="*", help="Card names (quote multi-word names).")
    parser.add_argument(
        "-f", "--file",
        help="Path to a file with one card name per line (# for comments).",
    )
    parser.add_argument(
        "-o", "--output", type=Path, default=DEFAULT_OUTPUT,
        help="Output JSON path (default: ./oracle_subset.json).",
    )
    args = parser.parse_args()

    names = sf.read_names(args)
    if not names:
        parser.error("no card names provided (pass names as args or via --file)")

    coll = storage.cards_collection()
    # An empty collection is a setup error (fetcher hasn't been run
    # yet), not a bug in this script — surface it clearly rather than
    # silently producing an empty output file.
    if coll.estimated_document_count() == 0:
        parser.error(
            f"cards collection '{coll.database.name}.{coll.name}' is empty\n"
            "run scryfall-fetch first to populate it"
        )

    # Build the output dict in one pass, tracking misses and ambiguities
    # so we can report them all at the end instead of interleaving with
    # progress output.
    subset: dict[str, list[dict]] = {}
    missing: list[str] = []
    ambiguous: list[tuple[str, int]] = []

    for name in names:
        matches = sf.find_cards_by_name(coll, name)
        if not matches:
            missing.append(name)
            continue
        subset[name] = [project_for_output(m) for m in matches]
        if len(matches) > 1:
            ambiguous.append((name, len(matches)))

    # mkdir(parents=True) is safe if the directory already exists.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        # sort_keys keeps output diffs stable; ensure_ascii=False keeps
        # Unicode symbols (mana, curly quotes) readable rather than
        # escaped as \uXXXX.
        json.dumps(subset, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )

    # Summary line: total card projections written and how many distinct
    # input names produced at least one match.
    total = sum(len(v) for v in subset.values())
    print(f"wrote {total} card entries ({len(subset)} unique input names) to {args.output}")

    if ambiguous:
        # Ambiguity isn't an error — it's real data. Report it so the
        # caller knows their downstream code will need to pick.
        print(
            f"\nnote: {len(ambiguous)} input name(s) matched multiple cached cards:",
            file=sys.stderr,
        )
        for name, count in ambiguous:
            print(f"  {name}: {count} matches", file=sys.stderr)

    if missing:
        # Missing names ARE actionable — either the spelling was wrong,
        # or the card hasn't been fetched yet. Exit non-zero so callers
        # in a pipeline notice.
        print(
            f"\nwarning: {len(missing)} name(s) not found in collection:",
            file=sys.stderr,
        )
        for name in missing:
            print(f"  {name}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
