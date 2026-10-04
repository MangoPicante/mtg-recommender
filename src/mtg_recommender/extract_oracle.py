"""Extract oracle text for a specified set of cards from the local cache.

Reads the unified cache written by `scryfall_fetch.py` (default path:
`cache/oracle_texts.json`) and writes a much smaller JSON file
containing only the cards the caller asked for. Handy for shipping a
minimal subset alongside a notebook, a model, or a downstream tool
without dragging along the full ~40 k-entry snapshot.

Input reuse:

    - `read_names` from scryfall_fetch handles --file + positional args,
      with '#' comments and case-insensitive dedup.
    - `resolve_by_name` from scryfall_fetch does the two-step
      name -> [scryfall_ids] -> [cards] lookup, honoring the list-valued
      alias index that keeps ambiguous names (art variants, meld pieces,
      shared face names) visible.

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

`updated_at` is intentionally dropped — it's cache metadata about when
we last refreshed the entry, not information about the card itself.

Usage (after `pip install -e .`, which registers the `extract-oracle` script):
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

# Intra-package relative import — pairs with the `src/mtg_recommender/` layout
# declared in pyproject.toml so this works both when installed (pip install -e .)
# and when run as `python -m mtg_recommender.extract_oracle`.
from . import scryfall_fetch as sf

# Default output lives next to wherever the user runs the CLI from. CWD-relative
# rather than module-relative so the file lands somewhere the user can see,
# rather than inside the installed package tree. Override with -o.
DEFAULT_OUTPUT = Path.cwd() / "oracle_subset.json"

# Fields to copy per card into the output. Chosen to be everything a
# downstream text-based recommender would want, minus the cache-only
# `updated_at` field.
FIELDS = ("name", "mana_cost", "type_line", "oracle_text", "scryfall_id")


def project_for_output(card: dict) -> dict:
    """Copy just the downstream-facing fields from a cached card entry."""
    return {k: card.get(k) for k in FIELDS}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract oracle text for a specified set of cards from the local cache.",
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
    parser.add_argument(
        "--cache", type=Path, default=sf.CACHE_PATH,
        help="Input cache file path (default: cache/oracle_texts.json).",
    )
    args = parser.parse_args()

    names = sf.read_names(args)
    if not names:
        parser.error("no card names provided (pass names as args or via --file)")

    # A missing cache is a setup error (fetcher hasn't been run yet), not
    # a bug in this script — surface it clearly.
    if not args.cache.exists():
        parser.error(
            f"cache file not found: {args.cache}\n"
            "run scryfall_fetch.py first to populate it"
        )

    cache = sf.load_cache(args.cache)

    # Build the output dict in one pass, tracking misses and ambiguities
    # so we can report them all at the end instead of interleaving with
    # progress output.
    subset: dict[str, list[dict]] = {}
    missing: list[str] = []
    ambiguous: list[tuple[str, int]] = []

    for name in names:
        matches = sf.resolve_by_name(cache, name)
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
            f"\nwarning: {len(missing)} name(s) not found in cache:",
            file=sys.stderr,
        )
        for name in missing:
            print(f"  {name}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
