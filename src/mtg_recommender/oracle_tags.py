"""Download, cache, and attach Scryfall oracle tags.

Oracle tags are a community-sourced classification of cards by what they
DO ("spot removal", "evasion", "tutor-creature-giant", ...) rather than
what their rules text literally says. They are maintained by the Scryfall
Tagger project and published as their own bulk file at /bulk-data
(type = "oracle_tags"), separate from the oracle_cards bulk the main
fetcher consumes.

Shape of the raw oracle_tags JSONL (one line per tag):

    {
      "object": "tag",
      "id": "<uuid>",
      "label": "spot removal",
      "slug": "spot-removal",
      "type": "oracle",
      "description": "..." | null,
      "parent_ids": ["<tag uuid>", ...],
      "child_ids":  ["<tag uuid>", ...],
      "aliases":    ["other-slug", ...],
      "taggings": [
        {"oracle_id": "<uuid>", "weight": "median", "annotation": "..."},
        ...
      ]
    }

The join to cached cards is by `oracle_id` (NOT scryfall_id). The oracle
id names the GAMEPLAY card — a single oracle_id can map to many
scryfall_ids across reprints. `scryfall_fetch.extract_card_fields` now
carries oracle_id on every cached card, so the join is a direct dict
lookup.

What this module writes:

1. cache/oracle_tags.json — the tag catalog, slug-keyed:

    {
      "snapshot_updated_at": "<Scryfall snapshot UTC ISO 8601>",
      "tags": {
        "<slug>": {
          "id": "<uuid>",            # the Scryfall tag id, kept for cross-ref
          "label": "...",            # human-readable form of the slug
          "description": "..." | null,
          "parent_slugs": ["..."],   # parent uuids resolved to slugs for readability
          "child_slugs":  ["..."],
          "aliases":      ["..."]
        },
        ...
      }
    }

   Slugs are used as keys (rather than uuids) because they're stable,
   URL-safe, and human-readable — far more useful at the REPL than
   32-char hex ids. Parent/child references are translated from uuids
   to slugs during catalog construction via a one-pass index.

2. cache/oracle_texts.json — the existing cards cache gains a
   `tags: ["<slug>", ...]` field on every card entry. Weight and
   annotation are dropped: 99.7 % of weights are "median" (and the
   remaining categories are rare enough to ignore for similarity
   scoring in v1), and annotations are tagger notes rather than
   scoring signal. Cards whose oracle_id was never tagged get
   `tags: []` so downstream code can rely on the field existing.

Taggings whose oracle_id doesn't match any cached card are counted and
reported — users with a partial cache (e.g. a single decklist's worth of
cards) will see a large unmatched count, which is expected. A near-100 %
unmatched rate usually means the cards cache was built before oracle_id
was added to the projection; `scryfall-fetch --refresh` repopulates.

Usage (after `pip install -e .`, which registers `scryfall-fetch-tags`):

    scryfall-fetch-tags              # downloads if snapshot is new; attaches
    scryfall-fetch-tags --refresh    # redownload even if snapshot matches

Equivalently from a source checkout without installing:

    python -m mtg_recommender.oracle_tags
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
import urllib.request
from pathlib import Path

# Pull HEADERS / SCRYFALL_BASE / http_get_json / cache I/O from the sibling
# module. Keeping the Scryfall request details in one place means a change
# to the User-Agent, timeout policy, or JSON shape only has one home.
from . import scryfall_fetch as sf


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# Tag catalog lives next to the cards cache under ./cache/. CWD-relative for
# the same reason scryfall_fetch's CACHE_DIR is: the modules live inside an
# installable package tree, so a module-relative path would either bury the
# cache inside src/ during development or vanish into site-packages when
# installed as a wheel.
TAGS_CACHE_PATH = Path.cwd() / "cache" / "oracle_tags.json"


# ---------------------------------------------------------------------------
# Bulk metadata + download
# ---------------------------------------------------------------------------

def get_bulk_oracle_tags_metadata() -> dict:
    """Return the /bulk-data entry for the oracle_tags dump.

    Mirrors `scryfall_fetch.get_bulk_oracle_metadata` but filters for the
    `oracle_tags` type instead of `oracle_cards`. Raises if Scryfall ever
    stops publishing the type (unlikely — the Tagger project depends on
    it) so a schema change surfaces loudly rather than silently.
    """
    data = sf.http_get_json(f"{sf.SCRYFALL_BASE}/bulk-data")
    for entry in data.get("data", []):
        if entry.get("type") == "oracle_tags":
            return entry
    raise RuntimeError("oracle_tags entry not found in /bulk-data response")


def download_bulk_oracle_tags(meta: dict) -> list[dict]:
    """Download the oracle_tags bulk file and return it as a list of tag dicts.

    Same JSONL + gzip handling as `scryfall_fetch.download_bulk_oracle_cards`.
    Deliberately not sharing the function because the two bulks have slightly
    different metadata fields (oracle_tags has no `size` field, only
    `compressed_size`) and the user-facing progress message differs.
    """
    # oracle_tags only exposes `jsonl_download_uri`; there is no
    # `download_uri` field on this bulk type.
    download_uri = meta["jsonl_download_uri"]
    # `compressed_size` is the .gz payload size in bytes; show MB so the
    # user has a rough sense of what's downloading. Oracle_tags is small
    # (~6 MB gz, ~19 MB JSONL) compared to oracle_cards (~24 MB gz).
    size_mb = meta.get("compressed_size", 0) / 1_000_000
    print(f"bulk  : downloading oracle_tags (~{size_mb:.0f} MB gz, snapshot {meta['updated_at']})")

    req = urllib.request.Request(download_uri, headers=sf.HEADERS)
    # Shorter timeout than oracle_cards (300 s) because this file is far
    # smaller; 120 s is still generous for 6 MB on a slow link.
    with urllib.request.urlopen(req, timeout=120) as resp:
        raw = resp.read()

    # Explicit gzip detection via magic bytes — same pattern as
    # scryfall_fetch. urllib does not auto-decompress without an
    # Accept-Encoding header, and we don't want to depend on proxy
    # behaviour.
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)

    tags: list[dict] = []
    for line in raw.decode("utf-8").splitlines():
        line = line.strip()
        if line:
            tags.append(json.loads(line))
    return tags


# ---------------------------------------------------------------------------
# Catalog + inverted index construction
# ---------------------------------------------------------------------------

def _build_id_to_slug(tags: list[dict]) -> dict[str, str]:
    """Index tag UUIDs to their slugs, for resolving parent/child references.

    The raw bulk stores parent_ids and child_ids as uuids; the catalog
    stores them as slugs. One pass through the data is enough since
    every tag appears as its own JSONL line.
    """
    return {t["id"]: t["slug"] for t in tags if t.get("id") and t.get("slug")}


def extract_tag_fields(raw: dict, id_to_slug: dict[str, str]) -> dict:
    """Project a raw tag object down to the catalog-level fields we keep.

    Taggings are intentionally stripped — they're inverted into the cards
    cache by `build_oracle_id_to_slugs` and have no reason to live in the
    tag catalog, which is meant for human lookup and Phase 2 embedding.
    """
    # Translate parent/child uuids to slugs via the pre-built index. An
    # unknown uuid (shouldn't happen, but data drift is a thing) is
    # dropped silently rather than leaving a dangling uuid in the output.
    parent_slugs = [id_to_slug[pid] for pid in raw.get("parent_ids", []) if pid in id_to_slug]
    child_slugs = [id_to_slug[cid] for cid in raw.get("child_ids", []) if cid in id_to_slug]
    return {
        "id": raw.get("id"),
        "label": raw.get("label"),
        "description": raw.get("description"),
        "parent_slugs": parent_slugs,
        "child_slugs": child_slugs,
        "aliases": list(raw.get("aliases", []) or []),
    }


def build_tag_catalog(tags: list[dict]) -> dict[str, dict]:
    """Produce the slug-keyed catalog dict from the raw bulk list."""
    id_to_slug = _build_id_to_slug(tags)
    catalog: dict[str, dict] = {}
    for raw in tags:
        slug = raw.get("slug")
        # Skip entries without a slug — nothing to key them under. In
        # practice every oracle tag has a slug; this is a defensive guard.
        if not slug:
            continue
        catalog[slug] = extract_tag_fields(raw, id_to_slug)
    return catalog


def build_oracle_id_to_slugs(tags: list[dict]) -> dict[str, list[str]]:
    """Invert tag.taggings into oracle_id -> sorted list of slugs.

    This is the per-card view of the tagging data: for each oracle entity,
    what are all the tags that apply to it? Slugs are sorted so the output
    is deterministic (stable diffs when the cards cache is re-saved).

    Dedup via a set per card: a tag should never apply twice to the same
    card in a well-formed bulk, but we don't want to assume that.
    """
    by_oracle_id: dict[str, set[str]] = {}
    for tag in tags:
        slug = tag.get("slug")
        if not slug:
            continue
        for tagging in tag.get("taggings", []) or []:
            oid = tagging.get("oracle_id")
            if not oid:
                continue
            by_oracle_id.setdefault(oid, set()).add(slug)
    return {oid: sorted(slugs) for oid, slugs in by_oracle_id.items()}


# ---------------------------------------------------------------------------
# Attach tags to the cards cache
# ---------------------------------------------------------------------------

def attach_tags_to_cards(
    cards_cache: dict,
    oracle_id_to_slugs: dict[str, list[str]],
) -> tuple[int, int, int]:
    """Set `tags: [slug, ...]` on every card in the cache.

    Returns a tuple:

        (cards_matched, cards_without_tags, taggings_unmatched)

    Semantics:
      - cards_matched         : card had an oracle_id that appeared in the
                                tagger data, and tags were attached.
      - cards_without_tags    : card had an oracle_id but no tags apply
                                to it (empty list attached), OR the card
                                had no oracle_id at all (also empty list).
                                The important invariant is that every
                                card ends up with a `tags` field so
                                downstream code can rely on it existing.
      - taggings_unmatched    : oracle_ids in the tagger data that didn't
                                correspond to any cached card. Expected
                                to be large when the cards cache is a
                                partial subset (e.g. a single decklist).

    `tags` is always overwritten, never merged. A fresh bulk snapshot is
    authoritative: if a tag was removed from the tagger project, the
    card should lose it on the next import.
    """
    matched = 0
    empty = 0
    cards = cards_cache.get("cards", {})
    # Track which oracle_ids we successfully attached so we can compute
    # the unmatched set without a second pass over the cards cache.
    attached_oracle_ids: set[str] = set()
    for card in cards.values():
        oid = card.get("oracle_id")
        if oid and oid in oracle_id_to_slugs:
            card["tags"] = list(oracle_id_to_slugs[oid])
            attached_oracle_ids.add(oid)
            matched += 1
        else:
            # Either no oracle_id (stale cache from before the field was
            # added) or no tags for this card. Either way, give it an
            # empty list so consumers don't need to .get("tags", []).
            card["tags"] = []
            empty += 1
    taggings_unmatched = len(oracle_id_to_slugs) - len(attached_oracle_ids)
    return matched, empty, taggings_unmatched


# ---------------------------------------------------------------------------
# Tag-cache I/O
# ---------------------------------------------------------------------------

def load_tag_cache(path: Path) -> dict:
    """Load the tag catalog from disk, or return a fresh shell on first run.

    Shell shape ({"snapshot_updated_at": None, "tags": {}}) mirrors
    scryfall_fetch.load_cache so callers can always assume the keys
    exist.
    """
    if not path.exists():
        return {"snapshot_updated_at": None, "tags": {}}
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    # Defensive: a cache saved under a future schema with missing keys
    # should still look like a valid shell.
    data.setdefault("snapshot_updated_at", None)
    data.setdefault("tags", {})
    return data


def save_tag_cache(path: Path, catalog: dict) -> None:
    """Write the tag catalog to disk, pretty-printed and UTF-8."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        # sort_keys for stable diffs; ensure_ascii=False to keep Unicode
        # readable rather than \uXXXX-escaped.
        json.dump(catalog, f, indent=2, ensure_ascii=False, sort_keys=True)


# ---------------------------------------------------------------------------
# Freshness check
# ---------------------------------------------------------------------------

def is_fresh(tag_cache: dict, snapshot_updated_at: str) -> bool:
    """True if the on-disk tag cache already matches the current snapshot.

    oracle_tags is small enough that we always redownload when stale —
    there is no per-tag timestamp like the cards cache has. Exact string
    match on the snapshot timestamp is sufficient because Scryfall writes
    its own ISO 8601 values with microsecond precision and no reformatting.
    """
    stored = tag_cache.get("snapshot_updated_at")
    return bool(stored) and stored == snapshot_updated_at


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download the Scryfall oracle_tags bulk and attach tags to the cards cache.",
    )
    parser.add_argument(
        "--refresh", action="store_true",
        help="Redownload even if the on-disk tag cache already matches the current snapshot.",
    )
    parser.add_argument(
        "--cards", type=Path, default=sf.CACHE_PATH,
        help="Path to the cards cache written by scryfall-fetch (default: cache/oracle_texts.json).",
    )
    parser.add_argument(
        "--tags", type=Path, default=TAGS_CACHE_PATH,
        help="Path to the tag catalog file (default: cache/oracle_tags.json).",
    )
    args = parser.parse_args()

    # Cards cache is required: tag attachment can't do its job without it.
    # Keep the error message pointing to the obvious next step.
    if not args.cards.exists():
        parser.error(
            f"cards cache not found: {args.cards}\n"
            "run scryfall-fetch first to populate it"
        )

    tag_cache = load_tag_cache(args.tags)

    # Step 1: metadata. One tiny JSON request so we can compare timestamps
    # before committing to the 6 MB download.
    meta = get_bulk_oracle_tags_metadata()
    snapshot_updated_at = meta["updated_at"]
    if is_fresh(tag_cache, snapshot_updated_at) and not args.refresh:
        print(f"tags  : already fresh (snapshot {snapshot_updated_at}); pass --refresh to force")
        # Still (re-)attach to the cards cache in case cards have been
        # added since the last tag import. Reload the catalog from disk
        # as the inverted-index source so we don't rebuild it here.
        # Actually — we only have the catalog on disk, not the taggings
        # (we stripped those during the first import). So attach is a
        # no-op in this branch; a user who added new cards and wants
        # them tagged should pass --refresh. Make that explicit:
        print(
            "        new cards added since last import? pass --refresh to re-attach.",
            file=sys.stderr,
        )
        return 0

    # Step 2: download the bulk + build the catalog and inverted index.
    tags_raw = download_bulk_oracle_tags(meta)
    print(f"parsed: {len(tags_raw)} tag entries")
    catalog = build_tag_catalog(tags_raw)
    oracle_id_to_slugs = build_oracle_id_to_slugs(tags_raw)
    print(f"catalog: {len(catalog)} tags")
    print(f"index  : {len(oracle_id_to_slugs)} oracle_ids with at least one tag")

    # Step 3: save the catalog with the snapshot timestamp so a future
    # run can tell whether this snapshot has already been processed.
    new_cache = {"snapshot_updated_at": snapshot_updated_at, "tags": catalog}
    save_tag_cache(args.tags, new_cache)
    print(f"saved  : {args.tags}")

    # Step 4: attach to the cards cache. Load / mutate / save using the
    # cards module's helpers so the shell shape stays consistent.
    cards_cache = sf.load_cache(args.cards)
    matched, empty, unmatched = attach_tags_to_cards(cards_cache, oracle_id_to_slugs)
    sf.save_cache(args.cards, cards_cache)
    total_cards = matched + empty
    print(
        f"cards  : {matched}/{total_cards} tagged, {empty} without tags, "
        f"{unmatched} tagger oracle_ids didn't match any cached card"
    )
    if total_cards and matched == 0:
        # Loud failure mode worth calling out: the user's cards cache
        # has no oracle_ids at all, which usually means it was written
        # before oracle_id was added to the projection. Point at the
        # obvious remediation rather than leave them puzzled.
        print(
            "warning: no cards were tagged. The cards cache may predate the "
            "oracle_id field — run `scryfall-fetch --refresh ...` to repopulate it.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    # sys.exit propagates the int return code so shells / CI can react.
    sys.exit(main())
