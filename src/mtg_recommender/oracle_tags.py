"""Download Scryfall oracle tags into MongoDB and attach them to cards.

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

The join to cards is by `oracle_id` (NOT scryfall_id). The oracle id
names the GAMEPLAY card — a single oracle_id can map to many
scryfall_ids across reprints. `scryfall_fetch.extract_card_fields`
carries oracle_id on every card document and `storage.ensure_indexes`
creates a Mongo index on it, so the join is an indexed query.

What this module writes:

1. tags collection — one document per oracle tag, _id = slug. Shape:

    {
      "_id":            "<slug>",
      "scryfall_tag_id":"<uuid>",         # kept for cross-ref with the API
      "label":          "...",            # human-readable form of the slug
      "description":    "..." | null,
      "parent_slugs":   ["..."],          # parent uuids resolved to slugs
      "child_slugs":    ["..."],
      "aliases":        ["..."]
    }

   Slugs are used as _id (rather than uuids) because they're stable,
   URL-safe, and human-readable — far more useful at the REPL than
   32-char hex ids. Parent/child references are translated from uuids
   to slugs during catalog construction via a one-pass index.

2. cards collection — every card document gains a `tags: ["<slug>", ...]`
   field. Weight and annotation are dropped: 99.7 % of weights are
   "median" (and the remaining categories are rare enough to ignore
   for similarity scoring in v1), and annotations are tagger notes
   rather than scoring signal. Cards whose oracle_id was never tagged
   get `tags: []` so downstream code can rely on the field existing.

3. meta collection — a single doc with `_id = "oracle_tags"` records
   the snapshot timestamp so a future run can skip the download when
   the Scryfall snapshot hasn't moved.

Taggings whose oracle_id doesn't match any card are counted and
reported — users with a partial collection (e.g. a single decklist's
worth of cards) will see a large unmatched count, which is expected. A
near-100 % unmatched rate usually means the cards cache was built
before oracle_id was added to the projection; `scryfall-fetch
--refresh` repopulates.

Note on the --refresh / fresh-snapshot interaction: when the stored
snapshot timestamp matches the current /bulk-data snapshot, the
module skips the download AND the re-attach step. That's because the
raw taggings aren't retained after import — only the catalog and the
per-card tags array. A user who added new cards to the collection
since the last tag import and wants them tagged should pass
--refresh to force a redownload.

Usage (after `pip install -e .`, with MONGODB_URI set in .env):

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

from pymongo import UpdateMany
from pymongo.collection import Collection

from . import scryfall_fetch as sf
from . import storage


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Key used in the meta collection to record the oracle_tags snapshot
# timestamp. Namespaced so a future "oracle_cards" meta entry doesn't
# collide.
META_SOURCE = "oracle_tags"


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
# Catalog + inverted index construction (pure, no DB access)
# ---------------------------------------------------------------------------

def _build_id_to_slug(tags: list[dict]) -> dict[str, str]:
    """Index tag UUIDs to their slugs, for resolving parent/child references.

    The raw bulk stores parent_ids and child_ids as uuids; the catalog
    stores them as slugs. One pass through the data is enough since
    every tag appears as its own JSONL line.
    """
    return {t["id"]: t["slug"] for t in tags if t.get("id") and t.get("slug")}


def extract_tag_fields(raw: dict, id_to_slug: dict[str, str]) -> dict:
    """Project a raw tag object into the Mongo document shape (minus _id).

    _id is set by the inserter from the slug, so it's intentionally
    absent here — this function is reused by tests that only care
    about field projection.

    `id` on the raw object is Scryfall's internal tag UUID; the
    projection renames it to `scryfall_tag_id` so it doesn't look like
    an attempt to set Mongo's reserved `_id` and so readers of the
    stored doc know what they're looking at.

    Taggings are intentionally stripped — they're inverted into the
    per-card tags array by `build_oracle_id_to_slugs` and have no
    reason to live in the tag catalog, which is meant for human lookup
    and Phase 2 embedding.
    """
    # Translate parent/child uuids to slugs via the pre-built index. An
    # unknown uuid (shouldn't happen, but data drift is a thing) is
    # dropped silently rather than leaving a dangling uuid in the output.
    parent_slugs = [id_to_slug[pid] for pid in raw.get("parent_ids", []) if pid in id_to_slug]
    child_slugs = [id_to_slug[cid] for cid in raw.get("child_ids", []) if cid in id_to_slug]
    return {
        "scryfall_tag_id": raw.get("id"),
        "label": raw.get("label"),
        "description": raw.get("description"),
        "parent_slugs": parent_slugs,
        "child_slugs": child_slugs,
        "aliases": list(raw.get("aliases", []) or []),
    }


def build_tag_catalog(tags: list[dict]) -> dict[str, dict]:
    """Produce the slug-keyed catalog dict from the raw bulk list.

    Returned as a dict (slug -> projected fields) rather than a list of
    docs so callers can look tags up by slug without a second pass.
    `upsert_tag_catalog` serialises to {_id: slug, ...} docs at insert
    time.
    """
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
    is deterministic (stable updates when the collection is re-tagged).

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
# Mongo write helpers
# ---------------------------------------------------------------------------

def upsert_tag_catalog(coll: Collection, catalog: dict[str, dict]) -> None:
    """Replace the tags collection contents with `catalog`.

    Delete-then-insert rather than per-doc upsert because the catalog
    is authoritative: a tag removed from the Scryfall tagger project
    should disappear from Mongo too. Insert-many sends everything in a
    single command (4,560 small docs is comfortably under Mongo's
    16 MB cmd size limit), so this is one round trip per side.
    """
    coll.delete_many({})
    if not catalog:
        return
    docs = [{"_id": slug, **fields} for slug, fields in catalog.items()]
    coll.insert_many(docs)


def attach_tags_to_cards(
    coll: Collection,
    oracle_id_to_slugs: dict[str, list[str]],
) -> tuple[int, int, int]:
    """Set `tags` on every card doc. Returns (matched, empty, unmatched).

    Semantics:
      - matched  : card had an oracle_id that appeared in the tagger
                   data, and tags were attached.
      - empty    : card had an oracle_id but no tags apply to it (empty
                   list set), OR the card had no oracle_id at all. The
                   important invariant is that every card ends up with
                   a `tags` field so downstream code can rely on it
                   existing.
      - unmatched: oracle_ids in the tagger data that didn't correspond
                   to any card. Expected to be large when the cards
                   collection is a partial subset (e.g. one decklist).

    Implementation: clear every card's `tags` first (so a tag removed
    upstream disappears from affected cards), then bulk-apply the new
    tags one UpdateMany per oracle_id. The index on `oracle_id` keeps
    this efficient; the full ~36k-op bulk_write fits well under
    Mongo's single-command 16 MB limit.
    """
    # Step 1: clear every card's tags so previously-tagged cards whose
    # oracle_id has lost all its tags get an empty list rather than
    # keeping stale data from a prior import.
    coll.update_many({}, {"$set": {"tags": []}})

    # Step 2: bulk-apply new tags. UpdateMany per oracle_id uses the
    # multikey-safe equality query against the indexed field.
    ops = [
        UpdateMany({"oracle_id": oid}, {"$set": {"tags": slugs}})
        for oid, slugs in oracle_id_to_slugs.items()
    ]
    if ops:
        coll.bulk_write(ops, ordered=False)

    # Step 3: tally up the three stats for the user-facing summary.
    total = coll.estimated_document_count()
    matched = coll.count_documents({"tags": {"$ne": []}})
    empty = total - matched

    # distinct() reads only the index metadata (not full docs); for
    # ~40k cards this is a single tiny request. Filter out None in case
    # some stale card lacks oracle_id.
    cards_oracle_ids = {oid for oid in coll.distinct("oracle_id") if oid}
    tagger_oracle_ids = set(oracle_id_to_slugs.keys())
    unmatched = len(tagger_oracle_ids - cards_oracle_ids)

    return matched, empty, unmatched


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download the Scryfall oracle_tags bulk and attach tags to the cards collection.",
    )
    parser.add_argument(
        "--refresh", action="store_true",
        help="Redownload even if the stored snapshot already matches the current one.",
    )
    args = parser.parse_args()

    # Make sure the indexes we rely on exist. ensure_indexes is idempotent,
    # so a brand-new deployment and a long-running cluster hit the same
    # fast path.
    storage.ensure_indexes()
    cards_coll = storage.cards_collection()
    tags_coll = storage.tags_collection()

    # Cards collection is required: tag attachment can't do its job
    # without cards to attach to.
    if cards_coll.estimated_document_count() == 0:
        parser.error(
            f"cards collection '{cards_coll.database.name}.{cards_coll.name}' is empty\n"
            "run scryfall-fetch first to populate it"
        )

    # Step 1: metadata. One tiny JSON request so we can compare timestamps
    # before committing to the 6 MB download.
    meta = get_bulk_oracle_tags_metadata()
    snapshot_updated_at = meta["updated_at"]
    stored_snapshot = storage.get_snapshot_timestamp(META_SOURCE)
    if stored_snapshot == snapshot_updated_at and not args.refresh:
        print(f"tags  : already fresh (snapshot {snapshot_updated_at}); pass --refresh to force")
        # The raw taggings aren't retained after import, so the attach
        # step can't be rerun from state alone. If the user added new
        # cards since the last tag import, --refresh is the way to pick
        # them up.
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

    # Step 3: push the catalog to Mongo.
    upsert_tag_catalog(tags_coll, catalog)
    print(f"saved  : {tags_coll.database.name}.{tags_coll.name}")

    # Step 4: attach tags to cards.
    matched, empty, unmatched = attach_tags_to_cards(cards_coll, oracle_id_to_slugs)
    total_cards = matched + empty
    print(
        f"cards  : {matched}/{total_cards} tagged, {empty} without tags, "
        f"{unmatched} tagger oracle_ids didn't match any cached card"
    )

    # Step 5: record the snapshot timestamp so a future run can skip
    # the download when nothing has moved. Done last so a failure
    # anywhere above forces a retry on the next invocation.
    storage.set_snapshot_timestamp(META_SOURCE, snapshot_updated_at)

    if total_cards and matched == 0:
        # Loud failure mode worth calling out: the cards collection has
        # no oracle_ids at all, which usually means it was populated
        # before oracle_id was added to the projection. Point at the
        # obvious remediation.
        print(
            "warning: no cards were tagged. The cards collection may predate the "
            "oracle_id field — run `scryfall-fetch --refresh ...` to repopulate it.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    # sys.exit propagates the int return code so shells / CI can react.
    sys.exit(main())
