"""Fetch Scryfall oracle text into MongoDB (bulk-only, diff-aware).

Downloads Scryfall's `oracle_cards` bulk file — a single gzip-compressed
JSON Lines dump of every unique card — and merges only what actually
changed into the cards collection.

Per-card diff via `oracle_text_sha`:

    Each card doc carries `oracle_text_sha` — the first 16 hex chars of
    sha256(oracle_text). On a bulk merge, incoming cards are classified
    as new / changed / unchanged by comparing the fresh sha against the
    stored one. Unchanged cards get no write at all. Changed cards get
    the owned fields re-set AND `text_embedding` + `card_vector`
    `$unset` so downstream `mtg-embed cards` / `mtg-embed fuse` runs
    re-encode them automatically (their default "skip if field exists"
    path then rebuilds exactly what was invalidated — no --refresh
    needed).

    On a typical Scryfall snapshot bump only a handful of oracle texts
    revise, so the merge becomes a tiny bulk_write rather than ~40k
    pointless $set operations.

Storage (MongoDB — see `storage.py` for connection details):

    cards collection — one document per card, _id = scryfall_id:

        {
          "_id":              "<scryfall_id>",       # Mongo primary key;
                                                     # consumers that want
                                                     # a semantic field name
                                                     # rename at export time
                                                     # (extract_oracle does).
          "oracle_id":        "<join key for oracle_tags>",
          "name":             "...",
          "names":            ["lightning bolt", ...],  # lowered aliases
          "mana_cost":        "...",
          "type_line":        "...",
          "oracle_text":      "...",
          "oracle_text_sha":  "<16-hex sha256 prefix>", # the diff key
          "keywords":         ["Flying", ...],       # Scryfall keywords array
          "colors":           ["W", "U", ...],       # face colors (WUBRG)
          "color_identity":   ["W", "U", ...],       # Commander-rules legality
          "cmc":              <float>,               # converted mana value
          "power":            "<str>" | None,        # creatures only
          "toughness":        "<str>" | None,        # creatures only
        }

    Per-card `updated_at` is not stored — the snapshot timestamp lives
    once in the `meta` collection under `_id = "oracle_cards"` (every
    card in a given merge shares the same value, so repeating it on
    40 k docs was pure overhead).

    `tags`, `text_embedding`, and `card_vector` are owned by other
    modules; the fetcher uses $set on only its own fields and `$unset`
    on the two downstream embedding fields when a card's oracle_text
    changed. An unchanged card keeps every pre-existing field intact.

Why id-primary + per-doc names array:

    Scryfall IDs are the stable identity for a card — a name can be
    changed by errata, but the ID doesn't move. Keying by ID means the
    card is stored exactly once even when it has multiple names
    (double-faced, split, adventure, modal DFC). The `names` array on
    each doc, backed by a multikey index, bridges the user-facing
    lookup ("Lightning Bolt") to the id-keyed store without a separate
    alias collection. The lookup helper lives here so downstream modules
    (`extract_oracle`, `explore`) can reuse it.

Download triggers:

    1. Empty cards collection, OR
    2. The `meta` collection's stored snapshot timestamp doesn't match
       the current `/bulk-data` metadata (snapshot moved since last run).

    Otherwise skip the ~24 MB download: one `/bulk-data` call + one
    `meta` read and we're done. This is the common `just populate`
    rerun path. There is no `--refresh` flag — the diff detects every
    legitimate reason to re-encode a card. To force a full rebuild (for
    corruption recovery), drop the cards collection and rerun.

Usage (after `pip install -e .`, with MONGODB_URI set in .env):

    scryfall-fetch                 # idempotent: no-op if snapshot already covered

Equivalently from a source checkout without installing:
    python -m mtg_recommender.scryfall_fetch
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import sys
import urllib.request
from pathlib import Path

from pymongo import UpdateOne
from pymongo.collection import Collection

from . import storage

# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------

# Base URL for every Scryfall API request. HTTPS is required; Python's default
# SSL context on modern systems negotiates TLS 1.2 or 1.3 automatically.
SCRYFALL_BASE = "https://api.scryfall.com"

# Key the oracle_cards snapshot timestamp lives under in the `meta` collection.
# Namespaced parallel to oracle_tags' entry so one meta collection serves both.
META_SOURCE = "oracle_cards"

# Scryfall requires every request to include a User-Agent (identifying the
# caller) and an Accept header (declaring the media type we want back). If
# either is missing, the API will reject the request.
HEADERS = {
    "User-Agent": "MTG-Recommender/0.1 (github.com/MangoPicante/MTG-Recommender)",
    "Accept": "application/json",
}

# ---------------------------------------------------------------------------
# Card projection
# ---------------------------------------------------------------------------

def oracle_text_sha(oracle_text: str | None) -> str:
    """Short content signature of the stored oracle_text.

    First 16 hex chars of sha256 — 64 bits is overkill for distinguishing
    ~40 k cards (birthday-collision probability on 40 k docs is ~4e-11).
    The input is normalised to the empty string when None so a card
    without oracle text still gets a stable sha rather than crashing.
    """
    return hashlib.sha256((oracle_text or "").encode("utf-8")).hexdigest()[:16]


def build_names(raw: dict) -> list[str]:
    """Return the lowered-name array a card doc gets stored with.

    The names array powers alias lookup (`cards.find({"names": lowered})`).
    It contains the lowered form of the card's main `name` plus one entry
    per face name on multi-faced cards (transform DFC, MDFC, split,
    adventure, meld). Duplicates are collapsed because art-card variants
    sometimes repeat the same face name twice (e.g. the "Delver of
    Secrets // Delver of Secrets" art card contributes a single
    "delver of secrets" entry rather than two).

    Order is preserved by dict insertion, which keeps the stored array
    stable across runs — the combined name always comes first, then
    face names left-to-right. Stability matters for diff-friendly
    document comparisons (useful when inspecting records by hand).
    """
    seen: dict[str, None] = {}
    name = raw.get("name")
    if name:
        seen[name.lower()] = None
    for face in raw.get("card_faces", []) or []:
        face_name = face.get("name")
        if face_name:
            seen[face_name.lower()] = None
    return list(seen.keys())


def extract_card_fields(data: dict) -> dict:
    """Reduce a full Scryfall card object into the Mongo document shape.

    Pure projection — no hidden time dependency, no I/O. The snapshot
    timestamp lives once in the `meta` collection; stamping it on
    every card doc was pure overhead (identical value across ~40 k
    rows) and no longer written.

    Multi-faced card handling (transform DFC, MDFC, split, adventure,
    meld) — Scryfall doesn't populate the same top-level fields for every
    layout:

        - Transform DFCs (e.g. Delver of Secrets): top-level `mana_cost`
          is the front face's cost, top-level `type_line` is combined
          ("Creature — Human Wizard // Creature — Human Insect").
        - Modal DFCs (e.g. Bala Ged Recovery // Bala Ged Sanctuary):
          top-level `mana_cost` is null, top-level `oracle_text` is null,
          top-level `type_line` is combined ("Sorcery // Land"). All the
          real per-face data lives under `card_faces[*]`.

    So for `mana_cost` and `type_line` we prefer the top-level value when
    it's populated (truthy) and fall back to a per-face string joined
    with " // " otherwise — the same convention Scryfall uses for
    combined card names ("Bedroom // Livingroom", "Bala Ged Recovery //
    Bala Ged Sanctuary"). Single-faced cards always take the top-level
    value. `oracle_text` keeps its `\\n---\\n`-joined form because
    downstream text processing wants a single searchable blob and the
    clearer face boundary helps NLP tokenisers.

    The returned dict uses `_id = scryfall_id` so it can be upserted
    into Mongo directly. We don't duplicate the id into a separate
    `scryfall_id` field — the storage saving (~37 bytes × 40 k cards ≈
    1.5 MB) is small but the duplicate was pure noise. Consumers that
    want a semantic field name (e.g. `extract_oracle`'s JSON output)
    rename `_id → scryfall_id` at export time. `oracle_id` names the
    ORACLE entity (the gameplay card) rather than a specific printing;
    the oracle_tags import joins on it.
    """
    faces = data.get("card_faces") or []

    def pick(field: str):
        """Top-level value if populated, else per-face values joined with " // ".

        Single-faced cards (no `card_faces`) always take the top-level
        value verbatim — including None / empty string, which faithfully
        reflects Scryfall's own representation. When falling back to
        per-face values, None / missing entries are normalised to empty
        strings so the joined output stays a valid string (e.g. an MDFC
        with a land back face becomes "{2}{G} // ").
        """
        top = data.get(field)
        if top or not faces:
            return top
        return " // ".join(f.get(field) or "" for f in faces)

    oracle_text = data.get("oracle_text")
    if oracle_text is None and faces:
        # Join each face's oracle text with a visible separator so the
        # combined string preserves face boundaries for downstream code.
        oracle_text = "\n---\n".join(
            face.get("oracle_text", "") for face in faces
        )

    return {
        # `_id` is the Mongo primary key AND the scryfall id. No
        # separate `scryfall_id` field — downstream code that wants
        # the semantic name reads `_id` or renames at export.
        "_id": data.get("id"),
        # Join key for the oracle_tags bulk. Falls back to None if the
        # raw object omits it (shouldn't happen for real Scryfall
        # responses but is defended against so an odd test fixture
        # doesn't crash the fetcher).
        "oracle_id": data.get("oracle_id"),
        "name": data.get("name"),
        "names": build_names(data),
        "mana_cost": pick("mana_cost"),
        "type_line": pick("type_line"),
        "oracle_text": oracle_text,
        # Content signature of the exact oracle_text we store (not the
        # raw Scryfall field, since multi-face layouts join faces here).
        # bulk_upsert_cards compares this against the stored sha to
        # decide which docs actually need re-writing and which embedding
        # fields to invalidate.
        "oracle_text_sha": oracle_text_sha(oracle_text),
        # Scryfall-provided structural fields used by Phase 3 cluster
        # mode as weighted features. Stored verbatim — downstream code
        # does its own parsing of mana_cost + type_line and doesn't
        # need Scryfall's redundant representations beyond what's here.
        #
        # Nulls: non-creatures carry power/toughness = None; a card
        # without a mana_cost still has cmc=0.0. Keywords and colors
        # arrays are always present (empty list if Scryfall doesn't
        # populate them), so downstream code can rely on the key
        # existing even when the content is empty.
        "keywords": list(data.get("keywords") or []),
        "colors": list(data.get("colors") or []),
        "color_identity": list(data.get("color_identity") or []),
        "cmc": float(data.get("cmc") or 0.0),
        "power": data.get("power"),
        "toughness": data.get("toughness"),
    }


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def http_get_json(url: str, timeout: int = 30) -> dict:
    """GET a URL and parse the response as JSON. Applies the required headers."""
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


# ---------------------------------------------------------------------------
# Mongo storage ops
# ---------------------------------------------------------------------------
#
# Every mutating op targets only the fields THIS module owns and uses $set
# (not replace) so an existing card's `tags` array — written by the
# oracle_tags module — survives a bulk merge. The fields owned here are:
#
#     _id, oracle_id, name, names, mana_cost, type_line, oracle_text,
#     oracle_text_sha, keywords, colors, color_identity, cmc, power,
#     toughness
#
# A content-change merge also $unsets `text_embedding` and `card_vector`
# — not owned here, but conceptually downstream of oracle_text, so a
# fresh text invalidates both. See bulk_upsert_cards.

# Field keys scryfall_fetch sets on an upsert. `_id` is handled by the
# filter clause (Mongo refuses to $set the primary key), so it's absent
# from this list even though it's an owned field.
_OWNED_FIELDS = (
    "oracle_id", "name", "names",
    "mana_cost", "type_line", "oracle_text", "oracle_text_sha",
    "keywords", "colors", "color_identity",
    "cmc", "power", "toughness",
)

# Downstream fields the fetcher $unsets when a card's oracle_text changes.
# Clearing them triggers re-encode / re-fuse on the next `mtg-embed` run
# via that CLI's default "skip when field exists" path. Listed as a
# constant so the invalidation surface is explicit.
_DOWNSTREAM_INVALIDATED = ("text_embedding", "card_vector")


def _set_payload(doc: dict) -> dict:
    """Project a full card doc down to the $set payload for an upsert.

    _id is handled by the filter clause, not $set — Mongo refuses to
    $set the primary key. Everything else owned by this module is
    included; `tags` is deliberately absent.
    """
    return {k: doc[k] for k in _OWNED_FIELDS if k in doc}


def upsert_card(coll: Collection, raw: dict) -> bool:
    """Upsert a single card into Mongo. Returns True on success, False on no-id.

    "Success" here just means the raw object had an `id` field to key the
    document under — a missing `id` would silently vanish without a
    place to go, so we skip and tell the caller.
    """
    doc = extract_card_fields(raw)
    sid = doc["_id"]
    if not sid:
        return False
    coll.update_one({"_id": sid}, {"$set": _set_payload(doc)}, upsert=True)
    return True


def bulk_upsert_cards(
    coll: Collection, bulk: list[dict]
) -> tuple[int, int, int, int]:
    """Diff-aware upsert. Returns (new, changed, patched, unchanged).

    Loads existing docs' sha AND owned-field presence into memory first
    (one Mongo projection scan of ~40k tiny docs), then classifies each
    incoming card:

      - new (no stored doc): $set the owned fields with upsert=True;
        nothing downstream to invalidate yet.
      - changed (stored sha differs from incoming): $set the owned
        fields AND $unset `text_embedding` + `card_vector` so the next
        `mtg-embed` run re-encodes only the cards that actually need it.
      - patched (sha matches but at least one newly-owned field is
        missing from the stored doc): $set the owned fields WITHOUT
        $unset. The content is the same — just the schema is older
        than this module expects. Keeps embeddings valid across a
        schema add (the whole point of this case).
      - unchanged (sha matches, every owned field present): no write.

    `bulk_write(ordered=False)` lets individual failed ops not stop the
    rest. A card without an `id` in the raw dict is skipped (same
    semantics as the single-doc path).

    Why return a 4-tuple instead of just a count: the CLI reports the
    breakdown to the user, and tests assert on it so a regression that
    secretly rewrites every card (or quietly misses the schema patch)
    is caught.
    """
    # Project every owned field so we can detect schema drift (field
    # missing from the stored doc even though sha matches). The index
    # on _id keeps the server-side scan cheap; the extra bytes per
    # doc are modest.
    projection: dict[str, int] = {"_id": 1}
    for field in _OWNED_FIELDS:
        projection[field] = 1
    existing: dict[str, dict] = {
        doc["_id"]: doc for doc in coll.find({}, projection)
    }

    new_count = changed_count = patched_count = unchanged_count = 0
    ops: list[UpdateOne] = []
    for raw in bulk:
        doc = extract_card_fields(raw)
        sid = doc["_id"]
        if not sid:
            continue
        stored = existing.get(sid)
        if stored is None:
            # New card — nothing downstream to invalidate.
            ops.append(UpdateOne({"_id": sid}, {"$set": _set_payload(doc)}, upsert=True))
            new_count += 1
            continue
        new_sha = doc["oracle_text_sha"]
        stored_sha = stored.get("oracle_text_sha")
        if stored_sha != new_sha:
            # Changed card — clear the two downstream fields so the
            # next `mtg-embed cards` / `mtg-embed fuse` re-encodes it.
            ops.append(
                UpdateOne(
                    {"_id": sid},
                    {
                        "$set": _set_payload(doc),
                        "$unset": {field: "" for field in _DOWNSTREAM_INVALIDATED},
                    },
                )
            )
            changed_count += 1
            continue
        # sha matches. Check for schema drift: any owned field that
        # the stored doc is missing. `None` as a stored value is NOT
        # drift — it's a legitimately-null field (e.g. non-creature
        # power/toughness); only the key being absent counts.
        missing_fields = [f for f in _OWNED_FIELDS if f not in stored]
        if missing_fields:
            # Patch rewrite — same content, newer schema. Do NOT $unset
            # the embedding fields; the sha hasn't changed, so the
            # cached text_embedding + card_vector are still valid.
            ops.append(
                UpdateOne({"_id": sid}, {"$set": _set_payload(doc)})
            )
            patched_count += 1
        else:
            unchanged_count += 1

    if ops:
        coll.bulk_write(ops, ordered=False)
    return new_count, changed_count, patched_count, unchanged_count


def find_cards_by_name(coll: Collection, name: str) -> list[dict]:
    """Return every card whose `names` array contains the lowered input.

    Mongo returns a cursor; this helper materialises it to a list because
    a single name may legitimately match more than one card (art-card
    variants, meld pieces, cards named after their faces). The multikey
    index on `names` makes this O(log n) despite the array semantics.

    Callers who only want one card apply their own picking logic — this
    function stays neutral and returns every matching doc.
    """
    return list(coll.find({"names": name.lower()}))


# ---------------------------------------------------------------------------
# Bulk-data download & merge
# ---------------------------------------------------------------------------

def get_bulk_oracle_metadata() -> dict:
    """Ask Scryfall for the metadata of the current oracle_cards bulk dump.

    The /bulk-data endpoint lists every available bulk file (oracle_cards,
    default_cards, all_cards, etc.). We only care about oracle_cards, which
    contains exactly one entry per unique card (no reprint duplicates).
    """
    data = http_get_json(f"{SCRYFALL_BASE}/bulk-data")
    for entry in data.get("data", []):
        if entry.get("type") == "oracle_cards":
            return entry
    raise RuntimeError("oracle_cards entry not found in /bulk-data response")


def download_bulk_oracle_cards(meta: dict) -> list[dict]:
    """Download the oracle_cards bulk file into memory and return raw card dicts.

    We deliberately do NOT persist the raw JSONL to disk — the caller merges
    every card into the unified cache (stamped with `meta["updated_at"]`),
    and per-card timestamps are enough to know whether we already hold
    this snapshot.
    """
    download_uri = meta["jsonl_download_uri"]
    # `compressed_size` is the size of the .gz payload in bytes. We report
    # it in MB so the user has a sense of what's downloading.
    size_mb = meta.get("compressed_size", 0) / 1_000_000
    print(f"bulk  : downloading oracle_cards (~{size_mb:.0f} MB gz, snapshot {meta['updated_at']})")

    req = urllib.request.Request(download_uri, headers=HEADERS)
    # 5-minute timeout is generous for ~24 MB on a slow connection.
    with urllib.request.urlopen(req, timeout=300) as resp:
        raw = resp.read()

    # Detect gzip via the 0x1f 0x8b magic bytes and decompress. urllib does
    # not auto-decompress unless we set Accept-Encoding, so we handle it
    # explicitly. Also covers the case where a proxy already decoded it.
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)

    # The file is JSON Lines: one full card object per line. Parse each line
    # separately rather than loading a single giant JSON array.
    cards: list[dict] = []
    for line in raw.decode("utf-8").splitlines():
        line = line.strip()
        if line:
            cards.append(json.loads(line))
    return cards


# ---------------------------------------------------------------------------
# Input parsing
# ---------------------------------------------------------------------------

# Strips a leading sideboard marker like "SB: " that some deck export
# formats use to separate sideboard cards from the main deck.
_DECKLIST_SIDEBOARD_RE = re.compile(r"^\s*SB:\s*", re.IGNORECASE)
# Strips a leading copy-count prefix like "1 " or "4x " that the standard
# MTG decklist format uses. `x` after the digit(s) is optional and case-
# insensitive; trailing whitespace after the count is required so we
# don't clip the first word of a card name that happens to start with a
# digit (which shouldn't exist for real cards, but the guard is cheap).
_DECKLIST_LEADING_QTY_RE = re.compile(r"^\s*\d+x?\s+", re.IGNORECASE)
# Strips a trailing set+collector suffix like " (STA) 42" or " (DOM) 123"
# that many deck exporters append. The collector number is optional
# (Moxfield often omits it, e.g. "Lightning Bolt (STA)").
_DECKLIST_TRAILING_SET_RE = re.compile(r"\s*\([^)]+\)(?:\s+\S+)?\s*$")


def parse_decklist_line(line: str) -> str | None:
    """Extract a card name from a single decklist line, or return None to skip.

    Handles the common formats we see from Moxfield, Archidekt, MTGGoldfish,
    Arena exports, and hand-rolled text files:

        "Card Name"                        # bare
        "1 Card Name"                      # basic MTG decklist
        "4x Card Name"                     # alt qty syntax
        "1 Card Name (STA) 42"             # with set + collector number
        "1 Card Name (STA)"                # with set only
        "SB: 1 Card Name"                  # sideboard prefix

    Also skips blank lines and comment lines starting with '#'. Double-
    faced card names using the "Front // Back" convention pass through
    unchanged — the leading-qty strip is anchored to the start of the
    line, and `//` never appears inside the trailing set-parens block.
    """
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    line = _DECKLIST_SIDEBOARD_RE.sub("", line)
    line = _DECKLIST_LEADING_QTY_RE.sub("", line)
    line = _DECKLIST_TRAILING_SET_RE.sub("", line)
    line = line.strip()
    return line or None


def read_names(args: argparse.Namespace) -> list[str]:
    """Collect card names from the --file input and CLI positional args.

    Every line/arg is fed through `parse_decklist_line`, so decklist-style
    inputs ("1 Lightning Bolt (STA) 42") are normalised to bare names.
    Blank lines and '#' comments are dropped. Names are de-duplicated
    case-insensitively while preserving the order in which they were
    first seen (helpful for readable progress output).
    """
    raw: list[str] = []
    if args.file:
        for line in Path(args.file).read_text(encoding="utf-8").splitlines():
            raw.append(line)
    raw.extend(args.cards)

    seen: set[str] = set()
    out: list[str] = []
    for line in raw:
        name = parse_decklist_line(line)
        if name is None:
            continue
        key = name.lower()
        if key not in seen:
            seen.add(key)
            out.append(name)
    return out


# ---------------------------------------------------------------------------
# Mode implementations
# ---------------------------------------------------------------------------

def run_bulk_mode(coll: Collection) -> bool:
    """Ensure Mongo holds the current Scryfall oracle_cards snapshot.

    Fast-path (the common `just populate` rerun): if `meta` records this
    exact snapshot and the cache is non-empty, skip without reading any
    card docs. One `/bulk-data` call + one `meta` read.

    Download-path: either `meta` is unset (fresh cluster), the stored
    snapshot timestamp doesn't match the current one, or the collection
    is empty. `bulk_upsert_cards` then diffs each incoming card against
    its stored `oracle_text_sha` — unchanged cards get no write;
    content-changed cards get re-set plus `$unset` on `text_embedding`
    and `card_vector` so the next `mtg-embed` run picks up the real
    delta instead of either re-encoding the world or missing the staleness.

    Returns True iff the collection changed, so `main` can print the
    final count only when it's useful. There is no `--refresh` flag —
    the diff is the source of truth. To force a full rebuild, drop the
    cards collection and rerun.
    """
    meta = get_bulk_oracle_metadata()
    snapshot_updated_at = meta["updated_at"]

    cards_present = coll.estimated_document_count() > 0
    stored_snapshot = storage.get_meta_value(META_SOURCE)

    # Fast path: meta says we're already covering this snapshot.
    if cards_present and stored_snapshot == snapshot_updated_at:
        print(f"bulk  : already fresh (snapshot {snapshot_updated_at})")
        return False

    bulk = download_bulk_oracle_cards(meta)
    new_count, changed_count, patched_count, unchanged_count = bulk_upsert_cards(
        coll, bulk
    )
    total = new_count + changed_count + patched_count + unchanged_count
    print(
        f"bulk  : {new_count} new, {changed_count} changed, "
        f"{patched_count} patched, {unchanged_count} unchanged "
        f"(of {total} in snapshot)"
    )
    # Record the snapshot last so a crash mid-merge leaves meta un-set and
    # the next run retries rather than falsely claiming freshness.
    storage.set_meta_value(META_SOURCE, snapshot_updated_at)
    # "Did we touch Mongo?" — new/changed are obvious; patched also
    # writes (schema migration), so flip the summary message to "saved…"
    # for the user instead of "no changes" even though no embeddings
    # got invalidated.
    return new_count > 0 or changed_count > 0 or patched_count > 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Download Scryfall's oracle_cards bulk and merge it into MongoDB. "
            "Idempotent and diff-aware — rerun freely; only new/changed cards "
            "produce writes, and content-changed cards invalidate stale "
            "embeddings so `mtg-embed` picks them up automatically."
        )
    )
    parser.parse_args()  # no flags; parse to surface -h/--help cleanly

    # Ensure indexes before any read/write — idempotent on Mongo's side,
    # so the cost is one trip per CLI run and brand-new deployments work
    # without a separate migration step.
    storage.ensure_indexes()
    coll = storage.cards_collection()

    changed = run_bulk_mode(coll)
    if changed:
        total = coll.estimated_document_count()
        print(f"\nsaved {total} unique cards to MongoDB ({coll.database.name}.{coll.name})")
    else:
        print("\nno changes")
    return 0


if __name__ == "__main__":
    # sys.exit propagates the int return code so shells / CI can react.
    sys.exit(main())
