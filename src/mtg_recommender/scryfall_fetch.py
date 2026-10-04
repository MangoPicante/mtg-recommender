"""Fetch Scryfall oracle text into MongoDB.

The mode is chosen automatically based on how many cards were requested:

    1 card   : One HTTPS request via /cards/named. Fast, no bulk download.
               The resulting entry is stamped with the current UTC time.

    2+ cards : Downloads Scryfall's oracle_cards bulk file (a single
               gzip-compressed JSON Lines dump of every unique card), then
               upserts every card from that dump into the cards collection.
               Every upserted entry is stamped with the snapshot's own
               `updated_at` timestamp (the value Scryfall gives us for that
               dump), NOT the current wall-clock time.

Storage (MongoDB — see `storage.py` for connection details):

    cards collection — one document per card, _id = scryfall_id:

        {
          "_id":        "<scryfall_id>",       # Mongo primary key
          "scryfall_id":"<same as _id>",
          "oracle_id":  "<Scryfall oracle id — the join key for tags>",
          "name":       "...",
          "names":      ["lightning bolt", "lightning bolt // lightning bolt"],
                                               # lowered aliases on this doc;
                                               # a multikey index makes
                                               # `find({"names": lowered})`
                                               # fast and ambiguity-tolerant.
          "mana_cost":  "...",
          "type_line":  "...",
          "oracle_text":"...",
          "updated_at": "<UTC ISO 8601>"
        }

    `tags` is owned by the oracle_tags module and is deliberately NOT set
    by this module — upserts use $set on the fields the fetcher owns so
    an existing card's tags survive a bulk refresh.

Why id-primary + per-doc names array:

    Scryfall IDs are the stable identity for a card — a name can be
    changed by errata, but the ID doesn't move. Keying by ID means the
    card is stored exactly once even when it has multiple names
    (double-faced, split, adventure, modal DFC). The `names` array on
    each doc, backed by a multikey index, bridges the user-facing
    lookup ("Lightning Bolt") to the id-keyed store without a separate
    alias collection.

    Multiple cards may share a lowered name (art-card variants, meld
    pieces, cards named after their faces); `find({"names": lowered})`
    returns a cursor of every matching doc, and the caller decides how
    to disambiguate.

Per-card `updated_at` semantics:

    - Bulk merge  : set to the snapshot's `updated_at` (identical for
                    every card in a given merge).
    - Single fetch: set to `datetime.now(timezone.utc)` at fetch time.
    - Multiple bulk merges over time or a mix of bulk + single fetches
      leave each card with the correct time it was last refreshed from
      Scryfall.

Bulk-download triggers (any one is enough):

    1. --refresh flag set.
    2. Empty cards collection.
    3. Any requested card isn't in the collection yet — the whole point of
       bulk mode is to serve the request, so if we're missing something,
       download and try to satisfy it.
    4. Any cached card's `updated_at` is older than the current bulk
       snapshot's `updated_at` — merging refreshes that entry.

    If none apply, skip the ~24 MB download.

Usage (after `pip install -e .`, with MONGODB_URI set in .env):

    scryfall-fetch "Lightning Bolt"                    # 1 card  -> /cards/named
    scryfall-fetch "Lightning Bolt" "Counterspell"     # 2+ cards -> bulk
    scryfall-fetch --file cards.txt                    # from a file
    scryfall-fetch --file cards.txt --refresh          # force refetch/redownload

Equivalently from a source checkout without installing:
    python -m mtg_recommender.scryfall_fetch "Lightning Bolt"
"""
from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

from pymongo import UpdateOne
from pymongo.collection import Collection

from . import storage

# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------

# Base URL for every Scryfall API request. HTTPS is required; Python's default
# SSL context on modern systems negotiates TLS 1.2 or 1.3 automatically.
SCRYFALL_BASE = "https://api.scryfall.com"

# Scryfall requires every request to include a User-Agent (identifying the
# caller) and an Accept header (declaring the media type we want back). If
# either is missing, the API will reject the request.
HEADERS = {
    "User-Agent": "MTG-Recommender/0.1 (github.com/MangoPicante/MTG-Recommender)",
    "Accept": "application/json",
}

# ---------------------------------------------------------------------------
# Time helper
# ---------------------------------------------------------------------------

def now_utc_iso() -> str:
    """ISO 8601 timestamp for 'right now' in UTC (e.g. '2026-07-31T12:34:56.789+00:00').

    Kept as a tiny helper so every single-fetch stamps its cache entries in
    exactly the same format Scryfall uses for its bulk snapshot timestamps.
    """
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Card projection
# ---------------------------------------------------------------------------

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


def extract_card_fields(data: dict, updated_at: str) -> dict:
    """Reduce a full Scryfall card object into the Mongo document shape.

    `updated_at` is passed in explicitly so the caller decides what "last
    updated" means for this entry — the snapshot timestamp for bulk merges,
    the current wall-clock for single fetches. This keeps the projection
    function pure (no hidden time dependency) and lets both fetch paths
    produce identical-shaped documents.

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

    The returned dict uses `_id = scryfall_id` so it can be upserted into
    Mongo directly. `scryfall_id` is kept as a separate field for API
    symmetry — callers that iterate a cursor get the id without having
    to pop `_id`. `oracle_id` names the ORACLE entity (the gameplay
    card) rather than a specific printing; the oracle_tags import joins
    on it.
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

    scryfall_id = data.get("id")
    return {
        # `_id` is the Mongo primary key; keeping scryfall_id duplicated
        # as a top-level field means code reading documents doesn't need
        # to know about Mongo's reserved key naming.
        "_id": scryfall_id,
        "scryfall_id": scryfall_id,
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
        "updated_at": updated_at,
    }


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def http_get_json(url: str, timeout: int = 30) -> dict:
    """GET a URL and parse the response as JSON. Applies the required headers."""
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def fetch_card_raw_named(name: str) -> dict | None:
    """Fetch the raw Scryfall card object for `name` via /cards/named.

    Returns the raw response dict on success (unfiltered — the caller
    projects it with `extract_card_fields`), or None if Scryfall responds
    404 (unknown card). Any other HTTP error is re-raised for the caller
    to log and skip.
    """
    # quote() percent-encodes spaces and special characters so the URL is
    # valid even for names like "Jace, the Mind Sculptor".
    url = f"{SCRYFALL_BASE}/cards/named?exact={quote(name)}"
    try:
        return http_get_json(url, timeout=15)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            print(f"  not found: {name}", file=sys.stderr)
            return None
        # 429 (rate limited), 5xx (server errors), etc. bubble up to the
        # caller so they can be logged with the offending card name.
        raise


# ---------------------------------------------------------------------------
# Mongo storage ops
# ---------------------------------------------------------------------------
#
# Every mutating op targets only the fields THIS module owns and uses $set
# (not replace) so an existing card's `tags` array — written by the
# oracle_tags module — survives a bulk refresh. The fields owned here are:
#
#     _id, scryfall_id, oracle_id, name, names, mana_cost, type_line,
#     oracle_text, updated_at

# Field keys scryfall_fetch sets on an upsert. Kept as a module-level
# constant so the single-mode and bulk-mode paths stay in lockstep.
_OWNED_FIELDS = (
    "scryfall_id", "oracle_id", "name", "names",
    "mana_cost", "type_line", "oracle_text", "updated_at",
)


def _set_payload(doc: dict) -> dict:
    """Project a full card doc down to the $set payload for an upsert.

    _id is handled by the filter clause, not $set — Mongo refuses to
    $set the primary key. Everything else owned by this module is
    included; `tags` is deliberately absent.
    """
    return {k: doc[k] for k in _OWNED_FIELDS if k in doc}


def upsert_card(coll: Collection, raw: dict, updated_at: str) -> bool:
    """Upsert a single card into Mongo. Returns True on success, False on no-id.

    "Success" here just means the raw object had an `id` field to key the
    document under — a missing `id` would silently vanish without a
    place to go, so we skip and tell the caller.
    """
    doc = extract_card_fields(raw, updated_at=updated_at)
    sid = doc["_id"]
    if not sid:
        return False
    coll.update_one({"_id": sid}, {"$set": _set_payload(doc)}, upsert=True)
    return True


def bulk_upsert_cards(coll: Collection, bulk: list[dict], snapshot_updated_at: str) -> int:
    """Upsert every card from a bulk list in a single pymongo bulk_write call.

    `bulk_write` is massively faster than per-doc `update_one` for the
    ~40k oracle_cards dump — pymongo batches ops under the hood and
    sends them to the server in groups. `ordered=False` lets failures
    on individual ops not stop the rest; a card with no `id` is just
    skipped as it is in the single-doc path.

    Returns the count of ops queued (which equals the count of cards
    with a usable id — practically all of them for a real snapshot).
    """
    ops: list[UpdateOne] = []
    for raw in bulk:
        doc = extract_card_fields(raw, updated_at=snapshot_updated_at)
        sid = doc["_id"]
        if not sid:
            continue
        ops.append(UpdateOne({"_id": sid}, {"$set": _set_payload(doc)}, upsert=True))
    if ops:
        coll.bulk_write(ops, ordered=False)
    return len(ops)


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


def has_stale_cards(coll: Collection, snapshot_updated_at: str) -> bool:
    """Does the cards collection hold any entry older than the current snapshot?

    Timestamps are stored as ISO 8601 strings. ISO 8601 strings written
    with the same fractional-second precision are directly string-comparable,
    but Scryfall is not quite consistent about trailing zeros, so string
    compare of e.g. "...40.749+00:00" vs "...40.749000+00:00" would
    spuriously flag the former as older. We compare with
    `datetime.fromisoformat` on both sides by doing the scan client-side
    — one tiny read of `updated_at` plus `_id` per document is enough.

    Returns True on the first stale entry found; iteration stops there.
    Malformed or missing `updated_at` values are skipped (treated as
    "not stale") so a corrupted doc doesn't force a 24 MB redownload.
    """
    snapshot_dt = datetime.fromisoformat(snapshot_updated_at)
    # Project only the fields we need to keep the scan cheap. The server
    # streams one small subdocument per card rather than the full ~2 KB
    # payload.
    for entry in coll.find({}, {"updated_at": 1}):
        entry_ts = entry.get("updated_at")
        if not entry_ts:
            continue
        try:
            entry_dt = datetime.fromisoformat(entry_ts)
        except ValueError:
            continue
        if entry_dt < snapshot_dt:
            return True
    return False


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

def run_single_mode(name: str, coll: Collection, refresh: bool) -> bool:
    """Fetch a single card via /cards/named and upsert it into Mongo.

    Returns True if the collection changed (used by main() to decide
    the summary line). Cached entries are skipped unless --refresh is
    set. No rate limiting is needed because we only make one API call
    per invocation.
    """
    if not refresh and find_cards_by_name(coll, name):
        # Any non-empty match list counts as a cache hit for the CLI's
        # skip-fetch decision. Disambiguation across multiple matches is
        # the downstream consumer's problem.
        print(f"cached: {name}")
        return False

    print(f"fetch : {name}")
    try:
        raw = fetch_card_raw_named(name)
    except urllib.error.HTTPError as e:
        print(f"  http error {e.code}: {name}", file=sys.stderr)
        return False
    except urllib.error.URLError as e:
        print(f"  network error: {e.reason}", file=sys.stderr)
        return False

    if raw is None:
        return False
    # Real-time timestamp — this is when *we* pulled the card. It won't
    # coincide with a Scryfall bulk snapshot's timestamp, which is what
    # the freshness check in run_bulk_mode relies on.
    return upsert_card(coll, raw, now_utc_iso())


def run_bulk_mode(names: list[str], coll: Collection, refresh: bool) -> bool:
    """Ensure Mongo holds what we need from the current snapshot, then report.

    Downloads the bulk file when any of these apply:
      - --refresh forces it,
      - the cards collection is empty,
      - a requested name isn't in the collection (we clearly need it),
      - any card is older than the current snapshot (stale).
    Otherwise skips the ~24 MB download.

    After the merge (or skip), report each requested name as cached or
    not found. Missing names after a fresh download are real
    "not found"s — the card doesn't exist in the current Scryfall
    snapshot (misspelling, unreleased, or a token / meme card).
    """
    meta = get_bulk_oracle_metadata()
    snapshot_updated_at = meta["updated_at"]

    # Evaluate cheap checks first so we can short-circuit before the
    # potentially O(n) staleness scan. estimated_document_count() is a
    # metadata read on Mongo — cheaper than count_documents({}).
    cards_present = coll.estimated_document_count() > 0
    any_missing = any(not find_cards_by_name(coll, n) for n in names)
    needs_download = (
        refresh
        or not cards_present
        or any_missing
        or has_stale_cards(coll, snapshot_updated_at)
    )
    changed = False

    if needs_download:
        bulk = download_bulk_oracle_cards(meta)
        before = coll.estimated_document_count()
        bulk_upsert_cards(coll, bulk, snapshot_updated_at)
        changed = True
        after = coll.estimated_document_count()
        print(f"bulk  : collection now holds {after} unique cards (was {before})")
    else:
        print(f"bulk  : collection already covers snapshot {snapshot_updated_at}")

    for name in names:
        matches = find_cards_by_name(coll, name)
        if not matches:
            # After a fresh merge, a missing name is a real "not found" —
            # the card isn't in the current Scryfall snapshot.
            print(f"  not found: {name}", file=sys.stderr)
        elif len(matches) == 1:
            print(f"cached: {name}")
        else:
            # Surface the ambiguity so the user knows a downstream
            # consumer will have to pick between these ids.
            print(f"cached: {name} ({len(matches)} matches)")
            for m in matches:
                print(f"          - {m.get('name')} [{m.get('scryfall_id')}]")
    return changed


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch Scryfall oracle text into MongoDB.")
    parser.add_argument("cards", nargs="*", help="Card names (quote multi-word names).")
    parser.add_argument("-f", "--file",
                        help="Path to a file with one card name per line (# for comments).")
    parser.add_argument("--refresh", action="store_true",
                        help="Refetch even if already cached (single card) or force redownload the bulk snapshot.")
    args = parser.parse_args()

    names = read_names(args)
    if not names:
        # parser.error() prints usage and exits with code 2.
        parser.error("no card names provided (pass names as args or via --file)")

    # Ensure indexes before any read/write — idempotent on Mongo's side,
    # so the cost is one trip per CLI run and brand-new deployments work
    # without a separate migration step.
    storage.ensure_indexes()
    coll = storage.cards_collection()

    # Mode is chosen by count: a single card hits the API directly, anything
    # more falls to the bulk path (which upserts every card from the snapshot).
    if len(names) == 1:
        changed = run_single_mode(names[0], coll, args.refresh)
    else:
        changed = run_bulk_mode(names, coll, args.refresh)

    if changed:
        total = coll.estimated_document_count()
        print(f"\nsaved {total} unique cards to MongoDB ({coll.database.name}.{coll.name})")
    else:
        print("\nno changes")
    return 0


if __name__ == "__main__":
    # sys.exit propagates the int return code so shells / CI can react.
    sys.exit(main())
