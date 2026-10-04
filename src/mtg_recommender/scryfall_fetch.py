"""Fetch and cache Scryfall oracle texts for a list of cards.

The mode is chosen automatically based on how many cards were requested:

    1 card   : One HTTPS request via /cards/named. Fast, no bulk download.
               The resulting entry is stamped with the current UTC time.

    2+ cards : Downloads Scryfall's oracle_cards bulk file (a single
               gzip-compressed JSON Lines dump of every unique card), then
               merges every card from that dump into the shared cache.
               Every merged entry is stamped with the snapshot's own
               `updated_at` timestamp (the value Scryfall gives us for that
               dump), NOT the current wall-clock time.

Cache shape (cache/oracle_texts.json):

    {
      "cards": {
        "<scryfall_id>": {
          "name": "...",
          "mana_cost": "...",
          "type_line": "...",
          "oracle_text": "...",
          "scryfall_id": "<same as key>",
          "updated_at": "<UTC ISO 8601>"
        },
        ...
      },
      "aliases": {
        "<lowercased card name or face name>": ["<scryfall_id>", ...],
        ...
      }
    }

Why id-primary + list-valued aliases:

    Scryfall IDs are the stable identity for a card — a name can be
    changed by errata, but the ID doesn't move. Keying by ID means the
    card is stored exactly once even when it has multiple names (double-
    faced, split, adventure, modal DFC). The aliases index bridges the
    user-facing lookup ("Lightning Bolt") to the id-keyed store in two
    O(1) hops.

    Alias VALUES are lists so that name collisions surface to the caller
    instead of being silently resolved. For example, both the real
    "Delver of Secrets // Insectile Aberration" and a hypothetical art-
    card "Delver of Secrets // Delver of Secrets" contribute a face
    alias for "delver of secrets"; both ids end up in the list, and the
    downstream consumer picks based on whatever criterion matters to
    them (type line, canonical name shape, set code, etc.).

Per-card `updated_at` semantics:

    - Bulk merge  : set to the snapshot's `updated_at` (identical for
                    every card in a given merge).
    - Single fetch: set to `datetime.now(timezone.utc)` at fetch time.
    - Multiple bulk merges over time or a mix of bulk + single fetches
      leave each card with the correct time it was last refreshed from
      Scryfall.

Bulk-download triggers (any one is enough):

    1. --refresh flag set.
    2. Empty cache.
    3. Any requested card isn't in the cache yet — the whole point of
       bulk mode is to serve the request, so if we're missing something,
       download and try to satisfy it.
    4. Any cached card's `updated_at` is older than the current bulk
       snapshot's `updated_at` — merging refreshes that entry.

    If none apply, skip the 24 MB download.

The cache/ directory is gitignored so nothing here leaks into the repo.

Usage (after `pip install -e .`, which registers the `scryfall-fetch` script):
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

# All cached data lives in cache/, which .gitignore excludes from the repo.
#
# CWD-relative rather than module-relative because the module now lives inside
# an installable package (src/mtg_recommender/) — a module-relative path would
# either bury cache/ inside the package tree during development or vanish into
# site-packages when installed as a wheel. Users running the CLI from the
# project root (the common case) still get ./cache/ as expected; anything else
# can be overridden with --cache.
CACHE_DIR = Path.cwd() / "cache"
# Single unified cache: id-keyed card store plus name -> id alias index.
CACHE_PATH = CACHE_DIR / "oracle_texts.json"


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
# Cache I/O
# ---------------------------------------------------------------------------

def load_cache(path: Path) -> dict:
    """Load the unified cache. Returns a fresh shell if the file doesn't exist.

    The shell shape ({"cards": {}, "aliases": {}}) is what the rest of the
    script expects, so callers can always assume both keys are present.
    Any stray top-level fields left over from earlier schemas are silently
    ignored, and a cache saved in a previous name-keyed schema will look
    like it has "cards" but no "aliases" — the next bulk merge repopulates
    both correctly.
    """
    if not path.exists():
        return {"cards": {}, "aliases": {}}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        "cards": data.get("cards", {}),
        "aliases": data.get("aliases", {}),
    }


def save_cache(path: Path, cache: dict) -> None:
    """Write the unified cache back to disk (pretty-printed, deterministic order)."""
    # mkdir(parents=True) is safe if the directory already exists.
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        # sort_keys keeps diffs stable when the cache is inspected by hand.
        # ensure_ascii=False preserves Unicode symbols in oracle text
        # (mana symbols encoded as characters, curly quotes, etc.).
        json.dumps(cache, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Card projection
# ---------------------------------------------------------------------------

def extract_card_fields(data: dict, updated_at: str) -> dict:
    """Reduce a full Scryfall card object down to the fields we care about.

    `updated_at` is passed in explicitly so the caller decides what "last
    updated" means for this entry — the snapshot timestamp for bulk merges,
    the current wall-clock for single fetches. This keeps the projection
    function pure (no hidden time dependency) and lets both fetch paths
    produce identical-shaped entries.

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

    Note: `scryfall_id` is kept inside the value even though it's the key
    in the cache, so downstream code that reads a card value can identify
    it without needing to know which key it came from.
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
        "name": data.get("name"),
        "mana_cost": pick("mana_cost"),
        "type_line": pick("type_line"),
        "oracle_text": oracle_text,
        "scryfall_id": data.get("id"),
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
# Storage & alias management
# ---------------------------------------------------------------------------

def _add_alias(aliases: dict, name: str | None, card_id: str) -> None:
    """Append `card_id` to the alias list for `name` (lowercased), avoiding duplicates."""
    if not name:
        return
    ids = aliases.setdefault(name.lower(), [])
    if card_id not in ids:
        ids.append(card_id)


def register_aliases(aliases: dict, raw: dict, card_id: str) -> None:
    """Add `card_id` to the alias list for the card's full name and each face name.

    Every alias value is a list of ids. Multiple cards can share the same
    lowered name (art-card variants, meld pieces, etc.) — they all end up
    in the list, and the caller resolves the ambiguity at lookup time.
    Internal deduplication means calling this twice for the same card
    (e.g. across a re-merge) doesn't grow the list unbounded.
    """
    _add_alias(aliases, raw.get("name"), card_id)
    # `card_faces` may be missing, None, or a list; guard for all three.
    for face in raw.get("card_faces", []) or []:
        _add_alias(aliases, face.get("name"), card_id)


def store_card(cache: dict, raw: dict, updated_at: str, extra_aliases=()) -> bool:
    """Project `raw` and store it in the id-keyed card store, plus register aliases.

    Returns True on success, False if the raw response is missing a
    Scryfall id (which would leave us with no key to store it under —
    should be impossible in practice but we guard rather than crash).

    `extra_aliases` lets the caller register additional lowered-name
    aliases beyond what `register_aliases` derives from the raw object.
    Single-mode uses this to record whatever the user typed, in case
    their spelling differs from Scryfall's canonical name.
    """
    card_id = raw.get("id")
    if not card_id:
        return False
    cache["cards"][card_id] = extract_card_fields(raw, updated_at=updated_at)
    register_aliases(cache["aliases"], raw, card_id)
    for alias in extra_aliases:
        _add_alias(cache["aliases"], alias, card_id)
    return True


def resolve_by_name(cache: dict, name: str) -> list[dict]:
    """Two-step lookup: name -> [ids] -> [cards]. Returns [] if either step misses.

    Returns a list because a single name can legitimately map to more than
    one card (art-card variants, meld pieces, cards named after their
    faces). Callers who only want one card should apply their own picking
    logic (e.g. prefer entries whose `name` doesn't self-repeat, or match
    on a specific `type_line`); this function stays neutral and just
    returns every card that claims this name.
    """
    ids = cache["aliases"].get(name.lower(), [])
    return [cache["cards"][cid] for cid in ids if cid in cache["cards"]]


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


def merge_bulk_into_cache(bulk: list[dict], cache: dict, snapshot_updated_at: str) -> int:
    """Store every card in `bulk` under its scryfall_id and refresh aliases.

    Returns the count of cards successfully stored (i.e. those with a
    valid `id` — practically all of them). Every stored card is stamped
    with `snapshot_updated_at`.
    """
    merged = 0
    for raw in bulk:
        if store_card(cache, raw, snapshot_updated_at):
            merged += 1
    return merged


def has_stale_cards(cards: dict, snapshot_updated_at: str) -> bool:
    """Does the card store hold any entry older than the current bulk snapshot?

    Iterates the id-keyed store (each card exactly once — no double-
    counting via aliases). Returns True as soon as it finds an entry
    whose `updated_at` is earlier than `snapshot_updated_at`, meaning
    that entry could be refreshed by merging the current bulk file.

    Timestamps are parsed with `datetime.fromisoformat` rather than
    compared as raw strings. String compare of e.g. "...40.749+00:00"
    vs "...40.749000+00:00" would spuriously mark the former as older
    even though they're the same instant; parsing normalises both sides.

    Malformed or missing `updated_at` values are treated as "not stale"
    (skipped) — a corrupted entry shouldn't force a 24 MB redownload.
    """
    snapshot_dt = datetime.fromisoformat(snapshot_updated_at)
    for entry in cards.values():
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

def run_single_mode(name: str, cache: dict, refresh: bool) -> bool:
    """Fetch a single card via /cards/named and store it in the shared cache.

    Returns True if the cache changed (used by main() to decide whether to
    save). Cached entries are skipped unless --refresh is set. No rate
    limiting is needed because we only make one API call per invocation.

    On store, we also register the user's typed name as an alias — Scryfall's
    exact-match resolver can accept slight variations, so the canonical
    response name might not match what the user typed verbatim, and we want
    the next lookup for that exact spelling to still hit the cache.
    """
    if not refresh and resolve_by_name(cache, name):
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
    return store_card(cache, raw, now_utc_iso(), extra_aliases=[name])


def run_bulk_mode(names: list[str], cache: dict, refresh: bool) -> bool:
    """Ensure the cache holds what we need from the current snapshot, then report.

    Downloads the bulk file when any of these apply:
      - --refresh forces it,
      - the cache is empty,
      - a requested name isn't in the cache (we clearly need it),
      - any cached entry is older than the current snapshot (stale).
    Otherwise skips the 24 MB download.

    After the merge (or skip), report each requested name as cached or not
    found. Missing names after a fresh download are real "not found"s —
    the card doesn't exist in the current Scryfall snapshot (misspelling,
    unreleased, or a token / meme card).
    """
    meta = get_bulk_oracle_metadata()
    snapshot_updated_at = meta["updated_at"]

    # Evaluate cheap checks first so we can short-circuit before the
    # potentially O(n) staleness scan.
    any_missing = any(not resolve_by_name(cache, n) for n in names)
    needs_download = (
        refresh
        or not cache["cards"]
        or any_missing
        or has_stale_cards(cache["cards"], snapshot_updated_at)
    )
    changed = False

    if needs_download:
        bulk = download_bulk_oracle_cards(meta)
        before = len(cache["cards"])
        merge_bulk_into_cache(bulk, cache, snapshot_updated_at)
        changed = True
        print(f"bulk  : cache now holds {len(cache['cards'])} unique cards (was {before})")
    else:
        print(f"bulk  : cache already covers snapshot {snapshot_updated_at}")

    for name in names:
        matches = resolve_by_name(cache, name)
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
    parser = argparse.ArgumentParser(description="Fetch and cache Scryfall oracle text.")
    parser.add_argument("cards", nargs="*", help="Card names (quote multi-word names).")
    parser.add_argument("-f", "--file",
                        help="Path to a file with one card name per line (# for comments).")
    parser.add_argument("--refresh", action="store_true",
                        help="Refetch even if already cached (single card) or force redownload the bulk snapshot.")
    parser.add_argument("--cache", type=Path, default=CACHE_PATH,
                        help="Unified cache file path.")
    args = parser.parse_args()

    names = read_names(args)
    if not names:
        # parser.error() prints usage and exits with code 2.
        parser.error("no card names provided (pass names as args or via --file)")

    cache = load_cache(args.cache)
    # Mode is chosen by count: a single card hits the API directly, anything
    # more falls to the bulk path (which merges every card into the cache).
    if len(names) == 1:
        changed = run_single_mode(names[0], cache, args.refresh)
    else:
        changed = run_bulk_mode(names, cache, args.refresh)

    if changed:
        save_cache(args.cache, cache)
        print(
            f"\nsaved {len(cache['cards'])} unique cards "
            f"({len(cache['aliases'])} aliases) to {args.cache}"
        )
    else:
        print("\nno changes")
    return 0


if __name__ == "__main__":
    # sys.exit propagates the int return code so shells / CI can react.
    sys.exit(main())
