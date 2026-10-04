"""Fetch per-commander signals from EDHREC's public JSON API.

EDHREC has no documented REST API, but it exposes the data its own
Next.js front-end consumes at `https://json.edhrec.com/pages/commanders/
<slug>.json` — keyless, stable, and the same endpoint multiple
third-party scrapers rely on. CloudFlare in front of the host blocks
direct browser GETs via a fingerprint challenge, but Python's `urllib`
sidesteps the challenge cleanly (verified against the live endpoint).

What this module provides:

    commander_slug(name)         — normalise "Atraxa, Praetors' Voice"
                                   to EDHREC's url slug "atraxa-praetors-voice".
    fetch_payload(slug)          — one HTTP round trip, returns raw JSON.
    get_commander_payload(name)  — cache-aware wrapper. 7-day TTL in Mongo.
    extract_card_signals(payload)— flatten the nested cardlists into a deduped
                                   list of CardSignal records.
    get_commander_signals(name)  — convenience: cached payload → signals.

Per-card signals we pull out of the payload:

    name            : card name as EDHREC spells it (matches Scryfall)
    scryfall_id     : Scryfall UUID — direct join to our cards._id
    lift            : likelihood ratio vs. baseline (>1 = this card is
                      more associated with this commander than with
                      random decks of the same color identity). This is
                      the primary signal for Phase 3's cluster-eval
                      step.
    synergy         : correlation score (−1..1); legacy EDHREC signal,
                      kept for backwards comparison
    num_decks       : how many decks of this commander run the card
    potential_decks : total decks of this commander in the sample
                      (num_decks / potential_decks → inclusion rate)
    trend_zscore    : recency / momentum signal

Caching: payloads land in the `edhrec` Mongo collection keyed by slug;
a stored entry is reused if it's less than EDHREC_CACHE_TTL_DAYS old.
EDHREC updates slowly (daily batch recomputes), so 7 days is a safe
default — tune via the `EDHREC_CACHE_TTL_DAYS` env var if needed. Pass
`force=True` to bypass the cache for one call.

Ethics: this is a public endpoint, but we're a guest. The default
User-Agent identifies the project and includes a contact. The cache
makes repeat lookups free. One commander lookup is one HTTP request.
"""
from __future__ import annotations

import json
import os
import re
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

from pymongo.collection import Collection

from . import storage

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EDHREC_JSON_BASE = "https://json.edhrec.com"

# Default cache TTL. EDHREC recomputes once a day at most and the deltas
# for any given commander are tiny; 7 days is cheap for us and friendly
# to them. Override with the EDHREC_CACHE_TTL_DAYS env var.
EDHREC_CACHE_TTL_DAYS = 7
EDHREC_CACHE_TTL_ENV = "EDHREC_CACHE_TTL_DAYS"

# User-Agent identifies the project + a contact so EDHREC's ops can reach
# us if we misbehave. Hard-coded (not env-overridable) so the identifier
# is honest — nobody can impersonate a browser from a config file.
USER_AGENT = (
    "MTG-Recommender/0.1 "
    "(github.com/MangoPicante/mtg-recommender; +Phase-3 recommender data)"
)

# Standard JSON Accept header. Doesn't change the response (the endpoint
# only serves JSON) but it's the honest thing to send.
_HEADERS = {"User-Agent": USER_AGENT, "Accept": "application/json"}

# Non-alphanumeric → single hyphen. We strip apostrophes BEFORE this so
# "Praetors' Voice" collapses to "praetors-voice" rather than
# "praetors--voice".
_SLUG_NONWORD_RE = re.compile(r"[^a-z0-9]+")


# ---------------------------------------------------------------------------
# Signal shape
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CardSignal:
    """One card's EDHREC signals for a given commander.

    Frozen so a list of CardSignal is safely hashable per-element and
    the caller can't accidentally mutate values they received by reference.
    """

    name: str
    scryfall_id: str
    lift: float
    synergy: float
    num_decks: int
    potential_decks: int
    trend_zscore: Optional[float] = None

    @property
    def inclusion_rate(self) -> float:
        """`num_decks / potential_decks` — what fraction of this
        commander's decks run the card. Guards against divide-by-zero
        (shouldn't happen in real EDHREC data, but a sentinel 0 is
        safer than a RuntimeError in a recommender hot path).
        """
        if self.potential_decks <= 0:
            return 0.0
        return self.num_decks / self.potential_decks


# ---------------------------------------------------------------------------
# Slug derivation
# ---------------------------------------------------------------------------

def commander_slug(name: str) -> str:
    """Convert a commander's card name to EDHREC's URL slug.

    Examples (verified against live edhrec.com URLs):

        "Atraxa, Praetors' Voice" -> "atraxa-praetors-voice"
        "Yuriko, the Tiger's Shadow" -> "yuriko-the-tigers-shadow"
        "Jhoira, Weatherlight Captain" -> "jhoira-weatherlight-captain"

    Partner / background commanders use a different URL shape
    (`<slug1>-<slug2>`) and are deferred to a follow-up — this helper
    is single-name-only. A `None` or empty-string input is a programmer
    error (we never ask EDHREC for an unnamed commander), so we raise.

    The transformation is deliberately simple so slug mismatches surface
    as 404s rather than being silently papered over by over-eager
    normalization — EDHREC has slug canonicalisation we don't want to
    second-guess.
    """
    if not name:
        raise ValueError("commander_slug() requires a non-empty name")
    # Lowercase first so the regex is simpler.
    lowered = name.lower()
    # Strip apostrophes (both ASCII and curly) before the non-word
    # collapse so "praetors' voice" doesn't become "praetors--voice".
    stripped = lowered.replace("'", "").replace("’", "")
    # Replace any run of non-[a-z0-9] with a single hyphen, then trim
    # leading/trailing hyphens that commas / dots at the boundaries leave.
    return _SLUG_NONWORD_RE.sub("-", stripped).strip("-")


# ---------------------------------------------------------------------------
# HTTP fetch
# ---------------------------------------------------------------------------

def _commander_url(slug: str) -> str:
    """URL for a commander's EDHREC JSON page."""
    return f"{EDHREC_JSON_BASE}/pages/commanders/{slug}.json"


def fetch_payload(slug: str, *, timeout: int = 30) -> dict:
    """One HTTP round trip to json.edhrec.com. Returns parsed JSON dict.

    Raises `urllib.error.HTTPError` on non-2xx — the caller decides
    whether a 404 ("commander not in EDHREC") is fatal or a soft miss.
    This function stays dumb; retry policy and 404 handling live at the
    callsite.
    """
    req = urllib.request.Request(_commander_url(slug), headers=_HEADERS)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


# ---------------------------------------------------------------------------
# Mongo-backed cache
# ---------------------------------------------------------------------------

def _ttl_days() -> int:
    """Current cache TTL in days, from env override or default."""
    raw = os.environ.get(EDHREC_CACHE_TTL_ENV)
    if not raw:
        return EDHREC_CACHE_TTL_DAYS
    try:
        return max(0, int(raw))
    except ValueError:
        # Bad override shouldn't silently fall back to no caching —
        # loudly reject so misconfiguration is obvious.
        raise ValueError(
            f"{EDHREC_CACHE_TTL_ENV} must be an integer number of days, got {raw!r}"
        )


def _is_fresh(fetched_at_iso: str) -> bool:
    """Is a cache entry fetched at `fetched_at_iso` still within TTL?"""
    try:
        fetched_at = datetime.fromisoformat(fetched_at_iso)
    except (TypeError, ValueError):
        return False
    age = datetime.now(timezone.utc) - fetched_at
    return age < timedelta(days=_ttl_days())


def _cache_get(coll: Collection, slug: str) -> Optional[dict]:
    """Return the cached payload for `slug` if it exists and is fresh."""
    doc = coll.find_one({"_id": slug})
    if not doc:
        return None
    if not _is_fresh(doc.get("fetched_at", "")):
        return None
    return doc.get("payload")


def _cache_put(coll: Collection, slug: str, payload: dict) -> None:
    """Upsert a payload into the cache with the current UTC timestamp."""
    coll.update_one(
        {"_id": slug},
        {
            "$set": {
                "fetched_at": datetime.now(timezone.utc).isoformat(),
                "payload": payload,
            }
        },
        upsert=True,
    )


def get_commander_payload(
    name: str,
    *,
    force: bool = False,
    coll: Optional[Collection] = None,
) -> dict:
    """Return the EDHREC payload for `name`, hitting the cache first.

    - `force=True` bypasses the cache and refetches, overwriting the
      cached entry.
    - `coll` lets a caller inject a specific collection handle (tests
      do this). The default resolves to `storage.edhrec_collection()`.

    The slug derived from `name` becomes the cache key. If a commander
    renames on EDHREC's side (rare) the old key quietly expires.
    """
    if coll is None:
        coll = storage.edhrec_collection()
    slug = commander_slug(name)
    if not force:
        cached = _cache_get(coll, slug)
        if cached is not None:
            return cached
    payload = fetch_payload(slug)
    _cache_put(coll, slug, payload)
    return payload


# ---------------------------------------------------------------------------
# Signal extraction
# ---------------------------------------------------------------------------

def extract_card_signals(payload: dict) -> list[CardSignal]:
    """Flatten `container.json_dict.cardlists` into deduped `CardSignal`s.

    EDHREC groups each commander's cards into ~12 themed sections
    (`cardlists`), and a single card can appear in several sections
    (e.g. "Top Cards" and "Creatures"). The lift / synergy numbers in
    each section are computed from the same underlying data, so they
    agree — but if they don't, we keep the entry with the highest
    `lift` on the theory that the user cares most about the
    strongest-correlated reading.

    A payload without the expected `container.json_dict.cardlists`
    structure returns `[]` rather than raising, so a schema shift at
    EDHREC's end degrades gracefully (the recommender falls back to
    tag-only ranking).
    """
    try:
        cardlists = payload["container"]["json_dict"]["cardlists"]
    except (KeyError, TypeError):
        return []
    best: dict[str, CardSignal] = {}
    for section in cardlists or []:
        for raw in section.get("cardviews") or []:
            sig = _signal_from_cardview(raw)
            if sig is None:
                continue
            existing = best.get(sig.scryfall_id)
            if existing is None or sig.lift > existing.lift:
                best[sig.scryfall_id] = sig
    return list(best.values())


def _signal_from_cardview(raw: dict) -> Optional[CardSignal]:
    """Project one `cardviews[*]` entry into a CardSignal.

    Returns None if the required fields (id, name, lift, num_decks)
    aren't present — EDHREC occasionally includes partial entries for
    cards with insufficient data, and we'd rather drop them than
    carry zero/None sentinels into downstream math.
    """
    scryfall_id = raw.get("id")
    name = raw.get("name")
    lift = raw.get("lift")
    num_decks = raw.get("num_decks")
    if not scryfall_id or not name or lift is None or num_decks is None:
        return None
    return CardSignal(
        name=name,
        scryfall_id=scryfall_id,
        lift=float(lift),
        synergy=float(raw.get("synergy") or 0.0),
        num_decks=int(num_decks),
        potential_decks=int(raw.get("potential_decks") or 0),
        trend_zscore=_optional_float(raw.get("trend_zscore")),
    )


def _optional_float(value) -> Optional[float]:
    """Coerce to float if not None; keep None otherwise."""
    if value is None:
        return None
    return float(value)


# ---------------------------------------------------------------------------
# Convenience top-level
# ---------------------------------------------------------------------------

def get_commander_signals(
    name: str,
    *,
    force: bool = False,
    coll: Optional[Collection] = None,
) -> list[CardSignal]:
    """One-shot: cached payload → extracted signals for `name`."""
    return extract_card_signals(get_commander_payload(name, force=force, coll=coll))
