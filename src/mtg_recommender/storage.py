"""MongoDB-backed persistence layer.

Owns the "how do we talk to Mongo?" question so no other module has to:
reads connection config from the environment (via a `.env` file in
development), caches a single MongoClient per process, and exposes
handles to the two collections the rest of the package uses.

Why Mongo (and only Mongo):

    PLAN.md Phase 1 declares MongoDB Atlas the source of truth for card
    + tag data and explicitly rules out a dual-write / offline-fallback
    story — "once Mongo is wired up, the on-disk JSON cache goes away".
    Keeping that decision in one module means every other module just
    takes a Mongo collection as its input and doesn't need to care
    whether it was mocked, pointed at Atlas, or pointed at a local
    Docker mongod.

Collections:

    cards : one document per card, _id = scryfall_id. Shape:
        {
          "_id":        "<scryfall_id>",       # Mongo primary key AND the
                                               # scryfall id (no duplicate
                                               # field — extract_oracle
                                               # renames at export when a
                                               # consumer wants the
                                               # semantic name).
          "oracle_id":  "<Scryfall oracle id>",# join key to the tags bulk
          "name":       "...",
          "names":      ["lightning bolt", "lightning bolt // lightning bolt"],
                                               # lowered aliases; replaces
                                               # the old top-level alias
                                               # dict with an on-document
                                               # array indexed below.
          "mana_cost":  "...",
          "type_line":  "...",
          "oracle_text":"...",
          "oracle_text_sha": "<16-hex sha256>",# diff key for incremental merges
          "tags":       ["spot-removal", ...], # populated by oracle_tags import
        }
        Per-card `updated_at` is intentionally absent: the snapshot
        timestamp lives once in the `meta` collection under
        `_id = "oracle_cards"`, since every card from a given bulk
        merge shares the same value.

    tags  : one document per oracle tag, _id = slug. Shape:
        {
          "_id":           "<slug>",
          "label":         "...",
          "description":   "..." | None,
          "parent_slugs":  [...],
          "child_slugs":   [...],
          "aliases":       [...]
        }
        The raw Scryfall tag UUID isn't stored — nothing in the project
        cross-refs it against the API, so persisting it was overhead.

    meta  : one document per persisted scalar, _id = key. Shape:
        {
          "_id":    "<key>",      # "oracle_cards", "oracle_tags",
                                  # "text_embedding_model",
                                  # "tag_embedding_model", "fuse_alpha", …
          "value":  "<any scalar>"
        }
      Used for two purposes: bulk-snapshot timestamps (so repeat
      `scryfall-fetch` / `scryfall-fetch-tags` runs skip the download
      when nothing moved) and embedding-config signatures (so
      `mtg-embed` auto-invalidates when the active model or alpha
      changes). `get_meta_value` / `set_meta_value` are the only API.

Indexes created by `ensure_indexes`:

    cards.names    : alias / name-based lookup (multi-key index because
                     `names` is an array; a lookup like
                     `cards.find_one({"names": "lightning bolt"})` is
                     an index hit regardless of position inside the
                     array).
    cards.oracle_id: the join key for the tag importer.

    _id on every collection is indexed automatically by Mongo and does
    not need to be declared here.

Env vars (read on first client access, from `.env` if present):

    MONGODB_URI                : required. Atlas connection string or any
                                 pymongo-compatible URI (local Docker,
                                 replica set, in-memory for tests). The
                                 storage module raises a clear
                                 RuntimeError if this is missing rather
                                 than letting a cryptic pymongo error
                                 bubble up.
    MONGODB_DB                 : database name (default: `mtg_recommender`).
    MONGODB_CARDS_COLLECTION   : (default: `cards`).
    MONGODB_TAGS_COLLECTION    : (default: `tags`).
    MONGODB_META_COLLECTION    : (default: `meta`).

Testability:

    Tests swap the cached MongoClient out for a mongomock instance:

        from mtg_recommender import storage
        storage.reset_client(mongomock.MongoClient())

    `reset_client` also clears any stale client so each test starts
    with the expected state. The default `get_client()` path only
    reaches `MongoClient(...)` when the cache is empty, so tests
    never touch pymongo's real driver.
"""
from __future__ import annotations

import os
from typing import Optional

from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database

# python-dotenv is a runtime dep (see pyproject.toml) but the import is
# wrapped so a stripped-down environment (CI without the dev extras, say)
# still loads this module. If dotenv isn't available, env vars must come
# from the shell / process environment itself.
try:
    from dotenv import load_dotenv

    # load_dotenv() silently does nothing if .env is absent, so this is
    # safe to call at import time. Values already set in the real
    # environment take precedence (dotenv's default behaviour) — handy
    # for prod / CI where MONGODB_URI is exported directly.
    load_dotenv()
except ImportError:  # pragma: no cover - dotenv is a declared dep
    pass


# ---------------------------------------------------------------------------
# Env var names + defaults
# ---------------------------------------------------------------------------

MONGODB_URI_ENV = "MONGODB_URI"
MONGODB_DB_ENV = "MONGODB_DB"
MONGODB_CARDS_COLLECTION_ENV = "MONGODB_CARDS_COLLECTION"
MONGODB_TAGS_COLLECTION_ENV = "MONGODB_TAGS_COLLECTION"
MONGODB_META_COLLECTION_ENV = "MONGODB_META_COLLECTION"

DEFAULT_DB = "mtg_recommender"
DEFAULT_CARDS_COLLECTION = "cards"
DEFAULT_TAGS_COLLECTION = "tags"
DEFAULT_META_COLLECTION = "meta"


# ---------------------------------------------------------------------------
# Client cache
# ---------------------------------------------------------------------------

# A single MongoClient per process is the pymongo recommendation — the
# client manages its own connection pool internally, so reusing it keeps
# authentication, DNS resolution and TLS handshakes out of the hot path.
_client: Optional[MongoClient] = None


def _require_uri() -> str:
    """Fetch MONGODB_URI from the environment or raise with a usable message.

    The storage module can't do anything without a URI, so a missing
    value is a hard error. The error text points directly at the
    remediation (`.env.example` + copy to `.env`) so the first-run
    experience doesn't require spelunking through code.
    """
    uri = os.environ.get(MONGODB_URI_ENV)
    if not uri:
        raise RuntimeError(
            f"{MONGODB_URI_ENV} is not set. "
            "Copy .env.example to .env and fill in your Atlas connection string, "
            f"or export {MONGODB_URI_ENV} directly in your shell."
        )
    return uri


def get_client() -> MongoClient:
    """Return the cached MongoClient, constructing it on first access."""
    global _client
    if _client is None:
        _client = MongoClient(_require_uri())
    return _client


def reset_client(client: Optional[MongoClient] = None) -> None:
    """Replace the cached client (or clear it so the next call rebuilds).

    Tests call this with a mongomock.MongoClient() to swap in the
    offline backend. Passing None clears the cache, which is useful
    between tests that want to re-exercise the lazy-init path.
    """
    global _client
    _client = client


# ---------------------------------------------------------------------------
# Database + collection handles
# ---------------------------------------------------------------------------

def get_database(name: Optional[str] = None) -> Database:
    """Return the configured Database.

    `name` lets a test override the env default without touching
    os.environ. Production callers pass nothing and get whatever
    MONGODB_DB resolves to (or the DEFAULT_DB fallback).
    """
    db_name = name or os.environ.get(MONGODB_DB_ENV, DEFAULT_DB)
    return get_client()[db_name]


def cards_collection(db: Optional[Database] = None) -> Collection:
    """Return the `cards` collection handle."""
    db = db if db is not None else get_database()
    name = os.environ.get(MONGODB_CARDS_COLLECTION_ENV, DEFAULT_CARDS_COLLECTION)
    return db[name]


def tags_collection(db: Optional[Database] = None) -> Collection:
    """Return the `tags` collection handle."""
    db = db if db is not None else get_database()
    name = os.environ.get(MONGODB_TAGS_COLLECTION_ENV, DEFAULT_TAGS_COLLECTION)
    return db[name]


def meta_collection(db: Optional[Database] = None) -> Collection:
    """Return the `meta` collection handle.

    Used to track the snapshot timestamps of imported bulk sources
    (oracle_cards, oracle_tags) so freshness checks don't need to
    re-scan the data collections.
    """
    db = db if db is not None else get_database()
    name = os.environ.get(MONGODB_META_COLLECTION_ENV, DEFAULT_META_COLLECTION)
    return db[name]


# ---------------------------------------------------------------------------
# Index management
# ---------------------------------------------------------------------------

def ensure_indexes(db: Optional[Database] = None) -> None:
    """Create the indexes the rest of the package relies on.

    pymongo's create_index is idempotent — calling it with an existing
    index is a no-op — so this is safe to call every time a CLI starts.
    Doing so means a brand-new database works without a separate
    migration step, and a code change that adds an index gets picked up
    on the next CLI run.
    """
    cards = cards_collection(db)
    # `names` is an array field (one card can have many lowered aliases),
    # so Mongo creates a multikey index that matches a lookup against
    # any element of the array. This is what makes
    # `cards.find({"names": "lightning bolt"})` fast.
    cards.create_index("names", name="names_lookup")
    # oracle_id is the join key for the oracle_tags importer.
    cards.create_index("oracle_id", name="oracle_id_lookup")


# ---------------------------------------------------------------------------
# Meta helpers (persisted scalar key/value: snapshot timestamps and embedding
# config signatures)
# ---------------------------------------------------------------------------

def get_meta_value(key: str, db: Optional[Database] = None):
    """Return the stored value for `key`, or None if not set.

    Serves two kinds of callers:
      - freshness-skip: `scryfall-fetch` / `scryfall-fetch-tags` store a
        snapshot `updated_at` under `oracle_cards` / `oracle_tags`.
      - auto-invalidation: `mtg-embed` stores the active model name
        under `text_embedding_model` / `tag_embedding_model` and the
        active alpha under `fuse_alpha`, so a config change wipes the
        now-stale embeddings on the next run.

    The stored value's type is whatever the setter wrote — Mongo
    serialises strings and floats transparently, so the helper is
    untyped on purpose.
    """
    doc = meta_collection(db).find_one({"_id": key})
    return doc.get("value") if doc else None


def set_meta_value(key: str, value, db: Optional[Database] = None) -> None:
    """Upsert the stored value for `key`."""
    meta_collection(db).update_one(
        {"_id": key},
        {"$set": {"value": value}},
        upsert=True,
    )
