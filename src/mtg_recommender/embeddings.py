"""Sentence-transformer embeddings for cards and tags.

Encodes each card's oracle text into `text_embedding` on the card doc and
each tag's label + description into `embedding` on the tag doc. The
weighted-average fuse that produces a single `card_vector` is deferred
to the next Phase 2 slice.

Default model is `sentence-transformers/all-mpnet-base-v2` (768-dim,
~420 MB on first download). Override with the `MTG_EMBEDDING_MODEL`
env var when a different model is wanted; the module doesn't bake the
dimensionality in anywhere, so swapping is a one-line change.

Dependency: `sentence-transformers` (which pulls in `torch`). It's an
optional extra — `pip install -e ".[embeddings]"` — because the Phase 1
fetchers don't need it and we don't want to force every user to pay the
~1 GB torch download. A clear error points at that install if the import
fails.

Mongo storage:

    cards.text_embedding : list[float]   # one entry per oracle_text embed
    tags.embedding       : list[float]   # one entry per label+description embed

Vectors are stored as plain BSON double arrays (what pymongo serialises
`list[float]` as). That costs ~8 bytes per dim, so a 768-dim vector adds
~6 KB per doc; 38k cards is ~230 MB total. Compact enough that we don't
need to reach for BSON Binary with float32 yet.

Usage (after `pip install -e ".[embeddings]"`):

    mtg-embed cards                 # embed cards that don't yet have text_embedding
    mtg-embed cards --refresh       # re-embed every card (expensive)
    mtg-embed tags                  # same shape for tags
    mtg-embed tags --refresh
    mtg-embed cards --limit 100     # cap the work (useful for smoke-tests)

Progress is reported every `log_every` cards/tags so a long run shows signs
of life. The default is 500.
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Iterable, Optional

from pymongo import UpdateOne
from pymongo.collection import Collection

from . import storage

# Default model. Can be overridden per invocation via the env var; keeping it
# a module-level constant means tests reading the default don't need to
# duplicate the string literal.
DEFAULT_MODEL = "sentence-transformers/all-mpnet-base-v2"
MODEL_ENV = "MTG_EMBEDDING_MODEL"

# Batch size for both the encoder forward pass and the Mongo bulk_write. 64
# is a safe default on CPU; a GPU would want more. Overridable via CLI flag
# but not often worth tuning.
DEFAULT_BATCH_SIZE = 64

# How often `embed_cards` / `embed_tags` print progress. Printing every doc
# would spam; every 500 keeps a long run legible on one screen.
DEFAULT_LOG_EVERY = 500


# ---------------------------------------------------------------------------
# Encoder cache
# ---------------------------------------------------------------------------

# A single encoder per process. sentence-transformers models hold a torch
# model in memory (~400 MB for mpnet), so we don't want to reload per call.
_encoder = None


def _load_encoder(model_name: str):
    """Import sentence-transformers lazily and construct the encoder.

    The import is deferred so running `mtg-check` or the fetchers doesn't
    pay the torch import cost (which can take seconds). A missing dep
    raises a clear RuntimeError pointing at the `[embeddings]` extra.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise RuntimeError(
            "sentence-transformers is not installed. "
            "Run `pip install -e \".[embeddings]\"` to add it."
        ) from e
    return SentenceTransformer(model_name)


def get_encoder():
    """Return the cached encoder, loading it on first access."""
    global _encoder
    if _encoder is None:
        model_name = os.environ.get(MODEL_ENV, DEFAULT_MODEL)
        _encoder = _load_encoder(model_name)
    return _encoder


def reset_encoder(encoder=None):
    """Replace the cached encoder (tests inject a fake; pass None to clear)."""
    global _encoder
    _encoder = encoder


# ---------------------------------------------------------------------------
# Core embed loops
# ---------------------------------------------------------------------------

def _tag_text(tag: dict) -> str:
    """Compose the text we feed to the encoder for a tag doc.

    Label alone is slug-ish and short; description adds meaningful detail
    ("Removes a single permanent"). Joining them with a period gives the
    encoder a longer, more informative input. A missing description is
    skipped so we don't send a trailing ". " that looks like a sentence
    boundary with no sentence after it.
    """
    label = tag.get("label") or tag.get("_id") or ""
    description = tag.get("description")
    if description:
        return f"{label}. {description}"
    return label


def embed_cards(
    coll: Collection,
    *,
    refresh: bool = False,
    batch_size: int = DEFAULT_BATCH_SIZE,
    limit: Optional[int] = None,
    encoder=None,
    log_every: int = DEFAULT_LOG_EVERY,
) -> int:
    """Encode oracle_text on every card and $set `text_embedding`.

    Skips docs that already have a `text_embedding` field unless `refresh`
    is True. Docs without oracle_text (meld pieces with top-level null,
    tokens with no rules text) are also skipped — there's nothing to embed.

    Returns the number of docs actually written. `limit` caps the work;
    useful for a smoke test before committing to a full ~40k encode.
    """
    encoder = encoder if encoder is not None else get_encoder()
    query: dict = {"oracle_text": {"$exists": True, "$nin": [None, ""]}}
    if not refresh:
        query["text_embedding"] = {"$exists": False}

    total = coll.count_documents(query)
    if limit is not None:
        total = min(total, limit)
    if total == 0:
        print("no cards to embed")
        return 0
    print(f"embedding oracle_text for {total} cards (batch={batch_size})")

    cursor = coll.find(query, {"_id": 1, "oracle_text": 1})
    if limit is not None:
        cursor = cursor.limit(limit)

    return _run_embed_loop(
        coll=coll,
        rows=((doc["_id"], doc["oracle_text"]) for doc in cursor),
        total=total,
        batch_size=batch_size,
        encoder=encoder,
        field="text_embedding",
        log_every=log_every,
        label="cards",
    )


def embed_tags(
    coll: Collection,
    *,
    refresh: bool = False,
    batch_size: int = DEFAULT_BATCH_SIZE,
    limit: Optional[int] = None,
    encoder=None,
    log_every: int = DEFAULT_LOG_EVERY,
) -> int:
    """Encode label + description on every tag and $set `embedding`.

    Mirror of `embed_cards` but keyed on the tag collection and using the
    tag's own `_id` (slug) as the primary key.
    """
    encoder = encoder if encoder is not None else get_encoder()
    query: dict = {}
    if not refresh:
        query["embedding"] = {"$exists": False}

    total = coll.count_documents(query)
    if limit is not None:
        total = min(total, limit)
    if total == 0:
        print("no tags to embed")
        return 0
    print(f"embedding label+description for {total} tags (batch={batch_size})")

    cursor = coll.find(query, {"_id": 1, "label": 1, "description": 1})
    if limit is not None:
        cursor = cursor.limit(limit)

    return _run_embed_loop(
        coll=coll,
        rows=((doc["_id"], _tag_text(doc)) for doc in cursor),
        total=total,
        batch_size=batch_size,
        encoder=encoder,
        field="embedding",
        log_every=log_every,
        label="tags",
    )


def _run_embed_loop(
    *,
    coll: Collection,
    rows: Iterable[tuple[str, str]],
    total: int,
    batch_size: int,
    encoder,
    field: str,
    log_every: int,
    label: str,
) -> int:
    """Core batched encode-and-write loop shared by cards and tags.

    Keeps I/O (Mongo bulk_write) and compute (encoder.encode) at the same
    cadence so neither stalls on the other.
    """
    done = 0
    batch_ids: list[str] = []
    batch_texts: list[str] = []

    def flush() -> int:
        if not batch_ids:
            return 0
        vectors = encoder.encode(
            batch_texts,
            batch_size=len(batch_texts),
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        ops = [
            UpdateOne({"_id": sid}, {"$set": {field: vec.tolist()}})
            for sid, vec in zip(batch_ids, vectors)
        ]
        coll.bulk_write(ops, ordered=False)
        return len(batch_ids)

    for _id, text in rows:
        batch_ids.append(_id)
        batch_texts.append(text)
        if len(batch_ids) >= batch_size:
            done += flush()
            batch_ids.clear()
            batch_texts.clear()
            if done % log_every < batch_size:
                print(f"  embedded {done}/{total} {label}")

    done += flush()
    print(f"  embedded {done}/{total} {label}")
    return done


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _cmd_cards(args: argparse.Namespace) -> int:
    storage.ensure_indexes()
    written = embed_cards(
        storage.cards_collection(),
        refresh=args.refresh,
        batch_size=args.batch_size,
        limit=args.limit,
    )
    print(f"\nwrote text_embedding on {written} cards")
    return 0


def _cmd_tags(args: argparse.Namespace) -> int:
    written = embed_tags(
        storage.tags_collection(),
        refresh=args.refresh,
        batch_size=args.batch_size,
        limit=args.limit,
    )
    print(f"\nwrote embedding on {written} tags")
    return 0


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="mtg-embed",
        description="Encode oracle text and tags into dense vectors stored on Mongo docs.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument(
            "--refresh", action="store_true",
            help="Re-embed documents that already have an embedding field.",
        )
        sp.add_argument(
            "--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
            help=f"Encoder + bulk-write batch size (default: {DEFAULT_BATCH_SIZE}).",
        )
        sp.add_argument(
            "--limit", type=int, default=None,
            help="Cap the number of docs to embed (useful for smoke-tests).",
        )

    cards_p = sub.add_parser("cards", help="Embed oracle_text onto each card.")
    add_common(cards_p)
    cards_p.set_defaults(func=_cmd_cards)

    tags_p = sub.add_parser("tags", help="Embed label+description onto each tag.")
    add_common(tags_p)
    tags_p.set_defaults(func=_cmd_tags)

    args = parser.parse_args(list(argv) if argv is not None else None)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
