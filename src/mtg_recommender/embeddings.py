"""Sentence-transformer embeddings for cards and tags.

Encodes each card's oracle text into `text_embedding` on the card doc,
each tag's label + description into `embedding` on the tag doc, and
fuses the two into a single `card_vector` per card via a weighted
average (``alpha * text_unit + (1 - alpha) * tag_unit``, result
renormalized).

Default model is `sentence-transformers/all-mpnet-base-v2` (768-dim,
~420 MB on first download). Override with the `MTG_EMBEDDING_MODEL`
env var when a different model is wanted; the module doesn't bake the
dimensionality in anywhere, so swapping is a one-line change. Fusing
assumes text and tag vectors share dimensionality (true when they come
from the same encoder); `fuse_card_vectors` raises if they don't.

`sentence-transformers` (and transitively `torch`, ~1 GB) is a runtime
dependency declared in `pyproject.toml`. The import inside
`_load_encoder` stays lazy so the Phase 1 CLIs and the test suite don't
pay torch's multi-second import cost when nothing in them touches the
encoder.

Mongo storage:

    cards.text_embedding : list[float]   # one entry per oracle_text embed
    cards.card_vector    : list[float]   # fused text + aggregated tag vector
    tags.embedding       : list[float]   # one entry per label+description embed

Vectors are stored as plain BSON double arrays (what pymongo serialises
`list[float]` as). That costs ~8 bytes per dim, so a 768-dim vector adds
~6 KB per doc; 38k cards is ~230 MB total. Compact enough that we don't
need to reach for BSON Binary with float32 yet.

Usage (after `pip install -e .`):

    mtg-embed cards                 # embed cards that don't yet have text_embedding
    mtg-embed cards --refresh       # re-embed every card (expensive)
    mtg-embed tags                  # same shape for tags
    mtg-embed tags --refresh
    mtg-embed cards --limit 100     # cap the work (useful for smoke-tests)
    mtg-embed fuse                  # build card_vector for cards that don't have one
    mtg-embed fuse --alpha 0.7      # weight text more heavily (default 0.6)
    mtg-embed fuse --refresh        # re-fuse every card

Freshness of `card_vector` is still user-driven: a `text_embedding` or
`tags` change requires `mtg-embed fuse --refresh` to pick it up.
Auto-invalidation lands in a later slice.

Progress is reported every `log_every` cards/tags so a long run shows signs
of life. The default is 500.
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Iterable, Optional

import numpy as np
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

# Weight on the text side of the fuse. 0.6 leans slightly on oracle_text —
# it carries the richer per-card signal — while still letting the tag
# aggregate pull similar cards together. Tunable per invocation via
# `mtg-embed fuse --alpha`; a sweep over an evaluation set is a Phase 3 task.
DEFAULT_ALPHA = 0.6


# ---------------------------------------------------------------------------
# Encoder cache
# ---------------------------------------------------------------------------

# A single encoder per process. sentence-transformers models hold a torch
# model in memory (~400 MB for mpnet), so we don't want to reload per call.
_encoder = None


def _load_encoder(model_name: str):
    """Import sentence-transformers lazily and construct the encoder.

    The import is deferred so running `mtg-check` or the Phase 1 fetchers
    doesn't pay torch's multi-second import cost just by living in the
    same package. `sentence-transformers` is a declared runtime dep, so
    a missing import here is a broken install — let it raise naturally.
    """
    from sentence_transformers import SentenceTransformer
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
# Fuse: text_embedding + aggregated tag embedding -> card_vector
# ---------------------------------------------------------------------------

def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    """Return `vec` scaled to unit length, or `vec` unchanged if it's zero.

    Normalizing before the weighted sum means `alpha` controls direction
    rather than magnitude — a long tag vector can't swamp a shorter text
    vector just by being bigger.
    """
    norm = float(np.linalg.norm(vec))
    if norm == 0.0:
        # A zero vector has no direction; dividing would NaN. The caller
        # treats this as "no useful signal" and the fuse falls back to
        # the other side.
        return vec
    return vec / norm


def _aggregate_tag_vector(
    tag_slugs: Iterable[str], tag_embeddings: dict[str, np.ndarray]
) -> Optional[np.ndarray]:
    """Mean of the tag embeddings for `tag_slugs`; None if nothing resolves.

    Unknown slugs are silently dropped — a card may carry a tag that
    hasn't been embedded yet (new tag import, --limit on `embed tags`),
    and we'd rather fuse with what we have than skip the card entirely.
    """
    vecs = [tag_embeddings[slug] for slug in tag_slugs if slug in tag_embeddings]
    if not vecs:
        return None
    # float64 avoids accumulating rounding error when averaging many tags.
    return np.array(vecs, dtype=np.float64).mean(axis=0)


def _fuse_vectors(
    text_vec: np.ndarray, tag_vec: Optional[np.ndarray], alpha: float
) -> np.ndarray:
    """Blend L2-normalized text and tag vectors by `alpha`; renormalize.

    If `tag_vec` is None (card has no tags, or none of its tags have
    embeddings) the fused vector is just the normalized text vector —
    `alpha` effectively becomes 1.0 for that card. Returning a unit
    vector keeps cosine-similarity downstream well-behaved.
    """
    text_unit = _l2_normalize(text_vec.astype(np.float64))
    if tag_vec is None:
        return text_unit
    tag_unit = _l2_normalize(tag_vec)
    fused = alpha * text_unit + (1.0 - alpha) * tag_unit
    return _l2_normalize(fused)


def _load_tag_embeddings(tags_coll: Collection) -> dict[str, np.ndarray]:
    """Load every tag that has an `embedding` into {slug: np.ndarray}.

    The tag catalog is small (~2k entries), so holding it in memory for
    the fuse pass is trivial — much cheaper than one Mongo round-trip
    per card.
    """
    return {
        doc["_id"]: np.array(doc["embedding"], dtype=np.float64)
        for doc in tags_coll.find(
            {"embedding": {"$exists": True}}, {"_id": 1, "embedding": 1}
        )
    }


def fuse_card_vectors(
    cards_coll: Collection,
    tags_coll: Collection,
    *,
    refresh: bool = False,
    batch_size: int = DEFAULT_BATCH_SIZE,
    limit: Optional[int] = None,
    alpha: float = DEFAULT_ALPHA,
    log_every: int = DEFAULT_LOG_EVERY,
) -> int:
    """Fuse per-card text + aggregated tag vectors into `card_vector`.

    Only cards that already carry `text_embedding` are considered — the
    fuse is a pure read of existing embeddings, not an encoder pass.
    Default behaviour skips cards that already have `card_vector`;
    `refresh=True` re-fuses every eligible card (needed when `alpha`
    changes or an input vector was re-embedded upstream).

    Raises `ValueError` if a card's text and tag vectors have mismatched
    dimensionality — in practice that means tags were embedded with a
    different model than text, and the fix is to re-run `mtg-embed tags
    --refresh` with the same encoder.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")

    tag_embeddings = _load_tag_embeddings(tags_coll)

    # Only text-embedded cards are candidates; tag side falls back to
    # text-only when the card has no (or no embedded) tags.
    query: dict = {"text_embedding": {"$exists": True, "$ne": None}}
    if not refresh:
        query["card_vector"] = {"$exists": False}

    total = cards_coll.count_documents(query)
    if limit is not None:
        total = min(total, limit)
    if total == 0:
        print("no cards to fuse")
        return 0
    print(
        f"fusing card_vector for {total} cards "
        f"(alpha={alpha}, batch={batch_size}, tag_vocab={len(tag_embeddings)})"
    )

    cursor = cards_coll.find(
        query, {"_id": 1, "text_embedding": 1, "tags": 1}
    )
    if limit is not None:
        cursor = cursor.limit(limit)

    done = 0
    batch_ops: list[UpdateOne] = []

    def flush() -> int:
        if not batch_ops:
            return 0
        cards_coll.bulk_write(batch_ops, ordered=False)
        n = len(batch_ops)
        batch_ops.clear()
        return n

    for doc in cursor:
        text_vec = np.array(doc["text_embedding"], dtype=np.float64)
        tag_slugs = doc.get("tags") or []
        tag_vec = _aggregate_tag_vector(tag_slugs, tag_embeddings)
        if tag_vec is not None and tag_vec.shape != text_vec.shape:
            # Shape mismatch means a mixed-encoder state — refuse rather
            # than silently produce a vector in neither space.
            raise ValueError(
                f"text/tag embedding dims differ for card {doc['_id']!r}: "
                f"text={text_vec.shape[0]}, tag={tag_vec.shape[0]}. "
                "Re-run `mtg-embed tags --refresh` with the same encoder."
            )
        fused = _fuse_vectors(text_vec, tag_vec, alpha)
        batch_ops.append(
            UpdateOne({"_id": doc["_id"]}, {"$set": {"card_vector": fused.tolist()}})
        )
        if len(batch_ops) >= batch_size:
            done += flush()
            if done % log_every < batch_size:
                print(f"  fused {done}/{total} cards")

    done += flush()
    print(f"  fused {done}/{total} cards")
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


def _cmd_fuse(args: argparse.Namespace) -> int:
    # Fuse touches only `card_vector`, which already-covered indexes don't
    # care about — no `ensure_indexes` call needed here.
    written = fuse_card_vectors(
        storage.cards_collection(),
        storage.tags_collection(),
        refresh=args.refresh,
        batch_size=args.batch_size,
        limit=args.limit,
        alpha=args.alpha,
    )
    print(f"\nwrote card_vector on {written} cards")
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

    fuse_p = sub.add_parser(
        "fuse",
        help="Fuse text_embedding + aggregated tag embedding into card_vector.",
    )
    add_common(fuse_p)
    fuse_p.add_argument(
        "--alpha", type=float, default=DEFAULT_ALPHA,
        help=(
            f"Weight on the text side of the fuse, in [0, 1] "
            f"(default: {DEFAULT_ALPHA}). 1.0 ignores tags; 0.0 ignores text."
        ),
    )
    fuse_p.set_defaults(func=_cmd_fuse)

    args = parser.parse_args(list(argv) if argv is not None else None)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
