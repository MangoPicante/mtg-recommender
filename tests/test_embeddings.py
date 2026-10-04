"""Offline tests for mtg_recommender.embeddings.

Tests never load the real sentence-transformer model — too slow and
requires a 420 MB download on first use. Instead `reset_encoder` drops
a `FakeEncoder` into the module's cached slot, which produces
deterministic numpy vectors fast enough that the full file finishes in
well under a second.

Test classes:
    TestTagText           — label/description composition rules
    TestEmbedCards        — batching, refresh semantics, skip-no-text, limit
    TestEmbedTags         — mirror coverage for the tags collection
    TestEncoderCache      — reset_encoder; missing dep raises cleanly
    TestMainCLI           — argparse dispatch (cards / tags / --refresh / --limit)
"""
from __future__ import annotations

import io
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import patch

import mongomock
import numpy as np

from mtg_recommender import embeddings as emb
from mtg_recommender import storage

# ---------------------------------------------------------------------------
# Fake encoder
# ---------------------------------------------------------------------------

class FakeEncoder:
    """Deterministic numpy-backed stand-in for SentenceTransformer.

    Each input text produces a small fixed-dim vector where the first
    element is the hash of the text (so different inputs produce different
    vectors). Dimensionality is small (8) because test assertions only
    care about `is this vector present and the right length`.
    """

    def __init__(self, dim: int = 8):
        self.dim = dim
        self.call_log: list[list[str]] = []

    def encode(self, texts, **kwargs):
        # Record every batch so tests can assert on batching behaviour.
        self.call_log.append(list(texts))
        return np.array(
            [[sum(ord(c) for c in t) % 1000 + i for i in range(self.dim)] for t in texts],
            dtype=np.float32,
        )


def _run(argv: list[str]) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err), patch("sys.argv", ["mtg-embed", *argv]):
        rc = emb.main(argv)
    return rc, out.getvalue(), err.getvalue()


class _MongoBackedTestCase(unittest.TestCase):
    def setUp(self):
        storage.reset_client(mongomock.MongoClient())
        storage.ensure_indexes()
        self.cards = storage.cards_collection()
        self.tags = storage.tags_collection()
        self.encoder = FakeEncoder()
        emb.reset_encoder(self.encoder)

    def tearDown(self):
        emb.reset_encoder(None)
        storage.reset_client(None)


# ---------------------------------------------------------------------------
# _tag_text
# ---------------------------------------------------------------------------

class TestTagText(unittest.TestCase):

    def test_label_and_description_joined(self):
        self.assertEqual(
            emb._tag_text({"label": "Spot removal", "description": "Removes a single permanent."}),
            "Spot removal. Removes a single permanent.",
        )

    def test_no_description_returns_label_only(self):
        self.assertEqual(emb._tag_text({"label": "Evasion"}), "Evasion")

    def test_falls_back_to_slug_when_label_missing(self):
        self.assertEqual(emb._tag_text({"_id": "mana-rock"}), "mana-rock")


# ---------------------------------------------------------------------------
# embed_cards
# ---------------------------------------------------------------------------

class TestEmbedCards(_MongoBackedTestCase):

    def _seed(self, *rows):
        """Each row is (scryfall_id, oracle_text). Minimal fixture shape."""
        docs = [
            {"_id": sid, "scryfall_id": sid, "name": sid, "oracle_text": text}
            for sid, text in rows
        ]
        self.cards.insert_many(docs)

    def test_writes_text_embedding_as_list_of_floats(self):
        self._seed(("a", "deal 3 damage"))
        with redirect_stdout(io.StringIO()):
            emb.embed_cards(self.cards)
        doc = self.cards.find_one({"_id": "a"})
        self.assertIn("text_embedding", doc)
        self.assertIsInstance(doc["text_embedding"], list)
        self.assertEqual(len(doc["text_embedding"]), self.encoder.dim)
        # Must be plain Python floats, not numpy scalars (BSON would
        # choke on numpy types without a type coder).
        self.assertIsInstance(doc["text_embedding"][0], float)

    def test_skips_cards_without_oracle_text(self):
        self.cards.insert_many([
            {"_id": "a", "scryfall_id": "a", "name": "A", "oracle_text": "something"},
            {"_id": "b", "scryfall_id": "b", "name": "B"},  # no oracle_text at all
            {"_id": "c", "scryfall_id": "c", "name": "C", "oracle_text": None},
            {"_id": "d", "scryfall_id": "d", "name": "D", "oracle_text": ""},
        ])
        with redirect_stdout(io.StringIO()):
            written = emb.embed_cards(self.cards)
        self.assertEqual(written, 1)
        self.assertIn("text_embedding", self.cards.find_one({"_id": "a"}))
        for sid in ("b", "c", "d"):
            self.assertNotIn("text_embedding", self.cards.find_one({"_id": sid}))

    def test_default_skips_already_embedded(self):
        self._seed(("a", "text one"), ("b", "text two"))
        # Pre-populate `a` to simulate it being already embedded from a
        # prior run. Only `b` should get the new vector.
        self.cards.update_one({"_id": "a"}, {"$set": {"text_embedding": [0.0] * 8}})
        with redirect_stdout(io.StringIO()):
            written = emb.embed_cards(self.cards)
        self.assertEqual(written, 1)
        self.assertEqual(self.cards.find_one({"_id": "a"})["text_embedding"], [0.0] * 8)
        # `b` got a non-zero vector from the fake encoder.
        self.assertNotEqual(self.cards.find_one({"_id": "b"})["text_embedding"], [0.0] * 8)

    def test_refresh_re_embeds_everything(self):
        self._seed(("a", "text one"))
        self.cards.update_one({"_id": "a"}, {"$set": {"text_embedding": [0.0] * 8}})
        with redirect_stdout(io.StringIO()):
            emb.embed_cards(self.cards, refresh=True)
        self.assertNotEqual(self.cards.find_one({"_id": "a"})["text_embedding"], [0.0] * 8)

    def test_limit_caps_work(self):
        self._seed(*[(f"c{i}", f"text {i}") for i in range(10)])
        with redirect_stdout(io.StringIO()):
            written = emb.embed_cards(self.cards, limit=3)
        self.assertEqual(written, 3)
        embedded = list(self.cards.find({"text_embedding": {"$exists": True}}))
        self.assertEqual(len(embedded), 3)

    def test_batching_respects_batch_size(self):
        # 5 docs, batch_size=2 -> three encoder.encode calls of sizes 2,2,1.
        self._seed(*[(f"c{i}", f"text {i}") for i in range(5)])
        with redirect_stdout(io.StringIO()):
            emb.embed_cards(self.cards, batch_size=2)
        batch_sizes = [len(b) for b in self.encoder.call_log]
        self.assertEqual(batch_sizes, [2, 2, 1])

    def test_empty_collection_prints_message_and_returns_zero(self):
        with redirect_stdout(io.StringIO()) as out:
            written = emb.embed_cards(self.cards)
        self.assertEqual(written, 0)
        self.assertIn("no cards to embed", out.getvalue())


# ---------------------------------------------------------------------------
# embed_tags
# ---------------------------------------------------------------------------

class TestEmbedTags(_MongoBackedTestCase):

    def test_writes_embedding_on_each_tag(self):
        self.tags.insert_many([
            {"_id": "spot-removal", "label": "Spot removal", "description": "Removes a permanent."},
            {"_id": "evasion", "label": "Evasion", "description": None},
        ])
        with redirect_stdout(io.StringIO()):
            written = emb.embed_tags(self.tags)
        self.assertEqual(written, 2)
        for slug in ("spot-removal", "evasion"):
            doc = self.tags.find_one({"_id": slug})
            self.assertIn("embedding", doc)
            self.assertEqual(len(doc["embedding"]), self.encoder.dim)

    def test_label_plus_description_fed_to_encoder(self):
        self.tags.insert_one(
            {"_id": "spot-removal", "label": "Spot removal", "description": "Removes a permanent."}
        )
        with redirect_stdout(io.StringIO()):
            emb.embed_tags(self.tags)
        # The fake encoder stashes each batch; the only one here is a
        # single-tag batch whose text is the joined form.
        self.assertEqual(
            self.encoder.call_log,
            [["Spot removal. Removes a permanent."]],
        )

    def test_default_skips_already_embedded_tags(self):
        self.tags.insert_many([
            {"_id": "a", "label": "A"},
            {"_id": "b", "label": "B", "embedding": [0.0] * 8},
        ])
        with redirect_stdout(io.StringIO()):
            written = emb.embed_tags(self.tags)
        self.assertEqual(written, 1)
        self.assertIn("embedding", self.tags.find_one({"_id": "a"}))
        self.assertEqual(self.tags.find_one({"_id": "b"})["embedding"], [0.0] * 8)


# ---------------------------------------------------------------------------
# Encoder cache
# ---------------------------------------------------------------------------

class TestEncoderCache(unittest.TestCase):

    def tearDown(self):
        emb.reset_encoder(None)

    def test_reset_installs_the_given_encoder(self):
        fake = FakeEncoder()
        emb.reset_encoder(fake)
        self.assertIs(emb.get_encoder(), fake)

    def test_missing_sentence_transformers_dep_raises_actionable_error(self):
        emb.reset_encoder(None)
        with patch.dict("sys.modules", {"sentence_transformers": None}):
            with self.assertRaises(RuntimeError) as ctx:
                emb.get_encoder()
        self.assertIn("sentence-transformers", str(ctx.exception))
        self.assertIn("[embeddings]", str(ctx.exception))


# ---------------------------------------------------------------------------
# main() CLI
# ---------------------------------------------------------------------------

class TestMainCLI(_MongoBackedTestCase):

    def test_cards_subcommand_embeds_cards(self):
        self.cards.insert_one(
            {"_id": "a", "scryfall_id": "a", "name": "A", "oracle_text": "text"}
        )
        rc, out, _ = _run(["cards"])
        self.assertEqual(rc, 0)
        self.assertIn("text_embedding", self.cards.find_one({"_id": "a"}))
        self.assertIn("wrote text_embedding on 1 cards", out)

    def test_tags_subcommand_embeds_tags(self):
        self.tags.insert_one({"_id": "s", "label": "S"})
        rc, out, _ = _run(["tags"])
        self.assertEqual(rc, 0)
        self.assertIn("embedding", self.tags.find_one({"_id": "s"}))
        self.assertIn("wrote embedding on 1 tags", out)

    def test_cards_refresh_flag_passed_through(self):
        self.cards.insert_one(
            {"_id": "a", "scryfall_id": "a", "name": "A", "oracle_text": "text",
             "text_embedding": [0.0] * 8}
        )
        rc, _, _ = _run(["cards", "--refresh"])
        self.assertEqual(rc, 0)
        # The pre-existing [0,0,...] got overwritten.
        self.assertNotEqual(self.cards.find_one({"_id": "a"})["text_embedding"], [0.0] * 8)

    def test_cards_limit_flag_passed_through(self):
        for i in range(5):
            self.cards.insert_one(
                {"_id": f"c{i}", "scryfall_id": f"c{i}", "name": f"C{i}", "oracle_text": "x"}
            )
        rc, _, _ = _run(["cards", "--limit", "2"])
        self.assertEqual(rc, 0)
        self.assertEqual(self.cards.count_documents({"text_embedding": {"$exists": True}}), 2)

    def test_missing_subcommand_errors(self):
        with self.assertRaises(SystemExit):
            _run([])


if __name__ == "__main__":
    unittest.main()
