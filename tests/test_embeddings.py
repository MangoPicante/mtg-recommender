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
    TestFuseHelpers       — _l2_normalize / _aggregate_tag_vector / _fuse_vectors
    TestFuseCardVectors   — fuse loop: happy path, no-tags, dim mismatch,
                            skip-already-fused, --refresh, --limit, alpha extremes
    TestMainCLI           — argparse dispatch (cards / tags / fuse / flags)
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
        """Each row is (card id, oracle_text). Minimal fixture shape."""
        docs = [
            {"_id": sid, "name": sid, "oracle_text": text}
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
            {"_id": "a", "name": "A", "oracle_text": "something"},
            {"_id": "b", "name": "B"},  # no oracle_text at all
            {"_id": "c", "name": "C", "oracle_text": None},
            {"_id": "d", "name": "D", "oracle_text": ""},
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


# ---------------------------------------------------------------------------
# Fuse helpers
# ---------------------------------------------------------------------------

class TestFuseHelpers(unittest.TestCase):

    def test_l2_normalize_returns_unit_vector(self):
        vec = np.array([3.0, 4.0])  # length 5
        out = emb._l2_normalize(vec)
        self.assertAlmostEqual(float(np.linalg.norm(out)), 1.0, places=6)
        np.testing.assert_allclose(out, [0.6, 0.8])

    def test_l2_normalize_zero_vector_returned_unchanged(self):
        # No direction to point in; dividing would NaN. We return the
        # zero vector so the fuse logic can fall back to the other side.
        vec = np.zeros(4)
        out = emb._l2_normalize(vec)
        np.testing.assert_array_equal(out, np.zeros(4))

    def test_aggregate_tag_vector_mean(self):
        tags = {
            "a": np.array([1.0, 0.0, 0.0]),
            "b": np.array([0.0, 2.0, 0.0]),
        }
        out = emb._aggregate_tag_vector(["a", "b"], tags)
        np.testing.assert_allclose(out, [0.5, 1.0, 0.0])

    def test_aggregate_tag_vector_drops_unknown_slugs(self):
        # A card may reference a tag that hasn't been embedded yet.
        tags = {"a": np.array([1.0, 0.0])}
        out = emb._aggregate_tag_vector(["a", "unknown"], tags)
        np.testing.assert_allclose(out, [1.0, 0.0])

    def test_aggregate_tag_vector_returns_none_when_nothing_resolves(self):
        self.assertIsNone(emb._aggregate_tag_vector([], {}))
        self.assertIsNone(emb._aggregate_tag_vector(["ghost"], {"a": np.ones(3)}))

    def test_fuse_vectors_produces_unit_length_result(self):
        text = np.array([1.0, 0.0, 0.0, 0.0])
        tag = np.array([0.0, 1.0, 0.0, 0.0])
        out = emb._fuse_vectors(text, tag, alpha=0.5)
        self.assertAlmostEqual(float(np.linalg.norm(out)), 1.0, places=6)

    def test_fuse_vectors_alpha_one_ignores_tag(self):
        text = np.array([2.0, 0.0])  # not unit — fuse must normalize it
        tag = np.array([0.0, 7.0])
        out = emb._fuse_vectors(text, tag, alpha=1.0)
        np.testing.assert_allclose(out, [1.0, 0.0])

    def test_fuse_vectors_alpha_zero_ignores_text(self):
        text = np.array([5.0, 0.0])
        tag = np.array([0.0, 3.0])
        out = emb._fuse_vectors(text, tag, alpha=0.0)
        np.testing.assert_allclose(out, [0.0, 1.0])

    def test_fuse_vectors_none_tag_returns_normalized_text(self):
        text = np.array([0.0, 4.0, 0.0])
        out = emb._fuse_vectors(text, None, alpha=0.6)
        np.testing.assert_allclose(out, [0.0, 1.0, 0.0])


# ---------------------------------------------------------------------------
# fuse_card_vectors
# ---------------------------------------------------------------------------

class TestFuseCardVectors(_MongoBackedTestCase):
    """Exercise the fuse loop against mongomock-seeded embeddings.

    These tests don't need the FakeEncoder — fuse reads existing
    text_embedding / tag.embedding fields rather than calling an encoder.
    """

    DIM = 4

    def _seed_tag(self, slug: str, vec: list[float]) -> None:
        self.tags.insert_one(
            {"_id": slug, "label": slug.title(), "embedding": vec}
        )

    def _seed_card(self, sid: str, text_vec: list[float], tags: list[str]) -> None:
        self.cards.insert_one(
            {
                "_id": sid, "name": sid,
                "oracle_text": f"text for {sid}",
                "text_embedding": text_vec,
                "tags": tags,
            }
        )

    def test_happy_path_writes_unit_card_vector(self):
        self._seed_tag("a", [0.0, 1.0, 0.0, 0.0])
        self._seed_card("c1", [1.0, 0.0, 0.0, 0.0], ["a"])
        with redirect_stdout(io.StringIO()):
            written = emb.fuse_card_vectors(self.cards, self.tags, alpha=0.5)
        self.assertEqual(written, 1)
        doc = self.cards.find_one({"_id": "c1"})
        self.assertIn("card_vector", doc)
        self.assertEqual(len(doc["card_vector"]), self.DIM)
        self.assertAlmostEqual(
            float(np.linalg.norm(np.array(doc["card_vector"]))), 1.0, places=6
        )
        # BSON needs plain floats, not numpy scalars.
        self.assertIsInstance(doc["card_vector"][0], float)

    def test_card_with_no_tags_fuses_to_normalized_text(self):
        self._seed_card("c1", [0.0, 2.0, 0.0, 0.0], [])
        with redirect_stdout(io.StringIO()):
            emb.fuse_card_vectors(self.cards, self.tags, alpha=0.6)
        got = self.cards.find_one({"_id": "c1"})["card_vector"]
        np.testing.assert_allclose(got, [0.0, 1.0, 0.0, 0.0])

    def test_tags_with_no_embedding_fall_back_to_text(self):
        # Card references a tag slug that has no embedding yet — fuse
        # must silently drop it rather than crash or skip the card.
        self.tags.insert_one({"_id": "a", "label": "A"})  # no embedding
        self._seed_card("c1", [0.0, 0.0, 3.0, 0.0], ["a"])
        with redirect_stdout(io.StringIO()):
            emb.fuse_card_vectors(self.cards, self.tags, alpha=0.5)
        got = self.cards.find_one({"_id": "c1"})["card_vector"]
        np.testing.assert_allclose(got, [0.0, 0.0, 1.0, 0.0])

    def test_dim_mismatch_raises(self):
        # Text is 4-dim, tag is 3-dim — mixed-encoder state.
        self._seed_tag("a", [1.0, 0.0, 0.0])
        self._seed_card("c1", [1.0, 0.0, 0.0, 0.0], ["a"])
        with redirect_stdout(io.StringIO()), self.assertRaises(ValueError) as ctx:
            emb.fuse_card_vectors(self.cards, self.tags)
        self.assertIn("text=4", str(ctx.exception))
        self.assertIn("tag=3", str(ctx.exception))

    def test_default_skips_already_fused(self):
        self._seed_tag("a", [0.0, 1.0, 0.0, 0.0])
        self._seed_card("c1", [1.0, 0.0, 0.0, 0.0], ["a"])
        self.cards.update_one(
            {"_id": "c1"}, {"$set": {"card_vector": [9.0] * self.DIM}}
        )
        with redirect_stdout(io.StringIO()):
            written = emb.fuse_card_vectors(self.cards, self.tags)
        self.assertEqual(written, 0)
        self.assertEqual(self.cards.find_one({"_id": "c1"})["card_vector"], [9.0] * self.DIM)

    def test_refresh_re_fuses(self):
        self._seed_tag("a", [0.0, 1.0, 0.0, 0.0])
        self._seed_card("c1", [1.0, 0.0, 0.0, 0.0], ["a"])
        self.cards.update_one(
            {"_id": "c1"}, {"$set": {"card_vector": [9.0] * self.DIM}}
        )
        with redirect_stdout(io.StringIO()):
            written = emb.fuse_card_vectors(self.cards, self.tags, refresh=True)
        self.assertEqual(written, 1)
        self.assertNotEqual(
            self.cards.find_one({"_id": "c1"})["card_vector"], [9.0] * self.DIM
        )

    def test_limit_caps_work(self):
        self._seed_tag("a", [0.0, 1.0, 0.0, 0.0])
        for i in range(5):
            self._seed_card(f"c{i}", [1.0, 0.0, 0.0, 0.0], ["a"])
        with redirect_stdout(io.StringIO()):
            written = emb.fuse_card_vectors(self.cards, self.tags, limit=2)
        self.assertEqual(written, 2)
        self.assertEqual(
            self.cards.count_documents({"card_vector": {"$exists": True}}), 2
        )

    def test_cards_without_text_embedding_are_skipped(self):
        # Only the embedded card gets a card_vector; the bare one is left alone.
        self._seed_tag("a", [0.0, 1.0, 0.0, 0.0])
        self._seed_card("c1", [1.0, 0.0, 0.0, 0.0], ["a"])
        self.cards.insert_one(
            {"_id": "c2", "name": "c2", "oracle_text": "x", "tags": ["a"]}
        )
        with redirect_stdout(io.StringIO()):
            written = emb.fuse_card_vectors(self.cards, self.tags)
        self.assertEqual(written, 1)
        self.assertIn("card_vector", self.cards.find_one({"_id": "c1"}))
        self.assertNotIn("card_vector", self.cards.find_one({"_id": "c2"}))

    def test_empty_collection_returns_zero(self):
        with redirect_stdout(io.StringIO()) as out:
            written = emb.fuse_card_vectors(self.cards, self.tags)
        self.assertEqual(written, 0)
        self.assertIn("no cards to fuse", out.getvalue())

    def test_alpha_out_of_range_raises(self):
        with self.assertRaises(ValueError):
            emb.fuse_card_vectors(self.cards, self.tags, alpha=1.5)
        with self.assertRaises(ValueError):
            emb.fuse_card_vectors(self.cards, self.tags, alpha=-0.1)


# ---------------------------------------------------------------------------
# main() CLI
# ---------------------------------------------------------------------------

class TestMainCLI(_MongoBackedTestCase):

    def test_cards_subcommand_embeds_cards(self):
        self.cards.insert_one(
            {"_id": "a", "name": "A", "oracle_text": "text"}
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
            {"_id": "a", "name": "A", "oracle_text": "text",
             "text_embedding": [0.0] * 8}
        )
        rc, _, _ = _run(["cards", "--refresh"])
        self.assertEqual(rc, 0)
        # The pre-existing [0,0,...] got overwritten.
        self.assertNotEqual(self.cards.find_one({"_id": "a"})["text_embedding"], [0.0] * 8)

    def test_cards_limit_flag_passed_through(self):
        for i in range(5):
            self.cards.insert_one(
                {"_id": f"c{i}", "name": f"C{i}", "oracle_text": "x"}
            )
        rc, _, _ = _run(["cards", "--limit", "2"])
        self.assertEqual(rc, 0)
        self.assertEqual(self.cards.count_documents({"text_embedding": {"$exists": True}}), 2)

    def test_missing_subcommand_errors(self):
        with self.assertRaises(SystemExit):
            _run([])

    def _seed_fuse_fixture(self) -> None:
        """Minimal 4-dim fixture: one tag, one card with both inputs."""
        self.tags.insert_one(
            {"_id": "a", "label": "A", "embedding": [0.0, 1.0, 0.0, 0.0]}
        )
        self.cards.insert_one(
            {"_id": "c1", "name": "C1",
             "oracle_text": "x",
             "text_embedding": [1.0, 0.0, 0.0, 0.0], "tags": ["a"]}
        )

    def test_fuse_subcommand_writes_card_vector(self):
        self._seed_fuse_fixture()
        rc, out, _ = _run(["fuse"])
        self.assertEqual(rc, 0)
        doc = self.cards.find_one({"_id": "c1"})
        self.assertIn("card_vector", doc)
        self.assertEqual(len(doc["card_vector"]), 4)
        self.assertIn("wrote card_vector on 1 cards", out)

    def test_fuse_alpha_flag_passed_through(self):
        # With alpha=1.0 the tag is ignored → card_vector == normalized text.
        self._seed_fuse_fixture()
        rc, _, _ = _run(["fuse", "--alpha", "1.0"])
        self.assertEqual(rc, 0)
        got = self.cards.find_one({"_id": "c1"})["card_vector"]
        np.testing.assert_allclose(got, [1.0, 0.0, 0.0, 0.0])

    def test_fuse_refresh_flag_passed_through(self):
        self._seed_fuse_fixture()
        self.cards.update_one(
            {"_id": "c1"}, {"$set": {"card_vector": [9.0] * 4}}
        )
        rc, _, _ = _run(["fuse", "--refresh"])
        self.assertEqual(rc, 0)
        self.assertNotEqual(
            self.cards.find_one({"_id": "c1"})["card_vector"], [9.0] * 4
        )


if __name__ == "__main__":
    unittest.main()
