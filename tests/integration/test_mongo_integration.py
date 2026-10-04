"""Opt-in integration tests against a real MongoDB cluster.

These tests only run when `MONGODB_INTEGRATION_URI` is set in the
environment; otherwise every class is skipped. Each test class creates
a disposable database (uuid-suffixed) in setUpClass and drops it in
tearDownClass, so running the suite against your real Atlas project
is safe and leaves no residue.

Run against your cluster:

    # bash/zsh:
    MONGODB_INTEGRATION_URI="mongodb+srv://..." python -m unittest discover tests
    # PowerShell:
    $env:MONGODB_INTEGRATION_URI = "mongodb+srv://..."
    python -m unittest discover tests

Without the env var set, these are silent skips — the normal offline
test suite is unaffected.

Scope: connectivity, index creation, round-trip upsert + read on cards,
oracle_tags attach cycle, meta timestamp roundtrip. Not a performance
or stress test.
"""
from __future__ import annotations

import os
import unittest
import uuid

from mtg_recommender import oracle_tags, storage
from mtg_recommender import scryfall_fetch as sf

INTEGRATION_URI = os.environ.get("MONGODB_INTEGRATION_URI")

_SKIP_REASON = "set MONGODB_INTEGRATION_URI to enable integration tests"


class _IntegrationBase(unittest.TestCase):
    """Shared setup: disposable db per class, env restored after tearDown."""

    @classmethod
    def setUpClass(cls):
        if not INTEGRATION_URI:
            raise unittest.SkipTest(_SKIP_REASON)
        # uuid4 keeps parallel runs isolated and makes residue easy to
        # spot in Atlas if a teardown ever fails.
        cls._db_name = f"mtg_recommender_integ_{uuid.uuid4().hex[:8]}"
        cls._saved_env = {
            k: os.environ.get(k) for k in ("MONGODB_URI", "MONGODB_DB")
        }
        os.environ["MONGODB_URI"] = INTEGRATION_URI
        os.environ["MONGODB_DB"] = cls._db_name
        storage.reset_client(None)  # force rebuild under new env

    @classmethod
    def tearDownClass(cls):
        if not INTEGRATION_URI:
            return
        try:
            storage.get_client().drop_database(cls._db_name)
        finally:
            # Always restore env — even if the drop raised — so tests
            # that run after this one aren't polluted.
            for k, v in cls._saved_env.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
            storage.reset_client(None)


@unittest.skipUnless(INTEGRATION_URI, _SKIP_REASON)
class TestConnectivity(_IntegrationBase):

    def test_ping_server(self):
        info = storage.get_client().admin.command("ping")
        # pymongo returns 1.0; some server versions use int 1.
        self.assertEqual(float(info.get("ok", 0)), 1.0)

    def test_server_info_has_version(self):
        info = storage.get_client().server_info()
        self.assertIn("version", info)


@unittest.skipUnless(INTEGRATION_URI, _SKIP_REASON)
class TestIndexes(_IntegrationBase):

    def test_ensure_indexes_creates_expected_indexes(self):
        storage.ensure_indexes()
        info = storage.cards_collection().index_information()
        self.assertIn("names_lookup", info)
        self.assertIn("oracle_id_lookup", info)

    def test_ensure_indexes_is_idempotent(self):
        storage.ensure_indexes()
        storage.ensure_indexes()
        info = storage.cards_collection().index_information()
        self.assertIn("names_lookup", info)


@unittest.skipUnless(INTEGRATION_URI, _SKIP_REASON)
class TestCardRoundtrip(_IntegrationBase):

    def test_upsert_then_find_by_name(self):
        storage.ensure_indexes()
        raw = {
            "id": "integ-bolt",
            "oracle_id": "integ-oracle-bolt",
            "name": "Integration Lightning Bolt",
            "mana_cost": "{R}",
            "type_line": "Instant",
            "oracle_text": "deals 3 damage",
        }
        coll = storage.cards_collection()
        ok = sf.upsert_card(coll, raw, updated_at="2026-10-04T00:00:00+00:00")
        self.assertTrue(ok)
        hits = sf.find_cards_by_name(coll, "Integration Lightning Bolt")
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]["scryfall_id"], "integ-bolt")
        self.assertEqual(hits[0]["oracle_id"], "integ-oracle-bolt")

    def test_upsert_preserves_existing_tags(self):
        # If oracle_tags has already written a `tags` array, a fetcher
        # refetch of the same card must not clobber it.
        storage.ensure_indexes()
        coll = storage.cards_collection()
        coll.insert_one({
            "_id": "integ-wrath", "scryfall_id": "integ-wrath",
            "name": "Integration Wrath",
            "names": ["integration wrath"],
            "updated_at": "2020-01-01T00:00:00+00:00",
            "tags": ["sweeper", "removal-creature"],
        })
        sf.upsert_card(
            coll,
            {"id": "integ-wrath", "oracle_id": "o", "name": "Integration Wrath",
             "mana_cost": "{2}{W}{W}", "type_line": "Sorcery",
             "oracle_text": "Destroy all creatures."},
            updated_at="2026-10-04T00:00:00+00:00",
        )
        got = coll.find_one({"_id": "integ-wrath"})
        self.assertEqual(got["tags"], ["sweeper", "removal-creature"])
        self.assertEqual(got["mana_cost"], "{2}{W}{W}")


@unittest.skipUnless(INTEGRATION_URI, _SKIP_REASON)
class TestOracleTagsAttach(_IntegrationBase):

    def test_attach_sets_tags_from_oracle_id_join(self):
        storage.ensure_indexes()
        cards = storage.cards_collection()
        cards.insert_many([
            {"_id": "c1", "scryfall_id": "c1", "oracle_id": "o-bolt", "name": "C1"},
            {"_id": "c2", "scryfall_id": "c2", "oracle_id": "o-wrath", "name": "C2"},
            {"_id": "c3", "scryfall_id": "c3", "oracle_id": "o-none", "name": "C3"},
        ])
        index = {"o-bolt": ["burn-any", "spot-removal"], "o-wrath": ["sweeper"]}
        matched, empty, unmatched = oracle_tags.attach_tags_to_cards(cards, index)
        self.assertEqual(matched, 2)
        self.assertEqual(empty, 1)
        self.assertEqual(unmatched, 0)
        self.assertEqual(cards.find_one({"_id": "c1"})["tags"], ["burn-any", "spot-removal"])
        self.assertEqual(cards.find_one({"_id": "c2"})["tags"], ["sweeper"])
        self.assertEqual(cards.find_one({"_id": "c3"})["tags"], [])


@unittest.skipUnless(INTEGRATION_URI, _SKIP_REASON)
class TestMetaCollection(_IntegrationBase):

    def test_snapshot_timestamp_roundtrip(self):
        storage.set_snapshot_timestamp("integ_source", "2026-10-04T00:00:00+00:00")
        got = storage.get_snapshot_timestamp("integ_source")
        self.assertEqual(got, "2026-10-04T00:00:00+00:00")

    def test_snapshot_overwrite(self):
        storage.set_snapshot_timestamp("integ_source", "2020-01-01T00:00:00+00:00")
        storage.set_snapshot_timestamp("integ_source", "2026-10-04T00:00:00+00:00")
        got = storage.get_snapshot_timestamp("integ_source")
        self.assertEqual(got, "2026-10-04T00:00:00+00:00")


if __name__ == "__main__":
    unittest.main()
