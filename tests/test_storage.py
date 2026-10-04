"""Offline tests for mtg_recommender.storage.

All Mongo operations go through mongomock, an in-memory pymongo-compatible
backend. The storage module exposes `reset_client` so each test can swap
the cached client out for a fresh mongomock instance; no network, no
real Mongo required.

Run with:

    python -m unittest discover tests
    # or, more targeted:
    python -m unittest tests.test_storage

Test classes:

    TestClientCache          get_client / reset_client lazy-init + swap
    TestConfigResolution     env vars override defaults, defaults apply
                             otherwise, missing MONGODB_URI raises
    TestCollectionHandles    cards/tags/meta collections point at the
                             configured names on the configured db
    TestEnsureIndexes        creates the indexes the rest of the package
                             needs; idempotent
    TestMetaHelpers          snapshot timestamp get / set / overwrite
"""
from __future__ import annotations

import os
import unittest
from unittest.mock import patch

import mongomock

from mtg_recommender import storage


class StorageTestCase(unittest.TestCase):
    """Shared setup: fresh mongomock client, env scrubbed to defaults."""

    def setUp(self):
        # Give every test its own mongomock client so collection state
        # doesn't leak between tests. Clearing it in tearDown is also
        # important so the next test doesn't accidentally see a prior
        # test's cached client.
        self._client = mongomock.MongoClient()
        storage.reset_client(self._client)
        # Scrub any env values that might leak in from a developer's
        # shell (MONGODB_URI etc). patch.dict with clear=False lets us
        # delete specific keys without wiping the whole environment,
        # but it's simpler to use patch.dict with a copy and remove
        # the ones we care about.
        self._env_patch = patch.dict(
            os.environ,
            {k: v for k, v in os.environ.items()
             if not k.startswith("MONGODB_")},
            clear=True,
        )
        self._env_patch.start()
        # Re-insert PATH etc. for subprocess-ish code paths; we only
        # wanted to drop MONGODB_* above. Simpler to just set what we
        # need per-test instead.

    def tearDown(self):
        self._env_patch.stop()
        storage.reset_client(None)


# ---------------------------------------------------------------------------
# get_client / reset_client
# ---------------------------------------------------------------------------

class TestClientCache(StorageTestCase):

    def test_reset_installs_the_given_client(self):
        self.assertIs(storage.get_client(), self._client)

    def test_reset_with_none_clears_the_cache(self):
        storage.reset_client(None)
        # Now a call to get_client() would try to build a real one,
        # which requires MONGODB_URI. We don't set it, so this should
        # raise rather than silently hitting the network.
        with self.assertRaises(RuntimeError):
            storage.get_client()

    def test_lazy_init_only_builds_once(self):
        storage.reset_client(None)
        with patch.dict(os.environ, {"MONGODB_URI": "mongodb://unused/"}), \
             patch("mtg_recommender.storage.MongoClient") as MC:
            MC.return_value = mongomock.MongoClient()
            first = storage.get_client()
            second = storage.get_client()
        # Second call must hit the cache, not re-build.
        self.assertIs(first, second)
        self.assertEqual(MC.call_count, 1)


# ---------------------------------------------------------------------------
# URI + db + collection name resolution
# ---------------------------------------------------------------------------

class TestConfigResolution(StorageTestCase):

    def test_missing_uri_raises_with_actionable_message(self):
        storage.reset_client(None)
        with self.assertRaises(RuntimeError) as ctx:
            storage.get_client()
        # The message must mention the env var name AND the .env file,
        # which are the two places a user would look.
        self.assertIn("MONGODB_URI", str(ctx.exception))
        self.assertIn(".env", str(ctx.exception))

    def test_default_db_name_applies_when_env_absent(self):
        self.assertEqual(storage.get_database().name, storage.DEFAULT_DB)

    def test_db_name_env_override(self):
        with patch.dict(os.environ, {"MONGODB_DB": "custom_db"}):
            self.assertEqual(storage.get_database().name, "custom_db")

    def test_explicit_db_name_arg_overrides_env(self):
        with patch.dict(os.environ, {"MONGODB_DB": "env_name"}):
            self.assertEqual(storage.get_database("explicit").name, "explicit")


# ---------------------------------------------------------------------------
# cards_collection / tags_collection / meta_collection
# ---------------------------------------------------------------------------

class TestCollectionHandles(StorageTestCase):

    def test_default_collection_names(self):
        self.assertEqual(storage.cards_collection().name, storage.DEFAULT_CARDS_COLLECTION)
        self.assertEqual(storage.tags_collection().name, storage.DEFAULT_TAGS_COLLECTION)
        self.assertEqual(storage.meta_collection().name, storage.DEFAULT_META_COLLECTION)

    def test_env_overrides_cards_collection(self):
        with patch.dict(os.environ, {"MONGODB_CARDS_COLLECTION": "cards_v2"}):
            self.assertEqual(storage.cards_collection().name, "cards_v2")

    def test_collections_live_on_the_configured_db(self):
        with patch.dict(os.environ, {"MONGODB_DB": "x"}):
            self.assertEqual(storage.cards_collection().database.name, "x")

    def test_explicit_db_passed_through_to_collection(self):
        other = storage.get_client()["elsewhere"]
        self.assertEqual(storage.cards_collection(other).database.name, "elsewhere")


# ---------------------------------------------------------------------------
# ensure_indexes
# ---------------------------------------------------------------------------

class TestEnsureIndexes(StorageTestCase):

    def test_creates_names_and_oracle_id_indexes(self):
        storage.ensure_indexes()
        cards = storage.cards_collection()
        # Mongo (and mongomock) returns an index_information() dict
        # keyed by index name. _id_ is always present; we care that
        # our two named indexes show up alongside it.
        info = cards.index_information()
        self.assertIn("names_lookup", info)
        self.assertIn("oracle_id_lookup", info)

    def test_idempotent(self):
        # Calling twice must not raise. mongomock mirrors pymongo's
        # behaviour here: a second create_index on an identical spec
        # is a no-op.
        storage.ensure_indexes()
        storage.ensure_indexes()
        info = storage.cards_collection().index_information()
        self.assertIn("names_lookup", info)


# ---------------------------------------------------------------------------
# Meta helpers (snapshot timestamps)
# ---------------------------------------------------------------------------

class TestMetaHelpers(StorageTestCase):

    def test_get_returns_none_before_any_set(self):
        self.assertIsNone(storage.get_snapshot_timestamp("oracle_tags"))

    def test_set_then_get_roundtrips(self):
        storage.set_snapshot_timestamp("oracle_tags", "2026-10-03T21:00:32.494+00:00")
        self.assertEqual(
            storage.get_snapshot_timestamp("oracle_tags"),
            "2026-10-03T21:00:32.494+00:00",
        )

    def test_set_overwrites_previous_value(self):
        storage.set_snapshot_timestamp("oracle_tags", "2020-01-01T00:00:00+00:00")
        storage.set_snapshot_timestamp("oracle_tags", "2026-10-03T21:00:32.494+00:00")
        self.assertEqual(
            storage.get_snapshot_timestamp("oracle_tags"),
            "2026-10-03T21:00:32.494+00:00",
        )

    def test_sources_are_isolated(self):
        storage.set_snapshot_timestamp("oracle_tags", "ts-tags")
        storage.set_snapshot_timestamp("oracle_cards", "ts-cards")
        self.assertEqual(storage.get_snapshot_timestamp("oracle_tags"), "ts-tags")
        self.assertEqual(storage.get_snapshot_timestamp("oracle_cards"), "ts-cards")


if __name__ == "__main__":
    unittest.main()
