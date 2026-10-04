"""Offline unit tests for mtg_recommender.check.

mongomock covers the happy path; failures are driven by patching the
storage client to raise. No real Mongo required.
"""
from __future__ import annotations

import io
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import patch

import mongomock
from pymongo.errors import ServerSelectionTimeoutError

from mtg_recommender import check as hc
from mtg_recommender import storage


def _run(argv: list[str]) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        rc = hc.main(argv)
    return rc, out.getvalue(), err.getvalue()


class _MongoBackedTestCase(unittest.TestCase):
    def setUp(self):
        storage.reset_client(mongomock.MongoClient())
        storage.ensure_indexes()

    def tearDown(self):
        storage.reset_client(None)


# ---------------------------------------------------------------------------
# Individual check functions
# ---------------------------------------------------------------------------

class TestCheckConnectivity(_MongoBackedTestCase):

    def test_passes_against_live_client(self):
        ok, detail = hc.check_connectivity()
        self.assertTrue(ok)
        self.assertEqual(detail, "")

    def test_verbose_includes_server_and_pymongo_version(self):
        ok, detail = hc.check_connectivity(verbose=True)
        self.assertTrue(ok)
        self.assertIn("pymongo", detail)
        # mongomock returns a dict with a `version` key from server_info().
        self.assertIn("server", detail)

    def test_missing_uri_fails_with_runtime_error(self):
        # Clear the mongomock client so get_client() falls back to the
        # real path, which raises RuntimeError when MONGODB_URI is unset.
        storage.reset_client(None)
        with patch.dict("os.environ", {}, clear=False):
            # Strip any pre-existing MONGODB_URI to force the error.
            import os
            os.environ.pop("MONGODB_URI", None)
            ok, detail = hc.check_connectivity()
        self.assertFalse(ok)
        self.assertIn("MONGODB_URI", detail)

    def test_server_timeout_fails_cleanly(self):
        with patch.object(
            storage.get_client().admin, "command",
            side_effect=ServerSelectionTimeoutError("no servers"),
        ):
            ok, detail = hc.check_connectivity()
        self.assertFalse(ok)
        self.assertIn("ServerSelectionTimeoutError", detail)


class TestCheckDatabase(_MongoBackedTestCase):

    def test_reports_db_name_on_success(self):
        ok, detail = hc.check_database()
        self.assertTrue(ok)
        self.assertEqual(detail, storage.DEFAULT_DB)


class TestCheckIndexes(_MongoBackedTestCase):

    def test_passes_when_indexes_present(self):
        ok, detail = hc.check_indexes()
        self.assertTrue(ok)
        self.assertIn("names_lookup", detail)
        self.assertIn("oracle_id_lookup", detail)

    def test_fails_when_an_index_is_missing(self):
        # Drop one of the required indexes to simulate a missed migration.
        storage.cards_collection().drop_index("names_lookup")
        ok, detail = hc.check_indexes()
        self.assertFalse(ok)
        self.assertIn("names_lookup", detail)


class TestCheckCounts(_MongoBackedTestCase):

    def test_reports_counts_for_each_collection(self):
        storage.cards_collection().insert_many([{"_id": "a"}, {"_id": "b"}])
        storage.tags_collection().insert_one({"_id": "t"})
        storage.set_meta_value("oracle_tags", "2026-10-04T00:00:00+00:00")
        ok, rows = hc.check_counts()
        self.assertTrue(ok)
        by_name = dict(rows)
        self.assertEqual(by_name[storage.DEFAULT_CARDS_COLLECTION], 2)
        self.assertEqual(by_name[storage.DEFAULT_TAGS_COLLECTION], 1)
        self.assertEqual(by_name[storage.DEFAULT_META_COLLECTION], 1)

    def test_zero_counts_are_fine(self):
        # Empty collections should still report OK with count 0 — brand-
        # new deployment case.
        ok, rows = hc.check_counts()
        self.assertTrue(ok)
        self.assertTrue(all(count == 0 for _, count in rows))


# ---------------------------------------------------------------------------
# Orchestrator (run_checks + main)
# ---------------------------------------------------------------------------

class TestRunChecks(_MongoBackedTestCase):

    def test_all_pass_returns_zero(self):
        rc, out, _ = _run([])
        self.assertEqual(rc, 0)
        self.assertIn("all checks passed", out)
        # No FAIL rows:
        self.assertNotIn("FAIL", out)

    def test_connectivity_fail_skips_downstream_and_returns_one(self):
        # Simulate a connection timeout for the ping call.
        with patch.object(
            storage.get_client().admin, "command",
            side_effect=ServerSelectionTimeoutError("no servers"),
        ):
            rc, out, _ = _run([])
        self.assertEqual(rc, 1)
        # Connectivity failed, so Database/Indexes/Collections should SKIP.
        self.assertIn("Connectivity", out)
        self.assertIn("FAIL", out)
        self.assertIn("SKIP", out)
        self.assertIn("one or more checks failed", out)

    def test_missing_index_fails_overall_but_other_checks_still_run(self):
        storage.cards_collection().drop_index("names_lookup")
        rc, out, _ = _run([])
        self.assertEqual(rc, 1)
        # Indexes row says FAIL; Collections still gets a row (not SKIP).
        self.assertIn("Indexes", out)
        self.assertIn("FAIL", out)
        self.assertIn("Collections", out)

    def test_verbose_flag_passes_through_to_connectivity(self):
        rc, out, _ = _run(["--verbose"])
        self.assertEqual(rc, 0)
        self.assertIn("pymongo", out)

    def test_output_has_header_and_divider(self):
        _, out, _ = _run([])
        self.assertIn("MongoDB health check", out)
        self.assertIn("─" * 10, out)  # part of the divider


if __name__ == "__main__":
    unittest.main()
