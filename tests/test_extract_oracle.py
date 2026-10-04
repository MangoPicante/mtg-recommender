"""Offline tests for mtg_recommender.extract_oracle.

mongomock stands in for Mongo; the file-system side writes to a
TemporaryDirectory. Nothing hits the network or a real cluster.
"""
from __future__ import annotations

import io
import json
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import mongomock

from mtg_recommender import extract_oracle as eo
from mtg_recommender import storage


def _run(argv: list[str]) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        rc = eo.main()
    return rc, out.getvalue(), err.getvalue()


def _run_with_argv(argv: list[str]) -> tuple[int, str, str]:
    with patch("sys.argv", ["extract-oracle", *argv]):
        return _run(argv)


BOLT = {
    "_id": "id-bolt", "scryfall_id": "id-bolt", "oracle_id": "oracle-bolt",
    "name": "Lightning Bolt", "names": ["lightning bolt"],
    "mana_cost": "{R}", "type_line": "Instant",
    "oracle_text": "Lightning Bolt deals 3 damage to any target.",
    "tags": ["spot-removal"], "updated_at": "2026-10-04T00:00:00+00:00",
}

WRATH = {
    "_id": "id-wrath", "scryfall_id": "id-wrath", "oracle_id": "oracle-wrath",
    "name": "Wrath of God", "names": ["wrath of god"],
    "mana_cost": "{2}{W}{W}", "type_line": "Sorcery",
    "oracle_text": "Destroy all creatures. They can't be regenerated.",
    "tags": ["sweeper"], "updated_at": "2026-10-04T00:00:00+00:00",
}

# An art-card variant whose names array collides with the normal Bolt
# printing — used to exercise the ambiguity branch.
BOLT_ART = {
    "_id": "id-bolt-art", "scryfall_id": "id-bolt-art", "oracle_id": "oracle-bolt-art",
    "name": "Lightning Bolt // Lightning Bolt",
    "names": ["lightning bolt // lightning bolt", "lightning bolt"],
    "mana_cost": None, "type_line": None, "oracle_text": None,
    "tags": [], "updated_at": "2026-10-04T00:00:00+00:00",
}


class _MongoBackedTestCase(unittest.TestCase):
    def setUp(self):
        storage.reset_client(mongomock.MongoClient())
        storage.ensure_indexes()
        self.cards = storage.cards_collection()

    def tearDown(self):
        storage.reset_client(None)


class TestProjectForOutput(unittest.TestCase):

    def test_keeps_only_the_downstream_fields(self):
        projected = eo.project_for_output(BOLT)
        self.assertEqual(
            set(projected.keys()),
            {"name", "mana_cost", "type_line", "oracle_text", "scryfall_id"},
        )
        self.assertEqual(projected["name"], "Lightning Bolt")

    def test_drops_mongo_bookkeeping_and_tags(self):
        projected = eo.project_for_output(BOLT)
        for k in ("_id", "names", "oracle_id", "tags", "updated_at"):
            self.assertNotIn(k, projected)

    def test_missing_fields_default_to_none(self):
        projected = eo.project_for_output({})
        self.assertEqual(set(projected.values()), {None})


class TestMainCLI(_MongoBackedTestCase):

    def test_no_names_errors(self):
        self.cards.insert_one(BOLT)
        with self.assertRaises(SystemExit):
            _run_with_argv([])

    def test_empty_collection_errors(self):
        # Collection has no docs — a setup error, not a silent empty output.
        with self.assertRaises(SystemExit):
            _run_with_argv(["Lightning Bolt"])

    def test_happy_path_writes_json_subset(self):
        self.cards.insert_many([BOLT, WRATH])
        with TemporaryDirectory() as td:
            out_path = Path(td) / "subset.json"
            argv = ["Lightning Bolt", "Wrath of God", "-o", str(out_path)]
            rc, _, _ = _run_with_argv(argv)
            self.assertEqual(rc, 0)
            data = json.loads(out_path.read_text(encoding="utf-8"))
        self.assertIn("Lightning Bolt", data)
        self.assertIn("Wrath of God", data)
        self.assertEqual(len(data["Lightning Bolt"]), 1)
        self.assertEqual(data["Lightning Bolt"][0]["name"], "Lightning Bolt")

    def test_missing_name_exits_nonzero_but_writes_file(self):
        # The found cards still make it to disk; the missing one gets
        # logged to stderr; main() returns 1 so pipelines notice.
        self.cards.insert_one(BOLT)
        with TemporaryDirectory() as td:
            out_path = Path(td) / "subset.json"
            argv = ["Lightning Bolt", "Nonexistent", "-o", str(out_path)]
            rc, _, err = _run_with_argv(argv)
            self.assertEqual(rc, 1)
            self.assertIn("Nonexistent", err)
            data = json.loads(out_path.read_text(encoding="utf-8"))
        self.assertIn("Lightning Bolt", data)
        self.assertNotIn("Nonexistent", data)

    def test_ambiguous_name_reports_match_count(self):
        # Two cards share the lowered "lightning bolt" alias; both land
        # under the user-supplied key and the ambiguity surfaces on stderr.
        self.cards.insert_many([BOLT, BOLT_ART])
        with TemporaryDirectory() as td:
            out_path = Path(td) / "subset.json"
            argv = ["Lightning Bolt", "-o", str(out_path)]
            rc, _, err = _run_with_argv(argv)
            self.assertEqual(rc, 0)
            data = json.loads(out_path.read_text(encoding="utf-8"))
        self.assertEqual(len(data["Lightning Bolt"]), 2)
        self.assertIn("multiple cached cards", err)

    def test_output_parent_directory_is_created(self):
        # mkdir(parents=True) under the hood — writing to a nested path
        # whose parent doesn't exist yet should work.
        self.cards.insert_one(BOLT)
        with TemporaryDirectory() as td:
            out_path = Path(td) / "nested" / "deeper" / "subset.json"
            argv = ["Lightning Bolt", "-o", str(out_path)]
            rc, _, _ = _run_with_argv(argv)
            self.assertEqual(rc, 0)
            self.assertTrue(out_path.exists())


if __name__ == "__main__":
    unittest.main()
