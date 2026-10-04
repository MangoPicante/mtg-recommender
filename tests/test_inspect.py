"""Offline tests for mtg_recommender.inspect.

mongomock backs every DB call; stdout is captured so we can assert on
the human-readable output.

    TestFmtCard       card formatting edge cases
    TestFmtTag        tag formatting
    TestCmdCard       card subcommand: hit, miss, ambiguity
    TestCmdTag        tag subcommand: hit, miss, counts cards
    TestCmdList       list subcommand: all, --tag filter, empty
    TestCmdStats      stats subcommand: numbers + top-tag aggregation
    TestMainDispatch  argparse wiring: each subcommand reaches its handler
"""
from __future__ import annotations

import io
import unittest
from contextlib import redirect_stderr, redirect_stdout

import mongomock

from mtg_recommender import inspect as mi
from mtg_recommender import storage


def _run(argv: list[str]) -> tuple[int, str, str]:
    """Invoke inspect.main with argv, returning (rc, stdout, stderr)."""
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        rc = mi.main(argv)
    return rc, out.getvalue(), err.getvalue()


class _MongoBackedTestCase(unittest.TestCase):
    def setUp(self):
        storage.reset_client(mongomock.MongoClient())
        storage.ensure_indexes()
        self.cards = storage.cards_collection()
        self.tags = storage.tags_collection()

    def tearDown(self):
        storage.reset_client(None)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

WRATH = {
    "_id": "id-wrath", "scryfall_id": "id-wrath", "oracle_id": "oracle-wrath",
    "name": "Wrath of God", "names": ["wrath of god"],
    "mana_cost": "{2}{W}{W}", "type_line": "Sorcery",
    "oracle_text": "Destroy all creatures. They can't be regenerated.",
    "tags": ["removal-creature", "sweeper"],
    "updated_at": "2026-10-03T21:00:00+00:00",
}

BOLT = {
    "_id": "id-bolt", "scryfall_id": "id-bolt", "oracle_id": "oracle-bolt",
    "name": "Lightning Bolt", "names": ["lightning bolt"],
    "mana_cost": "{R}", "type_line": "Instant",
    "oracle_text": "Lightning Bolt deals 3 damage to any target.",
    "tags": ["burn-any", "spot-removal"],
    "updated_at": "2026-10-03T21:00:00+00:00",
}

BOLT_ART = {
    "_id": "id-bolt-art", "scryfall_id": "id-bolt-art", "oracle_id": "oracle-bolt-art",
    "name": "Lightning Bolt // Lightning Bolt",
    "names": ["lightning bolt // lightning bolt", "lightning bolt"],
    "mana_cost": None, "type_line": None, "oracle_text": None,
    "tags": [],
    "updated_at": "2026-10-03T21:00:00+00:00",
}

TAG_SPOT = {
    "_id": "spot-removal", "scryfall_tag_id": "u-spot",
    "label": "Spot removal", "description": "Removes a single permanent.",
    "parent_slugs": ["removal"], "child_slugs": ["doom-blade"],
    "aliases": ["targeted-removal"],
}


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

class TestFmtCard(unittest.TestCase):

    def test_full_card_shows_name_cost_type_oracle_tags(self):
        out = mi._fmt_card(WRATH)
        self.assertIn("Wrath of God", out)
        self.assertIn("{2}{W}{W}", out)
        self.assertIn("Sorcery", out)
        self.assertIn("Destroy all creatures", out)
        self.assertIn("removal-creature, sweeper", out)

    def test_missing_tags_renders_none_label(self):
        out = mi._fmt_card(BOLT_ART)
        self.assertIn("tags: (none)", out)

    def test_missing_oracle_text_skipped(self):
        # Should not raise, should not print an empty oracle line.
        out = mi._fmt_card(BOLT_ART)
        # No stray colon-after-blank-space from an empty oracle text.
        self.assertNotIn("    \n", out)


class TestFmtTag(unittest.TestCase):

    def test_full_tag_doc(self):
        out = mi._fmt_tag(TAG_SPOT, card_count=42)
        self.assertIn("spot-removal", out)
        self.assertIn("Spot removal", out)
        self.assertIn("Removes a single permanent", out)
        self.assertIn("parents     : removal", out)
        self.assertIn("children    : doom-blade", out)
        self.assertIn("aliases     : targeted-removal", out)
        self.assertIn("cards tagged: 42", out)

    def test_missing_description_shows_placeholder(self):
        tag = dict(TAG_SPOT, description=None)
        out = mi._fmt_tag(tag, 0)
        self.assertIn("(no description)", out)

    def test_empty_relations_show_none(self):
        tag = dict(TAG_SPOT, parent_slugs=[], child_slugs=[], aliases=[])
        out = mi._fmt_tag(tag, 0)
        self.assertIn("parents     : (none)", out)
        self.assertIn("children    : (none)", out)
        self.assertIn("aliases     : (none)", out)


# ---------------------------------------------------------------------------
# `mtg-inspect card`
# ---------------------------------------------------------------------------

class TestCmdCard(_MongoBackedTestCase):

    def test_hit_prints_card_and_returns_zero(self):
        self.cards.insert_one(WRATH)
        rc, out, err = _run(["card", "Wrath of God"])
        self.assertEqual(rc, 0)
        self.assertIn("Wrath of God", out)
        self.assertIn("Sorcery", out)
        self.assertEqual(err, "")

    def test_case_insensitive(self):
        self.cards.insert_one(WRATH)
        rc, out, _ = _run(["card", "wRaTh Of GoD"])
        self.assertEqual(rc, 0)
        self.assertIn("Wrath of God", out)

    def test_miss_returns_one_and_writes_stderr(self):
        rc, out, err = _run(["card", "Nonexistent"])
        self.assertEqual(rc, 1)
        self.assertEqual(out, "")
        self.assertIn("not found", err)

    def test_ambiguous_prints_all_matches(self):
        self.cards.insert_many([BOLT, BOLT_ART])
        rc, out, _ = _run(["card", "Lightning Bolt"])
        self.assertEqual(rc, 0)
        self.assertIn("2 matches", out)
        self.assertIn("id-bolt", out)
        self.assertIn("id-bolt-art", out)


# ---------------------------------------------------------------------------
# `mtg-inspect tag`
# ---------------------------------------------------------------------------

class TestCmdTag(_MongoBackedTestCase):

    def test_hit_shows_catalog_and_card_count(self):
        self.tags.insert_one(TAG_SPOT)
        self.cards.insert_many([BOLT, dict(BOLT, _id="other", scryfall_id="other", names=["other"])])
        rc, out, err = _run(["tag", "spot-removal"])
        self.assertEqual(rc, 0)
        self.assertIn("spot-removal", out)
        self.assertIn("cards tagged: 2", out)
        self.assertEqual(err, "")

    def test_miss_returns_one(self):
        rc, out, err = _run(["tag", "nonexistent"])
        self.assertEqual(rc, 1)
        self.assertEqual(out, "")
        self.assertIn("not found", err)

    def test_zero_card_count_shown(self):
        self.tags.insert_one(TAG_SPOT)
        rc, out, _ = _run(["tag", "spot-removal"])
        self.assertEqual(rc, 0)
        self.assertIn("cards tagged: 0", out)


# ---------------------------------------------------------------------------
# `mtg-inspect list`
# ---------------------------------------------------------------------------

class TestCmdList(_MongoBackedTestCase):

    def test_list_all_sorted_by_name(self):
        self.cards.insert_many([WRATH, BOLT])
        rc, out, _ = _run(["list"])
        self.assertEqual(rc, 0)
        # Lightning Bolt sorts before Wrath of God.
        self.assertLess(out.index("Lightning Bolt"), out.index("Wrath of God"))

    def test_filter_by_tag(self):
        self.cards.insert_many([WRATH, BOLT])
        rc, out, _ = _run(["list", "--tag", "sweeper"])
        self.assertEqual(rc, 0)
        self.assertIn("Wrath of God", out)
        self.assertNotIn("Lightning Bolt", out)

    def test_limit_caps_output_rows(self):
        self.cards.insert_many([WRATH, BOLT])
        rc, out, _ = _run(["list", "--limit", "1"])
        self.assertEqual(rc, 0)
        # Only one card name should appear; and the header mentions "1 of 2".
        self.assertIn("showing 1 of 2", out)

    def test_empty_collection_returns_one(self):
        rc, out, err = _run(["list"])
        self.assertEqual(rc, 1)
        self.assertEqual(out, "")
        self.assertIn("no cards", err)

    def test_tag_with_no_matches_returns_one(self):
        self.cards.insert_one(WRATH)
        rc, _, err = _run(["list", "--tag", "nonexistent"])
        self.assertEqual(rc, 1)
        self.assertIn("no cards tagged", err)


# ---------------------------------------------------------------------------
# `mtg-inspect stats`
# ---------------------------------------------------------------------------

class TestCmdStats(_MongoBackedTestCase):

    def test_counts_cards_tags_oracle_id_coverage(self):
        self.cards.insert_many([WRATH, BOLT, BOLT_ART])  # BOLT_ART has tags=[]
        self.tags.insert_one(TAG_SPOT)
        rc, out, _ = _run(["stats"])
        self.assertEqual(rc, 0)
        # 3 cards total; 2 tagged (WRATH + BOLT); BOLT_ART has empty tags.
        self.assertIn("total          : 3", out)
        self.assertIn("with tags      : 2", out)
        # 1 tag in catalog.
        self.assertIn("total          : 1", out)

    def test_top_tags_aggregation(self):
        # 3 cards all tagged "spot-removal", 1 of them also "burn-any".
        docs = [
            dict(BOLT, _id="a", scryfall_id="a", names=["a"], tags=["spot-removal", "burn-any"]),
            dict(BOLT, _id="b", scryfall_id="b", names=["b"], tags=["spot-removal"]),
            dict(BOLT, _id="c", scryfall_id="c", names=["c"], tags=["spot-removal"]),
        ]
        self.cards.insert_many(docs)
        rc, out, _ = _run(["stats", "--top", "5"])
        self.assertEqual(rc, 0)
        # spot-removal has 3 taggings, burn-any has 1.
        spot_idx = out.index("spot-removal")
        burn_idx = out.index("burn-any")
        self.assertLess(spot_idx, burn_idx)  # top by count comes first
        self.assertIn("spot-removal", out)
        self.assertIn("burn-any", out)

    def test_snapshot_timestamp_reported(self):
        storage.set_snapshot_timestamp("oracle_tags", "2026-10-03T21:00:32.494+00:00")
        rc, out, _ = _run(["stats"])
        self.assertEqual(rc, 0)
        self.assertIn("2026-10-03T21:00:32.494+00:00", out)

    def test_missing_snapshot_shows_placeholder(self):
        rc, out, _ = _run(["stats"])
        self.assertEqual(rc, 0)
        self.assertIn("(not set)", out)


# ---------------------------------------------------------------------------
# main() argparse dispatch
# ---------------------------------------------------------------------------

class TestMainDispatch(_MongoBackedTestCase):

    def test_missing_subcommand_errors(self):
        # argparse exits with SystemExit(2) when a required subparser isn't given.
        with self.assertRaises(SystemExit):
            _run([])

    def test_unknown_subcommand_errors(self):
        with self.assertRaises(SystemExit):
            _run(["bogus"])

    def test_help_renders_without_formatting_errors(self):
        # Regression: argparse runs help strings through %-formatting, so a
        # bare `%` in a subparser help raises ValueError at render time.
        # SystemExit with code 0 means --help printed cleanly.
        with self.assertRaises(SystemExit) as ctx:
            _run(["--help"])
        self.assertEqual(ctx.exception.code, 0)


if __name__ == "__main__":
    unittest.main()
