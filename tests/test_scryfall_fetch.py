"""Repeatable, offline tests for scryfall_fetch.py.

Every HTTP call is mocked and every Mongo op routes through mongomock,
so the suite is deterministic and takes well under a second. No network
access, no real Mongo required.

Run with:

    python -m unittest discover tests
    # or, more targeted:
    python -m unittest tests.test_scryfall_fetch

Test classes are grouped by concern so a failure narrows the search:

    TestCardProjection          extract_card_fields + build_names
    TestUpsertCard              upsert_card / bulk_upsert_cards + $set payload
    TestFindCardsByName         find_cards_by_name via the names index
    TestHasStaleCards           freshness comparison + precision quirks
    TestReadNames               --file + positional arg parsing (pure; still
                                 used by extract_oracle)
    TestGetBulkOracleMetadata   /bulk-data response filtering
    TestDownloadBulkOracleCards gzip detection + JSONL parsing
    TestBulkMode                run_bulk_mode: every download trigger plus the
                                 meta-based fast-path skip and the pre-meta
                                 cluster upgrade path
"""
from __future__ import annotations

import argparse
import gzip
import io
import json
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

import mongomock

from mtg_recommender import scryfall_fetch as sf
from mtg_recommender import storage

# ---------------------------------------------------------------------------
# Fixture data
# ---------------------------------------------------------------------------

# ISO 8601 timestamps chosen so LATER > MIDDLE > EARLIER by both string
# and datetime comparison. Used across freshness tests.
EARLIER = "2020-01-01T00:00:00+00:00"
MIDDLE = "2024-06-15T12:00:00+00:00"
LATER = "2026-07-31T09:03:40.749+00:00"

# Minimal raw Scryfall card objects, in the shape /cards/named and the
# oracle_cards bulk file both return. Kept small so assertions stay
# readable.

LIGHTNING_BOLT_RAW = {
    "id": "id-lb",
    "oracle_id": "oracle-lb",
    "name": "Lightning Bolt",
    "mana_cost": "{R}",
    "type_line": "Instant",
    "oracle_text": "Lightning Bolt deals 3 damage to any target.",
}

SOL_RING_RAW = {
    "id": "id-sol-ring",
    "oracle_id": "oracle-sol-ring",
    "name": "Sol Ring",
    "mana_cost": "{1}",
    "type_line": "Artifact",
    "oracle_text": "{T}: Add {C}{C}.",
}

# A real double-faced card: primary name is the combined form, per-face
# names live under card_faces.
DELVER_RAW = {
    "id": "id-delver",
    "oracle_id": "oracle-delver",
    "name": "Delver of Secrets // Insectile Aberration",
    "mana_cost": "{U}",
    "type_line": "Creature — Human Wizard // Creature — Human Insect",
    "card_faces": [
        {"name": "Delver of Secrets", "oracle_text": "Front text"},
        {"name": "Insectile Aberration", "oracle_text": "Flying"},
    ],
}

# A modal double-faced card (MDFC) — Scryfall gives us null for the
# top-level oracle_text and mana_cost; only card_faces[*] has the real
# per-face data. Used to verify the per-face fallback in
# extract_card_fields.
BALA_GED_MDFC_RAW = {
    "id": "id-bala-ged",
    "oracle_id": "oracle-bala-ged",
    "name": "Bala Ged Recovery // Bala Ged Sanctuary",
    "mana_cost": None,
    "type_line": "Sorcery // Land",
    "oracle_text": None,
    "card_faces": [
        {
            "name": "Bala Ged Recovery",
            "mana_cost": "{2}{G}",
            "type_line": "Sorcery",
            "oracle_text": "Return target card from your graveyard to your hand.",
        },
        {
            "name": "Bala Ged Sanctuary",
            "mana_cost": "",
            "type_line": "Land",
            "oracle_text": "This land enters tapped.\n{T}: Add {G}.",
        },
    ],
}


# An art-card variant whose faces both share a name. Used to exercise
# alias-collision cases that motivated the multikey names index.
DELVER_ART_RAW = {
    "id": "id-delver-art",
    "oracle_id": "oracle-delver-art",
    "name": "Delver of Secrets // Delver of Secrets",
    "mana_cost": None,
    "type_line": None,
    "card_faces": [
        {"name": "Delver of Secrets", "oracle_text": "art front"},
        {"name": "Delver of Secrets", "oracle_text": "art back"},
    ],
}


def call_silent(fn, *args, **kwargs):
    """Invoke fn with stdout/stderr swallowed. Returns fn's return value.

    scryfall_fetch's mode functions print progress messages that would
    otherwise clutter test output. We're asserting on Mongo state and
    return values, not on printed text, so silencing keeps the test
    output clean.
    """
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        return fn(*args, **kwargs)


class _MongoBackedTestCase(unittest.TestCase):
    """Shared setup: fresh mongomock client per test, indexes pre-created."""

    def setUp(self):
        storage.reset_client(mongomock.MongoClient())
        storage.ensure_indexes()
        self.coll = storage.cards_collection()

    def tearDown(self):
        storage.reset_client(None)


# ---------------------------------------------------------------------------
# Pure card projection (extract_card_fields, build_names)
# ---------------------------------------------------------------------------

class TestCardProjection(unittest.TestCase):

    def test_single_face_card(self):
        result = sf.extract_card_fields(LIGHTNING_BOLT_RAW, updated_at=LATER)
        self.assertEqual(result["name"], "Lightning Bolt")
        self.assertEqual(result["mana_cost"], "{R}")
        self.assertEqual(result["oracle_text"], LIGHTNING_BOLT_RAW["oracle_text"])
        self.assertEqual(result["scryfall_id"], "id-lb")
        self.assertEqual(result["_id"], "id-lb")
        self.assertEqual(result["updated_at"], LATER)

    def test_names_contains_lowered_main_name(self):
        result = sf.extract_card_fields(LIGHTNING_BOLT_RAW, updated_at=LATER)
        self.assertEqual(result["names"], ["lightning bolt"])

    def test_names_contains_combined_and_face_names_for_dfc(self):
        result = sf.extract_card_fields(DELVER_RAW, updated_at=LATER)
        # Order: combined name first, then face names left-to-right.
        self.assertEqual(
            result["names"],
            [
                "delver of secrets // insectile aberration",
                "delver of secrets",
                "insectile aberration",
            ],
        )

    def test_names_dedupes_when_faces_share_name(self):
        # Art-card variant whose two faces are both "Delver of Secrets"
        # must contribute a single "delver of secrets" to the names array.
        result = sf.extract_card_fields(DELVER_ART_RAW, updated_at=LATER)
        self.assertEqual(
            result["names"],
            [
                "delver of secrets // delver of secrets",
                "delver of secrets",
            ],
        )

    def test_multifaced_card_missing_top_level_oracle_joins_faces(self):
        # No top-level oracle_text; must join face texts with the separator.
        result = sf.extract_card_fields(DELVER_RAW, updated_at=MIDDLE)
        self.assertIn("Front text", result["oracle_text"])
        self.assertIn("Flying", result["oracle_text"])
        self.assertIn("\n---\n", result["oracle_text"])

    def test_uses_passed_updated_at_verbatim(self):
        # The projection is deliberately time-agnostic — whatever the
        # caller passes ends up in the entry unchanged.
        for ts in (EARLIER, MIDDLE, LATER):
            result = sf.extract_card_fields(LIGHTNING_BOLT_RAW, updated_at=ts)
            self.assertEqual(result["updated_at"], ts)

    def test_missing_fields_yield_none(self):
        result = sf.extract_card_fields({"id": "x", "name": "X"}, updated_at=LATER)
        self.assertIsNone(result["mana_cost"])
        self.assertIsNone(result["type_line"])
        self.assertIsNone(result["oracle_text"])

    def test_mdfc_null_toplevel_falls_back_to_joined_string(self):
        result = sf.extract_card_fields(BALA_GED_MDFC_RAW, updated_at=LATER)
        self.assertEqual(result["mana_cost"], "{2}{G} // ")
        self.assertEqual(result["type_line"], "Sorcery // Land")
        self.assertIn("Return target card", result["oracle_text"])
        self.assertIn("Add {G}", result["oracle_text"])
        self.assertIn("\n---\n", result["oracle_text"])

    def test_null_toplevel_type_line_joins_per_face_with_slashes(self):
        raw = {
            "id": "x",
            "name": "Front // Back",
            "mana_cost": None,
            "type_line": None,
            "card_faces": [
                {"name": "Front", "mana_cost": "{2}{U}", "type_line": "Creature"},
                {"name": "Back", "mana_cost": "{1}{R}", "type_line": "Enchantment"},
            ],
        }
        result = sf.extract_card_fields(raw, updated_at=LATER)
        self.assertEqual(result["mana_cost"], "{2}{U} // {1}{R}")
        self.assertEqual(result["type_line"], "Creature // Enchantment")

    def test_transform_dfc_prefers_populated_toplevel_fields(self):
        result = sf.extract_card_fields(DELVER_RAW, updated_at=LATER)
        self.assertEqual(result["mana_cost"], "{U}")
        self.assertEqual(result["type_line"], DELVER_RAW["type_line"])

    def test_oracle_id_is_copied_when_present(self):
        result = sf.extract_card_fields(LIGHTNING_BOLT_RAW, updated_at=LATER)
        self.assertEqual(result["oracle_id"], "oracle-lb")

    def test_oracle_id_is_none_when_missing(self):
        raw_without_oracle_id = {"id": "x", "name": "X"}
        result = sf.extract_card_fields(raw_without_oracle_id, updated_at=LATER)
        self.assertIsNone(result["oracle_id"])


# ---------------------------------------------------------------------------
# upsert_card / bulk_upsert_cards
# ---------------------------------------------------------------------------

class TestUpsertCard(_MongoBackedTestCase):

    def test_single_upsert_stores_under_scryfall_id(self):
        ok = sf.upsert_card(self.coll, LIGHTNING_BOLT_RAW, updated_at=LATER)
        self.assertTrue(ok)
        doc = self.coll.find_one({"_id": "id-lb"})
        self.assertIsNotNone(doc)
        self.assertEqual(doc["name"], "Lightning Bolt")
        self.assertEqual(doc["updated_at"], LATER)
        self.assertEqual(doc["names"], ["lightning bolt"])
        self.assertEqual(doc["oracle_id"], "oracle-lb")

    def test_returns_false_and_stores_nothing_without_id(self):
        ok = sf.upsert_card(self.coll, {"name": "No id"}, updated_at=LATER)
        self.assertFalse(ok)
        self.assertEqual(self.coll.count_documents({}), 0)

    def test_upsert_preserves_existing_tags_field(self):
        # tags is owned by oracle_tags, not scryfall_fetch. A refetch of
        # the same card must NOT clobber the tags array an earlier
        # tag-import run wrote.
        self.coll.insert_one({
            "_id": "id-lb",
            "scryfall_id": "id-lb",
            "name": "Lightning Bolt (old)",
            "updated_at": EARLIER,
            "tags": ["spot-removal", "burn-any"],
        })
        sf.upsert_card(self.coll, LIGHTNING_BOLT_RAW, updated_at=LATER)
        doc = self.coll.find_one({"_id": "id-lb"})
        # Fetcher-owned fields updated:
        self.assertEqual(doc["name"], "Lightning Bolt")
        self.assertEqual(doc["updated_at"], LATER)
        # Fetcher-foreign field intact:
        self.assertEqual(doc["tags"], ["spot-removal", "burn-any"])

    def test_bulk_upsert_stores_everything(self):
        count = sf.bulk_upsert_cards(
            self.coll, [LIGHTNING_BOLT_RAW, SOL_RING_RAW, DELVER_RAW], LATER
        )
        self.assertEqual(count, 3)
        self.assertEqual(self.coll.count_documents({}), 3)
        self.assertEqual(self.coll.find_one({"_id": "id-delver"})["name"],
                         "Delver of Secrets // Insectile Aberration")

    def test_bulk_upsert_skips_entries_missing_id(self):
        count = sf.bulk_upsert_cards(
            self.coll, [LIGHTNING_BOLT_RAW, {"name": "no id"}], LATER
        )
        self.assertEqual(count, 1)
        self.assertEqual(self.coll.count_documents({}), 1)

    def test_bulk_upsert_preserves_existing_tags(self):
        # Same guarantee as single-upsert but through the bulk path.
        self.coll.insert_one({
            "_id": "id-lb", "scryfall_id": "id-lb", "name": "Old",
            "updated_at": EARLIER, "tags": ["spot-removal"],
        })
        sf.bulk_upsert_cards(self.coll, [LIGHTNING_BOLT_RAW], LATER)
        doc = self.coll.find_one({"_id": "id-lb"})
        self.assertEqual(doc["tags"], ["spot-removal"])
        self.assertEqual(doc["updated_at"], LATER)

    def test_empty_bulk_is_a_noop(self):
        # Bulk-write doesn't like being called with an empty op list;
        # the helper must handle this cleanly.
        count = sf.bulk_upsert_cards(self.coll, [], LATER)
        self.assertEqual(count, 0)


# ---------------------------------------------------------------------------
# find_cards_by_name
# ---------------------------------------------------------------------------

class TestFindCardsByName(_MongoBackedTestCase):

    def _seed(self, *raws, updated_at=LATER):
        sf.bulk_upsert_cards(self.coll, list(raws), updated_at)

    def test_hit_returns_single_element_list(self):
        self._seed(LIGHTNING_BOLT_RAW)
        matches = sf.find_cards_by_name(self.coll, "Lightning Bolt")
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0]["scryfall_id"], "id-lb")

    def test_lookup_is_case_insensitive(self):
        self._seed(LIGHTNING_BOLT_RAW)
        self.assertEqual(
            sf.find_cards_by_name(self.coll, "LIGHTNING BOLT")[0]["scryfall_id"],
            "id-lb",
        )

    def test_miss_returns_empty_list(self):
        self._seed(LIGHTNING_BOLT_RAW)
        self.assertEqual(sf.find_cards_by_name(self.coll, "Unknown Card"), [])

    def test_dfc_resolvable_via_either_face_name(self):
        self._seed(DELVER_RAW)
        for q in ("Delver of Secrets", "Insectile Aberration",
                  "Delver of Secrets // Insectile Aberration"):
            with self.subTest(q=q):
                matches = sf.find_cards_by_name(self.coll, q)
                self.assertEqual(len(matches), 1)
                self.assertEqual(matches[0]["scryfall_id"], "id-delver")

    def test_ambiguous_name_returns_all_matching_cards(self):
        self._seed(DELVER_RAW, DELVER_ART_RAW)
        matches = sf.find_cards_by_name(self.coll, "Delver of Secrets")
        self.assertEqual(
            {m["scryfall_id"] for m in matches},
            {"id-delver", "id-delver-art"},
        )


# ---------------------------------------------------------------------------
# has_stale_cards
# ---------------------------------------------------------------------------

class TestHasStaleCards(_MongoBackedTestCase):

    def test_empty_collection_is_not_stale(self):
        self.assertFalse(sf.has_stale_cards(self.coll, LATER))

    def test_all_entries_at_snapshot_are_not_stale(self):
        self.coll.insert_many([
            {"_id": "a", "updated_at": LATER},
            {"_id": "b", "updated_at": LATER},
        ])
        self.assertFalse(sf.has_stale_cards(self.coll, LATER))

    def test_entry_newer_than_snapshot_is_not_stale(self):
        self.coll.insert_one({"_id": "a", "updated_at": "2999-01-01T00:00:00+00:00"})
        self.assertFalse(sf.has_stale_cards(self.coll, LATER))

    def test_any_entry_older_than_snapshot_triggers_stale(self):
        self.coll.insert_many([
            {"_id": "fresh", "updated_at": LATER},
            {"_id": "old", "updated_at": EARLIER},
        ])
        self.assertTrue(sf.has_stale_cards(self.coll, LATER))

    def test_precision_agnostic_across_fractional_second_widths(self):
        # Same instant written with different fractional-second widths.
        # String compare would mark the shorter form as older; the
        # fromisoformat parse normalises both sides.
        self.coll.insert_one({"_id": "a", "updated_at": "2026-07-31T09:03:40.749+00:00"})
        snapshot = "2026-07-31T09:03:40.749000+00:00"
        self.assertFalse(sf.has_stale_cards(self.coll, snapshot))

    def test_malformed_timestamp_is_skipped_not_treated_as_stale(self):
        self.coll.insert_many([
            {"_id": "good", "updated_at": LATER},
            {"_id": "bad", "updated_at": "definitely not a date"},
        ])
        self.assertFalse(sf.has_stale_cards(self.coll, LATER))

    def test_missing_timestamp_is_skipped(self):
        self.coll.insert_many([
            {"_id": "noop"},
            {"_id": "good", "updated_at": LATER},
        ])
        self.assertFalse(sf.has_stale_cards(self.coll, LATER))


# ---------------------------------------------------------------------------
# read_names (input parsing — pure, unchanged by the Mongo switch)
# ---------------------------------------------------------------------------

class TestReadNames(unittest.TestCase):

    @staticmethod
    def _args(cards=None, file=None):
        return argparse.Namespace(cards=cards or [], file=file)

    def test_args_only_preserves_input_order(self):
        names = sf.read_names(self._args(cards=["Sol Ring", "Lightning Bolt"]))
        self.assertEqual(names, ["Sol Ring", "Lightning Bolt"])

    def test_dedupes_case_insensitively_keeping_first_seen_form(self):
        names = sf.read_names(self._args(cards=["Sol Ring", "sol ring", "SOL RING"]))
        self.assertEqual(names, ["Sol Ring"])

    def test_file_input_skips_blanks_and_comments(self):
        with TemporaryDirectory() as d:
            p = Path(d) / "list.txt"
            p.write_text(
                "# a comment\nSol Ring\n\n# another comment\nLightning Bolt\n",
                encoding="utf-8",
            )
            names = sf.read_names(self._args(file=str(p)))
            self.assertEqual(names, ["Sol Ring", "Lightning Bolt"])

    def test_file_then_args_merged_and_deduped(self):
        with TemporaryDirectory() as d:
            p = Path(d) / "list.txt"
            p.write_text("Sol Ring\nLightning Bolt\n", encoding="utf-8")
            names = sf.read_names(
                self._args(file=str(p), cards=["Lightning Bolt", "Counterspell"])
            )
            self.assertEqual(names, ["Sol Ring", "Lightning Bolt", "Counterspell"])

    def test_decklist_format_strips_leading_quantities(self):
        with TemporaryDirectory() as d:
            p = Path(d) / "deck.txt"
            p.write_text(
                "1 Sol Ring\n10 Forest\n4x Lightning Bolt\n",
                encoding="utf-8",
            )
            names = sf.read_names(self._args(file=str(p)))
            self.assertEqual(names, ["Sol Ring", "Forest", "Lightning Bolt"])

    def test_decklist_format_preserves_dfc_slashes(self):
        with TemporaryDirectory() as d:
            p = Path(d) / "deck.txt"
            p.write_text(
                "1 Bala Ged Recovery // Bala Ged Sanctuary\n"
                "1 Emeritus of Abundance // Regrowth\n",
                encoding="utf-8",
            )
            names = sf.read_names(self._args(file=str(p)))
            self.assertEqual(
                names,
                [
                    "Bala Ged Recovery // Bala Ged Sanctuary",
                    "Emeritus of Abundance // Regrowth",
                ],
            )

    def test_decklist_format_strips_set_and_collector_suffix(self):
        with TemporaryDirectory() as d:
            p = Path(d) / "deck.txt"
            p.write_text(
                "1 Lightning Bolt (STA) 42\n"
                "4 Counterspell (LEA)\n"
                "1 Bala Ged Recovery // Bala Ged Sanctuary (ZNR) 180\n",
                encoding="utf-8",
            )
            names = sf.read_names(self._args(file=str(p)))
            self.assertEqual(
                names,
                [
                    "Lightning Bolt",
                    "Counterspell",
                    "Bala Ged Recovery // Bala Ged Sanctuary",
                ],
            )

    def test_decklist_format_strips_sideboard_prefix(self):
        with TemporaryDirectory() as d:
            p = Path(d) / "deck.txt"
            p.write_text("SB: 1 Force of Will\nSB: 2 Flusterstorm\n", encoding="utf-8")
            names = sf.read_names(self._args(file=str(p)))
            self.assertEqual(names, ["Force of Will", "Flusterstorm"])

    def test_parse_decklist_line_directly(self):
        cases = [
            ("1 Sol Ring", "Sol Ring"),
            ("10 Forest", "Forest"),
            ("4x Lightning Bolt", "Lightning Bolt"),
            ("Sol Ring", "Sol Ring"),
            ("  1  Sol Ring  ", "Sol Ring"),
            ("SB: 1 Force of Will (EMA) 49", "Force of Will"),
            ("1 Bala Ged Recovery // Bala Ged Sanctuary", "Bala Ged Recovery // Bala Ged Sanctuary"),
            ("# comment", None),
            ("", None),
            ("   ", None),
        ]
        for raw, expected in cases:
            with self.subTest(raw=raw):
                self.assertEqual(sf.parse_decklist_line(raw), expected)


# ---------------------------------------------------------------------------
# get_bulk_oracle_metadata
# ---------------------------------------------------------------------------

class TestGetBulkOracleMetadata(unittest.TestCase):

    def test_returns_oracle_cards_entry_from_response(self):
        response = {
            "data": [
                {"type": "default_cards", "updated_at": EARLIER},
                {"type": "oracle_cards", "updated_at": LATER, "jsonl_download_uri": "u"},
            ]
        }
        with patch.object(sf, "http_get_json", return_value=response):
            meta = sf.get_bulk_oracle_metadata()
        self.assertEqual(meta["type"], "oracle_cards")
        self.assertEqual(meta["updated_at"], LATER)

    def test_raises_when_oracle_cards_entry_missing(self):
        response = {"data": [{"type": "default_cards"}]}
        with patch.object(sf, "http_get_json", return_value=response):
            with self.assertRaises(RuntimeError):
                sf.get_bulk_oracle_metadata()


# ---------------------------------------------------------------------------
# download_bulk_oracle_cards (gzip + JSONL parsing)
# ---------------------------------------------------------------------------

class TestDownloadBulkOracleCards(unittest.TestCase):

    FAKE_META = {
        "updated_at": LATER,
        "jsonl_download_uri": "https://example.invalid/oracle.jsonl.gz",
        "compressed_size": 100,
    }

    @staticmethod
    def _make_response(body_bytes):
        """Build a MagicMock that behaves like urlopen's context manager."""
        m = MagicMock()
        m.read.return_value = body_bytes
        m.__enter__.return_value = m
        m.__exit__.return_value = False
        return m

    def test_parses_uncompressed_jsonl_payload(self):
        payload = (
            json.dumps(LIGHTNING_BOLT_RAW).encode()
            + b"\n"
            + json.dumps(SOL_RING_RAW).encode()
        )
        with patch(
            "mtg_recommender.scryfall_fetch.urllib.request.urlopen",
            return_value=self._make_response(payload),
        ):
            cards = call_silent(sf.download_bulk_oracle_cards, self.FAKE_META)
        self.assertEqual(len(cards), 2)
        self.assertEqual({c["id"] for c in cards}, {"id-lb", "id-sol-ring"})

    def test_decompresses_gzipped_payload(self):
        payload = json.dumps(LIGHTNING_BOLT_RAW).encode()
        gz = gzip.compress(payload)
        with patch(
            "mtg_recommender.scryfall_fetch.urllib.request.urlopen",
            return_value=self._make_response(gz),
        ):
            cards = call_silent(sf.download_bulk_oracle_cards, self.FAKE_META)
        self.assertEqual(len(cards), 1)
        self.assertEqual(cards[0]["id"], "id-lb")

    def test_skips_blank_lines_in_payload(self):
        payload = (
            b"\n"
            + json.dumps(LIGHTNING_BOLT_RAW).encode()
            + b"\n\n"
            + json.dumps(SOL_RING_RAW).encode()
            + b"\n"
        )
        with patch(
            "mtg_recommender.scryfall_fetch.urllib.request.urlopen",
            return_value=self._make_response(payload),
        ):
            cards = call_silent(sf.download_bulk_oracle_cards, self.FAKE_META)
        self.assertEqual(len(cards), 2)


# ---------------------------------------------------------------------------
# Bulk mode (every download trigger + happy paths)
# ---------------------------------------------------------------------------

FAKE_META = {
    "type": "oracle_cards",
    "updated_at": LATER,
    "jsonl_download_uri": "https://example.invalid/oracle.jsonl.gz",
    "compressed_size": 100,
}


class TestBulkMode(_MongoBackedTestCase):

    def test_download_triggered_when_collection_empty(self):
        with patch.object(sf, "get_bulk_oracle_metadata", return_value=FAKE_META), \
             patch.object(
                 sf, "download_bulk_oracle_cards",
                 return_value=[LIGHTNING_BOLT_RAW, SOL_RING_RAW],
             ) as dl:
            changed = call_silent(sf.run_bulk_mode, self.coll, False)
        dl.assert_called_once()
        self.assertTrue(changed)
        self.assertEqual(self.coll.count_documents({}), 2)

    def test_download_skipped_when_snapshot_already_covered(self):
        # Pre-populate at the snapshot's own timestamp — nothing to do.
        sf.bulk_upsert_cards(self.coll, [LIGHTNING_BOLT_RAW, SOL_RING_RAW], LATER)
        with patch.object(sf, "get_bulk_oracle_metadata", return_value=FAKE_META), \
             patch.object(sf, "download_bulk_oracle_cards") as dl:
            changed = call_silent(sf.run_bulk_mode, self.coll, False)
        dl.assert_not_called()
        self.assertFalse(changed)

    def test_download_triggered_when_collection_has_stale_entries(self):
        sf.upsert_card(self.coll, LIGHTNING_BOLT_RAW, updated_at=EARLIER)
        with patch.object(sf, "get_bulk_oracle_metadata", return_value=FAKE_META), \
             patch.object(
                 sf, "download_bulk_oracle_cards",
                 return_value=[LIGHTNING_BOLT_RAW],
             ) as dl:
            changed = call_silent(sf.run_bulk_mode, self.coll, False)
        dl.assert_called_once()
        self.assertTrue(changed)
        # The stale entry was restamped with the snapshot timestamp.
        self.assertEqual(
            self.coll.find_one({"_id": "id-lb"})["updated_at"], LATER
        )

    def test_refresh_forces_download_even_when_everything_current(self):
        sf.upsert_card(self.coll, LIGHTNING_BOLT_RAW, updated_at=LATER)
        with patch.object(sf, "get_bulk_oracle_metadata", return_value=FAKE_META), \
             patch.object(
                 sf, "download_bulk_oracle_cards",
                 return_value=[LIGHTNING_BOLT_RAW],
             ) as dl:
            call_silent(sf.run_bulk_mode, self.coll, True)
        dl.assert_called_once()

    def test_merged_cards_resolvable_by_shared_face_name(self):
        # The names-index path still works end-to-end after a bulk merge:
        # art-card and front-face variants both land in the collection
        # and both answer a lookup by the shared face name.
        with patch.object(sf, "get_bulk_oracle_metadata", return_value=FAKE_META), \
             patch.object(
                 sf, "download_bulk_oracle_cards",
                 return_value=[DELVER_RAW, DELVER_ART_RAW],
             ):
            call_silent(sf.run_bulk_mode, self.coll, False)
        matches = sf.find_cards_by_name(self.coll, "Delver of Secrets")
        self.assertEqual(
            {m["scryfall_id"] for m in matches},
            {"id-delver", "id-delver-art"},
        )

    def test_meta_timestamp_persisted_after_successful_merge(self):
        # The fast-path skip only works if the merge records the snapshot
        # it just loaded. Verify meta carries the right value after a
        # fresh download.
        with patch.object(sf, "get_bulk_oracle_metadata", return_value=FAKE_META), \
             patch.object(
                 sf, "download_bulk_oracle_cards",
                 return_value=[LIGHTNING_BOLT_RAW],
             ):
            call_silent(sf.run_bulk_mode, self.coll, False)
        self.assertEqual(
            storage.get_snapshot_timestamp(sf.META_SOURCE), LATER
        )

    def test_fast_path_skips_staleness_scan_when_meta_matches(self):
        # After a prior merge recorded the snapshot in meta, a rerun with
        # the same snapshot must NOT scan every card doc. We assert that
        # by patching has_stale_cards and refusing to let it be called.
        sf.bulk_upsert_cards(self.coll, [LIGHTNING_BOLT_RAW], LATER)
        storage.set_snapshot_timestamp(sf.META_SOURCE, LATER)
        with patch.object(sf, "get_bulk_oracle_metadata", return_value=FAKE_META), \
             patch.object(sf, "download_bulk_oracle_cards") as dl, \
             patch.object(sf, "has_stale_cards") as stale:
            changed = call_silent(sf.run_bulk_mode, self.coll, False)
        self.assertFalse(changed)
        dl.assert_not_called()
        stale.assert_not_called()

    def test_fast_path_bypassed_when_meta_timestamp_stale(self):
        # Meta recorded an older snapshot than the current /bulk-data —
        # the fast-path check must fail and we must fall through to the
        # staleness scan + download.
        sf.bulk_upsert_cards(self.coll, [LIGHTNING_BOLT_RAW], EARLIER)
        storage.set_snapshot_timestamp(sf.META_SOURCE, EARLIER)
        with patch.object(sf, "get_bulk_oracle_metadata", return_value=FAKE_META), \
             patch.object(
                 sf, "download_bulk_oracle_cards",
                 return_value=[LIGHTNING_BOLT_RAW],
             ) as dl:
            call_silent(sf.run_bulk_mode, self.coll, False)
        dl.assert_called_once()
        self.assertEqual(
            storage.get_snapshot_timestamp(sf.META_SOURCE), LATER
        )

    def test_pre_meta_cluster_upgrades_to_fast_path_after_clean_scan(self):
        # Simulates a cluster populated before meta tracking existed:
        # cards are present and already fresh, but no meta entry exists.
        # The slow path should determine nothing is stale, skip the
        # download, AND write the meta entry so the next run is fast.
        sf.bulk_upsert_cards(self.coll, [LIGHTNING_BOLT_RAW], LATER)
        self.assertIsNone(storage.get_snapshot_timestamp(sf.META_SOURCE))
        with patch.object(sf, "get_bulk_oracle_metadata", return_value=FAKE_META), \
             patch.object(sf, "download_bulk_oracle_cards") as dl:
            changed = call_silent(sf.run_bulk_mode, self.coll, False)
        dl.assert_not_called()
        self.assertFalse(changed)
        self.assertEqual(
            storage.get_snapshot_timestamp(sf.META_SOURCE), LATER
        )

    def test_refresh_bypasses_fast_path_even_when_meta_matches(self):
        # --refresh must always force the download, no matter how fresh
        # meta claims the cache is.
        sf.bulk_upsert_cards(self.coll, [LIGHTNING_BOLT_RAW], LATER)
        storage.set_snapshot_timestamp(sf.META_SOURCE, LATER)
        with patch.object(sf, "get_bulk_oracle_metadata", return_value=FAKE_META), \
             patch.object(
                 sf, "download_bulk_oracle_cards",
                 return_value=[LIGHTNING_BOLT_RAW],
             ) as dl:
            call_silent(sf.run_bulk_mode, self.coll, True)
        dl.assert_called_once()


if __name__ == "__main__":
    unittest.main(verbosity=2)
