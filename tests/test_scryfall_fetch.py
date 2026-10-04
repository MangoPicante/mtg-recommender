"""Repeatable, offline tests for scryfall_fetch.py.

Every HTTP call is mocked and every Mongo op routes through mongomock,
so the suite is deterministic and takes well under a second. No network
access, no real Mongo required.

Run with:

    python -m unittest discover tests
    # or, more targeted:
    python -m unittest tests.test_scryfall_fetch

Test classes are grouped by concern so a failure narrows the search:

    TestCardProjection          extract_card_fields + build_names + sha
    TestUpsertCard              upsert_card / bulk_upsert_cards + the diff
                                 classification (new / changed / unchanged)
    TestFindCardsByName         find_cards_by_name via the names index
    TestReadNames               --file + positional arg parsing (pure; still
                                 used by extract_oracle)
    TestGetBulkOracleMetadata   /bulk-data response filtering
    TestDownloadBulkOracleCards gzip detection + JSONL parsing
    TestBulkMode                run_bulk_mode: fast-path meta skip, snapshot-
                                 mismatch download, pre-meta cluster upgrade
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
        result = sf.extract_card_fields(LIGHTNING_BOLT_RAW)
        self.assertEqual(result["name"], "Lightning Bolt")
        self.assertEqual(result["mana_cost"], "{R}")
        self.assertEqual(result["oracle_text"], LIGHTNING_BOLT_RAW["oracle_text"])
        self.assertEqual(result["_id"], "id-lb")
        # sha of the (post-projection) oracle_text is populated and
        # matches a direct call to the helper on the same string.
        self.assertEqual(
            result["oracle_text_sha"], sf.oracle_text_sha(result["oracle_text"])
        )

    def test_sha_is_stable_and_distinguishes_different_text(self):
        a = sf.oracle_text_sha("Deals 3 damage to any target.")
        a_dup = sf.oracle_text_sha("Deals 3 damage to any target.")
        b = sf.oracle_text_sha("Deals 4 damage to any target.")
        self.assertEqual(a, a_dup)
        self.assertNotEqual(a, b)
        # 16 hex chars by convention.
        self.assertEqual(len(a), 16)

    def test_sha_handles_none_oracle_text(self):
        # Meld pieces and some tokens have no oracle text at all — the
        # helper must not crash and must still return a stable string.
        self.assertEqual(sf.oracle_text_sha(None), sf.oracle_text_sha(""))

    def test_names_contains_lowered_main_name(self):
        result = sf.extract_card_fields(LIGHTNING_BOLT_RAW)
        self.assertEqual(result["names"], ["lightning bolt"])

    def test_names_contains_combined_and_face_names_for_dfc(self):
        result = sf.extract_card_fields(DELVER_RAW)
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
        result = sf.extract_card_fields(DELVER_ART_RAW)
        self.assertEqual(
            result["names"],
            [
                "delver of secrets // delver of secrets",
                "delver of secrets",
            ],
        )

    def test_multifaced_card_missing_top_level_oracle_joins_faces(self):
        # No top-level oracle_text; must join face texts with the separator.
        result = sf.extract_card_fields(DELVER_RAW)
        self.assertIn("Front text", result["oracle_text"])
        self.assertIn("Flying", result["oracle_text"])
        self.assertIn("\n---\n", result["oracle_text"])

    def test_missing_fields_yield_none(self):
        result = sf.extract_card_fields({"id": "x", "name": "X"})
        self.assertIsNone(result["mana_cost"])
        self.assertIsNone(result["type_line"])
        self.assertIsNone(result["oracle_text"])

    def test_mdfc_null_toplevel_falls_back_to_joined_string(self):
        result = sf.extract_card_fields(BALA_GED_MDFC_RAW)
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
        result = sf.extract_card_fields(raw)
        self.assertEqual(result["mana_cost"], "{2}{U} // {1}{R}")
        self.assertEqual(result["type_line"], "Creature // Enchantment")

    def test_transform_dfc_prefers_populated_toplevel_fields(self):
        result = sf.extract_card_fields(DELVER_RAW)
        self.assertEqual(result["mana_cost"], "{U}")
        self.assertEqual(result["type_line"], DELVER_RAW["type_line"])

    def test_oracle_id_is_copied_when_present(self):
        result = sf.extract_card_fields(LIGHTNING_BOLT_RAW)
        self.assertEqual(result["oracle_id"], "oracle-lb")

    def test_oracle_id_is_none_when_missing(self):
        raw_without_oracle_id = {"id": "x", "name": "X"}
        result = sf.extract_card_fields(raw_without_oracle_id)
        self.assertIsNone(result["oracle_id"])


# ---------------------------------------------------------------------------
# upsert_card / bulk_upsert_cards
# ---------------------------------------------------------------------------

class TestUpsertCard(_MongoBackedTestCase):

    def test_single_upsert_stores_under_id(self):
        ok = sf.upsert_card(self.coll, LIGHTNING_BOLT_RAW)
        self.assertTrue(ok)
        doc = self.coll.find_one({"_id": "id-lb"})
        self.assertIsNotNone(doc)
        self.assertEqual(doc["name"], "Lightning Bolt")
        self.assertEqual(doc["names"], ["lightning bolt"])
        self.assertEqual(doc["oracle_id"], "oracle-lb")

    def test_returns_false_and_stores_nothing_without_id(self):
        ok = sf.upsert_card(self.coll, {"name": "No id"})
        self.assertFalse(ok)
        self.assertEqual(self.coll.count_documents({}), 0)

    def test_upsert_preserves_existing_tags_field(self):
        # tags is owned by oracle_tags, not scryfall_fetch. A refetch of
        # the same card must NOT clobber the tags array an earlier
        # tag-import run wrote.
        self.coll.insert_one({
            "_id": "id-lb",
            "name": "Lightning Bolt (old)",
            "tags": ["spot-removal", "burn-any"],
        })
        sf.upsert_card(self.coll, LIGHTNING_BOLT_RAW)
        doc = self.coll.find_one({"_id": "id-lb"})
        # Fetcher-owned fields updated:
        self.assertEqual(doc["name"], "Lightning Bolt")
        # Fetcher-foreign field intact:
        self.assertEqual(doc["tags"], ["spot-removal", "burn-any"])

    def test_bulk_upsert_stores_everything(self):
        new, changed, unchanged = sf.bulk_upsert_cards(
            self.coll, [LIGHTNING_BOLT_RAW, SOL_RING_RAW, DELVER_RAW]
        )
        self.assertEqual((new, changed, unchanged), (3, 0, 0))
        self.assertEqual(self.coll.count_documents({}), 3)
        self.assertEqual(self.coll.find_one({"_id": "id-delver"})["name"],
                         "Delver of Secrets // Insectile Aberration")

    def test_bulk_upsert_skips_entries_missing_id(self):
        new, changed, unchanged = sf.bulk_upsert_cards(
            self.coll, [LIGHTNING_BOLT_RAW, {"name": "no id"}]
        )
        self.assertEqual((new, changed, unchanged), (1, 0, 0))
        self.assertEqual(self.coll.count_documents({}), 1)

    def test_bulk_upsert_preserves_existing_tags(self):
        # Same guarantee as single-upsert but through the bulk path.
        # Insert with the current sha so this counts as unchanged (if we
        # inserted with no sha, the diff would write and $unset tags
        # instead — a hypothetical we exercise separately).
        bolt_sha = sf.oracle_text_sha(sf.extract_card_fields(LIGHTNING_BOLT_RAW)["oracle_text"])
        self.coll.insert_one({
            "_id": "id-lb", "name": "Old",
            "oracle_text_sha": bolt_sha,
            "tags": ["spot-removal"],
        })
        sf.bulk_upsert_cards(self.coll, [LIGHTNING_BOLT_RAW])
        doc = self.coll.find_one({"_id": "id-lb"})
        # Unchanged content → no write at all; tags survive untouched.
        self.assertEqual(doc["tags"], ["spot-removal"])

    def test_empty_bulk_is_a_noop(self):
        new, changed, unchanged = sf.bulk_upsert_cards(self.coll, [])
        self.assertEqual((new, changed, unchanged), (0, 0, 0))

    def test_unchanged_card_produces_no_write(self):
        # Pre-seed at a stored sha matching what extract_card_fields will
        # produce for the incoming raw. The bulk pass should classify
        # this as unchanged and leave the doc intact.
        sha = sf.oracle_text_sha(sf.extract_card_fields(LIGHTNING_BOLT_RAW)["oracle_text"])
        self.coll.insert_one({
            "_id": "id-lb", "name": "Lightning Bolt",
            "oracle_text_sha": sha,
            "text_embedding": [0.0] * 4, "card_vector": [0.0] * 4,
        })
        new, changed, unchanged = sf.bulk_upsert_cards(
            self.coll, [LIGHTNING_BOLT_RAW]
        )
        self.assertEqual((new, changed, unchanged), (0, 0, 1))
        doc = self.coll.find_one({"_id": "id-lb"})
        # No write means downstream fields stay as they were.
        self.assertEqual(doc["text_embedding"], [0.0] * 4)
        self.assertEqual(doc["card_vector"], [0.0] * 4)

    def test_changed_content_rewrites_and_clears_downstream_fields(self):
        # Stored sha matches an OLD oracle_text; the incoming raw has
        # new text, so the diff detects a change and $unsets both
        # embedding fields.
        self.coll.insert_one({
            "_id": "id-lb", "name": "Old",
            "oracle_text_sha": "deadbeefdeadbeef",
            "text_embedding": [0.0] * 4, "card_vector": [0.0] * 4,
            "tags": ["spot-removal"],
        })
        new, changed, unchanged = sf.bulk_upsert_cards(
            self.coll, [LIGHTNING_BOLT_RAW]
        )
        self.assertEqual((new, changed, unchanged), (0, 1, 0))
        doc = self.coll.find_one({"_id": "id-lb"})
        self.assertEqual(doc["name"], "Lightning Bolt")
        # Owned-by-oracle_tags field intact across the content change.
        self.assertEqual(doc["tags"], ["spot-removal"])
        # Downstream fields invalidated so mtg-embed re-encodes.
        self.assertNotIn("text_embedding", doc)
        self.assertNotIn("card_vector", doc)

    def test_pre_sha_doc_is_treated_as_changed(self):
        # A doc from an older schema (no oracle_text_sha) must be
        # rewritten so the sha is populated for future diff passes.
        # Downstream fields get cleared too — safest default when we
        # can't prove the text is unchanged.
        self.coll.insert_one({
            "_id": "id-lb", "name": "Lightning Bolt",
            "text_embedding": [0.0] * 4,
        })
        new, changed, unchanged = sf.bulk_upsert_cards(
            self.coll, [LIGHTNING_BOLT_RAW]
        )
        self.assertEqual((new, changed, unchanged), (0, 1, 0))
        doc = self.coll.find_one({"_id": "id-lb"})
        self.assertIn("oracle_text_sha", doc)
        self.assertNotIn("text_embedding", doc)


# ---------------------------------------------------------------------------
# find_cards_by_name
# ---------------------------------------------------------------------------

class TestFindCardsByName(_MongoBackedTestCase):

    def _seed(self, *raws):
        sf.bulk_upsert_cards(self.coll, list(raws))

    def test_hit_returns_single_element_list(self):
        self._seed(LIGHTNING_BOLT_RAW)
        matches = sf.find_cards_by_name(self.coll, "Lightning Bolt")
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0]["_id"], "id-lb")

    def test_lookup_is_case_insensitive(self):
        self._seed(LIGHTNING_BOLT_RAW)
        self.assertEqual(
            sf.find_cards_by_name(self.coll, "LIGHTNING BOLT")[0]["_id"],
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
                self.assertEqual(matches[0]["_id"], "id-delver")

    def test_ambiguous_name_returns_all_matching_cards(self):
        self._seed(DELVER_RAW, DELVER_ART_RAW)
        matches = sf.find_cards_by_name(self.coll, "Delver of Secrets")
        self.assertEqual(
            {m["_id"] for m in matches},
            {"id-delver", "id-delver-art"},
        )


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
            changed = call_silent(sf.run_bulk_mode, self.coll)
        dl.assert_called_once()
        self.assertTrue(changed)
        self.assertEqual(self.coll.count_documents({}), 2)

    def test_fast_path_skips_when_meta_matches_snapshot(self):
        # Pre-populate + record meta — the common rerun case.
        sf.bulk_upsert_cards(self.coll, [LIGHTNING_BOLT_RAW, SOL_RING_RAW])
        storage.set_snapshot_timestamp(sf.META_SOURCE, LATER)
        with patch.object(sf, "get_bulk_oracle_metadata", return_value=FAKE_META), \
             patch.object(sf, "download_bulk_oracle_cards") as dl:
            changed = call_silent(sf.run_bulk_mode, self.coll)
        dl.assert_not_called()
        self.assertFalse(changed)

    def test_pre_meta_cluster_downloads_once_then_sets_meta(self):
        # A cluster populated before meta tracking existed — cards are
        # present and already match the current snapshot's content, but
        # the meta entry is missing. The module downloads once (the diff
        # then classifies every card as unchanged, so no writes) and
        # sets meta so the next run hits the fast path.
        sf.bulk_upsert_cards(self.coll, [LIGHTNING_BOLT_RAW])
        self.assertIsNone(storage.get_snapshot_timestamp(sf.META_SOURCE))
        with patch.object(sf, "get_bulk_oracle_metadata", return_value=FAKE_META), \
             patch.object(
                 sf, "download_bulk_oracle_cards",
                 return_value=[LIGHTNING_BOLT_RAW],
             ) as dl:
            changed = call_silent(sf.run_bulk_mode, self.coll)
        dl.assert_called_once()
        # No real content delta → report False to the caller.
        self.assertFalse(changed)
        self.assertEqual(
            storage.get_snapshot_timestamp(sf.META_SOURCE), LATER
        )

    def test_snapshot_mismatch_triggers_download(self):
        # Meta recorded an older snapshot than the current /bulk-data —
        # fast-path check fails, download fires.
        sf.bulk_upsert_cards(self.coll, [LIGHTNING_BOLT_RAW])
        storage.set_snapshot_timestamp(sf.META_SOURCE, EARLIER)
        # Flip the sha so the incoming card is "changed", not "unchanged".
        changed_raw = {**LIGHTNING_BOLT_RAW, "oracle_text": "Deals 4 damage to any target."}
        with patch.object(sf, "get_bulk_oracle_metadata", return_value=FAKE_META), \
             patch.object(
                 sf, "download_bulk_oracle_cards",
                 return_value=[changed_raw],
             ) as dl:
            changed = call_silent(sf.run_bulk_mode, self.coll)
        dl.assert_called_once()
        self.assertTrue(changed)
        self.assertEqual(
            storage.get_snapshot_timestamp(sf.META_SOURCE), LATER
        )
        # The new oracle_text landed.
        self.assertIn(
            "4 damage",
            self.coll.find_one({"_id": "id-lb"})["oracle_text"],
        )

    def test_merged_cards_resolvable_by_shared_face_name(self):
        # The names-index path still works end-to-end after a bulk merge:
        # art-card and front-face variants both land in the collection
        # and both answer a lookup by the shared face name.
        with patch.object(sf, "get_bulk_oracle_metadata", return_value=FAKE_META), \
             patch.object(
                 sf, "download_bulk_oracle_cards",
                 return_value=[DELVER_RAW, DELVER_ART_RAW],
             ):
            call_silent(sf.run_bulk_mode, self.coll)
        matches = sf.find_cards_by_name(self.coll, "Delver of Secrets")
        self.assertEqual(
            {m["_id"] for m in matches},
            {"id-delver", "id-delver-art"},
        )

    def test_meta_timestamp_persisted_after_successful_merge(self):
        with patch.object(sf, "get_bulk_oracle_metadata", return_value=FAKE_META), \
             patch.object(
                 sf, "download_bulk_oracle_cards",
                 return_value=[LIGHTNING_BOLT_RAW],
             ):
            call_silent(sf.run_bulk_mode, self.coll)
        self.assertEqual(
            storage.get_snapshot_timestamp(sf.META_SOURCE), LATER
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
