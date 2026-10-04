"""Repeatable, offline tests for scryfall_fetch.py.

Every HTTP call is mocked so the suite is deterministic and takes well
under a second. No network access, no writes outside `TemporaryDirectory`.

Run with:

    python -m unittest discover tests
    # or, more targeted:
    python -m unittest tests.test_scryfall_fetch

Test classes are grouped by concern so a failure narrows the search:

    TestCardProjection            extract_card_fields
    TestAliases                   _add_alias, register_aliases
    TestStoreCard                 store_card
    TestResolveByName             resolve_by_name (including ambiguity)
    TestHasStaleCards             freshness comparison + precision quirks
    TestCacheIO                   load_cache / save_cache roundtrip
    TestReadNames                 --file + positional arg parsing
    TestGetBulkOracleMetadata     /bulk-data response filtering
    TestDownloadBulkOracleCards   gzip detection + JSONL parsing
    TestSingleMode                run_single_mode with a mocked API
    TestBulkMode                  run_bulk_mode + every download trigger
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
from urllib.error import HTTPError

from mtg_recommender import scryfall_fetch as sf


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
    "name": "Lightning Bolt",
    "mana_cost": "{R}",
    "type_line": "Instant",
    "oracle_text": "Lightning Bolt deals 3 damage to any target.",
}

SOL_RING_RAW = {
    "id": "id-sol-ring",
    "name": "Sol Ring",
    "mana_cost": "{1}",
    "type_line": "Artifact",
    "oracle_text": "{T}: Add {C}{C}.",
}

# A real double-faced card: primary name is the combined form, per-face
# names live under card_faces.
DELVER_RAW = {
    "id": "id-delver",
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
# alias-collision cases that motivated list-valued aliases.
DELVER_ART_RAW = {
    "id": "id-delver-art",
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
    otherwise clutter test output. We're asserting on cache state and
    return values, not on printed text, so silencing keeps the test
    output clean.
    """
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        return fn(*args, **kwargs)


# ---------------------------------------------------------------------------
# Pure card projection
# ---------------------------------------------------------------------------

class TestCardProjection(unittest.TestCase):

    def test_single_face_card(self):
        result = sf.extract_card_fields(LIGHTNING_BOLT_RAW, updated_at=LATER)
        self.assertEqual(result["name"], "Lightning Bolt")
        self.assertEqual(result["mana_cost"], "{R}")
        self.assertEqual(result["oracle_text"], LIGHTNING_BOLT_RAW["oracle_text"])
        self.assertEqual(result["scryfall_id"], "id-lb")
        self.assertEqual(result["updated_at"], LATER)

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
        # Bala Ged Recovery: top-level mana_cost / oracle_text are null,
        # per-face values live under card_faces. The projection joins
        # per-face mana costs with " // " (Scryfall's own convention for
        # combined card names), preserving the empty back-face cost as
        # a trailing empty string ("{2}{G} // ").
        result = sf.extract_card_fields(BALA_GED_MDFC_RAW, updated_at=LATER)
        self.assertEqual(result["mana_cost"], "{2}{G} // ")
        # type_line was populated at the top level, so it stays as-is
        # (Scryfall combined it for us).
        self.assertEqual(result["type_line"], "Sorcery // Land")
        # oracle_text still gets the "\n---\n" treatment for downstream
        # text processing — clearer face boundary for NLP tokenisers.
        self.assertIn("Return target card", result["oracle_text"])
        self.assertIn("Add {G}", result["oracle_text"])
        self.assertIn("\n---\n", result["oracle_text"])

    def test_null_toplevel_type_line_joins_per_face_with_slashes(self):
        # Hypothetical layout where Scryfall didn't combine type_line at
        # the top level either (e.g. some Room / split layouts). We
        # should still emit "Creature // Enchantment"-style output.
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
        # Delver's front-face mana cost bubbles up to the top level, so
        # we prefer the top-level string rather than making up a list.
        # (Same for its combined type_line.)
        result = sf.extract_card_fields(DELVER_RAW, updated_at=LATER)
        self.assertEqual(result["mana_cost"], "{U}")
        self.assertEqual(result["type_line"], DELVER_RAW["type_line"])

    def test_single_face_card_keeps_scalar_fields(self):
        # Regression guard: adding per-face fallback must not turn every
        # card into a list. Single-faced cards keep their string values.
        result = sf.extract_card_fields(LIGHTNING_BOLT_RAW, updated_at=LATER)
        self.assertEqual(result["mana_cost"], "{R}")
        self.assertEqual(result["type_line"], "Instant")


# ---------------------------------------------------------------------------
# Alias registration
# ---------------------------------------------------------------------------

class TestAliases(unittest.TestCase):

    def test_add_alias_creates_list_with_lowered_key(self):
        aliases: dict = {}
        sf._add_alias(aliases, "Lightning Bolt", "id-lb")
        self.assertEqual(aliases, {"lightning bolt": ["id-lb"]})

    def test_add_alias_dedupes_same_id(self):
        aliases: dict = {}
        sf._add_alias(aliases, "LB", "id1")
        sf._add_alias(aliases, "LB", "id1")
        self.assertEqual(aliases, {"lb": ["id1"]})

    def test_add_alias_accumulates_distinct_ids(self):
        aliases: dict = {}
        sf._add_alias(aliases, "Same Name", "id1")
        sf._add_alias(aliases, "Same Name", "id2")
        self.assertEqual(aliases, {"same name": ["id1", "id2"]})

    def test_add_alias_ignores_empty_or_none_name(self):
        aliases: dict = {}
        sf._add_alias(aliases, "", "id1")
        sf._add_alias(aliases, None, "id2")
        self.assertEqual(aliases, {})

    def test_register_aliases_single_face(self):
        aliases: dict = {}
        sf.register_aliases(aliases, LIGHTNING_BOLT_RAW, "id-lb")
        self.assertEqual(aliases, {"lightning bolt": ["id-lb"]})

    def test_register_aliases_dfc_registers_combined_name_and_each_face(self):
        aliases: dict = {}
        sf.register_aliases(aliases, DELVER_RAW, "id-delver")
        self.assertEqual(
            set(aliases),
            {
                "delver of secrets // insectile aberration",
                "delver of secrets",
                "insectile aberration",
            },
        )

    def test_two_cards_sharing_face_name_both_appear_in_alias_list(self):
        aliases: dict = {}
        sf.register_aliases(aliases, DELVER_RAW, "id-delver")
        sf.register_aliases(aliases, DELVER_ART_RAW, "id-delver-art")
        # "Delver of Secrets" is now a claim from both cards. Order isn't
        # guaranteed to matter, so compare as a set.
        self.assertEqual(
            set(aliases["delver of secrets"]),
            {"id-delver", "id-delver-art"},
        )


# ---------------------------------------------------------------------------
# store_card
# ---------------------------------------------------------------------------

class TestStoreCard(unittest.TestCase):

    def test_stores_under_scryfall_id(self):
        cache = {"cards": {}, "aliases": {}}
        ok = sf.store_card(cache, LIGHTNING_BOLT_RAW, updated_at=LATER)
        self.assertTrue(ok)
        self.assertIn("id-lb", cache["cards"])
        self.assertEqual(cache["cards"]["id-lb"]["name"], "Lightning Bolt")
        self.assertEqual(cache["cards"]["id-lb"]["updated_at"], LATER)

    def test_registers_aliases_from_raw(self):
        cache = {"cards": {}, "aliases": {}}
        sf.store_card(cache, LIGHTNING_BOLT_RAW, updated_at=LATER)
        self.assertEqual(cache["aliases"], {"lightning bolt": ["id-lb"]})

    def test_extra_aliases_are_registered_and_deduped(self):
        cache = {"cards": {}, "aliases": {}}
        sf.store_card(
            cache,
            LIGHTNING_BOLT_RAW,
            updated_at=LATER,
            extra_aliases=["lightning bolt", "LiGhTnInG bOlT", ""],
        )
        # Both extras collapse to the same lowered key as the raw's name,
        # so we still expect exactly one id in that list.
        self.assertEqual(cache["aliases"]["lightning bolt"], ["id-lb"])

    def test_returns_false_and_stores_nothing_without_id(self):
        cache = {"cards": {}, "aliases": {}}
        ok = sf.store_card(cache, {"name": "No id"}, updated_at=LATER)
        self.assertFalse(ok)
        self.assertEqual(cache["cards"], {})
        self.assertEqual(cache["aliases"], {})


# ---------------------------------------------------------------------------
# resolve_by_name
# ---------------------------------------------------------------------------

class TestResolveByName(unittest.TestCase):

    @staticmethod
    def _cache_with(*raws, updated_at=LATER):
        cache = {"cards": {}, "aliases": {}}
        for r in raws:
            sf.store_card(cache, r, updated_at=updated_at)
        return cache

    def test_hit_returns_single_element_list(self):
        cache = self._cache_with(LIGHTNING_BOLT_RAW)
        matches = sf.resolve_by_name(cache, "Lightning Bolt")
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0]["scryfall_id"], "id-lb")

    def test_lookup_is_case_insensitive(self):
        cache = self._cache_with(LIGHTNING_BOLT_RAW)
        self.assertEqual(
            sf.resolve_by_name(cache, "LIGHTNING BOLT")[0]["scryfall_id"], "id-lb"
        )

    def test_miss_returns_empty_list(self):
        cache = self._cache_with(LIGHTNING_BOLT_RAW)
        self.assertEqual(sf.resolve_by_name(cache, "Unknown Card"), [])

    def test_dfc_resolvable_via_either_face_name(self):
        cache = self._cache_with(DELVER_RAW)
        by_front = sf.resolve_by_name(cache, "Delver of Secrets")
        by_back = sf.resolve_by_name(cache, "Insectile Aberration")
        by_combined = sf.resolve_by_name(cache, "Delver of Secrets // Insectile Aberration")
        for ms in (by_front, by_back, by_combined):
            self.assertEqual(len(ms), 1)
            self.assertEqual(ms[0]["scryfall_id"], "id-delver")

    def test_ambiguous_name_returns_all_matching_cards(self):
        cache = self._cache_with(DELVER_RAW, DELVER_ART_RAW)
        matches = sf.resolve_by_name(cache, "Delver of Secrets")
        self.assertEqual(
            {m["scryfall_id"] for m in matches},
            {"id-delver", "id-delver-art"},
        )

    def test_dangling_alias_id_is_filtered_out(self):
        # If an alias points to an id no longer in cards (e.g. hand-edited
        # cache), we skip it rather than KeyError.
        cache = {"cards": {}, "aliases": {"ghost": ["nonexistent-id"]}}
        self.assertEqual(sf.resolve_by_name(cache, "ghost"), [])


# ---------------------------------------------------------------------------
# has_stale_cards
# ---------------------------------------------------------------------------

class TestHasStaleCards(unittest.TestCase):

    def test_empty_cache_is_not_stale(self):
        # Emptiness is a separate trigger (handled in run_bulk_mode) —
        # the staleness check itself should just say "nothing older exists."
        self.assertFalse(sf.has_stale_cards({}, LATER))

    def test_all_entries_at_snapshot_are_not_stale(self):
        cards = {"id1": {"updated_at": LATER}, "id2": {"updated_at": LATER}}
        self.assertFalse(sf.has_stale_cards(cards, LATER))

    def test_entry_newer_than_snapshot_is_not_stale(self):
        # A single-fetch happened after the snapshot's publication time.
        cards = {"id1": {"updated_at": "2999-01-01T00:00:00+00:00"}}
        self.assertFalse(sf.has_stale_cards(cards, LATER))

    def test_any_entry_older_than_snapshot_triggers_stale(self):
        cards = {"fresh": {"updated_at": LATER}, "old": {"updated_at": EARLIER}}
        self.assertTrue(sf.has_stale_cards(cards, LATER))

    def test_precision_agnostic_across_fractional_second_widths(self):
        # Same instant, different fractional-second precision. Naive
        # string comparison would falsely mark the shorter form as older.
        cards = {"a": {"updated_at": "2026-07-31T09:03:40.749+00:00"}}
        snapshot = "2026-07-31T09:03:40.749000+00:00"
        self.assertFalse(sf.has_stale_cards(cards, snapshot))

    def test_malformed_timestamp_is_skipped_not_treated_as_stale(self):
        cards = {
            "good": {"updated_at": LATER},
            "bad": {"updated_at": "definitely not a date"},
        }
        # We don't want a single corrupt entry to force a 24 MB redownload.
        self.assertFalse(sf.has_stale_cards(cards, LATER))

    def test_missing_timestamp_is_skipped(self):
        cards = {"noop": {}, "good": {"updated_at": LATER}}
        self.assertFalse(sf.has_stale_cards(cards, LATER))


# ---------------------------------------------------------------------------
# Cache I/O
# ---------------------------------------------------------------------------

class TestCacheIO(unittest.TestCase):

    def test_load_missing_file_returns_empty_shell(self):
        with TemporaryDirectory() as d:
            path = Path(d) / "nope.json"
            cache = sf.load_cache(path)
            self.assertEqual(cache, {"cards": {}, "aliases": {}})

    def test_roundtrip_preserves_structure(self):
        with TemporaryDirectory() as d:
            path = Path(d) / "cache.json"
            projected = sf.extract_card_fields(LIGHTNING_BOLT_RAW, updated_at=LATER)
            cache = {
                "cards": {"id-lb": projected},
                "aliases": {"lightning bolt": ["id-lb"]},
            }
            sf.save_cache(path, cache)
            reloaded = sf.load_cache(path)
            self.assertEqual(reloaded, cache)

    def test_load_tolerates_missing_aliases_key(self):
        # An older or hand-edited cache without an "aliases" key must
        # still load cleanly with an empty aliases dict.
        with TemporaryDirectory() as d:
            path = Path(d) / "cache.json"
            path.write_text(json.dumps({"cards": {}}), encoding="utf-8")
            cache = sf.load_cache(path)
            self.assertEqual(cache, {"cards": {}, "aliases": {}})


# ---------------------------------------------------------------------------
# read_names (input parsing)
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
            # "Lightning Bolt" from args is deduped against the file entry.
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
        # DFC names have "//" inside them — the qty strip must not clip
        # anything past the first word.
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
                "4 Counterspell (LEA)\n"        # collector number omitted
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
        # A few edge cases surfaced as a table for easier scanning.
        cases = [
            ("1 Sol Ring", "Sol Ring"),
            ("10 Forest", "Forest"),
            ("4x Lightning Bolt", "Lightning Bolt"),
            ("Sol Ring", "Sol Ring"),                                # already bare
            ("  1  Sol Ring  ", "Sol Ring"),                         # extra whitespace
            ("SB: 1 Force of Will (EMA) 49", "Force of Will"),       # everything
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
        # Detection happens via the 0x1f 0x8b magic bytes, independent of
        # any Content-Encoding header.
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
# Single-card mode
# ---------------------------------------------------------------------------

class TestSingleMode(unittest.TestCase):

    def test_cache_hit_skips_fetch(self):
        cache = {"cards": {}, "aliases": {}}
        sf.store_card(cache, LIGHTNING_BOLT_RAW, updated_at=LATER)
        with patch.object(sf, "fetch_card_raw_named") as mocked:
            changed = call_silent(sf.run_single_mode, "Lightning Bolt", cache, False)
        mocked.assert_not_called()
        self.assertFalse(changed)

    def test_refresh_forces_fetch_even_when_cached(self):
        cache = {"cards": {}, "aliases": {}}
        sf.store_card(cache, LIGHTNING_BOLT_RAW, updated_at=EARLIER)
        with patch.object(
            sf, "fetch_card_raw_named", return_value=LIGHTNING_BOLT_RAW
        ) as mocked:
            changed = call_silent(sf.run_single_mode, "Lightning Bolt", cache, True)
        mocked.assert_called_once_with("Lightning Bolt")
        self.assertTrue(changed)

    def test_fetch_stores_card_and_registers_user_typed_alias(self):
        cache = {"cards": {}, "aliases": {}}
        # User's typed casing differs from Scryfall's canonical.
        with patch.object(sf, "fetch_card_raw_named", return_value=LIGHTNING_BOLT_RAW):
            changed = call_silent(sf.run_single_mode, "lightning BOLT", cache, False)
        self.assertTrue(changed)
        self.assertIn("id-lb", cache["cards"])
        # Both the user's spelling and the canonical form resolve.
        self.assertEqual(
            sf.resolve_by_name(cache, "lightning bolt")[0]["scryfall_id"], "id-lb"
        )
        self.assertEqual(
            sf.resolve_by_name(cache, "LIGHTNING BOLT")[0]["scryfall_id"], "id-lb"
        )

    def test_404_returns_false_and_leaves_cache_unchanged(self):
        cache = {"cards": {}, "aliases": {}}
        # fetch_card_raw_named returns None on 404.
        with patch.object(sf, "fetch_card_raw_named", return_value=None):
            changed = call_silent(sf.run_single_mode, "Unknown Card", cache, False)
        self.assertFalse(changed)
        self.assertEqual(cache["cards"], {})
        self.assertEqual(cache["aliases"], {})

    def test_http_error_returns_false_gracefully(self):
        cache = {"cards": {}, "aliases": {}}
        err = HTTPError("url", 500, "Server Error", {}, None)
        with patch.object(sf, "fetch_card_raw_named", side_effect=err):
            changed = call_silent(sf.run_single_mode, "Anything", cache, False)
        self.assertFalse(changed)


# ---------------------------------------------------------------------------
# Bulk mode (all four download triggers + happy paths)
# ---------------------------------------------------------------------------

# Metadata dict returned by the mocked get_bulk_oracle_metadata across
# all bulk-mode tests. Its updated_at is LATER.
FAKE_META = {
    "type": "oracle_cards",
    "updated_at": LATER,
    "jsonl_download_uri": "https://example.invalid/oracle.jsonl.gz",
    "compressed_size": 100,
}


class TestBulkMode(unittest.TestCase):

    def _cache(self):
        return {"cards": {}, "aliases": {}}

    def test_download_triggered_when_cache_empty(self):
        cache = self._cache()
        with patch.object(sf, "get_bulk_oracle_metadata", return_value=FAKE_META), \
             patch.object(
                 sf, "download_bulk_oracle_cards",
                 return_value=[LIGHTNING_BOLT_RAW, SOL_RING_RAW],
             ) as dl:
            changed = call_silent(
                sf.run_bulk_mode, ["Lightning Bolt", "Sol Ring"], cache, False
            )
        dl.assert_called_once()
        self.assertTrue(changed)
        self.assertEqual(len(cache["cards"]), 2)

    def test_download_skipped_when_all_requested_present_and_fresh(self):
        cache = self._cache()
        sf.store_card(cache, LIGHTNING_BOLT_RAW, updated_at=LATER)
        sf.store_card(cache, SOL_RING_RAW, updated_at=LATER)
        with patch.object(sf, "get_bulk_oracle_metadata", return_value=FAKE_META), \
             patch.object(sf, "download_bulk_oracle_cards") as dl:
            changed = call_silent(
                sf.run_bulk_mode, ["Lightning Bolt", "Sol Ring"], cache, False
            )
        dl.assert_not_called()
        self.assertFalse(changed)

    def test_download_triggered_when_any_requested_card_missing(self):
        # Cache is fresh but doesn't cover Sol Ring — must download.
        cache = self._cache()
        sf.store_card(cache, LIGHTNING_BOLT_RAW, updated_at=LATER)
        with patch.object(sf, "get_bulk_oracle_metadata", return_value=FAKE_META), \
             patch.object(
                 sf, "download_bulk_oracle_cards",
                 return_value=[LIGHTNING_BOLT_RAW, SOL_RING_RAW],
             ) as dl:
            changed = call_silent(
                sf.run_bulk_mode, ["Lightning Bolt", "Sol Ring"], cache, False
            )
        dl.assert_called_once()
        self.assertTrue(changed)
        self.assertIn("id-sol-ring", cache["cards"])

    def test_download_triggered_when_cache_has_stale_entries(self):
        cache = self._cache()
        # Everything requested IS in cache, but stamped older than snapshot.
        sf.store_card(cache, LIGHTNING_BOLT_RAW, updated_at=EARLIER)
        with patch.object(sf, "get_bulk_oracle_metadata", return_value=FAKE_META), \
             patch.object(
                 sf, "download_bulk_oracle_cards",
                 return_value=[LIGHTNING_BOLT_RAW],
             ) as dl:
            changed = call_silent(
                sf.run_bulk_mode, ["Lightning Bolt"], cache, False
            )
        dl.assert_called_once()
        self.assertTrue(changed)
        # The stale entry was restamped with the snapshot timestamp.
        self.assertEqual(cache["cards"]["id-lb"]["updated_at"], LATER)

    def test_refresh_forces_download_even_when_everything_current(self):
        cache = self._cache()
        sf.store_card(cache, LIGHTNING_BOLT_RAW, updated_at=LATER)
        with patch.object(sf, "get_bulk_oracle_metadata", return_value=FAKE_META), \
             patch.object(
                 sf, "download_bulk_oracle_cards",
                 return_value=[LIGHTNING_BOLT_RAW],
             ) as dl:
            call_silent(sf.run_bulk_mode, ["Lightning Bolt"], cache, True)
        dl.assert_called_once()

    def test_ambiguous_shared_face_name_ends_up_with_multiple_ids(self):
        cache = self._cache()
        with patch.object(sf, "get_bulk_oracle_metadata", return_value=FAKE_META), \
             patch.object(
                 sf, "download_bulk_oracle_cards",
                 return_value=[DELVER_RAW, DELVER_ART_RAW],
             ):
            call_silent(sf.run_bulk_mode, ["Delver of Secrets"], cache, False)
        matches = sf.resolve_by_name(cache, "Delver of Secrets")
        self.assertEqual(
            {m["scryfall_id"] for m in matches},
            {"id-delver", "id-delver-art"},
        )

    def test_unknown_card_after_fresh_merge_is_not_found(self):
        # Empty cache, request includes a card that isn't in the bulk.
        # The download runs (empty cache trigger), but "Nonexistent" still
        # can't be resolved.
        cache = self._cache()
        with patch.object(sf, "get_bulk_oracle_metadata", return_value=FAKE_META), \
             patch.object(
                 sf, "download_bulk_oracle_cards",
                 return_value=[LIGHTNING_BOLT_RAW],
             ):
            call_silent(sf.run_bulk_mode, ["Lightning Bolt", "Nonexistent"], cache, False)
        self.assertEqual(sf.resolve_by_name(cache, "Nonexistent"), [])
        self.assertNotEqual(sf.resolve_by_name(cache, "Lightning Bolt"), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
