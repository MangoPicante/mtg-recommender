"""Offline tests for mtg_recommender.edhrec_fetch.

Every HTTP call is mocked and every Mongo op routes through mongomock.
A trimmed sample of the live EDHREC payload lives under
tests/fixtures/edhrec_atraxa_trimmed.json and backs the extraction
tests — small enough to keep diff-friendly but structurally identical
to the real response (same `container.json_dict.cardlists` nesting,
multiple sections, duplicate cards across sections, one partial entry
to exercise the drop-on-missing-fields path).

Test classes:
    TestCommanderSlug      — URL slug derivation (apostrophes, commas, unicode)
    TestExtractCardSignals — dedup across sections, max-lift resolution,
                              partial entry dropped, schema-shift returns []
    TestCacheHelpers       — _is_fresh, _ttl_days env override
    TestFetchPayload       — mocked urllib response
    TestGetCommanderPayload— cache miss → HTTP + write; hit → no HTTP;
                              expired → refetch; force=True → refetch
    TestGetCommanderSignals— end-to-end wrapper
"""
from __future__ import annotations

import json
import os
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import mongomock

from mtg_recommender import edhrec_fetch as ef
from mtg_recommender import storage

FIXTURE = Path(__file__).parent / "fixtures" / "edhrec_atraxa_trimmed.json"


def _load_fixture() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# commander_slug
# ---------------------------------------------------------------------------

class TestCommanderSlug(unittest.TestCase):

    def test_plain_name(self):
        self.assertEqual(ef.commander_slug("Jhoira"), "jhoira")

    def test_apostrophe_is_stripped_not_replaced(self):
        # The point of stripping apostrophes BEFORE the non-word collapse
        # is that "Praetors' Voice" → "praetors-voice", not "praetors--voice".
        self.assertEqual(
            ef.commander_slug("Atraxa, Praetors' Voice"),
            "atraxa-praetors-voice",
        )

    def test_curly_apostrophe_also_stripped(self):
        # Scryfall occasionally has U+2019 instead of U+0027.
        self.assertEqual(
            ef.commander_slug("Yuriko, the Tiger’s Shadow"),
            "yuriko-the-tigers-shadow",
        )

    def test_commas_and_periods_become_hyphens_and_trim(self):
        self.assertEqual(
            ef.commander_slug("Jhoira, Weatherlight Captain"),
            "jhoira-weatherlight-captain",
        )

    def test_consecutive_separators_collapse(self):
        # Multiple commas, spaces, etc. should collapse to one hyphen.
        self.assertEqual(ef.commander_slug("A ,  B"), "a-b")

    def test_leading_trailing_separators_trimmed(self):
        self.assertEqual(ef.commander_slug(", edge case ,"), "edge-case")

    def test_empty_or_none_raises(self):
        with self.assertRaises(ValueError):
            ef.commander_slug("")
        with self.assertRaises(ValueError):
            ef.commander_slug(None)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# extract_card_signals
# ---------------------------------------------------------------------------

class TestExtractCardSignals(unittest.TestCase):

    def test_signals_extracted_with_expected_fields(self):
        payload = _load_fixture()
        sigs = ef.extract_card_signals(payload)
        by_id = {s.scryfall_id: s for s in sigs}
        self.assertIn("11111111-1111-1111-1111-111111111111", by_id)
        bolt = by_id["11111111-1111-1111-1111-111111111111"]
        self.assertEqual(bolt.name, "Sanctum Lurker")
        self.assertAlmostEqual(bolt.lift, 1.64)
        self.assertAlmostEqual(bolt.synergy, 0.035)
        self.assertEqual(bolt.num_decks, 236)
        self.assertEqual(bolt.potential_decks, 2633)

    def test_duplicate_across_sections_keeps_max_lift(self):
        # In the fixture, Tekuthal appears in both "New Cards" (lift 1.68)
        # and "Top Cards" (lift 1.72). The 1.72 reading wins.
        payload = _load_fixture()
        sigs = ef.extract_card_signals(payload)
        tek = next(s for s in sigs if s.scryfall_id == "22222222-2222-2222-2222-222222222222")
        self.assertAlmostEqual(tek.lift, 1.72)

    def test_partial_entry_without_lift_is_dropped(self):
        # The fixture's "Partial Entry" section has a card missing lift
        # / num_decks. It must not appear in the extracted list.
        payload = _load_fixture()
        ids = {s.scryfall_id for s in ef.extract_card_signals(payload)}
        self.assertNotIn("44444444-4444-4444-4444-444444444444", ids)

    def test_dedupes_so_total_count_matches_unique_cards(self):
        # Fixture has 3 unique cards eligible for extraction
        # (Sanctum Lurker, Tekuthal, Rhystic Study). The partial-entry
        # card is dropped. So we expect exactly 3.
        payload = _load_fixture()
        sigs = ef.extract_card_signals(payload)
        self.assertEqual(len(sigs), 3)

    def test_missing_cardlists_structure_returns_empty_list(self):
        # If EDHREC changes shape, the extractor must degrade gracefully
        # rather than raising — the recommender then just skips the
        # EDHREC overlay.
        for bad in [{}, {"container": {}}, {"container": {"json_dict": {}}}]:
            self.assertEqual(ef.extract_card_signals(bad), [])

    def test_inclusion_rate_property(self):
        payload = _load_fixture()
        sigs = ef.extract_card_signals(payload)
        bolt = next(s for s in sigs if s.name == "Sanctum Lurker")
        self.assertAlmostEqual(bolt.inclusion_rate, 236 / 2633, places=6)

    def test_inclusion_rate_guards_against_zero_denominator(self):
        sig = ef.CardSignal(
            name="x", scryfall_id="y", lift=1.0, synergy=0.0,
            num_decks=5, potential_decks=0,
        )
        self.assertEqual(sig.inclusion_rate, 0.0)


# ---------------------------------------------------------------------------
# TTL / freshness
# ---------------------------------------------------------------------------

class TestCacheHelpers(unittest.TestCase):

    def test_fresh_entry_within_ttl(self):
        now = datetime.now(timezone.utc).isoformat()
        self.assertTrue(ef._is_fresh(now))

    def test_entry_older_than_ttl_is_stale(self):
        old = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
        self.assertFalse(ef._is_fresh(old))

    def test_malformed_timestamp_is_treated_as_stale(self):
        # A corrupt cache entry mustn't crash the caller — just refetch.
        self.assertFalse(ef._is_fresh("not a date"))
        self.assertFalse(ef._is_fresh(""))

    def test_env_override_ttl_days(self):
        with patch.dict(os.environ, {ef.EDHREC_CACHE_TTL_ENV: "1"}, clear=False):
            self.assertEqual(ef._ttl_days(), 1)
            # An entry 2 days old must now be stale under TTL=1.
            two_days_ago = (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()
            self.assertFalse(ef._is_fresh(two_days_ago))

    def test_bad_env_override_raises(self):
        with patch.dict(os.environ, {ef.EDHREC_CACHE_TTL_ENV: "not-a-number"}, clear=False):
            with self.assertRaises(ValueError):
                ef._ttl_days()


# ---------------------------------------------------------------------------
# fetch_payload (HTTP, mocked)
# ---------------------------------------------------------------------------

class TestFetchPayload(unittest.TestCase):

    def _mock_urlopen(self, body: dict):
        """Build a MagicMock that mimics urlopen's context manager + read()."""
        resp = MagicMock()
        resp.read.return_value = json.dumps(body).encode("utf-8")
        resp.__enter__.return_value = resp
        resp.__exit__.return_value = False
        return resp

    def test_returns_parsed_json(self):
        payload = {"hello": "world"}
        with patch(
            "mtg_recommender.edhrec_fetch.urllib.request.urlopen",
            return_value=self._mock_urlopen(payload),
        ):
            got = ef.fetch_payload("atraxa-praetors-voice")
        self.assertEqual(got, payload)

    def test_url_shape(self):
        with patch(
            "mtg_recommender.edhrec_fetch.urllib.request.urlopen",
            return_value=self._mock_urlopen({}),
        ) as mocked:
            ef.fetch_payload("atraxa-praetors-voice")
        called_url = mocked.call_args.args[0].full_url
        self.assertEqual(
            called_url,
            "https://json.edhrec.com/pages/commanders/atraxa-praetors-voice.json",
        )

    def test_sends_project_user_agent(self):
        with patch(
            "mtg_recommender.edhrec_fetch.urllib.request.urlopen",
            return_value=self._mock_urlopen({}),
        ) as mocked:
            ef.fetch_payload("x")
        headers = mocked.call_args.args[0].headers
        # urllib lowercases header keys in `.headers`.
        self.assertIn("MTG-Recommender", headers.get("User-agent", ""))


# ---------------------------------------------------------------------------
# get_commander_payload (cache behavior)
# ---------------------------------------------------------------------------

class _MongoBackedTestCase(unittest.TestCase):
    def setUp(self):
        storage.reset_client(mongomock.MongoClient())
        self.coll = storage.edhrec_collection()

    def tearDown(self):
        storage.reset_client(None)


class TestGetCommanderPayload(_MongoBackedTestCase):

    def _mock_fetch(self, body: dict):
        return patch.object(ef, "fetch_payload", return_value=body)

    def test_cache_miss_fetches_and_writes(self):
        payload = {"a": 1}
        with self._mock_fetch(payload) as mocked:
            got = ef.get_commander_payload("Atraxa, Praetors' Voice", coll=self.coll)
        mocked.assert_called_once_with("atraxa-praetors-voice")
        self.assertEqual(got, payload)
        stored = self.coll.find_one({"_id": "atraxa-praetors-voice"})
        self.assertEqual(stored["payload"], payload)
        self.assertIn("fetched_at", stored)

    def test_fresh_cache_hit_skips_http(self):
        self.coll.insert_one({
            "_id": "atraxa-praetors-voice",
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "payload": {"from": "cache"},
        })
        with patch.object(ef, "fetch_payload") as mocked:
            got = ef.get_commander_payload("Atraxa, Praetors' Voice", coll=self.coll)
        mocked.assert_not_called()
        self.assertEqual(got, {"from": "cache"})

    def test_stale_cache_refetches(self):
        self.coll.insert_one({
            "_id": "atraxa-praetors-voice",
            "fetched_at": (datetime.now(timezone.utc) - timedelta(days=30)).isoformat(),
            "payload": {"from": "cache"},
        })
        with self._mock_fetch({"from": "fresh"}) as mocked:
            got = ef.get_commander_payload("Atraxa, Praetors' Voice", coll=self.coll)
        mocked.assert_called_once()
        self.assertEqual(got, {"from": "fresh"})
        # Cache now carries the fresh payload.
        self.assertEqual(
            self.coll.find_one({"_id": "atraxa-praetors-voice"})["payload"],
            {"from": "fresh"},
        )

    def test_force_bypasses_fresh_cache(self):
        self.coll.insert_one({
            "_id": "atraxa-praetors-voice",
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "payload": {"from": "cache"},
        })
        with self._mock_fetch({"from": "forced"}) as mocked:
            got = ef.get_commander_payload(
                "Atraxa, Praetors' Voice", force=True, coll=self.coll
            )
        mocked.assert_called_once()
        self.assertEqual(got, {"from": "forced"})


# ---------------------------------------------------------------------------
# End-to-end wrapper
# ---------------------------------------------------------------------------

class TestGetCommanderSignals(_MongoBackedTestCase):

    def test_fetches_and_extracts_signals(self):
        payload = _load_fixture()
        with patch.object(ef, "fetch_payload", return_value=payload):
            sigs = ef.get_commander_signals(
                "Atraxa, Praetors' Voice", coll=self.coll
            )
        # Same 3-card expectation as the extraction test — guards against
        # the wrapper doing something unexpected on top of the pieces.
        self.assertEqual(len(sigs), 3)
        self.assertTrue(all(isinstance(s, ef.CardSignal) for s in sigs))


if __name__ == "__main__":
    unittest.main()
