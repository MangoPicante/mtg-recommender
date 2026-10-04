"""Offline tests for mtg_recommender.oracle_tags.

Every HTTP call is mocked. Keeps parity with the scryfall_fetch suite in
style and run time — the full file should finish well under a second and
touch nothing outside TemporaryDirectory.

Run with:

    python -m unittest discover tests
    # or, more targeted:
    python -m unittest tests.test_oracle_tags

Test classes are grouped by concern:

    TestIdToSlug              _build_id_to_slug
    TestExtractTagFields      extract_tag_fields — projection + uuid->slug
    TestBuildTagCatalog       build_tag_catalog — end-to-end slug map
    TestBuildOracleIndex      build_oracle_id_to_slugs — inversion + dedup
    TestAttachTagsToCards     attach_tags_to_cards — join by oracle_id
    TestIsFresh               is_fresh — snapshot timestamp comparison
    TestTagCacheIO            load_tag_cache / save_tag_cache
    TestGetBulkTagsMetadata   /bulk-data filtering (mocked urlopen)
    TestDownloadBulkTags      gzip detection + JSONL parsing (mocked urlopen)
    TestMainCLI               main() end-to-end (mocked urlopen)
"""
from __future__ import annotations

import gzip
import io
import json
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

from mtg_recommender import oracle_tags as ot
from mtg_recommender import scryfall_fetch as sf


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# A compact tag graph used across tests. "removal" is a parent of both
# "spot-removal" and "mass-removal"; "evasion" stands alone. Keeps slugs
# human-readable so assertion failures are easy to read.
TAG_REMOVAL = {
    "object": "tag",
    "id": "u-removal",
    "label": "Removal",
    "slug": "removal",
    "type": "oracle",
    "description": "Cards that remove permanents.",
    "parent_ids": [],
    "child_ids": ["u-spot", "u-mass"],
    "aliases": [],
    "taggings": [],
}

TAG_SPOT = {
    "object": "tag",
    "id": "u-spot",
    "label": "Spot removal",
    "slug": "spot-removal",
    "type": "oracle",
    "description": "Removes a single permanent.",
    "parent_ids": ["u-removal"],
    "child_ids": [],
    "aliases": ["targeted-removal"],
    "taggings": [
        {"oracle_id": "oracle-bolt", "weight": "median"},
        {"oracle_id": "oracle-doom", "weight": "very_strong"},
    ],
}

TAG_MASS = {
    "object": "tag",
    "id": "u-mass",
    "label": "Mass removal",
    "slug": "mass-removal",
    "type": "oracle",
    "description": None,
    "parent_ids": ["u-removal"],
    "child_ids": [],
    "aliases": [],
    "taggings": [
        {"oracle_id": "oracle-wrath", "weight": "median", "annotation": "symmetric"},
    ],
}

TAG_EVASION = {
    "object": "tag",
    "id": "u-evasion",
    "label": "Evasion",
    "slug": "evasion",
    "type": "oracle",
    "description": "Hard to block.",
    "parent_ids": [],
    "child_ids": [],
    "aliases": [],
    "taggings": [
        {"oracle_id": "oracle-bolt", "weight": "median"},  # double-tagging: bolt gets both
    ],
}

ALL_TAGS = [TAG_REMOVAL, TAG_SPOT, TAG_MASS, TAG_EVASION]


def call_silent(fn, *args, **kwargs):
    """Invoke fn with stdout/stderr swallowed. Mirrors the scryfall_fetch tests."""
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        return fn(*args, **kwargs)


# ---------------------------------------------------------------------------
# _build_id_to_slug
# ---------------------------------------------------------------------------

class TestIdToSlug(unittest.TestCase):

    def test_builds_full_mapping(self):
        index = ot._build_id_to_slug(ALL_TAGS)
        self.assertEqual(
            index,
            {
                "u-removal": "removal",
                "u-spot": "spot-removal",
                "u-mass": "mass-removal",
                "u-evasion": "evasion",
            },
        )

    def test_skips_entries_missing_id_or_slug(self):
        # Defensive: a malformed line should be skipped, not crash the
        # whole import. Real Scryfall data has both fields on every tag,
        # but a schema drift should surface quietly.
        tags = ALL_TAGS + [{"id": "u-orphan"}, {"slug": "orphan"}]
        index = ot._build_id_to_slug(tags)
        self.assertNotIn("u-orphan", index)
        # No slug-keyed entries either — this is an id->slug map.
        self.assertEqual(len(index), 4)


# ---------------------------------------------------------------------------
# extract_tag_fields
# ---------------------------------------------------------------------------

class TestExtractTagFields(unittest.TestCase):

    def setUp(self):
        self.id_to_slug = ot._build_id_to_slug(ALL_TAGS)

    def test_projects_core_fields(self):
        out = ot.extract_tag_fields(TAG_SPOT, self.id_to_slug)
        self.assertEqual(out["id"], "u-spot")
        self.assertEqual(out["label"], "Spot removal")
        self.assertEqual(out["description"], "Removes a single permanent.")
        self.assertEqual(out["aliases"], ["targeted-removal"])

    def test_resolves_parent_uuids_to_slugs(self):
        out = ot.extract_tag_fields(TAG_SPOT, self.id_to_slug)
        # u-removal -> "removal"
        self.assertEqual(out["parent_slugs"], ["removal"])

    def test_resolves_child_uuids_to_slugs(self):
        out = ot.extract_tag_fields(TAG_REMOVAL, self.id_to_slug)
        self.assertEqual(sorted(out["child_slugs"]), ["mass-removal", "spot-removal"])

    def test_dangling_uuid_is_dropped(self):
        # A parent id that isn't in the index (e.g. mid-import schema
        # drift) should be silently dropped rather than left in the
        # output as a confusing uuid.
        raw = {
            "id": "u-x", "slug": "x", "label": "X", "description": None,
            "parent_ids": ["u-nonexistent", "u-removal"],
            "child_ids": [], "aliases": [], "taggings": [],
        }
        out = ot.extract_tag_fields(raw, self.id_to_slug)
        self.assertEqual(out["parent_slugs"], ["removal"])

    def test_strips_taggings(self):
        # Taggings are inverted into the cards cache, so they must NOT
        # appear in the catalog projection.
        out = ot.extract_tag_fields(TAG_SPOT, self.id_to_slug)
        self.assertNotIn("taggings", out)

    def test_missing_description_passes_through_as_none(self):
        out = ot.extract_tag_fields(TAG_MASS, self.id_to_slug)
        self.assertIsNone(out["description"])


# ---------------------------------------------------------------------------
# build_tag_catalog
# ---------------------------------------------------------------------------

class TestBuildTagCatalog(unittest.TestCase):

    def test_builds_slug_keyed_dict(self):
        catalog = ot.build_tag_catalog(ALL_TAGS)
        self.assertEqual(
            set(catalog.keys()),
            {"removal", "spot-removal", "mass-removal", "evasion"},
        )

    def test_catalog_entries_have_expected_shape(self):
        catalog = ot.build_tag_catalog(ALL_TAGS)
        entry = catalog["spot-removal"]
        self.assertEqual(
            set(entry.keys()),
            {"id", "label", "description", "parent_slugs", "child_slugs", "aliases"},
        )

    def test_entry_without_slug_is_skipped(self):
        # The catalog is slug-keyed, so a tag without a slug has nowhere
        # to go. Skip it rather than crash — same resilience as the
        # cards fetcher's "no id, no store" guard.
        tags = ALL_TAGS + [{"id": "u-noslug", "label": "No slug", "parent_ids": [], "child_ids": [], "taggings": []}]
        catalog = ot.build_tag_catalog(tags)
        self.assertEqual(len(catalog), 4)


# ---------------------------------------------------------------------------
# build_oracle_id_to_slugs
# ---------------------------------------------------------------------------

class TestBuildOracleIndex(unittest.TestCase):

    def test_inverts_tagging_lists(self):
        index = ot.build_oracle_id_to_slugs(ALL_TAGS)
        # oracle-bolt is tagged by both spot-removal and evasion.
        self.assertEqual(index["oracle-bolt"], ["evasion", "spot-removal"])
        # oracle-doom only by spot-removal.
        self.assertEqual(index["oracle-doom"], ["spot-removal"])
        # oracle-wrath only by mass-removal.
        self.assertEqual(index["oracle-wrath"], ["mass-removal"])

    def test_slugs_are_sorted_within_each_card(self):
        # Stable ordering keeps the saved cards cache diff-friendly.
        index = ot.build_oracle_id_to_slugs(ALL_TAGS)
        for slugs in index.values():
            self.assertEqual(slugs, sorted(slugs))

    def test_dedupes_double_taggings(self):
        # A single tag applied twice to the same oracle_id should not
        # appear twice in the output. Shouldn't happen in well-formed
        # bulk data but shouldn't blow up if it does.
        tag_dup = {
            "slug": "dup", "id": "u-dup",
            "taggings": [
                {"oracle_id": "oracle-x", "weight": "median"},
                {"oracle_id": "oracle-x", "weight": "median"},
            ],
        }
        index = ot.build_oracle_id_to_slugs([tag_dup])
        self.assertEqual(index["oracle-x"], ["dup"])

    def test_ignores_taggings_without_oracle_id(self):
        tag_broken = {
            "slug": "broken", "id": "u-b",
            "taggings": [{"oracle_id": None, "weight": "median"}],
        }
        index = ot.build_oracle_id_to_slugs([tag_broken])
        self.assertEqual(index, {})


# ---------------------------------------------------------------------------
# attach_tags_to_cards
# ---------------------------------------------------------------------------

class TestAttachTagsToCards(unittest.TestCase):

    def _cards_cache(self, *entries):
        return {"cards": {e["scryfall_id"]: e for e in entries}, "aliases": {}}

    def test_matches_by_oracle_id_and_attaches_slugs(self):
        cache = self._cards_cache(
            {"scryfall_id": "id-bolt", "oracle_id": "oracle-bolt", "name": "Bolt"},
            {"scryfall_id": "id-doom", "oracle_id": "oracle-doom", "name": "Doom"},
        )
        index = {"oracle-bolt": ["evasion", "spot-removal"], "oracle-doom": ["spot-removal"]}
        matched, empty, unmatched = ot.attach_tags_to_cards(cache, index)
        self.assertEqual(matched, 2)
        self.assertEqual(empty, 0)
        self.assertEqual(unmatched, 0)
        self.assertEqual(cache["cards"]["id-bolt"]["tags"], ["evasion", "spot-removal"])
        self.assertEqual(cache["cards"]["id-doom"]["tags"], ["spot-removal"])

    def test_card_without_oracle_id_gets_empty_tags(self):
        # Stale cache from before oracle_id was added to the projection:
        # we don't error out, we just give the card an empty tag list so
        # downstream consumers can rely on the field existing.
        cache = self._cards_cache(
            {"scryfall_id": "id-x", "name": "X"},  # no oracle_id
        )
        matched, empty, unmatched = ot.attach_tags_to_cards(cache, {"oracle-y": ["foo"]})
        self.assertEqual(matched, 0)
        self.assertEqual(empty, 1)
        self.assertEqual(cache["cards"]["id-x"]["tags"], [])

    def test_card_with_untagged_oracle_id_gets_empty_tags(self):
        cache = self._cards_cache(
            {"scryfall_id": "id-untagged", "oracle_id": "oracle-untagged", "name": "U"},
        )
        matched, empty, unmatched = ot.attach_tags_to_cards(cache, {"oracle-bolt": ["evasion"]})
        self.assertEqual(matched, 0)
        self.assertEqual(empty, 1)
        self.assertEqual(cache["cards"]["id-untagged"]["tags"], [])

    def test_overwrites_rather_than_merges(self):
        # A tag removed upstream must disappear from the card on the
        # next import. If we merged, stale tags would linger forever.
        cache = self._cards_cache(
            {
                "scryfall_id": "id-bolt", "oracle_id": "oracle-bolt",
                "tags": ["old-tag", "another-stale-tag"],
            },
        )
        ot.attach_tags_to_cards(cache, {"oracle-bolt": ["evasion"]})
        self.assertEqual(cache["cards"]["id-bolt"]["tags"], ["evasion"])

    def test_unmatched_count_reflects_oracle_ids_not_in_cache(self):
        # Tagger has three oracle_ids but the cards cache only contains
        # one of them — the other two count as unmatched.
        cache = self._cards_cache(
            {"scryfall_id": "id-bolt", "oracle_id": "oracle-bolt"},
        )
        index = {"oracle-bolt": ["evasion"], "oracle-doom": ["spot-removal"], "oracle-wrath": ["mass-removal"]}
        _, _, unmatched = ot.attach_tags_to_cards(cache, index)
        self.assertEqual(unmatched, 2)


# ---------------------------------------------------------------------------
# is_fresh
# ---------------------------------------------------------------------------

class TestIsFresh(unittest.TestCase):

    def test_match_is_fresh(self):
        cache = {"snapshot_updated_at": "2026-10-03T21:00:32.494+00:00", "tags": {}}
        self.assertTrue(ot.is_fresh(cache, "2026-10-03T21:00:32.494+00:00"))

    def test_mismatch_is_stale(self):
        cache = {"snapshot_updated_at": "2020-01-01T00:00:00+00:00", "tags": {}}
        self.assertFalse(ot.is_fresh(cache, "2026-10-03T21:00:32.494+00:00"))

    def test_empty_cache_is_stale(self):
        self.assertFalse(ot.is_fresh({"snapshot_updated_at": None, "tags": {}}, "any"))


# ---------------------------------------------------------------------------
# load_tag_cache / save_tag_cache
# ---------------------------------------------------------------------------

class TestTagCacheIO(unittest.TestCase):

    def test_missing_file_returns_fresh_shell(self):
        with TemporaryDirectory() as td:
            shell = ot.load_tag_cache(Path(td) / "nonexistent.json")
        self.assertEqual(shell, {"snapshot_updated_at": None, "tags": {}})

    def test_roundtrip_preserves_data(self):
        payload = {
            "snapshot_updated_at": "2026-10-03T21:00:32.494+00:00",
            "tags": {"spot-removal": {"id": "u-spot", "label": "Spot removal"}},
        }
        with TemporaryDirectory() as td:
            path = Path(td) / "cache" / "oracle_tags.json"  # tests parent mkdir
            ot.save_tag_cache(path, payload)
            self.assertTrue(path.exists())
            self.assertEqual(ot.load_tag_cache(path), payload)

    def test_load_fills_in_missing_keys(self):
        # A cache saved with a missing key (older schema) should load
        # as a valid shell — not crash the fetcher on the next run.
        with TemporaryDirectory() as td:
            path = Path(td) / "oracle_tags.json"
            path.write_text("{}", encoding="utf-8")
            loaded = ot.load_tag_cache(path)
        self.assertIsNone(loaded["snapshot_updated_at"])
        self.assertEqual(loaded["tags"], {})


# ---------------------------------------------------------------------------
# get_bulk_oracle_tags_metadata
# ---------------------------------------------------------------------------

class TestGetBulkTagsMetadata(unittest.TestCase):

    def test_picks_oracle_tags_entry(self):
        payload = {
            "data": [
                {"type": "oracle_cards", "updated_at": "x"},
                {"type": "oracle_tags", "updated_at": "y", "jsonl_download_uri": "http://z"},
                {"type": "art_tags", "updated_at": "w"},
            ],
        }
        with patch(
            "mtg_recommender.scryfall_fetch.urllib.request.urlopen",
            return_value=_json_response(payload),
        ):
            meta = ot.get_bulk_oracle_tags_metadata()
        self.assertEqual(meta["type"], "oracle_tags")
        self.assertEqual(meta["jsonl_download_uri"], "http://z")

    def test_raises_when_entry_absent(self):
        payload = {"data": [{"type": "oracle_cards"}]}
        with patch(
            "mtg_recommender.scryfall_fetch.urllib.request.urlopen",
            return_value=_json_response(payload),
        ):
            with self.assertRaises(RuntimeError):
                ot.get_bulk_oracle_tags_metadata()


# ---------------------------------------------------------------------------
# download_bulk_oracle_tags
# ---------------------------------------------------------------------------

class TestDownloadBulkTags(unittest.TestCase):

    def test_gzip_payload_is_decompressed(self):
        jsonl = "\n".join(json.dumps(t) for t in [TAG_SPOT, TAG_EVASION]).encode("utf-8")
        gzipped = gzip.compress(jsonl)
        meta = {"jsonl_download_uri": "http://x", "updated_at": "ts", "compressed_size": len(gzipped)}
        with patch(
            "mtg_recommender.oracle_tags.urllib.request.urlopen",
            return_value=_bytes_response(gzipped),
        ):
            tags = call_silent(ot.download_bulk_oracle_tags, meta)
        self.assertEqual([t["slug"] for t in tags], ["spot-removal", "evasion"])

    def test_plain_jsonl_is_parsed_without_decompression(self):
        # A proxy may have already decompressed the gzip; our magic-byte
        # check should pass the bytes through untouched.
        jsonl = (json.dumps(TAG_SPOT) + "\n" + json.dumps(TAG_EVASION) + "\n").encode("utf-8")
        meta = {"jsonl_download_uri": "http://x", "updated_at": "ts"}
        with patch(
            "mtg_recommender.oracle_tags.urllib.request.urlopen",
            return_value=_bytes_response(jsonl),
        ):
            tags = call_silent(ot.download_bulk_oracle_tags, meta)
        self.assertEqual(len(tags), 2)

    def test_blank_lines_skipped(self):
        jsonl = (json.dumps(TAG_SPOT) + "\n\n   \n" + json.dumps(TAG_EVASION) + "\n").encode("utf-8")
        meta = {"jsonl_download_uri": "http://x", "updated_at": "ts"}
        with patch(
            "mtg_recommender.oracle_tags.urllib.request.urlopen",
            return_value=_bytes_response(jsonl),
        ):
            tags = call_silent(ot.download_bulk_oracle_tags, meta)
        self.assertEqual(len(tags), 2)


# ---------------------------------------------------------------------------
# main — end-to-end CLI
# ---------------------------------------------------------------------------

class TestMainCLI(unittest.TestCase):

    def test_missing_cards_cache_errors(self):
        # No cards cache -> parser.error -> SystemExit(2). Nothing is
        # downloaded; the user needs to run scryfall-fetch first.
        with TemporaryDirectory() as td:
            tags_path = Path(td) / "oracle_tags.json"
            cards_path = Path(td) / "nonexistent.json"
            with patch("sys.argv", ["scryfall-fetch-tags", "--cards", str(cards_path), "--tags", str(tags_path)]):
                with self.assertRaises(SystemExit):
                    call_silent(ot.main)

    def test_happy_path_downloads_and_attaches(self):
        # Seed a cards cache with cards whose oracle_ids match two of
        # our fixture tags, then run main() with a mocked urlopen that
        # serves the /bulk-data metadata first and the gzipped bulk
        # payload second (urllib.request.urlopen is called twice).
        metadata_payload = {
            "data": [{
                "type": "oracle_tags",
                "jsonl_download_uri": "http://bulk",
                "updated_at": "2026-10-03T21:00:32.494+00:00",
                "compressed_size": 100,
            }],
        }
        jsonl = "\n".join(json.dumps(t) for t in ALL_TAGS).encode("utf-8")
        gzipped = gzip.compress(jsonl)

        with TemporaryDirectory() as td:
            tags_path = Path(td) / "oracle_tags.json"
            cards_path = Path(td) / "oracle_texts.json"
            cards_cache = {
                "cards": {
                    "id-bolt": {"scryfall_id": "id-bolt", "oracle_id": "oracle-bolt", "name": "Bolt"},
                    "id-wrath": {"scryfall_id": "id-wrath", "oracle_id": "oracle-wrath", "name": "Wrath"},
                    # Card not referenced by any fixture tag; should end
                    # with an empty tags list.
                    "id-other": {"scryfall_id": "id-other", "oracle_id": "oracle-other", "name": "Other"},
                },
                "aliases": {},
            }
            cards_path.write_text(json.dumps(cards_cache), encoding="utf-8")

            # Two sequential urlopen calls: metadata (JSON) first, then
            # bulk (gzipped JSONL). urllib.request is cached as a single
            # module object across our two source modules, so one patch
            # with a sequential side_effect covers both call sites.
            metadata_resp = _json_response(metadata_payload)
            bulk_resp = _bytes_response(gzipped)
            argv = ["scryfall-fetch-tags", "--cards", str(cards_path), "--tags", str(tags_path)]
            with patch(
                "mtg_recommender.scryfall_fetch.urllib.request.urlopen",
                side_effect=[metadata_resp, bulk_resp],
            ), patch("sys.argv", argv):
                rc = call_silent(ot.main)
            self.assertEqual(rc, 0)

            # Reload both caches from disk and assert end state. These
            # assertions MUST live inside the TemporaryDirectory `with`
            # block — once it exits, the directory and everything in it
            # are deleted.
            saved_tags = json.loads(tags_path.read_text(encoding="utf-8"))
            self.assertEqual(saved_tags["snapshot_updated_at"], "2026-10-03T21:00:32.494+00:00")
            self.assertEqual(
                set(saved_tags["tags"].keys()),
                {"removal", "spot-removal", "mass-removal", "evasion"},
            )
            saved_cards = json.loads(cards_path.read_text(encoding="utf-8"))
            self.assertEqual(saved_cards["cards"]["id-bolt"]["tags"], ["evasion", "spot-removal"])
            self.assertEqual(saved_cards["cards"]["id-wrath"]["tags"], ["mass-removal"])
            self.assertEqual(saved_cards["cards"]["id-other"]["tags"], [])


# ---------------------------------------------------------------------------
# Response-mock helpers
# ---------------------------------------------------------------------------

def _json_response(payload: dict) -> MagicMock:
    """A MagicMock that mimics urlopen's context-manager + .read() shape."""
    resp = MagicMock()
    resp.__enter__.return_value = resp
    resp.__exit__.return_value = False
    resp.read.return_value = json.dumps(payload).encode("utf-8")
    return resp


def _bytes_response(data: bytes) -> MagicMock:
    resp = MagicMock()
    resp.__enter__.return_value = resp
    resp.__exit__.return_value = False
    resp.read.return_value = data
    return resp


if __name__ == "__main__":
    unittest.main()
