"""Offline tests for mtg_recommender.oracle_tags.

Every HTTP call is mocked and every Mongo op routes through mongomock,
so the suite is deterministic and offline. Parity with the scryfall_fetch
test suite in style and run time — the full file finishes well under a
second.

Run with:

    python -m unittest discover tests
    # or, more targeted:
    python -m unittest tests.test_oracle_tags

Test classes are grouped by concern:

    TestIdToSlug              _build_id_to_slug
    TestExtractTagFields      extract_tag_fields — projection + uuid->slug
    TestBuildTagCatalog       build_tag_catalog — end-to-end slug map
    TestBuildOracleIndex      build_oracle_id_to_slugs — inversion + dedup
    TestUpsertTagCatalog      Mongo-write path for the catalog
    TestAttachTagsToCards     attach_tags_to_cards — Mongo join by oracle_id
    TestGetBulkTagsMetadata   /bulk-data filtering (mocked urlopen)
    TestDownloadBulkTags      gzip detection + JSONL parsing (mocked urlopen)
    TestMainCLI               main() end-to-end (mongomock + mocked urlopen)
"""
from __future__ import annotations

import gzip
import io
import json
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import MagicMock, patch

import mongomock

from mtg_recommender import oracle_tags as ot
from mtg_recommender import storage

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
    """Invoke fn with stdout/stderr swallowed."""
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        return fn(*args, **kwargs)


class _MongoBackedTestCase(unittest.TestCase):
    """Shared setup: fresh mongomock client per test, indexes pre-created."""

    def setUp(self):
        storage.reset_client(mongomock.MongoClient())
        storage.ensure_indexes()
        self.cards = storage.cards_collection()
        self.tags = storage.tags_collection()
        self.meta = storage.meta_collection()

    def tearDown(self):
        storage.reset_client(None)


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
        tags = ALL_TAGS + [{"id": "u-orphan"}, {"slug": "orphan"}]
        index = ot._build_id_to_slug(tags)
        self.assertNotIn("u-orphan", index)
        self.assertEqual(len(index), 4)


# ---------------------------------------------------------------------------
# extract_tag_fields
# ---------------------------------------------------------------------------

class TestExtractTagFields(unittest.TestCase):

    def setUp(self):
        self.id_to_slug = ot._build_id_to_slug(ALL_TAGS)

    def test_projects_core_fields(self):
        out = ot.extract_tag_fields(TAG_SPOT, self.id_to_slug)
        self.assertEqual(out["scryfall_tag_id"], "u-spot")
        self.assertEqual(out["label"], "Spot removal")
        self.assertEqual(out["description"], "Removes a single permanent.")
        self.assertEqual(out["aliases"], ["targeted-removal"])

    def test_renames_raw_id_to_scryfall_tag_id(self):
        # Raw Scryfall uses `id` for the tag UUID; the projection renames
        # it so it doesn't look like an attempt to set Mongo's reserved
        # _id field once the doc lands in a collection.
        out = ot.extract_tag_fields(TAG_SPOT, self.id_to_slug)
        self.assertNotIn("id", out)
        self.assertIn("scryfall_tag_id", out)

    def test_resolves_parent_uuids_to_slugs(self):
        out = ot.extract_tag_fields(TAG_SPOT, self.id_to_slug)
        self.assertEqual(out["parent_slugs"], ["removal"])

    def test_resolves_child_uuids_to_slugs(self):
        out = ot.extract_tag_fields(TAG_REMOVAL, self.id_to_slug)
        self.assertEqual(sorted(out["child_slugs"]), ["mass-removal", "spot-removal"])

    def test_dangling_uuid_is_dropped(self):
        raw = {
            "id": "u-x", "slug": "x", "label": "X", "description": None,
            "parent_ids": ["u-nonexistent", "u-removal"],
            "child_ids": [], "aliases": [], "taggings": [],
        }
        out = ot.extract_tag_fields(raw, self.id_to_slug)
        self.assertEqual(out["parent_slugs"], ["removal"])

    def test_strips_taggings(self):
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
            {"scryfall_tag_id", "label", "description", "parent_slugs", "child_slugs", "aliases"},
        )

    def test_entry_without_slug_is_skipped(self):
        tags = ALL_TAGS + [{"id": "u-noslug", "label": "No slug", "parent_ids": [], "child_ids": [], "taggings": []}]
        catalog = ot.build_tag_catalog(tags)
        self.assertEqual(len(catalog), 4)


# ---------------------------------------------------------------------------
# build_oracle_id_to_slugs
# ---------------------------------------------------------------------------

class TestBuildOracleIndex(unittest.TestCase):

    def test_inverts_tagging_lists(self):
        index = ot.build_oracle_id_to_slugs(ALL_TAGS)
        self.assertEqual(index["oracle-bolt"], ["evasion", "spot-removal"])
        self.assertEqual(index["oracle-doom"], ["spot-removal"])
        self.assertEqual(index["oracle-wrath"], ["mass-removal"])

    def test_slugs_are_sorted_within_each_card(self):
        index = ot.build_oracle_id_to_slugs(ALL_TAGS)
        for slugs in index.values():
            self.assertEqual(slugs, sorted(slugs))

    def test_dedupes_double_taggings(self):
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
# upsert_tag_catalog (Mongo write)
# ---------------------------------------------------------------------------

class TestUpsertTagCatalog(_MongoBackedTestCase):

    def test_inserts_slug_keyed_docs(self):
        catalog = ot.build_tag_catalog(ALL_TAGS)
        ot.upsert_tag_catalog(self.tags, catalog)
        self.assertEqual(self.tags.count_documents({}), 4)
        spot = self.tags.find_one({"_id": "spot-removal"})
        self.assertIsNotNone(spot)
        self.assertEqual(spot["label"], "Spot removal")
        self.assertEqual(spot["parent_slugs"], ["removal"])

    def test_replaces_existing_catalog_contents(self):
        # Seed with a tag that's no longer in the authoritative catalog;
        # it must be gone after upsert.
        self.tags.insert_one({"_id": "stale", "label": "stale"})
        ot.upsert_tag_catalog(self.tags, ot.build_tag_catalog(ALL_TAGS))
        self.assertIsNone(self.tags.find_one({"_id": "stale"}))
        self.assertEqual(self.tags.count_documents({}), 4)

    def test_empty_catalog_clears_collection(self):
        self.tags.insert_one({"_id": "x"})
        ot.upsert_tag_catalog(self.tags, {})
        self.assertEqual(self.tags.count_documents({}), 0)


# ---------------------------------------------------------------------------
# attach_tags_to_cards (Mongo write + stats)
# ---------------------------------------------------------------------------

class TestAttachTagsToCards(_MongoBackedTestCase):

    def _seed_cards(self, *docs):
        for d in docs:
            d.setdefault("tags", [])
            self.cards.insert_one(d)

    def test_matches_by_oracle_id_and_attaches_slugs(self):
        self._seed_cards(
            {"_id": "id-bolt", "oracle_id": "oracle-bolt", "name": "Bolt"},
            {"_id": "id-doom", "oracle_id": "oracle-doom", "name": "Doom"},
        )
        index = {"oracle-bolt": ["evasion", "spot-removal"], "oracle-doom": ["spot-removal"]}
        changed, matched, empty, unmatched = ot.attach_tags_to_cards(self.cards, index)
        self.assertEqual(changed, 2)
        self.assertEqual(matched, 2)
        self.assertEqual(empty, 0)
        self.assertEqual(unmatched, 0)
        self.assertEqual(
            self.cards.find_one({"_id": "id-bolt"})["tags"],
            ["evasion", "spot-removal"],
        )
        self.assertEqual(
            self.cards.find_one({"_id": "id-doom"})["tags"],
            ["spot-removal"],
        )

    def test_card_without_oracle_id_gets_empty_tags(self):
        self._seed_cards(
            {"_id": "id-x", "name": "X"},  # no oracle_id
        )
        _, matched, empty, _ = ot.attach_tags_to_cards(self.cards, {"oracle-y": ["foo"]})
        self.assertEqual(matched, 0)
        self.assertEqual(empty, 1)
        self.assertEqual(self.cards.find_one({"_id": "id-x"})["tags"], [])

    def test_card_with_untagged_oracle_id_gets_empty_tags(self):
        self._seed_cards(
            {"_id": "id-untagged",
             "oracle_id": "oracle-untagged", "name": "U"},
        )
        _, matched, empty, _ = ot.attach_tags_to_cards(
            self.cards, {"oracle-bolt": ["evasion"]}
        )
        self.assertEqual(matched, 0)
        self.assertEqual(empty, 1)
        self.assertEqual(self.cards.find_one({"_id": "id-untagged"})["tags"], [])

    def test_overwrites_rather_than_merges(self):
        # A tag removed upstream must disappear from the card on the
        # next import. If we merged, stale tags would linger forever.
        self._seed_cards(
            {"_id": "id-bolt", "oracle_id": "oracle-bolt",
             "tags": ["old-tag", "another-stale-tag"]},
        )
        ot.attach_tags_to_cards(self.cards, {"oracle-bolt": ["evasion"]})
        self.assertEqual(self.cards.find_one({"_id": "id-bolt"})["tags"], ["evasion"])

    def test_unmatched_count_reflects_oracle_ids_not_in_collection(self):
        self._seed_cards(
            {"_id": "id-bolt", "oracle_id": "oracle-bolt"},
        )
        index = {"oracle-bolt": ["evasion"], "oracle-doom": ["spot-removal"],
                 "oracle-wrath": ["mass-removal"]}
        _, _, _, unmatched = ot.attach_tags_to_cards(self.cards, index)
        self.assertEqual(unmatched, 2)

    def test_unchanged_oracle_id_produces_no_write(self):
        # Card already has exactly the tags the new mapping would set —
        # the diff should classify as unchanged (changed_oids = 0) and
        # leave `card_vector` intact.
        self._seed_cards(
            {"_id": "id-bolt", "oracle_id": "oracle-bolt",
             "tags": ["evasion"], "card_vector": [0.0] * 4},
        )
        changed, _, _, _ = ot.attach_tags_to_cards(
            self.cards, {"oracle-bolt": ["evasion"]}
        )
        self.assertEqual(changed, 0)
        # Content unchanged → no $unset — card_vector survives.
        self.assertEqual(
            self.cards.find_one({"_id": "id-bolt"})["card_vector"], [0.0] * 4
        )

    def test_changed_oracle_id_unsets_card_vector(self):
        # Card has existing card_vector; the new tags differ, so the
        # fused vector is stale and gets cleared.
        self._seed_cards(
            {"_id": "id-bolt", "oracle_id": "oracle-bolt",
             "tags": ["old-tag"], "card_vector": [0.0] * 4},
        )
        changed, _, _, _ = ot.attach_tags_to_cards(
            self.cards, {"oracle-bolt": ["evasion"]}
        )
        self.assertEqual(changed, 1)
        doc = self.cards.find_one({"_id": "id-bolt"})
        self.assertEqual(doc["tags"], ["evasion"])
        self.assertNotIn("card_vector", doc)

    def test_card_missing_tags_field_gets_empty_default(self):
        # Simulates a card freshly added by scryfall-fetch after a prior
        # tag import: the doc has no `tags` field at all. Attach must
        # ensure every card ends up with `tags` set so downstream code
        # can rely on it.
        self.cards.insert_one(
            {"_id": "id-new", "oracle_id": "oracle-new"}
        )
        ot.attach_tags_to_cards(self.cards, {"oracle-other": ["foo"]})
        self.assertEqual(self.cards.find_one({"_id": "id-new"})["tags"], [])


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
# main — end-to-end CLI (mongomock + mocked urlopen)
# ---------------------------------------------------------------------------

class TestMainCLI(_MongoBackedTestCase):

    def test_empty_cards_collection_errors(self):
        # Cards collection empty -> parser.error -> SystemExit(2). No
        # download attempted; user needs to run scryfall-fetch first.
        with patch("sys.argv", ["scryfall-fetch-tags"]):
            with self.assertRaises(SystemExit):
                call_silent(ot.main)

    def test_happy_path_downloads_and_attaches(self):
        # Seed a cards collection with cards whose oracle_ids match some
        # fixture tags, then run main() with a mocked urlopen that
        # serves the /bulk-data metadata first and the gzipped bulk
        # payload second.
        metadata_payload = {
            "data": [{
                "type": "oracle_tags",
                "jsonl_download_uri": "http://bulk",
                "updated_at": "2026-10-03T21:00:32.494+00:00",
                "compressed_size": 100,
            }],
        }
        jsonl = "\n".join(json.dumps(t) for t in ALL_TAGS).encode()
        gzipped = gzip.compress(jsonl)

        self.cards.insert_many([
            {"_id": "id-bolt", "oracle_id": "oracle-bolt", "name": "Bolt"},
            {"_id": "id-wrath", "oracle_id": "oracle-wrath", "name": "Wrath"},
            # No fixture tag applies to this card; should end with [].
            {"_id": "id-other", "oracle_id": "oracle-other", "name": "Other"},
        ])

        with patch(
            "mtg_recommender.scryfall_fetch.urllib.request.urlopen",
            side_effect=[_json_response(metadata_payload), _bytes_response(gzipped)],
        ), patch("sys.argv", ["scryfall-fetch-tags"]):
            rc = call_silent(ot.main)
        self.assertEqual(rc, 0)

        # Catalog landed in Mongo.
        self.assertEqual(self.tags.count_documents({}), 4)
        self.assertEqual(
            {t["_id"] for t in self.tags.find({}, {"_id": 1})},
            {"removal", "spot-removal", "mass-removal", "evasion"},
        )

        # Cards got the right tags.
        self.assertEqual(
            self.cards.find_one({"_id": "id-bolt"})["tags"],
            ["evasion", "spot-removal"],
        )
        self.assertEqual(
            self.cards.find_one({"_id": "id-wrath"})["tags"],
            ["mass-removal"],
        )
        self.assertEqual(self.cards.find_one({"_id": "id-other"})["tags"], [])

        # Snapshot timestamp recorded for the freshness check.
        self.assertEqual(
            storage.get_snapshot_timestamp(ot.META_SOURCE),
            "2026-10-03T21:00:32.494+00:00",
        )

    def test_fresh_snapshot_skips_download(self):
        # Pre-record the current snapshot timestamp. Running main() with
        # a bulk-data response claiming the same timestamp should skip
        # the download (second urlopen call) entirely. The seeded card
        # has `tags: []` already — if that field were missing, the
        # untagged-cards auto-detect would force a redownload.
        self.cards.insert_one(
            {"_id": "id-x", "oracle_id": "oracle-x", "tags": []}
        )
        storage.set_snapshot_timestamp(ot.META_SOURCE, "2026-10-03T21:00:32.494+00:00")
        metadata_payload = {
            "data": [{
                "type": "oracle_tags",
                "jsonl_download_uri": "http://bulk",
                "updated_at": "2026-10-03T21:00:32.494+00:00",
                "compressed_size": 100,
            }],
        }
        with patch(
            "mtg_recommender.scryfall_fetch.urllib.request.urlopen",
            side_effect=[_json_response(metadata_payload)],
        ), patch("sys.argv", ["scryfall-fetch-tags"]):
            rc = call_silent(ot.main)
        self.assertEqual(rc, 0)
        # tags collection untouched because the download was skipped.
        self.assertEqual(self.tags.count_documents({}), 0)

    def test_untagged_cards_trigger_redownload_even_with_matching_snapshot(self):
        # The raw taggings aren't retained after import, so when a prior
        # scryfall-fetch added new cards with no `tags` field, we have
        # to redownload to tag them — even if the oracle_tags snapshot
        # hasn't moved.
        self.cards.insert_one(
            {"_id": "id-bolt", "oracle_id": "oracle-bolt"}
        )  # no `tags` field
        storage.set_snapshot_timestamp(ot.META_SOURCE, "2026-10-03T21:00:32.494+00:00")
        metadata_payload = {
            "data": [{
                "type": "oracle_tags",
                "jsonl_download_uri": "http://bulk",
                "updated_at": "2026-10-03T21:00:32.494+00:00",
                "compressed_size": 100,
            }],
        }
        jsonl = json.dumps(TAG_EVASION).encode()
        gzipped = gzip.compress(jsonl)

        with patch(
            "mtg_recommender.scryfall_fetch.urllib.request.urlopen",
            side_effect=[_json_response(metadata_payload), _bytes_response(gzipped)],
        ), patch("sys.argv", ["scryfall-fetch-tags"]):
            rc = call_silent(ot.main)
        self.assertEqual(rc, 0)
        # Download fired and attached despite matching snapshot timestamp.
        self.assertEqual(self.tags.count_documents({}), 1)
        self.assertEqual(
            self.cards.find_one({"_id": "id-bolt"})["tags"], ["evasion"]
        )


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
