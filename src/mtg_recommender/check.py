"""Mongo health-check CLI.

Verifies that the configured MongoDB cluster is reachable, authenticated,
has the expected database + indexes, and reports doc counts per
collection. Read-only; nothing is written. Suitable as a smoke test
after `.env` setup or as a one-off diagnostic when something looks off.

Exit code: 0 if every check passes, 1 if any check fails.

Usage (after `pip install -e .`):
    mtg-check
    mtg-check --verbose   # add pymongo/server version in the connectivity row
"""
from __future__ import annotations

import argparse
import sys
from typing import Iterable, Optional

import pymongo
from pymongo.errors import PyMongoError

from . import storage


# Indexes the rest of the package relies on. Keep in sync with
# `storage.ensure_indexes` — the check is the inverse of that function.
_REQUIRED_CARDS_INDEXES = ("names_lookup", "oracle_id_lookup")

# Fixed widths so rows line up regardless of status text length.
_LABEL_WIDTH = 18
_STATUS_WIDTH = 4


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def _row(label: str, status: str, detail: str = "") -> str:
    """One check row: `label  STATUS  detail`."""
    detail_part = f"  {detail}" if detail else ""
    return f"{label:<{_LABEL_WIDTH}}{status:<{_STATUS_WIDTH}}{detail_part}"


def _ok(label: str, detail: str = "") -> str:
    return _row(label, "OK", detail)


def _fail(label: str, detail: str = "") -> str:
    return _row(label, "FAIL", detail)


def _skip(label: str, detail: str = "") -> str:
    return _row(label, "SKIP", detail)


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def check_connectivity(verbose: bool = False) -> tuple[bool, str]:
    """Ping the Mongo server. Returns (passed, detail).

    `ping` is a cheap no-auth-required command that forces pymongo to
    actually open the connection (which is otherwise deferred until the
    first real op).
    """
    try:
        client = storage.get_client()
        client.admin.command("ping")
        if not verbose:
            return True, ""
        server_version = client.server_info().get("version", "?")
        return True, f"server {server_version}, pymongo {pymongo.__version__}"
    except (PyMongoError, RuntimeError) as e:
        # RuntimeError covers the "MONGODB_URI not set" case from storage.
        return False, f"{type(e).__name__}: {e}"


def check_database() -> tuple[bool, str]:
    """Confirm the configured db is addressable. Returns (passed, detail).

    Doesn't fail on an empty db — brand-new deployments are legitimate.
    """
    try:
        db = storage.get_database()
        # list_collection_names is a round-trip that would fail on auth or
        # permissions issues even if the connection succeeded.
        db.list_collection_names()
        return True, db.name
    except PyMongoError as e:
        return False, f"{type(e).__name__}: {e}"


def check_indexes() -> tuple[bool, str]:
    """Confirm the two indexes the fetchers rely on exist on the cards collection."""
    try:
        info = storage.cards_collection().index_information()
    except PyMongoError as e:
        return False, f"{type(e).__name__}: {e}"
    missing = [name for name in _REQUIRED_CARDS_INDEXES if name not in info]
    if missing:
        return False, f"missing: {', '.join(missing)}"
    return True, f"{', '.join(_REQUIRED_CARDS_INDEXES)} on cards"


def check_counts() -> tuple[bool, list[tuple[str, int]]]:
    """Return a doc count per collection. Any Mongo error fails the whole check."""
    try:
        rows = [
            (storage.cards_collection().name, storage.cards_collection().estimated_document_count()),
            (storage.tags_collection().name, storage.tags_collection().estimated_document_count()),
            (storage.meta_collection().name, storage.meta_collection().estimated_document_count()),
        ]
    except PyMongoError as e:
        return False, [("error", 0)] + [(type(e).__name__, 0)]
    return True, rows


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_checks(verbose: bool = False) -> int:
    """Run every check in order, print rows, return a shell-style exit code."""
    print("MongoDB health check")
    print("─" * 50)

    overall_ok = True

    # Connectivity first — if it fails, downstream checks skip since they
    # all require a live connection.
    conn_ok, conn_detail = check_connectivity(verbose=verbose)
    print(_ok("Connectivity", conn_detail) if conn_ok else _fail("Connectivity", conn_detail))

    if not conn_ok:
        print(_skip("Database"))
        print(_skip("Indexes"))
        print(_skip("Collections"))
        print()
        print("one or more checks failed")
        return 1

    db_ok, db_detail = check_database()
    print(_ok("Database", db_detail) if db_ok else _fail("Database", db_detail))
    overall_ok = overall_ok and db_ok

    idx_ok, idx_detail = check_indexes()
    print(_ok("Indexes", idx_detail) if idx_ok else _fail("Indexes", idx_detail))
    overall_ok = overall_ok and idx_ok

    counts_ok, rows = check_counts()
    if counts_ok:
        print("Collections")
        # Indent each collection row under the "Collections" header; the
        # label gets the per-row offset.
        for name, count in rows:
            print(f"  {_ok(name, f'{count:,} docs')}")
    else:
        print(_fail("Collections", rows[0][0]))
        overall_ok = False

    print()
    print("all checks passed" if overall_ok else "one or more checks failed")
    return 0 if overall_ok else 1


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="mtg-check",
        description="Verify the configured MongoDB cluster is reachable and healthy.",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Include server/pymongo versions in the connectivity row.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    return run_checks(verbose=args.verbose)


if __name__ == "__main__":
    sys.exit(main())
