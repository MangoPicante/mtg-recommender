"""mtg_recommender — Magic: The Gathering card recommender grounded in Scryfall oracle text.

Public surface area is intentionally thin for now: the package exists mainly to
host the console-script entry points declared in `pyproject.toml`. See
PLAN.md for the roadmap.
"""
from importlib.metadata import PackageNotFoundError, version

try:
    # Single source of truth is pyproject.toml; this reads it back from the
    # installed package's metadata so the two can't drift.
    __version__ = version("mtg-recommender")
except PackageNotFoundError:
    # Package isn't installed (e.g. running from the source tree without
    # `pip install -e .`). Return a sentinel rather than raising so the
    # package can still be imported for ad-hoc exploration.
    __version__ = "0.0.0+unknown"
