# justfile — common commands for the MTG recommender.
#
# Install `just` from https://just.systems. `just` with no target lists
# every recipe. Recipes assume `just install` has been run — the editable
# install registers the mtg-* and scryfall-* console scripts that most
# recipes delegate to. Pass extra flags through to the underlying CLIs;
# each CLI's `--help` is the authoritative reference.
#
# Recipes run through bash. On Windows this points at Git Bash, which
# ships with git-for-windows and is on PATH by default.

set windows-shell := ["bash.exe", "-c"]


# List every recipe (default when no target is given).
default:
    @just --list


# Install the package in editable mode with dev extras.
install:
    pip install -e ".[dev]"

# Remove local caches and generated outputs. Does NOT touch Mongo.
clean:
    rm -rf cache
    rm -f oracle_subset.json
    find . -name __pycache__ -type d -prune -exec rm -rf {} +


# Lint the codebase with ruff (CI runs this too).
lint:
    ruff check .

# Apply ruff's auto-fixes where it can.
lint-fix:
    ruff check . --fix

# Run the offline test suite (set MONGODB_INTEGRATION_URI for the real suite).
test:
    python -m unittest discover tests

# Same as `test`, one line per test case.
test-verbose:
    python -m unittest discover tests -v


# Verify the Mongo cluster is reachable and healthy (`mtg-check` wrapper).
check *ARGS:
    mtg-check {{ARGS}}

# Fetch oracle text into Mongo (`scryfall-fetch` wrapper; pass cards or --file).
fetch *ARGS:
    scryfall-fetch {{ARGS}}

# Download oracle_tags bulk and attach tags to cards (`scryfall-fetch-tags`).
tags *ARGS:
    scryfall-fetch-tags {{ARGS}}

# Inspect what's in Mongo (`mtg-inspect` wrapper: card / tag / list / stats).
inspect *ARGS:
    mtg-inspect {{ARGS}}

# Encode oracle text + tags into dense vectors on their Mongo docs
# (`mtg-embed` wrapper: cards / tags, --refresh, --limit, --batch-size).
# Requires `pip install -e ".[embeddings]"`.
embed *ARGS:
    mtg-embed {{ARGS}}

# Export a trimmed JSON subset for a decklist file (`extract-oracle` wrapper).
extract FILE *ARGS:
    extract-oracle --file {{FILE}} {{ARGS}}
