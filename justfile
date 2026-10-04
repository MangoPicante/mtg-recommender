# justfile — common commands for the MTG recommender.
#
# Install `just` from https://just.systems. `just` with no target lists
# every recipe. Recipes assume `just install` has been run — the editable
# install registers the mtg-* and scryfall-* console scripts that most
# recipes delegate to. Pass extra flags through to the underlying CLIs;
# each CLI's `--help` is the authoritative reference.
#
# Recipes use POSIX-shell syntax. On Windows we point at `sh` rather
# than `bash.exe`: `bash.exe` on PATH almost always resolves to the WSL
# shim at C:\Windows\System32\bash.exe, which fails with
# `execvpe(/bin/bash)` when no WSL distro is installed. Windows ships
# no System32-level `sh.exe`, so `sh` on PATH reliably picks up Git
# for Windows's POSIX shell (`C:\Program Files\Git\bin\sh.exe`) — and
# scoop/choco installs of just pull Git Bash in automatically. If
# `sh` is missing from PATH, add Git's bin dir to PATH or install
# Git for Windows.

set windows-shell := ["sh", "-c"]


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

# Run the offline test suite.
test:
    python -m unittest discover tests

# Same as `test`, one line per test case.
test-verbose:
    python -m unittest discover tests -v

# Verify the Mongo cluster is reachable and healthy (`mtg-check` wrapper).
check *ARGS:
    mtg-check {{ARGS}}

# Fetch oracle text into Mongo (`scryfall-fetch` wrapper; idempotent + diff-aware, no flags).
fetch:
    scryfall-fetch

# Download oracle_tags bulk and attach tags to cards (`scryfall-fetch-tags`; idempotent, no flags).
tags:
    scryfall-fetch-tags

# Inspect what's in Mongo (`mtg-inspect` wrapper: card / tag / list / stats).
inspect *ARGS:
    mtg-inspect {{ARGS}}

# Encode text + tags into dense vectors (`mtg-embed` wrapper: cards / tags / flags).
embed *ARGS:
    mtg-embed {{ARGS}}

# Smoke-test the embedding pipeline end-to-end on 20 cards.
embed-smoke:
    mtg-embed cards --limit 20

# Fuse text + aggregated tag embeddings into card_vector (`mtg-embed fuse` wrapper).
fuse *ARGS:
    mtg-embed fuse {{ARGS}}

# Populate Mongo end-to-end: fetch → tag → embed cards + tags → fuse. Idempotent and diff-aware — reruns re-encode only cards whose oracle_text or tags changed; `--refresh` on an embed step forces that step.
populate *ARGS:
    scryfall-fetch
    scryfall-fetch-tags
    mtg-embed cards {{ARGS}}
    mtg-embed tags {{ARGS}}
    mtg-embed fuse {{ARGS}}

# Cluster a decklist's tags into themes (`mtg-deck-profile` wrapper). Pass a decklist with --file or positional names.
profile *ARGS:
    mtg-deck-profile {{ARGS}}

# Export a trimmed JSON subset for a decklist file (`extract-oracle` wrapper).
extract FILE *ARGS:
    extract-oracle --file {{FILE}} {{ARGS}}
