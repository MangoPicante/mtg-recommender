# MTG Recommender

A tool that takes a Magic: The Gathering decklist and recommends cards for it,
grounded in Scryfall oracle text. Format scope for v1 is **Commander only**;
scoring is **similarity** ("plays like"), not synergy. See [PLAN.md](PLAN.md)
for the full roadmap.

## Current state

Phase 1 is partially in place: Scryfall oracle text can be fetched and cached
locally, and a decklist-scoped subset can be exported. The persistence layer
(MongoDB Atlas), oracletag acquisition, and everything downstream of that
(embeddings, recommendation, UI) are still TODO.

### Modules

Under `src/mtg_recommender/`:

| Module | What it does |
| --- | --- |
| `scryfall_fetch` | Fetches Scryfall oracle text for one card or a decklist, caching to `./cache/oracle_texts.json`. Switches between `/cards/named` (single card) and the oracle-cards bulk download (2+ cards) automatically. Cache is id-keyed with a list-valued name alias index so ambiguous names (art variants, meld pieces, shared face names) surface rather than silently resolve. |
| `extract_oracle` | Writes a trimmed per-decklist subset of the cache for downstream consumers. |

Tests live under `tests/` and are offline — every HTTP call is mocked.

## Setup

Python 3.10+ is required (modules use `from __future__ import annotations` plus
3.10-style union syntax in a few spots). The package is pure stdlib for now —
no runtime dependencies — but is set up as an installable PEP 621 package so
`pip install -e .` registers the two CLI commands and keeps the layout honest
when real deps start arriving in Phase 1.

```bash
# recommended: isolated venv
python -m venv .venv

# PowerShell:
.venv\Scripts\Activate.ps1
# bash/zsh:
source .venv/bin/activate

# installs the package in editable mode and registers the console scripts
pip install -e .
```

A `.env.example` is provided for future configuration (e.g. the Mongo Atlas
connection string that lands in Phase 1). Copy it to `.env` when wiring that
up — `.env` itself is gitignored.

```bash
cp .env.example .env
```

## Usage

Both CLIs are installed as console scripts by `pip install -e .`. Equivalent
`python -m mtg_recommender.<module>` forms also work from a source checkout.

### Fetch oracle text

```bash
# single card -> /cards/named (fast, one request)
scryfall-fetch "Lightning Bolt"

# multiple cards -> oracle-cards bulk download, merged into the shared cache
scryfall-fetch "Lightning Bolt" "Counterspell"

# from a decklist file (one card per line, '#' for comments; Moxfield / Arena /
# MTGGoldfish export formats are tolerated)
scryfall-fetch --file deck.txt

# force refetch / redownload even if the cache looks fresh
scryfall-fetch --file deck.txt --refresh
```

The cache is written to `./cache/oracle_texts.json` (relative to the current
working directory) and is gitignored. Override the location with `--cache
/some/other/path.json`. Shape and semantics are documented in the top-of-file
docstring of `src/mtg_recommender/scryfall_fetch.py`.

### Extract a decklist subset

```bash
# writes ./oracle_subset.json by default
extract-oracle --file deck.txt

# or specify output
extract-oracle --file deck.txt -o deck_oracle.json
```

The subset drops cache-only fields (`updated_at`) and keeps the downstream-
facing ones (`name`, `mana_cost`, `type_line`, `oracle_text`, `scryfall_id`).

## Testing

```bash
python -m unittest discover tests
```

Every HTTP call in the test suite is mocked — the tests do not touch the
network and will run offline. 61 tests, well under a second total.

## Project layout

```
mtg-recommender/
├── src/
│   └── mtg_recommender/        # installable package
│       ├── __init__.py
│       ├── scryfall_fetch.py   # fetcher + unified cache
│       └── extract_oracle.py   # per-decklist subset exporter
├── tests/
│   └── test_scryfall_fetch.py  # offline unit tests
├── cache/                      # gitignored; created at runtime
├── pyproject.toml              # PEP 621 metadata, build config, entry points
├── .env.example                # placeholder for Phase 1 Mongo URI etc.
├── CLAUDE.md                   # workflow + style conventions
├── PLAN.md                     # scope, roadmap, open questions
└── README.md                   # you are here
```

The `src/` layout (PyPA-recommended) prevents accidental imports from the repo
root — code must go through the installed package, which catches packaging
bugs early.

## Workflow

Branch per concern, rebase-merge only, tests in the same commit as the feature.
See [CLAUDE.md](CLAUDE.md) for the full set of conventions.
