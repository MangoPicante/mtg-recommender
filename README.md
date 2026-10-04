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

| File | What it does |
| --- | --- |
| `scryfall_fetch.py` | Fetches Scryfall oracle text for one card or a decklist, caching to `cache/oracle_texts.json`. Switches between `/cards/named` (single card) and the oracle-cards bulk download (2+ cards) automatically. Cache is id-keyed with a list-valued name alias index so ambiguous names (art variants, meld pieces, shared face names) surface rather than silently resolve. |
| `extract_oracle.py` | Writes a trimmed per-decklist subset of the cache for downstream consumers. |
| `test_scryfall_fetch.py` | Offline unit tests for the fetcher — every HTTP call is mocked. |

## Setup

Python 3.10+ is required (the modules use `from __future__ import annotations`
plus 3.10-style union syntax in a few spots). No external dependencies are
needed yet — the fetcher is intentionally stdlib-only.

```bash
# optional but recommended
python -m venv .venv
# PowerShell:
.venv\Scripts\Activate.ps1
# bash/zsh:
source .venv/bin/activate

# dependencies will arrive as later phases land (see requirements.txt)
pip install -r requirements.txt
```

A `.env.example` is provided for future configuration (e.g. the Mongo Atlas
connection string that lands in Phase 1). Copy it to `.env` when wiring that up:

```bash
cp .env.example .env
```

## Usage

### Fetch oracle text

```bash
# single card -> /cards/named (fast, one request)
python scryfall_fetch.py "Lightning Bolt"

# multiple cards -> oracle-cards bulk download, merged into the shared cache
python scryfall_fetch.py "Lightning Bolt" "Counterspell"

# from a decklist file (one card per line, '#' for comments; Moxfield / Arena /
# MTGGoldfish export formats are tolerated)
python scryfall_fetch.py --file deck.txt

# force refetch / redownload even if the cache looks fresh
python scryfall_fetch.py --file deck.txt --refresh
```

The cache is written to `cache/oracle_texts.json` and is gitignored. Shape and
semantics are documented in the top-of-file docstring of `scryfall_fetch.py`.

### Extract a decklist subset

```bash
# writes ./oracle_subset.json by default
python extract_oracle.py --file deck.txt

# or specify output
python extract_oracle.py --file deck.txt -o deck_oracle.json
```

The subset drops cache-only fields (`updated_at`) and keeps the downstream-
facing ones (`name`, `mana_cost`, `type_line`, `oracle_text`, `scryfall_id`).

## Testing

```bash
python -m unittest discover
```

Every HTTP call in the test suite is mocked — the tests do not touch the
network and will run offline.

## Project layout

```
mtg-recommender/
├── cache/                  # gitignored; populated by scryfall_fetch.py
├── scryfall_fetch.py       # Scryfall fetcher + unified cache
├── extract_oracle.py       # per-decklist subset exporter
├── test_scryfall_fetch.py  # offline unit tests
├── requirements.txt        # Python dependencies (empty for now)
├── .env.example            # placeholder for Phase 1 Mongo URI etc.
├── CLAUDE.md               # workflow + style conventions
├── PLAN.md                 # scope, roadmap, open questions
└── README.md               # you are here
```

## Workflow

Branch per concern, rebase-merge only, tests in the same commit as the feature.
See [CLAUDE.md](CLAUDE.md) for the full set of conventions.
