# MTG Recommender

A tool that takes a Magic: The Gathering decklist and recommends cards for it,
grounded in Scryfall oracle text. Format scope for v1 is **Commander only**;
scoring is **similarity** ("plays like"), not synergy. See [PLAN.md](PLAN.md)
for the full roadmap.

## Current state

Phase 1 is done: Scryfall oracle text and oracle tags both land in MongoDB
Atlas (two collections, `cards` and `tags`, with a `meta` collection for
snapshot timestamps). Everything downstream — embeddings, recommendation,
UI — is still TODO.

### Modules

Under `src/mtg_recommender/`:

| Module | What it does |
| --- | --- |
| `storage` | Thin wrapper around pymongo. Reads `MONGODB_URI` (and optional overrides) from `.env`, caches a single `MongoClient` per process, exposes handles to the `cards` / `tags` / `meta` collections, and declares the indexes the rest of the package relies on (`cards.names` for alias lookup, `cards.oracle_id` for the tag join). |
| `scryfall_fetch` | Fetches Scryfall oracle text for one card or a decklist into the `cards` collection. Switches between `/cards/named` (single card) and the oracle-cards bulk download (2+ cards) automatically. Each card document is keyed by `scryfall_id` with a `names: [lowered, ...]` array for alias lookup. The `tags` field is left for `oracle_tags` to populate. |
| `oracle_tags` | Downloads the Scryfall oracle-tags bulk, writes a slug-keyed catalog to the `tags` collection (hierarchy + aliases + descriptions preserved), and attaches `tags: [slug, ...]` arrays to each card by joining on `oracle_id`. |
| `extract_oracle` | Writes a trimmed per-decklist JSON subset of the cards collection for downstream consumers that don't speak Mongo. |
| `inspect` | Read-only ad-hoc inspection CLI. Subcommands: `card`, `tag`, `list`, `stats`. Human-readable output for one-off exploration and debugging. |
| `check` | Read-only health check. Pings the Mongo server, verifies the configured database + indexes, and reports a doc count per collection. Exit 0 on success, 1 if any check fails. |

Tests live under `tests/` and are offline — every HTTP call is mocked, every
Mongo op goes through `mongomock`.

## Setup

Python 3.10+ is required. The package has two runtime dependencies today
(`pymongo`, `python-dotenv`); heavier libraries arrive with Phase 2 embeddings.

```bash
# recommended: isolated venv
python -m venv .venv

# PowerShell:
.venv\Scripts\Activate.ps1
# bash/zsh:
source .venv/bin/activate

# installs the package in editable mode and registers the console scripts.
# Append [dev] to pull mongomock for the offline test suite.
pip install -e ".[dev]"
```

### MongoDB

The three CLIs read and write via MongoDB — set up Atlas (or any
pymongo-compatible Mongo) and plug the connection string into `.env`:

```bash
cp .env.example .env
# edit .env, set MONGODB_URI=mongodb+srv://... (plus MONGODB_DB if you want
# to override the default "mtg_recommender")
```

`.env` is gitignored; `.env.example` is the checked-in template.

Collections (`cards`, `tags`, `meta`) and their indexes are created on first
run — no separate migration step. See the docstring of `storage.py` for the
full document schemas.

### Verifying your cluster

After editing `.env`, confirm Mongo is reachable with the health check:

```bash
mtg-check            # one row per check: connectivity, database, indexes, counts
mtg-check -v         # also prints server + pymongo versions
```

Exit code is 0 on success and 1 if any check fails, so the command is
scriptable. A failed connectivity row short-circuits the rest to `SKIP` so
the error you care about shows up first.

## Usage

All three CLIs are installed as console scripts by `pip install -e .`.
Equivalent `python -m mtg_recommender.<module>` forms also work from a source
checkout.

### Fetch oracle text

```bash
# single card -> /cards/named (fast, one request)
scryfall-fetch "Lightning Bolt"

# multiple cards -> oracle-cards bulk download, upserted into the cards collection
scryfall-fetch "Lightning Bolt" "Counterspell"

# from a decklist file (one card per line, '#' for comments; Moxfield / Arena /
# MTGGoldfish export formats are tolerated)
scryfall-fetch --file deck.txt

# force refetch / redownload even if the collection looks fresh
scryfall-fetch --file deck.txt --refresh
```

Upserts touch only the fields `scryfall_fetch` owns, so a card's `tags` array
(written by `oracle_tags`) survives a refetch. Full document shape and
semantics are in the top-of-file docstring of `scryfall_fetch.py`.

### Attach oracle tags

Scryfall's Tagger project classifies cards by what they *do* ("spot-removal",
"mana-rock", "sweeper", …). Pull the tag catalog into the `tags` collection
and attach per-card tag lists to the `cards` collection:

```bash
# downloads oracle_tags bulk (~6 MB), writes catalog, attaches tags to each card
scryfall-fetch-tags

# force redownload even if the stored snapshot timestamp matches
scryfall-fetch-tags --refresh
```

Join is by `oracle_id`; cards the Tagger project doesn't cover get `tags: []`.
The snapshot timestamp lives in the `meta` collection so repeat runs skip the
download when nothing has moved. The command requires the `cards` collection
to be populated first — run `scryfall-fetch` on your decklists before this.

### Inspect what's in Mongo

Read-only subcommand CLI for ad-hoc exploration of the collections:

```bash
mtg-inspect card "Wrath of God"             # full doc(s); multiple matches surface ambiguity
mtg-inspect tag spot-removal                # catalog entry (hierarchy, aliases, card count)
mtg-inspect list --tag spot-removal         # list cards carrying a tag
mtg-inspect list --tag spot-removal --limit 50
mtg-inspect stats                           # collection sizes, tagged %, top tags, snapshot ts
mtg-inspect stats --top 20                  # top-N tag aggregation
```

Output is human-readable text — use `extract-oracle` for a machine-readable
JSON subset. Nothing in this CLI writes to Mongo.

### Extract a decklist subset

```bash
# writes ./oracle_subset.json by default
extract-oracle --file deck.txt

# or specify output
extract-oracle --file deck.txt -o deck_oracle.json
```

The subset drops Mongo bookkeeping (`_id`, `names`, `updated_at`, `oracle_id`,
`tags`) and keeps the downstream-facing fields (`name`, `mana_cost`,
`type_line`, `oracle_text`, `scryfall_id`).

## Testing

```bash
python -m unittest discover tests
```

Every HTTP call is mocked; every Mongo op routes through `mongomock`. 145
offline tests, well under a second total — no network, no real Mongo required.

### Integration tests (opt-in)

A small suite under `tests/integration/` exercises the real pymongo driver
against a real Mongo cluster. These tests silently skip unless
`MONGODB_INTEGRATION_URI` is set — pointing them at your Atlas cluster is
safe because each test class creates a disposable uuid-suffixed database and
drops it on teardown:

```bash
# bash/zsh:
MONGODB_INTEGRATION_URI="mongodb+srv://..." python -m unittest discover tests

# PowerShell:
$env:MONGODB_INTEGRATION_URI = "mongodb+srv://..."
python -m unittest discover tests
```

Nine integration tests cover connectivity, `ensure_indexes`, card
round-tripping, tag attachment, and the meta timestamp — see
`tests/integration/test_mongo_integration.py` for the full list.

## Project layout

```text
mtg-recommender/
├── src/
│   └── mtg_recommender/         # installable package
│       ├── __init__.py
│       ├── storage.py           # Mongo client/config/indexes
│       ├── scryfall_fetch.py    # oracle-text fetcher -> cards collection
│       ├── oracle_tags.py       # oracle-tags importer -> tags collection + attach
│       ├── extract_oracle.py    # per-decklist JSON subset exporter
│       ├── inspect.py           # read-only ad-hoc inspection CLI
│       └── check.py             # read-only Mongo health check
├── tests/
│   ├── test_storage.py          # offline, mongomock-backed
│   ├── test_scryfall_fetch.py
│   ├── test_oracle_tags.py
│   ├── test_inspect.py
│   ├── test_check.py
│   └── integration/             # opt-in; needs MONGODB_INTEGRATION_URI
│       └── test_mongo_integration.py
├── pyproject.toml               # PEP 621 metadata, build config, entry points
├── .env.example                 # template; copy to .env + fill in MONGODB_URI
├── CLAUDE.md                    # workflow + style conventions
├── PLAN.md                      # scope, roadmap, open questions
└── README.md                    # you are here
```

The `src/` layout (PyPA-recommended) prevents accidental imports from the repo
root — code must go through the installed package, which catches packaging
bugs early.

## Workflow

Branch per concern, rebase-merge only, tests in the same commit as the feature.
See [CLAUDE.md](CLAUDE.md) for the full set of conventions.
