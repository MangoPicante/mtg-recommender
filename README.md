# MTG Recommender

A tool that takes a Magic: The Gathering decklist and recommends cards for it,
grounded in Scryfall oracle text. Format scope for v1 is **Commander only**;
scoring is **similarity** ("plays like"), not synergy. See [PLAN.md](PLAN.md)
for the full roadmap.

## Current state

Phases 1 and 2 are done. Scryfall oracle text and tags land in MongoDB
Atlas; every card has a 768-dim `text_embedding`, every tag has an
`embedding`, and each card carries a fused `card_vector` ready for
similarity search. Phase 3 has started: the EDHREC JSON client is in
(`edhrec_fetch` module, cached per commander in a new `edhrec` collection
with a 7-day TTL); still TODO are per-deck tag clustering, cluster
evaluation against EDHREC lift, candidate ranking, and the `recommend`
CLI. See [PLAN.md](PLAN.md) for the full Phase 3 roadmap.

### Modules

Under `src/mtg_recommender/`:

| Module | What it does |
| --- | --- |
| `storage` | Thin wrapper around pymongo. Reads `MONGODB_URI` (and optional overrides) from `.env`, caches a single `MongoClient` per process, exposes handles to the `cards` / `tags` / `meta` collections, and declares the indexes the rest of the package relies on (`cards.names` for alias lookup, `cards.oracle_id` for the tag join). |
| `scryfall_fetch` | Downloads Scryfall's `oracle_cards` bulk and merges it into the `cards` collection. Bulk-only: no `/cards/named` single-card path. Each card document is keyed by `scryfall_id` with a `names: [lowered, ...]` array for alias lookup (used by `extract_oracle` and `explore`). The `tags` field is left for `oracle_tags` to populate. |
| `oracle_tags` | Downloads the Scryfall oracle-tags bulk, writes a slug-keyed catalog to the `tags` collection (hierarchy + aliases + descriptions preserved), and attaches `tags: [slug, ...]` arrays to each card by joining on `oracle_id`. |
| `extract_oracle` | Writes a trimmed per-decklist JSON subset of the cards collection for downstream consumers that don't speak Mongo. |
| `explore` | Read-only ad-hoc inspection CLI (installed as `mtg-inspect`). Subcommands: `card`, `tag`, `list`, `stats`. Human-readable output for one-off exploration and debugging. |
| `check` | Read-only health check. Pings the Mongo server, verifies the configured database + indexes, and reports a doc count per collection. Exit 0 on success, 1 if any check fails. |
| `embeddings` | Phase 2. Encodes each card's `oracle_text` into `text_embedding` on the card doc and each tag's label+description into `embedding` on the tag doc; also fuses the two into a single unit `card_vector` per card via a weighted average (`alpha * text + (1 - alpha) * tag`, defaults to `alpha=0.6`). Uses `sentence-transformers/all-mpnet-base-v2` (768-dim) by default; `MTG_EMBEDDING_MODEL` env var overrides. |
| `edhrec_fetch` | Phase 3. Thin client over EDHREC's public keyless JSON API at `json.edhrec.com`. Fetches per-commander card signals (lift, synergy, inclusion rate, trend) and caches the payload in the `edhrec` Mongo collection with a 7-day TTL. Each signal carries a Scryfall UUID so joining back to our `cards` collection is a direct `_id` lookup. |
| `deck_profile` | Phase 3. Given a decklist, resolves cards against the `names` index and classifies each card into one of ~30 themes discovered from the oracle_tags hierarchy. Multi-theme cards are resolved by cosine similarity between the card's `text_embedding` and each theme's representative vector. Returns a `DeckProfile` carrying per-theme centroids + member tags + deck-card membership + an unassigned bucket. |
| `theme_classifier` | Phase 3. Owns the data-driven theme discovery (filtering ~900 top-level tags down to ~30 "real themes" via blocklist + min-children + card-coverage window) and the per-card classification logic (candidate intersection → text-embedding tiebreak). Also owns the optional post-classification merge pass that collapses themes whose `card_vector` profiles are cosine-similar. Called into by `deck_profile` in theme mode. |
| `card_clusterer` | Phase 3. Alternative to `theme_classifier`: HDBSCAN over `card_vector` to cluster deck cards directly, then labels each cluster by its most-frequent non-blocklisted top-level tag (fallback: `cluster-N`). Called into by `deck_profile` in cluster mode (`--mode cluster`). |
| `weight_trainer` | Phase 3. Per-deck random-search tuner for the cluster-mode structural weight knobs. Scores each candidate configuration by eta-squared of EDHREC per-card lift explained by the clustering (0 = uninformative, 1 = perfectly stratified). Called into by `deck_profile`'s `--train` CLI flag. |

Tests live under `tests/` and are offline — every HTTP call is mocked, every
Mongo op goes through `mongomock`.

## Setup

Python 3.10+ is required. Runtime dependencies: `pymongo`, `python-dotenv`,
and `sentence-transformers` (which pulls `torch`, ~1 GB on first install —
needed for the Phase 2 encoder).

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

### Common commands via `just`

A [`justfile`](justfile) wraps the common dev + CLI flows. Install
[just](https://just.systems) and run `just` with no target to list recipes.
`just install` runs the editable install above; `just test` runs the offline
suite; `just lint` runs ruff; `just check`, `just inspect`, `just extract`,
`just embed`, `just fuse`, `just profile` delegate to the console scripts
(the embed and profile ones with argument pass-through); `just fetch` and
`just tags` are flagless because the underlying fetchers are idempotent +
diff-aware. `just populate` runs the whole `fetch → tag → embed → fuse`
pipeline end-to-end — the single command that takes a cold Mongo cluster
to a recommendation-ready state (slow on first run: embedding ~40k cards
on CPU is a few minutes plus a one-time ~420 MB sentence-transformers
download; reruns on an unchanged Scryfall snapshot are near-instant).

### MongoDB

Every CLI reads and writes via MongoDB — set up Atlas (or any
pymongo-compatible Mongo) and plug the connection string into `.env`:

```bash
cp .env.example .env
# edit .env, set MONGODB_URI=mongodb+srv://... (plus MONGODB_DB if you want
# to override the default "mtg_recommender")
```

`.env` is gitignored; `.env.example` is the checked-in template.

Collections (`cards`, `tags`, `meta`, `edhrec`) and their indexes are
created on first run — no separate migration step. See the docstring of
`storage.py` for the full document schemas. `meta` holds both snapshot
timestamps (from `scryfall-fetch` / `scryfall-fetch-tags`) and
embedding-config signatures (from `mtg-embed`'s auto-invalidation);
`edhrec` holds the per-commander payload cache for Phase 3.

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

The console scripts (`scryfall-fetch`, `scryfall-fetch-tags`, `mtg-embed`,
`mtg-inspect`, `mtg-check`, `extract-oracle`, `mtg-deck-profile`) are
installed by `pip install -e .`. Equivalent `python -m mtg_recommender.<module>`
forms also work from a source checkout. Phase 3's `edhrec_fetch` ships
as a library only for now — the full recommender CLI will land in a
follow-up PR.

### Fetch oracle text

Pulls Scryfall's `oracle_cards` bulk (~24 MB gzipped, one entry per unique
card) and upserts every card into the `cards` collection. There's no
per-card HTTP mode: the real workflows downstream (embeddings, tag attach,
recommendations) all want the full cache, and carving out a decklist-sized
subset is `extract-oracle`'s job once the cache is populated.

```bash
# Idempotent + diff-aware. No flags needed.
scryfall-fetch
```

Reruns are cheap on two levels:

1. **Fast-path skip.** The current snapshot timestamp lives in the `meta`
   collection under `_id = "oracle_cards"`. If the stored timestamp matches
   the current `/bulk-data` metadata and the cache is non-empty, the module
   skips without reading any card docs — one meta call total.
2. **Diff-aware merge.** When the snapshot moves, each card carries an
   `oracle_text_sha` (16-hex sha256 prefix). The merge classifies incoming
   cards as new / changed / **patched** / unchanged against the stored
   sha and writes only new/changed/patched entries. A content change
   also `$unset`s `text_embedding` and `card_vector` so the next
   `mtg-embed cards` / `mtg-embed fuse` re-encodes only what moved;
   "patched" (sha unchanged but newly-owned schema fields missing from
   the stored doc) rewrites the owned fields WITHOUT invalidating
   embeddings — self-healing when the fetcher's owned-field list grows.
   On a typical Scryfall snapshot bump that's a tiny bulk_write instead
   of ~40k pointless `$set` operations.

   Owned fields per card: `oracle_id`, `name`, `names`, `mana_cost`,
   `type_line`, `oracle_text`, `oracle_text_sha`, `keywords`, `colors`,
   `color_identity`, `cmc`, `power`, `toughness`. The `keywords` /
   `colors` / `color_identity` / `cmc` / `power` / `toughness` block
   feeds `mtg-deck-profile --mode cluster`'s structural feature knobs
   (`--type-weight`, `--mana-cost-weight`, `--keyword-weight`,
   `--color-weight`, `--color-identity-weight`, `--power-weight`,
   `--toughness-weight`).

Upserts touch only the fields `scryfall_fetch` owns, so a card's `tags`
array (written by `oracle_tags`) survives every merge. For corruption
recovery, drop the `cards` collection and rerun. Full document shape
and semantics are in the top-of-file docstring of `scryfall_fetch.py`.

### Attach oracle tags

Scryfall's Tagger project classifies cards by what they *do* ("spot-removal",
"mana-rock", "sweeper", …). Pull the tag catalog into the `tags` collection
and attach per-card tag lists to the `cards` collection:

```bash
# Idempotent + diff-aware. No flags needed.
scryfall-fetch-tags
```

Download triggers: no stored snapshot, snapshot moved since last run, OR any
card in `cards` is missing the `tags` field (a `scryfall-fetch` run added
new cards since the last tag import). The attach pass itself is diff-aware
— it writes only to oracle_ids whose tag list actually changed, and
`$unset`s `card_vector` on those cards so `mtg-embed fuse` re-fuses only
what moved. Unchanged oracle_ids get no write.

Join is by `oracle_id`; cards the Tagger project doesn't cover get `tags: []`.
The command requires the `cards` collection to be populated first — run
`scryfall-fetch` before this.

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

### Embed cards and tags (Phase 2)

`sentence-transformers` (and its transitive `torch` dep, ~1 GB on first
install) is a runtime dep now, so no extra install step is needed. Just
encode:

```bash
# Default: embed every card whose oracle_text isn't already encoded.
mtg-embed cards

# Same for tags. The text fed to the encoder is "<label>. <description>".
mtg-embed tags

# Smoke-test with a cap before committing to the full encode:
mtg-embed cards --limit 50

# Nuclear override — re-embed everything (expensive). You rarely need
# this: a MTG_EMBEDDING_MODEL change auto-invalidates the stale vectors
# via the `meta` signature so a plain `mtg-embed cards` picks them up.
mtg-embed cards --refresh
```

Vectors land as BSON Binary (packed little-endian float32) on the card doc
(`text_embedding`) and the tag doc (`embedding`) — 4 bytes per dim instead
of the 8 bytes a BSON double array would take. Lossless relative to the
encoder's native float32 output, and halves the on-disk cost: a 40 k-card
cluster with text + card_vector lands at ~240 MB instead of ~490 MB,
comfortable on Atlas's free tier. Default model is
`sentence-transformers/all-mpnet-base-v2` (768-dim); override via the
`MTG_EMBEDDING_MODEL` env var without touching code.

Once both sides are populated, fuse them into a single per-card vector:

```bash
# Default: fuse every card that has text_embedding but no card_vector.
mtg-embed fuse

# Tweak the text/tag blend (default alpha=0.6, i.e. 60% text / 40% tags).
# Changing --alpha auto-invalidates stale card_vectors via the `meta`
# signature, so the fuse loop recomputes them without --refresh.
mtg-embed fuse --alpha 0.7
mtg-embed fuse --alpha 1.0        # ignore tags entirely
mtg-embed fuse --alpha 0.0        # ignore text entirely
```

The result lands as a unit-length `card_vector` on each card doc. Fuse logic
L2-normalizes `text_embedding` and the mean of the card's tag embeddings
before combining them, then renormalizes — so cosine similarity on
`card_vector` is well-behaved and `alpha` only steers direction, never
magnitude. Cards with no tags fall back to the normalized text vector
(effectively `alpha=1.0` for those cards). Freshness is fully automatic
across the pipeline: `scryfall-fetch` / `scryfall-fetch-tags` invalidate
downstream fields on content change, and `mtg-embed *` compares the active
model/alpha against the previous run's values in `meta` and clears stale
vectors before re-encoding. `--refresh` remains as a nuclear override.

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

### EDHREC commander signals (Phase 3)

The `edhrec_fetch` module pulls per-commander card signals from EDHREC's
public, keyless JSON API at `json.edhrec.com`. No CLI yet — this is a
library used by the Phase 3 recommender. Example from Python:

```python
from mtg_recommender import edhrec_fetch as ef

# One HTTP round trip the first time, Mongo cache (7-day TTL) thereafter.
signals = ef.get_commander_signals("Atraxa, Praetors' Voice")

# Each signal: name, scryfall_id, lift, synergy, num_decks,
# potential_decks, trend_zscore, inclusion_rate (computed property).
top_by_lift = sorted(signals, key=lambda s: s.lift, reverse=True)[:10]
for s in top_by_lift:
    print(f"{s.name:<40} lift={s.lift:>5.2f}  decks={s.num_decks:>5}")
```

The response is cached in the `edhrec` Mongo collection keyed by commander
slug. Pass `force=True` to bypass the cache. Set `EDHREC_CACHE_TTL_DAYS`
to override the default 7-day TTL (set lower during active development).

**Scope:** single-name commanders. Partner / background commanders use a
different URL shape on EDHREC and are deferred to a follow-up helper.

**Why direct JSON, not `pyedhrec`:** that wrapper hasn't shipped since Feb
2024 and doesn't expose `lift` (which EDHREC promoted to its primary metric
in late 2026). Hitting `json.edhrec.com` ourselves is ~200 lines of code,
one fewer transitive dep, and gives us the signals we actually need.

### Deck profile and theme classification (Phase 3)

Classify a decklist's cards into themes derived from the oracle_tags
hierarchy. Requires `cards` + `tags` collections populated (`just
populate` or the equivalent individual steps):

```bash
# From a decklist file (one card per line; '#' comments, Moxfield /
# MTGGoldfish / Arena export formats tolerated).
mtg-deck-profile --file deck.txt

# Or inline card names.
mtg-deck-profile "Lightning Bolt" "Wrath of God" "Sol Ring" "Brainstorm"

# Tune the theme filter. All three decide which top-level tags count
# as "real themes" worth classifying against; defaults are the ones
# that survive an audit of the live oracle_tags bulk and feel sensible
# for Commander play.
#   --min-children N     min number of child tags under the top-level
#                         (default 2 — a point concept isn't a theme)
#   --min-coverage N     min distinct cards tagged under the subtree
#                         (default 100 — below is niche)
#   --max-coverage N     max distinct cards (default 5000 — above is
#                         mechanical, applies to half the format)
mtg-deck-profile --file deck.txt --min-coverage 50 --max-coverage 8000

# Override the built-in structural/catalog blocklist. Default drops
# `triggered-ability`, `cycle`, `card-names`, etc. Pass "" to clear it
# entirely (useful for exploratory audits).
mtg-deck-profile --file deck.txt --theme-blocklist "card-names,type-errata"
mtg-deck-profile --file deck.txt --theme-blocklist ""

# After classification, themes whose cards' card_vector profiles are
# cosine-similar enough get merged into one cluster — collapses
# attributions like "removal" + "mass-removal" when the deck's
# removal cards semantically overlap. Default 0.9 (cautious). Lower
# merges more aggressively; negative disables the pass.
mtg-deck-profile --file deck.txt --merge-threshold 0.75
mtg-deck-profile --file deck.txt --merge-threshold -1       # off

# Cap per-cluster display (default: 10 tags + 10 cards; everything
# beyond collapses to a "(+N more)" marker).
mtg-deck-profile --file deck.txt --limit 20

# ------- alternative: unsupervised clustering mode -------
# Cluster cards directly by card_vector (HDBSCAN, cosine) instead of
# sorting them into pre-defined themes. Each cluster is labelled by
# the most frequent non-blocklisted top-level tag its member cards
# share (deduped per-card); falls back to "cluster-N" if every
# top-level is blocked. Useful for surfacing archetypes the curated
# top-level vocabulary doesn't name.
mtg-deck-profile --file deck.txt --mode cluster
mtg-deck-profile --file deck.txt --mode cluster --cluster-min-size 5
mtg-deck-profile --file deck.txt --mode cluster --cluster-selection-epsilon 0.1

# Add weighted structural features onto card_vector before clustering.
# Each sub-vector is L2-normalized, scaled by its weight, and
# concatenated onto the card_vector so cosine similarity becomes a
# weighted mean of per-feature cosines.
#   --type-weight             supertypes + card types + deck-local subtypes (multi-hot)
#   --mana-cost-weight        10-dim shape: generic, WUBRG, X, hybrid, phyrexian, snow
#   --keyword-weight          multi-hot over deck-local keyword vocab
#   --color-weight            5-dim WUBRG multi-hot of printed colors
#   --color-identity-weight   5-dim WUBRG multi-hot of oracle-text-inclusive identity
#   --power-weight            2-dim [value, has-power flag] for creature stats
#   --toughness-weight        2-dim [value, has-toughness flag]
# All default to 0.0 (feature off). Combine freely.
mtg-deck-profile --file deck.txt --mode cluster --type-weight 0.4
mtg-deck-profile --file deck.txt --mode cluster --mana-cost-weight 0.3
mtg-deck-profile --file deck.txt --mode cluster --color-identity-weight 0.5
mtg-deck-profile --file deck.txt --mode cluster \
    --type-weight 0.4 --power-weight 0.3 --toughness-weight 0.3

# --train auto-tunes the knobs against EDHREC lift for a commander.
# Random search over --train-trials configurations; the search picks the
# weights that make eta^2 of per-card lift explained by clustering
# as high as possible (clusters stratify high- vs. low-lift cards).
# Prints the best weights found, then renders the final clustering
# with them. Replaces any manual --*-weight flags.
mtg-deck-profile --file deck.txt --mode cluster --train \
    --commander "Atraxa, Praetors' Voice"
# Restrict the search to a subset of knobs for a cheaper / more
# interpretable run.
mtg-deck-profile --file deck.txt --mode cluster --train \
    --commander "Atraxa, Praetors' Voice" \
    --train-knobs types,mana_cost,color_identity --train-trials 100
```

Sample output:

```text
resolved 99 unique cards (0 missing)
tag universe: 142 unique tags
4 theme(s) matched, 3 card(s) unassigned, 8 tag(s) outside any theme

theme 'removal'  (18 tags from deck, 12 cards)
  tags : burn-any, exile-creature, removal-creature, spot-removal, ...
  cards: Anguished Unmaking, Beast Within, Despark, ..., (+2 more)

theme 'ramp'  (6 tags from deck, 14 cards)
  ...

unassigned cards: Weird Niche Enchantment, Random Vanilla Creature, ...
```

Also usable as a library — the CLI is a thin wrapper over
`build_deck_profile`:

```python
from mtg_recommender import deck_profile as dp

profile = dp.build_deck_profile(["Lightning Bolt", "Wrath of God", ...])
for cluster in profile.clusters:
    print(cluster.label, cluster.tags, cluster.centroid.shape)
print("unassigned:", profile.unassigned_card_ids)
```

**How the classification works:**

1. **Theme discovery** (`theme_classifier.discover_themes`) walks the
   top-level tags, drops the ones on the blocklist or outside the
   children/coverage window, and keeps the rest. For each survivor
   it computes the subtree of descendant slugs plus a representative
   vector = L2-normalized mean of the subtree's tag embeddings.
2. **Per-card classification** (`theme_classifier.classify_card`)
   intersects the card's tag array with each theme's subtree. Zero
   matches → unassigned. One match → that theme. Two or more → cosine
   similarity between the card's stored `text_embedding` and each
   candidate theme's representative vector; the max wins. (Falls
   back to the candidate with the highest card_coverage if the
   card has no `text_embedding` stored.)
3. **Theme merge** (`theme_classifier.merge_similar_themes`) builds
   each theme's L2-normalized mean `card_vector` profile and runs
   Union-Find across pairs whose cosine similarity hits the
   `--merge-threshold`. Chains (A~B, B~C) collapse into one
   component even when A~C alone wouldn't. The keeper's label is
   the component member with the highest `card_coverage`; every
   constituent name is preserved in `DeckCluster.constituent_themes`
   so attribution still shows "merged: X, Y, Z" in the render.
4. **Attribution shape:** each `DeckCluster.tags` lists only the slugs
   the deck's cards actually brought in — not the theme's full
   subtree — so the output reflects what mattered on this specific
   deck. Phase 3 step 4 (candidate ranking) will use each cluster's
   `centroid` as a query vector against `card_vector`, with EDHREC
   lift (from `edhrec_fetch`) layered on as a per-commander quality
   signal.

## Testing

```bash
python -m unittest discover tests
```

Every HTTP call is mocked; every Mongo op routes through `mongomock`. Full
suite runs in well under a second — no network, no real Mongo required. For
real-cluster coverage, run `mtg-check` against your own `MONGODB_URI`.

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
│       ├── explore.py           # read-only ad-hoc inspection CLI (mtg-inspect)
│       ├── check.py             # read-only Mongo health check
│       ├── embeddings.py        # Phase 2 encoder + mtg-embed CLI
│       ├── edhrec_fetch.py      # Phase 3 EDHREC JSON client + per-commander cache
│       ├── theme_classifier.py  # Phase 3 hierarchy-driven theme discovery + per-card classification
│       ├── card_clusterer.py    # Phase 3 unsupervised card_vector clustering + top-level labelling (--mode cluster)
│       ├── weight_trainer.py    # Phase 3 random-search tuner for cluster-mode structural weight knobs (--train)
│       └── deck_profile.py      # Phase 3 deck profile builder (theme and cluster modes) + mtg-deck-profile CLI
├── tests/
│   ├── test_storage.py          # offline, mongomock-backed
│   ├── test_scryfall_fetch.py
│   ├── test_oracle_tags.py
│   ├── test_explore.py
│   ├── test_check.py
│   ├── test_extract_oracle.py
│   ├── test_embeddings.py
│   ├── test_edhrec_fetch.py
│   ├── test_weight_trainer.py
│   ├── test_deck_profile.py     # also covers theme_classifier + card_clusterer (shared fixtures)
│   └── fixtures/                # trimmed sample payloads used by HTTP-mocked tests
│       └── edhrec_atraxa_trimmed.json
├── pyproject.toml               # PEP 621 metadata, build config, entry points
├── justfile                     # task runner for the common dev + CLI flows
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
