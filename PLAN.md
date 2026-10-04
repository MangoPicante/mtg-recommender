# MTG Recommender — Plan

A tool that takes a Magic: The Gathering decklist and recommends cards for it, grounded in Scryfall oracle text.

## Scope

- **Format: Commander only** for v1. Multi-format comes later and will need legality / colour-identity fields in the schema.
- **Scoring: similarity** ("plays like") only. Synergy ("plays well with") is deferred — EDHREC already publishes a "lift" score we can fold in later if we want that mode.

## Current state

Phase 1 is done. All reads and writes go through MongoDB; the on-disk JSON cache from early prototyping is gone.

Modules under `src/mtg_recommender/`:

- `storage.py` — Mongo client/config/indexes; owns `cards` + `tags` + `meta` collection handles and the two multikey indexes the fetchers depend on.
- `scryfall_fetch.py` — Scryfall oracle-text fetcher (single + bulk modes) → `cards` collection. Carries `oracle_id` on every doc so tag import can join.
- `oracle_tags.py` — oracle-tags bulk importer. Writes the slug-keyed catalog to `tags` and attaches `tags: [slug, ...]` arrays to every card.
- `extract_oracle.py` — per-decklist JSON subset exporter for downstream consumers.
- `inspect.py` — read-only inspection CLI (`card` / `tag` / `list` / `stats` subcommands).
- `check.py` — Mongo health check (connectivity + indexes + counts).

Tests: offline unittest + mongomock suite plus an opt-in integration suite (`tests/integration/`) that runs against a real cluster when `MONGODB_INTEGRATION_URI` is set. `justfile` wraps the common dev + CLI flows.

## Phases

### Phase 1 — foundations _(done)_

- Oracle-text store with scryfall-id identity and a multikey `names` index for alias lookup.
- Decklist parser tolerant of Moxfield / Arena / MTGGoldfish exports.
- Oracletag acquisition from Scryfall's `/bulk-data` endpoint (`oracle_tags` type), slug-keyed catalog, hierarchy preserved.
- MongoDB persistence — `cards` + `tags` + `meta` collections, indexes created idempotently, snapshot timestamps tracked in `meta`.

### Phase 2 — card representation (next)

- **Oracle-text embeddings.** Embed each card's oracle text into a vector. Pick an encoder (sentence-transformers off-the-shelf vs. an MTG-token-aware fine-tune).
- **Oracletag embeddings + clustering.** Embed each unique oracletag and cluster them so semantically-similar tags (e.g. "removal" / "destroy creature" / "exile creature") land near each other. Lets us treat a tag as "this + its neighbours" rather than as an opaque string.
- **Combined card vector.** Fuse the oracle-text embedding with the aggregated tag embedding into one vector per card via **weighted average** (`alpha * text_vec + (1 - alpha) * tag_vec`). Requires the two embedding spaces to share dimensionality; `alpha` starts as a tunable constant and can be revisited once we have evaluation signal.
- Persist embeddings on the card's Mongo document; regenerate when `updated_at` or the card's tag set changes.

### Phase 3 — recommendation

- From a Commander decklist, derive a deck profile (e.g. mean + spread of embeddings, maybe weighted by role).
- Rank candidate cards by cosine similarity to the profile, filtered to the deck's **colour identity** and excluding cards already in the deck.
- CLI: `recommend.py --file deck.txt --top 20`.

### Phase 4 — quality / UX

- Explainability: surface the strongest text-overlap signals behind a recommendation.
- Beyond-Commander format support (Modern, Pioneer, …) — needs legality fields added to the Mongo schema.
- Optional synergy mode layered on top of similarity, using EDHREC's lift score as the signal.
- Optional thin web UI.

## Open questions

### Phase 2 kickoff

- **Encoder choice.** Off-the-shelf sentence-transformers (`all-MiniLM-L6-v2`, 384-dim; or `all-mpnet-base-v2`, 768-dim) vs. an MTG-fine-tuned model. Start off-the-shelf and only fine-tune if evaluation signal demands it.
- **Where embeddings live.** A 384-dim float32 vector is ~1.5 KB per card doc on 38k cards (~60 MB). A 768-dim vector doubles that. Options: inline on the card doc (simplest, keeps the join free) vs. a sibling `embeddings` collection (keeps card docs lean for non-recommender reads). Inline is the default; revisit if doc size becomes a problem.
- **`alpha` for the weighted fuse.** The text-vs-tag blend starts as a module-level constant; make it tunable before the recommender runs so we can sweep it against an evaluation set.
