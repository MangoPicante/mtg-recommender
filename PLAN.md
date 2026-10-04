# MTG Recommender — Plan

A tool that takes a Magic: The Gathering decklist and recommends cards for it, grounded in Scryfall oracle text.

## Scope

- **Format: Commander only** for v1. Multi-format comes later and will need legality / colour-identity fields in the schema.
- **Scoring: similarity** ("plays like") only. Synergy ("plays well with") is deferred — EDHREC already publishes a "lift" score we can fold in later if we want that mode.

## Current state

Phase 1 is done. All reads and writes go through MongoDB; the on-disk JSON cache from early prototyping is gone.

Modules under `src/mtg_recommender/`:

- `storage.py` — Mongo client/config/indexes; owns `cards` + `tags` + `meta` collection handles and the two multikey indexes the fetchers depend on.
- `scryfall_fetch.py` — Scryfall oracle-text fetcher: downloads the `oracle_cards` bulk and merges every card into the `cards` collection. Carries `oracle_id` on every doc so tag import can join.
- `oracle_tags.py` — oracle-tags bulk importer. Writes the slug-keyed catalog to `tags` and attaches `tags: [slug, ...]` arrays to every card.
- `extract_oracle.py` — per-decklist JSON subset exporter for downstream consumers.
- `explore.py` — read-only inspection CLI, invoked as `mtg-inspect` (`card` / `tag` / `list` / `stats` subcommands).
- `check.py` — Mongo health check (connectivity + indexes + counts).

Tests: offline unittest + mongomock suite only — no integration suite. `mtg-check` serves as the real-cluster smoke. `justfile` wraps the common dev + CLI flows.

## Phases

### Phase 1 — foundations _(done)_

- Oracle-text store with scryfall-id identity and a multikey `names` index for alias lookup.
- Decklist parser tolerant of Moxfield / Arena / MTGGoldfish exports.
- Oracletag acquisition from Scryfall's `/bulk-data` endpoint (`oracle_tags` type), slug-keyed catalog, hierarchy preserved.
- MongoDB persistence — `cards` + `tags` + `meta` collections, indexes created idempotently, snapshot timestamps tracked in `meta`.

### Phase 2 — card representation (in progress)

- **Oracle-text embeddings.** _(done)_ Each card's `oracle_text` is encoded via `sentence-transformers/all-mpnet-base-v2` (768-dim) and persisted as `text_embedding` on the card doc. Encoder is overridable through the `MTG_EMBEDDING_MODEL` env var. The CLI (`mtg-embed cards`) skips docs that already carry a vector; `--refresh` re-embeds.
- **Oracletag embeddings.** _(done)_ Each tag's `label + description` is encoded with the same model and persisted as `embedding` on the tag doc. `mtg-embed tags` runs the same skip / `--refresh` pattern. Clustering of these embeddings — "treat a tag as this + its neighbours" — is the next sub-step.
- **Combined card vector.** _(done)_ `mtg-embed fuse` reads `text_embedding` + the card's tag embeddings, L2-normalizes each side, blends them via `alpha * text + (1 - alpha) * tag` (default `alpha=0.6`, overridable via `--alpha`), renormalizes, and writes the result as `card_vector` on the card doc. Cards with no (resolvable) tags collapse to the normalized text vector — tags-only would fail silently otherwise. A dim mismatch between the two spaces raises instead of producing a vector in neither space.
- **Freshness.** _(mostly done)_ `scryfall-fetch` stores an `oracle_text_sha` per card; a bulk merge diffs incoming cards against the stored sha, writes only new/changed entries, and `$unset`s `text_embedding` + `card_vector` on changed cards so `mtg-embed cards` / `mtg-embed fuse` re-encode exactly those. `scryfall-fetch-tags` does the same per oracle_id, `$unset`ing `card_vector` on cards whose tag list changed. `--refresh` is gone from both fetchers. Still TODO: `mtg-embed * --refresh` is kept for `MTG_EMBEDDING_MODEL` / `--alpha` changes, which the system can't cheaply self-detect. A model-signature + alpha-signature follow-up would subsume those too.

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
