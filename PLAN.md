# MTG Recommender — Plan

A tool that takes a Magic: The Gathering decklist and recommends cards for it, grounded in Scryfall oracle text.

## Scope

- **Format: Commander only** for v1. Multi-format comes later and will need legality / colour-identity fields in the schema.
- **Scoring: similarity** ("plays like") only. Synergy ("plays well with") is deferred — EDHREC already publishes a "lift" score we can fold in later if we want that mode.

## Current state

- `scryfall_fetch.py` — fetches Scryfall oracle text for a card or decklist, caching to `cache/oracle_texts.json` (id-keyed, with a name-alias index). Chooses single-card vs. bulk-download mode automatically.
- `extract_oracle.py` — writes a trimmed per-decklist subset of the cache for downstream consumers.
- `test_scryfall_fetch.py` — offline unit tests; every HTTP call is mocked.

## Phases

### Phase 1 — foundations

- Oracle-text cache with scryfall-id identity + name-alias index. _(done)_
- Decklist parser tolerant of Moxfield / Arena / MTGGoldfish exports. _(done)_
- **Oracletag acquisition.** Pulled from the Scryfall `/bulk-data` endpoint (the tagger project's tags are published there alongside the oracle bulk files). Need to verify which bulk type carries them and in what shape before writing the import.
- **MongoDB persistence.** Upload the fetched data (oracle text, oracletags, and later the embeddings) to **MongoDB Atlas** and make it the source of truth. Schema: **one wide document per card**, `_id = scryfall_id`, with `oracle_text`, `tags`, `text_embedding`, `tag_embedding`, etc. as fields on that document. The on-disk JSON cache (`cache/oracle_texts.json`) goes away once Mongo is wired up — no dual-write, no offline fallback.

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

_None blocking right now._
