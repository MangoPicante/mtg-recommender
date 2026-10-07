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

### Phase 2 — card representation _(done)_

- **Oracle-text embeddings.** Each card's `oracle_text` is encoded via `sentence-transformers/all-mpnet-base-v2` (768-dim) and persisted as `text_embedding` on the card doc, stored as BSON Binary of packed little-endian float32 (4 bytes/dim). Encoder is overridable through the `MTG_EMBEDDING_MODEL` env var.
- **Oracletag embeddings.** Each tag's `label + description` is encoded with the same model and persisted as `embedding` on the tag doc, same packed-float32 shape. (A possible follow-up: smooth each tag's representation with its k-NN neighbours — "treat a tag as this + its neighbours" — but no evaluation harness exists to measure the impact, so deferred until Phase 3's recommender gives us one.)
- **Combined card vector.** `mtg-embed fuse` reads `text_embedding` + the card's tag embeddings, L2-normalizes each side, blends them via `alpha * text + (1 - alpha) * tag` (default `alpha=0.6`, overridable via `--alpha`), renormalizes, and writes the result as `card_vector` on the card doc. Cards with no (resolvable) tags collapse to the normalized text vector — tags-only would fail silently otherwise. A dim mismatch between the two spaces raises instead of producing a vector in neither space.
- **Freshness.** Fully automatic across every input:
    - `scryfall-fetch` stores an `oracle_text_sha` per card; a bulk merge diffs incoming cards against the stored sha, writes only new/changed entries, and `$unset`s `text_embedding` + `card_vector` on changed cards.
    - `scryfall-fetch-tags` does the same per `oracle_id`, `$unset`ing `card_vector` on cards whose tag list changed.
    - `mtg-embed cards` / `tags` / `fuse` compare the active `MTG_EMBEDDING_MODEL` / `--alpha` against the previous run's values stored in `meta`, and `$unset` the stale fields on drift. The subsequent "skip if field exists" loop then re-encodes exactly what was invalidated.
  `--refresh` remains on `mtg-embed *` as a nuclear override but no normal workflow needs it.

### Phase 3 — recommendation

The recommender takes a Commander decklist and produces a ranked list of candidate cards. The ranking combines the deck's own tag-cluster themes with EDHREC's per-commander lift / rank signal, so suggestions are "cards that fit the deck's themes AND empirically show up in strong decks for this commander."

1. **EDHREC client (`edhrec_fetch.py`).** _(done)_ Thin wrapper around EDHREC's public keyless JSON API at `json.edhrec.com/pages/commanders/<slug>.json`. Responses cached in a new `edhrec` Mongo collection with a 7-day TTL (EDHREC updates slowly, and we want to be a good citizen). Per-card signals extracted: `lift` (likelihood ratio vs. baseline — the main theme signal), `synergy` (correlation), `num_decks` / `potential_decks` (popularity), `trend_zscore` (recency). The response includes each card's Scryfall `id` so we join to our `cards` collection by `_id` with no name-matching.
2. **Per-deck theme classification (`deck_profile.py` + `theme_classifier.py` + `card_clusterer.py`).** _(done)_

   Two modes: **theme** (default) and **cluster**.

   **Theme mode (`build_deck_profile` + `theme_classifier`):** `build_deck_profile(card_names)` resolves cards against the `names` index, then classifies each into one of ~30 themes discovered from the oracle_tags hierarchy. Theme discovery (`theme_classifier.discover_themes`) walks the top-level tags and keeps those that: aren't on a hand-shaped blocklist of structural/catalog slugs (`triggered-ability`, `cycle`, `card-names`, etc.), have ≥ `min_children=2`, and cover between `min_coverage=100` and `max_coverage=5000` distinct cards. Each surviving top-level becomes a `Theme` carrying its full subtree of tag slugs and a unit representative vector (mean of the subtree's tag embeddings). Per-card classification intersects the card's tag array with each theme's subtree; zero candidates → unassigned, one → that theme, two+ → cosine similarity between the card's stored `text_embedding` and each candidate's representative (max wins — the "based on oracle text" tiebreak). A post-classification merge pass (`merge_similar_themes`, default `merge_threshold=0.9`) computes each theme's L2-normalized mean `card_vector` profile, pairwise cosines the profiles, and Union-Finds themes whose profiles are at or above the threshold into one cluster — chains collapse, keeper is the component member with the highest `card_coverage`.

   **Cluster mode (`build_cluster_profile` + `card_clusterer`):** `--mode cluster`. Clusters deck cards directly on their `card_vector` via `sklearn.cluster.HDBSCAN` (cosine metric, same inclusivity defaults as the theme-mode HDBSCAN we used earlier). Each resulting cluster is labelled by the most-frequent non-blocklisted top-level tag reached from member cards' tags (deduped per-card so one tag-heavy card doesn't dominate); fallback label `cluster-N` when every candidate is blocked. Useful for surfacing archetypes the curated top-level vocabulary doesn't name. No merge pass in this mode — clustering is the primary knob.

   **Structural feature weights (cluster mode).** The clusterer accepts an optional `feature_weights` dict (`{feature_name: weight}`) that concatenates L2-normalized * weighted structural sub-vectors onto each card's `card_vector` before HDBSCAN runs. Under cosine the overall similarity becomes a weighted mean of per-feature cosines with weights ∝ w_i². Shipped feature builders:
   - **types** (CLI `--type-weight`): multi-hot of a card's supertypes + card types + deck-local subtypes.
   - **mana_cost** (CLI `--mana-cost-weight`): 10-dim symbol-count vector (generic, WUBRG, X, hybrid, phyrexian, snow).
   - **keywords** (CLI `--keyword-weight`): multi-hot over deck-local keyword vocab (Flying, Lifelink, Haste, …).
   - **colors** (CLI `--color-weight`): 5-dim WUBRG multi-hot of the card's printed colours.
   - **color_identity** (CLI `--color-identity-weight`): 5-dim WUBRG multi-hot of the card's identity (printed colours + oracle-text mana symbols) — usually what Commander splits care about.
   - **power** (CLI `--power-weight`): 2-dim `[value, has-power flag]`. The flag distinguishes non-creatures (`[0, 0]`) from 0-power creatures (`[0, 1]`).
   - **toughness** (CLI `--toughness-weight`): same shape as power.

   All weights default to 0.0 (feature off); combinations compose linearly under cosine. Centroids returned from the clusterer stay in base `card_vector` space (not augmented) so Phase 3 step 4's candidate ranking matches the stored per-card vectors.

   Both modes return a `DeckProfile` with per-cluster unit centroids, deck tag slugs that landed in each cluster, `constituent_themes` naming the theme(s) that collapsed in (or the label alone in cluster mode), deck card ids per cluster, tags outside any theme (`noise_tags`, always `()` in cluster mode), and cards that matched zero themes or lack a `card_vector` (`unassigned_card_ids`).
3. **Cluster evaluation via EDHREC.** For each cluster, cross-reference the deck's cards within it against EDHREC's lift/rank for the commander. A cluster whose member cards consistently show high lift is a validated theme for this commander; one with low lift suggests the clustering grouped tags that don't actually co-occur in play. Clusters below a threshold are either dropped or down-weighted when ranking.
4. **Deck profile → candidate ranking.** For each validated cluster, use its centroid in `card_vector` space to rank candidates (cards within the deck's colour identity, not already in the deck). Combine the per-cluster cosine scores with EDHREC lift as a tie-breaker / multiplier. Final output: top-N candidates with per-recommendation attribution ("suggested by cluster 2 (removal theme) + EDHREC lift 2.1").
5. **CLI.** `recommend.py --file deck.txt --commander "..." --top 20`. Human-readable output showing cluster labels and attribution.

Partner / background commanders are deferred to a follow-up — the EDHREC client handles single-name commanders first; partner slugs follow a different URL pattern that needs its own helper.

### Phase 4 — quality / UX

- Explainability: surface the strongest text-overlap signals behind a recommendation.
- Beyond-Commander format support (Modern, Pioneer, …) — needs legality fields added to the Mongo schema.
- Optional synergy mode layered on top of similarity, using EDHREC's lift score as the signal.
- Optional thin web UI.

## Open questions

### Phase 2 kickoff

- **Encoder choice.** Off-the-shelf sentence-transformers (`all-MiniLM-L6-v2`, 384-dim; or `all-mpnet-base-v2`, 768-dim) vs. an MTG-fine-tuned model. Start off-the-shelf and only fine-tune if evaluation signal demands it.
- **Where embeddings live.** _(answered)_ Inline on the card doc, stored as BSON Binary of packed little-endian float32 — 4 bytes per dim instead of the 8 a BSON double array takes. A 768-dim text + card_vector pair lands at ~240 MB on 40k cards (fits Atlas's free tier); the sibling-collection variant was ruled out because downstream reads always want the card + its vector together, so the join would be pure cost.
- **`alpha` for the weighted fuse.** The text-vs-tag blend starts as a module-level constant; make it tunable before the recommender runs so we can sweep it against an evaluation set.
