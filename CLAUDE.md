# MTG Recommender

A tool for recommending Magic: The Gathering cards for a given decklist, grounded in Scryfall oracle text.

See `PLAN.md` for scope and roadmap.

## Branching & PR Workflow

Feature branches, not direct commits to `main`. Rebase-merge only.

```bash
git checkout main && git pull origin main
git checkout -b feat/<short-slug>        # prefix: feat/ fix/ chore/ docs/ refactor/ test/

# work in small commits
git add <files> && git commit -m "..."

git push -u origin feat/<short-slug>
gh pr create --fill
# do not auto-merge — wait for the user to approve
gh pr merge --rebase --delete-branch     # only after user approves

git checkout main && git pull origin main && git fetch --prune
```

## Comment Style

Comments should be **verbose and frequent** — this project leans toward over-explaining rather than under-explaining. The default "self-documenting code needs no comments" rule does **not** apply here.

- Every module gets a top-of-file docstring describing what it does, the shape of its inputs/outputs, and any non-obvious invariants.
- Every function gets a docstring. Public helpers get the full "why does this exist / what are the edge cases / what does the caller need to know" treatment — see `scryfall_fetch.py` for the house style.
- Inline comments are encouraged whenever logic isn't immediately obvious from the code alone — especially around regexes, data-shape assumptions, timestamp handling, and anywhere a reader might ask "why this way?".
- Prefer explaining the **why** (the hidden constraint, the Scryfall quirk, the past bug) over restating the **what**. But when the "what" is dense (nested comprehensions, non-trivial parsing), spell it out too.

If a block of logic would make a new reader pause to figure out what's happening, add a comment.

## Commit Hygiene

One concern per commit — don't batch everything at the end. Prefer "parse decklist formats" → "add tests for decklist parser" → "wire parser into fetcher" over one sweeping commit.

## Keeping README in Sync

Update `README.md` in the same commit whenever you add or change anything user-visible (new script, new CLI flag, new output file, setup step, env var). Skip for internal refactors, test-only changes, or comment edits.

## Testing Policy

Tests are written in the same commit as the feature.

**Always write tests for:**
- New function with branching logic → unit test covering key branches
- Bug fix → regression test
- Anything touching the cache schema or alias resolution

**Minimum per module:** happy path · one edge case · one failure mode (missing input, HTTP error, malformed data)

**Skip:** pure config, one-shot throwaway scripts, trivial pass-throughs

**Mechanics:** `test_<module>.py` next to the module · `unittest` + `unittest.mock` · network calls must be mocked · run `python -m unittest discover` before committing
