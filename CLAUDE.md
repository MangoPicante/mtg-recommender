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

Comments should be **frequent but concise**. The default "self-documenting code needs no comments" rule does **not** apply — assume a reader cold-starting on the file; most non-trivial lines deserve some explanation. But explain in a sentence, not a paragraph.

- Every module gets a top-of-file docstring: what it does, input/output shape, any non-obvious invariant. One or two short paragraphs — not a tour of every function.
- Every function gets a docstring. Lead with one line stating what it does. Add a second paragraph only when the **why**, a hidden constraint, or an edge case genuinely needs it.
- Inline comments go in wherever logic isn't obvious from the code — regexes, data-shape assumptions, timestamp quirks, workarounds. One or two lines each.
- Prefer the **why** (the hidden constraint, the Scryfall quirk, the past bug) over the **what**. Restate the **what** only when the code is genuinely dense (nested comprehensions, bit-twiddling, non-trivial parsing).

If a block would make a new reader pause, add a comment. If a comment can be cut in half without losing meaning, cut it.

## Commit Hygiene

One concern per commit — don't batch everything at the end. Prefer "parse decklist formats" → "add tests for decklist parser" → "wire parser into fetcher" over one sweeping commit.

## Keeping README, justfile, and PLAN in Sync

Three top-level docs need to move with the code — update them in the same commit that introduces the change, not in a trailing cleanup pass:

- `README.md` — add or change anything user-visible: new script, new CLI flag, new output file, setup step, env var. Skip for internal refactors, test-only changes, or comment edits.
- `justfile` — add or change a common command a developer would reach for: new console script, new CLI entry point, new routine check like lint/test/smoke. Keep recipe comments one line, matching the surrounding style. Skip for one-off scripts, flags that just tweak existing recipes, or internal-only tooling.
- `PLAN.md` — the slice you just shipped should land done, a sub-step added, an open question answered or dropped, a trade-off shifted. Skip for pure implementation details that don't move the roadmap or change scope.

## Testing Policy

Tests are written in the same commit as the feature.

**Always write tests for:**

- New function with branching logic → unit test covering key branches
- Bug fix → regression test
- Anything touching the cache schema or alias resolution

**Minimum per module:** happy path · one edge case · one failure mode (missing input, HTTP error, malformed data)

**Skip:** pure config, one-shot throwaway scripts, trivial pass-throughs

**Mechanics:** `test_<module>.py` next to the module · `unittest` + `unittest.mock` · network calls must be mocked · run `python -m unittest discover` before committing
