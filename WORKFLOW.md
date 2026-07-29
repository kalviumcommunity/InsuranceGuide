# GitHub Workflow

This document describes how the team collaborates on this repository: branching, commit messages, pull request review, and issue tracking.

## Branching Strategy

- `main` holds only releasable, working code. Nothing is committed directly to `main`.
- All work happens on feature branches named `[type]/[short-description]`, where `type` is one of:
  - `feature` — new functionality
  - `fix` — bug fixes
  - `docs` — documentation only
  - `refactor` — code restructuring with no behavior change
  - `chore` — tooling, config, or maintenance work
- Branches are merged into `main` via Pull Request, never merged directly.
- Branches are deleted after a successful merge to keep the branch list clean.

## Commit Message Convention

Format:

```
[type]: [description]

[optional body explaining why]
```

Types used: `feat`, `fix`, `docs`, `refactor`, `chore`, `test`.

Why: a consistent format keeps `git log` readable, makes it possible to generate a changelog automatically, and makes the intent of a change clear without opening the diff.

Examples:

```
feat: add data validation function

Validates incoming CSV files for schema completeness and encoding.

docs: document branching strategy for team

chore: update requirements.txt with validation library
```

## Pull Request Review Process

- Every PR requires at least one approval before merging.
- Reviews focus on: correctness, clarity, data integrity, and test coverage.
- Commit messages are reviewed as part of code review — history should read clearly on its own.
- The PR description must explain what changed and why, and link the issue(s) it addresses.

## GitHub Issue Tracking

- Every feature or fix starts with an issue before code is written.
- Issues include a clear title, a description of why the work matters and what "done" means, at least one label, and an assignee.
- Issues are closed automatically when the linked PR is merged (via `Closes #<issue-number>` in the PR description).
