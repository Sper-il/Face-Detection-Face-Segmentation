# CI / CD

This directory holds GitHub Actions workflows that run automatically on every
push and pull request.

## `ci.yml`

Two jobs:

### `test`

Runs on every push / PR. Validates that the codebase still works across
Python 3.10 / 3.11 / 3.12.

1. `ruff check src tests` — lints the source tree.
2. `pytest tests/` — runs the unit + integration test suite (~60 tests).
   These are synthetic / small and finish in ~2 minutes on a 2-core runner.

### `eval-smoke`

Runs only after `test` succeeds. Verifies that the standalone evaluation
scripts (`src.scripts.eval_retinaface_wider`, `src.scripts.eval_segmentation`)
work end-to-end against a tiny synthetic dataset + random checkpoint.

This catches regressions like "the eval script can't find its inputs" or
"the model checkpoint loader is broken" *without* needing a real trained
model in the repository.

## Adding a new workflow

Add a new YAML file under `.github/workflows/`. Keep the same matrix of
Python versions as `test.yml` and re-use the existing `pip cache` step.

## Required secrets

None — CI runs entirely on public runners.
