# ADR-0003 — Evaluation logging policy

- **Status:** Accepted
- **Date:** 2026-09-16
- **Deciders:** AI Engineer (project owner) + Cursor Agent

## Context

A single `progress_status.md` will quickly become unreadable if every
training run's metric table is pasted into it. We also want a tamper-evident
history of results — i.e. a contributor should not be able to silently edit
a previous run's number to hide a regression.

## Decision

1. **All** metric tables go into `data/output/eval_results.md`.
2. The file is **append-only** — historical entries are never edited.
3. Each entry starts with a header of the form
   `## [YYYY-MM-DD HH:MM] <model-name> on <dataset-slice>`.
4. Each entry contains the metric table in the format prescribed in
   `.cursor/skills/face-detection-segmentation/references/evaluation.md`,
   plus a free-form "Notes" section.
5. `progress_status.md` mentions the *latest* result but never raw numbers.

## Consequences

- The progress doc stays a narrative; the eval log stays a metric ledger.
- It is easy to spot regressions (new entries added at the bottom).
- Any attempt to rewrite history is visible (the file is under version
  control and changes are reviewable).

## References

- `progress_status.md` §5.1, §7.
- `.cursor/skills/face-detection-segmentation/references/evaluation.md`.
