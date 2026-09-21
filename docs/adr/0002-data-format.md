# ADR-0002 — Data format & split policy

- **Status:** Accepted
- **Date:** 2026-09-16
- **Deciders:** AI Engineer (project owner) + Cursor Agent

## Context

The project must ingest two datasets (WIDER FACE for detection,
CelebAMask-HQ for segmentation) and produce reproducible 80/10/10 splits.
Without a clear format & split rule, contributors can silently leak
information between splits or degrade image quality by re-encoding to JPEG.

## Decision

1. All images and masks are stored as **PNG** (lossless).
2. Annotations are stored as **CSV** with columns
   `image_id, x_min, y_min, x_max, y_max, confidence, mask_path`
   (the `confidence` and `mask_path` columns are empty for detection rows
   and only `mask_path` is empty for segmentation rows; this lets a single
   schema describe both datasets).
3. Splits are **80 / 10 / 10** with `SEED = 42` and applied *per dataset*.
4. WIDER FACE's `ignore` flag is honoured at **evaluation** time only
   (predictions inside `ignore` boxes do not count as false positives,
   but they also do not count as true positives).
5. Each split directory contains a `DATASET.md` card with:
   - source URL, version, license;
   - exact file count per split;
   - sha256 of the index file.

## Consequences

- Reproducible: re-running any preprocessing script with the same seed
  produces the same split.
- Lossless: PNG keeps the original pixel values; downstream models never
  see JPEG artefacts.
- Auditable: each split can be verified with a no-overlap test
  (`tests/test_data.py::test_no_overlap`).

## References

- `progress_status.md` §4 (Data) and §8 (Working Rules).
- `.cursor/skills/face-detection-segmentation/references/data.md`.
