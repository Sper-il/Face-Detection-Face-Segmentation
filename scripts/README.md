# Scripts

Helper scripts to support the Kaggle training workflow for the v23 pipeline.

| Script | Purpose |
|---|---|
| `check_kaggle_status.py` | Query the status of a Kaggle kernel + list its output files (Kaggle Python SDK). Pull outputs locally with `--download`. |
| `poll_kaggle_kernel.py` | Poll a Kaggle kernel every `--interval` seconds until it finishes (or fails). Writes a timestamped log + `last_status.json`. |
| `sync_results.py` | Copy `models/retinaface_best.pth` → `retinaface_demo.pth` (and `unet_best.pth` → `unet_demo.pth`); extract `models.zip` if present; recompute `model_hashes.json`; merge `models/training_history.json` + `runs/eval/*_test_metrics.json` into `models/training_metrics.json`; copy weights + extras into the repo. |
| `update_progress_status.py` | Re-render section 14 of `progress_status.md` and the latest-results block in `README.md` from the merged metrics. |

## Typical flow (after Kaggle kernel finishes)

1. **Check status.**

   ```bash
   python scripts/check_kaggle_status.py
   ```

   Output example:
   ```
   kernel : speril/face-detection-face-segmentation
   status : KernelWorkerStatus.COMPLETE
   files  : 4 in output
     - models.zip
     - result.png
     - retinaface_demo.pth
     - unet_demo.pth
   ```

2. **Pull outputs locally.**

   ```bash
   python scripts/check_kaggle_status.py --download
   # or for a specific older kernel:
   python scripts/check_kaggle_status.py --kernel speril/face-detection-segmentation-patched --download
   ```

   Files land in `./kaggle_outputs_v23/<kernel-slug>/`.

3. **Sync weights + metrics into the repo.**

   ```bash
   python scripts/sync_results.py
   ```

   - Extracts `models.zip` (strips `models/` prefix so files land flat in `kaggle_outputs_v23/models/`).
   - Copies `retinaface_best.pth` → `retinaface_demo.pth` (and the same for U-Net).
   - Hashes both `.pth` files into `model_hashes.json`.
   - Copies weights + `result.png` + `runs/eval/*_test_metrics.json` into `./models/` and `./runs/`.
   - Merges training history + test metrics into `models/training_metrics.json`.

4. **Refresh documentation.**

   ```bash
   python scripts/update_progress_status.py
   ```

   Re-renders section 14 of `progress_status.md` and the "Latest Training Results" block in `README.md`. Bails out (no-op) when no new test metrics are available, so it's safe to run while the kernel is still training.

## Background polling

If you don't want to refresh the Kaggle web UI every few minutes, run:

```bash
python scripts/poll_kaggle_kernel.py --interval 600 --max-wait 43200
```

This logs status changes to `kaggle_outputs_v23/poll.log` and writes `last_status.json` when the kernel reaches a terminal state. Exits automatically after `max-wait` seconds.

For a single check: `python scripts/poll_kaggle_kernel.py --once`.

## Notes

- All scripts use the Kaggle Python SDK (`from kaggle.api.kaggle_api_extended import KaggleApi`), which authenticates via `~/.kaggle/kaggle.json` or the `KAGGLE_USERNAME`/`KAGGLE_KEY` env vars.
- The sync + update scripts are safe to re-run: they skip cleanly when expected inputs are missing and never overwrite existing repo files with stale data (hash check before copy).
- Run from the repo root (or pass `--models-dir` / `--runs-dir` / `--kernel` to override).
- On Windows the Kaggle SDK's log file uses the system default encoding, which fails on emoji characters in the runner output. `check_kaggle_status.py` monkey-patches `open()` to force UTF-8 when pulling outputs.
