# Scripts

Helper scripts for the Face Detection & Segmentation project, organized by purpose.

## Folder structure

| Subfolder | Purpose |
|---|---|
| `preprocessing/` | Data download, validation, and preprocessing pipelines (WIDERFace, CelebAMask-HQ) |
| `inference/` | Model inference and demo scripts (RetinaFace, U-Net) |
| `evaluation/` | Evaluation scripts and report generation (metrics, figures, DOCX) |
| `checkpoints/` | Checkpoint inspection, hashing, and weight download utilities |
| `kaggle/` | Kaggle kernel status polling, output download, and result sync |
| `diagrams/` | Pipeline architecture diagram generation |
| `misc/` | Project maintenance scripts (progress status updates) |

---

## Kaggle workflow (most common usage)

The `kaggle/` folder holds the scripts you actually run day-to-day:

| Script | Purpose |
|---|---|
| `check_kaggle_status.py` | Query the status of a Kaggle kernel + list its output files (Kaggle Python SDK). Pull outputs locally with `--download`. |
| `poll_kaggle_kernel.py` | Poll a Kaggle kernel every `--interval` seconds until it finishes (or fails). Writes a timestamped log + `last_status.json`. |
| `sync_results.py` | Copy `models/retinaface_best.pth` → `retinaface_demo.pth` (and `unet_best.pth` → `unet_demo.pth`); extract `models.zip` if present; recompute `model_hashes.json`; merge `models/training_history.json` + `runs/eval/*_test_metrics.json` into `models/training_metrics.json`; copy weights + extras into the repo. |
| `update_progress_status.py` | Re-render section 14 of `progress_status.md` and the latest-results block in `README.md` from the merged metrics. |

### Typical flow (after Kaggle kernel finishes)

1. **Check status.**

   ```bash
   python scripts/kaggle/check_kaggle_status.py
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
   python scripts/kaggle/check_kaggle_status.py --download
   # or for a specific older kernel:
   python scripts/kaggle/check_kaggle_status.py --kernel speril/face-detection-segmentation-patched --download
   ```

   Files land in `./kaggle_outputs_v23/<kernel-slug>/`.

3. **Sync weights + metrics into the repo.**

   ```bash
   python scripts/kaggle/sync_results.py
   ```

   - Extracts `models.zip` (strips `models/` prefix so files land flat in `kaggle_outputs_v23/models/`).
   - Copies `retinaface_best.pth` → `retinaface_demo.pth` (and the same for U-Net).
   - Hashes both `.pth` files into `model_hashes.json`.
   - Copies weights + `result.png` + `runs/eval/*_test_metrics.json` into `./models/` and `./runs/`.
   - Merges training history + test metrics into `models/training_metrics.json`.

4. **Refresh documentation.**

   ```bash
   python scripts/misc/update_progress_status.py
   ```

   Re-renders section 14 of `progress_status.md` and the "Latest Training Results" block in `README.md`. Bails out (no-op) when no new test metrics are available, so it's safe to run while the kernel is still training.

### Background polling

If you don't want to refresh the Kaggle web UI every few minutes, run:

```bash
python scripts/kaggle/poll_kaggle_kernel.py --interval 600 --max-wait 43200
```

This logs status changes to `kaggle_outputs_v23/poll.log` and writes `last_status.json` when the kernel reaches a terminal state. Exits automatically after `max-wait` seconds.

For a single check: `python scripts/kaggle/poll_kaggle_kernel.py --once`.

---

## Preprocessing

```bash
# Run the full preprocessing pipeline
python scripts/preprocessing/run_preprocessing.py

# Validate preprocessed data
python scripts/preprocessing/validate_preprocessing.py

# Audit data integrity
python scripts/preprocessing/data_audit.py

# Process WIDERFace specifically
python scripts/preprocessing/process_widerface.py

# Quick smoke test for preprocessing
python scripts/preprocessing/test_preprocessing_quick.py
```

## Inference

```bash
# Run inference with RetinaFace (Yakhyo variant)
python scripts/inference/inference_retinaface_yakhyo.py

# Run the interactive demo
python scripts/inference/demo.py

# Visualize segmentation results
python scripts/inference/visualize_segmentation.py
```

## Evaluation

```bash
# Evaluate segmentation model
python scripts/evaluation/eval_segmentation.py

# Generate figures for the report
python scripts/evaluation/generate_report_figures.py

# Insert figures into DOCX report
python scripts/evaluation/insert_figures_to_docx.py

# Quick test with U-Net on a single image
python scripts/evaluation/test_unet_image.py

# Quick test with Yakhyo model
python scripts/evaluation/test_yakhyo_model.py
```

## Checkpoints

```bash
# Inspect a checkpoint
python scripts/checkpoints/inspect_checkpoint.py
python scripts/checkpoints/inspect_yakhyo_checkpoint.py

# Quick checkpoint sanity check
python scripts/checkpoints/check_checkpoint.py

# Test if RetinaFace weights are loadable
python scripts/checkpoints/test_retinaface_weights.py

# Download RetinaFace pretrained weights
python scripts/checkpoints/download_retinaface_weights.py
```

## Diagrams

```bash
# Generate pipeline architecture diagram
python scripts/diagrams/create_pipeline_diagram.py
```

---

## Notes

- All Kaggle scripts use the Kaggle Python SDK (`from kaggle.api.kaggle_api_extended import KaggleApi`), which authenticates via `~/.kaggle/kaggle.json` or the `KAGGLE_USERNAME`/`KAGGLE_KEY` env vars.
- The sync + update scripts are safe to re-run: they skip cleanly when expected inputs are missing and never overwrite existing repo files with stale data (hash check before copy).
- Run from the repo root (or pass `--models-dir` / `--runs-dir` / `--kernel` to override).
- On Windows the Kaggle SDK's log file uses the system default encoding, which fails on emoji characters in the runner output. `check_kaggle_status.py` monkey-patches `open()` to force UTF-8 when pulling outputs.
- After reorganization, always run scripts from the new subfolder path (e.g. `python scripts/kaggle/check_kaggle_status.py`).
