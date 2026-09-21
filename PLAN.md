# Orchestration: Wait for Kaggle kernel -> Download -> Clean -> Finalize

## Phases (must run sequentially)

### Phase A: Poll kernel until COMPLETE / ERROR (BLOCKING, may take 30-60 min)
- Run `poll_kaggle_long.py` in foreground (it already auto-exits on terminal status)
- Read poll_log_v4.txt to see final status
- If ERROR: download logs, analyze, STOP for user decision
- If COMPLETE: continue to Phase B

### Phase B: Download kernel outputs
- `kaggle kernels output speril/face-detection-segmentation-patched -p <download_dir>`
- Expected: `retinaface_best.pth`, `unet_best.pth`, `result.png`, `metrics.json`,
  `__results__.html`, log files

### Phase C: Move models into project structure
- Copy/move downloaded `retinaface_best.pth` -> `models/retinaface_best.pth`
- Copy/move downloaded `unet_best.pth` -> `models/unet_best.pth`
- Copy/move downloaded `result.png` -> `docs/figures/result.png` (create dir)
- Copy/move downloaded `metrics.json` -> `runs/metrics.json` (or models/)

### Phase D: Clean workspace
Delete these temporary files (sau khi đã chuyển các file cần thiết vào project):
- `kaggle_train.ipynb.bak` (backup cũ)
- `kaggle_train.ipynb.bak2`
- `patch_notebook.py`, `patch_notebook2.py`, `patch_notebook3.py`
- `fix_cell12_v3.py`
- `get_csv_paths.py`, `get_err5.py`, `get_full_log.py`, `get_log2.py`, `get_log3.py`
- `find_collate.py`, `list_ds.py`, `list_ds2.py`, `list_ds3.py`
- `poll_kaggle.py`, `poll_kaggle_long.py`
- `poll_log.txt`, `poll_out.txt`, `poll_err.txt`
- `poll_log_v4.txt`, `poll_stdout_v4.txt`, `poll_stderr_v4.txt`
- `kaggle_log_v3.txt`, `kaggle_log_v4.txt`
- `logs2.txt`, `full_log.txt`, `ds_files.txt`, `ds_list.txt`, `ds_list2.txt`, `ds_list3.txt`
- `push4.txt`, `push5.txt`
- `verify_syntax.py`
- `kernel_error_logs/` (already analyzed)
- `kaggle_output/`, `kaggle_output_utf8/` (temporary UTF8 fixes)
Keep:
- `kaggle_train.ipynb`
- `kernel-metadata.json`, `dataset-metadata.json`
- `KAGGLE_RUN.md`, `KAGGLE_TRAINING_GUIDE.md`, `DETECTION_DEBUG_GUIDE.md`
- `progress_status.md`, `README.md`, `ROADMAP.md`
- `models/`, `data/`, `src/`, `tests/`, `demos/`, `deploy/`, `docs/`, `runs/`,
  `notebooks/`, `backups_v23/`

### Phase E: Finalize project (write summary)
- Update `README.md` with training results (mIoU, AP, etc. from metrics.json)
- Update `ROADMAP.md` to mark milestone "trained model with 20 epochs" as done
- Update `progress_status.md` with final status
- Generate final `models/retinaface_best.pth` sha256 & sizes
- Generate project tree (only top-level files, depth 2)
- Print summary to console

### Phase F: Git commit (if user wants)
- After everything looks clean, ask user if they want to git commit
