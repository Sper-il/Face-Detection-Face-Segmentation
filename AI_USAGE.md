# AI Usage Log

This file tracks every AI-assisted change made to the project, as required by
the project's working rules (`progress_status.md` §8).

| # | Date | Tool | Task | Outcome | Notes |
|---|------|------|------|---------|-------|
| 1 | 2026-09-15 | Cursor | Initial repo scaffold | Created `src/`, `tests/`, `demos/`, `deploy/`, `docs/` | Setup follow progress_status.md template |
| 2 | 2026-09-15 | Cursor | Implement RetinaFace model | `src/detection/retinaface.py` | Based on InsightFace reference |
| 3 | 2026-09-15 | Cursor | Implement U-Net model | `src/segmentation/unet.py` | ResNet-34 encoder + standard decoder |
| 4 | 2026-09-15 | Cursor | Build pipeline orchestrator | `src/pipeline/orchestrator.py` | Graceful degradation on no-face |
| 5 | 2026-09-17 | Cursor | Audit & fix `train.py` bugs | `src/detection/train.py`, `src/segmentation/train.py` | D1: epoch_state global removed |
| 6 | 2026-09-17 | Cursor | Write full audit report | `docs/audit/full_audit_report.md` | 14 issues identified |
| 7 | 2026-09-19 | Cursor | Train U-Net on Kaggle (24K CelebAMask-HQ images) | `models/unet_final.pth` | IoU=0.9682 on 3K test set |
| 8 | 2026-09-21 | Cursor | Reverse-engineer retinaface_final.pth | `src/detection/retinaface.py` | Custom ResNet34+FPN+SSH backbone |
| 9 | 2026-09-21 | Cursor | End-to-end smoke test | `scripts/kaggle/end_to_end_smoke_test.py` | Both stages forward-pass OK |
| 10 | 2026-09-21 | Cursor | Refactor retinaface.py to support both API modes | `src/detection/retinaface.py` | Original+checkpoint classes |
| 11 | 2026-09-21 | Cursor | Apply AI Project Framework skill | All docs updated, 67 tests pass | This entry |

## Standing AI Rules

- **Naming convention**: snake_case for files, PascalCase for classes, UPPER_SNAKE_CASE for constants.
- **Folder convention**: see `progress_status.md` §8.2.
- **Discussion rule**: every architectural decision goes into `docs/adr/`.
- **Edition rule**: PR review for trained models, conventional commits.
- **Always read** the project's working rules before generating code.
