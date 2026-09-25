# AI Usage Log

This file tracks every AI-assisted change made to the project, as required by
the project's working rules (`progress_status.md` §8).

| # | Date | Tool | Task | Outcome | Notes |
|---|------|------|------|---------|-------|
| 1 | 2026-09-15 | Cursor | Initial repo scaffold | Created `src/`, `tests/`, `docs/` | Setup follow progress_status.md template |
| 2 | 2026-09-15 | Cursor | Implement RetinaFace model | `src/detection/retinaface.py` | Based on InsightFace reference |
| 3 | 2026-09-15 | Cursor | Implement U-Net model | `src/segmentation/unet.py` | ResNet-34 encoder + standard decoder |
| 4 | 2026-09-15 | Cursor | Build pipeline orchestrator | `src/pipeline/orchestrator.py` | Graceful degradation on no-face |
| 5 | 2026-09-17 | Cursor | Audit & fix `train.py` bugs | `src/detection/train.py`, `src/segmentation/train.py` | D1: epoch_state global removed |
| 6 | 2026-09-17 | Cursor | Write full audit report | `docs/audit/full_audit_report.md` | 14 issues identified |
| 7 | 2026-09-19 | Cursor | Train U-Net on Kaggle (24K CelebAMask-HQ images) | `models/unet_final.pth` | IoU=0.9682 on 3K test set |
| 8 | 2026-09-21 | Cursor | Reverse-engineer retinaface_final.pth | `src/detection/retinaface.py` | Custom ResNet34+FPN+SSH backbone |
| 9 | 2026-09-21 | Cursor | End-to-end smoke test | `scripts/kaggle/end_to_end_smoke_test.py` | Both stages forward-pass OK |
| 10 | 2026-09-21 | Cursor | Refactor retinaface.py to support both API modes | `src/detection/retinaface.py` | Original+checkpoint classes |
| 11 | 2026-09-21 | Cursor | Apply AI Project Framework skill | All docs updated, 67 tests pass | Framework applied |
| 12 | 2026-09-23 | Cursor | Fix scripts/kaggle/end_to_end_smoke_test.py | REPO path + sample_dir to data/processed/segmentation | Both fixes merged |
| 13 | 2026-09-23 | Cursor | Re-run segmentation evaluation (100 test + 100 val) | `runs/evaluation/segmentation_test_metrics.json` | IoU=0.9660, Dice=0.9824 |
| 14 | 2026-09-23 | Cursor | Generate detection visualizations from real models | `runs/visualizations/detection/` | 8 samples from real RetinaFace |
| 15 | 2026-09-23 | Cursor | Remove stale outputs/visualizations (from dummy data) | Cleaned | Old dummy-data visualizations deleted |
| 16 | 2026-09-23 | Cursor | Reorganize all .md files into docs/ subfolders | All docs/guides/planning/status/references/ | 12 files moved into thematic folders |
| 17 | 2026-09-23 | Cursor | Update README.md with current project structure | Real model perf + actual structure | Real metrics |
| 18 | 2026-09-23 | Cursor | Update all .md files to reflect current status | progress_status, ROADMAP, PLAN, TEAM_WORK_PLAN, evaluation report, AI_USAGE | All md files now accurate |
| 19 | 2026-09-25 | Cursor | Fix all evaluation scripts (defaults + paths) | `src/eval.py`, `notebooks/eval_100_samples.ipynb`, eval scripts | All evals use real trained models & real data |
| 20 | 2026-09-25 | Cursor | Re-run full evaluation pipeline | `runs/evaluation/`, `runs/visualizations/` | IoU=**0.9679** test, **0.9666** val |

---

## Standing AI Rules

- **Naming convention**: snake_case for files, PascalCase for classes, UPPER_SNAKE_CASE for constants.
- **Folder convention**: see `progress_status.md` §8.2 and the project structure in `README.md`.
- **Discussion rule**: every architectural decision goes into `docs/adr/`.
- **Edition rule**: PR review for trained models, conventional commits.
- **Always read** the project's working rules before generating code.
- **Always cross-check** file paths and verify file existence before referencing in docs.

---

## Action Statistics (2026-09-25)

- **Evaluation re-run** with real models and real data:
  - **Segmentation test:** IoU=**0.9679**, Dice=**0.9834**, PixelAcc=**0.9770**
  - **Segmentation val:** IoU=**0.9666**, Dice=**0.9826**, PixelAcc=**0.9756**
  - **RetinaFace smoke:** forward 0.56s, detector 0.44s, top_score 1.42
  - **Visualizations:** 8 new samples in `runs/visualizations/test/`
- **3 commits** this session: docs sync, eval fix, eval re-run
- **Files updated:** 6 markdown files + 3 JSON metric files + 8 PNG visualizations
