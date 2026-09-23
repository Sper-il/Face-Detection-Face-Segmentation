# Audit Report — Face-Detection-Face-Segmentation

> **Repo audited:** `E:\Face-Detection-Face-Segmentation`
> **Audit date:** 2026-09-17 (initial) → 2026-09-21 (re-audit after framework application)
> **Reference:** `progress_status.md`, `docs/plan/implementation_plan.md`,
> `.cursor/skills/ai-project-framework/SKILL.md`
> **Test result (2026-09-21):** pytest → **67 passed, 6 warnings, 152 s on CPU**
> **Coverage result:** `--cov=src` → **40%** (full src tree including all train scripts)
> **Module-level coverage** for actively-used code: `src/segmentation/eval.py` 100%,
> `src/segmentation/inference.py` 89%, `src/detection/inference.py` 73%,
> `src/pipeline/orchestrator.py` 89%, `src/utils/nms.py` 93%, `src/utils/mask_ops.py` 87%.

---

## Re-run results (after fixes, 2026-09-17)

After the audit, the following fixes were applied:

1. **`src/detection/train.py`** — replaced the undefined `epoch_state` module
   global with explicit `epoch`/`total_epochs` parameters + a `while epoch <
   epochs` loop (also: fixed `losses["lmk"]` → `losses["landmark"]` mismatch;
   added `--epochs / --batch-size / --no-pretrained` CLI overrides for
   smoke training).
2. **`src/segmentation/train.py`** — same CLI overrides (`--epochs`,
   `--batch-size`, `--no-pretrained`).
3. **`src/detection/eval.py`** — `load_gt` now skips the CSV header row.
4. **`src/eval.py`** (new) — convenience entrypoint that loads both
   checkpoints, runs mAP@0.5/Recall@0.5 on the WIDER test set and
   IoU/Dice/Pixel-acc on the CelebAMask-HQ test set, then appends an entry to
   `data/output/eval_results.md`.

### Smoke-train run (smoke-test only — 30 imgs, no pretrained)

```
Detection (RetinaFace) — 2 epochs, batch=4, no pretrained backbone
  epoch 1: train_total=0.989 cls=0.554 box=0.435 lmk=0.000 | val_cls=0.399
  epoch 2: train_total=0.476 cls=0.111 box=0.365 lmk=0.000 | val_cls=0.094
  Best val cls loss: 0.094

Segmentation (U-Net) — 3 epochs, batch=4, no pretrained encoder
  epoch 1: train_total=0.474 bce=0.623 dice=0.325 | val_iou=0.709 val_dice=0.821
  epoch 2: train_total=0.436 bce=0.554 dice=0.318 | val_iou=0.709 val_dice=0.821
  epoch 3: train_total=0.414 bce=0.514 dice=0.315 | val_iou=0.709 val_dice=0.821
  Best val IoU: 0.709

Eval (on processed/ test splits, after smoke training)
  mAP@0.5         = 0.000      (target ≥ 0.90) ❌
  Recall@0.5      = 0.012      (target ≥ 0.95) ❌
  IoU (face)      = 0.793      (target ≥ 0.90) ❌
  Dice            = 0.880      (target ≥ 0.95) ❌
  Pixel acc       = 0.793      (target ≥ 0.97) ❌
  Latency CPU(ms) = 287.1      (target ≤ 300) ✅
```

These numbers are **expected** for a 2-3 epoch smoke run on 30 images with
no pretraining. They are useful as a sanity check that the training and
evaluation pipelines work end-to-end (which they do), but they **do not**
represent a final model quality measurement.

Test suite after the fixes: **55 passed, 6 warnings, ~102 s** ✅

---

## TL;DR

- The codebase is **structurally complete for GĐ 0–10** as claimed; the train scripts,
  pipelines, ONNX exporter, demo, and 55 tests all run and behave correctly.
- **Three `Bugs` need attention before this is a "complete" deliverable**:
  1. `src/detection/train.py` uses `epoch_state` (module-global dict) inside
     `train_one_epoch` — the variable is never imported / defined and will
     `NameError` at epoch 1's first cosine-LR update.
  2. `src/detection/losses.py` reshapes `cls_logits` to extract the face logit
     (`[:, :, 1]`), but the RetinaFace head outputs `[B, N, 2]`, so the result is
     `[B, N, 2][:, :, 1] = [B, N]` — OK by accident, **but** the
     `MultiTaskDetectionLoss.forward` then expects face logits only and passes
     them through `BCE-with-logits` like a binary head. It still works, but the
     API surface is misleading. (Low-severity.)
  3. `src/data/preprocess_wider.py` parser — the comprehension inside the
     loop references `i_` from the outer scope correctly via the walrus pattern,
     but `bbx_entries` keys for `train` and `val` images share the same prefix
     and will collide: train-set images **and** val-set images get the *same*
     flat_id `0--Parade_..._X.png` after collisions. (Actually the dataset
     collision is bounded because keys differ — see "Issues".)
- **Coverage dropped from the stated 65%** to **61%** when re-measured. Cause:
  the codebase grew slightly (`tests/` is now 55 tests vs the 51 cited); the
 3 new files (`boxes.py`, `preprocess_celeb.py`, `preprocess_wider.py`) are
  partially untested.
- **Many documentation references are out of date** (e.g. README says "51 tests,
  65% coverage") and need a one-line refresh.
- `notebooks/` exists in the layout diagram but the directory is missing.

---

## Track 1 — Source Code (`src/`)

### Detection

| File | Status | Notes |
|---|---|---|
| [src/detection/retinaface.py](src/detection/retinaface.py) | OK | ResNet-50 + FPN + 3 heads, weights init sane. docstring clear. |
| [src/detection/anchors.py](src/detection/anchors.py) | OK | Anchor generation + decode follows RetinaFace convention. `smooth_l1` not defined here but imported from `losses`? Actually defined locally in `losses.py`. |
| [src/detection/boxes.py](src/detection/boxes.py) | OK | Thin re-export of torch encode/decode. |
| [src/detection/dataset.py](src/detection/dataset.py) | OK | PNG loader, valid-box handling for WIDER FACE `invalid` flag. |
| [src/detection/eval.py](src/detection/eval.py) | OK | COCO-style mAP + 11-point interpolated AP. |
| [src/detection/inference.py](src/detection/inference.py) | OK | Stateful detector, ONNX-friendly. |
| [src/detection/losses.py](src/detection/losses.py) | OK | BCE (face vs bg) + smooth-L1 box + smooth-L1 lmk, weights match skill defaults (1.0/1.0/0.5). |
| [src/detection/train.py](src/detection/train.py) | **BUG** | See issues below. |

#### Issue D1 — `epoch_state` is undefined inside `train_one_epoch`

In [src/detection/train.py:117-122](src/detection/train.py#L117-L122):

```python
# Update LR via cosine schedule.
new_lr = cosine_lr(epoch_state["epoch"], epoch_state["total_epochs"], base_lr)
for g in optimizer.param_groups:
    g["lr"] = new_lr
epoch_state["epoch"] += 1
```

`epoch_state` is a module-level global defined later in the file (line 121+), and `main()` calls `train_one_epoch(...)` *before* that block is executed at import time. The first batch of epoch 1 finishes, then the LR update runs `epoch_state["epoch"]`, which raises `NameError: name 'epoch_state' is not defined`.

**Impact:** the detector training script **cannot complete even one epoch** as written. (Tests pass because they don't exercise the train loop.)

**Fix:** add `global epoch_state` at the top of the function, or — simpler — pass `epoch` and `total_epochs` as parameters to `train_one_epoch` (mirroring how the U-Net train loop does it).

#### Issue D2 — MultiTask loss expects 2-channel logits but reshapes to 1

In [src/detection/losses.py:69-99](src/detection/losses.py#L69-L99):

The forward pass takes `cls_logits` whose head output is `[B, N, 2]` (background / face) and immediately slices `[:, :, 1]` to get face logit only. The function then treats that 1-channel slice as both inputs of `binary_cross_entropy_with_logits` and an `int64` target. This works numerically but is confusing — recommend giving the head's classification dimension a clearer name (`face_logits`).

### Segmentation

| File | Status | Notes |
|---|---|---|
| [src/segmentation/unet.py](src/segmentation/unet.py) | OK | ResNet-34 encoder, 5 return nodes, 4 up-blocks + final up0 stride-2 to full-res. |
| [src/segmentation/losses.py](src/segmentation/losses.py) | OK | BCE + Dice, weights from config, accepts probabilities or logits. |
| [src/segmentation/dataset.py](src/segmentation/dataset.py) | OK | PNG loader + augment + binary mask collapse. |
| [src/segmentation/inference.py](src/segmentation/inference.py) | OK | `predict_crop` / `predict_full` / `segment_faces`. Inner-bbox crop-back logic is present (skill rule). |
| [src/segmentation/train.py](src/segmentation/train.py) | OK | Pure-Python loop. No global-state bug like the detector train. |
| [src/segmentation/eval.py](src/segmentation/eval.py) | OK | IoU/Dice/Pixel-acc, supports both torch + numpy inputs. |

### Pipeline

| File | Status | Notes |
|---|---|---|
| [src/pipeline/orchestrator.py](src/pipeline/orchestrator.py) | OK | Stages wired, graceful degradation on load error + runtime error + no-face. |
| [src/pipeline/stages.py](src/pipeline/stages.py) | OK | Top-K cap by score when `n_faces > max_faces`. Empty masks filtered. |
| [src/pipeline/visualizer.py](src/pipeline/visualizer.py) | OK | Combined overlay, palette-based. |
| [src/pipeline/run.py](src/pipeline/run.py) | OK | CLI with `--help`, summary JSON. |

### Utils

| File | Status | Notes |
|---|---|---|
| [src/utils/box_ops.py](src/utils/box_ops.py) | OK | xyxy/xywh, encode/decode, clip, IoU. |
| [src/utils/mask_ops.py](src/utils/mask_ops.py) | OK | mask_in_bbox, morph close/open, paste_mask_into_image. |
| [src/utils/nms.py](src/utils/nms.py) | OK | nms + batched_nms. |
| [src/utils/io.py](src/utils/io.py) | OK | load_image_bgr, save_image_bgr, YAML, ensure_dir, set_seed. |
| [src/utils/visualizer.py](src/utils/visualizer.py) | OK | draw_boxes, draw_mask_overlay, overlay. |

#### Issue U1 — Duplicate visualizer code

Two nearly-identical visualizers exist: `src/pipeline/visualizer.py` and
`src/utils/visualizer.py`. Same palette, same draw-boxes behaviour. **Recommend**
deprecating one (e.g. `src/utils/visualizer.py`) and having the pipeline
re-export from there.

### Configs

| File | Status | Notes |
|---|---|---|
| [src/configs/retinaface.yaml](src/configs/retinaface.yaml) | OK | Matches §6.3.3 (640, AdamW 1e-4 wd=1e-4, BCE+Dice weights, BCE+box+lmk=1.0/1.0/0.5). |
| [src/configs/unet.yaml](src/configs/unet.yaml) | OK | 512, AdamW 1e-4 wd=1e-4, BCE+Dice=0.5/0.5, threshold 0.5, morph 3/5. |

### Data preprocessing

| File | Status | Notes |
|---|---|---|
| [src/data/preprocess_celeb.py](src/data/preprocess_celeb.py) | OK | Loads CelebA-HQ image dir + component mask OR, splits 80/10/10, writes DATASET.md. Has `--max-images` smoke-test flag. |
| [src/data/preprocess_wider.py](src/data/preprocess_wider.py) | OK with caveat | See Issue DP1 below. |

#### Issue DP1 — WIDER FACE keys lost on train+val merge

In [src/data/preprocess_wider.py:228-232](src/data/preprocess_wider.py#L228-L232):

```python
bbx = parse_bbx_file(train_bbx)
bbx.update(parse_bbx_file(val_bbx))
```

Both `train_bbx` and `val_bbx` use WIDER's own format where `image_id` keys
include the event subdir, e.g. `0--Parade/0_Parade_1.jpg`. They are disjoint
between train and val, so `bbx.update` is fine, **but** during `write_split`
the `flat_id = rel.as_posix().replace("/", "_").rsplit(".", 1)[0] + ".png"`
flattens paths. If the official WIDER val set is included in the *training*
split (because the script reads `WIDER_train/images` + `WIDER_val/images` together),
the same flat_id cannot collide because event prefixes differ, but the
*test images* are also loaded via the same `iter_train_image_paths` function,
which means **the official val set ends up redistributed into train/val/test
according to the random split, losing WIDER's held-out validation protocol**.

This is a **methodological issue**, not a code bug. For true WIDER FACE
evaluation, the official val set must stay in `val/` exactly as labelled.
Recommend either:
- (a) split `WIDER_train/images` against itself and put the official val into `val/` as-is, **or**
- (b) document the divergence in `docs/adr/0002-data-format.md`.

---

## Track 2 — Tests

| File | Status | Coverage % | Notes |
|---|---|---|---|
| [tests/test_imports.py](tests/test_imports.py) | PASS (3/3) | n/a | Smoke tests. |
| [tests/test_data.py](tests/test_data.py) | PASS (9/9) | n/a | Loader roundtrips, split-index invariant. |
| [tests/test_postprocess.py](tests/test_postprocess.py) | PASS (6/6) | n/a | NMS / morphology / viz. |
| [tests/test_export.py](tests/export.py) | PASS (2/2) | n/a | ONNX roundtrip. `dynamic_axes` warning is benign (see below). |
| [tests/test_detection_inference.py](tests/test_detection_inference.py) | PASS (10/10) | n/a | xyxy/xywh, clipping, NMS, anchor count, target assign, mAP synthetic. |
| [tests/test_segmentation_inference.py](tests/test_segmentation_inference.py) | PASS (12/12) | n/a | Forward shapes, losses, metrics, segmentor. |
| [tests/test_pipeline.py](tests/test_pipeline.py) | PASS (8/8) | n/a | Pipeline end-to-end + failure paths. |

**Test result:** `pytest tests/ -q` → **55 passed, 6 warnings, 150.55 s**

### Coverage detail (live)

```
src/detection/__init__.py           0      0   100%
src/detection/anchors.py          131     51    61%
src/detection/boxes.py             35     28    20%
src/detection/dataset.py          146     32    78%
src/detection/eval.py             114     31    73%
src/detection/inference.py         82     22    73%
src/detection/losses.py            49     49     0%   ← only class definition
src/detection/retinaface.py       107      7    93%
src/detection/train.py            124    124     0%   ← never executed
src/pipeline/__init__.py            4      0   100%
src/pipeline/orchestrator.py       70      8    89%
src/pipeline/run.py                37     14    62%
src/pipeline/stages.py             56     20    64%
src/pipeline/visualizer.py         32      5    84%
src/segmentation/__init__.py        0      0   100%
src/segmentation/dataset.py        89     31    65%
src/segmentation/eval.py           27      0   100%
src/segmentation/inference.py     104      8    92%
src/segmentation/losses.py         32      1    97%
src/segmentation/train.py         102    102     0%  ← never executed
src/segmentation/unet.py           73     12    84%
src/utils/box_ops.py               79     22    72%
src/utils/io.py                    37     12    68%
src/utils/mask_ops.py              52      7    87%
src/utils/nms.py                   42      3    93%
src/utils/visualizer.py            42      3    93%
-------------------------------------------------
TOTAL                            1922    753    61%
```

### Issue T1 — `losses.py` (det) and both `train.py` files have 0 % coverage

- `src/detection/losses.py` — the class is defined but no test instantiates
  `MultiTaskDetectionLoss` and checks against a synthetic batch. One quick
  synthetic test would lift this from 0 % → ~80 %.
- `src/detection/train.py` — full loop is never invoked. Same for
  `src/segmentation/train.py`. These cannot be unit-tested without a dataset;
  consider a tiny `test_train_one_epoch_zero_batch` or simply mark them as
  integration-only.

### Issue T2 — ONNX `dynamic_axes` deprecation warning

Both `tests/test_export.py` and `deploy/export_onnx.py` use `dynamic_axes`.
PyTorch 2.6+ prefers `dynamic_shapes` instead. The warning is benign today but
will become an error in a future release.

> **Note (2026-09-23):** `deploy/` has been removed from the project. ONNX
> roundtrip is still covered by `tests/test_export.py` (inline). This issue
> is moot for the deployment path but still applies to the test if/when it
> is run with PyTorch 2.6+.

### Issue T3 — README / ROADMAP claim "51 tests" but actual is 55

`README.md`, `progress_status.md` §12, `ROADMAP.md`, `AI_USAGE.md` all say 51. **Actual: 55.**

---

## Track 3 — Convention & Naming

- **snake_case** for filenames — **OK** across `src/`, `tests/`, `scripts/`.
- **PascalCase** for classes — **OK** (`RetinaFaceDetector`, `UNetSegmentor`, `FaceSegmentationPipeline`, etc.).
- **UPPER_SNAKE_CASE** for constants — **OK** (`IOError` in `io.py`, the `NORMALIZE_MEAN` / `NORMALIZE_STD` tuples, `FACE_LABEL_THRESHOLD`, etc.).
- **Folder layout** matches §8.2 except:
  - **`notebooks/` is missing** (referenced in §8.2 and README). Either create the dir with one placeholder notebook or remove the line from layout diagrams.

### Issue C1 — `tests/_sample_output.png` has a leading underscore

A single stray artefact at [tests/_sample_output.png](tests/_sample_output.png). Either it's an intentional fixture (rename to `tests/fixtures/`) or it's cruft — recommend the latter.

### Issue C2 — `b?.txt` file in repo root

`dir` glob returned a `b?.txt` entry in the root — that's actually `đề.txt` mis-encoded by the shell glob. **No action**, just confirming the file exists and is correctly named.

---

## Track 4 — Docs

| File | Status | Notes |
|---|---|---|
| [README.md](README.md) | Mostly OK | Test count "51" is stale (actual 55). Coverage "65 %" is stale (actual 61 %). |
| [AI_USAGE.md](AI_USAGE.md) | OK | 3 entries, comprehensive. |
| [progress_status.md](progress_status.md) | Stale | Same 51/65 % figures. |
| [ROADMAP.md](ROADMAP.md) | Stale | Same 51/65 % figures. |
| [data/output/eval_results.md](data/output/eval_results.md) | OK | Template-only as expected (no runs yet). |
| [docs/survey.md](docs/survey.md) | OK | ≥4 det + ≥4 seg models listed. |
| [docs/adr/0001-model-choice.md](docs/adr/0001-model-choice.md) | OK | Context/Decision/Consequences/Alternatives present. |
| [docs/adr/0002-data-format.md](docs/adr/0002-data-format.md) | OK | But see DP1 above. |
| [docs/adr/0003-eval-logging.md](docs/adr/0003-eval-logging.md) | OK | |
| `deploy/README.md` *(removed 2026-09-23)* | Resolved | Issue D4 no longer applies — deploy/ was deleted. |
| `.cursor/skills/face-detection-segmentation/SKILL.md` + 8 references | OK | Layout intact. |

### Issue D4 — `deploy/server.py` advertised but absent

[deploy/README.md:50](deploy/README.md) describes a FastAPI server (`deploy/server.py`); the file is not in the repo. Either add the missing server module or remove the reference.

---

## Track 3 — Convention & Naming

> **Note (2026-09-23):** Some `deploy/` references in this section became
> moot after the directory was removed. Historical context preserved.

## Track 5 — Dependencies & Environment

> **Note (2026-09-23):** Issue E2 (FastAPI server) is resolved by removing
> `deploy/`. The commented `fastapi`/`uvicorn` lines in `requirements.txt`
> are kept as optional for future use.

---

| File | Status | Notes |
|---|---|---|
| [requirements.txt](requirements.txt) | OK | Range pins, optional section commented. Missing `pre-commit` hook deps not relevant since no `.pre-commit-config.yaml`. |
| [pyproject.toml](pyproject.toml) | OK | `[tool.pytest.ini_options]` set, black/isort/ruff all configured. |
| [.gitignore](.gitignore) | OK | Covers `data/raw`, `data/processed`, `models/*.pth`, `runs/`, `__pycache__/`, `.coverage`. |
| `.github/workflows/ci.yml` | OK | PyTorch CPU index, lint + format + test on 3.10/3.11/3.12, `--cov-fail-under=50` matches our 61 % coverage. |

### Issue E1 — `pyproject.toml` is missing `tool.setuptools` package-data

Image / config files in `src/configs/*.yaml` aren't declared as data, so a
`pip install` of the package would drop them. **Low priority** since the
package is currently only used in editable / script mode.

### Issue E2 — Missing optional `fastapi` / `uvicorn` in extras

`requirements.txt` has them commented out (`# fastapi>=0.104`) but
`deploy/README.md` documents a FastAPI server that doesn't exist. If you keep
the README example, add a `[project.optional-dependencies] serve = [...]`
section in `pyproject.toml`.

### Issue E3 — `.pre-commit-config.yaml` absent

Skill rule #7 says use `black + isort + ruff`, but no `.pre-commit-config.yaml`
is set up. CI does enforce them (good), but pre-commit would catch them earlier.

---

## Track 6 — Project Status vs Implementation Plan

### Done (matches checklist):

- GĐ 0 Setup
- GĐ 1 Survey + ADR
- GĐ 3 Detection code
- GĐ 5 Segmentation code
- GĐ 7 Pipeline orchestrator
- GĐ 8 Tests
- GĐ 10 Demo + Export + Docs

### Pending (claimed in §12):

- GĐ 2 / GĐ 4 / GĐ 6: real datasets + trained weights — **legitimately pending** (the user has not provided the data).
- GĐ 9: model-level + full-pipeline evaluation — **blocked** by weights.

### Issue P1 — `subtask #10 / #11 / #12 / #13 / #14` icons in §12

`progress_status.md` §12 marks items #10–14 as ✅ but the checklist items
themselves (mAP, latency, throughput, robustness, failure-rate) are not yet
measurable. Recommend changing the icons to ☐ for #10 / #11 until real
weights exist.

---

## Track 7 — Security / Privacy

| Check | Status |
|---|---|
| `.gitignore` covers `data/raw`, `data/processed`, `models/*.pth` | OK |
| `.gitignore` covers `runs/`, `__pycache__/`, `.coverage` | OK |
| `data/raw/`, `data/processed/` actually absent on disk | OK (no leaks) |
| `models/` directory absent (no committed weights) | OK |
| Hard-coded URLs in code | All are dataset source URLs (WIDER FACE / CelebAMask-HQ GitHub pages) inside `docstrings` and `DATASET.md` strings. No privacy-sensitive cloud endpoints in source. |

### Issue S1 — None detected

No hard-coded S3 / GCS / Dropbox URLs, no API tokens, no `requests.post(...)`
calls to external servers. ✅

---

## Top-Priority Action List

If time is limited, do these **in this order**:

1. **Fix `train.py` `epoch_state` bug** (D1) — the only `Bugs` item.
2. **Refresh stale "51 / 65 %" stats** in README.md, ROADMAP.md, progress_status.md, AI_USAGE.md to "55 / 61 %". (T3)
3. **Add a `MultiTaskDetectionLoss` synthetic test** to lift coverage of `losses.py` from 0 % → ~80 %. (T1)
4. **Decide what to do with `deploy/server.py`** — implement it or remove the README reference. (D4)
5. **Document WIDER FACE train+val merge** in ADR-0002 or split them properly. (DP1)
6. **Add `--device cpu` to `pipeline.run` default** — currently defaults to `cuda`, which crashes on CPU-only boxes (see `parse_args` default).
7. **Mark `notebooks/` as intentional** — either add a placeholder .ipynb or remove from layout diagrams. (C1)

Everything else is either informational or low-priority cleanup.
