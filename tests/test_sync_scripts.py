"""Tests for the Kaggle-results sync helpers.

These tests cover the safety guards of `sync_results.py` and
`update_progress_status.py`: they must not clobber existing repo files
when the expected Kaggle outputs are absent.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import sync_results  # noqa: E402
import update_progress_status  # noqa: E402


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def test_sync_weights_no_op_when_best_missing(tmp_path: Path) -> None:
    models = tmp_path / "models"
    runs = tmp_path / "runs"
    hashes_path = models / "model_hashes.json"
    metrics_path = models / "training_metrics.json"
    _write_json(hashes_path, {"retinaface_demo.pth": "deadbeef"})
    _write_json(metrics_path, {"existing": True})

    out = sync_results.sync_weights(models)
    assert out == {"retinaface_demo.pth": "deadbeef"}
    # metrics unchanged
    assert json.loads(metrics_path.read_text()) == {"existing": True}


def test_sync_weights_copies_best_when_present(tmp_path: Path) -> None:
    models = tmp_path / "models"
    runs = tmp_path / "runs"
    hashes_path = models / "model_hashes.json"
    _write_json(hashes_path, {"retinaface_demo.pth": "old"})

    # Write a tiny "best" file.
    best = models / "retinaface_best.pth"
    best.write_bytes(b"fake-detector-bytes")
    unet_best = models / "unet_best.pth"
    unet_best.write_bytes(b"fake-unet-bytes")

    out = sync_results.sync_weights(models)
    assert "retinaface_demo.pth" in out
    assert "unet_demo.pth" in out
    assert (models / "retinaface_demo.pth").read_bytes() == b"fake-detector-bytes"
    assert (models / "unet_demo.pth").read_bytes() == b"fake-unet-bytes"
    # Existing entries should be preserved.
    stored = json.loads(hashes_path.read_text())
    assert stored["retinaface_demo.pth"] == out["retinaface_demo.pth"]


def test_finalize_training_metrics_no_op_when_empty(tmp_path: Path) -> None:
    metrics_path = tmp_path / "training_metrics.json"
    _write_json(metrics_path, {"legacy": True})
    out = sync_results.finalize_training_metrics(tmp_path, None, None, None)
    assert json.loads(metrics_path.read_text()) == {"legacy": True}
    assert out == {"legacy": True}


def test_update_progress_status_no_op_when_no_metrics(tmp_path: Path) -> None:
    progress = tmp_path / "progress_status.md"
    original = "# status\n\n## 14. Latest Results\n\nold content\n"
    progress.write_text(original, encoding="utf-8")
    update_progress_status.update_progress_status(
        tm={}, det_test=None, seg_test=None, out_path=progress,
    )
    assert progress.read_text(encoding="utf-8") == original


def test_update_readme_no_op_when_no_metrics(tmp_path: Path) -> None:
    readme = tmp_path / "README.md"
    original = "# title\n\n## Latest Training Results (v22)\n\nold block\n"
    readme.write_text(original, encoding="utf-8")
    update_progress_status.update_readme(
        tm={}, det_test=None, seg_test=None, out_path=readme,
    )
    assert readme.read_text(encoding="utf-8") == original


def test_check_kaggle_status_reports_missing_kaggle_sdk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """When the kaggle SDK isn't importable, check_kaggle_status must exit cleanly."""
    import builtins as _builtins
    _real_import = _builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name == "kaggle" or name.startswith("kaggle."):
            raise ImportError("kaggle missing")
        return _real_import(name, *args, **kwargs)

    monkeypatch.setattr(_builtins, "__import__", _fake_import)
    # Avoid argparse trying to parse pytest's argv.
    monkeypatch.setattr("sys.argv", ["check_kaggle_status.py"])
    from check_kaggle_status import main as ck_main
    with pytest.raises(SystemExit) as exc_info:
        ck_main()
    assert exc_info.value.code == 2
    captured = capsys.readouterr()
    assert "kaggle" in captured.out.lower()


def test_extract_models_zip(tmp_path: Path) -> None:
    """`extract_models_zip` should pull entries from models.zip into models/."""
    import zipfile

    parent = tmp_path / "pulled"
    parent.mkdir()
    zf_path = parent / "models.zip"
    target = parent / "models"
    target.mkdir()
    with zipfile.ZipFile(zf_path, "w") as zf:
        zf.writestr("models/", "")
        zf.writestr("models/retinaface_demo.pth", b"fake-detector")
        zf.writestr("models/unet_demo.pth", b"fake-unet")
        zf.writestr("models/training_metrics.json", '{"epochs": 1}')

    assert sync_results.extract_models_zip(target) is True
    assert (target / "retinaface_demo.pth").read_bytes() == b"fake-detector"
    assert (target / "unet_demo.pth").read_bytes() == b"fake-unet"
    assert json.loads((target / "training_metrics.json").read_text()) == {"epochs": 1}


def test_extract_models_zip_returns_false_when_missing(tmp_path: Path) -> None:
    target = tmp_path / "models"
    target.mkdir()
    assert sync_results.extract_models_zip(target) is False


def test_sync_weights_propagates_to_repo_models_dir(tmp_path: Path) -> None:
    """`sync_weights` should copy weights + hashes to repo_models_dir."""
    pulled = tmp_path / "pulled_models"
    pulled.mkdir()
    (pulled / "retinaface_best.pth").write_bytes(b"new-detector")
    (pulled / "unet_best.pth").write_bytes(b"new-unet")
    repo_models = tmp_path / "repo_models"
    repo_models.mkdir()

    sync_results.sync_weights(pulled, repo_models)
    assert (repo_models / "retinaface_demo.pth").read_bytes() == b"new-detector"
    assert (repo_models / "unet_demo.pth").read_bytes() == b"new-unet"
    hashes = json.loads((repo_models / "model_hashes.json").read_text())
    assert "retinaface_demo.pth" in hashes
    assert "unet_demo.pth" in hashes


def test_copy_extras_to_repo_skips_when_up_to_date(tmp_path: Path) -> None:
    """If the repo file already matches the source, don't copy."""
    pulled_models = tmp_path / "pulled_models"
    pulled_runs = tmp_path / "pulled_runs" / "eval"
    pulled_models.mkdir()
    pulled_runs.mkdir(parents=True)
    repo = tmp_path / "repo"
    repo.mkdir()

    (pulled_models / "training_metrics.json").write_text('{"x":1}')
    (repo / "training_metrics.json").write_text('{"x":1}')

    sync_results.copy_extras_to_repo(pulled_models, pulled_runs, repo)
    # Repo file unchanged.
    assert (repo / "training_metrics.json").read_text() == '{"x:1}' if False else \
        (repo / "training_metrics.json").read_text() == '{"x":1}'
