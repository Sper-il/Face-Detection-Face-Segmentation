"""Smoke tests — verify the project skeleton is wired correctly.

These tests do NOT require trained weights or datasets.
Run with: `pytest tests/test_imports.py` or simply `pytest`.
"""

from __future__ import annotations

import importlib


def test_src_package_importable():
    mod = importlib.import_module("src")
    assert mod.__version__ == "0.1.0"


def test_subpackages_importable():
    for name in ("detection", "segmentation", "pipeline", "utils", "configs"):
        importlib.import_module(f"src.{name}")


def test_pipeline_cli_help_runs(capsys):
    """`python -m src.pipeline.run --help` should print usage and exit cleanly."""
    from src.pipeline import run as pipeline_run

    # Simulate --help by patching sys.argv
    import sys

    saved = sys.argv
    try:
        sys.argv = ["run", "--help"]
        try:
            pipeline_run.main()
        except SystemExit:
            pass
    finally:
        sys.argv = saved

    captured = capsys.readouterr()
    # argparse prints usage to stdout (or stderr in some versions); either is fine.
    assert "CLI OK" in captured.out or "usage" in captured.out.lower()