"""Kiểm tra trạng thái Kaggle kernel + pull outputs.

Sử dụng Kaggle Python API (không phụ thuộc `kaggle` CLI). Cần credentials
ở `~/.kaggle/kaggle.json` hoặc biến môi trường `KAGGLE_USERNAME` / `KAGGLE_KEY`.

Usage::

    python scripts/check_kaggle_status.py                # status + file list
    python scripts/check_kaggle_status.py --download     # status + pull outputs
    python scripts/check_kaggle_status.py --kernel speril/face-detection-segmentation-patched
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

DEFAULT_KERNEL = "speril/face-detection-face-segmentation"


def get_api():
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        print("[ERROR] kaggle SDK not installed. Install: pip install kaggle")
        sys.exit(2)
    api = KaggleApi()
    try:
        api.authenticate()
    except Exception as exc:
        print(f"[ERROR] Kaggle auth failed: {exc}")
        print("        Set KAGGLE_USERNAME / KAGGLE_KEY or place ~/.kaggle/kaggle.json")
        sys.exit(2)
    return api


def check_status(api, kernel: str) -> dict:
    raw = api.kernels_status(kernel)
    if isinstance(raw, str):
        # Older SDKs return a JSON string.
        try:
            return json.loads(raw)
        except Exception:
            return {"status": raw}
    return {"status": getattr(raw, "status", None),
            "failure_message": getattr(raw, "failureMessage", None)}


def list_outputs(api, kernel: str) -> list[dict]:
    response = api.kernels_list_files(kernel)
    files = getattr(response, "files", []) or []
    parsed: list[dict] = []
    for entry in files:
        if isinstance(entry, str):
            parsed.append({"name": entry})
            continue
        if isinstance(entry, dict):
            parsed.append({
                "name": entry.get("name") or entry.get("fileName"),
                "size": entry.get("size") or entry.get("totalBytes"),
                "creation_date": entry.get("creationDate"),
            })
            continue
        # Protobuf-style object.
        parsed.append({
            "name": getattr(entry, "name", None) or getattr(entry, "fileName", None),
            "size": getattr(entry, "size", None) or getattr(entry, "totalBytes", None),
            "creation_date": getattr(entry, "creationDate", None),
        })
    return parsed


def pull_outputs(api, kernel: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[pull] {kernel} -> {out_dir}", flush=True)
    # Workaround: Kaggle SDK writes a `<kernel-slug>.log` next to the pulled
    # files using locale-default encoding (cp1252 on Windows), which crashes
    # on emoji characters emitted by Kaggle's runner. Force UTF-8 by replacing
    # `open` inside the SDK for the duration of the call.
    import builtins
    _orig_open = builtins.open
    def _utf8_open(*args, **kwargs):
        if "encoding" not in kwargs and "b" not in (args[1] if len(args) > 1 else ""):
            kwargs["encoding"] = "utf-8"
            kwargs["errors"] = "replace"
        return _orig_open(*args, **kwargs)
    builtins.open = _utf8_open
    try:
        api.kernels_output(kernel, str(out_dir))
    finally:
        builtins.open = _orig_open


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernel", type=str, default=DEFAULT_KERNEL)
    parser.add_argument("--download", action="store_true",
                        help="Pull outputs into ./kaggle_outputs_v23/<kernel-slug>/")
    parser.add_argument("--out-dir", type=Path, default=Path("./kaggle_outputs_v23"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    api = get_api()
    status = check_status(api, args.kernel)
    print(f"kernel : {args.kernel}")
    print(f"status : {status.get('status')}")
    fail = status.get("failure_message")
    if fail:
        print(f"failure: {fail}")

    files = list_outputs(api, args.kernel)
    print(f"files  : {len(files)} in output")
    for f in files:
        size = f.get("size")
        size_s = f"({size} bytes)" if size else ""
        print(f"  - {f.get('name')}  {size_s}")

    if args.download:
        slug = args.kernel.split("/", 1)[1]
        pull_outputs(api, args.kernel, args.out_dir / slug)


if __name__ == "__main__":
    main()
