"""Poll Kaggle kernel status until it completes (or fails).

Sử dụng Kaggle Python API. In log mỗi interval giây và ghi `last_status.json`.

Usage::

    python scripts/poll_kaggle_kernel.py --interval 600   # check every 10 minutes
    python scripts/poll_kaggle_kernel.py --once           # single check
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_KERNEL = "speril/face-detection-face-segmentation"


def get_api():
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    return api


def status(api, kernel: str) -> str:
    raw = api.kernels_status(kernel)
    if isinstance(raw, str):
        try:
            return json.loads(raw).get("status", raw)
        except Exception:
            return raw
    return getattr(raw, "status", None) or str(raw)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernel", type=str, default=DEFAULT_KERNEL)
    parser.add_argument("--interval", type=int, default=600,
                        help="Seconds between checks (default 600s = 10min)")
    parser.add_argument("--max-wait", type=int, default=12 * 3600,
                        help="Stop after this many seconds total (default 12h)")
    parser.add_argument("--once", action="store_true",
                        help="Single check (no polling)")
    parser.add_argument("--log", type=Path, default=Path("kaggle_outputs_v23/poll.log"))
    return parser.parse_args()


def log_line(log_path: Path, line: str) -> None:
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    print(f"[{ts}] {line}")
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(f"[{ts}] {line}\n")


def main() -> None:
    args = parse_args()
    api = get_api()
    args.log.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    while True:
        try:
            st = status(api, args.kernel)
        except Exception as exc:
            st = f"ERROR: {exc}"
        log_line(args.log, f"status={st}")

        if st in {"KernelWorkerStatus.COMPLETE", "COMPLETE", "complete",
                  "KernelWorkerStatus.FAILED", "FAILED", "failed",
                  "KernelWorkerStatus.ERROR", "ERROR"}:
            log_line(args.log, "terminal state reached - exiting poll loop")
            Path("kaggle_outputs_v23/last_status.json").write_text(
                json.dumps({"kernel": args.kernel, "status": str(st),
                            "checked_at": datetime.now(timezone.utc).isoformat()},
                           indent=2))
            sys.exit(0 if "COMPLETE" in str(st) else 1)

        if args.once:
            sys.exit(0)
        if time.time() - started >= args.max_wait:
            log_line(args.log, "max-wait reached - exiting")
            sys.exit(0)
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
