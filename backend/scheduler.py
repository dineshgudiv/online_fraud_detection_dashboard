"""Simple scheduler loop for drift snapshots."""

from __future__ import annotations

import os
import time

from tasks.drift import run_drift_snapshot


def main() -> None:
    interval = int(os.getenv("DRIFT_INTERVAL_SECONDS", "3600"))
    while True:
        run_drift_snapshot()
        time.sleep(interval)


if __name__ == "__main__":
    main()
