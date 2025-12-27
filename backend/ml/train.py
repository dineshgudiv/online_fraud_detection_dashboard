"""Offline training script to produce a demo artifact."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

ARTIFACT_PATH = Path(__file__).resolve().parent / "artifacts" / "demo_model.json"


def main() -> None:
    artifact = {
        "name": "demo-linear",
        "version": datetime.utcnow().strftime("%Y.%m.%d"),
        "threshold": 0.7,
        "bias": -2.0,
        "weights": {
            "amount": 0.0025,
            "high_risk_category": 0.6,
            "high_risk_country": 0.5,
            "withdrawal": 0.4,
        },
    }
    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT_PATH.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    print(f"Wrote artifact to {ARTIFACT_PATH}")


if __name__ == "__main__":
    main()
