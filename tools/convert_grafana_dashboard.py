#!/usr/bin/env python3
"""Convert API-wrapped Grafana dashboard JSON to UI-importable JSON."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, Tuple

REQUIRED_UID = "fraud-ai-training"
REQUIRED_PANEL_IDS = {61, 62, 63, 64, 65}


def _load_json(path: Path) -> object:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _extract_dashboard(payload: object) -> object:
    if isinstance(payload, dict) and isinstance(payload.get("dashboard"), dict):
        return payload["dashboard"]
    return payload


def _validate_dashboard(dashboard: object) -> Tuple[bool, Iterable[str]]:
    errors = []
    if not isinstance(dashboard, dict):
        return False, ["Dashboard JSON is not an object."]

    uid = dashboard.get("uid")
    if uid != REQUIRED_UID:
        errors.append(f"Dashboard uid is '{uid}', expected '{REQUIRED_UID}'.")

    panels = dashboard.get("panels", [])
    found_ids = set()
    if isinstance(panels, list):
        for panel in panels:
            if isinstance(panel, dict):
                panel_id = panel.get("id")
                if isinstance(panel_id, int):
                    found_ids.add(panel_id)

    missing = REQUIRED_PANEL_IDS - found_ids
    if missing:
        errors.append(f"Missing panel IDs: {sorted(missing)}.")

    return len(errors) == 0, errors


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert Grafana API dashboard JSON to UI-importable JSON."
    )
    parser.add_argument(
        "--input",
        default="grafana/dashboards/fraud_ai_model_training_dashboard.json",
        help="Path to API-wrapped or raw dashboard JSON.",
    )
    parser.add_argument(
        "--output",
        default="grafana/dashboards/fraud_ai_model_training_dashboard.UI.json",
        help="Path to write the UI-importable dashboard JSON.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    try:
        payload = _load_json(input_path)
    except Exception as exc:  # noqa: BLE001
        print(f"Failed to read input JSON: {exc}", file=sys.stderr)
        return 1

    dashboard = _extract_dashboard(payload)
    ok, errors = _validate_dashboard(dashboard)
    if not ok:
        for err in errors:
            print(f"Validation error: {err}", file=sys.stderr)
        return 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(dashboard, handle, indent=2)
            handle.write("\n")
    except Exception as exc:  # noqa: BLE001
        print(f"Failed to write output JSON: {exc}", file=sys.stderr)
        return 1

    try:
        output_payload = _load_json(output_path)
    except Exception as exc:  # noqa: BLE001
        print(f"Failed to re-parse output JSON: {exc}", file=sys.stderr)
        return 1

    ok, errors = _validate_dashboard(output_payload)
    if not ok:
        for err in errors:
            print(f"Validation error: {err}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
