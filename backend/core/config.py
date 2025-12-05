"""Configuration and runtime state for the fraud detection demo."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, Optional

import numpy as np


BASE_DIR = Path(__file__).resolve().parents[2]
DATA_FILE = BASE_DIR / "data" / "transactions_sample.csv"
AUDIT_LOG_FILE = BASE_DIR / "data" / "audit_log.jsonl"
RUNTIME_CONFIG_FILE = BASE_DIR / "data" / "runtime_config.json"
RULES_FILE = BASE_DIR / "data" / "rules.json"
DATASET_CONFIG_FILE = BASE_DIR / "data" / "dataset_config.json"
CUSTOM_DATA_DIR = BASE_DIR / "data" / "custom"
RANDOM_SEED = 42
STREAM_BUFFER_SIZE = 240  # rolling points in the realtime simulator

DEFAULT_RUNTIME_CONFIG = {
    "decision_threshold": 0.7,
    "mode": "balanced",
    "rules_enabled": True,
}


random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def _load_json(path: Path, default: dict) -> dict:
    if not path.exists():
        return default
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def _save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception:
        return


def get_runtime_config() -> dict:
    cfg = _load_json(RUNTIME_CONFIG_FILE, DEFAULT_RUNTIME_CONFIG)
    # ensure defaults
    merged = {**DEFAULT_RUNTIME_CONFIG, **cfg}
    return merged


def update_runtime_config(
    decision_threshold: Optional[float] = None,
    mode: Optional[str] = None,
    rules_enabled: Optional[bool] = None,
) -> dict:
    current = get_runtime_config()
    if decision_threshold is not None:
        current["decision_threshold"] = float(decision_threshold)
    if mode is not None:
        current["mode"] = mode
    if rules_enabled is not None:
        current["rules_enabled"] = bool(rules_enabled)
    _save_json(RUNTIME_CONFIG_FILE, current)
    return current

