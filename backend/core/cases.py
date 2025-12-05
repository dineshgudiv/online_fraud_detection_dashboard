"""Simple case management persistence."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import List, Optional

from backend.core.config import BASE_DIR


CASES_FILE = BASE_DIR / "data" / "cases.json"


def _load_cases() -> List[dict]:
    if not CASES_FILE.exists():
        return []
    try:
        with CASES_FILE.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def _save_cases(cases: List[dict]) -> None:
    CASES_FILE.parent.mkdir(parents=True, exist_ok=True)
    try:
        with CASES_FILE.open("w", encoding="utf-8") as f:
            json.dump(cases, f, indent=2)
    except Exception:
        return


def list_cases() -> List[dict]:
    return _load_cases()


def _next_case_id(cases: List[dict]) -> str:
    return f"CASE-{len(cases)+1:06d}"


def create_case(data: dict) -> dict:
    cases = _load_cases()
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    case = {
        "case_id": _next_case_id(cases),
        "status": "OPEN",
        "created_at": now,
        "updated_at": now,
        **data,
    }
    cases.append(case)
    _save_cases(cases)
    return case


def update_case(case_id: str, updates: dict) -> Optional[dict]:
    cases = _load_cases()
    updated = None
    for c in cases:
        if c.get("case_id") == case_id:
            c.update({k: v for k, v in updates.items() if v is not None})
            c["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            updated = c
            break
    if updated:
        _save_cases(cases)
    return updated


def get_case(case_id: str) -> Optional[dict]:
    for c in _load_cases():
        if c.get("case_id") == case_id:
            return c
    return None


def create_case_from_alert(alert: dict) -> dict:
    data = {
        "title": f"Alert {alert.get('id')} investigation",
        "user_id": alert.get("user_id"),
        "transaction_id": alert.get("transaction_id"),
        "risk_level": alert.get("risk_level"),
        "fraud_score": alert.get("fraud_score"),
        "created_from_alert_id": alert.get("id"),
        "notes": alert.get("reason", ""),
    }
    return create_case(data)

