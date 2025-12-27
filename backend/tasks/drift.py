"""Drift metric computation (demo-capable)."""

from __future__ import annotations

import math
import os
import random
from datetime import datetime
from typing import Iterable

from db import models
from db.session import SessionLocal


def _utcnow() -> datetime:
    return datetime.utcnow()


def _model_version() -> str:
    return os.getenv("MODEL_VERSION", _utcnow().strftime("%Y.%m.%d"))


def _psi(expected: Iterable[float], actual: Iterable[float], buckets: int = 10) -> float:
    expected = list(expected)
    actual = list(actual)
    if not expected or not actual:
        return 0.0
    min_val = min(min(expected), min(actual))
    max_val = max(max(expected), max(actual))
    if min_val == max_val:
        return 0.0
    step = (max_val - min_val) / buckets
    psi = 0.0
    for i in range(buckets):
        low = min_val + step * i
        high = low + step
        expected_count = sum(1 for x in expected if low <= x < high)
        actual_count = sum(1 for x in actual if low <= x < high)
        expected_pct = max(expected_count / len(expected), 0.0001)
        actual_pct = max(actual_count / len(actual), 0.0001)
        psi += (actual_pct - expected_pct) * math.log(actual_pct / expected_pct)
    return round(float(psi), 4)


def _demo_series(seed: int, drift: float) -> tuple[list[float], list[float]]:
    random.seed(seed)
    baseline = [random.gauss(0, 1) for _ in range(500)]
    current = [random.gauss(drift, 1.1) for _ in range(500)]
    return baseline, current


def run_drift_snapshot() -> None:
    db = SessionLocal()
    try:
        now = _utcnow()
        model_version = _model_version()
        feature_specs = {
            "amount": 0.35,
            "velocity": 0.15,
            "device_risk": 0.05,
            "geo_risk": 0.2,
            "score": 0.1,
        }
        metrics: list[models.DriftMetric] = []
        psi_values = []
        for idx, (feature, drift) in enumerate(feature_specs.items(), start=1):
            baseline, current = _demo_series(idx * 100, drift)
            psi_value = _psi(baseline, current)
            psi_values.append(psi_value)
            metrics.append(
                models.DriftMetric(
                    timestamp=now,
                    model_version=model_version,
                    feature=feature,
                    psi=psi_value,
                    ks_pvalue=None,
                    overall_score=0.0,
                )
            )
        overall = round(max(psi_values) if psi_values else 0.0, 4)
        for metric in metrics:
            metric.overall_score = overall
            db.add(metric)
        db.commit()
    finally:
        db.close()
