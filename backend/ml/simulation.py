"""Simulation utilities for realtime metrics and drift reports."""

from __future__ import annotations

import random
from collections import deque
from datetime import datetime
from typing import Deque, Dict, Tuple

import numpy as np
import pandas as pd

from backend.core.config import STREAM_BUFFER_SIZE
from backend.ml.engine import load_or_create_dataset, score_transaction


class RealTimeBuffer:
    def __init__(self, maxlen: int = STREAM_BUFFER_SIZE):
        self.points: Deque[Tuple[str, float, float]] = deque(maxlen=maxlen)
        self.total_processed: int = 0

    def append(self, ts: str, events: float, fraud_events: float) -> None:
        self.points.append((ts, events, fraud_events))
        self.total_processed += int(events)


BUFFER = RealTimeBuffer()


def _sample_transactions(n: int = 12) -> pd.DataFrame:
    df = load_or_create_dataset()
    return df.sample(n=min(n, len(df)), replace=False)


def _generate_point() -> Tuple[str, float, float]:
    df = _sample_transactions(8)
    fraud_events = 0
    for _, row in df.iterrows():
        features = {
            "user_id": row["user_id"],
            "merchant_id": row["merchant_id"],
            "amount": row["amount"],
            "currency": row["currency"],
            "transaction_type": row["transaction_type"],
            "category": row["category"],
            "location": row["location"],
            "device_id": row["device_id"],
        }
        result = score_transaction(features)
        fraud_events += 1 if result["fraud_score"] > 0.65 else 0

    events = float(len(df))
    ts = datetime.utcnow().strftime("%H:%M:%S")
    return ts, events, float(fraud_events)


def get_realtime_metrics() -> Dict:
    ts, events, fraud_events = _generate_point()
    BUFFER.append(ts, events, fraud_events)

    if len(BUFFER.points) > 1:
        recent = list(BUFFER.points)[-10:]
    else:
        recent = list(BUFFER.points)

    events_per_second = round(events / 60.0 + random.uniform(0.2, 1.5), 3)
    error_rate = round(max(0.1, np.random.normal(0.6, 0.2)), 2)
    current_events_per_min = round(events + random.uniform(0, 3), 2)
    fraud_events_per_min = round(fraud_events + random.uniform(0, 1), 2)
    fraud_rate_percent = round((fraud_events / events) * 100 if events else 0, 2)

    points = [
        {"timestamp": p[0], "events": p[1], "fraud_events": p[2]}
        for p in recent
    ]

    return {
        "processed_events": BUFFER.total_processed,
        "events_per_second": events_per_second,
        "error_rate": error_rate,
        "current_events_per_min": current_events_per_min,
        "fraud_events_per_min": fraud_events_per_min,
        "realtime_fraud_rate": fraud_rate_percent,
        "points": points,
    }


def _drift_status(score: float) -> str:
    if score < 0.3:
        return "ok"
    if score < 0.7:
        return "warning"
    return "drifted"


def compute_drift_report(window_size: int = 200) -> Dict:
    """Compute a lightweight drift report using random reference/current samples."""
    df = load_or_create_dataset().copy()
    df["timestamp"] = pd.to_datetime(df.get("timestamp"))
    ref = df.sample(min(window_size, len(df)), replace=True, random_state=random.randint(0, 9999))
    cur = df.sample(min(window_size, len(df)), replace=True, random_state=random.randint(0, 9999))

    features = []

    # Amount drift: normalized mean shift
    ref_mean, cur_mean = ref["amount"].mean(), cur["amount"].mean()
    ref_std = max(ref["amount"].std(), 1)
    amount_score = min(1.0, abs(cur_mean - ref_mean) / (ref_std * 3) + random.uniform(0, 0.05))
    features.append(
        {
            "feature": "amount",
            "drift_score": round(amount_score, 3),
            "status": _drift_status(amount_score),
            "p_value": round(random.uniform(0.01, 0.2), 3),
            "comment": f"Mean shift {cur_mean:.2f} vs {ref_mean:.2f}",
        }
    )

    # Hour drift
    ref_hours = ref["timestamp"].dt.hour.value_counts(normalize=True)
    cur_hours = cur["timestamp"].dt.hour.value_counts(normalize=True)
    aligned = pd.concat([ref_hours, cur_hours], axis=1).fillna(0)
    aligned.columns = ["ref", "cur"]
    hour_score = min(1.0, (aligned["ref"] - aligned["cur"]).abs().sum())
    features.append(
        {
            "feature": "hour",
            "drift_score": round(hour_score, 3),
            "status": _drift_status(hour_score),
            "p_value": round(random.uniform(0.02, 0.3), 3),
            "comment": "Hourly distribution shift",
        }
    )

    # Location drift
    ref_loc = ref["location"].value_counts(normalize=True)
    cur_loc = cur["location"].value_counts(normalize=True)
    loc = pd.concat([ref_loc, cur_loc], axis=1).fillna(0)
    loc.columns = ["ref", "cur"]
    loc_score = min(1.0, (loc["ref"] - loc["cur"]).abs().sum())
    features.append(
        {
            "feature": "location",
            "drift_score": round(loc_score, 3),
            "status": _drift_status(loc_score),
            "p_value": round(random.uniform(0.05, 0.35), 3),
            "comment": "Geo distribution variance",
        }
    )

    return {
        "window_size": int(window_size),
        "reference_period": "Historical sample",
        "current_period": "Recent sample",
        "features": features,
    }

