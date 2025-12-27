"""Lightweight scoring module that loads a versioned artifact."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

ARTIFACT_PATH = Path(__file__).resolve().parent / "artifacts" / "demo_model.json"


@dataclass(frozen=True)
class ModelArtifact:
    name: str
    version: str
    threshold: float
    bias: float
    weights: Dict[str, float]


_ARTIFACT: ModelArtifact | None = None


def load_artifact() -> ModelArtifact:
    global _ARTIFACT
    if _ARTIFACT is not None:
        return _ARTIFACT
    if not ARTIFACT_PATH.exists():
        raise RuntimeError("Model artifact missing")
    data = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))
    _ARTIFACT = ModelArtifact(
        name=str(data["name"]),
        version=str(data["version"]),
        threshold=float(data["threshold"]),
        bias=float(data["bias"]),
        weights={k: float(v) for k, v in data["weights"].items()},
    )
    return _ARTIFACT


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _risk_level(score: float) -> str:
    if score >= 0.85:
        return "CRITICAL"
    if score >= 0.7:
        return "HIGH"
    if score >= 0.45:
        return "MEDIUM"
    return "LOW"


def score_transaction(features: dict) -> dict:
    artifact = load_artifact()
    amount = float(features.get("amount", 0.0))
    category = str(features.get("category", ""))
    txn_type = str(features.get("transaction_type", ""))
    location = str(features.get("location", ""))

    risk_factors: List[str] = []
    logit = artifact.bias
    logit += artifact.weights.get("amount", 0.0) * amount

    if category.lower() in {"crypto", "gaming"}:
        logit += artifact.weights.get("high_risk_category", 0.0)
        risk_factors.append("High-risk category")

    if location.upper() in {"CN", "BR", "RU"}:
        logit += artifact.weights.get("high_risk_country", 0.0)
        risk_factors.append("High-risk geography")

    if txn_type.lower() in {"withdrawal", "transfer"}:
        logit += artifact.weights.get("withdrawal", 0.0)
        risk_factors.append("Risky transaction type")

    score = round(_sigmoid(logit), 3)
    risk_level = _risk_level(score)
    decision = "REJECTED" if score >= artifact.threshold else "APPROVED"

    if not risk_factors:
        risk_factors.append("No strong risk signals detected")

    return {
        "fraud_score": score,
        "risk_level": risk_level,
        "decision": decision,
        "risk_factors": risk_factors,
        "model_name": artifact.name,
        "model_version": artifact.version,
    }

