"""Lightweight ML engine with runtime config, rules, audit logging, datasets, cases, batch scoring, and executive summary."""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from backend.core.cases import list_cases, create_case, update_case, create_case_from_alert
from backend.core.config import (
    AUDIT_LOG_FILE,
    CUSTOM_DATA_DIR,
    DATASET_CONFIG_FILE,
    DATA_FILE,
    DEFAULT_RUNTIME_CONFIG,
    RULES_FILE,
    RANDOM_SEED,
    get_runtime_config,
    update_runtime_config,
    _load_json,
    _save_json,
    BASE_DIR,
)


np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

BATCH_OUTPUT_DIR = BASE_DIR / "data" / "batch_results"
BATCH_INPUT_DIR = BASE_DIR / "data" / "batch_input"


@dataclass
class ModelStore:
    logistic_regression: LogisticRegression
    random_forest: RandomForestClassifier
    isolation_forest: IsolationForest
    feature_columns: List[str] = field(default_factory=list)
    accuracy: Dict[str, float] = field(default_factory=dict)


MODELS: ModelStore | None = None
DATASET: pd.DataFrame | None = None
MODEL_HEALTH: Dict | None = None
ACTIVE_DATASET_NAME: str = "default"


# ---------- Audit Log ----------
def append_audit_log(entry: dict) -> None:
    """Append a single audit entry to the JSONL log file."""
    try:
        AUDIT_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with AUDIT_LOG_FILE.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception:
        return


def read_audit_log(limit: int | None = None) -> List[dict]:
    """Read audit log entries, returning the most recent `limit` rows."""
    if not AUDIT_LOG_FILE.exists():
        return []
    try:
        with AUDIT_LOG_FILE.open("r", encoding="utf-8") as f:
            lines = f.readlines()
        if limit:
            lines = lines[-limit:]
        entries = []
        for line in lines:
            try:
                entries.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                continue
        return entries
    except Exception:
        return []


# ---------- Rules ----------
def _default_rules() -> List[dict]:
    return [
        {
            "id": "rule_high_amount_night",
            "enabled": True,
            "description": "Flag very high amount transactions at night.",
            "condition": {"amount_gt": 1000, "hour_in": [0, 1, 2, 3, 4, 23]},
            "action": {"set_min_score": 0.85, "force_decision": "REJECT"},
            "severity": "HIGH",
        }
    ]


def load_rules() -> List[dict]:
    if not RULES_FILE.exists():
        save_rules(_default_rules())
    try:
        with RULES_FILE.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else _default_rules()
    except Exception:
        return _default_rules()


def save_rules(rules: List[dict]) -> None:
    RULES_FILE.parent.mkdir(parents=True, exist_ok=True)
    try:
        with RULES_FILE.open("w", encoding="utf-8") as f:
            json.dump(rules, f, indent=2)
    except Exception:
        return


def _risk_level_from_score(score: float) -> str:
    if score < 0.3:
        return "LOW"
    if score < 0.5:
        return "MEDIUM"
    if score < 0.7:
        return "HIGH"
    return "CRITICAL"


def _apply_condition(rule_cond: dict, tx: dict, txn_hour: int) -> bool:
    if rule_cond.get("amount_gt") is not None and tx.get("amount", 0) <= rule_cond["amount_gt"]:
        return False
    if rule_cond.get("country_in"):
        if tx.get("location") not in rule_cond["country_in"]:
            return False
    if rule_cond.get("hour_in"):
        if txn_hour not in rule_cond["hour_in"]:
            return False
    if rule_cond.get("merchant_in"):
        if tx.get("merchant_id") not in rule_cond["merchant_in"]:
            return False
    if rule_cond.get("user_in"):
        if tx.get("user_id") not in rule_cond["user_in"]:
            return False
    return True


def apply_rules(
    base_score: float,
    base_risk: str,
    base_decision: str,
    tx_features: dict,
) -> Tuple[float, str, str, List[str]]:
    """Apply rule-based overrides to a scored transaction."""
    txn_time = tx_features.get("timestamp")
    try:
        txn_hour = pd.to_datetime(txn_time).hour if txn_time else pd.Timestamp.utcnow().hour
    except Exception:
        txn_hour = pd.Timestamp.utcnow().hour

    rules = load_rules()
    new_score = base_score
    new_risk = base_risk
    new_decision = base_decision
    hits: List[str] = []

    for rule in rules:
        if not rule.get("enabled", True):
            continue
        cond = rule.get("condition", {})
        if not _apply_condition(cond, tx_features, txn_hour):
            continue
        action = rule.get("action", {})
        hits.append(rule.get("description", rule.get("id", "rule")))

        if action.get("set_min_score") is not None:
            new_score = max(new_score, float(action["set_min_score"]))
        if action.get("bump_score") is not None:
            new_score = min(1.0, new_score + float(action["bump_score"]))
        if action.get("set_risk_level"):
            new_risk = action["set_risk_level"]
        else:
            new_risk = _risk_level_from_score(new_score)
        if action.get("force_decision"):
            new_decision = "REJECTED" if action["force_decision"].upper() == "REJECT" else "APPROVED"
        else:
            # recompute decision based on updated score later; keep as is for now
            pass

    return new_score, new_risk, new_decision, hits


# ---------- Dataset management ----------
def _load_dataset_config() -> dict:
    default_cfg = {"active": "default"}
    return _load_json(DATASET_CONFIG_FILE, default_cfg)


def _save_dataset_config(cfg: dict) -> None:
    _save_json(DATASET_CONFIG_FILE, cfg)


def _active_dataset_path() -> Path:
    cfg = _load_dataset_config()
    name = cfg.get("active", "default")
    global ACTIVE_DATASET_NAME
    ACTIVE_DATASET_NAME = name
    if name == "default":
        return DATA_FILE
    return CUSTOM_DATA_DIR / f"{name}.csv"


def list_datasets() -> dict:
    available: List[dict] = []
    paths = [(DATA_FILE, "default")]
    CUSTOM_DATA_DIR.mkdir(parents=True, exist_ok=True)
    for p in CUSTOM_DATA_DIR.glob("*.csv"):
        paths.append((p, p.stem))
    for path, name in paths:
        try:
            df = pd.read_csv(path)
            available.append({"name": name, "path": str(path), "num_rows": len(df)})
        except Exception:
            available.append({"name": name, "path": str(path), "num_rows": 0})
    active = _load_dataset_config().get("active", "default")
    return {"active": active, "available": available}


def set_active_dataset(name: str) -> dict:
    datasets = list_datasets()["available"]
    names = {d["name"] for d in datasets}
    if name not in names:
        raise ValueError(f"Dataset {name} not found")
    _save_dataset_config({"active": name})
    load_or_create_dataset(force_reload=True)
    train_models()
    return list_datasets()


def upload_dataset(file_bytes: bytes, filename: str) -> dict:
    CUSTOM_DATA_DIR.mkdir(parents=True, exist_ok=True)
    stem = Path(filename).stem
    target = CUSTOM_DATA_DIR / f"{stem}.csv"
    with target.open("wb") as f:
        f.write(file_bytes)
    try:
        df = pd.read_csv(target)
        num_rows = len(df)
    except Exception:
        num_rows = 0
    return {"name": stem, "path": str(target), "num_rows": num_rows}


# ---------- Data and models ----------
def _dataset_is_valid(df: pd.DataFrame) -> bool:
    """Basic validation to ensure dataset can be split and trained."""
    if df is None or df.empty:
        return False
    if "fraud_label" not in df.columns:
        return False
    if len(df) < 30:
        return False
    labels = df["fraud_label"].dropna().unique()
    if len(labels) < 2:
        return False
    counts = df["fraud_label"].value_counts()
    if counts.min() < 2:
        return False
    return True


def _generate_risk_level(fraud_label: int, amount: float) -> str:
    if fraud_label == 1 and amount > 900:
        return "CRITICAL"
    if fraud_label == 1 and amount > 500:
        return "HIGH"
    if amount > 400:
        return "HIGH"
    if fraud_label == 1:
        return "MEDIUM"
    return random.choice(["LOW", "MEDIUM"])


def _ensure_default_dataset() -> pd.DataFrame:
    """Generate a synthetic dataset if the default file is missing."""
    n_rows = 600
    user_ids = [f"U{1000+i}" for i in range(120)]
    merchant_ids = [f"M{200+i}" for i in range(40)]
    currencies = ["USD", "EUR", "GBP", "JPY", "INR"]
    categories = ["electronics", "fashion", "groceries", "gaming", "travel", "crypto"]
    transaction_types = ["purchase", "withdrawal", "transfer", "refund"]
    locations = ["US", "GB", "DE", "FR", "IN", "CN", "SG", "BR", "ZA", "AU"]
    device_ids = [f"D{500+i}" for i in range(100)]

    rows = []
    for i in range(n_rows):
        amount = np.round(np.random.exponential(scale=180) + np.random.uniform(10, 1200), 2)
        fraud_base = 0.12 + (0.1 if amount > 800 else 0) + (0.06 if random.random() < 0.15 else 0)
        fraud_label = int(np.random.rand() < min(fraud_base, 0.45))
        risk_level = _generate_risk_level(fraud_label, amount)
        rows.append(
            {
                "transaction_id": f"T{100000 + i}",
                "user_id": random.choice(user_ids),
                "merchant_id": random.choice(merchant_ids),
                "amount": amount,
                "currency": random.choice(currencies),
                "category": random.choice(categories),
                "transaction_type": random.choice(transaction_types),
                "timestamp": pd.Timestamp.now() - pd.Timedelta(minutes=np.random.randint(0, 720)),
                "location": random.choice(locations),
                "device_id": random.choice(device_ids),
                "fraud_label": fraud_label,
                "risk_level": risk_level,
            }
        )

    df = pd.DataFrame(rows)
    DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA_FILE, index=False)
    return df


def load_or_create_dataset(force_reload: bool = False) -> pd.DataFrame:
    """Load dataset based on active dataset config or create a synthetic one."""
    global DATASET
    path = _active_dataset_path()
    if DATASET is not None and not force_reload:
        return DATASET
    if path.exists():
        try:
            df = pd.read_csv(path)
            if _dataset_is_valid(df):
                DATASET = df
                return DATASET
        except Exception:
            pass

    # fallback to default synthetic dataset
    DATASET = _ensure_default_dataset()
    _save_dataset_config({"active": "default"})
    return DATASET


def _prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    feature_cols = [
        "amount",
        "currency",
        "category",
        "transaction_type",
        "location",
        "device_id",
        "merchant_id",
        "user_id",
    ]
    X = pd.get_dummies(df[feature_cols], drop_first=True)
    y = df["fraud_label"]
    return X, y, list(X.columns)


def _probabilities_from_models(models: ModelStore, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    lr_prob = models.logistic_regression.predict_proba(X)[:, 1]
    rf_prob = models.random_forest.predict_proba(X)[:, 1]
    iso_raw = models.isolation_forest.decision_function(X)
    iso_prob = 1 - (1 / (1 + np.exp(iso_raw * 5)))
    return lr_prob, rf_prob, iso_prob


def _metric_block(name: str, y_true: np.ndarray, probs: np.ndarray, threshold: float = 0.7) -> Dict:
    preds = (probs >= threshold).astype(int)
    acc = accuracy_score(y_true, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, preds, average="binary", zero_division=0)
    try:
        auc = roc_auc_score(y_true, probs)
    except Exception:
        auc = None
    return {
        "name": name,
        "accuracy": round(float(acc), 3),
        "precision": round(float(precision), 3),
        "recall": round(float(recall), 3),
        "f1": round(float(f1), 3),
        "auc": round(float(auc), 3) if auc is not None else None,
        "preds": preds,
    }


def train_models() -> ModelStore:
    """Train lightweight models, compute metrics, and keep them in memory."""
    global MODELS, DATASET, MODEL_HEALTH
    load_or_create_dataset(force_reload=True)
    X, y, feature_columns = _prepare_features(DATASET)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y)

    lr = LogisticRegression(max_iter=400)
    rf = RandomForestClassifier(
        n_estimators=120,
        max_depth=8,
        random_state=RANDOM_SEED,
        min_samples_leaf=3,
    )
    iso = IsolationForest(
        n_estimators=80,
        contamination=0.12,
        random_state=RANDOM_SEED,
    )

    lr.fit(X_train, y_train)
    rf.fit(X_train, y_train)
    iso.fit(X_train)

    MODELS = ModelStore(
        logistic_regression=lr,
        random_forest=rf,
        isolation_forest=iso,
        feature_columns=feature_columns,
        accuracy={},
    )

    lr_prob, rf_prob, iso_prob = _probabilities_from_models(MODELS, X_test)
    ensemble_prob = 0.45 * rf_prob + 0.45 * lr_prob + 0.1 * iso_prob

    m_lr = _metric_block("logistic_regression", y_test, lr_prob)
    m_rf = _metric_block("random_forest", y_test, rf_prob)
    m_iso = _metric_block("isolation_forest", y_test, iso_prob)
    m_ens = _metric_block("ensemble", y_test, ensemble_prob)

    cm = confusion_matrix(y_test, m_ens["preds"]).tolist()
    class_distribution = {str(int(cls)): int(count) for cls, count in zip(*np.unique(y_test, return_counts=True))}

    MODEL_HEALTH = {
        "overall": {k: v for k, v in m_ens.items() if k != "preds"},
        "models": [{k: v for k, v in m.items() if k != "preds"} for m in [m_lr, m_rf, m_iso]],
        "confusion_matrix": {"matrix": cm, "labels": ["legit", "fraud"]},
        "class_distribution": class_distribution,
    }
    return MODELS


def get_model_health() -> Dict:
    if MODEL_HEALTH is None:
        train_models()
    return MODEL_HEALTH or {}


def _vectorize(features: dict) -> pd.DataFrame:
    """Convert input features into the training feature space."""
    if MODELS is None:
        train_models()
    assert MODELS is not None
    df = pd.DataFrame([features])
    vec = pd.get_dummies(df, drop_first=True)
    vec = vec.reindex(columns=MODELS.feature_columns, fill_value=0)
    return vec


def _score_vector(vec: pd.DataFrame) -> Tuple[float, float, float, float]:
    assert MODELS is not None
    lr_prob = float(MODELS.logistic_regression.predict_proba(vec)[0][1])
    rf_prob = float(MODELS.random_forest.predict_proba(vec)[0][1])
    iso_score_raw = float(MODELS.isolation_forest.decision_function(vec)[0])
    iso_score = 1 - (1 / (1 + math.exp(iso_score_raw * 5)))
    fraud_score = round(0.45 * rf_prob + 0.45 * lr_prob + 0.1 * iso_score, 3)
    return fraud_score, lr_prob, rf_prob, iso_score


def score_transaction(features_dict: dict) -> dict:
    """Score a transaction, apply rules/modes, and log an audit record."""
    if MODELS is None:
        train_models()
    assert MODELS is not None

    vec = _vectorize(features_dict)
    fraud_score, lr_prob, rf_prob, iso_score = _score_vector(vec)

    runtime_cfg = get_runtime_config()
    threshold = float(runtime_cfg.get("decision_threshold", DEFAULT_RUNTIME_CONFIG["decision_threshold"]))
    mode = runtime_cfg.get("mode", "balanced")

    # Mode adjustments
    if mode == "strict":
        threshold = max(threshold, 0.6)
        fraud_score = min(1.0, fraud_score + 0.05)
    elif mode == "permissive":
        fraud_score = max(0.0, fraud_score - 0.05)
    # balanced: no change

    risk_level = _risk_level_from_score(fraud_score)
    decision = "APPROVED" if fraud_score < threshold else "REJECTED"

    rule_hits: List[str] = []
    if runtime_cfg.get("rules_enabled", True):
        fraud_score, rule_risk, rule_decision, hits = apply_rules(fraud_score, risk_level, decision, features_dict)
        rule_hits = hits
        risk_level = rule_risk if rule_risk else _risk_level_from_score(fraud_score)
        decision = rule_decision if rule_decision else ("APPROVED" if fraud_score < threshold else "REJECTED")

    risk_factors = []
    amount = features_dict.get("amount", 0)
    category = features_dict.get("category", "")
    txn_type = features_dict.get("transaction_type", "")
    location = features_dict.get("location", "")

    if amount > 900:
        risk_factors.append("Unusually high transaction amount")
    if category in {"crypto", "gaming"}:
        risk_factors.append(f"High-risk category detected: {category}")
    if txn_type in {"withdrawal", "transfer"}:
        risk_factors.append(f"Manual review due to {txn_type} pattern")
    if location in {"CN", "BR", "RU"}:
        risk_factors.append("Transaction from elevated-risk geography")
    if iso_score > 0.6:
        risk_factors.append("Anomaly detected by isolation forest")
    if not risk_factors:
        risk_factors.append("No strong risk signals detected")
    risk_factors.extend(rule_hits)

    recommendations = [
        "Trigger step-up authentication for the user",
        "Check device fingerprint and IP reputation",
        "Throttle velocity for similar merchants",
        "Enable manual review for repeated high-value attempts",
    ]
    if decision == "REJECTED":
        recommendations.insert(0, "Place temporary hold and notify fraud operations")

    txn_id = features_dict.get("transaction_id") or f"T-{int(time.time() * 1000)}-{random.randint(100,999)}"
    audit_entry = {
        "timestamp": pd.Timestamp.utcnow().isoformat(),
        "transaction_id": txn_id,
        "user_id": features_dict.get("user_id", "unknown"),
        "merchant_id": features_dict.get("merchant_id", "unknown"),
        "amount": float(features_dict.get("amount", 0)),
        "currency": features_dict.get("currency", "USD"),
        "model_name": "ensemble",
        "fraud_score": fraud_score,
        "risk_level": risk_level,
        "decision": decision,
        "risk_factors": risk_factors,
    }
    append_audit_log(audit_entry)

    return {
        "decision": decision,
        "risk_level": risk_level,
        "fraud_score": fraud_score,
        "risk_factors": risk_factors,
        "recommendations": recommendations,
        "model_breakdown": {
            "logistic_regression": round(lr_prob, 3),
            "random_forest": round(rf_prob, 3),
            "isolation_forest": round(iso_score, 3),
        },
        "transaction_id": txn_id,
        "threshold_used": threshold,
        "mode_used": mode,
        "rule_hits": rule_hits,
    }


# ---------- Alerts ----------
def get_alerts(limit: int = 50) -> dict:
    entries = read_audit_log(limit=None)
    items: List[dict] = []
    high_risk_count = 0
    user_rejections: Dict[str, int] = {}
    for entry in entries[-limit:]:
        rl = entry.get("risk_level", "LOW")
        if rl in {"HIGH", "CRITICAL"}:
            high_risk_count += 1
        if entry.get("decision") == "REJECTED":
            user_rejections[entry.get("user_id", "unknown")] = user_rejections.get(entry.get("user_id", "unknown"), 0) + 1
        items.append(
            {
                "id": entry.get("transaction_id", ""),
                "type": "high_risk" if rl in {"HIGH", "CRITICAL"} else "info",
                "user_id": entry.get("user_id"),
                "transaction_id": entry.get("transaction_id"),
                "risk_level": rl,
                "fraud_score": entry.get("fraud_score", 0),
                "decision": entry.get("decision", ""),
                "reason": "; ".join(entry.get("risk_factors", [])) if entry.get("risk_factors") else "Triggered alert",
                "timestamp": entry.get("timestamp", ""),
            }
        )
    total_alerts = len(items)
    users_flagged = len([u for u, cnt in user_rejections.items() if cnt >= 2])
    return {
        "total_alerts": total_alerts,
        "high_risk_count": high_risk_count,
        "users_flagged": users_flagged,
        "items": items[::-1],  # latest first
    }


# ---------- Batch scoring ----------
def score_csv_file(input_path: Path, output_name: str | None = None) -> Path:
    BATCH_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(input_path)
    results = []
    for _, row in df.iterrows():
        features = row.to_dict()
        try:
            scored = score_transaction(features)
            results.append(scored)
        except Exception:
            results.append({"fraud_score": 0, "risk_level": "LOW", "decision": "APPROVED"})
    df["fraud_score"] = [r.get("fraud_score", 0) for r in results]
    df["risk_level"] = [r.get("risk_level", "LOW") for r in results]
    df["decision"] = [r.get("decision", "APPROVED") for r in results]
    if not output_name:
        output_name = f"batch_scored_{int(time.time())}.csv"
    output_path = BATCH_OUTPUT_DIR / output_name
    df.to_csv(output_path, index=False)
    return output_path


def list_batch_results() -> List[dict]:
    BATCH_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    files = []
    for f in BATCH_OUTPUT_DIR.glob("*.csv"):
        files.append(
            {
                "name": f.name,
                "path": str(f),
                "size": f.stat().st_size,
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(f.stat().st_mtime)),
            }
        )
    return sorted(files, key=lambda x: x["created_at"], reverse=True)


# ---------- Executive summary ----------
def get_executive_summary() -> dict:
    cfg = get_runtime_config()
    alerts = get_alerts(limit=100)
    cases = list_cases()
    df = load_or_create_dataset()
    audit = read_audit_log(limit=None)

    total_tx = len(df) if df is not None else 0
    total_fraud = int(df["fraud_label"].sum()) if df is not None and "fraud_label" in df else 0
    total_rejected = len([a for a in audit if a.get("decision") == "REJECTED"])
    fraud_rate = round((total_fraud / total_tx) * 100, 2) if total_tx else 0

    status_counts = {"OPEN": 0, "IN_REVIEW": 0, "RESOLVED": 0}
    for c in cases:
        status_counts[c.get("status", "OPEN")] = status_counts.get(c.get("status", "OPEN"), 0) + 1

    return {
        "total_transactions": total_tx,
        "total_fraud": total_fraud,
        "fraud_rate": fraud_rate,
        "total_rejected": total_rejected,
        "models_loaded": ["random_forest", "logistic_regression", "isolation_forest"],
        "current_threshold": cfg.get("decision_threshold"),
        "current_mode": cfg.get("mode"),
        "rules_enabled": cfg.get("rules_enabled"),
        "recent_alerts_count": alerts.get("total_alerts", 0),
        "cases_open": status_counts.get("OPEN", 0),
        "cases_in_review": status_counts.get("IN_REVIEW", 0),
        "cases_resolved": status_counts.get("RESOLVED", 0),
    }
