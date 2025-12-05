"""FastAPI application for the fraud detection demo (Phase 5)."""

from __future__ import annotations

import random
import time
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from fastapi import FastAPI, Query, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from backend.core import schemas
from backend.core.config import AUDIT_LOG_FILE
from backend.core import cases as case_store
from backend.ml import engine
from backend.ml.simulation import compute_drift_report, get_realtime_metrics

START_TIME = time.time()

app = FastAPI(title="Online Fraud Detection Enterprise", version="5.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _startup() -> None:
    engine.load_or_create_dataset()
    engine.train_models()


@app.get("/health", response_model=schemas.HealthOut)
def health() -> schemas.HealthOut:
    return schemas.HealthOut(status="ok", api_version=app.version, uptime=round(time.time() - START_TIME, 2))


@app.get("/stats/summary", response_model=schemas.SummaryStatsOut)
def summary_stats() -> schemas.SummaryStatsOut:
    df = engine.load_or_create_dataset()
    total_transactions = int(len(df))
    total_fraud = int(df["fraud_label"].sum())
    fraud_rate_percent = round((total_fraud / total_transactions) * 100, 2) if total_transactions else 0.0
    fraud_amounts = df[df["fraud_label"] == 1]["amount"]
    avg_fraud_amount = round(float(fraud_amounts.mean() if not fraud_amounts.empty else 0.0), 2)

    return schemas.SummaryStatsOut(
        total_transactions=total_transactions,
        total_fraud=total_fraud,
        fraud_rate_percent=fraud_rate_percent,
        average_fraud_amount=avg_fraud_amount,
    )


@app.post("/score_transaction", response_model=schemas.TransactionScoreOut)
def score_transaction(payload: schemas.TransactionIn) -> schemas.TransactionScoreOut:
    features = payload.dict()
    features.setdefault("timestamp", None)
    result = engine.score_transaction(features)
    return schemas.TransactionScoreOut(**result)


@app.get("/stream/metrics", response_model=schemas.RealtimeStreamOut)
def stream_metrics() -> schemas.RealtimeStreamOut:
    metrics = get_realtime_metrics()
    points = [schemas.StreamPoint(**p) for p in metrics["points"]]
    return schemas.RealtimeStreamOut(
        processed_events=metrics["processed_events"],
        events_per_second=metrics["events_per_second"],
        error_rate=metrics["error_rate"],
        current_events_per_min=metrics["current_events_per_min"],
        fraud_events_per_min=metrics["fraud_events_per_min"],
        realtime_fraud_rate=metrics["realtime_fraud_rate"],
        points=points,
    )


COUNTRY_COORDS = {
    "US": (37.0902, -95.7129),
    "GB": (55.3781, -3.4360),
    "DE": (51.1657, 10.4515),
    "FR": (46.2276, 2.2137),
    "IN": (20.5937, 78.9629),
    "CN": (35.8617, 104.1954),
    "SG": (1.3521, 103.8198),
    "BR": (-14.235, -51.9253),
    "ZA": (-30.5595, 22.9375),
    "AU": (-25.2744, 133.7751),
    "CA": (56.1304, -106.3468),
    "JP": (36.2048, 138.2529),
}


def _risk_level_from_rate(rate: float) -> str:
    if rate > 0.35:
        return "CRITICAL"
    if rate > 0.2:
        return "HIGH"
    if rate > 0.1:
        return "MEDIUM"
    return "LOW"


@app.get("/network/countries", response_model=List[schemas.CountryStatsOut])
def network_countries() -> List[schemas.CountryStatsOut]:
    df = engine.load_or_create_dataset()
    grouped = df.groupby("location")
    results: List[schemas.CountryStatsOut] = []
    for country, group in grouped:
        total = int(len(group))
        fraud_tx = int(group["fraud_label"].sum())
        fraud_rate = round(fraud_tx / total, 3) if total else 0.0
        avg_score = round(min(1.0, fraud_rate + random.uniform(0.05, 0.25)), 3)
        total_amount = round(float(group["amount"].sum()), 2)
        lat, lng = COUNTRY_COORDS.get(country, (0.0, 0.0))
        results.append(
            schemas.CountryStatsOut(
                country=country,
                total_transactions=total,
                fraud_transactions=fraud_tx,
                fraud_rate=fraud_rate,
                avg_fraud_score=avg_score,
                total_amount=total_amount,
                lat=lat,
                lng=lng,
                risk_level=_risk_level_from_rate(fraud_rate),
            )
        )
    return results


def _build_fraud_rings() -> schemas.FraudRingsResponse:
    df = engine.load_or_create_dataset()
    risky = df[df["amount"] > df["amount"].quantile(0.8)]
    risky_users = risky["user_id"].unique().tolist()
    if len(risky_users) < 6:
        risky_users.extend([f"U{900+i}" for i in range(6)])
    random.shuffle(risky_users)

    rings: List[schemas.FraudRingOut] = []
    for idx in range(3):
        members = []
        member_ids = risky_users[idx * 4 : (idx + 1) * 4]
        for uid in member_ids:
            members.append(
                schemas.FraudRingMember(
                    user_id=uid,
                    role=random.choice(["mastermind", "mule", "account_taker", "device_spoofer"]),
                    risk_score=round(random.uniform(0.55, 0.98), 2),
                )
            )
        rings.append(
            schemas.FraudRingOut(
                ring_id=f"FR-{100+idx}",
                detection_method=random.choice(["graph_anomaly", "device_correlation", "velocity"]),
                risk_level=random.choice(["HIGH", "CRITICAL"]),
                status=random.choice(["active", "under_review"]),
                detection_date=pd.Timestamp.now().strftime("%Y-%m-%d"),
                total_amount=round(float(np.random.uniform(12000, 52000)), 2),
                members=members,
            )
        )

    total_amount = round(sum(r.total_amount for r in rings), 2)
    total_members = sum(len(r.members) for r in rings)
    critical_rings = sum(1 for r in rings if r.risk_level == "CRITICAL")

    return schemas.FraudRingsResponse(
        total_rings=len(rings),
        critical_rings=critical_rings,
        total_amount=total_amount,
        total_members=total_members,
        rings=rings,
    )


@app.get("/network/rings", response_model=schemas.FraudRingsResponse)
def network_rings() -> schemas.FraudRingsResponse:
    return _build_fraud_rings()


@app.get("/metrics/model_health", response_model=schemas.ModelHealthOut)
def model_health() -> schemas.ModelHealthOut:
    mh = engine.get_model_health()
    overall = mh.get("overall", {})
    models = mh.get("models", [])
    confusion_matrix = mh.get("confusion_matrix", {})
    class_distribution = mh.get("class_distribution", {})
    overall_model = schemas.ModelMetric(**overall) if overall else schemas.ModelMetric(
        name="ensemble", accuracy=0, precision=0, recall=0, f1=0, auc=None
    )
    model_list = [schemas.ModelMetric(**m) for m in models] if models else []
    return schemas.ModelHealthOut(
        overall=overall_model,
        models=model_list,
        confusion_matrix=confusion_matrix,
        class_distribution=class_distribution,
    )


@app.get("/metrics/drift", response_model=schemas.DriftReportOut)
def drift_report() -> schemas.DriftReportOut:
    report = compute_drift_report()
    features = [schemas.FeatureDrift(**f) for f in report.get("features", [])]
    return schemas.DriftReportOut(
        window_size=report.get("window_size", 0),
        reference_period=report.get("reference_period", "Historical"),
        current_period=report.get("current_period", "Current"),
        features=features,
    )


@app.get("/audit/logs", response_model=schemas.AuditLogOut)
def audit_logs(limit: int = Query(default=100, ge=1, le=2000)) -> schemas.AuditLogOut:
    entries = engine.read_audit_log(limit=limit)
    audit_items = [schemas.AuditEntry(**e) for e in entries]
    return schemas.AuditLogOut(total=len(entries), items=audit_items)


@app.get("/audit/export")
def audit_export():
    if not AUDIT_LOG_FILE.exists():
        AUDIT_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        AUDIT_LOG_FILE.touch()
    return FileResponse(AUDIT_LOG_FILE, media_type="text/plain", filename="audit_log.jsonl")


@app.post("/simulate/what_if", response_model=schemas.WhatIfOut)
def what_if_simulator(payload: schemas.WhatIfScenarioIn) -> schemas.WhatIfOut:
    base_features = payload.base.dict()
    base_result = engine.score_transaction(base_features)
    scenarios = []
    for idx, override in enumerate(payload.variations):
        variation_input = {**base_features, **override}
        result = engine.score_transaction(variation_input)
        scenarios.append(
            schemas.WhatIfResult(
                label=f"Scenario {idx+1}",
                input=variation_input,
                fraud_score=result["fraud_score"],
                risk_level=result["risk_level"],
                decision=result["decision"],
            )
        )
    base_out = schemas.WhatIfResult(
        label="Base",
        input=base_features,
        fraud_score=base_result["fraud_score"],
        risk_level=base_result["risk_level"],
        decision=base_result["decision"],
    )
    return schemas.WhatIfOut(base_result=base_out, scenarios=scenarios)


@app.get("/config/runtime", response_model=schemas.RuntimeConfig)
def get_config() -> schemas.RuntimeConfig:
    cfg = engine.get_runtime_config()
    return schemas.RuntimeConfig(**cfg)


@app.post("/config/runtime", response_model=schemas.RuntimeConfig)
def set_config(cfg: schemas.RuntimeConfig) -> schemas.RuntimeConfig:
    updated = engine.update_runtime_config(
        decision_threshold=cfg.decision_threshold,
        mode=cfg.mode,
        rules_enabled=cfg.rules_enabled,
    )
    return schemas.RuntimeConfig(**updated)


@app.get("/rules", response_model=schemas.RulesOut)
def get_rules() -> schemas.RulesOut:
    rules = engine.load_rules()
    items = [schemas.Rule(**r) for r in rules]
    return schemas.RulesOut(items=items)


@app.post("/rules", response_model=schemas.RulesOut)
def update_rules(payload: schemas.RulesUpdateIn) -> schemas.RulesOut:
    rules_dicts = [r.dict() for r in payload.items]
    engine.save_rules(rules_dicts)
    return schemas.RulesOut(items=payload.items)


@app.get("/datasets", response_model=schemas.DatasetListOut)
def datasets() -> schemas.DatasetListOut:
    info = engine.list_datasets()
    available = [schemas.DatasetInfo(**d) for d in info["available"]]
    return schemas.DatasetListOut(active=info["active"], available=available)


@app.post("/datasets/select", response_model=schemas.DatasetListOut)
def select_dataset(payload: dict) -> schemas.DatasetListOut:
    name = payload.get("name", "default")
    info = engine.set_active_dataset(name)
    available = [schemas.DatasetInfo(**d) for d in info["available"]]
    return schemas.DatasetListOut(active=info["active"], available=available)


@app.post("/datasets/upload", response_model=schemas.DatasetInfo)
async def upload_dataset(file: UploadFile = File(...)) -> schemas.DatasetInfo:
    data = await file.read()
    info = engine.upload_dataset(data, file.filename)
    return schemas.DatasetInfo(**info)


@app.get("/datasets/sample")
def dataset_sample(limit: int = 200):
    df = engine.load_or_create_dataset()
    sample = df.head(limit)
    return sample.to_dict(orient="records")


@app.get("/alerts", response_model=schemas.AlertsOut)
def alerts() -> schemas.AlertsOut:
    data = engine.get_alerts(limit=200)
    items = [schemas.AlertEntry(**i) for i in data.get("items", [])]
    return schemas.AlertsOut(
        total_alerts=data.get("total_alerts", 0),
        high_risk_count=data.get("high_risk_count", 0),
        users_flagged=data.get("users_flagged", 0),
        items=items,
    )


# ---------- Cases ----------
@app.get("/cases", response_model=schemas.CasesOut)
def cases_list() -> schemas.CasesOut:
    items = [schemas.Case(**c) for c in case_store.list_cases()]
    return schemas.CasesOut(items=items)


@app.get("/cases/{case_id}", response_model=schemas.CaseOut)
def get_case(case_id: str) -> schemas.CaseOut:
    c = case_store.get_case(case_id)
    if not c:
        raise HTTPException(status_code=404, detail="Case not found")
    return schemas.CaseOut(case=schemas.Case(**c))


@app.post("/cases", response_model=schemas.CaseOut)
def create_case(payload: schemas.CaseCreateIn) -> schemas.CaseOut:
    created = case_store.create_case(payload.dict())
    return schemas.CaseOut(case=schemas.Case(**created))


@app.post("/cases/from_alert/{alert_id}", response_model=schemas.CaseOut)
def create_case_from_alert(alert_id: str) -> schemas.CaseOut:
    alerts_data = engine.get_alerts(limit=500).get("items", [])
    alert = next((a for a in alerts_data if a.get("id") == alert_id), None)
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")
    created = case_store.create_case_from_alert(alert)
    return schemas.CaseOut(case=schemas.Case(**created))


@app.patch("/cases/{case_id}", response_model=schemas.CaseOut)
def update_case(case_id: str, payload: schemas.CaseUpdateIn) -> schemas.CaseOut:
    updates = payload.dict(exclude_unset=True)
    updated = case_store.update_case(case_id, updates)
    if not updated:
        raise HTTPException(status_code=404, detail="Case not found")
    return schemas.CaseOut(case=schemas.Case(**updated))


# ---------- Batch scoring ----------
@app.post("/batch/score")
async def batch_score(file: UploadFile = File(...)):
    data = await file.read()
    BATCH_INPUT_DIR = engine.BASE_DIR / "data" / "batch_input"
    BATCH_INPUT_DIR.mkdir(parents=True, exist_ok=True)
    input_path = BATCH_INPUT_DIR / file.filename
    with input_path.open("wb") as f:
        f.write(data)
    output_path = engine.score_csv_file(input_path)
    return {"output_name": output_path.name, "download_path": f"/batch/results/{output_path.name}"}


@app.get("/batch/results")
def batch_results():
    return engine.list_batch_results()


@app.get("/batch/results/{filename}")
def batch_result_file(filename: str):
    path = engine.BATCH_OUTPUT_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path, media_type="text/csv", filename=path.name)


# ---------- Executive report ----------
@app.get("/report/summary")
def report_summary():
    return engine.get_executive_summary()


@app.get("/")
def root() -> dict:
    return {"message": "Online Fraud Detection Enterprise API"}

