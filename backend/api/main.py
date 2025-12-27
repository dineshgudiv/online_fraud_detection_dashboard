"""FastAPI application for the fraud ops backend."""

from __future__ import annotations

import csv
import itertools
import json
import os
import time
import uuid
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from fastapi import Depends, FastAPI, File, HTTPException, Path as FastAPIPath, Query, Request, UploadFile, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from prometheus_client import make_asgi_app
from sqlalchemy import or_, text
from sqlalchemy.orm import Session

from core.audit import log_audit_event
from core.auth import get_current_user, require_roles, role_for_token
from core.config import CORS_ALLOWLIST, RATE_LIMIT_LOGIN, RATE_LIMIT_SCORE
from core.middleware import DemoPublicReadOnlyMiddleware, RequestIDMiddleware, RequestLoggingMiddleware
from core.observability import init_sentry
from core.pagination import paginate_query, pagination_params
from core.rate_limit import RateLimiter
from core.schemas import (
    AlertBulkUpdateIn,
    AlertOut,
    AlertStatus,
    AlertUpdateIn,
    AuditEntryOut,
    AuthTokenOut,
    BulkUpdateOut,
    CaseCreateIn,
    CaseOut,
    CaseStatus,
    CaseUpdateIn,
    ErrorResponse,
    FeedbackIn,
    FeedbackOut,
    HealthOut,
    LoginIn,
    GeoSummaryItem,
    GeoSummaryOut,
    ModelMetric,
    ModelMetricsOut,
    Page,
    PaginationParams,
    PipelineStatusOut,
    ReviewDecision,
    ReviewDecisionIn,
    ReviewDecisionOut,
    ReviewQueueItem,
    RiskLevel,
    Role,
    RoleUpdateIn,
    ScoreResponse,
    TransactionIn,
    UserOut,
)
from core.security import create_access_token, verify_password
from db import models
from db.session import SessionLocal, get_db
from ml.scorer import load_artifact, score_transaction as model_score
from routes.model_lab import router as model_lab_router
from routes.security import router as security_router

import pandas as pd

init_sentry()

START_TIME = time.time()

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
PROFILE_DIR = DATA_DIR / "profiles"
SCORED_DIR = DATA_DIR / "scored"
ACTIVE_DATASET_FILE = UPLOAD_DIR / "active_dataset.json"
MAX_ACTIVE_NAME_LEN = 255
for _dir in (UPLOAD_DIR, PROFILE_DIR, SCORED_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

SCORING_EXECUTOR = ThreadPoolExecutor(max_workers=2)

CHUNK_SIZE = 1024 * 1024  # 1MB
MAX_UPLOAD_BYTES = 500 * 1024 * 1024  # 500MB


def _safe_dataset_path(filename: str) -> Path:
    # Prevent path traversal
    clean = Path(filename).name
    if not clean or len(clean) > MAX_ACTIVE_NAME_LEN:
        raise HTTPException(status_code=400, detail="Invalid filename")

    candidate = (UPLOAD_DIR / clean).resolve()
    root = UPLOAD_DIR.resolve()
    if root not in candidate.parents:
        raise HTTPException(status_code=400, detail="Invalid filename")
    if not candidate.exists() or not candidate.is_file():
        raise HTTPException(status_code=404, detail="Dataset not found")
    return candidate


def _read_json_file(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _write_json_file(path: Path, payload: Any) -> None:
    tmp_path = path.with_suffix(f"{path.suffix}.tmp")
    tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def _dataset_to_dict(entry: models.DatasetVersion) -> dict[str, Any]:
    return {
        "version_id": entry.id,
        "original_filename": entry.original_filename,
        "stored_filename": entry.stored_path,
        "size_bytes": entry.size_bytes,
        "uploaded_at": entry.uploaded_at.isoformat() if entry.uploaded_at else None,
        "schema": entry.schema_json or [],
        "row_count": entry.row_count,
        "is_active": bool(entry.is_active),
    }


def _ensure_dataset_records(db: Session) -> None:
    existing = {row[0] for row in db.query(models.DatasetVersion.stored_path).all()}
    created = False
    for path in UPLOAD_DIR.iterdir():
        if not path.is_file():
            continue
        if path.name == ACTIVE_DATASET_FILE.name:
            continue
        if path.name in existing:
            continue
        entry = models.DatasetVersion(
            original_filename=path.name,
            stored_path=path.name,
            size_bytes=path.stat().st_size,
            uploaded_at=datetime.fromtimestamp(path.stat().st_mtime),
            schema_json=[],
            row_count=None,
            is_active=False,
        )
        db.add(entry)
        created = True
    if created:
        db.commit()


def _resolve_dataset(db: Session, version_ref: str) -> tuple[models.DatasetVersion, Path]:
    entry = (
        db.query(models.DatasetVersion)
        .filter(
            (models.DatasetVersion.id == version_ref)
            | (models.DatasetVersion.stored_path == version_ref)
        )
        .first()
    )
    if entry:
        path = _safe_dataset_path(entry.stored_path)
        return entry, path

    path = _safe_dataset_path(version_ref)
    if path.name == ACTIVE_DATASET_FILE.name:
        raise HTTPException(status_code=404, detail="Dataset not found")

    entry = models.DatasetVersion(
        original_filename=path.name,
        stored_path=path.name,
        size_bytes=path.stat().st_size,
        uploaded_at=datetime.fromtimestamp(path.stat().st_mtime),
        schema_json=[],
        row_count=None,
        is_active=False,
    )
    db.add(entry)
    db.commit()
    db.refresh(entry)
    return entry, path


def _scoring_job_to_dict(entry: models.ScoringJob) -> dict[str, Any]:
    output_format = None
    if entry.output_path:
        output_format = "parquet" if str(entry.output_path).endswith(".parquet") else "csv"
    created_at = entry.started_at.isoformat() if entry.started_at else None
    updated_at = entry.ended_at.isoformat() if entry.ended_at else created_at
    last_updated_at = entry.last_updated_at.isoformat() if entry.last_updated_at else updated_at
    return {
        "job_id": entry.id,
        "dataset_version_id": entry.dataset_version_id,
        "status": entry.status,
        "rows_done": entry.rows_done,
        "rows_total": entry.rows_total,
        "fraud_rows_written": entry.fraud_rows_written,
        "output_path": entry.output_path,
        "output_format": output_format,
        "threshold": entry.threshold,
        "model_version": entry.model_version,
        "created_at": created_at,
        "updated_at": updated_at,
        "last_updated_at": last_updated_at,
        "error": entry.error,
    }


def _get_scoring_job(db: Session, job_id: str) -> Optional[models.ScoringJob]:
    return db.query(models.ScoringJob).filter(models.ScoringJob.id == job_id).first()


def _update_scoring_job(job_id: str, updates: dict[str, Any]) -> None:
    db = SessionLocal()
    try:
        job = db.query(models.ScoringJob).filter(models.ScoringJob.id == job_id).first()
        if not job:
            return
        for key, value in updates.items():
            setattr(job, key, value)
        if "last_updated_at" not in updates:
            job.last_updated_at = datetime.utcnow()
        db.commit()
    finally:
        db.close()


def _normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    return df.where(pd.notnull(df), None)


def _to_jsonable(value: Any) -> Any:
    try:
        return json.loads(json.dumps(value, default=str))
    except Exception:
        return str(value)


def _pick_column(columns: list[str], candidates: list[str]) -> Optional[str]:
    lowered = {col.lower(): col for col in columns}
    for candidate in candidates:
        if candidate in lowered:
            return lowered[candidate]
    return None


def _detect_mapping(columns: list[str]) -> SchemaMappingRequest:
    return SchemaMappingRequest(
        amount_col=_pick_column(
            columns,
            ["amount", "transaction_amount", "transaction_amt", "amt", "value", "purchase_amount", "payment_amount"],
        ),
        timestamp_col=_pick_column(
            columns,
            ["timestamp", "transaction_time", "transaction_timestamp", "created_at", "event_time", "time", "date"],
        ),
        user_id_col=_pick_column(columns, ["user_id", "customer_id", "account_id", "client_id"]),
        merchant_col=_pick_column(columns, ["merchant_id", "merchant", "merchant_name"]),
        device_id_col=_pick_column(columns, ["device_id", "device", "device_fingerprint"]),
        country_col=_pick_column(columns, ["country", "country_code", "merchant_country", "billing_country"]),
        label_col=_pick_column(columns, ["label", "fraud_label", "is_fraud", "fraud", "chargeback"]),
    )


def _mapping_to_dict(mapping: Optional[models.DatasetSchemaMapping]) -> Optional[dict[str, Optional[str]]]:
    if not mapping:
        return None
    return {
        "amount_col": mapping.amount_col,
        "timestamp_col": mapping.timestamp_col,
        "user_id_col": mapping.user_id_col,
        "merchant_col": mapping.merchant_col,
        "device_id_col": mapping.device_id_col,
        "country_col": mapping.country_col,
        "label_col": mapping.label_col,
    }


def _mapping_effective(
    mapping: Optional[models.DatasetSchemaMapping], columns: list[str]
) -> tuple[SchemaMappingRequest, str]:
    if mapping:
        return SchemaMappingRequest(**_mapping_to_dict(mapping)), "custom"
    return _detect_mapping(columns), "auto"


def _get_dataset_columns(path: Path) -> list[str]:
    try:
        df = pd.read_csv(path, nrows=0)
        return list(df.columns)
    except Exception:
        try:
            with path.open("r", newline="", encoding="utf-8", errors="replace") as handle:
                reader = csv.reader(handle)
                return next(reader, []) or []
        except Exception:
            return []


def _score_dataframe(
    df: pd.DataFrame,
    threshold: float,
    dataset_id: str,
    start_row_index: int,
) -> pd.DataFrame:
    rows = len(df)
    scores = [0.05] * rows
    reasons: list[list[str]] = [[] for _ in range(rows)]

    columns = list(df.columns)
    amount_col = _pick_column(
        columns,
        [
            "amount",
            "transaction_amount",
            "transaction_amt",
            "amt",
            "value",
            "purchase_amount",
            "payment_amount",
        ],
    )
    if amount_col:
        amounts = pd.to_numeric(df[amount_col], errors="coerce")
        high_mask = amounts > 1000
        extreme_mask = amounts > 5000
        for idx, flag in enumerate(high_mask.fillna(False).tolist()):
            if flag:
                scores[idx] += 0.15
                reasons[idx].append("high_amount")
        for idx, flag in enumerate(extreme_mask.fillna(False).tolist()):
            if flag:
                scores[idx] += 0.2
                reasons[idx].append("extreme_amount")

    flag_cols = [
        col
        for col in columns
        if any(token in col.lower() for token in ["chargeback", "cbk", "fraud_flag", "is_fraud", "fraud"])
    ]
    for col in flag_cols:
        values = df[col].astype(str).str.lower().str.strip()
        mask = values.isin({"1", "true", "yes", "y", "fraud"})
        for idx, flag in enumerate(mask.tolist()):
            if flag:
                scores[idx] += 0.35
                reasons[idx].append("prior_fraud_flag")

    country_cols = [col for col in columns if "country" in col.lower()]
    high_risk_countries = {"NG", "GH", "PK", "RU", "CN", "BR", "VE"}
    for col in country_cols:
        values = df[col].astype(str).str.upper().str.strip()
        mask = values.isin(high_risk_countries)
        for idx, flag in enumerate(mask.tolist()):
            if flag:
                scores[idx] += 0.2
                reasons[idx].append("high_risk_country")

    if len(country_cols) >= 2:
        first = df[country_cols[0]].astype(str).str.upper().str.strip()
        second = df[country_cols[1]].astype(str).str.upper().str.strip()
        mask = (first.notna() & second.notna()) & (first != second)
        for idx, flag in enumerate(mask.tolist()):
            if flag:
                scores[idx] += 0.15
                reasons[idx].append("geo_mismatch")

    device_cols = [col for col in columns if "device" in col.lower()]
    if device_cols:
        values = df[device_cols[0]].astype(str).str.lower()
        mask = values.str.contains("emulator|root|jailbreak|vpn|proxy", regex=True, na=False)
        for idx, flag in enumerate(mask.tolist()):
            if flag:
                scores[idx] += 0.2
                reasons[idx].append("suspicious_device")

    velocity_col = _pick_column(
        columns,
        ["velocity", "tx_count_24h", "transactions_last_24h", "txn_count", "tx_count"],
    )
    if velocity_col:
        velocity = pd.to_numeric(df[velocity_col], errors="coerce")
        mask = velocity > 5
        for idx, flag in enumerate(mask.fillna(False).tolist()):
            if flag:
                scores[idx] += 0.2
                reasons[idx].append("high_velocity")

    for idx, reason_list in enumerate(reasons):
        if not reason_list:
            reason_list.append("low_signal")
        reasons[idx] = reason_list[:3]
        scores[idx] = min(0.99, max(0.0, scores[idx]))

    tx_id_col = _pick_column(
        columns,
        ["transaction_id", "tx_id", "txn_id", "transactionid", "payment_id"],
    )
    if not tx_id_col:
        tx_id_col = _pick_column(columns, ["id"])

    if tx_id_col:
        tx_ids = df[tx_id_col].astype(str).fillna("").tolist()
    else:
        tx_ids = [f"{dataset_id}-{start_row_index + idx}" for idx in range(rows)]

    df = df.copy()
    df["_risk_score"] = [round(score, 3) for score in scores]
    df["_prediction"] = ["FRAUD" if score >= threshold else "LEGIT" for score in scores]
    df["_reasons"] = reasons
    df["reason_codes"] = reasons
    df["_tx_id"] = tx_ids
    return df


def _estimate_row_count(path: Path) -> Optional[int]:
    try:
        count = 0
        for chunk in pd.read_csv(path, chunksize=50000):
            count += len(chunk)
        return int(count)
    except Exception:
        try:
            with path.open("r", encoding="utf-8", errors="replace") as handle:
                reader = csv.reader(handle)
                next(reader, None)
                return sum(1 for _ in reader)
        except Exception:
            return None


def _compute_profile_sample(
    path: Path,
    mapping: SchemaMappingRequest,
    sample_limit: int = 50000,
) -> dict[str, Any]:
    try:
        df = pd.read_csv(path, nrows=sample_limit)
    except Exception:
        with path.open("r", newline="", encoding="utf-8", errors="replace") as handle:
            reader = csv.DictReader(handle)
            rows = list(itertools.islice(reader, sample_limit))
        df = pd.DataFrame(rows)

    if df.empty:
        return {
            "sample_rows": 0,
            "columns": [],
            "missing_by_column": {},
            "duplicate_estimate_percent": 0.0,
            "invalid_timestamp_count": 0,
            "numeric_stats": {},
        }

    df = df.replace("", pd.NA)
    columns = list(df.columns)
    sample_rows = len(df)
    missing_by_column = ((df.isna().sum() / sample_rows) * 100).round(2).to_dict()
    duplicate_estimate = round(float(df.duplicated().mean() * 100), 2)

    invalid_timestamp_count = 0
    if mapping.timestamp_col and mapping.timestamp_col in df.columns:
        parsed = pd.to_datetime(df[mapping.timestamp_col], errors="coerce", utc=True)
        invalid_timestamp_count = int(parsed.isna().sum())

    amount_candidates = []
    if mapping.amount_col and mapping.amount_col in df.columns:
        amount_candidates.append(mapping.amount_col)
    else:
        amount_candidates = [
            col
            for col in columns
            if any(token in col.lower() for token in ["amount", "amt", "value", "purchase_amount", "payment"])
        ]

    numeric_stats = {}
    for col in amount_candidates:
        values = pd.to_numeric(df[col], errors="coerce").dropna()
        if values.empty:
            continue
        numeric_stats[col] = {
            "min": round(float(values.min()), 4),
            "max": round(float(values.max()), 4),
            "mean": round(float(values.mean()), 4),
            "p95": round(float(values.quantile(0.95)), 4),
        }

    return {
        "sample_rows": sample_rows,
        "columns": columns,
        "missing_by_column": missing_by_column,
        "duplicate_estimate_percent": duplicate_estimate,
        "invalid_timestamp_count": invalid_timestamp_count,
        "numeric_stats": numeric_stats,
    }


def _log_background_audit(
    action: str,
    transaction_id: str,
    decision: str,
    model_name: str = "dataset",
    model_version: str = "n/a",
    score: float = 0.0,
    risk_factors: Optional[list[str]] = None,
) -> None:
    db = SessionLocal()
    try:
        audit = models.AuditLog(
            actor="system",
            action=action,
            transaction_id=transaction_id,
            model_name=model_name,
            model_version=model_version,
            score=score,
            decision=decision,
            risk_factors_json=risk_factors or [],
        )
        db.add(audit)
        db.commit()
    finally:
        db.close()


def _run_scoring_job(
    job_id: str,
    dataset_version_id: str,
    stored_path: str,
    threshold: float,
    model_version: Optional[str],
) -> None:
    path = _safe_dataset_path(stored_path)
    rows_done = 0
    fraud_rows_written = 0
    output_format = "parquet"
    output_path = SCORED_DIR / f"{job_id}.parquet"
    fraud_path = SCORED_DIR / f"{job_id}_fraud.csv"
    fraud_header_written = fraud_path.exists() and fraud_path.stat().st_size > 0

    _update_scoring_job(
        job_id,
        {"status": "running", "rows_done": 0, "fraud_rows_written": 0, "started_at": datetime.utcnow()},
    )

    try:
        use_parquet = True
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError:
            use_parquet = False

        if use_parquet:
            writer = None
            for chunk in pd.read_csv(path, chunksize=5000):
                scored = _score_dataframe(chunk, threshold, dataset_version_id, rows_done)
                table = pa.Table.from_pandas(scored, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(output_path, table.schema)
                writer.write_table(table)
                fraud_chunk = scored[scored["_prediction"] == "FRAUD"]
                if not fraud_chunk.empty:
                    fraud_chunk.to_csv(
                        fraud_path,
                        mode="a",
                        index=False,
                        header=not fraud_header_written,
                    )
                    fraud_header_written = True
                    fraud_rows_written += len(fraud_chunk)
                rows_done += len(scored)
                _update_scoring_job(
                    job_id,
                    {"rows_done": rows_done, "fraud_rows_written": fraud_rows_written},
                )
            if writer:
                writer.close()
        else:
            output_format = "csv"
            output_path = SCORED_DIR / f"{job_id}.csv"
            header_written = False
            for chunk in pd.read_csv(path, chunksize=5000):
                scored = _score_dataframe(chunk, threshold, dataset_version_id, rows_done)
                scored.to_csv(output_path, mode="a", index=False, header=not header_written)
                fraud_chunk = scored[scored["_prediction"] == "FRAUD"]
                if not fraud_chunk.empty:
                    fraud_chunk.to_csv(
                        fraud_path,
                        mode="a",
                        index=False,
                        header=not fraud_header_written,
                    )
                    fraud_header_written = True
                    fraud_rows_written += len(fraud_chunk)
                header_written = True
                rows_done += len(scored)
                _update_scoring_job(
                    job_id,
                    {"rows_done": rows_done, "fraud_rows_written": fraud_rows_written},
                )

        _update_scoring_job(
            job_id,
            {
                "status": "done",
                "rows_done": rows_done,
                "rows_total": rows_done,
                "output_path": str(output_path),
                "ended_at": datetime.utcnow(),
            },
        )

        db = SessionLocal()
        try:
            dataset_entry = (
                db.query(models.DatasetVersion)
                .filter(models.DatasetVersion.id == dataset_version_id)
                .first()
            )
            if dataset_entry and not dataset_entry.row_count:
                dataset_entry.row_count = rows_done
                db.commit()
        finally:
            db.close()

        _log_background_audit(
            action="SCORE_JOB_COMPLETE",
            transaction_id=dataset_version_id,
            decision="done",
            model_name="scoring-job",
            model_version=job_id,
            risk_factors=[output_format],
        )
    except Exception as exc:
        _update_scoring_job(
            job_id,
            {
                "status": "failed",
                "error": str(exc),
                "rows_done": rows_done,
                "rows_total": rows_done,
                "fraud_rows_written": fraud_rows_written,
                "ended_at": datetime.utcnow(),
            },
        )
        _log_background_audit(
            action="SCORE_JOB_FAILED",
            transaction_id=dataset_version_id,
            decision="failed",
            model_name="scoring-job",
            model_version=job_id,
            risk_factors=[str(exc)],
        )


class ActiveDatasetRequest(BaseModel):
    filename: str


class ActiveDatasetVersionRequest(BaseModel):
    version_id: str


class ScoreDatasetRequest(BaseModel):
    threshold: float = 0.5
    model_version: Optional[str] = None


class CasesFromJobRequest(BaseModel):
    tx_ids: list[str]


class SchemaMappingRequest(BaseModel):
    amount_col: Optional[str] = None
    timestamp_col: Optional[str] = None
    user_id_col: Optional[str] = None
    merchant_col: Optional[str] = None
    device_id_col: Optional[str] = None
    country_col: Optional[str] = None
    label_col: Optional[str] = None

login_limiter = RateLimiter(RATE_LIMIT_LOGIN, 60)
score_limiter = RateLimiter(RATE_LIMIT_SCORE, 60)

app = FastAPI(
    title="Fraud Ops API",
    version="0.3.0",
    openapi_tags=[
        {"name": "Health", "description": "Service health and readiness."},
        {"name": "Auth", "description": "Authentication and session."},
        {"name": "Scoring", "description": "Fraud scoring and alert creation."},
        {"name": "Alerts", "description": "Alert queue and triage."},
        {"name": "Cases", "description": "Case management workflow."},
        {"name": "Feedback", "description": "Reviewer feedback loop."},
        {"name": "Audit", "description": "Audit log entries."},
        {"name": "Datasets", "description": "Dataset uploads and previews."},
        {"name": "Geo", "description": "Geo fraud summaries."},
        {"name": "Review", "description": "Manual review queue and decisions."},
        {"name": "Model", "description": "Model metrics and retraining."},
        {"name": "Pipeline", "description": "Pipeline status and health."},
        {"name": "Security", "description": "Security center posture, PKI inventory, and integrations."},
        {"name": "model-lab", "description": "Model lab snapshots and retraining."},
    ],
)

app.include_router(security_router)
app.include_router(model_lab_router)

app.add_middleware(DemoPublicReadOnlyMiddleware)
app.add_middleware(RequestIDMiddleware)
app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOWLIST,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)


def _rate_limit(limiter: RateLimiter):
    def dependency(request: Request):
        host = request.headers.get("X-Forwarded-For") or (request.client.host if request.client else "unknown")
        key = f"{host}:{request.url.path}"
        if not limiter.allow(key):
            raise HTTPException(status_code=429, detail="Rate limit exceeded")

    return dependency


def _request_id(request: Request) -> Optional[str]:
    return getattr(request.state, "request_id", None)


def _risk_level(value: str) -> RiskLevel:
    try:
        return RiskLevel(value)
    except Exception:
        return RiskLevel.LOW


LOCATION_PRIORITY = ["city", "state", "country", "country_code", "region", "location"]


def _pick_geo_field(alerts: list[models.Alert]) -> str:
    available: set[str] = set()
    for alert in alerts:
        features = alert.features_json
        if isinstance(features, dict):
            for key in LOCATION_PRIORITY:
                value = features.get(key)
                if value not in (None, ""):
                    available.add(key)
    for key in LOCATION_PRIORITY:
        if key in available:
            return key
    return "location"


def _geo_label(alert: models.Alert, key: str) -> str:
    features = alert.features_json
    if isinstance(features, dict):
        value = features.get(key)
        if value not in (None, ""):
            return str(value)
    return "Unknown"


def _alert_to_schema(alert: models.Alert) -> AlertOut:
    return AlertOut(
        id=alert.id,
        transaction_id=alert.transaction_id,
        created_at=alert.created_at.isoformat(),
        risk_score=alert.risk_score,
        risk_level=_risk_level(alert.risk_level),
        status=AlertStatus(alert.status),
        merchant_id=alert.merchant_id,
        user_id=alert.user_id,
        amount=alert.amount,
        currency=alert.currency,
        decision=alert.decision,
        reason=alert.reason,
        case_id=alert.case_id,
    )


def _case_to_schema(case: models.Case) -> CaseOut:
    return CaseOut(
        id=case.id,
        created_at=case.created_at.isoformat(),
        updated_at=case.updated_at.isoformat(),
        status=CaseStatus(case.status),
        title=case.title,
        created_by=case.created_by,
        assigned_to=case.assigned_to,
        alert_id=case.alert_id,
        transaction_id=case.transaction_id,
        user_id=case.user_id,
        risk_level=_risk_level(case.risk_level) if case.risk_level else None,
        risk_score=case.risk_score,
        notes=case.notes,
        notes_history=case.notes_history or [],
    )


def _audit_to_schema(entry: models.AuditLog) -> AuditEntryOut:
    return AuditEntryOut(
        id=entry.id,
        timestamp=entry.timestamp.isoformat(),
        actor=entry.actor,
        user_id=entry.user_id,
        role=entry.role,
        action=entry.action,
        resource_type=entry.resource_type,
        resource_id=entry.resource_id,
        ip=entry.ip,
        user_agent=entry.user_agent,
        correlation_id=entry.correlation_id,
        metadata=entry.metadata_json or None,
        transaction_id=entry.transaction_id,
        model_name=entry.model_name,
        model_version=entry.model_version,
        score=entry.score,
        decision=entry.decision,
        risk_factors=entry.risk_factors_json or [],
        alert_id=entry.alert_id,
        case_id=entry.case_id,
    )


def _feedback_to_schema(entry: models.Feedback) -> FeedbackOut:
    return FeedbackOut(
        id=entry.id,
        created_at=entry.created_at.isoformat(),
        alert_id=entry.alert_id,
        case_id=entry.case_id,
        reviewer=entry.reviewer,
        label=entry.label,
        notes=entry.notes,
    )


def _parse_dt(value: Optional[str], field_name: str) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid {field_name} datetime")


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    payload = ErrorResponse(detail=str(exc.detail), code="http_error", request_id=_request_id(request))
    return JSONResponse(status_code=exc.status_code, content=payload.dict())


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    payload = ErrorResponse(detail="Validation failed", code="validation_error", request_id=_request_id(request))
    return JSONResponse(status_code=422, content=payload.dict())


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    payload = ErrorResponse(detail="Internal server error", code="internal_error", request_id=_request_id(request))
    return JSONResponse(status_code=500, content=payload.dict())


@app.get("/", summary="API root", tags=["Health"])
def root() -> dict:
    return {
        "status": "ok",
        "message": "Fraud Ops API. See /docs for the OpenAPI schema.",
    }


@app.get("/health", response_model=HealthOut, summary="Liveness probe", tags=["Health"])
def health() -> HealthOut:
    return HealthOut(status="ok", api_version=app.version, uptime=round(time.time() - START_TIME, 2))


@app.get("/ready", response_model=HealthOut, summary="Readiness probe", tags=["Health"])
def ready(db: Session = Depends(get_db)) -> HealthOut:
    try:
        db.execute(text("SELECT 1"))
    except Exception:
        raise HTTPException(status_code=503, detail="Database unavailable")
    return HealthOut(status="ready", api_version=app.version, uptime=round(time.time() - START_TIME, 2))


@app.post(
    "/auth/login",
    response_model=AuthTokenOut,
    summary="Authenticate",
    tags=["Auth"],
    dependencies=[Depends(_rate_limit(login_limiter))],
)
def login(payload: LoginIn, request: Request, db: Session = Depends(get_db)) -> AuthTokenOut:
    user = db.query(models.User).filter(models.User.email == payload.email).first()
    if not user or not verify_password(payload.password, user.password_hash):
        log_audit_event(
            db,
            request,
            action="LOGIN_FAILED",
            resource_type="auth",
            resource_id=payload.email,
            actor_email=payload.email,
            metadata={"reason": "invalid_credentials"},
        )
        db.commit()
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
    token = create_access_token(subject=user.id, role=role_for_token(user.role))
    log_audit_event(
        db,
        request,
        action="LOGIN_SUCCESS",
        resource_type="auth",
        resource_id=user.id,
        user=user,
        metadata={"email": user.email},
    )
    db.commit()
    return AuthTokenOut(access_token=token)


@app.get("/auth/me", response_model=UserOut, summary="Current user", tags=["Auth"])
def me(current_user: models.User = Depends(get_current_user)) -> UserOut:
    normalized_role = Role(role_for_token(current_user.role))
    return UserOut(id=current_user.id, email=current_user.email, role=normalized_role)


@app.patch(
    "/settings/users/{user_id}/role",
    summary="Update user role",
    tags=["Auth"],
    dependencies=[Depends(require_roles(Role.ADMIN))],
)
def update_user_role(
    user_id: str,
    payload: RoleUpdateIn,
    request: Request,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(require_roles(Role.ADMIN)),
) -> dict:
    user = db.query(models.User).filter(models.User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    old_role = user.role
    user.role = role_for_token(payload.role.value)
    log_audit_event(
        db,
        request,
        action="ROLE_CHANGED",
        resource_type="user",
        resource_id=user.id,
        user=current_user,
        metadata={"old_role": old_role, "new_role": user.role},
    )
    db.commit()
    return {"id": user.id, "role": user.role}


@app.post(
    "/score",
    response_model=ScoreResponse,
    summary="Score a transaction",
    tags=["Scoring"],
    dependencies=[Depends(_rate_limit(score_limiter))],
)
def score_transaction(
    payload: TransactionIn,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(require_roles(Role.ANALYST, Role.ADMIN)),
) -> ScoreResponse:
    features = payload.dict()
    if not features.get("transaction_id"):
        features["transaction_id"] = f"TX-{uuid.uuid4().hex[:12]}"

    result = model_score(features)
    risk_level = _risk_level(result.get("risk_level", "LOW"))
    decision = result.get("decision", "APPROVED")
    model_name = result.get("model_name", "unknown")
    model_version = result.get("model_version", "unknown")

    alert = None
    if risk_level in {RiskLevel.HIGH, RiskLevel.CRITICAL} or decision == "REJECTED":
        alert = models.Alert(
            transaction_id=features["transaction_id"],
            risk_score=float(result.get("fraud_score", 0.0)),
            risk_level=risk_level.value,
            status=AlertStatus.NEW.value,
            merchant_id=features.get("merchant_id"),
            user_id=features.get("user_id"),
            amount=features.get("amount"),
            currency=features.get("currency"),
            decision=decision,
            reason="; ".join(result.get("risk_factors", [])),
            features_json=features,
        )
        db.add(alert)
        db.flush()

    audit = models.AuditLog(
        actor=current_user.email,
        action="SCORE",
        transaction_id=features["transaction_id"],
        model_name=model_name,
        model_version=model_version,
        score=float(result.get("fraud_score", 0.0)),
        decision=decision,
        risk_factors_json=list(result.get("risk_factors", [])),
        alert_id=alert.id if alert else None,
    )
    db.add(audit)
    db.commit()
    if alert:
        db.refresh(alert)

    return ScoreResponse(
        transaction_id=features["transaction_id"],
        decision=decision,
        risk_level=risk_level,
        risk_score=float(result.get("fraud_score", 0.0)),
        model_name=model_name,
        model_version=model_version,
        risk_factors=list(result.get("risk_factors", [])),
        alert_id=alert.id if alert else None,
    )


@app.get(
    "/alerts",
    response_model=Page[AlertOut],
    summary="List alerts",
    tags=["Alerts"],
    dependencies=[Depends(require_roles(Role.VIEWER, Role.ANALYST, Role.ADMIN))],
)
def list_alerts(
    params: PaginationParams = Depends(pagination_params),
    status: Optional[AlertStatus] = Query(None),
    risk_level: Optional[RiskLevel] = Query(None),
    search: Optional[str] = Query(None),
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None),
    sort: str = Query("created_at"),
    order: str = Query("desc"),
    db: Session = Depends(get_db),
) -> Page[AlertOut]:
    query = db.query(models.Alert)

    if status:
        query = query.filter(models.Alert.status == status.value)
    if risk_level:
        query = query.filter(models.Alert.risk_level == risk_level.value)
    if search:
        like = f"%{search}%"
        query = query.filter(
            or_(
                models.Alert.transaction_id.ilike(like),
                models.Alert.user_id.ilike(like),
                models.Alert.merchant_id.ilike(like),
            )
        )

    start_dt = _parse_dt(start_date, "start_date")
    end_dt = _parse_dt(end_date, "end_date")
    if start_dt:
        query = query.filter(models.Alert.created_at >= start_dt)
    if end_dt:
        query = query.filter(models.Alert.created_at <= end_dt)

    sort_map = {
        "created_at": models.Alert.created_at,
        "risk_score": models.Alert.risk_score,
        "risk_level": models.Alert.risk_level,
    }
    sort_col = sort_map.get(sort, models.Alert.created_at)
    if order.lower() == "asc":
        query = query.order_by(sort_col.asc())
    else:
        query = query.order_by(sort_col.desc())

    total, rows = paginate_query(query, params)
    items = [_alert_to_schema(row) for row in rows]
    return Page(page=params.page, page_size=params.page_size, total=total, items=items)


@app.get(
    "/alerts/{alert_id}",
    response_model=AlertOut,
    summary="Get alert details",
    tags=["Alerts"],
    dependencies=[Depends(require_roles(Role.VIEWER, Role.ANALYST, Role.ADMIN))],
)
def get_alert(alert_id: str, db: Session = Depends(get_db)) -> AlertOut:
    alert = db.query(models.Alert).filter(models.Alert.id == alert_id).first()
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")
    return _alert_to_schema(alert)


@app.patch(
    "/alerts/{alert_id}",
    response_model=AlertOut,
    summary="Update an alert",
    tags=["Alerts"],
)
def update_alert(
    alert_id: str,
    payload: AlertUpdateIn,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(require_roles(Role.ANALYST, Role.ADMIN)),
) -> AlertOut:
    alert = db.query(models.Alert).filter(models.Alert.id == alert_id).first()
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")
    alert.status = payload.status.value
    db.commit()
    db.refresh(alert)
    return _alert_to_schema(alert)


@app.post(
    "/alerts/bulk",
    response_model=BulkUpdateOut,
    summary="Bulk update alerts",
    tags=["Alerts"],
)
def bulk_update_alerts(
    payload: AlertBulkUpdateIn,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(require_roles(Role.ANALYST, Role.ADMIN)),
) -> BulkUpdateOut:
    if not payload.alert_ids:
        raise HTTPException(status_code=400, detail="No alert IDs provided")
    alerts = db.query(models.Alert).filter(models.Alert.id.in_(payload.alert_ids)).all()
    for alert in alerts:
        alert.status = payload.status.value
    db.commit()
    return BulkUpdateOut(updated=len(alerts), alert_ids=[a.id for a in alerts])


@app.get(
    "/cases",
    response_model=Page[CaseOut],
    summary="List cases",
    tags=["Cases"],
    dependencies=[Depends(require_roles(Role.VIEWER, Role.ANALYST, Role.ADMIN))],
)
def list_cases(
    params: PaginationParams = Depends(pagination_params),
    status: Optional[CaseStatus] = Query(None),
    search: Optional[str] = Query(None),
    db: Session = Depends(get_db),
) -> Page[CaseOut]:
    query = db.query(models.Case)
    if status:
        query = query.filter(models.Case.status == status.value)
    if search:
        like = f"%{search}%"
        query = query.filter(
            or_(
                models.Case.title.ilike(like),
                models.Case.transaction_id.ilike(like),
                models.Case.user_id.ilike(like),
            )
        )
    query = query.order_by(models.Case.updated_at.desc())

    total, rows = paginate_query(query, params)
    items = [_case_to_schema(row) for row in rows]
    return Page(page=params.page, page_size=params.page_size, total=total, items=items)


@app.get(
    "/cases/{case_id}",
    response_model=CaseOut,
    summary="Get case details",
    tags=["Cases"],
    dependencies=[Depends(require_roles(Role.VIEWER, Role.ANALYST, Role.ADMIN))],
)
def get_case(case_id: str, db: Session = Depends(get_db)) -> CaseOut:
    case = db.query(models.Case).filter(models.Case.id == case_id).first()
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")
    return _case_to_schema(case)


@app.post(
    "/cases",
    response_model=CaseOut,
    summary="Create a case",
    tags=["Cases"],
)
def create_case(
    payload: CaseCreateIn,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(require_roles(Role.ANALYST, Role.ADMIN)),
) -> CaseOut:
    alert = None
    if payload.alert_id:
        alert = db.query(models.Alert).filter(models.Alert.id == payload.alert_id).first()
        if not alert:
            raise HTTPException(status_code=404, detail="Alert not found")

    case = models.Case(
        status=CaseStatus.OPEN.value,
        title=payload.title,
        assigned_to=None,
        alert_id=payload.alert_id,
        transaction_id=payload.transaction_id or (alert.transaction_id if alert else None),
        user_id=payload.user_id or (alert.user_id if alert else None),
        risk_level=payload.risk_level.value if payload.risk_level else (alert.risk_level if alert else None),
        risk_score=payload.risk_score if payload.risk_score is not None else (alert.risk_score if alert else None),
        notes=payload.notes,
        notes_history=[],
        created_by=current_user.email,
    )
    db.add(case)
    db.flush()
    if alert:
        alert.case_id = case.id
        alert.status = AlertStatus.INVESTIGATING.value
    db.commit()
    db.refresh(case)
    return _case_to_schema(case)


@app.patch(
    "/cases/{case_id}",
    response_model=CaseOut,
    summary="Update a case",
    tags=["Cases"],
)
def update_case(
    case_id: str,
    payload: CaseUpdateIn,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(require_roles(Role.ANALYST, Role.ADMIN)),
) -> CaseOut:
    case = db.query(models.Case).filter(models.Case.id == case_id).first()
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")

    if payload.status:
        case.status = payload.status.value
    if payload.assigned_to is not None:
        case.assigned_to = payload.assigned_to
    if payload.notes is not None:
        case.notes = payload.notes
    if payload.note:
        history = list(case.notes_history or [])
        history.append(
            {
                "timestamp": datetime.utcnow().isoformat(),
                "author": payload.note_author or current_user.email,
                "note": payload.note,
            }
        )
        case.notes_history = history

    db.commit()
    db.refresh(case)
    return _case_to_schema(case)


@app.post(
    "/feedback",
    response_model=FeedbackOut,
    summary="Submit reviewer feedback",
    tags=["Feedback"],
)
def create_feedback(
    payload: FeedbackIn,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(require_roles(Role.ANALYST, Role.ADMIN)),
) -> FeedbackOut:
    if not payload.alert_id and not payload.case_id and not payload.tx_id:
        raise HTTPException(status_code=400, detail="alert_id, case_id, or tx_id is required")

    resolve_status = payload.label.lower() in {"fraud", "legit"}

    notes = payload.notes
    if payload.tx_id and not payload.alert_id and not payload.case_id:
        source_label = payload.source or "analyst"
        notes = f"tx_id={payload.tx_id} | source={source_label}" + (f" | {notes}" if notes else "")

    feedback = models.Feedback(
        alert_id=payload.alert_id,
        case_id=payload.case_id,
        reviewer=current_user.email,
        label=payload.label,
        notes=notes,
    )
    db.add(feedback)

    if payload.tx_id:
        label_entry = models.FeedbackLabel(
            tx_id=payload.tx_id,
            job_id=payload.job_id,
            dataset_version_id=payload.dataset_version_id,
            label=payload.label,
            source=payload.source or "analyst",
        )
        db.add(label_entry)

    alert = None
    if payload.alert_id:
        alert = db.query(models.Alert).filter(models.Alert.id == payload.alert_id).first()
        if not alert:
            raise HTTPException(status_code=404, detail="Alert not found")
        alert.status = (AlertStatus.RESOLVED if resolve_status else AlertStatus.INVESTIGATING).value

    case = None
    if payload.case_id:
        case = db.query(models.Case).filter(models.Case.id == payload.case_id).first()
        if not case:
            raise HTTPException(status_code=404, detail="Case not found")
        case.status = (CaseStatus.RESOLVED if resolve_status else CaseStatus.IN_REVIEW).value

    transaction_id = None
    if alert:
        transaction_id = alert.transaction_id
    elif case:
        transaction_id = case.transaction_id
    elif payload.tx_id:
        transaction_id = payload.tx_id

    audit = models.AuditLog(
        actor=current_user.email,
        action="FEEDBACK",
        transaction_id=transaction_id or f"TX-{uuid.uuid4().hex[:8]}",
        model_name=payload.source or "human-review",
        model_version="n/a",
        score=alert.risk_score if alert else (case.risk_score if case else 0.0),
        decision=payload.label,
        risk_factors_json=[notes] if notes else [],
        alert_id=payload.alert_id,
        case_id=payload.case_id,
    )
    db.add(audit)

    db.commit()
    db.refresh(feedback)
    return _feedback_to_schema(feedback)


@app.get(
    "/audit",
    response_model=Page[AuditEntryOut],
    summary="List audit log entries",
    tags=["Audit"],
    dependencies=[Depends(require_roles(Role.VIEWER, Role.ANALYST, Role.ADMIN))],
)
def list_audit(
    params: PaginationParams = Depends(pagination_params),
    transaction_id: Optional[str] = Query(None),
    case_id: Optional[str] = Query(None),
    alert_id: Optional[str] = Query(None),
    actor: Optional[str] = Query(None),
    action: Optional[str] = Query(None),
    resource_type: Optional[str] = Query(None),
    correlation_id: Optional[str] = Query(None),
    entity_type: Optional[str] = Query(None),
    entity_id: Optional[str] = Query(None, alias="id"),
    limit: Optional[int] = Query(None, ge=1, le=200),
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None),
    db: Session = Depends(get_db),
) -> Page[AuditEntryOut]:
    query = db.query(models.AuditLog)
    if transaction_id:
        query = query.filter(models.AuditLog.transaction_id == transaction_id)
    if case_id:
        query = query.filter(models.AuditLog.case_id == case_id)
    if alert_id:
        query = query.filter(models.AuditLog.alert_id == alert_id)
    if actor:
        query = query.filter(models.AuditLog.actor.ilike(f"%{actor}%"))
    if action:
        query = query.filter(models.AuditLog.action == action)
    if resource_type:
        query = query.filter(models.AuditLog.resource_type == resource_type)
    if correlation_id:
        query = query.filter(models.AuditLog.correlation_id == correlation_id)
    if entity_type == "dataset" and entity_id:
        query = query.filter(models.AuditLog.transaction_id == entity_id)

    if limit:
        params = PaginationParams(page=1, page_size=limit)

    start_dt = _parse_dt(start_date, "start_date")
    end_dt = _parse_dt(end_date, "end_date")
    if start_dt:
        query = query.filter(models.AuditLog.timestamp >= start_dt)
    if end_dt:
        query = query.filter(models.AuditLog.timestamp <= end_dt)

    query = query.order_by(models.AuditLog.timestamp.desc())

    total, rows = paginate_query(query, params)
    items = [_audit_to_schema(row) for row in rows]
    return Page(page=params.page, page_size=params.page_size, total=total, items=items)


@app.post("/datasets/upload", summary="Upload a dataset", tags=["Datasets"])
async def datasets_upload(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: models.User = Depends(require_roles(Role.VIEWER, Role.ANALYST, Role.ADMIN)),
) -> dict:
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")

    safe_name = file.filename.replace("/", "_").replace("\\", "_")
    out_path = UPLOAD_DIR / f"{int(time.time())}_{safe_name}"

    total = 0
    upload_complete = False
    try:
        with out_path.open("wb") as out_file:
            while True:
                chunk = await file.read(CHUNK_SIZE)
                if not chunk:
                    break
                total += len(chunk)
                if total > MAX_UPLOAD_BYTES:
                    raise HTTPException(status_code=413, detail="File too large (max 500MB)")
                out_file.write(chunk)

        columns: list[str] = []
        preview_rows: list[dict[str, Any]] = []
        try:
            df = pd.read_csv(out_path, nrows=20)
            columns = list(df.columns)
            preview_rows = _normalize_dataframe(df).to_dict(orient="records")
        except Exception:
            with out_path.open("r", newline="", encoding="utf-8", errors="replace") as handle:
                reader = csv.DictReader(handle)
                columns = reader.fieldnames or []
                preview_rows = list(itertools.islice(reader, 20))

        row_count = _estimate_row_count(out_path)
        is_first = db.query(models.DatasetVersion).count() == 0
        entry = models.DatasetVersion(
            original_filename=file.filename,
            stored_path=out_path.name,
            size_bytes=total,
            uploaded_at=datetime.utcnow(),
            schema_json=columns,
            row_count=row_count,
            is_active=is_first,
        )
        if is_first:
            db.query(models.DatasetVersion).update({models.DatasetVersion.is_active: False})
        db.add(entry)
        db.commit()
        db.refresh(entry)

        audit = models.AuditLog(
            actor=current_user.email,
            action="DATASET_UPLOAD",
            transaction_id=entry.id,
            model_name="dataset",
            model_version="n/a",
            score=0.0,
            decision="uploaded",
            risk_factors_json=[file.filename],
        )
        db.add(audit)
        db.commit()

        upload_complete = True
        return {
            "version_id": entry.id,
            "original_filename": file.filename,
            "stored_filename": out_path.name,
            "filename": out_path.name,
            "bytes": total,
            "row_count": row_count,
            "columns": columns,
            "preview": preview_rows,
        }
    finally:
        await file.close()
        if not upload_complete:
            out_path.unlink(missing_ok=True)


@app.get("/datasets/{dataset_id}/download", summary="Download an uploaded dataset", tags=["Datasets"])
async def download_dataset(
    dataset_id: str,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(require_roles(Role.VIEWER, Role.ANALYST, Role.ADMIN)),
):
    entry, path = _resolve_dataset(db, dataset_id)

    audit = models.AuditLog(
        actor=current_user.email,
        action="DATASET_DOWNLOAD",
        transaction_id=entry.id,
        model_name="dataset",
        model_version="n/a",
        score=0.0,
        decision="download",
        risk_factors_json=[entry.stored_path],
    )
    db.add(audit)
    db.commit()

    return FileResponse(
        path=str(path),
        filename=entry.stored_path or path.name,
        media_type="text/csv",
    )


@app.delete("/datasets/{dataset_id}", summary="Delete an uploaded dataset", tags=["Datasets"])
async def delete_dataset(
    dataset_id: str,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(require_roles(Role.ADMIN)),
) -> dict:
    entry, path = _resolve_dataset(db, dataset_id)
    version_id = entry.id

    path.unlink(missing_ok=True)

    profile_path = PROFILE_DIR / f"{version_id}.json"
    profile_path.unlink(missing_ok=True)

    jobs = db.query(models.ScoringJob).filter(models.ScoringJob.dataset_version_id == version_id).all()
    for job in jobs:
        if job.output_path:
            Path(job.output_path).unlink(missing_ok=True)
        db.delete(job)
    if entry.is_active:
        entry.is_active = False
    db.delete(entry)
    db.commit()

    audit = models.AuditLog(
        actor=current_user.email,
        action="DATASET_DELETE",
        transaction_id=version_id,
        model_name="dataset",
        model_version="n/a",
        score=0.0,
        decision="deleted",
        risk_factors_json=[entry.stored_path],
    )
    db.add(audit)
    db.commit()

    return {"ok": True}


@app.get("/datasets/active", summary="Get the active dataset", tags=["Datasets"])
async def get_active_dataset(
    db: Session = Depends(get_db),
    _current_user: models.User = Depends(require_roles(Role.VIEWER, Role.ANALYST, Role.ADMIN)),
) -> dict:
    entry = db.query(models.DatasetVersion).filter(models.DatasetVersion.is_active.is_(True)).first()
    return {
        "version_id": entry.id if entry else None,
        "filename": entry.stored_path if entry else None,
        "stored_filename": entry.stored_path if entry else None,
        "original_filename": entry.original_filename if entry else None,
    }


@app.post("/datasets/active", summary="Set the active dataset", tags=["Datasets"])
async def set_active_dataset(
    body: ActiveDatasetRequest,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(require_roles(Role.VIEWER, Role.ANALYST, Role.ADMIN)),
) -> dict:
    entry = (
        db.query(models.DatasetVersion)
        .filter(models.DatasetVersion.stored_path == body.filename)
        .first()
    )
    if not entry:
        path = _safe_dataset_path(body.filename)
        entry = (
            db.query(models.DatasetVersion)
            .filter(models.DatasetVersion.stored_path == path.name)
            .first()
        )
    if not entry:
        raise HTTPException(status_code=404, detail="Dataset not found")
    db.query(models.DatasetVersion).update({models.DatasetVersion.is_active: False})
    entry.is_active = True
    db.commit()

    audit = models.AuditLog(
        actor=current_user.email,
        action="DATASET_SET_ACTIVE",
        transaction_id=entry.id,
        model_name="dataset",
        model_version="n/a",
        score=0.0,
        decision="active",
        risk_factors_json=[entry.stored_path],
    )
    db.add(audit)
    db.commit()

    return {"ok": True, "version_id": entry.id, "filename": entry.stored_path}


@app.post("/datasets/set-active", summary="Set the active dataset version", tags=["Datasets"])
async def set_active_dataset_version(
    body: ActiveDatasetVersionRequest,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(require_roles(Role.VIEWER, Role.ANALYST, Role.ADMIN)),
) -> dict:
    entry = db.query(models.DatasetVersion).filter(models.DatasetVersion.id == body.version_id).first()
    if not entry:
        raise HTTPException(status_code=404, detail="Dataset not found")
    db.query(models.DatasetVersion).update({models.DatasetVersion.is_active: False})
    entry.is_active = True
    db.commit()

    audit = models.AuditLog(
        actor=current_user.email,
        action="DATASET_SET_ACTIVE",
        transaction_id=entry.id,
        model_name="dataset",
        model_version="n/a",
        score=0.0,
        decision="active",
        risk_factors_json=[entry.stored_path],
    )
    db.add(audit)
    db.commit()

    return {
        "ok": True,
        "version_id": entry.id,
        "stored_filename": entry.stored_path,
        "original_filename": entry.original_filename,
    }


@app.get("/datasets", summary="List uploaded datasets", tags=["Datasets"])
async def list_datasets(
    db: Session = Depends(get_db),
    _current_user: models.User = Depends(require_roles(Role.VIEWER, Role.ANALYST, Role.ADMIN)),
) -> list[dict]:
    _ensure_dataset_records(db)
    entries = (
        db.query(models.DatasetVersion)
        .order_by(models.DatasetVersion.uploaded_at.desc())
        .all()
    )
    items = []
    for entry in entries:
        if entry.stored_path == ACTIVE_DATASET_FILE.name:
            continue
        items.append(_dataset_to_dict(entry))
    return items


@app.get("/datasets/{dataset_id}/preview", summary="Preview an uploaded dataset", tags=["Datasets"])
async def preview_dataset(
    dataset_id: str = FastAPIPath(..., min_length=1),
    limit: int = Query(50, ge=1, le=200),
    db: Session = Depends(get_db),
    _current_user: models.User = Depends(require_roles(Role.VIEWER, Role.ANALYST, Role.ADMIN)),
) -> dict:
    entry, path = _resolve_dataset(db, dataset_id)

    columns: list[str] = []
    preview_rows: list[dict[str, Any]] = []
    try:
        df = pd.read_csv(path, nrows=limit)
        columns = list(df.columns)
        preview_rows = _normalize_dataframe(df).to_dict(orient="records")
    except Exception:
        with path.open("r", newline="", encoding="utf-8", errors="replace") as handle:
            reader = csv.DictReader(handle)
            columns = reader.fieldnames or []
            preview_rows = list(itertools.islice(reader, limit))

    return {
        "version_id": entry.id,
        "original_filename": entry.original_filename,
        "stored_filename": entry.stored_path,
        "bytes": path.stat().st_size,
        "columns": columns,
        "rows": preview_rows,
        "row_count": entry.row_count,
    }


@app.get("/datasets/{dataset_id}/profile", summary="Profile an uploaded dataset", tags=["Datasets"])
async def profile_dataset(
    dataset_id: str = FastAPIPath(..., min_length=1),
    db: Session = Depends(get_db),
    current_user: models.User = Depends(require_roles(Role.VIEWER, Role.ANALYST, Role.ADMIN)),
) -> dict:
    entry, path = _resolve_dataset(db, dataset_id)
    profile_path = PROFILE_DIR / f"{entry.id}.json"
    mapping = (
        db.query(models.DatasetSchemaMapping)
        .filter(models.DatasetSchemaMapping.dataset_version_id == entry.id)
        .first()
    )
    columns = _get_dataset_columns(path)
    effective_mapping, mapping_source = _mapping_effective(mapping, columns)
    if profile_path.exists():
        profile = _read_json_file(profile_path, {})
    else:
        profile = {}
    if not profile.get("missing_by_column") or not profile.get("columns"):
        profile = _compute_profile_sample(path, effective_mapping)
        profile["generated_at"] = datetime.utcnow().isoformat()
        _write_json_file(profile_path, profile)

    entry.schema_json = profile.get("columns", entry.schema_json or [])
    db.commit()

    audit = models.AuditLog(
        actor=current_user.email,
        action="PROFILE_RUN",
        transaction_id=entry.id,
        model_name="dataset",
        model_version="n/a",
        score=0.0,
        decision="profiled",
        risk_factors_json=[],
    )
    db.add(audit)
    db.commit()

    return {
        "version_id": entry.id,
        "original_filename": entry.original_filename,
        "stored_filename": entry.stored_path,
        "mapping": _mapping_to_dict(mapping),
        "mapping_source": mapping_source,
        **profile,
    }


@app.post("/datasets/{dataset_id}/schema-mapping", summary="Save dataset schema mapping", tags=["Datasets"])
async def save_schema_mapping(
    dataset_id: str = FastAPIPath(..., min_length=1),
    payload: SchemaMappingRequest | None = None,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(require_roles(Role.VIEWER, Role.ANALYST, Role.ADMIN)),
) -> dict:
    entry, path = _resolve_dataset(db, dataset_id)
    mapping = payload or SchemaMappingRequest()
    columns = set(_get_dataset_columns(path))
    invalid = [
        value
        for value in [
            mapping.amount_col,
            mapping.timestamp_col,
            mapping.user_id_col,
            mapping.merchant_col,
            mapping.device_id_col,
            mapping.country_col,
            mapping.label_col,
        ]
        if value and value not in columns
    ]
    if invalid:
        raise HTTPException(status_code=400, detail=f"Unknown columns: {', '.join(invalid)}")

    record = (
        db.query(models.DatasetSchemaMapping)
        .filter(models.DatasetSchemaMapping.dataset_version_id == entry.id)
        .first()
    )
    if not record:
        record = models.DatasetSchemaMapping(dataset_version_id=entry.id)
        db.add(record)

    record.amount_col = mapping.amount_col
    record.timestamp_col = mapping.timestamp_col
    record.user_id_col = mapping.user_id_col
    record.merchant_col = mapping.merchant_col
    record.device_id_col = mapping.device_id_col
    record.country_col = mapping.country_col
    record.label_col = mapping.label_col
    db.commit()

    profile_path = PROFILE_DIR / f"{entry.id}.json"
    profile_path.unlink(missing_ok=True)

    audit = models.AuditLog(
        actor=current_user.email,
        action="DATASET_SCHEMA_MAPPING",
        transaction_id=entry.id,
        model_name="dataset",
        model_version="n/a",
        score=0.0,
        decision="mapped",
        risk_factors_json=[col for col in invalid if col],
    )
    db.add(audit)
    db.commit()

    return {"ok": True, "mapping": _mapping_to_dict(record)}


@app.get("/datasets/{dataset_id}/scored-preview", summary="Preview scored dataset rows", tags=["Datasets"])
async def scored_preview_dataset(
    dataset_id: str = FastAPIPath(..., min_length=1),
    limit: int = Query(200, ge=1, le=500),
    threshold: float = Query(0.5, ge=0.0, le=1.0),
    db: Session = Depends(get_db),
    current_user: models.User = Depends(require_roles(Role.VIEWER, Role.ANALYST, Role.ADMIN)),
) -> dict:
    entry, path = _resolve_dataset(db, dataset_id)
    df = pd.read_csv(path, nrows=limit)
    scored = _score_dataframe(df, threshold, entry.id, 0)
    rows = _normalize_dataframe(scored).to_dict(orient="records")
    total_rows = len(scored)
    fraud_count = int((scored["_prediction"] == "FRAUD").sum()) if total_rows else 0
    legit_count = total_rows - fraud_count
    avg_score = float(scored["_risk_score"].mean()) if total_rows else 0.0
    fraud_rate = (fraud_count / total_rows) if total_rows else 0.0

    audit = models.AuditLog(
        actor=current_user.email,
        action="SCORED_PREVIEW",
        transaction_id=entry.id,
        model_name="dataset",
        model_version="n/a",
        score=threshold,
        decision="preview",
        risk_factors_json=[],
    )
    db.add(audit)
    db.commit()

    return {
        "version_id": entry.id,
        "original_filename": entry.original_filename,
        "stored_filename": entry.stored_path,
        "columns": list(scored.columns),
        "rows": rows,
        "summary": {
            "fraud": fraud_count,
            "legit": legit_count,
            "fraud_rate": round(fraud_rate, 4),
            "avg_score": round(avg_score, 4),
        },
    }


@app.post("/datasets/{dataset_id}/score", summary="Start a scoring job", tags=["Datasets"])
async def start_scoring_job(
    dataset_id: str = FastAPIPath(..., min_length=1),
    payload: Optional[ScoreDatasetRequest] = None,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(require_roles(Role.ANALYST, Role.ADMIN)),
) -> dict:
    entry, _path = _resolve_dataset(db, dataset_id)
    threshold = payload.threshold if payload else 0.5
    model_version = payload.model_version if payload else None
    job = models.ScoringJob(
        dataset_version_id=entry.id,
        status="queued",
        threshold=threshold,
        model_version=model_version,
        rows_total=entry.row_count,
        rows_done=0,
        fraud_rows_written=0,
        output_path=None,
        started_at=datetime.utcnow(),
        ended_at=None,
        last_updated_at=datetime.utcnow(),
        error=None,
    )
    db.add(job)
    db.commit()
    db.refresh(job)

    audit = models.AuditLog(
        actor=current_user.email,
        action="SCORE_JOB_START",
        transaction_id=entry.id,
        model_name="scoring-job",
        model_version=job.id,
        score=threshold,
        decision="queued",
        risk_factors_json=[],
    )
    db.add(audit)
    db.commit()

    SCORING_EXECUTOR.submit(_run_scoring_job, job.id, entry.id, entry.stored_path, threshold, model_version)
    return _scoring_job_to_dict(job)


@app.get("/scoring-jobs", summary="List scoring jobs", tags=["Datasets"])
async def list_scoring_jobs(
    db: Session = Depends(get_db),
    _current_user: models.User = Depends(require_roles(Role.VIEWER, Role.ANALYST, Role.ADMIN)),
) -> list[dict]:
    jobs = db.query(models.ScoringJob).order_by(models.ScoringJob.started_at.desc()).all()
    return [_scoring_job_to_dict(job) for job in jobs]


@app.get("/scoring-jobs/{job_id}", summary="Get scoring job status", tags=["Datasets"])
async def get_scoring_job(
    job_id: str,
    db: Session = Depends(get_db),
    _current_user: models.User = Depends(require_roles(Role.VIEWER, Role.ANALYST, Role.ADMIN)),
) -> dict:
    job = _get_scoring_job(db, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Scoring job not found")
    return _scoring_job_to_dict(job)


@app.get("/scoring-jobs/{job_id}/results", summary="Get scored rows for a job", tags=["Datasets"])
async def get_scoring_results(
    job_id: str,
    prediction: Optional[str] = Query(None),
    limit: int = Query(200, ge=1, le=500),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
    _current_user: models.User = Depends(require_roles(Role.VIEWER, Role.ANALYST, Role.ADMIN)),
) -> dict:
    job = _get_scoring_job(db, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Scoring job not found")
    output_path = job.output_path
    fraud_path = SCORED_DIR / f"{job_id}_fraud.csv"
    is_partial = job.status != "done"
    if is_partial:
        if not fraud_path.exists():
            return {
                "job_id": job_id,
                "offset": offset,
                "limit": limit,
                "has_more": False,
                "rows": [],
                "is_partial": True,
                "rows_available": job.fraud_rows_written,
                "job_status": job.status,
            }
    else:
        if not output_path:
            raise HTTPException(status_code=404, detail="Scored output not available")

    pred_filter = prediction.upper() if prediction else None
    rows: list[dict[str, Any]] = []
    has_more = False
    skipped = 0
    needed = limit + 1

    if is_partial:
        skiprows = range(1, offset + 1) if offset > 0 else None
        try:
            df = pd.read_csv(fraud_path, skiprows=skiprows, nrows=needed)
        except pd.errors.EmptyDataError:
            df = pd.DataFrame()
        if pred_filter:
            df = df[df["_prediction"] == pred_filter]
        rows = _normalize_dataframe(df).to_dict(orient="records") if not df.empty else []
        if len(rows) > limit:
            has_more = True
            rows = rows[:limit]
    else:
        output_format = "parquet" if str(output_path).endswith(".parquet") else "csv"
        if output_format == "parquet":
            import pyarrow.dataset as ds

            dataset = ds.dataset(output_path, format="parquet")
            filter_expr = ds.field("_prediction") == pred_filter if pred_filter else None
            scanner = dataset.scanner(filter=filter_expr, batch_size=1000)
            for batch in scanner.to_batches():
                df = batch.to_pandas()
                if skipped < offset:
                    if skipped + len(df) <= offset:
                        skipped += len(df)
                        continue
                    df = df.iloc[offset - skipped :]
                    skipped = offset
                if len(rows) >= needed:
                    break
                df = df.iloc[: needed - len(rows)]
                rows.extend(_normalize_dataframe(df).to_dict(orient="records"))
                if len(rows) >= needed:
                    break
        else:
            for chunk in pd.read_csv(output_path, chunksize=5000):
                if pred_filter:
                    chunk = chunk[chunk["_prediction"] == pred_filter]
                if skipped < offset:
                    if skipped + len(chunk) <= offset:
                        skipped += len(chunk)
                        continue
                    chunk = chunk.iloc[offset - skipped :]
                    skipped = offset
                if len(rows) >= needed:
                    break
                chunk = chunk.iloc[: needed - len(rows)]
                rows.extend(_normalize_dataframe(chunk).to_dict(orient="records"))
                if len(rows) >= needed:
                    break

    if len(rows) > limit:
        has_more = True
        rows = rows[:limit]

    return {
        "job_id": job_id,
        "offset": offset,
        "limit": limit,
        "has_more": has_more,
        "rows": rows,
        "is_partial": is_partial,
        "rows_available": job.fraud_rows_written if is_partial else job.fraud_rows_written,
        "job_status": job.status,
    }


@app.get("/scoring-jobs/{job_id}/download", summary="Download scored output", tags=["Datasets"])
async def download_scoring_job(
    job_id: str,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(require_roles(Role.VIEWER, Role.ANALYST, Role.ADMIN)),
):
    job = _get_scoring_job(db, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Scoring job not found")
    output_path = job.output_path
    if not output_path:
        raise HTTPException(status_code=404, detail="Scored output not available")

    audit = models.AuditLog(
        actor=current_user.email,
        action="SCORE_JOB_DOWNLOAD",
        transaction_id=job.dataset_version_id,
        model_name="scoring-job",
        model_version=job_id,
        score=0.0,
        decision="download",
        risk_factors_json=[],
    )
    db.add(audit)
    db.commit()

    return FileResponse(
        path=output_path,
        filename=Path(output_path).name,
        media_type="application/octet-stream",
    )


@app.post("/cases/from-job/{job_id}", summary="Create cases from a scoring job", tags=["Cases"])
async def create_cases_from_job(
    job_id: str,
    payload: CasesFromJobRequest,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(require_roles(Role.ANALYST, Role.ADMIN)),
) -> dict:
    job = _get_scoring_job(db, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Scoring job not found")
    output_path = job.output_path
    fraud_path = SCORED_DIR / f"{job_id}_fraud.csv"
    if job.status != "done":
        if not fraud_path.exists():
            raise HTTPException(status_code=409, detail="Fraud rows not available yet")
        output_path = str(fraud_path)
        output_format = "csv"
    else:
        if not output_path:
            raise HTTPException(status_code=404, detail="Scored output not available")
        output_format = "parquet" if str(output_path).endswith(".parquet") else "csv"

    tx_ids = {tx_id for tx_id in payload.tx_ids if tx_id}
    if not tx_ids:
        raise HTTPException(status_code=400, detail="No tx_ids provided")

    selected_rows: list[dict[str, Any]] = []

    if output_format == "parquet":
        import pyarrow.dataset as ds

        dataset = ds.dataset(output_path, format="parquet")
        scanner = dataset.scanner(batch_size=1000)
        for batch in scanner.to_batches():
            df = batch.to_pandas()
            df = df[df["_tx_id"].astype(str).isin(tx_ids)]
            if not df.empty:
                selected_rows.extend(df.to_dict(orient="records"))
            if len(selected_rows) >= len(tx_ids):
                break
    else:
        for chunk in pd.read_csv(output_path, chunksize=5000):
            chunk = chunk[chunk["_tx_id"].astype(str).isin(tx_ids)]
            if not chunk.empty:
                selected_rows.extend(chunk.to_dict(orient="records"))
            if len(selected_rows) >= len(tx_ids):
                break

    if not selected_rows:
        raise HTTPException(status_code=404, detail="No matching rows for provided tx_ids")

    top_risk = max(float(row.get("_risk_score", 0.0)) for row in selected_rows)
    first_tx = str(selected_rows[0].get("_tx_id"))
    case = models.Case(
        status=CaseStatus.OPEN.value,
        title=f"Scoring job {job_id} case",
        assigned_to=None,
        alert_id=None,
        transaction_id=first_tx,
        user_id=selected_rows[0].get("user_id") or selected_rows[0].get("customer_id"),
        risk_level=None,
        risk_score=top_risk,
        notes=f"{len(selected_rows)} items from scoring job {job_id}",
        notes_history=[],
        created_by=current_user.email,
    )
    db.add(case)
    db.flush()

    audit = models.AuditLog(
        actor=current_user.email,
        action="CASE_CREATE",
        transaction_id=first_tx,
        model_name="scoring-job",
        model_version=job_id,
        score=top_risk,
        decision="case_created",
        risk_factors_json=[],
        case_id=case.id,
    )
    db.add(audit)

    for row in selected_rows:
        tx_id = str(row.get("_tx_id"))
        risk_score = float(row.get("_risk_score", 0.0))
        case_item = models.CaseItem(
            case_id=case.id,
            job_id=job_id,
            tx_id=tx_id,
            risk_score=risk_score,
            payload_json=_to_jsonable(row),
        )
        db.add(case_item)

        reason_codes = row.get("reason_codes") or []
        audit = models.AuditLog(
            actor=current_user.email,
            action="CASE_ITEM_ADD",
            transaction_id=tx_id,
            model_name="scoring-job",
            model_version=job_id,
            score=risk_score,
            decision="case_item",
            risk_factors_json=reason_codes if isinstance(reason_codes, list) else [],
            case_id=case.id,
        )
        db.add(audit)

    db.commit()

    return {"created": len(selected_rows), "case_id": case.id}


@app.get(
    "/api/geo/summary",
    response_model=GeoSummaryOut,
    summary="Geo fraud summary",
    tags=["Geo"],
    dependencies=[Depends(require_roles(Role.VIEWER, Role.ANALYST, Role.ADMIN))],
)
def geo_summary(
    limit: int = Query(5000, ge=1, le=20000),
    db: Session = Depends(get_db),
) -> GeoSummaryOut:
    alerts = (
        db.query(models.Alert)
        .order_by(models.Alert.created_at.desc())
        .limit(limit)
        .all()
    )
    if not alerts:
        return GeoSummaryOut(group_by="location", items=[])

    group_by = _pick_geo_field(alerts)
    buckets = defaultdict(lambda: {"count": 0, "high_risk_count": 0, "total_amount": 0.0})
    has_amount = False

    for alert in alerts:
        label = _geo_label(alert, group_by)
        buckets[label]["count"] += 1
        if alert.risk_level in {RiskLevel.HIGH.value, RiskLevel.CRITICAL.value}:
            buckets[label]["high_risk_count"] += 1
        if alert.amount is not None:
            has_amount = True
            buckets[label]["total_amount"] += float(alert.amount)

    items = []
    for label, stats in buckets.items():
        total_amount = stats["total_amount"] if has_amount else None
        items.append(
            GeoSummaryItem(
                label=label,
                count=stats["count"],
                high_risk_count=stats["high_risk_count"],
                total_amount=total_amount,
            )
        )

    items.sort(key=lambda item: item.count, reverse=True)
    return GeoSummaryOut(group_by=group_by, items=items)


@app.get(
    "/api/review/queue",
    response_model=list[ReviewQueueItem],
    summary="Manual review queue",
    tags=["Review"],
    dependencies=[Depends(require_roles(Role.VIEWER, Role.ANALYST, Role.ADMIN))],
)
def review_queue(
    limit: int = Query(25, ge=1, le=200),
    db: Session = Depends(get_db),
) -> list[ReviewQueueItem]:
    cases = (
        db.query(models.Case)
        .filter(models.Case.status.in_([CaseStatus.OPEN.value, CaseStatus.IN_REVIEW.value]))
        .order_by(models.Case.updated_at.desc())
        .limit(limit)
        .all()
    )
    items = [
        ReviewQueueItem(
            id=case.id,
            source="case",
            title=case.title or f"Case {case.id}",
            status=case.status,
            risk_level=_risk_level(case.risk_level) if case.risk_level else None,
            risk_score=case.risk_score,
            transaction_id=case.transaction_id,
            assigned_to=case.assigned_to,
            created_at=case.created_at.isoformat(),
            updated_at=case.updated_at.isoformat(),
        )
        for case in cases
    ]
    if items:
        return items

    alerts = (
        db.query(models.Alert)
        .filter(models.Alert.status != AlertStatus.RESOLVED.value)
        .order_by(models.Alert.created_at.desc())
        .limit(limit)
        .all()
    )
    return [
        ReviewQueueItem(
            id=alert.id,
            source="alert",
            title=f"Review alert {alert.transaction_id}",
            status=alert.status,
            risk_level=_risk_level(alert.risk_level),
            risk_score=alert.risk_score,
            transaction_id=alert.transaction_id,
            assigned_to=None,
            created_at=alert.created_at.isoformat(),
            updated_at=alert.created_at.isoformat(),
        )
        for alert in alerts
    ]


@app.post(
    "/api/review/{case_id}/decision",
    response_model=ReviewDecisionOut,
    summary="Submit a manual review decision",
    tags=["Review"],
)
def review_decision(
    case_id: str,
    payload: ReviewDecisionIn,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(require_roles(Role.ANALYST, Role.ADMIN)),
) -> ReviewDecisionOut:
    decision_label = "APPROVED" if payload.decision == ReviewDecision.APPROVE else "REJECTED"

    case = db.query(models.Case).filter(models.Case.id == case_id).first()
    if case:
        case.status = CaseStatus.RESOLVED.value
        if payload.notes is not None:
            case.notes = payload.notes
        history = list(case.notes_history or [])
        history.append(
            {
                "timestamp": datetime.utcnow().isoformat(),
                "author": current_user.email,
                "note": f"{decision_label}: {payload.notes}" if payload.notes else decision_label,
            }
        )
        case.notes_history = history

        audit = models.AuditLog(
            actor=current_user.email,
            action="REVIEW_DECISION",
            transaction_id=case.transaction_id or f"TX-{uuid.uuid4().hex[:8]}",
            model_name="human-review",
            model_version="n/a",
            score=case.risk_score or 0.0,
            decision=decision_label,
            risk_factors_json=[payload.notes] if payload.notes else [],
            case_id=case.id,
        )
        db.add(audit)
        db.commit()
        return ReviewDecisionOut(
            id=case.id,
            source="case",
            status=case.status,
            message=f"Case resolved as {decision_label}.",
        )

    alert = db.query(models.Alert).filter(models.Alert.id == case_id).first()
    if not alert:
        raise HTTPException(status_code=404, detail="Case or alert not found")

    alert.status = AlertStatus.RESOLVED.value
    alert.decision = decision_label
    if payload.notes:
        alert.reason = payload.notes

    audit = models.AuditLog(
        actor=current_user.email,
        action="REVIEW_DECISION",
        transaction_id=alert.transaction_id,
        model_name="human-review",
        model_version="n/a",
        score=alert.risk_score,
        decision=decision_label,
        risk_factors_json=[payload.notes] if payload.notes else [],
        alert_id=alert.id,
    )
    db.add(audit)
    db.commit()
    return ReviewDecisionOut(
        id=alert.id,
        source="alert",
        status=alert.status,
        message=f"Alert resolved as {decision_label}.",
    )


@app.get(
    "/api/model/metrics",
    response_model=ModelMetricsOut,
    summary="Model performance metrics",
    tags=["Model"],
    dependencies=[Depends(require_roles(Role.VIEWER, Role.ANALYST, Role.ADMIN))],
)
def model_metrics() -> ModelMetricsOut:
    try:
        artifact = load_artifact()
        model_name = artifact.name
        model_version = artifact.version
    except Exception:
        model_name = "unknown"
        model_version = "unknown"

    metrics = [
        ModelMetric(name="auc", available=False, note="Not available"),
        ModelMetric(name="f1", available=False, note="Not available"),
        ModelMetric(name="latency_ms", available=False, note="Not available"),
    ]
    return ModelMetricsOut(model_name=model_name, model_version=model_version, metrics=metrics)


@app.post(
    "/api/model/retrain",
    summary="Trigger model retraining (stub)",
    tags=["Model"],
)
def model_retrain(
    current_user: models.User = Depends(require_roles(Role.ANALYST, Role.ADMIN)),
) -> dict:
    return {
        "status": "not_implemented",
        "message": "Model retraining is not implemented in this build.",
    }


@app.get(
    "/api/pipeline/status",
    response_model=PipelineStatusOut,
    summary="Pipeline status",
    tags=["Pipeline"],
    dependencies=[Depends(require_roles(Role.VIEWER, Role.ANALYST, Role.ADMIN))],
)
def pipeline_status(db: Session = Depends(get_db)) -> PipelineStatusOut:
    prometheus_url = os.getenv("PROMETHEUS_URL")
    try:
        alerts_total = db.query(models.Alert).count()
        high_risk_alerts = (
            db.query(models.Alert)
            .filter(models.Alert.risk_level.in_([RiskLevel.HIGH.value, RiskLevel.CRITICAL.value]))
            .count()
        )
        cases_total = db.query(models.Case).count()
        open_cases = (
            db.query(models.Case)
            .filter(models.Case.status.in_([CaseStatus.OPEN.value, CaseStatus.IN_REVIEW.value]))
            .count()
        )
        latest_alert = (
            db.query(models.Alert)
            .order_by(models.Alert.created_at.desc())
            .first()
        )
        last_ingest_at = latest_alert.created_at.isoformat() if latest_alert else None
        return PipelineStatusOut(
            status="ok",
            source="database",
            last_ingest_at=last_ingest_at,
            alerts_total=alerts_total,
            high_risk_alerts=high_risk_alerts,
            cases_total=cases_total,
            open_cases=open_cases,
            prometheus_enabled=bool(prometheus_url),
            prometheus_url=prometheus_url,
        )
    except Exception:
        return PipelineStatusOut(
            status="degraded",
            source="database",
            last_ingest_at=None,
            alerts_total=0,
            high_risk_alerts=0,
            cases_total=0,
            open_cases=0,
            prometheus_enabled=bool(prometheus_url),
            prometheus_url=prometheus_url,
        )
