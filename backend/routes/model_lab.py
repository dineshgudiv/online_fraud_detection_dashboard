"""Model Lab endpoints."""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from redis import Redis
from rq import Queue
from sqlalchemy.orm import Session

from core import config
from core.audit import log_audit_event
from core.auth import require_permission, require_roles
from core.schemas import DriftSummaryOut, LineageOut, RetrainJobOut, Role
from db import models
from db.session import get_db
from tasks.retrain import run_retrain

router = APIRouter(prefix="/ml", tags=["model-lab"])


def _demo_enabled() -> bool:
    return os.getenv("MODEL_DEMO_MODE", "true").lower() in ("1", "true", "yes")


def _phase2_demo_enabled() -> bool:
    return os.getenv("PHASE2_DEMO_MODE", "false").lower() in ("1", "true", "yes")


def _demo_snapshot() -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    return {
        "model_name": "demo-linear",
        "model_version": now.strftime("%Y.%m.%d"),
        "last_trained_at": (now - timedelta(hours=6)).isoformat(),
        "auc": 0.941,
        "f1": 0.812,
        "latency_ms_p95": 38,
        "drift_score": 0.12,
        "data_freshness_min": 8,
        "status": "OK",
    }


def _demo_lineage() -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    model_version = now.strftime("%Y.%m.%d.%H%M")
    return {
        "model": {
            "version": model_version,
            "trained_at": (now - timedelta(hours=6)).isoformat(),
            "metrics": {"auc": 0.94, "f1": 0.82, "latency_ms_p95": 38},
        },
        "dataset": {
            "id": "demo-dataset-001",
            "version": "dataset-v2025.01",
            "created_at": (now - timedelta(days=2)).isoformat(),
            "source": "demo",
        },
        "feature_set": {
            "id": "demo-features-001",
            "version": "features-v3",
            "schema_hash": "schema_demo_hash",
            "created_at": (now - timedelta(days=2)).isoformat(),
        },
        "retrain_job": {
            "id": "retrain-demo-001",
            "status": "SUCCEEDED",
            "requested_by": "demo@fraudops.dev",
            "requested_at": (now - timedelta(hours=6, minutes=10)).isoformat(),
            "started_at": (now - timedelta(hours=6)).isoformat(),
            "finished_at": (now - timedelta(hours=5, minutes=55)).isoformat(),
            "error_message": None,
        },
    }


def _queue() -> Queue:
    conn = Redis.from_url(config.REDIS_URL)
    return Queue(config.RQ_DEFAULT_QUEUE, connection=conn)


def _job_to_dict(job: models.RetrainJob) -> dict[str, Any]:
    duration = None
    if job.started_at and job.finished_at:
        duration = (job.finished_at - job.started_at).total_seconds()
    return {
        "id": job.id,
        "status": job.status,
        "requested_by": job.requested_by,
        "requested_at": job.requested_at.isoformat() if job.requested_at else None,
        "started_at": job.started_at.isoformat() if job.started_at else None,
        "finished_at": job.finished_at.isoformat() if job.finished_at else None,
        "error_message": job.error_message,
        "model_name": job.model_name,
        "model_version": job.model_version,
        "duration_seconds": duration,
    }


def _drift_level(score: float) -> tuple[str, str]:
    if score < 0.1:
        return "GREEN", "Drift within expected bounds"
    if score < 0.2:
        return "YELLOW", "Drift increasing; monitor closely"
    return "RED", "High drift detected; investigate and retrain"


@router.get("/model/snapshot")
def model_snapshot() -> dict[str, Any]:
    if _demo_enabled():
        return _demo_snapshot()
    return {
        "model_name": None,
        "model_version": None,
        "last_trained_at": None,
        "auc": None,
        "f1": None,
        "latency_ms_p95": None,
        "drift_score": None,
        "data_freshness_min": None,
        "status": "NOT_AVAILABLE",
    }


@router.post("/model/retrain")
def trigger_retrain(
    request: Request,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(require_permission("model:retrain")),
) -> dict[str, Any]:
    running = (
        db.query(models.RetrainJob)
        .filter(models.RetrainJob.status.in_(["QUEUED", "RUNNING"]))
        .first()
    )
    if running:
        raise HTTPException(status_code=409, detail="Retrain already in progress")

    job = models.RetrainJob(
        status="QUEUED",
        requested_by=current_user.email,
        model_name=_demo_snapshot().get("model_name"),
        model_version=_demo_snapshot().get("model_version"),
    )
    db.add(job)
    db.commit()
    db.refresh(job)

    queue = _queue()
    rq_job = queue.enqueue(run_retrain, job.id)
    job.rq_job_id = rq_job.id
    db.commit()

    log_audit_event(
        db,
        request,
        action="MODEL_RETRAIN_TRIGGERED",
        resource_type="model",
        resource_id=job.id,
        user=current_user,
        metadata={"job_id": job.id},
    )
    db.commit()
    return {"job_id": job.id, "queued": True, "message": "Retrain job queued"}


@router.get("/model/retrain/status", response_model=RetrainJobOut)
def retrain_status(
    job_id: str,
    db: Session = Depends(get_db),
    _user: models.User = Depends(require_permission("model:retrain")),
) -> dict[str, Any]:
    job = db.query(models.RetrainJob).filter(models.RetrainJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return _job_to_dict(job)


@router.get("/model/retrain/jobs")
def retrain_history(
    limit: int = 25,
    db: Session = Depends(get_db),
    _user: models.User = Depends(require_permission("model:retrain")),
) -> dict[str, Any]:
    limit = min(max(limit, 1), 200)
    rows = (
        db.query(models.RetrainJob)
        .order_by(models.RetrainJob.requested_at.desc())
        .limit(limit)
        .all()
    )
    return {"items": [_job_to_dict(row) for row in rows], "count": len(rows)}


@router.get("/drift/summary", response_model=DriftSummaryOut)
def drift_summary(
    hours: int = 168,
    db: Session = Depends(get_db),
    _user: models.User = Depends(require_roles(Role.READONLY, Role.ANALYST, Role.ADMIN)),
) -> dict[str, Any]:
    hours = min(max(hours, 1), 24 * 30)
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    rows = (
        db.query(models.DriftMetric)
        .filter(models.DriftMetric.timestamp >= cutoff)
        .order_by(models.DriftMetric.timestamp.asc())
        .all()
    )
    if not rows:
        return {"level": "GREEN", "message": "No drift data yet", "overall": [], "top_features": []}

    by_timestamp: dict[str, float] = {}
    for row in rows:
        key = row.timestamp.isoformat()
        by_timestamp[key] = max(by_timestamp.get(key, 0.0), row.overall_score)
    overall = [{"timestamp": ts, "overall_score": score} for ts, score in sorted(by_timestamp.items())]

    latest_ts = rows[-1].timestamp
    latest_rows = [row for row in rows if row.timestamp == latest_ts]
    latest_rows.sort(key=lambda item: item.psi, reverse=True)
    top_features = [
        {
            "feature": row.feature,
            "psi": row.psi,
            "ks_pvalue": row.ks_pvalue,
        }
        for row in latest_rows[:5]
    ]
    level, message = _drift_level(rows[-1].overall_score)
    return {"level": level, "message": message, "overall": overall, "top_features": top_features}


@router.get("/drift/metrics")
def drift_metrics(
    limit: int = 200,
    db: Session = Depends(get_db),
    _user: models.User = Depends(require_roles(Role.READONLY, Role.ANALYST, Role.ADMIN)),
) -> dict[str, Any]:
    limit = min(max(limit, 1), 500)
    rows = (
        db.query(models.DriftMetric)
        .order_by(models.DriftMetric.timestamp.desc())
        .limit(limit)
        .all()
    )
    items = [
        {
            "timestamp": row.timestamp.isoformat(),
            "model_version": row.model_version,
            "feature": row.feature,
            "psi": row.psi,
            "ks_pvalue": row.ks_pvalue,
            "overall_score": row.overall_score,
        }
        for row in rows
    ]
    return {"items": items, "count": len(items)}


@router.get("/lineage", response_model=LineageOut)
def model_lineage(
    db: Session = Depends(get_db),
    _user: models.User = Depends(require_roles(Role.READONLY, Role.ANALYST, Role.ADMIN)),
) -> dict[str, Any]:
    model = (
        db.query(models.ModelVersion)
        .order_by(models.ModelVersion.trained_at.desc())
        .first()
    )
    if not model:
        if _phase2_demo_enabled():
            return _demo_lineage()
        return {"model": None, "dataset": None, "feature_set": None, "retrain_job": None}

    dataset = None
    if model.dataset_id:
        dataset = db.query(models.Dataset).filter(models.Dataset.id == model.dataset_id).first()
    feature_set = None
    if model.feature_set_id:
        feature_set = (
            db.query(models.FeatureSet)
            .filter(models.FeatureSet.id == model.feature_set_id)
            .first()
        )

    retrain_job = (
        db.query(models.RetrainJob)
        .filter(models.RetrainJob.model_version == model.version)
        .order_by(models.RetrainJob.requested_at.desc())
        .first()
    )

    return {
        "model": {
            "version": model.version,
            "trained_at": model.trained_at.isoformat(),
            "metrics": model.metrics_json,
        }
        if model
        else None,
        "dataset": {
            "id": dataset.id,
            "version": dataset.version,
            "created_at": dataset.created_at.isoformat(),
            "source": dataset.source,
        }
        if dataset
        else None,
        "feature_set": {
            "id": feature_set.id,
            "version": feature_set.version,
            "schema_hash": feature_set.schema_hash,
            "created_at": feature_set.created_at.isoformat(),
        }
        if feature_set
        else None,
        "retrain_job": {
            "id": retrain_job.id,
            "status": retrain_job.status,
            "requested_by": retrain_job.requested_by,
            "requested_at": retrain_job.requested_at.isoformat() if retrain_job.requested_at else None,
            "started_at": retrain_job.started_at.isoformat() if retrain_job.started_at else None,
            "finished_at": retrain_job.finished_at.isoformat() if retrain_job.finished_at else None,
            "error_message": retrain_job.error_message,
        }
        if retrain_job
        else None,
    }
