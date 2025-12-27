"""Retrain job execution for the async pipeline."""

from __future__ import annotations

import time
from datetime import datetime

from db import models
from db.session import SessionLocal


def _utcnow() -> datetime:
    return datetime.utcnow()


def _ensure_dataset(db: SessionLocal) -> models.Dataset:
    dataset = db.query(models.Dataset).order_by(models.Dataset.created_at.desc()).first()
    if dataset:
        return dataset
    dataset = models.Dataset(
        version="demo-dataset-001",
        source="demo",
        created_at=_utcnow(),
    )
    db.add(dataset)
    db.commit()
    db.refresh(dataset)
    return dataset


def _ensure_feature_set(db: SessionLocal) -> models.FeatureSet:
    feature_set = db.query(models.FeatureSet).order_by(models.FeatureSet.created_at.desc()).first()
    if feature_set:
        return feature_set
    feature_set = models.FeatureSet(
        version="features-001",
        schema_hash="demo_schema_hash",
        created_at=_utcnow(),
    )
    db.add(feature_set)
    db.commit()
    db.refresh(feature_set)
    return feature_set


def run_retrain(job_id: str) -> None:
    db = SessionLocal()
    try:
        job = db.query(models.RetrainJob).filter(models.RetrainJob.id == job_id).first()
        if not job:
            return
        job.status = "RUNNING"
        job.started_at = _utcnow()
        dataset = _ensure_dataset(db)
        feature_set = _ensure_feature_set(db)
        job.dataset_id = dataset.id
        job.feature_set_id = feature_set.id
        db.commit()

        # Simulate retraining workload; replace with real training pipeline.
        time.sleep(3)

        model_version = _utcnow().strftime("%Y.%m.%d.%H%M")
        model_entry = models.ModelVersion(
            version=model_version,
            trained_at=_utcnow(),
            dataset_id=dataset.id,
            feature_set_id=feature_set.id,
            metrics_json={"auc": 0.94, "f1": 0.82, "latency_ms_p95": 38},
        )
        db.add(model_entry)
        job.model_version = model_version
        job.status = "SUCCEEDED"
        job.finished_at = _utcnow()
        db.commit()
    except Exception as exc:
        job = db.query(models.RetrainJob).filter(models.RetrainJob.id == job_id).first()
        if job:
            job.status = "FAILED"
            job.error_message = str(exc)
            job.finished_at = _utcnow()
            db.commit()
    finally:
        db.close()
