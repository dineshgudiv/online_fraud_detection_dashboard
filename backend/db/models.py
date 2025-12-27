"""SQLAlchemy models for fraud ops."""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, Index, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB

from db.base import Base


def _uuid() -> str:
    return str(uuid.uuid4())


def _utcnow() -> datetime:
    return datetime.utcnow()


class User(Base):
    __tablename__ = "users"

    id = Column(String, primary_key=True, default=_uuid)
    email = Column(String, unique=True, nullable=False, index=True)
    password_hash = Column(String, nullable=False)
    role = Column(String, nullable=False, index=True)
    created_at = Column(DateTime(timezone=True), default=_utcnow, nullable=False)
    updated_at = Column(DateTime(timezone=True), default=_utcnow, onupdate=_utcnow, nullable=False)


class Alert(Base):
    __tablename__ = "alerts"

    id = Column(String, primary_key=True, default=_uuid)
    transaction_id = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), default=_utcnow, nullable=False)
    risk_score = Column(Float, nullable=False)
    risk_level = Column(String, nullable=False)
    status = Column(String, nullable=False)
    merchant_id = Column(String, nullable=True)
    user_id = Column(String, nullable=True)
    amount = Column(Float, nullable=True)
    currency = Column(String, nullable=True)
    decision = Column(String, nullable=True)
    reason = Column(Text, nullable=True)
    features_json = Column(JSONB, nullable=True)
    case_id = Column(String, nullable=True)

    __table_args__ = (
        Index("ix_alerts_status", "status"),
        Index("ix_alerts_risk_level", "risk_level"),
        Index("ix_alerts_created_at", "created_at"),
        Index("ix_alerts_transaction_id", "transaction_id"),
    )


class Case(Base):
    __tablename__ = "cases"

    id = Column(String, primary_key=True, default=_uuid)
    created_at = Column(DateTime(timezone=True), default=_utcnow, nullable=False)
    updated_at = Column(DateTime(timezone=True), default=_utcnow, onupdate=_utcnow, nullable=False)
    status = Column(String, nullable=False)
    title = Column(String, nullable=False)
    assigned_to = Column(String, nullable=True)
    alert_id = Column(String, nullable=True)
    transaction_id = Column(String, nullable=True)
    user_id = Column(String, nullable=True)
    risk_level = Column(String, nullable=True)
    risk_score = Column(Float, nullable=True)
    notes = Column(Text, nullable=True)
    notes_history = Column(JSONB, nullable=True)
    created_by = Column(String, nullable=True)

    __table_args__ = (
        Index("ix_cases_status", "status"),
        Index("ix_cases_updated_at", "updated_at"),
    )


class AuditLog(Base):
    __tablename__ = "audit_log"

    id = Column(String, primary_key=True, default=_uuid)
    timestamp = Column(DateTime(timezone=True), default=_utcnow, nullable=False)
    actor = Column(String, nullable=True)
    action = Column(String, nullable=False)
    user_id = Column(String, nullable=True)
    role = Column(String, nullable=True)
    resource_type = Column(String, nullable=True)
    resource_id = Column(String, nullable=True)
    ip = Column(String, nullable=True)
    user_agent = Column(String, nullable=True)
    correlation_id = Column(String, nullable=True)
    metadata_json = Column(JSONB, nullable=True)
    transaction_id = Column(String, nullable=True)
    model_name = Column(String, nullable=True)
    model_version = Column(String, nullable=True)
    score = Column(Float, nullable=True)
    decision = Column(String, nullable=True)
    risk_factors_json = Column(JSONB, nullable=True)
    alert_id = Column(String, nullable=True)
    case_id = Column(String, nullable=True)

    __table_args__ = (
        Index("ix_audit_log_timestamp", "timestamp"),
        Index("ix_audit_log_transaction_id", "transaction_id"),
        Index("ix_audit_log_action", "action"),
        Index("ix_audit_log_resource_type", "resource_type"),
        Index("ix_audit_log_correlation_id", "correlation_id"),
    )


class Feedback(Base):
    __tablename__ = "feedback"

    id = Column(String, primary_key=True, default=_uuid)
    created_at = Column(DateTime(timezone=True), default=_utcnow, nullable=False)
    alert_id = Column(String, nullable=True)
    case_id = Column(String, nullable=True)
    reviewer = Column(String, nullable=False)
    label = Column(String, nullable=False)
    notes = Column(Text, nullable=True)


class CaseItem(Base):
    __tablename__ = "case_items"

    id = Column(String, primary_key=True, default=_uuid)
    case_id = Column(String, ForeignKey("cases.id"), nullable=False, index=True)
    job_id = Column(String, nullable=True, index=True)
    tx_id = Column(String, nullable=False, index=True)
    risk_score = Column(Float, nullable=True)
    payload_json = Column(JSONB, nullable=True)


class FeedbackLabel(Base):
    __tablename__ = "feedback_labels"

    id = Column(String, primary_key=True, default=_uuid)
    tx_id = Column(String, nullable=False, index=True)
    job_id = Column(String, nullable=True, index=True)
    dataset_version_id = Column(String, nullable=True, index=True)
    label = Column(String, nullable=False)
    source = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), default=_utcnow, nullable=False)


class Rule(Base):
    __tablename__ = "rules"

    id = Column(String, primary_key=True, default=_uuid)
    name = Column(String, nullable=False)
    enabled = Column(Integer, default=1, nullable=False)
    config_json = Column(JSONB, nullable=True)
    created_at = Column(DateTime(timezone=True), default=_utcnow, nullable=False)


class DatasetVersion(Base):
    __tablename__ = "dataset_versions"

    id = Column(String, primary_key=True, default=_uuid)
    original_filename = Column(String, nullable=False)
    stored_path = Column(String, nullable=False, index=True)
    size_bytes = Column(Integer, nullable=False)
    uploaded_at = Column(DateTime(timezone=True), default=_utcnow, nullable=False)
    schema_json = Column(JSONB, nullable=True)
    row_count = Column(Integer, nullable=True)
    is_active = Column(Boolean, default=False, nullable=False, index=True)


class DatasetSchemaMapping(Base):
    __tablename__ = "dataset_schema_mappings"

    id = Column(String, primary_key=True, default=_uuid)
    dataset_version_id = Column(String, ForeignKey("dataset_versions.id"), nullable=False, unique=True, index=True)
    amount_col = Column(String, nullable=True)
    timestamp_col = Column(String, nullable=True)
    user_id_col = Column(String, nullable=True)
    merchant_col = Column(String, nullable=True)
    device_id_col = Column(String, nullable=True)
    country_col = Column(String, nullable=True)
    label_col = Column(String, nullable=True)


class ScoringJob(Base):
    __tablename__ = "scoring_jobs"

    id = Column(String, primary_key=True, default=_uuid)
    dataset_version_id = Column(String, ForeignKey("dataset_versions.id"), nullable=False, index=True)
    status = Column(String, nullable=False)
    threshold = Column(Float, nullable=False)
    model_version = Column(String, nullable=True)
    rows_total = Column(Integer, nullable=True)
    rows_done = Column(Integer, nullable=False, default=0)
    fraud_rows_written = Column(Integer, nullable=False, default=0)
    output_path = Column(String, nullable=True)
    started_at = Column(DateTime(timezone=True), nullable=True)
    ended_at = Column(DateTime(timezone=True), nullable=True)
    last_updated_at = Column(DateTime(timezone=True), nullable=True)
    error = Column(Text, nullable=True)

    __table_args__ = (Index("ix_scoring_jobs_status", "status"),)


class RetrainJob(Base):
    __tablename__ = "retrain_jobs"

    id = Column(String, primary_key=True, default=_uuid)
    status = Column(String, nullable=False)
    requested_by = Column(String, nullable=True)
    requested_at = Column(DateTime(timezone=True), default=_utcnow, nullable=False)
    started_at = Column(DateTime(timezone=True), nullable=True)
    finished_at = Column(DateTime(timezone=True), nullable=True)
    error_message = Column(Text, nullable=True)
    model_name = Column(String, nullable=True)
    model_version = Column(String, nullable=True)
    rq_job_id = Column(String, nullable=True)
    dataset_id = Column(String, ForeignKey("datasets.id"), nullable=True)
    feature_set_id = Column(String, ForeignKey("feature_sets.id"), nullable=True)

    __table_args__ = (Index("ix_retrain_jobs_status", "status"),)


class DriftMetric(Base):
    __tablename__ = "drift_metrics"

    id = Column(String, primary_key=True, default=_uuid)
    timestamp = Column(DateTime(timezone=True), default=_utcnow, nullable=False, index=True)
    model_version = Column(String, nullable=True, index=True)
    feature = Column(String, nullable=False, index=True)
    psi = Column(Float, nullable=False)
    ks_pvalue = Column(Float, nullable=True)
    overall_score = Column(Float, nullable=False)

    __table_args__ = (
        Index("ix_drift_metrics_model_version", "model_version"),
        Index("ix_drift_metrics_feature", "feature"),
    )


class Dataset(Base):
    __tablename__ = "datasets"

    id = Column(String, primary_key=True, default=_uuid)
    version = Column(String, nullable=False, index=True)
    created_at = Column(DateTime(timezone=True), default=_utcnow, nullable=False)
    source = Column(String, nullable=True)


class FeatureSet(Base):
    __tablename__ = "feature_sets"

    id = Column(String, primary_key=True, default=_uuid)
    version = Column(String, nullable=False, index=True)
    schema_hash = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), default=_utcnow, nullable=False)


class ModelVersion(Base):
    __tablename__ = "model_versions"

    version = Column(String, primary_key=True)
    trained_at = Column(DateTime(timezone=True), default=_utcnow, nullable=False)
    dataset_id = Column(String, ForeignKey("datasets.id"), nullable=True, index=True)
    feature_set_id = Column(String, ForeignKey("feature_sets.id"), nullable=True, index=True)
    metrics_json = Column(JSONB, nullable=True)

    __table_args__ = (Index("ix_model_versions_trained_at", "trained_at"),)
