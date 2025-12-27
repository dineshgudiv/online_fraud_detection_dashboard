"""API schemas for the fraud ops backend."""

from __future__ import annotations

from enum import Enum
from typing import Generic, List, Optional, TypeVar

from pydantic import BaseModel, Field


T = TypeVar("T")


class ErrorResponse(BaseModel):
    detail: str
    code: str
    request_id: Optional[str] = None


class Page(BaseModel, Generic[T]):
    page: int
    page_size: int
    total: int
    items: List[T]


class HealthOut(BaseModel):
    status: str
    api_version: str
    uptime: float


class RiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class AlertStatus(str, Enum):
    NEW = "NEW"
    TRIAGED = "TRIAGED"
    INVESTIGATING = "INVESTIGATING"
    RESOLVED = "RESOLVED"


class CaseStatus(str, Enum):
    OPEN = "OPEN"
    IN_REVIEW = "IN_REVIEW"
    RESOLVED = "RESOLVED"


class Role(str, Enum):
    ADMIN = "ADMIN"
    ANALYST = "ANALYST"
    READONLY = "READONLY"
    VIEWER = "VIEWER"


class TransactionIn(BaseModel):
    user_id: str
    merchant_id: str
    amount: float
    currency: str
    transaction_type: str
    category: str
    location: str
    device_id: str
    timestamp: Optional[str] = None
    transaction_id: Optional[str] = None


class ScoreResponse(BaseModel):
    transaction_id: str
    decision: str
    risk_level: RiskLevel
    risk_score: float
    model_name: str
    model_version: str
    risk_factors: List[str]
    alert_id: Optional[str] = None


class AlertOut(BaseModel):
    id: str
    transaction_id: str
    created_at: str
    risk_score: float
    risk_level: RiskLevel
    status: AlertStatus
    merchant_id: Optional[str] = None
    user_id: Optional[str] = None
    amount: Optional[float] = None
    currency: Optional[str] = None
    decision: Optional[str] = None
    reason: Optional[str] = None
    case_id: Optional[str] = None


class AlertUpdateIn(BaseModel):
    status: AlertStatus


class AlertBulkUpdateIn(BaseModel):
    alert_ids: List[str]
    status: AlertStatus


class BulkUpdateOut(BaseModel):
    updated: int
    alert_ids: List[str]


class NoteEntry(BaseModel):
    timestamp: str
    author: str
    note: str


class CaseOut(BaseModel):
    id: str
    created_at: str
    updated_at: str
    status: CaseStatus
    title: str
    created_by: Optional[str] = None
    assigned_to: Optional[str] = None
    alert_id: Optional[str] = None
    transaction_id: Optional[str] = None
    user_id: Optional[str] = None
    risk_level: Optional[RiskLevel] = None
    risk_score: Optional[float] = None
    notes: Optional[str] = None
    notes_history: List[NoteEntry] = Field(default_factory=list)


class CaseCreateIn(BaseModel):
    title: str
    alert_id: Optional[str] = None
    transaction_id: Optional[str] = None
    user_id: Optional[str] = None
    risk_level: Optional[RiskLevel] = None
    risk_score: Optional[float] = None
    notes: Optional[str] = None


class CaseUpdateIn(BaseModel):
    status: Optional[CaseStatus] = None
    assigned_to: Optional[str] = None
    notes: Optional[str] = None
    note: Optional[str] = None
    note_author: Optional[str] = None


class FeedbackIn(BaseModel):
    alert_id: Optional[str] = None
    case_id: Optional[str] = None
    label: str
    notes: Optional[str] = None
    tx_id: Optional[str] = None
    source: Optional[str] = None
    job_id: Optional[str] = None
    dataset_version_id: Optional[str] = None


class FeedbackOut(BaseModel):
    id: str
    created_at: str
    alert_id: Optional[str] = None
    case_id: Optional[str] = None
    reviewer: str
    label: str
    notes: Optional[str] = None


class AuditEntryOut(BaseModel):
    id: str
    timestamp: str
    actor: Optional[str] = None
    user_id: Optional[str] = None
    role: Optional[str] = None
    action: str
    resource_type: Optional[str] = None
    resource_id: Optional[str] = None
    ip: Optional[str] = None
    user_agent: Optional[str] = None
    correlation_id: Optional[str] = None
    metadata: Optional[dict] = None
    transaction_id: Optional[str] = None
    model_name: Optional[str] = None
    model_version: Optional[str] = None
    score: Optional[float] = None
    decision: Optional[str] = None
    risk_factors: List[str] = []
    alert_id: Optional[str] = None
    case_id: Optional[str] = None


class PaginationParams(BaseModel):
    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=25, ge=1, le=200)


class LoginIn(BaseModel):
    email: str
    password: str


class RoleUpdateIn(BaseModel):
    role: Role


class AuthTokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserOut(BaseModel):
    id: str
    email: str
    role: Role


class GeoSummaryItem(BaseModel):
    label: str
    count: int
    high_risk_count: int = 0
    total_amount: Optional[float] = None


class GeoSummaryOut(BaseModel):
    group_by: str
    items: List[GeoSummaryItem]


class ReviewDecision(str, Enum):
    APPROVE = "APPROVE"
    REJECT = "REJECT"


class ReviewQueueItem(BaseModel):
    id: str
    source: str
    title: str
    status: str
    risk_level: Optional[RiskLevel] = None
    risk_score: Optional[float] = None
    transaction_id: Optional[str] = None
    assigned_to: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class ReviewDecisionIn(BaseModel):
    decision: ReviewDecision
    notes: Optional[str] = None


class ReviewDecisionOut(BaseModel):
    id: str
    source: str
    status: str
    message: str


class ModelMetric(BaseModel):
    name: str
    value: Optional[float] = None
    unit: Optional[str] = None
    available: bool
    note: Optional[str] = None


class ModelMetricsOut(BaseModel):
    model_name: str
    model_version: str
    metrics: List[ModelMetric]


class RetrainJobOut(BaseModel):
    id: str
    status: str
    requested_by: Optional[str] = None
    requested_at: Optional[str] = None
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    error_message: Optional[str] = None
    model_name: Optional[str] = None
    model_version: Optional[str] = None
    duration_seconds: Optional[float] = None


class DatasetLineageOut(BaseModel):
    id: str
    version: str
    created_at: str
    source: Optional[str] = None


class FeatureSetLineageOut(BaseModel):
    id: str
    version: str
    schema_hash: str
    created_at: str


class ModelVersionLineageOut(BaseModel):
    version: str
    trained_at: str
    metrics: Optional[dict] = None


class RetrainJobLineageOut(BaseModel):
    id: str
    status: str
    requested_by: Optional[str] = None
    requested_at: Optional[str] = None
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    error_message: Optional[str] = None


class LineageOut(BaseModel):
    model: Optional[ModelVersionLineageOut] = None
    dataset: Optional[DatasetLineageOut] = None
    feature_set: Optional[FeatureSetLineageOut] = None
    retrain_job: Optional[RetrainJobLineageOut] = None


class DriftPoint(BaseModel):
    timestamp: str
    overall_score: float


class DriftFeatureOut(BaseModel):
    feature: str
    psi: float
    ks_pvalue: Optional[float] = None


class DriftSummaryOut(BaseModel):
    level: str
    message: str
    overall: List[DriftPoint]
    top_features: List[DriftFeatureOut]


class PipelineStatusOut(BaseModel):
    status: str
    source: str
    last_ingest_at: Optional[str] = None
    alerts_total: int
    high_risk_alerts: int
    cases_total: int
    open_cases: int
    prometheus_enabled: bool = False
    prometheus_url: Optional[str] = None


