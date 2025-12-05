"""Pydantic schemas for FastAPI endpoints."""

from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


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


class TransactionScoreOut(BaseModel):
    decision: str
    risk_level: str
    fraud_score: float
    risk_factors: List[str]
    recommendations: List[str]
    model_breakdown: dict
    transaction_id: Optional[str] = None
    threshold_used: Optional[float] = None
    mode_used: Optional[str] = None
    rule_hits: Optional[List[str]] = None


class SummaryStatsOut(BaseModel):
    total_transactions: int
    total_fraud: int
    fraud_rate_percent: float
    average_fraud_amount: float


class StreamPoint(BaseModel):
    timestamp: str
    events: float
    fraud_events: float


class RealtimeStreamOut(BaseModel):
    processed_events: int
    events_per_second: float
    error_rate: float
    current_events_per_min: float
    fraud_events_per_min: float
    realtime_fraud_rate: float
    points: List[StreamPoint]


class CountryStatsOut(BaseModel):
    country: str
    total_transactions: int
    fraud_transactions: int
    fraud_rate: float
    avg_fraud_score: float
    total_amount: float
    lat: float
    lng: float
    risk_level: str


class FraudRingMember(BaseModel):
    user_id: str
    role: str
    risk_score: float


class FraudRingOut(BaseModel):
    ring_id: str
    risk_level: str
    detection_method: str
    status: str = "active"
    detection_date: Optional[str] = None
    total_amount: float = 0.0
    members: List[FraudRingMember] = Field(default_factory=list)


class FraudRingsResponse(BaseModel):
    total_rings: int
    critical_rings: int
    total_amount: float
    total_members: int
    rings: List[FraudRingOut]


class HealthOut(BaseModel):
    status: str
    api_version: str
    uptime: float


class ModelMetric(BaseModel):
    name: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    auc: Optional[float] = None


class ModelHealthOut(BaseModel):
    overall: ModelMetric
    models: List[ModelMetric]
    confusion_matrix: Dict
    class_distribution: Dict[str, int]


class FeatureDrift(BaseModel):
    feature: str
    drift_score: float
    status: str
    p_value: Optional[float] = None
    comment: Optional[str] = None


class DriftReportOut(BaseModel):
    window_size: int
    reference_period: str
    current_period: str
    features: List[FeatureDrift]


class AuditEntry(BaseModel):
    timestamp: str
    transaction_id: str
    user_id: str
    merchant_id: str
    amount: float
    currency: str
    model_name: str
    fraud_score: float
    risk_level: str
    decision: str
    risk_factors: List[str]


class AuditLogOut(BaseModel):
    total: int
    items: List[AuditEntry]


class WhatIfScenarioIn(BaseModel):
    base: TransactionIn
    variations: List[dict]


class WhatIfResult(BaseModel):
    label: str
    input: dict
    fraud_score: float
    risk_level: str
    decision: str


class WhatIfOut(BaseModel):
    base_result: WhatIfResult
    scenarios: List[WhatIfResult]


class RuntimeConfig(BaseModel):
    decision_threshold: float
    mode: str
    rules_enabled: bool


class RuleCondition(BaseModel):
    amount_gt: Optional[float] = None
    country_in: Optional[List[str]] = None
    hour_in: Optional[List[int]] = None
    merchant_in: Optional[List[str]] = None
    user_in: Optional[List[str]] = None


class RuleAction(BaseModel):
    set_min_score: Optional[float] = None
    bump_score: Optional[float] = None
    force_decision: Optional[str] = None
    set_risk_level: Optional[str] = None


class Rule(BaseModel):
    id: str
    enabled: bool
    description: str
    condition: RuleCondition
    action: RuleAction
    severity: str = "MEDIUM"


class RulesOut(BaseModel):
    items: List[Rule]


class RulesUpdateIn(BaseModel):
    items: List[Rule]


class DatasetInfo(BaseModel):
    name: str
    path: str
    num_rows: int


class DatasetListOut(BaseModel):
    active: str
    available: List[DatasetInfo]


class AlertEntry(BaseModel):
    id: str
    type: str
    user_id: Optional[str] = None
    transaction_id: Optional[str] = None
    risk_level: str
    fraud_score: float
    decision: str
    reason: str
    timestamp: str


class AlertsOut(BaseModel):
    total_alerts: int
    high_risk_count: int
    users_flagged: int
    items: List[AlertEntry]


# Phase 5: Case management
class CaseStatus(str, Enum):
    OPEN = "OPEN"
    IN_REVIEW = "IN_REVIEW"
    RESOLVED = "RESOLVED"


class CaseBase(BaseModel):
    title: str
    user_id: Optional[str] = None
    transaction_id: Optional[str] = None
    risk_level: Optional[str] = None
    fraud_score: Optional[float] = None
    created_from_alert_id: Optional[str] = None
    notes: Optional[str] = None


class Case(CaseBase):
    case_id: str
    status: CaseStatus
    created_at: str
    updated_at: str


class CaseCreateIn(CaseBase):
    pass


class CaseUpdateIn(BaseModel):
    status: Optional[CaseStatus] = None
    notes: Optional[str] = None


class CasesOut(BaseModel):
    items: List[Case]


class CaseOut(BaseModel):
    case: Case

