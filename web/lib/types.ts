export type RiskLevel = "LOW" | "MEDIUM" | "HIGH" | "CRITICAL";
export type AlertStatus = "NEW" | "TRIAGED" | "INVESTIGATING" | "RESOLVED";
export type CaseStatus = "OPEN" | "IN_REVIEW" | "RESOLVED";

export const RISK_LEVELS: RiskLevel[] = ["LOW", "MEDIUM", "HIGH", "CRITICAL"];
export const ALERT_STATUSES: AlertStatus[] = [
  "NEW",
  "TRIAGED",
  "INVESTIGATING",
  "RESOLVED"
];
export const CASE_STATUSES: CaseStatus[] = ["OPEN", "IN_REVIEW", "RESOLVED"];

export interface Page<T> {
  page: number;
  page_size: number;
  total: number;
  items: T[];
}

export interface Alert {
  id: string;
  transaction_id: string;
  created_at: string;
  risk_score: number;
  risk_level: RiskLevel;
  status: AlertStatus;
  merchant_id?: string | null;
  user_id?: string | null;
  amount?: number | null;
  currency?: string | null;
  decision?: string | null;
  reason?: string | null;
  case_id?: string | null;
}

export interface CaseNote {
  timestamp: string;
  author: string;
  note: string;
}

export interface CaseRecord {
  id: string;
  created_at: string;
  updated_at: string;
  status: CaseStatus;
  title: string;
  assigned_to?: string | null;
  alert_id?: string | null;
  transaction_id?: string | null;
  user_id?: string | null;
  risk_level?: RiskLevel | null;
  risk_score?: number | null;
  notes?: string | null;
  notes_history: CaseNote[];
}

export interface AuditEntry {
  id: string;
  timestamp: string;
  actor?: string | null;
  user_id?: string | null;
  role?: string | null;
  action: string;
  resource_type?: string | null;
  resource_id?: string | null;
  ip?: string | null;
  user_agent?: string | null;
  correlation_id?: string | null;
  metadata?: Record<string, unknown> | null;
  transaction_id?: string | null;
  model_name?: string | null;
  model_version?: string | null;
  score?: number | null;
  decision?: string | null;
  risk_factors: string[];
  alert_id?: string | null;
  case_id?: string | null;
}

export interface ScoreResponse {
  transaction_id: string;
  decision: string;
  risk_level: RiskLevel;
  risk_score: number;
  model_name: string;
  model_version: string;
  risk_factors: string[];
  alert_id?: string | null;
}

export interface FeedbackResponse {
  id: string;
  created_at: string;
  alert_id?: string | null;
  case_id?: string | null;
  reviewer: string;
  label: string;
  notes?: string | null;
}

export interface GeoSummaryItem {
  label: string;
  count: number;
  high_risk_count: number;
  total_amount?: number | null;
}

export interface GeoSummary {
  group_by: string;
  items: GeoSummaryItem[];
}

export interface ReviewQueueItem {
  id: string;
  source: "case" | "alert";
  title: string;
  status: string;
  risk_level?: RiskLevel | null;
  risk_score?: number | null;
  transaction_id?: string | null;
  assigned_to?: string | null;
  created_at?: string | null;
  updated_at?: string | null;
}

export type ReviewDecision = "APPROVE" | "REJECT";

export interface ReviewDecisionResponse {
  id: string;
  source: string;
  status: string;
  message: string;
}

export interface ModelMetric {
  name: string;
  value?: number | null;
  unit?: string | null;
  available: boolean;
  note?: string | null;
}

export interface ModelMetrics {
  model_name: string;
  model_version: string;
  metrics: ModelMetric[];
}

export interface PipelineStatus {
  status: string;
  source: string;
  last_ingest_at?: string | null;
  alerts_total: number;
  high_risk_alerts: number;
  cases_total: number;
  open_cases: number;
  prometheus_enabled?: boolean;
  prometheus_url?: string | null;
}

