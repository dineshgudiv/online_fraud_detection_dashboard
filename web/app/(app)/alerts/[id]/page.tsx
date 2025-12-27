"use client";

import { useEffect, useMemo, useState } from "react";

import PageHeader from "@/app/_components/PageHeader";
import StatCard from "@/app/_components/StatCard";
import Tabs from "@/app/_components/Tabs";
import DataTable from "@/app/_components/DataTable";
import { useAuth } from "@/app/_components/AuthProvider";
import { apiFetch } from "@/lib/api";
import { DEMO_PUBLIC_READONLY, DEMO_READONLY_MESSAGE } from "@/lib/demo";
import { formatDateUTC } from "@/lib/format";
import { Alert } from "@/lib/types";

type AlertDetail = Alert & {
  model_version?: string | null;
  device_id?: string | null;
  ip_address?: string | null;
  location?: string | null;
  channel?: string | null;
};

type EvidenceSignal = {
  label: string;
  severity: "HIGH" | "MEDIUM" | "LOW";
  detail: string;
};

type TimelineEvent = {
  title: string;
  actor: string;
  timestamp: string;
  note?: string;
};

const tabs = [
  { label: "Overview", value: "overview" },
  { label: "Evidence", value: "evidence" },
  { label: "Timeline", value: "timeline" },
  { label: "Related", value: "related" }
];

const formatDateTime = (value?: string | null) => formatDateUTC(value);

const formatMoney = (amount?: number | null, currency?: string | null) => {
  if (amount === null || amount === undefined) {
    return "n/a";
  }
  const safeCurrency = currency || "USD";
  try {
    return new Intl.NumberFormat("en-US", {
      style: "currency",
      currency: safeCurrency
    }).format(amount);
  } catch {
    return `${amount.toFixed(2)} ${safeCurrency}`;
  }
};

const DEMO_BASE_TIME = Date.parse("2025-01-20T12:00:00.000Z");

const demoTimestamp = (minutesAgo: number) =>
  new Date(DEMO_BASE_TIME - minutesAgo * 60 * 1000).toISOString();

const buildDemoAlert = (id: string): AlertDetail => ({
  id,
  transaction_id: "TX-20259",
  created_at: demoTimestamp(42),
  risk_score: 0.92,
  risk_level: "HIGH",
  status: "TRIAGED",
  decision: "REVIEW",
  model_version: "fraud-v3.4.1",
  merchant_id: "MER-4491",
  user_id: "user_291",
  amount: 1284.54,
  currency: "USD",
  device_id: "device_883a",
  ip_address: "198.51.100.24",
  location: "Austin, TX",
  channel: "web"
});

const demoSignals: EvidenceSignal[] = [
  {
    label: "Velocity spike",
    severity: "HIGH",
    detail: "8 transactions in 5 minutes across 3 merchants."
  },
  {
    label: "Geo mismatch",
    severity: "MEDIUM",
    detail: "Shipping country differs from prior user location."
  },
  {
    label: "Device risk",
    severity: "LOW",
    detail: "Browser fingerprint seen in one recent alert."
  }
];

const demoTimeline: TimelineEvent[] = [
  {
    title: "Alert created",
    actor: "Model pipeline",
    timestamp: demoTimestamp(44)
  },
  {
    title: "Assigned to analyst",
    actor: "Auto triage",
    timestamp: demoTimestamp(32),
    note: "Matched priority rules for high risk."
  },
  {
    title: "Status updated",
    actor: "Analyst: A. Perez",
    timestamp: demoTimestamp(12),
    note: "Marked triaged and queued for review."
  }
];

const demoRelated = [
  {
    id: "AL-10084",
    created_at: demoTimestamp(140),
    risk_level: "MEDIUM",
    risk_score: 0.63,
    status: "INVESTIGATING",
    device: "device_883a"
  },
  {
    id: "AL-10012",
    created_at: demoTimestamp(360),
    risk_level: "HIGH",
    risk_score: 0.88,
    status: "TRIAGED",
    device: "device_883a"
  }
];

const severityClass = (severity: EvidenceSignal["severity"]) => {
  if (severity === "HIGH") {
    return "chip high";
  }
  if (severity === "MEDIUM") {
    return "chip medium";
  }
  return "chip low";
};

const riskClass = (level: string) => {
  if (level === "HIGH" || level === "CRITICAL") {
    return "chip high";
  }
  if (level === "MEDIUM") {
    return "chip medium";
  }
  return "chip low";
};

type AlertDetailsPageProps = {
  params: { id: string };
};

const redirectToLogin = () => {
  if (typeof window === "undefined") {
    return;
  }
  const next = `${window.location.pathname}${window.location.search}`;
  window.location.href = `/login?next=${encodeURIComponent(next)}`;
};

export default function AlertDetailsPage({ params }: AlertDetailsPageProps) {
  const { user } = useAuth();
  const [alert, setAlert] = useState<AlertDetail | null>(null);
  const [activeTab, setActiveTab] = useState("overview");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const demoWriteBlocked = DEMO_PUBLIC_READONLY && !user;
  const demoWriteTitle = demoWriteBlocked ? DEMO_READONLY_MESSAGE : undefined;

  const ensureWriteAccess = () => {
    if (!demoWriteBlocked) {
      return true;
    }
    redirectToLogin();
    return false;
  };

  const loadAlert = async () => {
    setLoading(true);
    setError(null);
    try {
      // TODO: wire real backend endpoint for /api/alerts/:id
      const data = await apiFetch<AlertDetail>(`/api/alerts/${params.id}`);
      setAlert(data);
      return;
    } catch {
      try {
        const data = await apiFetch<AlertDetail>(`/alerts/${params.id}`);
        setAlert(data);
        return;
      } catch (err) {
        setError("Unable to load alert details. Showing demo data for now.");
        setAlert(buildDemoAlert(params.id));
      }
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadAlert();
  }, [params.id]);

  const stats = useMemo(() => {
    if (!alert) {
      return [];
    }
    return [
      { label: "Risk", value: alert.risk_level },
      {
        label: "Score",
        value: alert.risk_score ? alert.risk_score.toFixed(3) : "n/a"
      },
      { label: "Status", value: alert.status },
      { label: "Decision", value: alert.decision ?? "n/a" },
      { label: "Model version", value: alert.model_version ?? "n/a" }
    ];
  }, [alert]);

  const relatedRows = demoRelated.map((item) => [
    item.id,
    formatDateTime(item.created_at),
    <span key={`${item.id}-risk`} className={riskClass(item.risk_level)}>
      {item.risk_level}
    </span>,
    item.risk_score.toFixed(2),
    item.status,
    item.device
  ]);

  return (
    <section className="section">
      <PageHeader
        title={`Alert ${params.id}`}
        subtitle="Detailed alert context, evidence, and related activity."
        breadcrumbs={[
          { label: "Alerts", href: "/alerts" },
          { label: params.id }
        ]}
        actions={
          <div className="page-actions">
            <button
              className="button secondary"
              type="button"
              onClick={() => {
                if (!ensureWriteAccess()) {
                  return;
                }
                // TODO: implement case creation workflow.
                console.info("Create case clicked");
              }}
              disabled={demoWriteBlocked}
              title={demoWriteTitle}
            >
              Create case
            </button>
            <button
              className="button secondary"
              type="button"
              onClick={() => {
                if (!ensureWriteAccess()) {
                  return;
                }
                // TODO: implement assignment workflow.
                console.info("Assign clicked");
              }}
              disabled={demoWriteBlocked}
              title={demoWriteTitle}
            >
              Assign
            </button>
            <button
              className="button"
              type="button"
              onClick={() => {
                if (!ensureWriteAccess()) {
                  return;
                }
                // TODO: implement triage update.
                console.info("Mark triaged clicked");
              }}
              disabled={demoWriteBlocked}
              title={demoWriteTitle}
            >
              Mark triaged
            </button>
            <button
              className="button secondary"
              type="button"
              onClick={() => {
                if (!ensureWriteAccess()) {
                  return;
                }
                // TODO: implement export payload.
                console.info("Export JSON clicked");
              }}
              disabled={demoWriteBlocked}
              title={demoWriteTitle}
            >
              Export JSON
            </button>
          </div>
        }
      />

      {error && (
        <div className="error-banner">
          <span>{error}</span>
          <button className="button secondary" type="button" onClick={loadAlert}>
            Retry
          </button>
        </div>
      )}

      <div className="grid">
        {loading &&
          Array.from({ length: 5 }).map((_, index) => (
            <div key={`stat-skeleton-${index}`} className="card stat-card">
              <div className="skeleton skeleton-line" />
              <div className="skeleton skeleton-line" />
            </div>
          ))}
        {!loading &&
          stats.map((stat) => (
            <StatCard key={stat.label} label={stat.label} value={stat.value} />
          ))}
      </div>

      <Tabs tabs={tabs} active={activeTab} onChange={setActiveTab} />

      {activeTab === "overview" && (
        <div className="grid">
          <div className="card">
            <h3>Transaction summary</h3>
            <div className="kv-grid">
              <div className="kv-item">
                <div className="kv-label">Transaction ID</div>
                <div className="kv-value">{alert?.transaction_id ?? "n/a"}</div>
              </div>
              <div className="kv-item">
                <div className="kv-label">Merchant</div>
                <div className="kv-value">{alert?.merchant_id ?? "n/a"}</div>
              </div>
              <div className="kv-item">
                <div className="kv-label">User</div>
                <div className="kv-value">{alert?.user_id ?? "n/a"}</div>
              </div>
              <div className="kv-item">
                <div className="kv-label">Amount</div>
                <div className="kv-value">
                  {formatMoney(alert?.amount, alert?.currency)}
                </div>
              </div>
              <div className="kv-item">
                <div className="kv-label">Created</div>
                <div className="kv-value">
                  {formatDateTime(alert?.created_at)}
                </div>
              </div>
            </div>
          </div>

          <div className="card">
            <h3>Location and device</h3>
            <div className="kv-grid">
              <div className="kv-item">
                <div className="kv-label">Location</div>
                <div className="kv-value">{alert?.location ?? "n/a"}</div>
              </div>
              <div className="kv-item">
                <div className="kv-label">Device</div>
                <div className="kv-value">{alert?.device_id ?? "n/a"}</div>
              </div>
              <div className="kv-item">
                <div className="kv-label">IP address</div>
                <div className="kv-value">{alert?.ip_address ?? "n/a"}</div>
              </div>
              <div className="kv-item">
                <div className="kv-label">Channel</div>
                <div className="kv-value">{alert?.channel ?? "n/a"}</div>
              </div>
            </div>
          </div>
        </div>
      )}

      {activeTab === "evidence" && (
        <div className="card">
          <h3>Signals and evidence</h3>
          <div className="signal-list">
            {demoSignals.map((signal) => (
              <div key={signal.label} className="signal-item">
                <div>
                  <div className="signal-title">{signal.label}</div>
                  <div className="muted">{signal.detail}</div>
                </div>
                <span className={severityClass(signal.severity)}>
                  {signal.severity}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {activeTab === "timeline" && (
        <div className="card">
          <h3>Timeline</h3>
          <ul className="timeline">
            {demoTimeline.map((event) => (
              <li key={`${event.title}-${event.timestamp}`}>
                <div className="timeline-title">{event.title}</div>
                <div className="muted">
                  {event.actor} - {formatDateTime(event.timestamp)}
                </div>
                {event.note && <div className="timeline-note">{event.note}</div>}
              </li>
            ))}
          </ul>
        </div>
      )}

      {activeTab === "related" && (
        <DataTable
          title="Related alerts"
          subtitle="Same device, IP, or user."
          columns={[
            "Alert ID",
            "Created",
            "Risk",
            "Score",
            "Status",
            "Device"
          ]}
          rows={relatedRows}
          emptyText="No related alerts found."
          actions={
            <div className="table-controls">
              <select>
                <option>Sort by newest</option>
                <option>Sort by risk score</option>
                <option>Sort by status</option>
              </select>
              <button className="button secondary" type="button">
                Export
              </button>
            </div>
          }
          footer={
            <div className="table-pagination">
              <button className="button secondary" type="button">
                Prev
              </button>
              <span>Page 1 of 3</span>
              <button className="button secondary" type="button">
                Next
              </button>
            </div>
          }
        />
      )}
    </section>
  );
}
