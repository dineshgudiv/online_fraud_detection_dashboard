"use client";

import { useEffect, useMemo, useState } from "react";

import PageHeader from "@/app/_components/PageHeader";
import StatCard from "@/app/_components/StatCard";
import Tabs from "@/app/_components/Tabs";
import DataTable from "@/app/_components/DataTable";
import { apiFetch } from "@/lib/api";
import { formatDateUTC } from "@/lib/format";
import { CaseNote, CaseRecord } from "@/lib/types";

type CaseDetail = CaseRecord & {
  priority?: "Low" | "Medium" | "High" | "Critical";
  assignee?: string | null;
  sla_target?: string;
  sla_remaining?: string;
  linked_entities?: {
    user?: string;
    device?: string;
    ip?: string;
    merchant?: string;
  };
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

const statusOptions = ["Open", "Investigating", "Escalated", "Closed"];
const priorityOptions = ["Low", "Medium", "High", "Critical"];

const formatDateTime = (value?: string | null) => formatDateUTC(value);

const DEMO_BASE_TIME = Date.parse("2025-01-20T12:00:00.000Z");

const demoTimestamp = (minutesAgo: number) =>
  new Date(DEMO_BASE_TIME - minutesAgo * 60 * 1000).toISOString();

const buildDemoCase = (id: string): CaseDetail => ({
  id,
  title: "High risk login ring",
  created_at: demoTimestamp(280),
  updated_at: demoTimestamp(45),
  status: "IN_REVIEW",
  assigned_to: "Analyst A. Perez",
  transaction_id: "TX-20259",
  user_id: "user_291",
  risk_level: "HIGH",
  risk_score: 0.88,
  notes: "Escalated due to device reuse across merchants.",
  notes_history: [
    {
      timestamp: demoTimestamp(120),
      author: "Analyst A. Perez",
      note: "Marked for escalation pending additional evidence."
    }
  ],
  priority: "High",
  assignee: "A. Perez",
  sla_target: "4h response",
  sla_remaining: "2h 10m remaining",
  linked_entities: {
    user: "user_291",
    device: "device_883a",
    ip: "198.51.100.24",
    merchant: "MER-4491"
  }
});

const demoTimeline: TimelineEvent[] = [
  {
    title: "Case opened",
    actor: "System",
    timestamp: demoTimestamp(250)
  },
  {
    title: "Assigned to analyst",
    actor: "Auto triage",
    timestamp: demoTimestamp(180),
    note: "Priority set to high."
  },
  {
    title: "Status updated",
    actor: "Analyst A. Perez",
    timestamp: demoTimestamp(40),
    note: "Escalated pending additional verification."
  }
];

const demoEvidence = [
  { label: "Multiple accounts", severity: "HIGH", detail: "5 accounts share the same device." },
  { label: "Merchant overlap", severity: "MEDIUM", detail: "Transactions span 3 merchants in 1 hour." },
  { label: "Known proxy", severity: "LOW", detail: "IP is associated with a proxy provider." }
];

const demoRelatedAlerts = [
  {
    id: "AL-10084",
    created_at: demoTimestamp(140),
    status: "INVESTIGATING",
    risk_score: 0.63
  },
  {
    id: "AL-10012",
    created_at: demoTimestamp(360),
    status: "TRIAGED",
    risk_score: 0.88
  }
];

const toWorkflowStatus = (status?: string | null) => {
  if (status === "OPEN") {
    return "Open";
  }
  if (status === "IN_REVIEW") {
    return "Investigating";
  }
  if (status === "RESOLVED") {
    return "Closed";
  }
  return "Open";
};

const severityChip = (severity: string) => {
  if (severity === "HIGH") {
    return "chip high";
  }
  if (severity === "MEDIUM") {
    return "chip medium";
  }
  return "chip low";
};

type CaseDetailsPageProps = {
  params: { id: string };
};

export default function CaseDetailsPage({ params }: CaseDetailsPageProps) {
  const [caseData, setCaseData] = useState<CaseDetail | null>(null);
  const [activeTab, setActiveTab] = useState("overview");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [workflowStatus, setWorkflowStatus] = useState(statusOptions[0]);
  const [priority, setPriority] = useState(priorityOptions[2]);
  const [assignee, setAssignee] = useState("");
  const [noteDraft, setNoteDraft] = useState("");
  const [noteHistory, setNoteHistory] = useState<CaseNote[]>([]);

  const loadCase = async () => {
    setLoading(true);
    setError(null);
    try {
      // TODO: wire real backend endpoint for /api/cases/:id
      const data = await apiFetch<CaseDetail>(`/api/cases/${params.id}`);
      setCaseData(data);
      return;
    } catch {
      try {
        const data = await apiFetch<CaseDetail>(`/cases/${params.id}`);
        setCaseData(data);
        return;
      } catch {
        setError("Unable to load case details. Showing demo data for now.");
        setCaseData(buildDemoCase(params.id));
      }
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadCase();
  }, [params.id]);

  useEffect(() => {
    if (!caseData) {
      return;
    }
    setWorkflowStatus(toWorkflowStatus(caseData.status));
    setPriority(caseData.priority ?? "High");
    setAssignee(caseData.assignee ?? caseData.assigned_to ?? "");
    setNoteHistory(caseData.notes_history ?? []);
  }, [caseData]);

  const stats = useMemo(() => {
    if (!caseData) {
      return [];
    }
    return [
      { label: "Status", value: workflowStatus },
      { label: "Priority", value: priority },
      { label: "Assignee", value: assignee || "Unassigned" },
      { label: "SLA", value: caseData.sla_remaining ?? "n/a" }
    ];
  }, [caseData, workflowStatus, priority, assignee]);

  const relatedRows = demoRelatedAlerts.map((alert) => [
    alert.id,
    formatDateTime(alert.created_at),
    alert.status,
    alert.risk_score.toFixed(2)
  ]);

  return (
    <section className="section">
      <PageHeader
        title={`Case ${params.id}`}
        subtitle="Case workflow, linked entities, and investigation timeline."
        breadcrumbs={[
          { label: "Cases", href: "/cases" },
          { label: params.id }
        ]}
        actions={
          <div className="page-actions">
            <button className="button secondary" type="button">
              Assign
            </button>
            <button className="button secondary" type="button">
              Escalate
            </button>
            <button className="button" type="button">
              Close case
            </button>
          </div>
        }
      />

      {error && (
        <div className="error-banner">
          <span>{error}</span>
          <button className="button secondary" type="button" onClick={loadCase}>
            Retry
          </button>
        </div>
      )}

      <div className="grid">
        {loading &&
          Array.from({ length: 4 }).map((_, index) => (
            <div key={`case-stat-${index}`} className="card stat-card">
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
            <div className="panel-header">
              <div>
                <h3>Workflow</h3>
                <p className="muted">Status, assignment, and SLA targets.</p>
              </div>
              <div className="table-actions">
                <span className="pill">{caseData?.sla_target ?? "SLA not set"}</span>
              </div>
            </div>
            <div className="form-grid">
              <label className="field">
                <span>Status</span>
                <select
                  value={workflowStatus}
                  onChange={(event) => setWorkflowStatus(event.target.value)}
                >
                  {statusOptions.map((option) => (
                    <option key={option} value={option}>
                      {option}
                    </option>
                  ))}
                </select>
              </label>
              <label className="field">
                <span>Priority</span>
                <select
                  value={priority}
                  onChange={(event) => setPriority(event.target.value)}
                >
                  {priorityOptions.map((option) => (
                    <option key={option} value={option}>
                      {option}
                    </option>
                  ))}
                </select>
              </label>
              <label className="field">
                <span>Assignee</span>
                <input
                  value={assignee}
                  onChange={(event) => setAssignee(event.target.value)}
                  placeholder="Analyst name"
                />
              </label>
              <div className="field">
                <span>SLA</span>
                <div className="muted">
                  {caseData?.sla_remaining ?? "No SLA timer configured"}
                </div>
              </div>
            </div>
          </div>

          <div className="card">
            <div className="panel-header">
              <div>
                <h3>Notes</h3>
                <p className="muted">Capture investigation updates.</p>
              </div>
            </div>
            <div className="form">
              <textarea
                rows={4}
                placeholder="Add case note"
                value={noteDraft}
                onChange={(event) => setNoteDraft(event.target.value)}
              />
              <div className="toolbar">
                <button
                  className="button"
                  type="button"
                  onClick={() => {
                    if (!noteDraft.trim()) {
                      return;
                    }
                    // TODO: persist case notes to backend.
                    const nextNote: CaseNote = {
                      timestamp: new Date().toISOString(),
                      author: "Current analyst",
                      note: noteDraft.trim()
                    };
                    setNoteHistory((prev) => [nextNote, ...prev]);
                    setNoteDraft("");
                  }}
                >
                  Add note
                </button>
              </div>
            </div>
            {noteHistory.length > 0 ? (
              <ul className="timeline">
                {noteHistory.map((note) => (
                  <li key={`${note.timestamp}-${note.author}`}>
                    <div className="timeline-title">{note.author}</div>
                    <div className="muted">{formatDateTime(note.timestamp)}</div>
                    <div className="timeline-note">{note.note}</div>
                  </li>
                ))}
              </ul>
            ) : (
              <div className="empty">No notes yet.</div>
            )}
          </div>

          <div className="card">
            <div className="panel-header">
              <div>
                <h3>Linked entities</h3>
                <p className="muted">User, device, and merchant context.</p>
              </div>
              <button className="button secondary" type="button">
                View graph
              </button>
            </div>
            <div className="grid">
              <StatCard label="User" value={caseData?.linked_entities?.user ?? "n/a"} />
              <StatCard
                label="Device"
                value={caseData?.linked_entities?.device ?? "n/a"}
              />
              <StatCard label="IP" value={caseData?.linked_entities?.ip ?? "n/a"} />
              <StatCard
                label="Merchant"
                value={caseData?.linked_entities?.merchant ?? "n/a"}
              />
            </div>
          </div>
        </div>
      )}

      {activeTab === "evidence" && (
        <div className="card">
          <h3>Evidence summary</h3>
          <div className="signal-list">
            {demoEvidence.map((signal) => (
              <div key={signal.label} className="signal-item">
                <div>
                  <div className="signal-title">{signal.label}</div>
                  <div className="muted">{signal.detail}</div>
                </div>
                <span className={severityChip(signal.severity)}>
                  {signal.severity}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {activeTab === "timeline" && (
        <div className="card">
          <h3>Case timeline</h3>
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
          title="Alerts in this case"
          subtitle="Alerts linked to this case."
          columns={["Alert ID", "Created", "Status", "Score"]}
          rows={relatedRows}
          emptyText="No related alerts found."
          actions={
            <div className="table-controls">
              <select>
                <option>Sort by newest</option>
                <option>Sort by status</option>
                <option>Sort by score</option>
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
              <span>Page 1 of 2</span>
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
