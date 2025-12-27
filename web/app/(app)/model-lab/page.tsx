"use client";

import { useCallback, useEffect, useMemo, useState } from "react";

import StatCard from "@/app/_components/StatCard";
import { useAuth } from "@/app/_components/AuthProvider";
import { apiFetch } from "@/lib/api";
import { formatDateUTC } from "@/lib/format";

type ModelSnapshot = {
  model_name: string | null;
  model_version: string | null;
  last_trained_at: string | null;
  auc: number | null;
  f1: number | null;
  latency_ms_p95: number | null;
  drift_score: number | null;
  data_freshness_min: number | null;
  status: string;
};

type RetrainResponse = {
  job_id: string;
  queued: boolean;
  message: string;
};

type RetrainJob = {
  id: string;
  status: string;
  requested_by?: string | null;
  requested_at?: string | null;
  started_at?: string | null;
  finished_at?: string | null;
  duration_seconds?: number | null;
  error_message?: string | null;
  model_name?: string | null;
  model_version?: string | null;
};

type RetrainJobList = {
  items: RetrainJob[];
  count: number;
};

type DriftPoint = {
  timestamp: string;
  overall_score: number;
};

type DriftFeature = {
  feature: string;
  psi: number;
  ks_pvalue?: number | null;
};

type DriftSummary = {
  level: "GREEN" | "YELLOW" | "RED";
  message: string;
  overall: DriftPoint[];
  top_features: DriftFeature[];
};

type LineageModel = {
  version: string;
  trained_at: string;
  metrics?: Record<string, number> | null;
};

type LineageDataset = {
  id: string;
  version: string;
  created_at: string;
  source?: string | null;
};

type LineageFeatureSet = {
  id: string;
  version: string;
  schema_hash: string;
  created_at: string;
};

type LineageRetrainJob = {
  id: string;
  status: string;
  requested_by?: string | null;
  requested_at?: string | null;
  started_at?: string | null;
  finished_at?: string | null;
  error_message?: string | null;
};

type Lineage = {
  model: LineageModel | null;
  dataset: LineageDataset | null;
  feature_set: LineageFeatureSet | null;
  retrain_job: LineageRetrainJob | null;
};

const formatFloat = (value: number | null | undefined, digits = 3) => {
  if (value === null || value === undefined) {
    return "n/a";
  }
  return value.toFixed(digits);
};

const formatInt = (value: number | null | undefined) => {
  if (value === null || value === undefined) {
    return "n/a";
  }
  return String(Math.round(value));
};

const skeletonLine = (width = "60%") => (
  <div className="skeleton skeleton-line" style={{ width }} />
);

export default function ModelLabPage() {
  const { can, loading: authLoading } = useAuth();
  const [snapshot, setSnapshot] = useState<ModelSnapshot | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [actionMessage, setActionMessage] = useState<string | null>(null);
  const [running, setRunning] = useState(false);
  const [jobs, setJobs] = useState<RetrainJob[]>([]);
  const [jobsLoading, setJobsLoading] = useState(false);
  const [jobsError, setJobsError] = useState<string | null>(null);
  const [drift, setDrift] = useState<DriftSummary | null>(null);
  const [driftLoading, setDriftLoading] = useState(false);
  const [driftError, setDriftError] = useState<string | null>(null);
  const [lineage, setLineage] = useState<Lineage | null>(null);
  const [lineageLoading, setLineageLoading] = useState(false);
  const [lineageError, setLineageError] = useState<string | null>(null);
  const [showLineageDetail, setShowLineageDetail] = useState(false);

  const loadSnapshot = useCallback(async (showLoading = false) => {
    if (showLoading) {
      setLoading(true);
    }
    setError(null);
    try {
      const resp = await apiFetch<ModelSnapshot>("/ml/model/snapshot");
      setSnapshot(resp);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load model snapshot");
    } finally {
      if (showLoading) {
        setLoading(false);
      }
    }
  }, []);

  const loadJobs = useCallback(async () => {
    if (!can("model:retrain")) {
      return;
    }
    setJobsLoading(true);
    setJobsError(null);
    try {
      const resp = await apiFetch<RetrainJobList>("/ml/model/retrain/jobs");
      setJobs(resp.items);
    } catch (err) {
      setJobsError(err instanceof Error ? err.message : "Failed to load retrain jobs");
    } finally {
      setJobsLoading(false);
    }
  }, [can]);

  const loadDrift = useCallback(async () => {
    setDriftLoading(true);
    setDriftError(null);
    try {
      const resp = await apiFetch<DriftSummary>("/ml/drift/summary?hours=168");
      setDrift(resp);
    } catch (err) {
      setDriftError(err instanceof Error ? err.message : "Failed to load drift summary");
    } finally {
      setDriftLoading(false);
    }
  }, []);

  const loadLineage = useCallback(async () => {
    setLineageLoading(true);
    setLineageError(null);
    try {
      const resp = await apiFetch<Lineage>("/ml/lineage");
      setLineage(resp);
    } catch (err) {
      setLineageError(err instanceof Error ? err.message : "Failed to load lineage");
    } finally {
      setLineageLoading(false);
    }
  }, []);

  useEffect(() => {
    loadSnapshot(true);
    loadDrift();
    loadLineage();
    if (!authLoading && can("model:retrain")) {
      loadJobs();
    }
    const timer = setInterval(() => {
      loadSnapshot(false);
      loadDrift();
      loadLineage();
      if (can("model:retrain")) {
        loadJobs();
      }
    }, 15000);
    return () => clearInterval(timer);
  }, [authLoading, can, loadDrift, loadJobs, loadLineage, loadSnapshot]);

  const triggerRetrain = async () => {
    setRunning(true);
    setActionMessage(null);
    try {
      const resp = await apiFetch<RetrainResponse>("/ml/model/retrain", {
        method: "POST"
      });
      setActionMessage(`${resp.message} (${resp.job_id})`);
      await loadSnapshot(false);
      await loadJobs();
      await loadLineage();
    } catch (err) {
      setActionMessage(
        err instanceof Error ? err.message : "Failed to trigger retraining"
      );
    } finally {
      setRunning(false);
    }
  };

  const activeJob = useMemo(() => {
    return jobs.find((job) => job.status === "RUNNING" || job.status === "QUEUED") ?? null;
  }, [jobs]);

  const retrainDisabled = running || Boolean(activeJob);

  const modelName = snapshot?.model_name ?? "n/a";
  const modelVersion = snapshot?.model_version ?? "n/a";
  const lastTrained = formatDateUTC(snapshot?.last_trained_at ?? null);

  const driftLevelClass = drift?.level ? drift.level.toLowerCase() : "unknown";
  const lineageSummary = useMemo(() => {
    if (!lineage?.model || !lineage?.dataset || !lineage?.feature_set) {
      return "Lineage data not available yet.";
    }
    return `Model v${lineage.model.version} trained on Dataset ${lineage.dataset.version} (FeatureSet ${lineage.feature_set.version}).`;
  }, [lineage]);
  const hasLineage = Boolean(
    lineage?.model || lineage?.dataset || lineage?.feature_set || lineage?.retrain_job
  );

  return (
    <section className="section">
      <div>
        <h3>Model Lab</h3>
        <p className="muted">
          Model health snapshot and controlled retraining actions.
        </p>
      </div>

      {can("model:retrain") && (
        <div className="toolbar">
          <button className="button secondary" onClick={triggerRetrain} disabled={retrainDisabled}>
            {retrainDisabled ? "Retrain running" : "Trigger retrain"}
          </button>
          {activeJob && (
            <span className="muted">
              Active job: {activeJob.id} ({activeJob.status})
            </span>
          )}
          {actionMessage && <span className="muted">{actionMessage}</span>}
        </div>
      )}

      {loading && <div className="empty">Loading model metrics...</div>}
      {error && <div className="error">{error}</div>}

      <div className="grid">
        <div className="card">
          <h3>Model</h3>
          <p>{loading ? "Loading..." : modelName}</p>
          <p className="muted">Version: {loading ? "Loading..." : modelVersion}</p>
          <p className="muted">Last trained: {loading ? "Loading..." : lastTrained}</p>
        </div>
        <StatCard
          label="AUC"
          value={loading ? "Loading..." : formatFloat(snapshot?.auc)}
          hint="ROC-AUC (latest window)"
        />
        <StatCard
          label="F1"
          value={loading ? "Loading..." : formatFloat(snapshot?.f1)}
          hint="F1 score at tuned threshold"
        />
        <StatCard
          label="Latency p95 (ms)"
          value={loading ? "Loading..." : formatInt(snapshot?.latency_ms_p95)}
          hint="Inference latency p95"
        />
      </div>

      <div className="grid">
        <StatCard
          label="Drift score"
          value={loading ? "Loading..." : formatFloat(snapshot?.drift_score)}
          hint="Lower is better (demo metric)"
        />
        <StatCard
          label="Data freshness (min)"
          value={loading ? "Loading..." : formatInt(snapshot?.data_freshness_min)}
          hint="Minutes since last ingest"
        />
        <StatCard
          label="Status"
          value={loading ? "Loading..." : snapshot?.status ?? "n/a"}
          hint="Overall health"
        />
      </div>

      {can("model:retrain") && (
        <div className="card">
          <div className="panel-header">
            <div>
              <h3>Retrain jobs</h3>
              <p className="muted">Queue history and execution outcomes.</p>
            </div>
            <div className="table-actions">
              <button
                className="button secondary compact"
                type="button"
                onClick={loadJobs}
                disabled={jobsLoading}
              >
                {jobsLoading ? "Refreshing..." : "Refresh"}
              </button>
            </div>
          </div>

          {jobsError && <div className="error">{jobsError}</div>}
          {jobsLoading && <div className="empty">Loading retrain jobs...</div>}
          {!jobsLoading && jobs.length === 0 && (
            <div className="empty">No retrain jobs yet.</div>
          )}

          {!jobsLoading && jobs.length > 0 && (
            <table className="table">
              <thead>
                <tr>
                  <th>Status</th>
                  <th>Requested</th>
                  <th>Started</th>
                  <th>Finished</th>
                  <th>Duration</th>
                  <th>Requested by</th>
                  <th>Error</th>
                </tr>
              </thead>
              <tbody>
                {jobs.map((job) => (
                  <tr key={job.id}>
                    <td>{job.status}</td>
                    <td>{formatDateUTC(job.requested_at ?? null)}</td>
                    <td>{formatDateUTC(job.started_at ?? null)}</td>
                    <td>{formatDateUTC(job.finished_at ?? null)}</td>
                    <td>
                      {job.duration_seconds ? `${Math.round(job.duration_seconds)}s` : "n/a"}
                    </td>
                    <td>{job.requested_by ?? "n/a"}</td>
                    <td>{job.error_message ?? "n/a"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      )}

      <div className="card">
        <div className="panel-header">
          <div>
            <h3>Drift detection</h3>
            <p className="muted">Population stability and feature shift monitoring.</p>
          </div>
          {drift?.level && (
            <div className={`status-badge ${driftLevelClass}`}>
              {drift.level}
            </div>
          )}
        </div>

        {driftError && <div className="error">{driftError}</div>}
        {driftLoading && <div className="empty">Loading drift summary...</div>}

        {!driftLoading && drift && (
          <>
            <p className="muted">{drift.message}</p>
            <div className="drift-chart">
              <DriftChart points={drift.overall} />
            </div>
            <div className="grid">
              {drift.top_features.map((feature) => (
                <div key={feature.feature} className="card stat-card">
                  <div className="stat-label">{feature.feature}</div>
                  <div className="stat-value">{feature.psi.toFixed(3)}</div>
                  <div className="stat-hint">PSI score</div>
                </div>
              ))}
            </div>
          </>
        )}
      </div>

      <div className="card">
        <div className="panel-header">
          <div>
            <h3>Lineage & traceability</h3>
            <p className="muted">Dataset, feature set, and model version provenance.</p>
          </div>
          <div className="table-actions">
            <button
              className="button secondary compact"
              type="button"
              onClick={loadLineage}
              disabled={lineageLoading}
            >
              {lineageLoading ? "Refreshing..." : "Refresh"}
            </button>
            <button
              className="button secondary compact"
              type="button"
              onClick={() => setShowLineageDetail((prev) => !prev)}
              disabled={!hasLineage}
            >
              {showLineageDetail ? "Hide metadata" : "View metadata"}
            </button>
            <button
              className="button compact"
              type="button"
              onClick={() => {
                window.location.href = "/dataset";
              }}
              disabled={!lineage?.dataset}
            >
              Open dataset workspace
            </button>
          </div>
        </div>

        {lineageError && <div className="error">{lineageError}</div>}

        {lineageLoading && (
          <div className="kv-grid">
            <div className="kv-item">
              <div className="kv-label">Model version</div>
              {skeletonLine("65%")}
            </div>
            <div className="kv-item">
              <div className="kv-label">Trained at</div>
              {skeletonLine("55%")}
            </div>
            <div className="kv-item">
              <div className="kv-label">Dataset</div>
              {skeletonLine("70%")}
            </div>
            <div className="kv-item">
              <div className="kv-label">Feature set</div>
              {skeletonLine("60%")}
            </div>
          </div>
        )}

        {!lineageLoading && !hasLineage && (
          <div className="empty">No lineage data yet.</div>
        )}

        {!lineageLoading && hasLineage && (
          <>
            <p className="muted">{lineageSummary}</p>
            <div className="kv-grid">
              <div className="kv-item">
                <div className="kv-label">Model version</div>
                <div className="kv-value">
                  {lineage?.model ? `v${lineage.model.version}` : "Unknown"}
                </div>
              </div>
              <div className="kv-item">
                <div className="kv-label">Trained at</div>
                <div className="kv-value">
                  {formatDateUTC(lineage?.model?.trained_at ?? null)}
                </div>
              </div>
              <div className="kv-item">
                <div className="kv-label">Dataset</div>
                <div className="kv-value">
                  {lineage?.dataset?.version ?? "Unknown"}
                </div>
              </div>
              <div className="kv-item">
                <div className="kv-label">Feature set</div>
                <div className="kv-value">
                  {lineage?.feature_set?.version ?? "Unknown"}
                </div>
              </div>
            </div>

            {showLineageDetail && (
              <>
                <div className="kv-grid">
                  <div className="kv-item">
                    <div className="kv-label">Dataset ID</div>
                    <div className="kv-value mono">
                      {lineage?.dataset?.id ?? "Unknown"}
                    </div>
                  </div>
                  <div className="kv-item">
                    <div className="kv-label">Dataset source</div>
                    <div className="kv-value">
                      {lineage?.dataset?.source ?? "Unknown"}
                    </div>
                  </div>
                  <div className="kv-item">
                    <div className="kv-label">Dataset created</div>
                    <div className="kv-value">
                      {formatDateUTC(lineage?.dataset?.created_at ?? null)}
                    </div>
                  </div>
                  <div className="kv-item">
                    <div className="kv-label">Feature set ID</div>
                    <div className="kv-value mono">
                      {lineage?.feature_set?.id ?? "Unknown"}
                    </div>
                  </div>
                  <div className="kv-item">
                    <div className="kv-label">Schema hash</div>
                    <div className="kv-value mono">
                      {lineage?.feature_set?.schema_hash ?? "Unknown"}
                    </div>
                  </div>
                  <div className="kv-item">
                    <div className="kv-label">Feature set created</div>
                    <div className="kv-value">
                      {formatDateUTC(lineage?.feature_set?.created_at ?? null)}
                    </div>
                  </div>
                </div>

                <div className="kv-grid">
                  <div className="kv-item">
                    <div className="kv-label">Retrain job</div>
                    <div className="kv-value mono">
                      {lineage?.retrain_job?.id ?? "Unknown"}
                    </div>
                  </div>
                  <div className="kv-item">
                    <div className="kv-label">Job status</div>
                    <div className="kv-value">
                      {lineage?.retrain_job?.status ?? "Unknown"}
                    </div>
                  </div>
                  <div className="kv-item">
                    <div className="kv-label">Requested by</div>
                    <div className="kv-value">
                      {lineage?.retrain_job?.requested_by ?? "Unknown"}
                    </div>
                  </div>
                  <div className="kv-item">
                    <div className="kv-label">Requested at</div>
                    <div className="kv-value">
                      {formatDateUTC(lineage?.retrain_job?.requested_at ?? null)}
                    </div>
                  </div>
                  <div className="kv-item">
                    <div className="kv-label">Finished at</div>
                    <div className="kv-value">
                      {formatDateUTC(lineage?.retrain_job?.finished_at ?? null)}
                    </div>
                  </div>
                  <div className="kv-item">
                    <div className="kv-label">Error</div>
                    <div className="kv-value">
                      {lineage?.retrain_job?.error_message ?? "None"}
                    </div>
                  </div>
                </div>

                {lineage?.model?.metrics && (
                  <div className="kv-grid">
                    <div className="kv-item">
                      <div className="kv-label">AUC</div>
                      <div className="kv-value">
                        {formatFloat(lineage.model.metrics.auc)}
                      </div>
                    </div>
                    <div className="kv-item">
                      <div className="kv-label">F1</div>
                      <div className="kv-value">
                        {formatFloat(lineage.model.metrics.f1)}
                      </div>
                    </div>
                    <div className="kv-item">
                      <div className="kv-label">Latency p95 (ms)</div>
                      <div className="kv-value">
                        {formatInt(lineage.model.metrics.latency_ms_p95)}
                      </div>
                    </div>
                  </div>
                )}
              </>
            )}
          </>
        )}
      </div>
    </section>
  );
}

function DriftChart({ points }: { points: DriftPoint[] }) {
  if (!points.length) {
    return <div className="empty">No drift data available.</div>;
  }
  const width = 520;
  const height = 160;
  const padding = 24;
  const scores = points.map((p) => p.overall_score);
  const max = Math.max(...scores, 0.05);
  const min = Math.min(...scores, 0);
  const range = max - min || 1;
  const step = (width - padding * 2) / Math.max(points.length - 1, 1);
  const coords = points.map((point, index) => {
    const x = padding + index * step;
    const y = height - padding - ((point.overall_score - min) / range) * (height - padding * 2);
    return { x, y };
  });
  const path = coords
    .map((point, index) => `${index === 0 ? "M" : "L"}${point.x},${point.y}`)
    .join(" ");

  return (
    <svg viewBox={`0 0 ${width} ${height}`} role="img">
      <rect x="0" y="0" width={width} height={height} fill="transparent" />
      <path d={path} fill="none" stroke="currentColor" strokeWidth="2" />
      {coords.map((point, index) => (
        <circle key={index} cx={point.x} cy={point.y} r="3" fill="currentColor" />
      ))}
    </svg>
  );
}
