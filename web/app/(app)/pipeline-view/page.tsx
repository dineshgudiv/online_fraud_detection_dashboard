"use client";

import { useEffect, useMemo, useState } from "react";

import { apiFetch } from "@/lib/api";
import { formatDateUTC, formatNumber } from "@/lib/format";
import { PipelineStatus } from "@/lib/types";

export default function PipelineViewPage() {
  const [data, setData] = useState<PipelineStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const run = async () => {
      setLoading(true);
      setError(null);
      try {
        const resp = await apiFetch<PipelineStatus>("/api/pipeline/status");
        setData(resp);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to load pipeline status");
      } finally {
        setLoading(false);
      }
    };
    run();
  }, []);

  const stages = useMemo(() => {
    if (!data) {
      return [];
    }
    return [
      {
        name: "Ingest",
        detail: `${formatNumber(data.alerts_total)} events`
      },
      {
        name: "Scoring",
        detail: `${formatNumber(data.high_risk_alerts)} high-risk`
      },
      {
        name: "Review",
        detail: `${formatNumber(data.open_cases)} open cases`
      },
      {
        name: "Resolution",
        detail: `${formatNumber(data.cases_total)} total cases`
      }
    ];
  }, [data]);

  return (
    <section className="section">
      <div>
        <h3>Pipeline View</h3>
        <p className="muted">
          System health and transaction flow snapshot from the backend pipeline.
        </p>
      </div>

      {loading && <div className="empty">Loading pipeline status...</div>}
      {error && <div className="error">{error}</div>}

      {!loading && data && (
        <>
          <div className="grid">
            <div className="card">
              <h3>Status</h3>
              <p>{data.status.toUpperCase()}</p>
              <p className="muted">Source: {data.source}</p>
            </div>
            <div className="card">
              <h3>Last Ingest</h3>
              <p>
                {data.last_ingest_at
                  ? formatDateUTC(data.last_ingest_at)
                  : "Unknown"}
              </p>
            </div>
            <div className="card">
              <h3>Open Cases</h3>
              <p>{formatNumber(data.open_cases)}</p>
            </div>
            <div className="card">
              <h3>High-Risk Alerts</h3>
              <p>{formatNumber(data.high_risk_alerts)}</p>
            </div>
          </div>

          <div className="grid">
            {stages.map((stage) => (
              <div className="card" key={stage.name}>
                <h3>{stage.name}</h3>
                <p>{stage.detail}</p>
              </div>
            ))}
          </div>

          <div className="card">
            <h3>Observability</h3>
            <p className="muted">
              {data.prometheus_enabled && data.prometheus_url
                ? `Prometheus enabled: ${data.prometheus_url}`
                : "Prometheus not configured in this environment."}
            </p>
          </div>
        </>
      )}
    </section>
  );
}
