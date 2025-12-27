"use client";

import { useEffect, useMemo, useState } from "react";

import { apiFetch } from "@/lib/api";
import { formatNumber } from "@/lib/format";
import { GeoSummary } from "@/lib/types";

export default function GeoFraudMapPage() {
  const [data, setData] = useState<GeoSummary | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [hydrated, setHydrated] = useState(false);

  useEffect(() => {
    setHydrated(true);
    const run = async () => {
      setLoading(true);
      setError(null);
      try {
        const resp = await apiFetch<GeoSummary>("/api/geo/summary");
        setData(resp);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to load geo summary");
      } finally {
        setLoading(false);
      }
    };
    run();
  }, []);

  const summary = useMemo(() => {
    if (!data) {
      return { total: 0, highRisk: 0, maxCount: 0 };
    }
    const total = data.items.reduce((acc, item) => acc + item.count, 0);
    const highRisk = data.items.reduce((acc, item) => acc + item.high_risk_count, 0);
    const maxCount = data.items.reduce((acc, item) => Math.max(acc, item.count), 0);
    return { total, highRisk, maxCount };
  }, [data]);

  const groupBy = hydrated ? data?.group_by ?? "location" : "location";
  const totalLabel = hydrated ? formatNumber(summary.total) : "n/a";
  const highRiskLabel = hydrated ? formatNumber(summary.highRisk) : "n/a";
  const coverageLabel =
    hydrated && data ? `${formatNumber(data.items.length)} locations` : "n/a";

  return (
    <section className="section">
      <div>
        <h3>Geo Fraud Map</h3>
        <p className="muted">
          Location clustering based on recent alerts. Grouped by{" "}
          {groupBy}.
        </p>
      </div>

      <div className="grid">
        <div className="card">
          <h3>Total Alerts</h3>
          <p>{totalLabel}</p>
        </div>
        <div className="card">
          <h3>High-Risk Alerts</h3>
          <p>{highRiskLabel}</p>
        </div>
        <div className="card">
          <h3>Coverage</h3>
          <p>{coverageLabel}</p>
        </div>
      </div>

      {loading && <div className="empty">Loading geo summary...</div>}
      {error && <div className="error">{error}</div>}
      {!loading && data && data.items.length === 0 && (
        <div className="empty">No geo data available yet.</div>
      )}

      {!loading && data && data.items.length > 0 && (
        <table className="table">
          <thead>
            <tr>
              <th>Location</th>
              <th>Total</th>
              <th>High Risk</th>
              <th>Risk %</th>
              <th>Volume</th>
            </tr>
          </thead>
          <tbody>
            {data.items.map((item) => {
              const riskPct = item.count
                ? (item.high_risk_count / item.count) * 100
                : 0;
              const barWidth = summary.maxCount
                ? Math.round((item.count / summary.maxCount) * 100)
                : 0;
              return (
                <tr key={item.label}>
                  <td>{item.label}</td>
                  <td>{formatNumber(item.count)}</td>
                  <td>{formatNumber(item.high_risk_count)}</td>
                  <td>{riskPct.toFixed(1)}%</td>
                  <td>
                    <div
                      style={{
                        height: 6,
                        borderRadius: 999,
                        background: "rgba(148, 163, 184, 0.2)",
                        overflow: "hidden"
                      }}
                    >
                      <div
                        style={{
                          height: "100%",
                          width: `${barWidth}%`,
                          background: "linear-gradient(90deg, #22d3ee, #0ea5e9)"
                        }}
                      />
                    </div>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      )}
    </section>
  );
}
