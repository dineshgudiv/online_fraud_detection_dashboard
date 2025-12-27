"use client";

import { useEffect, useState } from "react";

import StatCard from "@/app/_components/StatCard";
import { apiFetch } from "@/lib/api";
import { SecurityHsmProvider } from "@/lib/security";

export default function HsmPage() {
  const [hsm, setHsm] = useState<SecurityHsmProvider | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let mounted = true;

    const load = async () => {
      setLoading(true);
      setError(null);
      try {
        const resp = await apiFetch<SecurityHsmProvider>("/security/hsm/provider");
        if (!mounted) {
          return;
        }
        setHsm(resp);
      } catch (err) {
        if (!mounted) {
          return;
        }
        setError(err instanceof Error ? err.message : "Failed to load HSM status");
      } finally {
        if (mounted) {
          setLoading(false);
        }
      }
    };

    load();
    return () => {
      mounted = false;
    };
  }, []);

  const statusLabel = hsm
    ? hsm.configured
      ? `HSM: ${hsm.provider ?? "configured"}`
      : "HSM: not configured"
    : "HSM: loading";

  const keysInUse = loading ? "Loading..." : String(hsm?.key_count ?? 0);
  const rotationOverdue = loading
    ? "Loading..."
    : hsm?.configured
    ? "0"
    : "Not configured";
  const lastKeyEvent = loading
    ? "Loading..."
    : hsm?.configured
    ? "No recent events"
    : "Not configured";

  return (
    <div className="section">
      {error && <div className="error">{error}</div>}

      <div className="card">
        <div className="panel-header">
          <div>
            <h3>HSM / Key Custody</h3>
            <p className="muted">
              Central view of signing and encryption keys, rotation state, and
              audit readiness.
            </p>
            {hsm && hsm.configured && (
              <p className="muted">
                Provider: {hsm.provider ?? "configured"} - Rotation policy:{" "}
                {hsm.rotation_policy ?? "Not set"} - Health: {hsm.health}
              </p>
            )}
          </div>
          <div className="status-pill">{statusLabel}</div>
        </div>
      </div>

      <div className="grid">
        <StatCard label="Keys in use" value={keysInUse} hint="Active keys in custody" />
        <StatCard label="Rotation overdue" value={rotationOverdue} hint="Keys past policy window" />
        <StatCard label="Last key event" value={lastKeyEvent} hint="Most recent key activity" />
      </div>

      <div className="card">
        <h3>Key operations (backend only)</h3>
        <p className="muted">
          Implement actions via the API and log to Audit. Do not expose private
          key material to the UI.
        </p>
      </div>
    </div>
  );
}
