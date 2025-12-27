"use client";

import { useCallback, useEffect, useMemo, useState } from "react";

import { useAuth } from "@/app/_components/AuthProvider";
import { DEMO_PUBLIC_READONLY, DEMO_READONLY_MESSAGE } from "@/lib/demo";

type DatasetItem = {
  version_id: string;
  original_filename: string;
  stored_filename: string;
  size_bytes: number;
  uploaded_at: string;
  is_active: boolean;
};

const API_BASE = "/api/proxy";

const redirectToLogin = () => {
  if (typeof window === "undefined") {
    return;
  }
  const next = `${window.location.pathname}${window.location.search}`;
  window.location.href = `/login?next=${encodeURIComponent(next)}`;
};

const parseErrorMessage = (raw: string) => {
  if (!raw) {
    return "Request failed. Please try again.";
  }
  try {
    const parsed = JSON.parse(raw) as { detail?: string };
    if (parsed && typeof parsed.detail === "string") {
      return parsed.detail;
    }
  } catch {
    // Fall through to raw text.
  }
  return raw;
};

export default function DatasetsSidebar() {
  const { user, loading: authLoading, refresh } = useAuth();
  const [datasets, setDatasets] = useState<DatasetItem[]>([]);
  const [active, setActive] = useState<string | null>(null);
  const [filter, setFilter] = useState("");
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [authRequired, setAuthRequired] = useState(false);
  const demoWriteBlocked = DEMO_PUBLIC_READONLY && !user;
  const demoWriteTitle = demoWriteBlocked ? DEMO_READONLY_MESSAGE : undefined;

  const ensureWriteAccess = () => {
    if (!demoWriteBlocked) {
      return true;
    }
    redirectToLogin();
    return false;
  };

  const handleAuthFailure = () => {
    setAuthRequired(true);
    setErr("Please login to continue.");
    redirectToLogin();
  };

  const fetchDatasets = useCallback(async () => {
    setLoading(true);
    setErr(null);
    setAuthRequired(false);
    try {
      const res = await fetch(`${API_BASE}/datasets`, { credentials: "include" });
      if (res.status === 401) {
        handleAuthFailure();
        return;
      }
      if (res.status === 403) {
        setErr("You do not have permission to view datasets.");
        return;
      }
      if (!res.ok) {
        setErr(parseErrorMessage(await res.text()));
        return;
      }
      const data = (await res.json()) as DatasetItem[];
      setDatasets(data);
      const activeItem = data.find((item) => item.is_active);
      setActive(activeItem?.original_filename ?? null);
    } catch {
      setErr("Failed to load datasets.");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchDatasets();
  }, [fetchDatasets]);

  const setActiveDataset = async (versionId: string) => {
    if (!ensureWriteAccess()) {
      return;
    }
    setErr(null);
    try {
      const res = await fetch(`${API_BASE}/datasets/set-active`, {
        method: "POST",
        credentials: "include",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ version_id: versionId })
      });
      if (res.status === 401) {
        handleAuthFailure();
        return;
      }
      if (res.status === 403) {
        setErr("You do not have permission to update datasets.");
        return;
      }
      if (!res.ok) {
        setErr(parseErrorMessage(await res.text()));
        return;
      }
      await fetchDatasets();
    } catch {
      setErr("Failed to set active dataset.");
    }
  };

  const deleteDataset = async (versionId: string, label: string) => {
    if (!ensureWriteAccess()) {
      return;
    }
    if (!confirm(`Delete dataset: ${label}?`)) {
      return;
    }
    setErr(null);
    try {
      const res = await fetch(`${API_BASE}/datasets/${encodeURIComponent(versionId)}`, {
        method: "DELETE",
        credentials: "include"
      });
      if (res.status === 401) {
        handleAuthFailure();
        return;
      }
      if (res.status === 403) {
        setErr("You do not have permission to delete datasets.");
        return;
      }
      if (!res.ok) {
        setErr(parseErrorMessage(await res.text()));
        return;
      }
      await fetchDatasets();
    } catch {
      setErr("Failed to delete dataset.");
    }
  };

  const downloadDataset = (versionId: string) => {
    const url = `${API_BASE}/datasets/${encodeURIComponent(versionId)}/download`;
    window.open(url, "_blank", "noopener,noreferrer");
  };

  const previewDataset = (versionId: string) => {
    if (typeof window === "undefined") {
      return;
    }
    window.location.href = `/dataset?file=${encodeURIComponent(versionId)}`;
  };

  const logout = async () => {
    setErr(null);
    try {
      const res = await fetch("/api/session/logout", { method: "POST" });
      if (!res.ok) {
        setErr("Failed to log out.");
        return;
      }
      await refresh();
      redirectToLogin();
    } catch {
      setErr("Failed to log out.");
    }
  };

  const visibleDatasets = useMemo(() => {
    const filtered = datasets.filter((dataset) =>
      dataset.original_filename.toLowerCase().includes(filter.toLowerCase())
    );
    return filtered.slice(0, 10);
  }, [datasets, filter]);

  return (
    <div style={{ display: "grid", gap: "0.75rem" }}>
      <div style={{ fontSize: "0.85rem", fontWeight: 700 }}>Datasets</div>
      <div className="muted" style={{ fontSize: "0.75rem" }}>
        Active: {active ?? "None"}
      </div>
      <div className="toolbar">
        <input
          placeholder="Search datasets"
          value={filter}
          onChange={(event) => setFilter(event.target.value)}
        />
      </div>
      {loading && <div className="muted">Loading...</div>}
      {err && (
        <div className="error" style={{ display: "flex", justifyContent: "space-between", gap: "0.6rem" }}>
          <span>{err}</span>
          {authRequired && (
            <button className="button secondary compact" type="button" onClick={redirectToLogin}>
              Login
            </button>
          )}
        </div>
      )}
      {!loading && visibleDatasets.length === 0 && (
        <div className="muted" style={{ fontSize: "0.75rem" }}>
          No datasets found.
        </div>
      )}
      {!loading &&
        visibleDatasets.map((dataset) => (
          <div
            key={dataset.version_id}
            style={{
              border: "1px solid rgba(148,163,184,0.2)",
              borderRadius: "0.8rem",
              padding: "0.6rem",
              display: "grid",
              gap: "0.5rem"
            }}
          >
            <div style={{ fontSize: "0.8rem", fontWeight: 600, wordBreak: "break-word" }}>
              {dataset.original_filename}
            </div>
            <div style={{ display: "flex", flexWrap: "wrap", gap: "0.4rem" }}>
              <button className="button secondary" type="button" onClick={() => previewDataset(dataset.version_id)}>
                Preview
              </button>
              <button className="button secondary" type="button" onClick={() => downloadDataset(dataset.version_id)}>
                Download
              </button>
              <button
                className="button"
                type="button"
                onClick={() => setActiveDataset(dataset.version_id)}
                disabled={demoWriteBlocked}
                title={demoWriteTitle}
              >
                Set Active
              </button>
              <button
                className="button secondary"
                type="button"
                onClick={() => deleteDataset(dataset.version_id, dataset.original_filename)}
                disabled={demoWriteBlocked}
                title={demoWriteTitle}
              >
                Delete
              </button>
            </div>
          </div>
        ))}

      <div style={{ marginTop: "1rem", display: "grid", gap: "0.35rem" }}>
        <div className="muted" style={{ fontSize: "0.75rem", fontWeight: 700 }}>
          Switch user
        </div>
        {authLoading ? (
          <div className="muted" style={{ fontSize: "0.75rem" }}>
            Loading session...
          </div>
        ) : user ? (
          <button className="button secondary compact" type="button" onClick={logout}>
            Logout
          </button>
        ) : (
          <button className="button compact" type="button" onClick={redirectToLogin}>
            Login
          </button>
        )}
      </div>
    </div>
  );
}
