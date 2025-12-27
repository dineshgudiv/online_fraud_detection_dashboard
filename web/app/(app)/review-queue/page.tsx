"use client";

import { Suspense, useEffect, useMemo, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";

import { apiFetch, buildQuery } from "@/lib/api";
import { formatDateUTC } from "@/lib/format";
import { Alert, Page } from "@/lib/types";

const PAGE_SIZE_OPTIONS = [10, 25, 50];

function ReviewQueuePageContent() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const [data, setData] = useState<Page<Alert> | null>(null);
  const [selected, setSelected] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [bulkLoading, setBulkLoading] = useState(false);
  const [refreshKey, setRefreshKey] = useState(0);

  const query = useMemo(() => {
    return {
      page: searchParams.get("page") ?? "1",
      page_size: searchParams.get("page_size") ?? "25",
      search: searchParams.get("search") ?? ""
    };
  }, [searchParams]);

  useEffect(() => {
    const run = async () => {
      setLoading(true);
      setError(null);
      try {
        const qs = buildQuery({
          page: query.page,
          page_size: query.page_size,
          search: query.search || undefined,
          status: "NEW"
        });
        const resp = await apiFetch<Page<Alert>>(`/alerts${qs}`);
        setData(resp);
        setSelected([]);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to load queue");
      } finally {
        setLoading(false);
      }
    };
    run();
  }, [query, refreshKey]);

  const updateQuery = (patch: Record<string, string>) => {
    const params = new URLSearchParams(searchParams.toString());
    Object.entries(patch).forEach(([key, value]) => {
      if (value) {
        params.set(key, value);
      } else {
        params.delete(key);
      }
    });
    router.push(`/review-queue?${params.toString()}`);
  };

  const toggleSelect = (id: string) => {
    setSelected((prev) =>
      prev.includes(id) ? prev.filter((item) => item !== id) : [...prev, id]
    );
  };

  const selectAll = () => {
    if (!data) return;
    setSelected(data.items.map((item) => item.id));
  };

  const clearSelection = () => setSelected([]);

  const bulkTriage = async () => {
    if (selected.length === 0) return;
    setBulkLoading(true);
    try {
      await apiFetch(`/alerts/bulk`, {
        method: "POST",
        body: JSON.stringify({
          alert_ids: selected,
          status: "TRIAGED"
        })
      });
      setRefreshKey((prev) => prev + 1);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to triage alerts");
    } finally {
      setBulkLoading(false);
    }
  };

  const page = Number(query.page);
  const totalPages = data ? Math.ceil(data.total / data.page_size) : 1;

  return (
    <section className="section">
      <div>
        <h3>Review Queue</h3>
        <p className="muted">Alerts awaiting analyst triage.</p>
      </div>

      <div className="toolbar">
        <input
          placeholder="Search transaction, user, merchant"
          value={query.search}
          onChange={(event) =>
            updateQuery({ search: event.target.value, page: "1" })
          }
        />
        <select
          value={query.page_size}
          onChange={(event) => updateQuery({ page_size: event.target.value })}
        >
          {PAGE_SIZE_OPTIONS.map((size) => (
            <option key={size} value={size}>
              {size} / page
            </option>
          ))}
        </select>
        <button className="button" onClick={bulkTriage} disabled={bulkLoading}>
          {bulkLoading ? "Updating..." : "Mark TRIAGED"}
        </button>
        <button className="button secondary" onClick={selectAll}>
          Select all
        </button>
        <button className="button secondary" onClick={clearSelection}>
          Clear
        </button>
      </div>

      {loading && <div className="empty">Loading queue...</div>}
      {error && <div className="error">{error}</div>}
      {!loading && data && data.items.length === 0 && (
        <div className="empty">No alerts in the review queue.</div>
      )}

      {!loading && data && data.items.length > 0 && (
        <table className="table">
          <thead>
            <tr>
              <th></th>
              <th>Created</th>
              <th>Transaction</th>
              <th>Risk</th>
              <th>Score</th>
            </tr>
          </thead>
          <tbody>
            {data.items.map((alert) => (
              <tr key={alert.id}>
                <td>
                  <input
                    type="checkbox"
                    checked={selected.includes(alert.id)}
                    onChange={() => toggleSelect(alert.id)}
                  />
                </td>
                <td>{formatDateUTC(alert.created_at)}</td>
                <td>{alert.transaction_id}</td>
                <td>{alert.risk_level}</td>
                <td>{alert.risk_score.toFixed(3)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}

      {!loading && data && (
        <div className="pagination">
          <button
            className="button secondary"
            onClick={() => updateQuery({ page: String(Math.max(1, page - 1)) })}
            disabled={page <= 1}
          >
            Prev
          </button>
          <span>
            Page {page} of {totalPages}
          </span>
          <button
            className="button secondary"
            onClick={() => updateQuery({ page: String(Math.min(totalPages, page + 1)) })}
            disabled={page >= totalPages}
          >
            Next
          </button>
        </div>
      )}
    </section>
  );
}

export default function ReviewQueuePage() {
  return (
    <Suspense
      fallback={
        <section className="section">
          <div className="empty">Loading review queue...</div>
        </section>
      }
    >
      <ReviewQueuePageContent />
    </Suspense>
  );
}

