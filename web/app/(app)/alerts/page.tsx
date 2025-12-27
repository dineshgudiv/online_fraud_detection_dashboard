"use client";

import { Suspense, useEffect, useMemo, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";

import { apiFetch, buildQuery } from "@/lib/api";
import { formatDateUTC } from "@/lib/format";
import { ALERT_STATUSES, Alert, Page, RISK_LEVELS } from "@/lib/types";

const PAGE_SIZE_OPTIONS = [10, 25, 50];

function AlertsPageContent() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const [data, setData] = useState<Page<Alert> | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const query = useMemo(() => {
    return {
      page: searchParams.get("page") ?? "1",
      page_size: searchParams.get("page_size") ?? "25",
      status: searchParams.get("status") ?? "",
      risk_level: searchParams.get("risk_level") ?? "",
      search: searchParams.get("search") ?? "",
      sort: searchParams.get("sort") ?? "created_at",
      order: searchParams.get("order") ?? "desc"
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
          status: query.status || undefined,
          risk_level: query.risk_level || undefined,
          search: query.search || undefined,
          sort: query.sort,
          order: query.order
        });
        const resp = await apiFetch<Page<Alert>>(`/alerts${qs}`);
        setData(resp);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to load alerts");
      } finally {
        setLoading(false);
      }
    };
    run();
  }, [query]);

  const updateQuery = (patch: Record<string, string>) => {
    const params = new URLSearchParams(searchParams.toString());
    Object.entries(patch).forEach(([key, value]) => {
      if (value) {
        params.set(key, value);
      } else {
        params.delete(key);
      }
    });
    router.push(`/alerts?${params.toString()}`);
  };

  const page = Number(query.page);
  const totalPages = data ? Math.ceil(data.total / data.page_size) : 1;

  return (
    <section className="section">
      <div>
        <h3>Alerts</h3>
        <p className="muted">Monitor and triage suspicious transactions.</p>
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
          value={query.status}
          onChange={(event) =>
            updateQuery({ status: event.target.value, page: "1" })
          }
        >
          <option value="">All status</option>
          {ALERT_STATUSES.map((status) => (
            <option key={status} value={status}>
              {status}
            </option>
          ))}
        </select>
        <select
          value={query.risk_level}
          onChange={(event) =>
            updateQuery({ risk_level: event.target.value, page: "1" })
          }
        >
          <option value="">All risk</option>
          {RISK_LEVELS.map((risk) => (
            <option key={risk} value={risk}>
              {risk}
            </option>
          ))}
        </select>
        <select
          value={query.sort}
          onChange={(event) => updateQuery({ sort: event.target.value })}
        >
          <option value="created_at">Newest</option>
          <option value="risk_score">Risk score</option>
          <option value="risk_level">Risk level</option>
        </select>
        <select
          value={query.order}
          onChange={(event) => updateQuery({ order: event.target.value })}
        >
          <option value="desc">Desc</option>
          <option value="asc">Asc</option>
        </select>
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
      </div>

      {loading && <div className="empty">Loading alerts...</div>}
      {error && <div className="error">{error}</div>}
      {!loading && data && data.items.length === 0 && (
        <div className="empty">No alerts match the current filters.</div>
      )}

      {!loading && data && data.items.length > 0 && (
        <table className="table">
          <thead>
            <tr>
              <th>Created</th>
              <th>Transaction</th>
              <th>Risk</th>
              <th>Score</th>
              <th>Status</th>
              <th>Decision</th>
            </tr>
          </thead>
          <tbody>
            {data.items.map((alert) => (
              <tr
                key={alert.id}
                onClick={() => router.push(`/alerts/${alert.id}`)}
                style={{ cursor: "pointer" }}
              >
                <td>{formatDateUTC(alert.created_at)}</td>
                <td>{alert.transaction_id}</td>
                <td>
                  <span
                    className={`chip ${
                      alert.risk_level === "CRITICAL" || alert.risk_level === "HIGH"
                        ? "high"
                        : alert.risk_level === "MEDIUM"
                        ? "medium"
                        : "low"
                    }`}
                  >
                    {alert.risk_level}
                  </span>
                </td>
                <td>{alert.risk_score.toFixed(3)}</td>
                <td>{alert.status}</td>
                <td>{alert.decision ?? "-"}</td>
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

export default function AlertsPage() {
  return (
    <Suspense
      fallback={
        <section className="section">
          <div className="empty">Loading alerts...</div>
        </section>
      }
    >
      <AlertsPageContent />
    </Suspense>
  );
}

