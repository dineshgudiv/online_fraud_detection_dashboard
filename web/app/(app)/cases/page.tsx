"use client";

import { Suspense, useEffect, useMemo, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";

import { apiFetch, buildQuery } from "@/lib/api";
import { formatDateUTC } from "@/lib/format";
import { CASE_STATUSES, CaseRecord, Page } from "@/lib/types";

const PAGE_SIZE_OPTIONS = [10, 25, 50];

function CasesPageContent() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const [data, setData] = useState<Page<CaseRecord> | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const query = useMemo(() => {
    return {
      page: searchParams.get("page") ?? "1",
      page_size: searchParams.get("page_size") ?? "25",
      status: searchParams.get("status") ?? "",
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
          status: query.status || undefined,
          search: query.search || undefined
        });
        const resp = await apiFetch<Page<CaseRecord>>(`/cases${qs}`);
        setData(resp);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to load cases");
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
    router.push(`/cases?${params.toString()}`);
  };

  const page = Number(query.page);
  const totalPages = data ? Math.ceil(data.total / data.page_size) : 1;

  return (
    <section className="section">
      <div>
        <h3>Cases</h3>
        <p className="muted">Manage escalations and investigations.</p>
      </div>

      <div className="toolbar">
        <input
          placeholder="Search title, transaction, user"
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
        <select
          value={query.status}
          onChange={(event) =>
            updateQuery({ status: event.target.value, page: "1" })
          }
        >
          <option value="">All status</option>
          {CASE_STATUSES.map((status) => (
            <option key={status} value={status}>
              {status}
            </option>
          ))}
        </select>
      </div>

      {loading && <div className="empty">Loading cases...</div>}
      {error && <div className="error">{error}</div>}
      {!loading && data && data.items.length === 0 && (
        <div className="empty">No cases match the current filters.</div>
      )}

      {!loading && data && data.items.length > 0 && (
        <table className="table">
          <thead>
            <tr>
              <th>Updated</th>
              <th>Title</th>
              <th>Status</th>
              <th>Assignee</th>
              <th>Transaction</th>
            </tr>
          </thead>
          <tbody>
            {data.items.map((item) => (
              <tr
                key={item.id}
                onClick={() => router.push(`/cases/${item.id}`)}
                style={{ cursor: "pointer" }}
              >
                <td>{formatDateUTC(item.updated_at)}</td>
                <td>{item.title}</td>
                <td>{item.status}</td>
                <td>{item.assigned_to ?? "Unassigned"}</td>
                <td>{item.transaction_id ?? "-"}</td>
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

export default function CasesPage() {
  return (
    <Suspense
      fallback={
        <section className="section">
          <div className="empty">Loading cases...</div>
        </section>
      }
    >
      <CasesPageContent />
    </Suspense>
  );
}

