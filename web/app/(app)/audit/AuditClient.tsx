"use client";

import { useEffect, useMemo, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";

import { apiFetch, buildQuery } from "@/lib/api";
import { formatDateUTC } from "@/lib/format";
import { AuditEntry, Page } from "@/lib/types";

const PAGE_SIZE_OPTIONS = [10, 25, 50];

function AuditPageContent() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const [data, setData] = useState<Page<AuditEntry> | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const query = useMemo(() => {
    return {
      page: searchParams.get("page") ?? "1",
      page_size: searchParams.get("page_size") ?? "25",
      transaction_id: searchParams.get("transaction_id") ?? "",
      case_id: searchParams.get("case_id") ?? "",
      actor: searchParams.get("actor") ?? "",
      action: searchParams.get("action") ?? "",
      resource_type: searchParams.get("resource_type") ?? "",
      correlation_id: searchParams.get("correlation_id") ?? "",
      start_date: searchParams.get("start_date") ?? "",
      end_date: searchParams.get("end_date") ?? ""
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
          transaction_id: query.transaction_id || undefined,
          case_id: query.case_id || undefined,
          actor: query.actor || undefined,
          action: query.action || undefined,
          resource_type: query.resource_type || undefined,
          correlation_id: query.correlation_id || undefined,
          start_date: query.start_date || undefined,
          end_date: query.end_date || undefined
        });
        const resp = await apiFetch<Page<AuditEntry>>(`/audit${qs}`);
        setData(resp);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to load audit log");
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
    router.push(`/audit?${params.toString()}`);
  };

  const page = Number(query.page);
  const totalPages = data ? Math.ceil(data.total / data.page_size) : 1;

  return (
    <>
      <div>
        <h3>Audit log</h3>
        <p className="muted">Immutable record of scoring and analyst actions.</p>
      </div>

      <div className="toolbar">
        <input
          placeholder="Actor email"
          value={query.actor}
          onChange={(event) =>
            updateQuery({ actor: event.target.value, page: "1" })
          }
        />
        <input
          placeholder="Action"
          value={query.action}
          onChange={(event) =>
            updateQuery({ action: event.target.value, page: "1" })
          }
        />
        <input
          placeholder="Resource type"
          value={query.resource_type}
          onChange={(event) =>
            updateQuery({ resource_type: event.target.value, page: "1" })
          }
        />
        <input
          placeholder="Transaction ID"
          value={query.transaction_id}
          onChange={(event) =>
            updateQuery({ transaction_id: event.target.value, page: "1" })
          }
        />
        <input
          placeholder="Case ID"
          value={query.case_id}
          onChange={(event) =>
            updateQuery({ case_id: event.target.value, page: "1" })
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
        <input
          placeholder="Correlation ID"
          value={query.correlation_id}
          onChange={(event) =>
            updateQuery({ correlation_id: event.target.value, page: "1" })
          }
        />
        <input
          placeholder="Start date (YYYY-MM-DD)"
          value={query.start_date}
          onChange={(event) => updateQuery({ start_date: event.target.value })}
        />
        <input
          placeholder="End date (YYYY-MM-DD)"
          value={query.end_date}
          onChange={(event) => updateQuery({ end_date: event.target.value })}
        />
      </div>

      {loading && <div className="empty">Loading audit log...</div>}
      {error && <div className="error">{error}</div>}
      {!loading && data && data.items.length === 0 && (
        <div className="empty">No audit entries found.</div>
      )}

      {!loading && data && data.items.length > 0 && (
        <table className="table">
          <thead>
            <tr>
              <th>Timestamp</th>
              <th>Action</th>
              <th>Resource</th>
              <th>Resource ID</th>
              <th>Actor</th>
              <th>Role</th>
              <th>Correlation</th>
            </tr>
          </thead>
          <tbody>
            {data.items.map((entry) => (
              <tr key={entry.id}>
                <td>{formatDateUTC(entry.timestamp)}</td>
                <td>{entry.action}</td>
                <td>{entry.resource_type ?? entry.model_name ?? "n/a"}</td>
                <td>
                  {entry.resource_id ??
                    entry.transaction_id ??
                    entry.alert_id ??
                    entry.case_id ??
                    "n/a"}
                </td>
                <td>{entry.actor ?? "system"}</td>
                <td>{entry.role ?? "n/a"}</td>
                <td className="mono">{entry.correlation_id ?? "n/a"}</td>
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
    </>
  );
}

export default function AuditClient() {
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  if (!mounted) {
    return <div className="empty">Loading audit log...</div>;
  }

  return <AuditPageContent />;
}
