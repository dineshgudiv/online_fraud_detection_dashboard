"use client";

import { Suspense, useEffect, useMemo, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";

import PageHeader from "@/app/_components/PageHeader";
import DataTable from "@/app/_components/DataTable";
import { apiFetch } from "@/lib/api";
import { formatDateUTC } from "@/lib/format";

type AlertResult = {
  id: string;
  created_at: string;
  risk_level: string;
  status: string;
  score: number;
};

type CaseResult = {
  id: string;
  title: string;
  status: string;
  updated_at: string;
  priority: string;
};

type EntityResult = {
  id: string;
  type: string;
  label: string;
  risk: string;
  last_seen: string;
};

type SearchResults = {
  alerts: AlertResult[];
  cases: CaseResult[];
  entities: EntityResult[];
};

const formatDateTime = (value?: string | null) => formatDateUTC(value);

const DEMO_BASE_TIME = Date.parse("2025-01-20T12:00:00.000Z");

const demoTimestamp = (minutesAgo: number) =>
  new Date(DEMO_BASE_TIME - minutesAgo * 60 * 1000).toISOString();

const demoResults: SearchResults = {
  alerts: [
    {
      id: "AL-10084",
      created_at: demoTimestamp(40),
      risk_level: "MEDIUM",
      status: "INVESTIGATING",
      score: 0.63
    },
    {
      id: "AL-10012",
      created_at: demoTimestamp(120),
      risk_level: "HIGH",
      status: "TRIAGED",
      score: 0.88
    }
  ],
  cases: [
    {
      id: "CASE-772",
      title: "Card testing burst",
      status: "Investigating",
      updated_at: demoTimestamp(95),
      priority: "High"
    }
  ],
  entities: [
    {
      id: "user_291",
      type: "User",
      label: "user_291",
      risk: "Medium",
      last_seen: demoTimestamp(10)
    },
    {
      id: "device_883a",
      type: "Device",
      label: "device_883a",
      risk: "High",
      last_seen: demoTimestamp(5)
    }
  ]
};

function SearchPageContent() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const currentQuery = searchParams.get("q") ?? "";

  const [query, setQuery] = useState(currentQuery);
  const [debouncedQuery, setDebouncedQuery] = useState(currentQuery);
  const [results, setResults] = useState<SearchResults | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const updateUrl = (value: string) => {
    const params = new URLSearchParams(searchParams.toString());
    if (value) {
      params.set("q", value);
    } else {
      params.delete("q");
    }
    const qs = params.toString();
    router.replace(qs ? `/search?${qs}` : "/search");
  };

  const runSearchNow = (value: string) => {
    const trimmed = value.trim();
    setDebouncedQuery(trimmed);
    updateUrl(trimmed);
  };

  const loadResults = async (term: string) => {
    if (!term) {
      setResults(null);
      setError(null);
      return;
    }
    setLoading(true);
    setError(null);
    try {
      // TODO: wire /api/search?q= to backend aggregation.
      const data = await apiFetch<SearchResults>(
        `/api/search?q=${encodeURIComponent(term)}`
      );
      setResults(data);
    } catch {
      setError("Search endpoint is unavailable. Showing demo results.");
      setResults(demoResults);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    const handle = setTimeout(() => {
      const trimmed = query.trim();
      if (trimmed !== debouncedQuery) {
        setDebouncedQuery(trimmed);
        updateUrl(trimmed);
      }
    }, 400);
    return () => clearTimeout(handle);
  }, [query, debouncedQuery]);

  useEffect(() => {
    setQuery(currentQuery);
    setDebouncedQuery(currentQuery);
  }, [currentQuery]);

  useEffect(() => {
    void loadResults(debouncedQuery);
  }, [debouncedQuery]);

  const counts = useMemo(() => {
    return {
      alerts: results?.alerts.length ?? 0,
      cases: results?.cases.length ?? 0,
      entities: results?.entities.length ?? 0
    };
  }, [results]);

  const hasQuery = debouncedQuery.length > 0;

  return (
    <section className="section">
      <PageHeader
        title="Global search"
        subtitle="Search alerts, cases, users, devices, and merchants."
      />

      <div className="toolbar">
        <input
          value={query}
          placeholder="Search by user, device, IP, transaction, case"
          onChange={(event) => setQuery(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter") {
              event.preventDefault();
              runSearchNow(query);
            }
            if (event.key === "Escape") {
              setQuery("");
              runSearchNow("");
            }
          }}
        />
        <button className="button secondary" type="button" onClick={() => runSearchNow(query)}>
          Search
        </button>
      </div>

      {error && (
        <div className="error-banner">
          <span>{error}</span>
          <button className="button secondary" type="button" onClick={() => loadResults(debouncedQuery)}>
            Retry
          </button>
        </div>
      )}

      {!hasQuery && <div className="empty">Enter a query to search.</div>}

      {hasQuery && (
        <div className="section">
          <DataTable
            title={`Alerts (${counts.alerts})`}
            subtitle="Top alert matches"
            columns={["Alert ID", "Created", "Risk", "Status", "Score"]}
            rows={results?.alerts.map((alert) => [
              alert.id,
              formatDateTime(alert.created_at),
              alert.risk_level,
              alert.status,
              alert.score.toFixed(2)
            ])}
            loading={loading}
            emptyText="No alerts matched this query."
            actions={
              <button className="button secondary" type="button">
                View all
              </button>
            }
            footer={
              <div className="table-pagination">
                <span>Previewing top matches</span>
              </div>
            }
          />

          <DataTable
            title={`Cases (${counts.cases})`}
            subtitle="Recent case matches"
            columns={["Case ID", "Title", "Status", "Priority", "Updated"]}
            rows={results?.cases.map((item) => [
              item.id,
              item.title,
              item.status,
              item.priority,
              formatDateTime(item.updated_at)
            ])}
            loading={loading}
            emptyText="No cases matched this query."
            actions={
              <button className="button secondary" type="button">
                View all
              </button>
            }
            footer={
              <div className="table-pagination">
                <span>Previewing top matches</span>
              </div>
            }
          />

          <DataTable
            title={`Entities (${counts.entities})`}
            subtitle="Users, devices, IPs, and merchants"
            columns={["Entity", "Type", "Risk", "Last seen"]}
            rows={results?.entities.map((entity) => [
              entity.label,
              entity.type,
              entity.risk,
              formatDateTime(entity.last_seen)
            ])}
            loading={loading}
            emptyText="No entities matched this query."
            actions={
              <button className="button secondary" type="button">
                View all
              </button>
            }
            footer={
              <div className="table-pagination">
                <span>Previewing top matches</span>
              </div>
            }
          />
        </div>
      )}
    </section>
  );
}

export default function SearchPage() {
  return (
    <Suspense
      fallback={
        <section className="section">
          <div className="empty">Loading search...</div>
        </section>
      }
    >
      <SearchPageContent />
    </Suspense>
  );
}
