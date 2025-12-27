"use client";

import { useEffect, useMemo, useState } from "react";

import { apiFetch } from "@/lib/api";
import { formatDateUTC } from "@/lib/format";
import {
  ReviewDecision,
  ReviewDecisionResponse,
  ReviewQueueItem
} from "@/lib/types";

export default function ManualReviewPage() {
  const [queue, setQueue] = useState<ReviewQueueItem[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [notes, setNotes] = useState("");
  const [search, setSearch] = useState("");
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const loadQueue = async () => {
    setLoading(true);
    setError(null);
    try {
      const resp = await apiFetch<ReviewQueueItem[]>("/api/review/queue");
      setQueue(resp);
      if (resp.length && !selectedId) {
        setSelectedId(resp[0].id);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load queue");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadQueue();
  }, []);

  const filteredQueue = useMemo(() => {
    if (!search) {
      return queue;
    }
    const term = search.toLowerCase();
    return queue.filter((item) => {
      return (
        item.title.toLowerCase().includes(term) ||
        (item.transaction_id ?? "").toLowerCase().includes(term) ||
        (item.assigned_to ?? "").toLowerCase().includes(term)
      );
    });
  }, [queue, search]);

  const selected = filteredQueue.find((item) => item.id === selectedId) ?? null;

  const submitDecision = async (decision: ReviewDecision) => {
    if (!selected) {
      return;
    }
    setSaving(true);
    setMessage(null);
    setError(null);
    try {
      const resp = await apiFetch<ReviewDecisionResponse>(
        `/api/review/${selected.id}/decision`,
        {
          method: "POST",
          body: JSON.stringify({
            decision,
            notes: notes || undefined
          })
        }
      );
      setMessage(resp.message);
      setNotes("");
      setSelectedId(null);
      await loadQueue();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to submit decision");
    } finally {
      setSaving(false);
    }
  };

  return (
    <section className="section">
      <div>
        <h3>Manual Review</h3>
        <p className="muted">
          Review open cases or high-risk alerts and capture decisions.
        </p>
      </div>

      <div className="toolbar">
        <input
          placeholder="Search title, transaction, assignee"
          value={search}
          onChange={(event) => setSearch(event.target.value)}
        />
        <button className="button secondary" onClick={loadQueue}>
          Refresh queue
        </button>
      </div>

      {loading && <div className="empty">Loading review queue...</div>}
      {error && <div className="error">{error}</div>}
      {!loading && !error && filteredQueue.length === 0 && (
        <div className="empty">No items require review right now.</div>
      )}

      {!loading && filteredQueue.length > 0 && (
        <div className="grid">
          <div className="card">
            <h3>Queue</h3>
            <table className="table">
              <thead>
                <tr>
                  <th>Updated</th>
                  <th>Title</th>
                  <th>Status</th>
                  <th>Risk</th>
                </tr>
              </thead>
              <tbody>
                {filteredQueue.map((item) => (
                  <tr
                    key={item.id}
                    onClick={() => setSelectedId(item.id)}
                    style={{
                      cursor: "pointer",
                      background:
                        item.id === selectedId
                          ? "rgba(34, 211, 238, 0.1)"
                          : undefined
                    }}
                  >
                    <td>
                      {item.updated_at
                        ? formatDateUTC(item.updated_at)
                        : "n/a"}
                    </td>
                    <td>{item.title}</td>
                    <td>{item.status}</td>
                    <td>
                      {item.risk_level ? (
                        <span
                          className={`chip ${
                            item.risk_level === "CRITICAL" ||
                            item.risk_level === "HIGH"
                              ? "high"
                              : item.risk_level === "MEDIUM"
                              ? "medium"
                              : "low"
                          }`}
                        >
                          {item.risk_level}
                        </span>
                      ) : (
                        "n/a"
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="card">
            <h3>Decision</h3>
            {selected ? (
              <div className="section">
                <div>
                  <p className="muted">Selected item</p>
                  <h4>{selected.title}</h4>
                  <p className="muted">
                    {selected.transaction_id ?? "No transaction ID"} -{" "}
                    {selected.source.toUpperCase()}
                  </p>
                </div>
                <div className="form">
                  <textarea
                    placeholder="Decision notes (optional)"
                    rows={5}
                    value={notes}
                    onChange={(event) => setNotes(event.target.value)}
                  />
                  <div className="toolbar">
                    <button
                      className="button secondary"
                      onClick={() => submitDecision("APPROVE")}
                      disabled={saving}
                    >
                      Approve
                    </button>
                    <button
                      className="button"
                      onClick={() => submitDecision("REJECT")}
                      disabled={saving}
                    >
                      Reject
                    </button>
                  </div>
                </div>
              </div>
            ) : (
              <p className="muted">Select an item to review.</p>
            )}
            {message && <div className="empty">{message}</div>}
          </div>
        </div>
      )}
    </section>
  );
}
