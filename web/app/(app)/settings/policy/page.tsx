"use client";

import { useRef, useState } from "react";

import RequireRole from "@/app/_components/RequireRole";
import PageHeader from "@/app/_components/PageHeader";
import DataTable from "@/app/_components/DataTable";

type ListItem = {
  id: string;
  type: "IP" | "Device" | "Merchant";
  value: string;
  note?: string;
};

export default function PolicySettingsPage() {
  const [thresholds, setThresholds] = useState({
    low: 0.2,
    medium: 0.5,
    high: 0.75,
    critical: 0.9
  });
  const [allowlist, setAllowlist] = useState<ListItem[]>([
    { id: "allow-1", type: "Merchant", value: "MER-4491", note: "Trusted partner" }
  ]);
  const [blocklist, setBlocklist] = useState<ListItem[]>([
    { id: "block-1", type: "IP", value: "203.0.113.10", note: "Known abuse" }
  ]);
  const [newAllowType, setNewAllowType] = useState<ListItem["type"]>("IP");
  const [newAllowValue, setNewAllowValue] = useState("");
  const [newBlockType, setNewBlockType] = useState<ListItem["type"]>("IP");
  const [newBlockValue, setNewBlockValue] = useState("");
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);
  const allowIdRef = useRef(2);
  const blockIdRef = useRef(2);

  const savePolicy = () => {
    setSaving(true);
    setSaved(false);
    // TODO: GET/POST /api/settings/policy
    setTimeout(() => {
      setSaving(false);
      setSaved(true);
    }, 600);
  };

  const addAllowlist = () => {
    if (!newAllowValue.trim()) {
      return;
    }
    const id = `allow-${allowIdRef.current}`;
    allowIdRef.current += 1;
    setAllowlist((prev) => [
      ...prev,
      {
        id,
        type: newAllowType,
        value: newAllowValue.trim()
      }
    ]);
    setNewAllowValue("");
  };

  const addBlocklist = () => {
    if (!newBlockValue.trim()) {
      return;
    }
    const id = `block-${blockIdRef.current}`;
    blockIdRef.current += 1;
    setBlocklist((prev) => [
      ...prev,
      {
        id,
        type: newBlockType,
        value: newBlockValue.trim()
      }
    ]);
    setNewBlockValue("");
  };

  return (
    <RequireRole roles={["ADMIN"]}>
      <section className="section">
        <PageHeader
          title="Policy settings"
          subtitle="Define thresholds and governance for fraud decisions."
          actions={
            <div className="page-actions">
              <button className="button" type="button" onClick={savePolicy}>
                {saving ? "Saving..." : "Save policy"}
              </button>
            </div>
          }
        />

        {saved && <div className="success-banner">Policy saved (client only).</div>}

        <div className="card">
          <h3>Risk thresholds</h3>
          <p className="muted">
            Tune score thresholds that map to Low, Medium, High, and Critical risk.
          </p>
          <div className="form-grid">
            <label className="field">
              <span>Low</span>
              <input
                type="number"
                step="0.01"
                value={thresholds.low}
                onChange={(event) =>
                  setThresholds((prev) => ({
                    ...prev,
                    low: Number(event.target.value)
                  }))
                }
              />
            </label>
            <label className="field">
              <span>Medium</span>
              <input
                type="number"
                step="0.01"
                value={thresholds.medium}
                onChange={(event) =>
                  setThresholds((prev) => ({
                    ...prev,
                    medium: Number(event.target.value)
                  }))
                }
              />
            </label>
            <label className="field">
              <span>High</span>
              <input
                type="number"
                step="0.01"
                value={thresholds.high}
                onChange={(event) =>
                  setThresholds((prev) => ({
                    ...prev,
                    high: Number(event.target.value)
                  }))
                }
              />
            </label>
            <label className="field">
              <span>Critical</span>
              <input
                type="number"
                step="0.01"
                value={thresholds.critical}
                onChange={(event) =>
                  setThresholds((prev) => ({
                    ...prev,
                    critical: Number(event.target.value)
                  }))
                }
              />
            </label>
          </div>
        </div>

        <div className="card">
          <div className="panel-header">
            <div>
              <h3>Allowlist</h3>
              <p className="muted">Trusted entities that should be exempt.</p>
            </div>
            <div className="inline-form">
              <select
                value={newAllowType}
                onChange={(event) =>
                  setNewAllowType(event.target.value as ListItem["type"])
                }
              >
                <option value="IP">IP</option>
                <option value="Device">Device</option>
                <option value="Merchant">Merchant</option>
              </select>
              <input
                placeholder="Value"
                value={newAllowValue}
                onChange={(event) => setNewAllowValue(event.target.value)}
              />
              <button className="button secondary" type="button" onClick={addAllowlist}>
                Add
              </button>
            </div>
          </div>

          <DataTable
            columns={["Type", "Value", "Note", ""]}
            rows={allowlist.map((item) => [
              item.type,
              item.value,
              item.note ?? "n/a",
              <button
                key={`remove-${item.id}`}
                className="button secondary"
                type="button"
                onClick={() =>
                  setAllowlist((prev) => prev.filter((entry) => entry.id !== item.id))
                }
              >
                Remove
              </button>
            ])}
            emptyText="No allowlist entries."
          />
        </div>

        <div className="card">
          <div className="panel-header">
            <div>
              <h3>Blocklist</h3>
              <p className="muted">Entities that should be blocked or escalated.</p>
            </div>
            <div className="inline-form">
              <select
                value={newBlockType}
                onChange={(event) =>
                  setNewBlockType(event.target.value as ListItem["type"])
                }
              >
                <option value="IP">IP</option>
                <option value="Device">Device</option>
                <option value="Merchant">Merchant</option>
              </select>
              <input
                placeholder="Value"
                value={newBlockValue}
                onChange={(event) => setNewBlockValue(event.target.value)}
              />
              <button className="button secondary" type="button" onClick={addBlocklist}>
                Add
              </button>
            </div>
          </div>

          <DataTable
            columns={["Type", "Value", "Note", ""]}
            rows={blocklist.map((item) => [
              item.type,
              item.value,
              item.note ?? "n/a",
              <button
                key={`remove-${item.id}`}
                className="button secondary"
                type="button"
                onClick={() =>
                  setBlocklist((prev) => prev.filter((entry) => entry.id !== item.id))
                }
              >
                Remove
              </button>
            ])}
            emptyText="No blocklist entries."
          />
        </div>
      </section>
    </RequireRole>
  );
}
