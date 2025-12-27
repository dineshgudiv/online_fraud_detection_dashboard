import { DEMO_MODE } from "./demo";

export const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE ||
  process.env.API_BASE ||
  "";

/** Query helper used by list pages */
export function buildQuery(params: Record<string, any> = {}) {
  const sp = new URLSearchParams();
  for (const [k, v] of Object.entries(params)) {
    if (v === undefined || v === null || v === "") continue;
    if (Array.isArray(v)) v.forEach((x) => sp.append(k, String(x)));
    else sp.set(k, String(v));
  }
  const s = sp.toString();
  return s ? `?${s}` : "";
}

/** Minimal demo fixtures so UI looks alive on LinkedIn */
function demoNowIso(minsAgo = 0) {
  return new Date(Date.now() - minsAgo * 60_000).toISOString();
}

async function demoResponse(path: string) {
  // Alerts list
  if (path.startsWith("/api/alerts") && !/\/api\/alerts\/[^/]+/.test(path)) {
    return {
      items: [
        {
          id: "ALRT-1042",
          created_at: demoNowIso(12),
          severity: "high",
          status: "open",
          score: 0.93,
          title: "Card testing burst detected",
          entity: "acct_92f1",
          reason_codes: ["velocity_spike", "bin_mismatch", "new_device"],
        },
        {
          id: "ALRT-1037",
          created_at: demoNowIso(48),
          severity: "medium",
          status: "triaged",
          score: 0.71,
          title: "High-risk geo + IP reputation",
          entity: "acct_77be",
          reason_codes: ["geo_anomaly", "ip_reputation"],
        },
      ],
      total: 2,
    };
  }

  // Alert detail
  if (/^\/api\/alerts\/[^/]+/.test(path)) {
    const id = path.split("/").pop() || "ALRT-0000";
    return {
      id,
      created_at: demoNowIso(12),
      severity: "high",
      status: "open",
      score: 0.93,
      title: "Card testing burst detected",
      entity: "acct_92f1",
      reason_codes: ["velocity_spike", "bin_mismatch", "new_device"],
      timeline: [
        { ts: demoNowIso(14), event: "Signal spike observed", actor: "detector" },
        { ts: demoNowIso(12), event: "Alert raised", actor: "rules-engine" },
        { ts: demoNowIso(5), event: "Queued for analyst review", actor: "ops" },
      ],
      attributes: {
        ip: "203.0.113.42",
        device: "Android 14 / Chrome",
        country: "IN",
        amount: 1499,
        currency: "INR",
      },
    };
  }

  // Cases list
  if (path.startsWith("/api/cases") && !/\/api\/cases\/[^/]+/.test(path)) {
    return {
      items: [
        {
          id: "CASE-884",
          created_at: demoNowIso(120),
          status: "investigating",
          priority: "p1",
          title: "Account takeover suspicion",
          entity: "acct_92f1",
        },
      ],
      total: 1,
    };
  }

  // Case detail
  if (/^\/api\/cases\/[^/]+/.test(path)) {
    const id = path.split("/").pop() || "CASE-000";
    return {
      id,
      created_at: demoNowIso(120),
      status: "investigating",
      priority: "p1",
      title: "Account takeover suspicion",
      entity: "acct_92f1",
      notes: [
        { ts: demoNowIso(110), author: "analyst", text: "Multiple failed logins + new device." },
      ],
    };
  }

  // Audit list
  if (path.startsWith("/api/audit")) {
    return {
      items: [
        { ts: demoNowIso(15), actor: "system", action: "ALERT_CREATED", target: "ALRT-1042" },
        { ts: demoNowIso(8), actor: "analyst", action: "ALERT_VIEWED", target: "ALRT-1042" },
      ],
      total: 2,
    };
  }

  // Dataset list / sidebar
  if (path.startsWith("/api/datasets")) {
    return {
      items: [
        { id: "ds_kaggle_ccfd", name: "Kaggle Credit Card Fraud", rows: 284807, updated_at: demoNowIso(1440) },
        { id: "ds_synth_stream", name: "Synthetic Streaming Events", rows: 120000, updated_at: demoNowIso(180) },
      ],
      total: 2,
    };
  }

  // Security center stubs (so pages don't break)
  if (path.startsWith("/api/security") || path.startsWith("/api/pki") || path.startsWith("/api/certificates")) {
    return {
      certificates: [
        { id: "cert_demo_01", subject: "CN=Fraud Console Demo", not_before: demoNowIso(60 * 24 * 20), not_after: demoNowIso(-60 * 24 * 10) },
      ],
      venafi: { connected: false },
      hsm: { provider: "demo", configured: true },
    };
  }

  // Default: return empty object to avoid crashes
  return {};
}

export async function apiFetch<T = unknown>(path: string, init?: RequestInit): Promise<T> {
  // Demo fallback: if DEMO_MODE enabled and no API_BASE configured, do not fetch network
  if (DEMO_MODE && !API_BASE) {
    return (await demoResponse(path)) as T;
  }

  const url = path.startsWith("http") ? path : `${API_BASE}${path}`;
  const res = await fetch(url, { ...init, cache: "no-store" });

  if (res.status === 204) return undefined as unknown as T;

  const text = await res.text();
  const data = text ? JSON.parse(text) : null;

  if (!res.ok) {
    const msg =
      (data && (data.detail || data.error || data.message)) ||
      `Request failed (${res.status})`;
    throw new Error(String(msg));
  }

  return data as T;
}
