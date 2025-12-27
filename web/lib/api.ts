export const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE ||
  process.env.API_BASE ||
  "http://127.0.0.1:8001";

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

export async function apiFetch<T = unknown>(path: string, init?: RequestInit): Promise<T> {
  const url = path.startsWith("http") ? path : `${API_BASE}${path}`;
  const res = await fetch(url, { ...init, cache: "no-store" });

  // If endpoint returns no body (204), avoid JSON parse errors
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
