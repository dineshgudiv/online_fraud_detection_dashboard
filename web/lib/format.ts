export function formatDateUTC(v?: string | number | Date | null) {
  if (v === undefined || v === null) return "—";
  const d = new Date(v);
  if (isNaN(d.getTime())) return String(v);
  return d.toISOString().replace("T", " ").replace("Z", " UTC");
}

export function formatNumber(n?: number | null, maximumFractionDigits: number = 2) {
  if (n === undefined || n === null) return "—";
  if (!Number.isFinite(n)) return String(n);
  return n.toLocaleString(undefined, { maximumFractionDigits });
}
