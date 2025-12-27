"use client";

import { ReactNode } from "react";

type StatCardProps = {
  label: string;
  value: ReactNode;
  hint?: string;
};

export default function StatCard({ label, value, hint }: StatCardProps) {
  return (
    <div className="card stat-card">
      <div className="stat-label">{label}</div>
      <div className="stat-value">{value}</div>
      {hint && <div className="stat-hint">{hint}</div>}
    </div>
  );
}
