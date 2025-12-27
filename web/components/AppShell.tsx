import Link from "next/link";
import { ReactNode } from "react";

import DatasetsSidebar from "../app/_components/DatasetsSidebar";
import { DEMO_PUBLIC_READONLY, DEMO_READONLY_BANNER } from "../lib/demo";

type NavItem =
  | { type: "link"; href: string; label: string }
  | { type: "label"; label: string };

const NAV_ITEMS: NavItem[] = [
  { type: "link", href: "/alerts", label: "Alerts" },
  { type: "link", href: "/review-queue", label: "Review Queue" },
  { type: "link", href: "/dataset", label: "Dataset" },
  { type: "link", href: "/cases", label: "Cases" },
  { type: "link", href: "/audit", label: "Audit" },
  { type: "link", href: "/search", label: "Search" },
  { type: "link", href: "/security", label: "Security Center" },
  { type: "link", href: "/geo-fraud-map", label: "Geo Fraud Map" },
  { type: "link", href: "/manual-review", label: "Manual Review" },
  { type: "link", href: "/model-lab", label: "Model Lab" },
  { type: "link", href: "/pipeline-view", label: "Pipeline View" },
  { type: "label", label: "Settings" },
  { type: "link", href: "/settings/policy", label: "Policy" },
  { type: "link", href: "/settings/users", label: "Users" }
];

export default function AppShell({ children }: { children: ReactNode }) {
  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <span className="badge">Fraud Ops</span>
          <h1>Risk Console</h1>
          <p>Operational view for fraud triage.</p>
        </div>
        <nav className="nav">
          {NAV_ITEMS.map((item) =>
            item.type === "label" ? (
              <div key={`label-${item.label}`} className="nav-label">
                {item.label}
              </div>
            ) : (
              <Link key={item.href} href={item.href} className="nav-link">
                {item.label}
              </Link>
            )
          )}
        </nav>
        <div className="sidebar-section">
          <DatasetsSidebar />
        </div>
        <div className="sidebar-footer">
          <Link href="/login" className="ghost-link">
            Switch user
          </Link>
        </div>
      </aside>
      <div className="main">
        <header className="topbar">
          <div>
            <h2>Fraud Operations</h2>
            <span>Live ops console</span>
          </div>
          <div className="status-pill">API: connected</div>
        </header>
        {DEMO_PUBLIC_READONLY && (
          <div className="demo-banner">{DEMO_READONLY_BANNER}</div>
        )}
        <main className="flex-1 min-w-0 overflow-x-hidden overflow-y-auto">
          {children}
        </main>
      </div>
    </div>
  );
}

