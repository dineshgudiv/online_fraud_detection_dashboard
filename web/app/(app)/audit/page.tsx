import { Suspense } from "react";

import AuditClient from "./AuditClient";

export default function AuditPage() {
  return (
    <section className="section">
      <Suspense fallback={<div className="empty">Loading audit log...</div>}>
        <AuditClient />
      </Suspense>
    </section>
  );
}
