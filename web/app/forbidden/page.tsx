"use client";

import Link from "next/link";
import { useSearchParams } from "next/navigation";

export default function ForbiddenPage() {
  const params = useSearchParams();
  const from = params.get("from");

  return (
    <main className="page">
      <section className="hero">
        <span className="badge">403</span>
        <h1>Not authorized</h1>
        <p>
          You do not have permission to access this area.
          {from ? ` Requested route: ${from}` : ""}
        </p>
        <div className="actions">
          <Link className="primary" href="/alerts">
            Back to console
          </Link>
          <Link className="ghost" href="/login">
            Switch user
          </Link>
        </div>
      </section>
    </main>
  );
}
