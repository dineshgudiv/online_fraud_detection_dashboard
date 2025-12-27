import Link from "next/link";

export default function Home() {
  return (
    <main className="page">
      <section className="hero">
        <span className="badge">Fraud Ops - Demo</span>
        <h1>Fraud Operations Console</h1>
        <p>
          A Vercel-native control plane for fraud alerts, case management, and
          analyst feedback. This page is a placeholder while the full workflow
          ships.
        </p>
        <div className="actions">
          <Link className="primary" href="/alerts">
            View alerts
          </Link>
          <Link className="ghost" href="/login">
            Sign in
          </Link>
        </div>
      </section>

      <section className="grid">
        <article className="card">
          <h3>Alerts and triage</h3>
          <p>Queue high-risk transactions, filter by risk, and route to review.</p>
        </article>
        <article className="card">
          <h3>Cases and evidence</h3>
          <p>Group alerts into cases with notes, status, and assignments.</p>
        </article>
        <article className="card">
          <h3>Audit and feedback</h3>
          <p>Every score, decision, and reviewer action is tracked for audit.</p>
        </article>
      </section>
    </main>
  );
}

