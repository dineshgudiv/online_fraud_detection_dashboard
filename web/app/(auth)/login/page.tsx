"use client";

import { Suspense, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";

function LoginPageContent() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const [email, setEmail] = useState("analyst@demo");
  const [password, setPassword] = useState("password");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (event: React.FormEvent) => {
    event.preventDefault();
    setLoading(true);
    setError(null);
    try {
      const resp = await fetch("/api/session/login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email, password })
      });
      if (!resp.ok) {
        const text = await resp.text();
        throw new Error(text || "Login failed");
      }
      const next = searchParams.get("next") ?? searchParams.get("from");
      const target = next && next.startsWith("/") && !next.startsWith("//") ? next : "/alerts";
      router.push(target);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Login failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="page">
      <section className="hero">
        <span className="badge">Secure access</span>
        <h1>Sign in</h1>
        <p>Use your analyst credentials to access the fraud console.</p>
        {error && <div className="error">{error}</div>}
        <form className="form" onSubmit={handleSubmit}>
          <input
            placeholder="Email"
            value={email}
            onChange={(event) => setEmail(event.target.value)}
          />
          <input
            type="password"
            placeholder="Password"
            value={password}
            onChange={(event) => setPassword(event.target.value)}
          />
          <button className="button" type="submit" disabled={loading}>
            {loading ? "Signing in..." : "Sign in"}
          </button>
        </form>
      </section>
    </main>
  );
}

export default function LoginPage() {
  return (
    <Suspense
      fallback={
        <main className="page">
          <section className="hero">
            <div className="muted">Loading login...</div>
          </section>
        </main>
      }
    >
      <LoginPageContent />
    </Suspense>
  );
}
