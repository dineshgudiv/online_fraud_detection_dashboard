import { NextResponse } from "next/server";

const API_BASE =
  process.env.API_BASE || (process.env.NODE_ENV === "development" ? "http://127.0.0.1:8001" : "");

export async function POST() {
  if (process.env.DEMO_MODE === "true") {
    return NextResponse.json(
      { ok: true, token: "demo-token", user: { email: "demo@fraudops.ai", role: "demo" } },
      { status: 200 }
    );
  }

  const request = arguments[0] as Request | undefined;
  if (!request) {
    return NextResponse.json(
      { error: "Missing request." },
      { status: 400, headers: { "Cache-Control": "no-store" } }
    );
  }

  const body = await request.json();

  if (!API_BASE) {
    return NextResponse.json(
      { error: "API_BASE not set. Set API_BASE or enable DEMO_MODE." },
      { status: 500, headers: { "Cache-Control": "no-store" } }
    );
  }

  const resp = await fetch(`${API_BASE}/api/session/login`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    cache: "no-store"
  });

  const responseBody = await resp.text();
  const response = new NextResponse(responseBody || null, { status: resp.status });
  const contentType = resp.headers.get("content-type");
  if (contentType) {
    response.headers.set("content-type", contentType);
  }
  response.headers.set("cache-control", "no-store");

  if (resp.ok && responseBody) {
    try {
      const data = JSON.parse(responseBody);
      const token = data?.access_token ?? data?.token;
      if (token) {
        response.cookies.set("session", token, {
          httpOnly: true,
          sameSite: "lax",
          secure: process.env.NODE_ENV === "production",
          path: "/"
        });
      }
    } catch {
      // Ignore non-JSON responses.
    }
  }

  return response;
}
