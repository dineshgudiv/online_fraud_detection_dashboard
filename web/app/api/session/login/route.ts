import { NextResponse } from "next/server";

const API_BASE = process.env.API_BASE ?? "http://127.0.0.1:8001";

export async function POST(request: Request) {
  const body = await request.json();

  const resp = await fetch(`${API_BASE}/auth/login`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body)
  });

  if (!resp.ok) {
    const text = await resp.text();
    return new NextResponse(text || "Login failed", { status: resp.status });
  }

  const data = await resp.json();
  const token = data.access_token;
  const response = NextResponse.json({ ok: true });
  response.cookies.set("session", token, {
    httpOnly: true,
    sameSite: "lax",
    secure: process.env.NODE_ENV === "production",
    path: "/"
  });
  return response;
}
