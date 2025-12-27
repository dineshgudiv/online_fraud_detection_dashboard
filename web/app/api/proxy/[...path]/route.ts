import { cookies } from "next/headers";
import { NextResponse } from "next/server";

const API_BASE = process.env.API_BASE ?? "http://127.0.0.1:8001";

async function proxy(request: Request, context: { params: { path: string[] } }) {
  const path = context.params.path.join("/");
  const url = `${API_BASE}/${path}${request.url.includes("?") ? request.url.slice(request.url.indexOf("?")) : ""}`;

  const headers = new Headers();
  const contentType = request.headers.get("content-type");
  if (contentType) {
    headers.set("content-type", contentType);
  }
  const token = cookies().get("session")?.value;
  if (token) {
    headers.set("Authorization", `Bearer ${token}`);
  }

  const method = request.method;
  const body = method === "GET" || method === "HEAD" ? undefined : await request.arrayBuffer();

  const resp = await fetch(url, {
    method,
    headers,
    body: body ? Buffer.from(body) : undefined
  });

  const responseBody = await resp.arrayBuffer();
  const response = new NextResponse(responseBody, { status: resp.status });
  resp.headers.forEach((value, key) => {
    response.headers.set(key, value);
  });
  return response;
}

export async function GET(request: Request, context: { params: { path: string[] } }) {
  return proxy(request, context);
}

export async function POST(request: Request, context: { params: { path: string[] } }) {
  return proxy(request, context);
}

export async function PATCH(request: Request, context: { params: { path: string[] } }) {
  return proxy(request, context);
}

export async function PUT(request: Request, context: { params: { path: string[] } }) {
  return proxy(request, context);
}

export async function DELETE(request: Request, context: { params: { path: string[] } }) {
  return proxy(request, context);
}
