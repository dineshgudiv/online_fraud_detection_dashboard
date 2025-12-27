import { NextResponse } from "next/server";
import type { NextRequest } from "next/server";

const DEMO_PUBLIC_READONLY =
  process.env.NEXT_PUBLIC_DEMO_PUBLIC_READONLY === "true";
const PUBLIC_PATHS = new Set(["/", "/login"]);
const DEMO_PUBLIC_PREFIXES = ["/alerts", "/audit", "/dataset", "/security"];

const isPublicPath = (pathname: string) => {
  if (PUBLIC_PATHS.has(pathname)) {
    return true;
  }
  if (!DEMO_PUBLIC_READONLY) {
    return false;
  }
  return DEMO_PUBLIC_PREFIXES.some(
    (prefix) => pathname === prefix || pathname.startsWith(`${prefix}/`)
  );
};

function applySecurityHeaders(response: NextResponse) {
  response.headers.set("X-Frame-Options", "DENY");
  response.headers.set("X-Content-Type-Options", "nosniff");
  response.headers.set("Referrer-Policy", "strict-origin-when-cross-origin");
  response.headers.set("Permissions-Policy", "camera=(), microphone=(), geolocation=()");
  response.headers.set("Cross-Origin-Opener-Policy", "same-origin");
  response.headers.set("Cross-Origin-Resource-Policy", "same-origin");
  return response;
}

export function middleware(request: NextRequest) {
  const { pathname, search } = request.nextUrl;
  if (!pathname.startsWith("/api") && !isPublicPath(pathname)) {
    const session = request.cookies.get("session")?.value;
    if (!session) {
      const loginUrl = request.nextUrl.clone();
      loginUrl.pathname = "/login";
      loginUrl.searchParams.set("next", `${pathname}${search}`);
      return applySecurityHeaders(NextResponse.redirect(loginUrl));
    }
  }
  return applySecurityHeaders(NextResponse.next());
}

export const config = {
  matcher: ["/((?!_next/static|_next/image|favicon.ico).*)"]
};
