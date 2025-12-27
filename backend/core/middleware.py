"""ASGI middleware for request IDs and logging."""

from __future__ import annotations

import json
import logging
import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from core import config
from core.auth import build_demo_user, extract_token, load_user_from_token
from core.schemas import ErrorResponse
from db.session import SessionLocal

logger = logging.getLogger("fraud_ops")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


class RequestIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        request_id = request.headers.get("X-Request-Id") or str(uuid.uuid4())
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-Id"] = request_id
        return response


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        start = time.perf_counter()
        response = await call_next(request)
        duration_ms = round((time.perf_counter() - start) * 1000, 2)
        payload = {
            "event": "request",
            "method": request.method,
            "path": request.url.path,
            "status": response.status_code,
            "duration_ms": duration_ms,
            "request_id": getattr(request.state, "request_id", None),
            "user_id": getattr(request.state, "user_id", None),
        }
        logger.info(json.dumps(payload))
        return response


class DemoPublicReadOnlyMiddleware(BaseHTTPMiddleware):
    _write_methods = {"POST", "PUT", "PATCH", "DELETE"}
    _read_methods = {"GET", "HEAD"}
    _public_read_prefixes = (
        "/alerts",
        "/audit",
        "/datasets",
        "/docs",
        "/metrics",
        "/redoc",
        "/security",
    )
    _public_read_exact = {
        "/",
        "/health",
        "/openapi.json",
        "/ready",
    }
    _write_exempt = {"/auth/login"}

    async def dispatch(self, request: Request, call_next) -> Response:
        if not config.DEMO_PUBLIC_READONLY or request.method == "OPTIONS":
            return await call_next(request)

        path = request.url.path
        token = extract_token(request)
        user = None
        if token:
            db = SessionLocal()
            try:
                user = load_user_from_token(token, db)
            finally:
                db.close()
        is_authenticated = user is not None

        if request.method in self._write_methods and not is_authenticated:
            if path in self._write_exempt:
                return await call_next(request)
            payload = ErrorResponse(
                detail="DEMO_READ_ONLY",
                code="demo_read_only",
                request_id=getattr(request.state, "request_id", None),
            )
            return JSONResponse(status_code=403, content=payload.dict())

        if request.method in self._read_methods and not is_authenticated:
            if self._is_public_read(path):
                request.state.demo_user = build_demo_user()

        return await call_next(request)

    def _is_public_read(self, path: str) -> bool:
        if path in self._public_read_exact:
            return True
        return any(path == prefix or path.startswith(f"{prefix}/") for prefix in self._public_read_prefixes)

