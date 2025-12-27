"""Runtime configuration for the fraud ops backend."""

from __future__ import annotations

import os


def _split_allowlist(raw: str) -> list[str]:
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


CORS_ALLOWLIST = _split_allowlist(os.getenv("CORS_ALLOWLIST", "http://localhost:3000,http://localhost:5173"))

JWT_SECRET = os.getenv("JWT_SECRET", "dev-secret")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
JWT_EXPIRES_MINUTES = int(os.getenv("JWT_EXPIRES_MINUTES", "60"))

RATE_LIMIT_LOGIN = int(os.getenv("RATE_LIMIT_LOGIN", "5"))
RATE_LIMIT_SCORE = int(os.getenv("RATE_LIMIT_SCORE", "30"))

SENTRY_DSN = os.getenv("SENTRY_DSN", "")


def _split_csv(raw: str) -> list[str]:
    return [value.strip() for value in raw.split(",") if value.strip()]


def _env_flag(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
RQ_QUEUES = _split_csv(os.getenv("RQ_QUEUES", "retrain,default"))
RQ_DEFAULT_QUEUE = os.getenv("RQ_DEFAULT_QUEUE", RQ_QUEUES[0] if RQ_QUEUES else "retrain")

DEMO_PUBLIC_READONLY = _env_flag("DEMO_PUBLIC_READONLY", "false")
