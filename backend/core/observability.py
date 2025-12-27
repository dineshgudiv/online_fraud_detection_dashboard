"""Observability hooks (Sentry, etc.)."""

from __future__ import annotations

import os

import sentry_sdk
from sentry_sdk.integrations.fastapi import FastApiIntegration

from core.config import SENTRY_DSN


def init_sentry() -> None:
    if not SENTRY_DSN:
        return
    sentry_sdk.init(
        dsn=SENTRY_DSN,
        integrations=[FastApiIntegration()],
        traces_sample_rate=float(os.getenv("SENTRY_TRACES_SAMPLE_RATE", "0.1")),
    )

