"""Audit trail helpers for compliance logging."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from fastapi import Request
from sqlalchemy.orm import Session

from db import models


def log_audit_event(
    db: Session,
    request: Request,
    *,
    action: str,
    resource_type: str,
    resource_id: Optional[str] = None,
    user: Optional[models.User] = None,
    actor_email: Optional[str] = None,
    metadata: Optional[dict[str, Any]] = None,
) -> models.AuditLog:
    actor = actor_email or (user.email if user else None)
    entry = models.AuditLog(
        timestamp=datetime.utcnow(),
        actor=actor,
        user_id=user.id if user else None,
        role=user.role if user else None,
        action=action,
        resource_type=resource_type,
        resource_id=resource_id,
        ip=request.client.host if request.client else None,
        user_agent=request.headers.get("user-agent"),
        correlation_id=getattr(request.state, "request_id", None),
        metadata_json=metadata or {},
    )
    db.add(entry)
    return entry
