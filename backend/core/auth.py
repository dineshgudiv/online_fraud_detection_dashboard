"""Authentication dependencies and RBAC helpers."""

from __future__ import annotations

from typing import Callable, Iterable

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from core import config
from core.schemas import Role
from core.security import decode_access_token
from db import models
from db.session import get_db

bearer_scheme = HTTPBearer(auto_error=False)

DEMO_USER_ID = "demo-public"
DEMO_USER_EMAIL = "demo@public"


def _assign_request_user(request: Request, user: models.User) -> None:
    request.state.user_id = user.id
    request.state.user_role = _normalize_role(user.role).value


def extract_token(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = None,
) -> str | None:
    if credentials:
        return credentials.credentials
    auth_header = request.headers.get("Authorization") or request.headers.get("authorization")
    if auth_header and auth_header.lower().startswith("bearer "):
        return auth_header.split(" ", 1)[1].strip()
    return request.cookies.get("session")


def load_user_from_token(token: str, db: Session) -> models.User | None:
    try:
        payload = decode_access_token(token)
    except ValueError:
        return None
    user_id = payload.get("sub")
    if not user_id:
        return None
    return db.query(models.User).filter(models.User.id == user_id).first()


def build_demo_user() -> models.User:
    return models.User(
        id=DEMO_USER_ID,
        email=DEMO_USER_EMAIL,
        password_hash="demo",
        role=Role.READONLY.value,
    )


def _normalize_role(role_value: str) -> Role:
    try:
        role = Role(role_value)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid role") from exc
    if role == Role.VIEWER:
        return Role.READONLY
    return role


def _normalize_roles(roles: Iterable[Role]) -> set[Role]:
    normalized = set()
    for role in roles:
        normalized.add(_normalize_role(role.value))
    return normalized


PERMISSIONS: dict[str, set[Role]] = {
    "model:retrain": {Role.ADMIN},
    "security:cert:download": {Role.ADMIN, Role.ANALYST},
    "security:settings": {Role.ADMIN},
}


def get_current_user(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    db: Session = Depends(get_db),
) -> models.User:
    token = extract_token(request, credentials)
    demo_user = getattr(request.state, "demo_user", None)
    if not token:
        if demo_user:
            _assign_request_user(request, demo_user)
            return demo_user
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing credentials")
    try:
        payload = decode_access_token(token)
    except ValueError:
        if demo_user:
            _assign_request_user(request, demo_user)
            return demo_user
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
    user_id = payload.get("sub")
    if not user_id:
        if demo_user:
            _assign_request_user(request, demo_user)
            return demo_user
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
    user = db.query(models.User).filter(models.User.id == user_id).first()
    if not user:
        if demo_user:
            _assign_request_user(request, demo_user)
            return demo_user
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
    _assign_request_user(request, user)
    return user


def require_roles(*roles: Role) -> Callable:
    def dependency(user: models.User = Depends(get_current_user)) -> models.User:
        current_role = _normalize_role(user.role)
        allowed_roles = _normalize_roles(roles)
        if current_role not in allowed_roles:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Insufficient permissions")
        return user

    return dependency


def require_role(*roles: Role) -> Callable:
    return require_roles(*roles)


def require_permission(permission: str) -> Callable:
    def dependency(request: Request, user: models.User = Depends(get_current_user)) -> models.User:
        if config.DEMO_PUBLIC_READONLY and getattr(request.state, "demo_user", None):
            return user
        current_role = _normalize_role(user.role)
        allowed = PERMISSIONS.get(permission)
        if not allowed:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Permission not configured")
        normalized_allowed = _normalize_roles(allowed)
        if current_role not in normalized_allowed:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Insufficient permissions")
        return user

    return dependency


def role_for_token(role_value: str) -> str:
    return _normalize_role(role_value).value
