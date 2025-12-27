"""Security Center endpoints."""

from __future__ import annotations

import os
import re
from datetime import datetime, timedelta, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import PlainTextResponse
from sqlalchemy.orm import Session

from core.audit import log_audit_event
from core.auth import require_permission, require_roles
from core.schemas import Role
from db import models
from db.session import get_db

router = APIRouter(prefix="/security", tags=["Security"])

_TRUTHY = ("1", "true", "yes")
_FILENAME_CLEAN = re.compile(r"[^a-zA-Z0-9._-]+")

_DEMO_PEM = """-----BEGIN CERTIFICATE-----
MIIDrzCCApegAwIBAgIUeVYcL4LkOqg1kYt6bH2bQ4dWZ2QwDQYJKoZIhvcNAQEL
BQAwVDELMAkGA1UEBhMCVVMxETAPBgNVBAoMCEZyYXVkT3BzMR0wGwYDVQQDDBRG
cmF1ZE9wcyBJbnRlcm5hbCBDQTAeFw0yNDAxMDEwMDAwMDBaFw0yNTAxMDEwMDAw
MDBaME8xCzAJBgNVBAYTAlVTMREwDwYDVQQKDAhGcmF1ZE9wczEYMBYGA1UEAwwP
YXBpLmZyYXVkLmxvY2FsMIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEA
xQ5q8mE0c3zShW5q5n9M1sQ8r6hZ2w0fE4Xb0q1rQGxO9k5d3gK7mQv7dRwx1m8S
d2p5Z1q5gYp3Hq1g6gQ2Gk4r3N8pYJw0QJf8bK2sI5nY0G8J4f3rC2b9GQ7kX1yH
wQIDAQABo2gwZjAfBgNVHSMEGDAWgBTrp5dVqJ2X8wz9d8mP1eYgB1GgATAPBgNV
HRMBAf8EBTADAQH/MB0GA1UdDgQWBBQ5a8Tz4xjXGqfKjL3p1tEwN0X2KDAOBgNV
HQ8BAf8EBAMCBaAwDQYJKoZIhvcNAQELBQADggEBADYhYlGqkHchj7Dg4X2uZx0f
2z0G0y6lE6bDdb7w6Jxk3B2g6rF9sK5d3Q0uX8g3W5c5g6mE0b8k3mD5yJc2y6xg
-----END CERTIFICATE-----"""


def _env_enabled(name: str, default: str) -> bool:
    return os.getenv(name, default).lower() in _TRUTHY


def _demo_enabled() -> bool:
    return _env_enabled("SECURITY_DEMO_MODE", "true") or _env_enabled("MODEL_DEMO_MODE", "false")


def _safe_filename(value: str) -> str:
    cleaned = _FILENAME_CLEAN.sub("_", value).strip("._")
    return cleaned or "certificate"


def _parse_dt(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _days_to_expiry(not_after: str) -> int | None:
    try:
        expires = _parse_dt(not_after)
    except Exception:
        return None
    now = datetime.now(timezone.utc)
    return int((expires - now).total_seconds() / 86400)


def _key_info(public_key_algorithm: str | None) -> tuple[str | None, int | None]:
    if not public_key_algorithm:
        return None, None
    lower = public_key_algorithm.lower()
    size = None
    for token in lower.replace("-", " ").split():
        if token.isdigit():
            size = int(token)
            break
    if "rsa" in lower:
        return "RSA", size
    if "ecdsa" in lower or "ec" in lower:
        return "EC", size
    return public_key_algorithm, size


def _trust_state(cert: dict[str, Any]) -> tuple[str, str | None]:
    status = cert.get("status")
    if status == "EXPIRED":
        return "EXPIRED", "Certificate is expired"
    if status == "REVOKED":
        return "CHAIN_INCOMPLETE", "Certificate is revoked"

    sig_alg = (cert.get("signature_algorithm") or "").lower()
    pub_alg = (cert.get("public_key_algorithm") or "").lower()
    if "sha1" in sig_alg or "md5" in sig_alg or "1024" in pub_alg:
        return "WEAK_ALGO", "Weak signature or key algorithm"

    if cert.get("issuer") == cert.get("common_name"):
        return "SELF_SIGNED", "Certificate is self-signed"

    chain = cert.get("chain") or []
    if len(chain) < 3:
        return "CHAIN_INCOMPLETE", "Certification path is incomplete"

    return "VALID", None


def _status_for(not_after: str, revoked: bool) -> str:
    if revoked:
        return "REVOKED"
    now = datetime.now(timezone.utc)
    try:
        if _parse_dt(not_after) < now:
            return "EXPIRED"
    except Exception:
        return "ACTIVE"
    return "ACTIVE"


def _demo_chain(common_name: str) -> list[dict[str, Any]]:
    return [
        {"label": "FraudOps Root CA", "type": "Root CA"},
        {"label": "FraudOps Issuing CA", "type": "Intermediate CA"},
        {"label": common_name, "type": "Leaf Certificate", "current": True},
    ]


def _demo_certs() -> list[dict[str, Any]]:
    now = datetime.now(timezone.utc)
    certs = [
        {
            "id": "cert_demo_001",
            "common_name": "api.fraud.local",
            "issuer": "FraudOps Internal CA",
            "not_before": (now - timedelta(days=30)).isoformat(),
            "not_after": (now + timedelta(days=80)).isoformat(),
            "serial": "01A1B2C3D4",
            "signature_algorithm": "sha256WithRSAEncryption",
            "public_key_algorithm": "RSA 2048-bit",
            "san": {"dns": ["api.fraud.local", "api.internal.local"], "ip": ["10.20.30.11"]},
            "key_usage": ["Digital Signature", "Key Encipherment"],
            "enhanced_key_usage": ["Server Authentication", "Client Authentication"],
            "revoked": False,
        },
        {
            "id": "cert_demo_002",
            "common_name": "grafana.fraud.local",
            "issuer": "FraudOps Internal CA",
            "not_before": (now - timedelta(days=100)).isoformat(),
            "not_after": (now + timedelta(days=10)).isoformat(),
            "serial": "0FFEEDDCCB",
            "signature_algorithm": "sha256WithRSAEncryption",
            "public_key_algorithm": "RSA 2048-bit",
            "san": {"dns": ["grafana.fraud.local"], "ip": []},
            "key_usage": ["Digital Signature", "Key Encipherment"],
            "enhanced_key_usage": ["Server Authentication"],
            "revoked": False,
        },
        {
            "id": "cert_demo_003",
            "common_name": "payments.fraud.local",
            "issuer": "FraudOps Internal CA",
            "not_before": (now - timedelta(days=420)).isoformat(),
            "not_after": (now - timedelta(days=15)).isoformat(),
            "serial": "0A11B22C33D4",
            "signature_algorithm": "sha256WithRSAEncryption",
            "public_key_algorithm": "RSA 2048-bit",
            "san": {"dns": ["payments.fraud.local"], "ip": ["10.20.30.20"]},
            "key_usage": ["Digital Signature", "Key Encipherment"],
            "enhanced_key_usage": [],
            "revoked": True,
        },
    ]
    for cert in certs:
        cert["status"] = _status_for(cert["not_after"], cert["revoked"])
        cert["days_to_expiry"] = _days_to_expiry(cert["not_after"])
        key_algorithm, key_size = _key_info(cert.get("public_key_algorithm"))
        cert["key_algorithm"] = key_algorithm
        cert["key_size"] = key_size
        trust_state, trust_reason = _trust_state(cert)
        cert["trust_state"] = trust_state
        cert["trust_reason"] = trust_reason
        cert["pem"] = _DEMO_PEM
        cert["chain"] = _demo_chain(cert["common_name"])
    return certs


def _cert_summary(cert: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": cert["id"],
        "common_name": cert["common_name"],
        "issuer": cert["issuer"],
        "not_before": cert["not_before"],
        "not_after": cert["not_after"],
        "serial": cert["serial"],
        "status": cert["status"],
        "days_to_expiry": cert.get("days_to_expiry"),
        "trust_state": cert.get("trust_state"),
    }


def _get_cert_detail(cert_id: str) -> dict[str, Any]:
    if not _demo_enabled():
        raise HTTPException(status_code=404, detail="Certificate not found")
    for cert in _demo_certs():
        if cert["id"] == cert_id:
            return cert
    raise HTTPException(status_code=404, detail="Certificate not found")


@router.get("/pki/certs")
def list_certs(
    _user: models.User = Depends(require_roles(Role.READONLY, Role.ANALYST, Role.ADMIN)),
) -> dict[str, Any]:
    certs = _demo_certs() if _demo_enabled() else []
    summaries = [_cert_summary(cert) for cert in certs]
    return {"items": summaries, "count": len(summaries)}


@router.get("/pki/certs/expiring")
def expiring_certs(
    days: int = 30,
    _user: models.User = Depends(require_roles(Role.READONLY, Role.ANALYST, Role.ADMIN)),
) -> dict[str, Any]:
    certs = _demo_certs() if _demo_enabled() else []
    cutoff = datetime.now(timezone.utc) + timedelta(days=days)

    expiring = []
    for cert in certs:
        not_after = _parse_dt(cert["not_after"])
        if not_after <= cutoff:
            expiring.append(_cert_summary(cert))

    return {"days": days, "items": expiring, "count": len(expiring)}


@router.get("/pki/certs/{cert_id}")
def cert_detail(
    cert_id: str,
    request: Request,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(require_roles(Role.READONLY, Role.ANALYST, Role.ADMIN)),
) -> dict[str, Any]:
    cert = _get_cert_detail(cert_id)
    log_audit_event(
        db,
        request,
        action="CERT_VIEWED",
        resource_type="certificate",
        resource_id=cert_id,
        user=current_user,
        metadata={"common_name": cert.get("common_name"), "status": cert.get("status")},
    )
    db.commit()
    return {
        **_cert_summary(cert),
        "signature_algorithm": cert.get("signature_algorithm"),
        "public_key_algorithm": cert.get("public_key_algorithm"),
        "key_algorithm": cert.get("key_algorithm"),
        "key_size": cert.get("key_size"),
        "san": cert.get("san"),
        "key_usage": cert.get("key_usage"),
        "enhanced_key_usage": cert.get("enhanced_key_usage"),
        "pem": cert.get("pem"),
        "chain": cert.get("chain"),
        "trust_state": cert.get("trust_state"),
        "trust_reason": cert.get("trust_reason"),
    }


@router.get("/pki/certs/{cert_id}/pem")
def cert_pem(
    cert_id: str,
    _user: models.User = Depends(require_roles(Role.READONLY, Role.ANALYST, Role.ADMIN)),
) -> PlainTextResponse:
    cert = _get_cert_detail(cert_id)
    return PlainTextResponse(cert.get("pem") or "", media_type="text/plain")


@router.get("/pki/certs/{cert_id}/download")
def cert_download(
    cert_id: str,
    request: Request,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(require_permission("security:cert:download")),
) -> Response:
    cert = _get_cert_detail(cert_id)
    log_audit_event(
        db,
        request,
        action="CERT_DOWNLOADED",
        resource_type="certificate",
        resource_id=cert_id,
        user=current_user,
        metadata={"common_name": cert.get("common_name"), "status": cert.get("status")},
    )
    db.commit()
    filename = _safe_filename(cert.get("common_name") or cert_id)
    headers = {"Content-Disposition": f'attachment; filename="{filename}.crt"'}
    return Response(
        content=cert.get("pem") or "",
        media_type="application/x-x509-ca-cert",
        headers=headers,
    )


@router.get("/venafi/status")
def venafi_status(
    _user: models.User = Depends(require_roles(Role.READONLY, Role.ANALYST, Role.ADMIN)),
) -> dict[str, Any]:
    if _demo_enabled():
        return {
            "connected": True,
            "last_sync": (datetime.now(timezone.utc) - timedelta(minutes=12)).isoformat(),
            "profile": "venafi-demo",
            "health": "OK",
        }
    return {"connected": False, "last_sync": None, "profile": None, "health": "NOT_CONNECTED"}


@router.get("/hsm/provider")
def hsm_provider(
    _user: models.User = Depends(require_roles(Role.READONLY, Role.ANALYST, Role.ADMIN)),
) -> dict[str, Any]:
    if _demo_enabled():
        return {
            "configured": True,
            "provider": "SoftHSM (dev)",
            "key_count": 6,
            "rotation_policy": "90d",
            "health": "OK",
        }
    return {"configured": False, "provider": None, "key_count": 0, "rotation_policy": None, "health": "NOT_CONFIGURED"}


@router.get("/settings")
def security_settings(_user=Depends(require_permission("security:settings"))) -> dict[str, Any]:
    return {
        "demo_mode": _demo_enabled(),
        "pki_inventory": "enabled" if _demo_enabled() else "not_configured",
        "venafi_integration": "connected" if _demo_enabled() else "not_connected",
        "hsm_provider": "soft_hsm" if _demo_enabled() else "not_configured",
    }
