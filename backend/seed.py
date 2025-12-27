"""Seed demo users and alerts for local development."""

from __future__ import annotations

import os
import random
import uuid
from datetime import datetime, timedelta

from sqlalchemy.orm import Session

from core.security import hash_password
from db import models
from db.session import SessionLocal

DEMO_USERS = [
    ("admin@demo", "ADMIN"),
    ("analyst@demo", "ANALYST"),
    ("viewer@demo", "VIEWER"),
]


def seed_users(db: Session) -> None:
    for email, role in DEMO_USERS:
        exists = db.query(models.User).filter(models.User.email == email).first()
        if exists:
            continue
        user = models.User(email=email, role=role, password_hash=hash_password("password"))
        db.add(user)
    db.commit()


def seed_alerts(db: Session, count: int = 50) -> None:
    if db.query(models.Alert).count() > 0:
        return

    for idx in range(count):
        amount = round(random.uniform(50, 3000), 2)
        risk_score = min(0.99, max(0.05, amount / 3000 + random.uniform(-0.05, 0.15)))
        risk_level = "CRITICAL" if risk_score > 0.85 else "HIGH" if risk_score > 0.7 else "MEDIUM" if risk_score > 0.45 else "LOW"
        status = random.choice(["NEW", "TRIAGED", "INVESTIGATING"])
        created_at = datetime.utcnow() - timedelta(minutes=random.randint(0, 720))
        txn_id = f"TX-{uuid.uuid4().hex[:10]}"

        alert = models.Alert(
            transaction_id=txn_id,
            created_at=created_at,
            risk_score=risk_score,
            risk_level=risk_level,
            status=status,
            merchant_id=f"M{random.randint(100, 999)}",
            user_id=f"U{random.randint(1000, 9999)}",
            amount=amount,
            currency="USD",
            decision="REJECTED" if risk_score > 0.7 else "APPROVED",
            reason="Seeded demo alert",
            features_json={"amount": amount, "seed": True},
        )
        db.add(alert)
        db.flush()

        audit = models.AuditLog(
            timestamp=created_at,
            actor="seed",
            action="SCORE",
            transaction_id=txn_id,
            model_name="demo-linear",
            model_version="seed",
            score=risk_score,
            decision=alert.decision,
            risk_factors_json=["seeded"],
            alert_id=alert.id,
        )
        db.add(audit)

    db.commit()


def seed_cases(db: Session, count: int = 5) -> None:
    if db.query(models.Case).count() > 0:
        return
    alerts = db.query(models.Alert).limit(count).all()
    for alert in alerts:
        case = models.Case(
            status="IN_REVIEW",
            title=f"Investigate {alert.transaction_id}",
            assigned_to="analyst@demo",
            alert_id=alert.id,
            transaction_id=alert.transaction_id,
            user_id=alert.user_id,
            risk_level=alert.risk_level,
            risk_score=alert.risk_score,
            notes="Seeded case",
            notes_history=[
                {
                    "timestamp": datetime.utcnow().isoformat(),
                    "author": "seed",
                    "note": "Seeded case note",
                }
            ],
        )
        db.add(case)
        db.flush()
        alert.case_id = case.id
        alert.status = "INVESTIGATING"
    db.commit()


def main() -> None:
    if os.getenv("SEED_DEMO_DATA", "true").lower() != "true":
        return

    db = SessionLocal()
    try:
        seed_users(db)
        seed_alerts(db)
        seed_cases(db)
    finally:
        db.close()


if __name__ == "__main__":
    main()
