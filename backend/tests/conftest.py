# ruff: noqa: E402
import os
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import sessionmaker

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from api.main import app
from core.security import hash_password
from db import models
from db.base import Base
from db.session import get_db

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+psycopg://fraud:fraud@localhost:5432/fraud")
connect_args = {"connect_timeout": 2} if DATABASE_URL.startswith("postgresql") else {}

engine = create_engine(DATABASE_URL, connect_args=connect_args)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


@pytest.fixture(scope="session", autouse=True)
def setup_db():
    try:
        with engine.connect():
            pass
    except OperationalError as exc:
        pytest.skip(f"Database not available: {exc}")
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)


@pytest.fixture()
def db_session():
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture()
def client(db_session):
    def override_get_db():
        try:
            yield db_session
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


@pytest.fixture()
def create_user(db_session):
    def _create(email: str, role: str, password: str = "password"):
        user = models.User(
            email=email,
            password_hash=hash_password(password),
            role=role,
        )
        db_session.add(user)
        db_session.commit()
        db_session.refresh(user)
        return user

    return _create


def login(client, email: str, password: str) -> str:
    resp = client.post("/auth/login", json={"email": email, "password": password})
    assert resp.status_code == 200
    return resp.json()["access_token"]
