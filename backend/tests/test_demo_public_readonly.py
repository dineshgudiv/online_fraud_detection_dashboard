from datetime import datetime

from .conftest import login
from core import config
from core.schemas import Role
from db import models


def test_demo_public_readonly_allows_get_datasets(client, monkeypatch):
    monkeypatch.setattr(config, "DEMO_PUBLIC_READONLY", True)

    resp = client.get("/datasets")

    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


def test_demo_public_readonly_blocks_write_without_auth(client, monkeypatch):
    monkeypatch.setattr(config, "DEMO_PUBLIC_READONLY", True)

    resp = client.post("/datasets")

    assert resp.status_code == 403
    payload = resp.json()
    assert payload["detail"] == "DEMO_READ_ONLY"
    assert payload["code"] == "demo_read_only"


def test_demo_public_readonly_allows_authenticated_write(client, db_session, create_user, monkeypatch):
    monkeypatch.setattr(config, "DEMO_PUBLIC_READONLY", True)

    create_user("admin@demo", Role.ADMIN.value)
    dataset = models.DatasetVersion(
        original_filename="demo.csv",
        stored_path="demo.csv",
        size_bytes=1,
        uploaded_at=datetime.utcnow(),
        schema_json=[],
        row_count=0,
        is_active=False,
    )
    db_session.add(dataset)
    db_session.commit()
    db_session.refresh(dataset)

    token = login(client, "admin@demo", "password")
    resp = client.post(
        "/datasets/set-active",
        json={"version_id": dataset.id},
        headers={"Authorization": f"Bearer {token}"},
    )

    assert resp.status_code == 200
