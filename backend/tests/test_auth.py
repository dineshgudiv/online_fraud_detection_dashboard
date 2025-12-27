from .conftest import login


def test_login_success_and_failure(client, create_user):
    create_user("analyst@demo", "ANALYST", "password")

    resp = client.post("/auth/login", json={"email": "analyst@demo", "password": "password"})
    assert resp.status_code == 200
    assert "access_token" in resp.json()

    bad = client.post("/auth/login", json={"email": "analyst@demo", "password": "wrong"})
    assert bad.status_code == 401


def test_viewer_cannot_mutate(client, create_user):
    create_user("viewer@demo", "VIEWER", "password")
    token = login(client, "viewer@demo", "password")

    resp = client.post(
        "/cases",
        json={"title": "Blocked case"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 403
