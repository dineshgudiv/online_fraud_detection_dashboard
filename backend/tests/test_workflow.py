from .conftest import login


def _headers(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def test_score_alert_case_feedback_flow(client, create_user):
    create_user("analyst@demo", "ANALYST", "password")
    token = login(client, "analyst@demo", "password")

    payload = {
        "user_id": "U100",
        "merchant_id": "M99",
        "amount": 2500,
        "currency": "USD",
        "transaction_type": "withdrawal",
        "category": "crypto",
        "location": "CN",
        "device_id": "D77",
    }

    score_resp = client.post("/score", json=payload, headers=_headers(token))
    assert score_resp.status_code == 200
    score = score_resp.json()
    assert score["transaction_id"]
    assert score["alert_id"]

    alerts_resp = client.get(
        f"/alerts?search={score['transaction_id']}&page=1&page_size=5",
        headers=_headers(token),
    )
    assert alerts_resp.status_code == 200
    alerts = alerts_resp.json()
    assert alerts["total"] >= 1

    audit_resp = client.get(
        f"/audit?transaction_id={score['transaction_id']}&page=1&page_size=5",
        headers=_headers(token),
    )
    assert audit_resp.status_code == 200
    assert audit_resp.json()["total"] >= 1

    case_resp = client.post(
        "/cases",
        json={"title": "Investigate alert", "alert_id": score["alert_id"]},
        headers=_headers(token),
    )
    assert case_resp.status_code == 200
    case_id = case_resp.json()["id"]

    update_resp = client.patch(
        f"/cases/{case_id}",
        json={"status": "IN_REVIEW", "note": "Initial triage"},
        headers=_headers(token),
    )
    assert update_resp.status_code == 200

    feedback_resp = client.post(
        "/feedback",
        json={
            "alert_id": score["alert_id"],
            "case_id": case_id,
            "label": "fraud",
            "notes": "Confirmed suspicious pattern",
        },
        headers=_headers(token),
    )
    assert feedback_resp.status_code == 200

    case_check = client.get(f"/cases/{case_id}", headers=_headers(token))
    assert case_check.status_code == 200
    assert case_check.json()["status"] == "RESOLVED"
