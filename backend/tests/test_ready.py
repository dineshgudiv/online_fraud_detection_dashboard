from fastapi.testclient import TestClient

from api.main import app
from db.session import get_db


def test_ready_handles_db_failure():
    def bad_db():
        raise Exception("db down")

    app.dependency_overrides[get_db] = bad_db
    with TestClient(app) as client:
        resp = client.get("/ready")
        assert resp.status_code == 503
    app.dependency_overrides.clear()
