from fastapi.testclient import TestClient
import pytest

import app.main as main
from app.main import app

client = TestClient(app)


@pytest.fixture(autouse=True)
def isolate_instance_token_environment(monkeypatch):
    monkeypatch.delenv("CITEFOLD_BACKEND_INSTANCE_TOKEN", raising=False)
    monkeypatch.delenv("VCC_BACKEND_INSTANCE_TOKEN", raising=False)


def test_health_check():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "application_id": "citefold",
        "api_version": 1,
        "instance_token": None,
    }


def test_health_exposes_the_owned_instance_token(monkeypatch):
    monkeypatch.setenv("CITEFOLD_BACKEND_INSTANCE_TOKEN", "owned-token")

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["instance_token"] == "owned-token"


def test_health_accepts_the_legacy_instance_token_environment(monkeypatch):
    monkeypatch.delenv("CITEFOLD_BACKEND_INSTANCE_TOKEN", raising=False)
    monkeypatch.setenv("VCC_BACKEND_INSTANCE_TOKEN", "legacy-token")

    response = client.get("/health")

    assert response.json()["instance_token"] == "legacy-token"


def test_quiesce_requires_the_owned_instance_token(monkeypatch):
    monkeypatch.setenv("CITEFOLD_BACKEND_INSTANCE_TOKEN", "owned-token")

    response = client.post("/runtime/quiesce")

    assert response.status_code == 403


def test_quiesce_reports_only_confirmed_idle(monkeypatch):
    class FakeManager:
        def __init__(self, result: bool) -> None:
            self.result = result

        def quiesce(self, *, timeout_seconds: float) -> bool:
            assert timeout_seconds == 4.0
            return self.result

    monkeypatch.setenv("CITEFOLD_BACKEND_INSTANCE_TOKEN", "owned-token")
    monkeypatch.setattr(
        main,
        "get_reliable_task_manager",
        lambda: FakeManager(True),
    )
    response = client.post(
        "/runtime/quiesce",
        headers={"X-Citefold-Instance-Token": "owned-token"},
    )
    assert response.status_code == 200
    assert response.json()["status"] == "quiesced"
    assert response.json()["instance_token"] == "owned-token"

    monkeypatch.setattr(
        main,
        "get_reliable_task_manager",
        lambda: FakeManager(False),
    )
    response = client.post(
        "/runtime/quiesce",
        headers={"X-Citefold-Instance-Token": "owned-token"},
    )
    assert response.status_code == 409
    assert response.json()["status"] == "timeout"


def test_quiesce_accepts_the_legacy_instance_token_header(monkeypatch):
    class FakeManager:
        def quiesce(self, *, timeout_seconds: float) -> bool:
            assert timeout_seconds == 4.0
            return True

    monkeypatch.setenv("CITEFOLD_BACKEND_INSTANCE_TOKEN", "owned-token")
    monkeypatch.setattr(
        main,
        "get_reliable_task_manager",
        lambda: FakeManager(),
    )

    response = client.post(
        "/runtime/quiesce",
        headers={"X-VCC-Instance-Token": "owned-token"},
    )

    assert response.status_code == 200
