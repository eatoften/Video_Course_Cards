from fastapi.testclient import TestClient

import app.main as main
from app.runtime_status import RuntimeDependencyStatus, RuntimeStatus


client = TestClient(main.app)


def test_get_runtime_status(monkeypatch) -> None:
    monkeypatch.setattr(
        main.runtime_service,
        "get_runtime_status",
        lambda: RuntimeStatus(
            ready=True,
            dependencies=[
                RuntimeDependencyStatus(
                    name="ffmpeg",
                    available=True,
                    version="ffmpeg version test",
                    required_for=["transcription"],
                )
            ],
        ),
    )

    response = client.get("/runtime/status")

    assert response.status_code == 200
    assert response.json() == {
        "ready": True,
        "dependencies": [
            {
                "name": "ffmpeg",
                "available": True,
                "version": "ffmpeg version test",
                "detail": None,
                "install_hint": None,
                "required_for": ["transcription"],
            }
        ],
    }


def test_post_runtime_check(monkeypatch) -> None:
    monkeypatch.setattr(
        main.runtime_service,
        "get_runtime_status",
        lambda: RuntimeStatus(
            ready=False,
            dependencies=[
                RuntimeDependencyStatus(
                    name="ollama",
                    available=False,
                    install_hint="ollama pull qwen3:4b",
                    required_for=["card generation"],
                )
            ],
        ),
    )

    response = client.post("/runtime/check")

    assert response.status_code == 200
    assert response.json()["dependencies"][0]["name"] == "ollama"
    assert response.json()["dependencies"][0]["available"] is False
