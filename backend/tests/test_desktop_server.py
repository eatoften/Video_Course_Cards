from app import desktop_server


def test_parse_args_uses_cli_values() -> None:
    config = desktop_server.parse_args(
        [
            "--host",
            "127.0.0.2",
            "--port",
            "8123",
            "--reload",
            "--no-reuse-existing",
        ]
    )

    assert config.host == "127.0.0.2"
    assert config.port == 8123
    assert config.reload is True
    assert config.reuse_existing is False
    assert config.base_url == "http://127.0.0.2:8123"


def test_main_reuses_existing_backend(monkeypatch) -> None:
    uvicorn_calls = []

    monkeypatch.setattr(
        desktop_server,
        "is_backend_ready",
        lambda base_url: True,
    )
    monkeypatch.setattr(
        desktop_server.uvicorn,
        "run",
        lambda *args, **kwargs: uvicorn_calls.append((args, kwargs)),
    )

    result = desktop_server.main(["--port", "8123"])

    assert result == 0
    assert uvicorn_calls == []


def test_main_starts_uvicorn_when_backend_is_not_ready(monkeypatch) -> None:
    uvicorn_calls = []

    monkeypatch.setattr(
        desktop_server,
        "is_backend_ready",
        lambda base_url: False,
    )
    monkeypatch.setattr(
        desktop_server.uvicorn,
        "run",
        lambda *args, **kwargs: uvicorn_calls.append((args, kwargs)),
    )

    result = desktop_server.main(["--host", "127.0.0.1", "--port", "8123"])

    assert result == 0
    assert uvicorn_calls == [
        (
            ("app.main:app",),
            {
                "host": "127.0.0.1",
                "port": 8123,
                "reload": False,
                "log_level": "info",
            },
        )
    ]
