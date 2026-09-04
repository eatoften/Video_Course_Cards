from pathlib import Path

import pytest

from app import settings


@pytest.fixture(autouse=True)
def isolate_app_path_environment(monkeypatch):
    for name in (
        "CITEFOLD_DATA_DIR",
        "CITEFOLD_DB_PATH",
        "CITEFOLD_UPLOAD_DIR",
        "CITEFOLD_TRANSCRIPT_DIR",
        "CITEFOLD_EXPORT_DIR",
        "CITEFOLD_LOG_DIR",
        "CITEFOLD_SOURCE_DIR",
        "CITEFOLD_BACKEND_LOG_FILE",
        "CITEFOLD_DESKTOP",
        "VCC_DATA_DIR",
        "VCC_DB_PATH",
        "VCC_UPLOAD_DIR",
        "VCC_TRANSCRIPT_DIR",
        "VCC_EXPORT_DIR",
        "VCC_LOG_DIR",
        "VCC_SOURCE_DIR",
        "VCC_BACKEND_LOG_FILE",
        "VCC_DESKTOP",
    ):
        monkeypatch.delenv(name, raising=False)
    settings.get_app_path_settings.cache_clear()
    yield
    settings.get_app_path_settings.cache_clear()


def test_app_path_settings_use_citefold_data_dir(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("CITEFOLD_DATA_DIR", str(tmp_path))
    settings.get_app_path_settings.cache_clear()

    paths = settings.get_app_path_settings()

    assert paths.data_dir == tmp_path
    assert paths.db_path == tmp_path / "data" / "jobs.db"
    assert paths.upload_dir == tmp_path / "uploads"
    assert paths.transcript_dir == tmp_path / "transcripts"
    assert paths.export_dir == tmp_path / "exports"
    assert paths.log_dir == tmp_path / "logs"
    assert paths.source_dir == tmp_path / "sources"

    settings.get_app_path_settings.cache_clear()


def test_app_path_settings_default_to_backend_data(monkeypatch) -> None:
    monkeypatch.delenv("CITEFOLD_DATA_DIR", raising=False)
    monkeypatch.delenv("CITEFOLD_DB_PATH", raising=False)
    monkeypatch.delenv("CITEFOLD_DESKTOP", raising=False)
    monkeypatch.delenv("VCC_DATA_DIR", raising=False)
    monkeypatch.delenv("VCC_DB_PATH", raising=False)
    monkeypatch.delenv("VCC_DESKTOP", raising=False)
    settings.get_app_path_settings.cache_clear()

    paths = settings.get_app_path_settings()

    assert paths.data_dir == Path(settings.BACKEND_DIR) / "data"
    assert paths.db_path == Path(settings.BACKEND_DIR) / "data" / "jobs.db"

    settings.get_app_path_settings.cache_clear()


def test_app_path_settings_desktop_mode_uses_app_data_database(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.delenv("CITEFOLD_DATA_DIR", raising=False)
    monkeypatch.delenv("CITEFOLD_DB_PATH", raising=False)
    monkeypatch.setenv("CITEFOLD_DESKTOP", "1")
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    settings.get_app_path_settings.cache_clear()

    paths = settings.get_app_path_settings()

    assert paths.data_dir == tmp_path / "Video Course Cards"
    assert (
        paths.db_path
        == tmp_path / "Video Course Cards" / "data" / "jobs.db"
    )

    settings.get_app_path_settings.cache_clear()


def test_app_path_settings_citefold_db_path_overrides_default(
    monkeypatch,
    tmp_path,
) -> None:
    custom_db_path = tmp_path / "custom" / "citefold.db"
    monkeypatch.setenv("CITEFOLD_DB_PATH", str(custom_db_path))
    settings.get_app_path_settings.cache_clear()

    paths = settings.get_app_path_settings()

    assert paths.db_path == custom_db_path

    settings.get_app_path_settings.cache_clear()


def test_app_path_settings_accept_legacy_vcc_environment(monkeypatch, tmp_path) -> None:
    legacy_data_dir = tmp_path / "legacy"
    monkeypatch.delenv("CITEFOLD_DATA_DIR", raising=False)
    monkeypatch.setenv("VCC_DATA_DIR", str(legacy_data_dir))
    settings.get_app_path_settings.cache_clear()

    paths = settings.get_app_path_settings()

    assert paths.data_dir == legacy_data_dir
    assert paths.db_path == legacy_data_dir / "data" / "jobs.db"

    settings.get_app_path_settings.cache_clear()


def test_citefold_environment_takes_precedence_over_legacy_vcc(
    monkeypatch,
    tmp_path,
) -> None:
    current_data_dir = tmp_path / "current"
    monkeypatch.setenv("CITEFOLD_DATA_DIR", str(current_data_dir))
    monkeypatch.setenv("VCC_DATA_DIR", str(tmp_path / "legacy"))
    settings.get_app_path_settings.cache_clear()

    assert settings.get_app_path_settings().data_dir == current_data_dir

    settings.get_app_path_settings.cache_clear()
