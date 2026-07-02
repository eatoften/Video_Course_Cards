from pathlib import Path

from app import settings


def test_app_path_settings_use_vcc_data_dir(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("VCC_DATA_DIR", str(tmp_path))
    settings.get_app_path_settings.cache_clear()

    paths = settings.get_app_path_settings()

    assert paths.data_dir == tmp_path
    assert paths.db_path == tmp_path / "data" / "jobs.db"
    assert paths.upload_dir == tmp_path / "uploads"
    assert paths.transcript_dir == tmp_path / "transcripts"
    assert paths.export_dir == tmp_path / "exports"
    assert paths.log_dir == tmp_path / "logs"

    settings.get_app_path_settings.cache_clear()


def test_app_path_settings_default_to_backend_data(monkeypatch) -> None:
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
    monkeypatch.delenv("VCC_DATA_DIR", raising=False)
    monkeypatch.delenv("VCC_DB_PATH", raising=False)
    monkeypatch.setenv("VCC_DESKTOP", "1")
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    settings.get_app_path_settings.cache_clear()

    paths = settings.get_app_path_settings()

    assert paths.data_dir == tmp_path / "Video Course Cards"
    assert paths.db_path == tmp_path / "Video Course Cards" / "data" / "jobs.db"

    settings.get_app_path_settings.cache_clear()


def test_app_path_settings_vcc_db_path_overrides_default(
    monkeypatch,
    tmp_path,
) -> None:
    custom_db_path = tmp_path / "custom" / "vcc.db"
    monkeypatch.setenv("VCC_DB_PATH", str(custom_db_path))
    settings.get_app_path_settings.cache_clear()

    paths = settings.get_app_path_settings()

    assert paths.db_path == custom_db_path

    settings.get_app_path_settings.cache_clear()
