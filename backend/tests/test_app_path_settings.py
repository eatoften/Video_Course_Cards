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
    monkeypatch.delenv("VCC_DESKTOP", raising=False)
    settings.get_app_path_settings.cache_clear()

    paths = settings.get_app_path_settings()

    assert paths.data_dir == Path(settings.BACKEND_DIR) / "data"

    settings.get_app_path_settings.cache_clear()
