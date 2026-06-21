import subprocess
import pytest

from pathlib import Path

import app.media_probe as media_probe


def test_probe_media_returns_parsed_json(monkeypatch):
    fake_stdout = """
    {
        "streams": [
            {
                "codec_type": "video",
                "codec_name": "h264"
            }
        ],
        "format": {
            "duration": "10.0"
        }
    }
    """

    fake_result = subprocess.CompletedProcess(
        args=["ffprobe"],
        returncode=0,
        stdout=fake_stdout,
        stderr="",
    )

    def fake_run_ffprobe(file_path: Path):
        assert file_path == Path("lecture.mp4")
        return fake_result

    monkeypatch.setattr(
        media_probe,
        "run_ffprobe",
        fake_run_ffprobe,
    )

    data = media_probe.probe_media(Path("lecture.mp4"))

    assert data == {
        "streams": [
            {
                "codec_type": "video",
                "codec_name": "h264",
            }
        ],
        "format": {
            "duration": "10.0",
        },
    }



def test_probe_media_raises_error_when_ffprobe_fails(monkeypatch):
    fake_result = subprocess.CompletedProcess(
        args=["ffprobe"],
        returncode=1,
        stdout="",
        stderr="moov atom not found",
    )

    def fake_run_ffprobe(file_path: Path):
        return fake_result

    monkeypatch.setattr(
        media_probe,
        "run_ffprobe",
        fake_run_ffprobe,
    )

    with pytest.raises(media_probe.MediaProbeError) as exc_info:
        media_probe.probe_media(Path("fake.mp4"))

    assert "moov atom not found" in str(exc_info.value)



def test_probe_video_returns_data_when_video_stream_exists(monkeypatch):
    fake_data = {
        "streams": [
            {
                "codec_type": "video",
                "codec_name": "h264",
            },
            {
                "codec_type": "audio",
                "codec_name": "aac",
            },
        ]
    }

    def fake_probe_media(file_path: Path):
        assert file_path == Path("lecture.mp4")
        return fake_data

    monkeypatch.setattr(
        media_probe,
        "probe_media",
        fake_probe_media,
    )

    result = media_probe.probe_video(Path("lecture.mp4"))

    assert result == fake_data


def test_probe_video_rejects_audio_only_media(monkeypatch):
    fake_data = {
        "streams": [
            {
                "codec_type": "audio",
                "codec_name": "aac",
            }
        ]
    }

    def fake_probe_media(file_path: Path):
        assert file_path == Path("audio.m4a")
        return fake_data

    monkeypatch.setattr(
        media_probe,
        "probe_media",
        fake_probe_media,
    )

    with pytest.raises(
        media_probe.MediaProbeError,
        match="media file contains no video stream",
    ):
        media_probe.probe_video(Path("audio.m4a"))