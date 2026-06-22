import subprocess
from pathlib import Path
import pytest

import app.audio_extractor as audio_extractor

def test_extract_audio_returns_created_output(monkeypatch, tmp_path):
    video_path = tmp_path / "lecture.mp4"
    output_path = tmp_path / "audio" / "lecture.wav"

    def fake_run(command, **kwargs):
        assert command[0] == "ffmpeg"
        assert command[command.index("-i") + 1] == str(video_path)
        assert command[command.index("-ac") + 1] == "1"
        assert command[command.index("-ar") + 1] == "16000"
        assert command[command.index("-c:a") + 1] == "pcm_s16le"
        assert command[-1] == str(output_path)

        output_path.write_bytes(b"fake wav content")

        return subprocess.CompletedProcess(
            args=command,
            returncode=0,
            stdout="",
            stderr="",
        )
    
    monkeypatch.setattr(
        audio_extractor.subprocess,
        "run",
        fake_run, 
    )

    result = audio_extractor.extract_audio(
        video_path,
        output_path,
    )

    assert result == output_path
    assert output_path.exists()
    assert output_path.read_bytes() == b"fake wav content"



def test_extract_audio_cleans_up_when_ffmpeg_fails(monkeypatch, tmp_path):
    video_path = tmp_path / "lecture.mp4"
    output_path = tmp_path / "audio" / "lecture.wav"

    video_path.write_bytes(b"fake video content")

    def fake_run(command, **kwargs):
        output_path.write_bytes(b"incomplete wav")

        return subprocess.CompletedProcess(
            args=command,
            returncode=1,
            stdout="",
            stderr="decoder failed",
        )

    monkeypatch.setattr(
        audio_extractor.subprocess,
        "run",
        fake_run,
    )

    with pytest.raises(
        audio_extractor.AudioExtractionError
    ) as exc_info:
        audio_extractor.extract_audio(
            video_path,
            output_path,
        )

    assert "decoder failed" in str(exc_info.value)
    assert not output_path.exists()