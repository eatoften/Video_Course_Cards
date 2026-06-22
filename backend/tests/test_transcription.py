from types import SimpleNamespace

import app.transcription as transcription

import pytest



def test_transcriber_returns_structured_result(monkeypatch, tmp_path):
    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"fake wav content")

    calls = {}

    class FakeWhisperModel:
        def __init__(
            self,
            model_size,
            device,
            compute_type,
        ):
            calls["init"] = {
                "model_size": model_size,
                "device": device,
                "compute_type": compute_type,
            }

        def transcribe(self, file_path, **kwargs):
            calls["transcribe"] = {
                "file_path": file_path,
                "kwargs": kwargs,
            }

            segments = iter(
                [
                    SimpleNamespace(
                        start=0.0,
                        end=1.5,
                        text=" Hello ",
                    ),
                    SimpleNamespace(
                        start=1.5,
                        end=2.0,
                        text="   ",
                    ),
                    SimpleNamespace(
                        start=2.0,
                        end=3.2,
                        text="world",
                    ),
                ]
            )

            info = SimpleNamespace(
                language="en",
                language_probability=0.98,
                duration=3.2,
            )

            return segments, info

    monkeypatch.setattr(
        transcription,
        "WhisperModel",
        FakeWhisperModel,
    )

    transcriber = transcription.FasterWhisperTranscriber(
        model_size="base",
        device="cpu",
        compute_type="int8",
    )

    result = transcriber.transcribe(audio_path)

    assert calls["init"] == {
        "model_size": "base",
        "device": "cpu",
        "compute_type": "int8",
    }

    assert calls["transcribe"] == {
        "file_path": str(audio_path),
        "kwargs": {
            "beam_size": 5,
            "task": "transcribe",
        },
    }

    assert result.model_dump() == {
        "language": "en",
        "language_probability": 0.98,
        "duration_seconds": 3.2,
        "segments": [
            {
                "start_seconds": 0.0,
                "end_seconds": 1.5,
                "text": "Hello",
            },
            {
                "start_seconds": 2.0,
                "end_seconds": 3.2,
                "text": "world",
            },
        ],
    }



def test_transcriber_rejects_missing_audio_file(monkeypatch, tmp_path):
    class FakeWhisperModel:
        def __init__(self, *args, **kwargs):
            pass

        def transcribe(self, *args, **kwargs):
            raise AssertionError(
                "Whisper should not run for a missing audio file"
            )

    monkeypatch.setattr(
        transcription,
        "WhisperModel",
        FakeWhisperModel,
    )

    transcriber = transcription.FasterWhisperTranscriber()

    missing_audio_path = tmp_path / "missing.wav"

    with pytest.raises(
        FileNotFoundError,
        match="Audio file not found",
    ):
        transcriber.transcribe(missing_audio_path)