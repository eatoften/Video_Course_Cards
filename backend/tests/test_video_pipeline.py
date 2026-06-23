import app.video_pipeline as video_pipeline
from app.media_metadata import VideoMetadata
from app.transcription import (
    TranscriptSegment,
    TranscriptionResult,
)


def test_video_pipeline_processes_video(monkeypatch, tmp_path):
    video_path = tmp_path / "lecture.mp4"
    artifact_root = tmp_path / "artifacts"

    video_path.write_bytes(b"fake video")

    metadata = VideoMetadata(
        duration_seconds=10.0,
        width=1920,
        height=1080,
        video_codec="h264",
        has_audio=True,
    )

    transcription = TranscriptionResult(
        language="en",
        language_probability=0.99,
        duration_seconds=10.0,
        segments=[
            TranscriptSegment(
                start_seconds=0.0,
                end_seconds=2.0,
                text="Hello world",
            )
        ],
    )

    calls = []

    raw_media_data = {
        "streams": [],
        "format": {},
    }

    expected_audio_path = (
        artifact_root
        / "audio"
        / "lecture.wav"
    )

    expected_transcript_path = (
        artifact_root
        / "transcripts"
        / "lecture.json"
    )

    def fake_probe_video(received_path):
        calls.append("probe_video")

        assert received_path == video_path

        return raw_media_data

    def fake_extract_video_metadata(received_data):
        calls.append("extract_video_metadata")

        assert received_data is raw_media_data

        return metadata

    def fake_extract_audio(
        received_video_path,
        output_path,
    ):
        calls.append("extract_audio")

        assert received_video_path == video_path
        assert output_path == expected_audio_path

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_path.write_bytes(b"fake wav")

        return output_path

    class FakeTranscriber:
        def transcribe(self, audio_path):
            calls.append("transcribe")

            assert audio_path == expected_audio_path
            assert audio_path.is_file()

            return transcription

    def fake_save_transcription(
        received_result,
        output_path,
    ):
        calls.append("save_transcription")

        assert received_result is transcription
        assert output_path == expected_transcript_path

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_path.write_text(
            "{}",
            encoding="utf-8",
        )

        return output_path

    monkeypatch.setattr(
        video_pipeline,
        "probe_video",
        fake_probe_video,
    )

    monkeypatch.setattr(
        video_pipeline,
        "extract_video_metadata",
        fake_extract_video_metadata,
    )

    monkeypatch.setattr(
        video_pipeline,
        "extract_audio",
        fake_extract_audio,
    )

    monkeypatch.setattr(
        video_pipeline,
        "save_transcription",
        fake_save_transcription,
    )

    pipeline = video_pipeline.VideoPipeline(
        FakeTranscriber()
    )

    result = pipeline.process(
        video_path,
        artifact_root,
    )

    assert calls == [
        "probe_video",
        "extract_video_metadata",
        "extract_audio",
        "transcribe",
        "save_transcription",
    ]

    assert result.metadata == metadata
    assert result.audio_path == expected_audio_path
    assert result.transcript_path == expected_transcript_path
    assert result.transcription == transcription

    assert expected_audio_path.is_file()
    assert expected_transcript_path.is_file()