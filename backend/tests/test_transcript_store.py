from app.transcript_store import (
    load_transcription,
    save_transcription,
)
from app.transcription import (
    TranscriptSegment,
    TranscriptionResult,
)


def test_save_and_load_transcription_round_trip(tmp_path):
    original_result = TranscriptionResult(
        language="zh",
        language_probability=0.98,
        duration_seconds=5.0,
        segments=[
            TranscriptSegment(
                start_seconds=0.0,
                end_seconds=2.5,
                text="你好，线性代数。",
            ),
            TranscriptSegment(
                start_seconds=2.5,
                end_seconds=5.0,
                text="This is a test.",
            ),
        ],
    )

    output_path = (
        tmp_path
        / "transcripts"
        / "lecture.json"
    )

    saved_path = save_transcription(
        original_result,
        output_path,
    )

    loaded_result = load_transcription(
        output_path
    )

    assert saved_path == output_path
    assert output_path.is_file()

    assert loaded_result.model_dump() == (
        original_result.model_dump()
    )