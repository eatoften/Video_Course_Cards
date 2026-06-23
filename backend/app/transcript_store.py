from pathlib import Path

from .transcription import TranscriptionResult


def save_transcription(
    result: TranscriptionResult,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    json_text = result.model_dump_json(indent=2)

    output_path.write_text(
        json_text,
        encoding="utf-8",
    )

    return output_path


def load_transcription(
    input_path: Path,
) -> TranscriptionResult:
    if not input_path.is_file():
        raise FileNotFoundError(
            f"Transcript file not found: {input_path}"
        )

    json_text = input_path.read_text(
        encoding="utf-8"
    )

    result = TranscriptionResult.model_validate_json(
        json_text
    )

    return result