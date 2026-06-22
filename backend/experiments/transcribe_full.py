from pathlib import Path
from time import perf_counter

from app.transcription import FasterWhisperTranscriber


BACKEND_DIR = Path(__file__).resolve().parent.parent
AUDIO_PATH = BACKEND_DIR / "data" / "audio" / "recording.wav"

OUTPUT_PATH = BACKEND_DIR/ "data" / "transcripts" / "recording.json"



def main() -> None:
    if not AUDIO_PATH.is_file():
        raise FileNotFoundError(f"Audio file not found: {AUDIO_PATH}")
    
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    model_load_start = perf_counter()

    transcriber = FasterWhisperTranscriber(
        model_size="base",
        device="cpu",
        compute_type="int8",
    )

    model_load_seconds = perf_counter() - model_load_start

    transcription_start = perf_counter()

    result = transcriber.transcribe(AUDIO_PATH)


    transcription_seconds = perf_counter() - transcription_start

    json_text = result.model_dump_json(indent=2)

    real_time_factor = (
        transcription_seconds / result.duration_seconds
    )

    print(f"Detected language: {result.language}")
    print(
        "Language probability: "
        f"{result.language_probability:.3f}"
    )
    print(
        f"Audio duration: {result.duration_seconds:.2f} seconds"
    )
    print(
        f"Model load time: {model_load_seconds:.2f} seconds"
    )
    print(
        f"Transcription time: {transcription_seconds:.2f} seconds"
    )
    print(f"Real-time factor: {real_time_factor:.3f}")

    print("\nTranscript:")

    for segment in result.segments:
        print(
            f"[{segment.start_seconds:7.2f}s -> "
            f"{segment.end_seconds:7.2f}s] "
            f"{segment.text}"
        )

    OUTPUT_PATH.write_text(json_text, encoding="utf-8")

    print(f"\nSegments: {len(result.segments)}")
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()