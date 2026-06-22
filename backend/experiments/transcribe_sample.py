from pathlib import Path
from time import perf_counter

from faster_whisper import WhisperModel

BACKEND_DIR = Path(__file__).resolve().parent.parent
AUDIO_PATH = BACKEND_DIR / "data" / "audio" / "sample-30s.wav"

def main():
    if not AUDIO_PATH.is_file():
        raise FileNotFoundError(f"Audio file not found: {AUDIO_PATH}")
    
    model_load_start = perf_counter()

    model = WhisperModel(
        "base",
        device = "cpu",
        compute_type="int8",
    )

    model_load_seconds = perf_counter() - model_load_start

    transcription_start = perf_counter()

    segment_iterator, info = model.transcribe(
        str(AUDIO_PATH),
        beam_size=5,
        task="transcribe",
    )

    segments = list(segment_iterator)

    transcription_seconds = perf_counter() - transcription_start

    print(f"Detected language: {info.language}")
    print(f"Language probability: {info.language_probability:.3f}")
    print(f"Audio duration: {info.duration:.2f} seconds")
    print(f"Model load time: {model_load_seconds:.2f} seconds")
    print(f"Transcription time: {transcription_seconds:.2f} seconds")

    real_time_factor = transcription_seconds / info.duration
    print(f"Real-time factor: {real_time_factor:.3f}")

    print("\nTranscript:")

    for segment in segments:
        text = segment.text.strip()

        if text:
            print(
                f"[{segment.start:7.2f}s -> "
                f"{segment.end:7.2f}s] {text}"
            )



if __name__ == "__main__":
    main()