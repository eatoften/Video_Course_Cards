from pathlib import Path

from faster_whisper import WhisperModel
from pydantic import BaseModel

class TranscriptSegment(BaseModel):
    start_seconds : float
    end_seconds : float
    text: str


class TranscriptionResult(BaseModel):
    language: str
    language_probability: float
    duration_seconds: float
    segments: list[TranscriptSegment]

class FasterWhisperTranscriber:
    def __init__(
            self,
            model_size: str = "base",
            device: str = "cpu",
            compute_type: str = "ints",
    ) -> None:
        self._model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
        )

    def transcribe(self, audio_path: Path) -> TranscriptionResult:
        if not audio_path.is_file():
            raise FileNotFoundError(
                f"Audio file not found: {audio_path}"
            )
        
        segment_iterator, info = self._model.transcribe(
            str(audio_path),
            beam_size=5,
            task="transcribe",
        )

        transcript_segments: list[TranscriptSegment] = []

        for segment in segment_iterator:
            text = segment.text.strip()

            if not text:
                continue
                
            transcript_segments.append(
                TranscriptSegment(
                    start_seconds=segment.start,
                    end_seconds= segment.end,
                    text=text,
                )
            )

        return TranscriptionResult(
            language=info.language or "unknown",
            language_probability=info.language_probability or 0.0,
            duration_seconds= info.duration,
            segments=transcript_segments
        )
        