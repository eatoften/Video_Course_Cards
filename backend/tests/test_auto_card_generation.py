from fastapi.testclient import TestClient

import app.auto_card_generation_service as auto_card_generation_service
import app.main as main
import app.transcript_chunk_service as transcript_chunk_service
from app.card_generation_run import AutoCardGenerationRequest
from app.job import VideoJob, VideoJobStatus
from app.job_store import create_job
from app.knowledge_card_store import list_cards_for_job
from app.llm_client import LLMMessage
from app.settings import LLMSettings
from app.transcript_chunk import TranscriptChunkGenerationRequest
from app.transcript_store import save_transcription
from app.transcription import TranscriptSegment, TranscriptionResult


client = TestClient(main.app)


class FakeEmbedder:
    def embed_texts(
        self,
        texts,
        *,
        batch_size=None,
    ) -> list[list[float]]:
        return [
            [1.0, 0.0],
            [0.99, 0.01],
            [0.0, 1.0],
            [0.01, 0.99],
        ][:len(texts)]


class FakeLLMClient:
    def __init__(self, outputs: list[str]) -> None:
        self.outputs = outputs
        self.calls: list[list[LLMMessage]] = []
        self.settings = LLMSettings(
            provider="ollama",
            base_url="http://localhost:11434/v1",
            model="qwen3:4b",
            api_key="local",
        )

    def create_chat_completion(
        self,
        messages: list[LLMMessage],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: dict[str, object] | None = None,
    ) -> str:
        self.calls.append(messages)

        return self.outputs.pop(0)


def create_completed_job_with_transcript(tmp_path) -> VideoJob:
    transcript = TranscriptionResult(
        language="en",
        language_probability=0.99,
        duration_seconds=120,
        segments=[
            TranscriptSegment(
                start_seconds=0,
                end_seconds=30,
                text="Image classification assigns labels to images.",
            ),
            TranscriptSegment(
                start_seconds=30,
                end_seconds=60,
                text="Nearest neighbor compares a test image to examples.",
            ),
            TranscriptSegment(
                start_seconds=60,
                end_seconds=90,
                text="Loss functions measure prediction mistakes.",
            ),
            TranscriptSegment(
                start_seconds=90,
                end_seconds=120,
                text="Optimization updates parameters using gradients.",
            ),
        ],
    )
    transcript_path = tmp_path / "transcripts" / "lecture.json"
    save_transcription(transcript, transcript_path)
    job = VideoJob(
        id="job-1",
        video_path=tmp_path / "lecture.mp4",
        status=VideoJobStatus.completed,
        original_filename="lecture.mp4",
        transcript_path=transcript_path,
    )
    create_job(job)

    return job


def card_output(
    *,
    title: str,
    summary: str,
    claim: str,
    quote: str,
    question: str,
    answer: str,
) -> str:
    return f"""
    {{
      "cards": [
        {{
          "title": "{title}",
          "summary": "{summary}",
          "key_points": ["{summary}"],
          "claims": [
            {{
              "text": "{claim}",
              "evidence_quotes": ["{quote}"]
            }}
          ],
          "question": "{question}",
          "answer": "{answer}",
          "difficulty": "easy"
        }}
      ]
    }}
    """


def auto_request() -> AutoCardGenerationRequest:
    return AutoCardGenerationRequest(
        card_count_per_chunk=1,
        chunking=TranscriptChunkGenerationRequest(
            context_radius=0,
            min_chunk_seconds=30,
            max_chunk_seconds=300,
            boundary_percentile=80,
        ),
    )


def test_auto_card_generation_run_creates_saved_cards(monkeypatch, tmp_path):
    job = create_completed_job_with_transcript(tmp_path)
    fake_llm = FakeLLMClient(
        [
            card_output(
                title="Image Classification",
                summary="Image classification assigns labels.",
                claim="Image classification assigns labels to images.",
                quote="Image classification assigns labels to images.",
                question="What does image classification do?",
                answer="It assigns labels to images.",
            ),
            card_output(
                title="Loss Functions",
                summary="Loss functions measure mistakes.",
                claim="Loss functions measure prediction mistakes.",
                quote="Loss functions measure prediction mistakes.",
                question="What do loss functions measure?",
                answer="They measure prediction mistakes.",
            ),
        ]
    )
    monkeypatch.setattr(
        transcript_chunk_service,
        "_create_default_embedder",
        lambda: FakeEmbedder(),
    )

    run = auto_card_generation_service.start_auto_card_generation(
        job.id,
        auto_request(),
    )
    auto_card_generation_service.run_auto_card_generation(
        run.id,
        lambda: fake_llm,
    )
    completed_run = auto_card_generation_service.get_card_generation_run(
        run.id
    )
    cards = list_cards_for_job(job.id)

    assert completed_run.status == "completed"
    assert completed_run.total_chunks == 2
    assert completed_run.completed_chunks == 2
    assert completed_run.succeeded_chunks == 2
    assert completed_run.failed_chunks == 0
    assert completed_run.cards_created == 2
    assert [
        card.title
        for card in cards
    ] == [
        "Image Classification",
        "Loss Functions",
    ]


def test_auto_card_generation_api_starts_and_reports_run(
    monkeypatch,
    tmp_path,
):
    job = create_completed_job_with_transcript(tmp_path)
    fake_llm = FakeLLMClient(
        [
            card_output(
                title="Image Classification",
                summary="Image classification assigns labels.",
                claim="Image classification assigns labels to images.",
                quote="Image classification assigns labels to images.",
                question="What does image classification do?",
                answer="It assigns labels to images.",
            ),
            card_output(
                title="Loss Functions",
                summary="Loss functions measure mistakes.",
                claim="Loss functions measure prediction mistakes.",
                quote="Loss functions measure prediction mistakes.",
                question="What do loss functions measure?",
                answer="They measure prediction mistakes.",
            ),
        ]
    )
    monkeypatch.setattr(
        transcript_chunk_service,
        "_create_default_embedder",
        lambda: FakeEmbedder(),
    )
    monkeypatch.setattr(
        main,
        "get_llm_client",
        lambda: fake_llm,
    )

    response = client.post(
        f"/jobs/{job.id}/cards/auto-generate",
        json=auto_request().model_dump(mode="json"),
    )

    assert response.status_code == 202

    run_id = response.json()["id"]
    run_response = client.get(f"/card-generation-runs/{run_id}")

    assert run_response.status_code == 200
    assert run_response.json()["status"] == "completed"
    assert run_response.json()["cards_created"] == 2


def test_auto_card_generation_returns_409_when_transcript_not_ready(
    tmp_path,
):
    job = VideoJob(
        id="job-1",
        video_path=tmp_path / "lecture.mp4",
        status=VideoJobStatus.uploaded,
    )
    create_job(job)

    response = client.post(f"/jobs/{job.id}/cards/auto-generate")

    assert response.status_code == 409
    assert response.json() == {
        "detail": "Transcript is not available for this job."
    }
