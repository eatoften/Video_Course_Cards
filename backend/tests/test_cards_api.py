from fastapi.testclient import TestClient

import app.main as main
from app import card_service
from app.job import VideoJob, VideoJobStatus
from app.job_store import create_job
from app.llm_client import (
    LLMClientError,
    LLMMessage,
    LLMModelList,
    LLMStatus,
    LLMTimeoutError,
)
from app.settings import LLMSettings
from app.transcript_store import save_transcription
from app.transcription import TranscriptSegment, TranscriptionResult


client = TestClient(main.app)


class FakeLLMClient:
    def __init__(
        self,
        outputs: list[str | Exception] | None = None,
    ) -> None:
        self.outputs = outputs or []
        self.calls: list[list[LLMMessage]] = []
        self.requested_models: list[str | None] = []
        self.requested_max_tokens: list[int | None] = []
        self.settings = LLMSettings(
            provider="ollama",
            base_url="http://localhost:11434/v1",
            model="qwen3:4b",
            api_key="local",
        )

    def check_status(self) -> LLMStatus:
        return LLMStatus(
            provider=self.settings.provider,
            base_url=self.settings.base_url,
            model=self.settings.model,
            available=True,
        )

    def list_models(self) -> LLMModelList:
        return LLMModelList(
            provider=self.settings.provider,
            base_url=self.settings.base_url,
            default_model=self.settings.model,
            models=["qwen3:4b", "qwen3:8b"],
            available=True,
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
        self.requested_models.append(model)
        self.requested_max_tokens.append(max_tokens)

        if not self.outputs:
            raise LLMClientError("No fake LLM output configured.")

        output = self.outputs.pop(0)

        if isinstance(output, Exception):
            raise output

        return output


def create_completed_job_with_transcript(tmp_path) -> VideoJob:
    transcript = TranscriptionResult(
        language="en",
        language_probability=0.99,
        duration_seconds=12.0,
        segments=[
            TranscriptSegment(
                start_seconds=0.0,
                end_seconds=4.0,
                text="Gradient descent updates parameters.",
            ),
            TranscriptSegment(
                start_seconds=4.0,
                end_seconds=8.0,
                text="The learning rate controls update size.",
            ),
            TranscriptSegment(
                start_seconds=8.0,
                end_seconds=12.0,
                text="A large learning rate can overshoot.",
            ),
        ],
    )
    transcript_path = (
        tmp_path
        / "transcripts"
        / "lecture.json"
    )
    save_transcription(transcript, transcript_path)

    job = VideoJob(
        id="job-123",
        video_path=tmp_path / "lecture.mp4",
        status=VideoJobStatus.completed,
        original_filename="lecture.mp4",
        transcript_path=transcript_path,
    )
    create_job(job)

    return job


def test_llm_status_returns_configured_local_model(monkeypatch):
    fake_client = FakeLLMClient()

    monkeypatch.setattr(
        main,
        "get_llm_client",
        lambda: fake_client,
    )

    response = client.get("/llm/status")

    assert response.status_code == 200
    assert response.json() == {
        "provider": "ollama",
        "base_url": "http://localhost:11434/v1",
        "model": "qwen3:4b",
        "available": True,
        "error_message": None,
    }


def test_llm_models_returns_installed_models(monkeypatch):
    fake_client = FakeLLMClient()

    monkeypatch.setattr(
        main,
        "get_llm_client",
        lambda: fake_client,
    )

    response = client.get("/llm/models")

    assert response.status_code == 200
    assert response.json() == {
        "provider": "ollama",
        "base_url": "http://localhost:11434/v1",
        "default_model": "qwen3:4b",
        "models": ["qwen3:4b", "qwen3:8b"],
        "available": True,
        "error_message": None,
    }


def test_draft_cards_returns_structured_cards(monkeypatch, tmp_path):
    job = create_completed_job_with_transcript(tmp_path)
    fake_client = FakeLLMClient(
        outputs=[
            """
            {
              "cards": [
                {
                  "title": "Learning Rate",
                  "summary": "The learning rate controls update size.",
                  "key_points": [
                    "It affects each parameter update.",
                    "Too large a value can overshoot."
                  ],
                  "claims": [
                    {
                      "text": "The learning rate controls update size.",
                      "evidence_quotes": [
                        "The learning rate controls update size."
                      ]
                    }
                  ],
                  "question": "What does the learning rate control?",
                  "answer": "It controls the size of parameter updates.",
                  "difficulty": "easy"
                }
              ]
            }
            """
        ]
    )

    monkeypatch.setattr(
        main,
        "get_llm_client",
        lambda: fake_client,
    )

    response = client.post(
        "/cards/draft",
        json={
            "job_id": job.id,
            "start_seconds": 0.0,
            "end_seconds": 10.0,
            "card_count": 2,
            "focus": "optimization",
            "model": "qwen3:8b",
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["job_id"] == job.id
    assert data["source_video"] == "lecture.mp4"
    assert data["provider"] == "ollama"
    assert data["model"] == "qwen3:8b"
    assert data["generation_metadata"]["provider"] == "ollama"
    assert data["generation_metadata"]["model"] == "qwen3:8b"
    assert data["generation_metadata"]["selected_segments_count"] == 3
    assert data["generation_metadata"]["requested_card_count"] == 2
    assert data["generation_metadata"]["raw_card_count"] == 1
    assert data["generation_metadata"]["returned_card_count"] == 1
    assert data["generation_metadata"]["raw_claim_count"] == 1
    assert data["generation_metadata"]["grounded_claim_count"] == 1
    assert data["generation_metadata"]["dropped_claim_count"] == 0
    assert data["generation_metadata"]["unsupported_terms_count"] == 0
    assert data["generation_metadata"]["elapsed_seconds"] >= 0
    assert data["generation_metadata"]["input_characters"] > 0
    assert (
        data["generation_metadata"]["selected_context_characters"]
        == len(
            "\n".join(
                [
                    "Gradient descent updates parameters.",
                    "The learning rate controls update size.",
                    "A large learning rate can overshoot.",
                ]
            )
        )
    )
    card = data["cards"][0]
    assert card["title"] == "Learning Rate"
    assert card["summary"] == "The learning rate controls update size."
    assert card["claims"][0]["id"]
    assert card["claims"][0]["text"] == (
        "The learning rate controls update size."
    )
    assert card["claims"][0]["evidence"][0]["id"]
    assert card["claims"][0]["evidence"][0]["segment_start_seconds"] == 4.0
    assert card["question"] == "What does the learning rate control?"
    assert card["answer"] == "It controls the size of parameter updates."
    assert "difficulty" not in card
    assert len(fake_client.calls) == 1
    assert "/no_think" in fake_client.calls[0][1].content
    assert fake_client.requested_models == ["qwen3:8b"]


def test_draft_cards_repairs_malformed_json(
    monkeypatch,
    tmp_path,
):
    job = create_completed_job_with_transcript(tmp_path)
    fake_client = FakeLLMClient(
        outputs=[
            "Here are the cards: not-json",
            """
            {
              "cards": [
                {
                  "title": "Gradient Descent",
                  "summary": "Gradient descent updates parameters.",
                  "key_points": ["It is an optimization method."],
                  "claims": [
                    {
                      "text": "Gradient descent updates parameters.",
                      "evidence_quotes": [
                        "Gradient descent updates parameters."
                      ]
                    }
                  ],
                  "question": "What does gradient descent update?",
                  "answer": "It updates parameters.",
                  "difficulty": "medium"
                }
              ]
            }
            """,
        ]
    )

    monkeypatch.setattr(
        main,
        "get_llm_client",
        lambda: fake_client,
    )

    response = client.post(
        "/cards/draft",
        json={
            "job_id": job.id,
            "start_seconds": 0.0,
            "end_seconds": 8.0,
        },
    )

    assert response.status_code == 200
    assert response.json()["cards"][0]["title"] == "Gradient Descent"
    assert len(fake_client.calls) == 2


def test_draft_cards_rejects_empty_llm_content(
    monkeypatch,
    tmp_path,
):
    job = create_completed_job_with_transcript(tmp_path)
    fake_client = FakeLLMClient(outputs=["", ""])

    monkeypatch.setattr(
        main,
        "get_llm_client",
        lambda: fake_client,
    )

    response = client.post(
        "/cards/draft",
        json={
            "job_id": job.id,
            "start_seconds": 0.0,
            "end_seconds": 8.0,
        },
    )

    assert response.status_code == 503
    assert response.json()["detail"].startswith(
        "Local LLM returned empty content."
    )
    assert len(fake_client.calls) == 2
    assert fake_client.requested_max_tokens == [8192, 16384]


def test_draft_cards_retries_empty_content_then_succeeds(
    monkeypatch,
    tmp_path,
):
    job = create_completed_job_with_transcript(tmp_path)
    fake_client = FakeLLMClient(
        outputs=[
            "",
            """
            {
              "cards": [
                {
                  "title": "Learning Rate",
                  "summary": "The learning rate controls update size.",
                  "key_points": ["It affects parameter updates."],
                  "claims": [
                    {
                      "text": "The learning rate controls update size.",
                      "evidence_quotes": [
                        "The learning rate controls update size."
                      ]
                    }
                  ],
                  "question": "What does the learning rate control?",
                  "answer": "It controls update size.",
                  "difficulty": "easy"
                }
              ]
            }
            """,
        ]
    )

    monkeypatch.setattr(
        main,
        "get_llm_client",
        lambda: fake_client,
    )

    response = client.post(
        "/cards/draft",
        json={
            "job_id": job.id,
            "start_seconds": 0.0,
            "end_seconds": 8.0,
        },
    )

    assert response.status_code == 200
    assert response.json()["cards"][0]["title"] == "Learning Rate"
    assert fake_client.requested_max_tokens == [8192, 16384]


def test_draft_cards_rejects_ungrounded_cards(
    monkeypatch,
    tmp_path,
):
    job = create_completed_job_with_transcript(tmp_path)
    fake_client = FakeLLMClient(
        outputs=[
            """
            {
              "cards": [
                {
                  "title": "Newton's Laws",
                  "summary": "Newton's laws describe motion and forces.",
                  "key_points": ["Force equals mass times acceleration."],
                  "claims": [
                    {
                      "text": "Force equals mass times acceleration.",
                      "evidence_quotes": [
                        "Force equals mass times acceleration."
                      ]
                    }
                  ],
                  "question": "What does Newton's second law state?",
                  "answer": "Force equals mass times acceleration.",
                  "difficulty": "easy"
                }
              ]
            }
            """
        ]
    )

    monkeypatch.setattr(
        main,
        "get_llm_client",
        lambda: fake_client,
    )

    response = client.post(
        "/cards/draft",
        json={
            "job_id": job.id,
            "start_seconds": 0.0,
            "end_seconds": 8.0,
        },
    )

    assert response.status_code == 503
    assert response.json() == {
        "detail": (
            "No grounded claims were found in the model output. Try "
            "selecting fewer transcript segments or using a larger model."
        )
    }


def test_draft_cards_rejects_context_that_is_too_long(
    monkeypatch,
    tmp_path,
):
    long_text = "optimization " * (
        card_service.MAX_CONTEXT_CHARACTERS // len("optimization ") + 2
    )
    transcript = TranscriptionResult(
        language="en",
        language_probability=0.99,
        duration_seconds=30.0,
        segments=[
            TranscriptSegment(
                start_seconds=0.0,
                end_seconds=30.0,
                text=long_text,
            ),
        ],
    )
    transcript_path = tmp_path / "transcripts" / "long.json"
    save_transcription(transcript, transcript_path)
    job = VideoJob(
        id="job-123",
        video_path=tmp_path / "lecture.mp4",
        status=VideoJobStatus.completed,
        original_filename="lecture.mp4",
        transcript_path=transcript_path,
    )
    create_job(job)
    fake_client = FakeLLMClient()

    monkeypatch.setattr(
        main,
        "get_llm_client",
        lambda: fake_client,
    )

    response = client.post(
        "/cards/draft",
        json={
            "job_id": job.id,
            "start_seconds": 0.0,
            "end_seconds": 30.0,
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"].startswith(
        "Selected context is too long:"
    )
    assert fake_client.calls == []


def test_draft_cards_returns_504_for_llm_timeout(
    monkeypatch,
    tmp_path,
):
    job = create_completed_job_with_transcript(tmp_path)
    fake_client = FakeLLMClient(
        outputs=[
            LLMTimeoutError("timeout"),
        ]
    )

    monkeypatch.setattr(
        main,
        "get_llm_client",
        lambda: fake_client,
    )

    response = client.post(
        "/cards/draft",
        json={
            "job_id": job.id,
            "start_seconds": 0.0,
            "end_seconds": 8.0,
        },
    )

    assert response.status_code == 504
    assert response.json() == {
        "detail": (
            "Local LLM request timed out. Try selecting fewer transcript "
            "segments or using a smaller/faster model."
        )
    }


def test_draft_cards_returns_409_when_transcript_is_not_ready(
    monkeypatch,
    tmp_path,
):
    fake_client = FakeLLMClient()
    job = VideoJob(
        id="job-123",
        video_path=tmp_path / "lecture.mp4",
        status=VideoJobStatus.uploaded,
    )
    create_job(job)

    monkeypatch.setattr(
        main,
        "get_llm_client",
        lambda: fake_client,
    )

    response = client.post(
        "/cards/draft",
        json={
            "job_id": job.id,
            "start_seconds": 0.0,
            "end_seconds": 8.0,
        },
    )

    assert response.status_code == 409
    assert response.json() == {
        "detail": "Transcript is not available for this job."
    }
    assert fake_client.calls == []
