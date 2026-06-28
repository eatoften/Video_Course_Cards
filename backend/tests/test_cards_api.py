from fastapi.testclient import TestClient

import app.main as main
from app.job import VideoJob, VideoJobStatus
from app.job_store import create_job
from app.llm_client import (
    LLMClientError,
    LLMMessage,
    LLMModelList,
    LLMStatus,
)
from app.settings import LLMSettings
from app.transcript_store import save_transcription
from app.transcription import TranscriptSegment, TranscriptionResult


client = TestClient(main.app)


class FakeLLMClient:
    def __init__(self, outputs: list[str] | None = None) -> None:
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

        return self.outputs.pop(0)


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
    assert data["cards"] == [
        {
            "title": "Learning Rate",
            "summary": "The learning rate controls update size.",
            "key_points": [
                "It affects each parameter update.",
                "Too large a value can overshoot.",
            ],
            "claims": [
                {
                    "text": "The learning rate controls update size.",
                    "evidence": [
                        {
                            "quote": (
                                "The learning rate controls update size."
                            ),
                            "segment_start_seconds": 4.0,
                            "segment_end_seconds": 8.0,
                        }
                    ],
                }
            ],
            "unsupported_terms": [],
            "question": "What does the learning rate control?",
            "answer": "It controls the size of parameter updates.",
            "difficulty": "easy",
            "source_start_seconds": 4.0,
            "source_end_seconds": 8.0,
        }
    ]
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
            "Local LLM generated cards that were not grounded in the "
            "selected transcript. Try a shorter window or a larger model."
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
