from app import card_service, job_service
from app.llm_client import LLMMessage
from app.settings import LLMSettings
from app.transcription import TranscriptSegment


class RecordingCardClient:
    def __init__(self) -> None:
        self.settings = LLMSettings(model="test-model")
        self.messages: list[LLMMessage] = []

    def create_chat_completion(
        self,
        messages: list[LLMMessage],
        **_: object,
    ) -> str:
        self.messages = messages
        return """
        {
          "cards": [{
            "title": "Convolution layer",
            "summary": "A convolution layer is an image-specific operator.",
            "key_points": ["It is used in convolutional networks."],
            "claims": [{
              "text": "The slide introduces a convolution layer.",
              "evidence_quotes": ["Convolution Layer"]
            }],
            "question": "Which layer is introduced?",
            "answer": "A convolution layer."
          }]
        }
        """


def test_explicit_slide_context_reuses_grounded_card_pipeline() -> None:
    client = RecordingCardClient()
    context = job_service.TranscriptContext(
        job_id="page-1",
        source_video="lecture-5-slide-18",
        start_seconds=810,
        end_seconds=811,
        segments=[
            TranscriptSegment(
                start_seconds=810,
                end_seconds=811,
                text="Convolution Layer",
            )
        ],
        text="Convolution Layer",
    )
    request = card_service.CardDraftRequest(
        job_id="page-1",
        start_seconds=810,
        end_seconds=811,
        card_count=1,
        model="test-model",
    )

    response = card_service.draft_knowledge_cards_from_context(
        request,
        context,
        llm_client=client,
        evidence_kind="slide_ocr",
    )

    assert len(response.cards) == 1
    assert response.cards[0].claims[0].evidence[0].quote == "Convolution Layer"
    assert "slide ocr evidence" in client.messages[0].content.lower()
    assert "<<<SLIDE_OCR" in client.messages[1].content
