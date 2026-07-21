import json

from app.llm_client import LLMMessage
from rag_lab.answering import (
    build_grounded_context,
    evaluate_grounded_answers,
    generate_grounded_answer,
)
from rag_lab.schemas import (
    RagBenchmarkItem,
    RagCorpusSnapshot,
    RagGroundedAnswerRecord,
    RankedCard,
    RetrievalRecord,
)


class FakeAnswerClient:
    def create_chat_completion(
        self,
        messages: list[LLMMessage],
        **_: object,
    ) -> str:
        assert "QUESTION" in messages[-1].content
        return json.dumps(
            {
                "answerable": True,
                "answer": "The supported fact.",
                "claims": [
                    {
                        "text": "The supported fact.",
                        "citations": [
                            {
                                "card_id": "card-a",
                                "claim_id": "claim-a",
                                "evidence_id": "evidence-a",
                            }
                        ],
                    }
                ],
            }
        )


class FakeEmbedder:
    def embed_texts(self, texts, **_):
        return [[1.0, 0.0] for _ in texts]


def _corpus() -> RagCorpusSnapshot:
    return RagCorpusSnapshot(
        snapshot_id="snapshot",
        course_id="course",
        source_database_sha256="a" * 64,
        snapshot_sha256="b" * 64,
        cards=[
            {
                "card_id": "card-a",
                "job_id": "job",
                "lecture_name": "lecture.mp4",
                "title": "Supported",
                "summary": "Summary",
                "document_text": "Document",
                "content_status": "draft",
                "source_start_seconds": 1,
                "source_end_seconds": 2,
                "claims": [
                    {
                        "claim_id": "claim-a",
                        "text": "The supported fact.",
                        "evidence": [
                            {
                                "evidence_id": "evidence-a",
                                "quote": "The supported fact.",
                                "start_seconds": 1,
                                "end_seconds": 2,
                            }
                        ],
                    }
                ],
            }
        ],
    )


def _retrieval() -> RetrievalRecord:
    return RetrievalRecord(
        question_id="q",
        category="factual",
        split="development",
        system="dense",
        elapsed_milliseconds=1,
        ranked_cards=[
            RankedCard(
                card_id="card-a",
                rank=1,
                score=0.9,
                retrieval_source="dense",
            )
        ],
    )


def test_context_generation_and_grounded_output_keep_exact_ids() -> None:
    context, characters = build_grounded_context(
        _corpus(),
        _retrieval(),
        top_k=1,
        character_budget=1000,
    )
    payload, _ = generate_grounded_answer(
        "What is supported?",
        context,
        llm_client=FakeAnswerClient(),
        model="fake",
        temperature=0,
        max_tokens=100,
    )

    assert characters <= 1000
    assert payload.claims[0].citations[0].evidence_id == "evidence-a"


def test_grounded_answer_metrics_score_gold_citations_and_abstention() -> None:
    item = RagBenchmarkItem(
        question_id="q",
        category="factual",
        split="development",
        question="What is supported?",
        answerable=True,
        reference_answer="The supported fact.",
        gold_card_ids=["card-a"],
        gold_claim_ids=["claim-a"],
        evidence=[
            {
                "card_id": "card-a",
                "claim_id": "claim-a",
                "evidence_id": "evidence-a",
                "quote": "The supported fact.",
                "start_seconds": 1,
                "end_seconds": 2,
            }
        ],
        authoring_method="manual",
    )
    answer = RagGroundedAnswerRecord(
        question_id="q",
        category="factual",
        split="development",
        system="dense",
        context_card_ids=["card-a"],
        context_characters=100,
        answerable_prediction=True,
        answer="The supported fact.",
        claims=[
            {
                "text": "The supported fact.",
                "citations": [
                    {
                        "card_id": "card-a",
                        "claim_id": "claim-a",
                        "evidence_id": "evidence-a",
                    }
                ],
            }
        ],
        latency_milliseconds=2,
    )

    report = evaluate_grounded_answers(
        [item],
        [_retrieval()],
        [answer],
        semantic_embedder=FakeEmbedder(),
    )

    assert report["gold_claim_citation_recall"] == 1.0
    assert report["citation_precision_against_gold"] == 1.0
    assert report["abstention_f1"] == 1.0
