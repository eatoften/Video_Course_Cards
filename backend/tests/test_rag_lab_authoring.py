import json

from app.llm_client import LLMMessage
from rag_lab.authoring import author_benchmark
from rag_lab.benchmark import audit_benchmark
from rag_lab.reviews import audit_annotation_review
from rag_lab.schemas import (
    RagBenchmarkSeed,
    RagCorpusCard,
    RagCorpusClaim,
    RagCorpusEvidence,
    RagCorpusSnapshot,
    RagPairQuestionSeed,
    RagSingleQuestionSeed,
    RagUnanswerableQuestionSeed,
)


class FakeQuestionClient:
    def create_chat_completion(
        self,
        messages: list[LLMMessage],
        **_: object,
    ) -> str:
        payload = json.loads(messages[-1].content.split("\n", 1)[1])
        if "source" in payload[0]:
            return json.dumps(
                {
                    "items": [
                        {
                            "index": item["index"],
                            "comparison_question": (
                                f"How are pair concepts {item['index']} related?"
                            ),
                            "multi_hop_question": (
                                f"What follows by combining pair {item['index']}?"
                            ),
                        }
                        for item in payload
                    ]
                }
            )
        return json.dumps(
            {
                "items": [
                    {
                        "index": item["index"],
                        "factual_question": f"What is fact {item['index']}?",
                        "concept_question": f"Why does concept {item['index']} matter?",
                    }
                    for item in payload
                ]
            }
        )


def _card(card_id: str, claim_id: str, evidence_id: str) -> RagCorpusCard:
    return RagCorpusCard(
        card_id=card_id,
        job_id="job-1",
        lecture_name="lecture.mp4",
        title=f"Title {card_id}",
        summary="Summary",
        document_text=f"Document {card_id}",
        content_status="draft",
        source_start_seconds=1.0,
        source_end_seconds=2.0,
        claims=[
            RagCorpusClaim(
                claim_id=claim_id,
                text=f"Supported claim {claim_id}.",
                evidence=[
                    RagCorpusEvidence(
                        evidence_id=evidence_id,
                        quote=f"Supported claim {claim_id}.",
                        start_seconds=1.0,
                        end_seconds=2.0,
                    )
                ],
            )
        ],
    )


def test_author_benchmark_keeps_ids_outside_llm_output() -> None:
    corpus_hash = "a" * 64
    corpus = RagCorpusSnapshot(
        snapshot_id="corpus",
        course_id="course",
        source_database_sha256="b" * 64,
        snapshot_sha256=corpus_hash,
        cards=[_card("card-a", "claim-a", "evidence-a"), _card("card-b", "claim-b", "evidence-b")],
    )
    seed = RagBenchmarkSeed(
        seed_id="seed",
        course_id="course",
        corpus_sha256=corpus_hash,
        single_questions=[
            RagSingleQuestionSeed(
                claim_id="claim-a",
                split="development",
                review_notes="Direct support.",
            )
        ],
        paired_questions=[
            RagPairQuestionSeed(
                source_claim_id="claim-a",
                target_claim_id="claim-b",
                relation_type="related",
                split="test",
                review_notes="Meaningful relation.",
            )
        ],
        unanswerable_questions=[
            RagUnanswerableQuestionSeed(
                question="What absent topic is discussed?",
                split="development",
                review_notes="Absent from corpus.",
            )
        ],
    )

    dataset, review = author_benchmark(
        seed,
        corpus,
        llm_client=FakeQuestionClient(),
        batch_size=1,
    )

    assert len(dataset.items) == 5
    assert dataset.items[0].gold_claim_ids == ["claim-a"]
    assert dataset.items[2].gold_card_ids == ["card-a", "card-b"]
    assert dataset.items[-1].answerable is False
    assert audit_benchmark(dataset, corpus, require_accepted=False)["passed"]
    assert audit_annotation_review(review, corpus)["passed"]
