from datetime import UTC, datetime

import pytest

from app.card_service import CardDraftResponse
from multimodal_lab.reader_card_cascade import (
    ReaderCardCascadeProtocol,
    ReaderCardGenerationRecord,
    ReaderCardGenerationStatus,
    card_input_sha256,
)
from multimodal_lab.reader_card_evaluation import (
    ReaderCardDecisionBundle,
    ReaderCardReviewRecord,
    apply_reader_card_decisions,
    best_source_line_match,
    build_reader_card_review_template,
    evaluate_reader_card_cascade,
    paired_binary_bootstrap_difference,
)
from multimodal_lab.schemas import StablePageReference


def make_protocol() -> ReaderCardCascadeProtocol:
    digest = "a" * 64
    return ReaderCardCascadeProtocol(
        protocol_id="card-cascade",
        source_comparison_protocol_id="comparison",
        source_comparison_path="comparison.json",
        source_comparison_sha256=digest,
        dataset_path="dataset.jsonl",
        dataset_sha256=digest,
        split_path="split.json",
        split_sha256=digest,
        test_lecture_id="lecture-5",
        references_path="references.jsonl",
        references_sha256=digest,
        vocabulary_sha256=digest,
        card_service_path="card_service.py",
        card_service_sha256=digest,
        text_sources=[
            {
                "system_name": "cnn_v2",
                "source_kind": "reader_evaluation",
                "path": "cnn.json",
                "sha256": digest,
            },
            {
                "system_name": "vit_v1",
                "source_kind": "reader_evaluation",
                "path": "vit.json",
                "sha256": digest,
            },
            {
                "system_name": "rapidocr_stored",
                "source_kind": "rapidocr_reviews",
                "path": "rapid.jsonl",
                "sha256": digest,
            },
        ],
        base_url="http://localhost:11434/v1",
        model="qwen3:4b",
        model_digest=digest,
        max_tokens=8192,
        timeout_seconds=300,
        focus="Recover the main concept.",
        review_method="Frozen single-auditor review.",
        citation_similarity_threshold=0.85,
        bootstrap_seed=7,
        bootstrap_iterations=1000,
        confidence_level=0.95,
    )


def make_response(system_name: str) -> CardDraftResponse:
    return CardDraftResponse.model_validate(
        {
            "job_id": "page-1",
            "source_video": "lecture-5/page-1",
            "start_seconds": 30,
            "end_seconds": 31,
            "provider": "ollama",
            "model": "qwen3:4b",
            "generation_metadata": {
                "provider": "ollama",
                "model": "qwen3:4b",
                "elapsed_seconds": 1,
                "input_characters": 100,
                "selected_context_characters": 28,
                "selected_segments_count": 1,
                "requested_card_count": 1,
                "raw_card_count": 1,
                "returned_card_count": 1,
                "raw_claim_count": 1,
                "grounded_claim_count": 1,
                "dropped_claim_count": 0,
                "unsupported_terms_count": 0,
                "max_context_characters": 4000,
                "max_selected_segments": 80,
            },
            "cards": [
                {
                    "title": f"Linear Classification ({system_name})",
                    "summary": "A linear classifier uses f(x) = Wx.",
                    "key_points": ["It maps inputs to class scores."],
                    "claims": [
                        {
                            "text": "A linear classifier uses f(x) = Wx.",
                            "evidence": [
                                {
                                    "quote": "Linear classifier uses f(x) = Wx",
                                    "segment_start_seconds": 30,
                                    "segment_end_seconds": 31,
                                }
                            ],
                        }
                    ],
                    "question": "What function does it use?",
                    "answer": "f(x) = Wx.",
                    "source_start_seconds": 30,
                    "source_end_seconds": 31,
                }
            ],
        }
    )


def make_record(system_name: str) -> ReaderCardGenerationRecord:
    digest = "a" * 64
    return ReaderCardGenerationRecord(
        protocol_id="card-cascade",
        protocol_sha256=digest,
        model="qwen3:4b",
        model_digest=digest,
        system_name=system_name,
        page_event_id="page-1",
        lecture_id="lecture-5",
        page_number=1,
        stable_frame_timestamp=30,
        schedule_position=0,
        sample_ids=["sample-1"],
        input_lines=["Linear classifier uses f(x) = Wx"],
        input_sha256=digest,
        status=ReaderCardGenerationStatus.succeeded,
        generated_at=datetime.now(UTC),
        elapsed_seconds=1,
        response=make_response(system_name),
    )


def make_reference() -> StablePageReference:
    return StablePageReference(
        page_event_id="page-1",
        lecture_id="lecture-5",
        stable_frame_timestamp=30,
        image_path="page.png",
        page_number=1,
        gold_text="Linear classifier uses f(x) = Wx",
        gold_concepts=["linear classification"],
    )


def complete_review(review: ReaderCardReviewRecord) -> ReaderCardReviewRecord:
    claims = []
    for claim in review.claim_reviews:
        claims.append(
            claim.model_copy(
                update={
                    "supported_by_source": True,
                    "citations": [
                        citation.model_copy(
                            update={"correct_against_source": True}
                        )
                        for citation in claim.citations
                    ],
                }
            )
        )
    return ReaderCardReviewRecord.model_validate(
        review.model_copy(
            update={
                "concept_recovered": True,
                "claim_reviews": claims,
                "accepted_without_edit": True,
                "usable_card": True,
                "completed": True,
                "reviewer_id": "reviewer",
                "review_method": "source audit",
                "reviewed_at": datetime.now(UTC),
            }
        ).model_dump()
    )


def test_card_input_hash_is_deterministic_and_order_sensitive() -> None:
    first = card_input_sha256("cnn", "page", ["a", "b"], ["one", "two"])
    second = card_input_sha256("cnn", "page", ["a", "b"], ["one", "two"])
    reordered = card_input_sha256("cnn", "page", ["b", "a"], ["two", "one"])

    assert first == second
    assert first != reordered


def test_source_line_match_preserves_ocr_error_as_a_similarity() -> None:
    line, score = best_source_line_match(
        "Linear classifer uses f(x) = Wx",
        ["Unrelated", "Linear classifier uses f(x) = Wx"],
    )

    assert line == "Linear classifier uses f(x) = Wx"
    assert score > 0.95


def test_review_template_and_evaluation_use_the_same_three_systems() -> None:
    protocol = make_protocol()
    records = [
        make_record(system_name)
        for system_name in ("cnn_v2", "vit_v1", "rapidocr_stored")
    ]
    reference = make_reference()
    reviews = build_reader_card_review_template(
        records,
        [reference],
        protocol=protocol,
        protocol_sha256="a" * 64,
    )
    completed_reviews = [complete_review(review) for review in reviews]

    report = evaluate_reader_card_cascade(
        records,
        completed_reviews,
        [reference],
        protocol=protocol,
        protocol_sha256="a" * 64,
        generation_records_sha256="b" * 64,
        review_records_sha256="c" * 64,
    )

    assert set(report.systems) == {"cnn_v2", "vit_v1", "rapidocr_stored"}
    assert report.systems["cnn_v2"].metrics.concept_recall == 1
    assert report.systems["vit_v1"].metrics.grounded_claim_precision == 1
    assert report.systems["rapidocr_stored"].metrics.usable_card_conversion == 1


def test_paired_binary_bootstrap_is_deterministic() -> None:
    first = paired_binary_bootstrap_difference(
        "cnn_v2",
        [1, 1, 1, 1],
        "vit_v1",
        [0, 0, 0, 0],
        metric="usable_card_conversion",
        seed=11,
        iterations=1000,
        confidence_level=0.95,
    )
    second = paired_binary_bootstrap_difference(
        "cnn_v2",
        [1, 1, 1, 1],
        "vit_v1",
        [0, 0, 0, 0],
        metric="usable_card_conversion",
        seed=11,
        iterations=1000,
        confidence_level=0.95,
    )

    assert first == second
    assert first.point_difference_a_minus_b == pytest.approx(1)
    assert first.lower_bound == pytest.approx(1)


def test_decision_bundle_rejects_claim_or_citation_drift() -> None:
    protocol = make_protocol()
    record = make_record("cnn_v2")
    review = build_reader_card_review_template(
        [record],
        [make_reference()],
        protocol=protocol,
        protocol_sha256="a" * 64,
    )[0]
    bundle = ReaderCardDecisionBundle(
        protocol_id="card-cascade",
        protocol_sha256="a" * 64,
        reviewer_id="reviewer",
        review_method="source audit",
        decisions=[
            {
                "system_name": "cnn_v2",
                "page_event_id": "page-1",
                "concept_recovered": True,
                "supported_claims": [True],
                "correct_citations": [[True]],
                "accepted_without_edit": True,
                "usable_card": True,
                "notes": "The card matches the source.",
            }
        ],
    )

    completed = apply_reader_card_decisions([review], bundle)

    assert completed[0].completed is True
    assert completed[0].claim_reviews[0].supported_by_source is True

    broken = bundle.model_copy(
        update={
            "decisions": [
                bundle.decisions[0].model_copy(
                    update={"correct_citations": []}
                )
            ]
        }
    )
    with pytest.raises(ValueError, match="Citation decisions do not align"):
        apply_reader_card_decisions([review], broken)
