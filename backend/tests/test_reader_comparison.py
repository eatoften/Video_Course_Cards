from pathlib import Path

import pytest

from multimodal_lab.reader_comparison import (
    claim_sealed_test_access,
    finalize_test_access,
    paired_bootstrap_cer_difference,
    score_reader_predictions,
)
from multimodal_lab.schemas import ReaderPrediction


def make_prediction(
    sample_id: str,
    reference: str,
    prediction: str,
) -> ReaderPrediction:
    return ReaderPrediction(
        sample_id=sample_id,
        reference=reference,
        prediction=prediction,
        scored_reference=reference,
        scored_prediction=prediction,
        exact_match=reference == prediction,
    )


def test_reader_slice_metrics_use_corpus_level_denominators() -> None:
    predictions = [
        make_prediction("one", "abc", "abc"),
        make_prediction("two", "long line", "long lane"),
    ]

    metrics = score_reader_predictions(predictions)

    assert metrics.sample_count == 2
    assert metrics.character_edits == 1
    assert metrics.character_count == 12
    assert metrics.character_error_rate == pytest.approx(1 / 12)
    assert metrics.word_edits == 1
    assert metrics.word_count == 3
    assert metrics.exact_match_count == 1


def test_paired_bootstrap_is_deterministic_and_preserves_pairing() -> None:
    better = [
        make_prediction(f"sample-{index}", "abcd", "abcd")
        for index in range(4)
    ]
    worse = [
        make_prediction(f"sample-{index}", "abcd", "abxd")
        for index in range(4)
    ]

    first = paired_bootstrap_cer_difference(
        "better",
        better,
        "worse",
        worse,
        seed=17,
        iterations=1000,
        confidence_level=0.95,
    )
    second = paired_bootstrap_cer_difference(
        "better",
        better,
        "worse",
        worse,
        seed=17,
        iterations=1000,
        confidence_level=0.95,
    )

    assert first == second
    assert first.point_difference_a_minus_b == pytest.approx(-0.25)
    assert first.upper_bound < 0
    assert first.probability_a_lower_than_b == 1


def test_sealed_test_ledger_refuses_a_second_open(tmp_path: Path) -> None:
    ledger_path = tmp_path / "test_access_ledger.json"
    payload = claim_sealed_test_access(
        ledger_path,
        protocol_id="reader-comparison",
        protocol_sha256="a" * 64,
        run_id="run-1",
    )
    finalize_test_access(
        ledger_path,
        payload,
        status="completed",
    )

    with pytest.raises(RuntimeError, match="already been opened"):
        claim_sealed_test_access(
            ledger_path,
            protocol_id="reader-comparison",
            protocol_sha256="a" * 64,
            run_id="run-2",
        )
