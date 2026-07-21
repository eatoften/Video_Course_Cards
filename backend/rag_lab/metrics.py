from __future__ import annotations

import math
import random
import statistics
from collections import defaultdict
from collections.abc import Iterable, Sequence

from .schemas import (
    RagBenchmarkItem,
    RagQuestionCategory,
    RetrievalMetricSlice,
    RetrievalRecord,
    RetrievalSystemReport,
)


def paired_bootstrap_metric_difference(
    items: Sequence[RagBenchmarkItem],
    left_records: Sequence[RetrievalRecord],
    right_records: Sequence[RetrievalRecord],
    *,
    metric: str,
    k: int,
    iterations: int,
    seed: int,
) -> dict[str, float | int | list[float]]:
    if iterations < 100:
        raise ValueError("Bootstrap requires at least 100 iterations.")
    left_pairs = _aligned_pairs(items, left_records)
    right_pairs = _aligned_pairs(items, right_records)
    right_by_id = {item.question_id: record for item, record in right_pairs}
    differences = [
        _question_metric(item, left_record, metric=metric, k=k)
        - _question_metric(
            item,
            right_by_id[item.question_id],
            metric=metric,
            k=k,
        )
        for item, left_record in left_pairs
    ]
    if not differences:
        raise ValueError("Bootstrap comparison requires at least one question.")
    observed = statistics.fmean(differences)
    generator = random.Random(seed)
    bootstrapped = sorted(
        statistics.fmean(
            differences[generator.randrange(len(differences))]
            for _ in differences
        )
        for _ in range(iterations)
    )
    lower = _quantile_nearest(bootstrapped, 0.025)
    upper = _quantile_nearest(bootstrapped, 0.975)
    probability_better = sum(value > 0 for value in bootstrapped) / iterations
    return {
        "question_count": len(differences),
        "observed_difference": observed,
        "confidence_interval_95": [lower, upper],
        "bootstrap_probability_left_better": probability_better,
        "iterations": iterations,
        "seed": seed,
    }


def select_confidence_threshold(
    items: Sequence[RagBenchmarkItem],
    records: Sequence[RetrievalRecord],
) -> float:
    pairs = _aligned_pairs(items, records)
    scores = sorted(
        {
            _record_confidence(record)
            for _, record in pairs
            if record.ranked_cards
        }
    )
    if not scores:
        return math.inf
    candidates = [math.nextafter(scores[0], -math.inf), *scores, math.inf]
    best_threshold = candidates[0]
    best_key = (-1.0, -1.0, -math.inf)
    for threshold in candidates:
        true_positive = false_positive = false_negative = 0
        for item, record in pairs:
            predicted_answerable = bool(record.ranked_cards) and (
                _record_confidence(record) >= threshold
            )
            if item.answerable and predicted_answerable:
                true_positive += 1
            elif not item.answerable and predicted_answerable:
                false_positive += 1
            elif item.answerable and not predicted_answerable:
                false_negative += 1
        f1 = _f1(true_positive, false_positive, false_negative)
        false_rate = false_positive / max(1, sum(not item.answerable for item, _ in pairs))
        key = (f1, -false_rate, threshold)
        if key > best_key:
            best_key = key
            best_threshold = threshold
    return best_threshold


def evaluate_retrieval_system(
    items: Sequence[RagBenchmarkItem],
    records: Sequence[RetrievalRecord],
    *,
    top_k_values: Sequence[int],
    confidence_threshold: float,
) -> RetrievalSystemReport:
    pairs = _aligned_pairs(items, records)
    if not pairs:
        raise ValueError("Retrieval evaluation requires at least one record.")
    systems = {record.system for _, record in pairs}
    splits = {record.split for _, record in pairs}
    if len(systems) != 1 or len(splits) != 1:
        raise ValueError("One report must contain one system and one split.")
    by_category: dict[RagQuestionCategory, RetrievalMetricSlice] = {}
    for category in sorted({item.category for item, _ in pairs}):
        category_pairs = [(item, record) for item, record in pairs if item.category == category]
        by_category[category] = _evaluate_slice(
            category_pairs,
            top_k_values=top_k_values,
            confidence_threshold=confidence_threshold,
        )
    return RetrievalSystemReport(
        system=next(iter(systems)),
        split=next(iter(splits)),
        overall=_evaluate_slice(
            pairs,
            top_k_values=top_k_values,
            confidence_threshold=confidence_threshold,
        ),
        by_category=by_category,
    )


def _evaluate_slice(
    pairs: Sequence[tuple[RagBenchmarkItem, RetrievalRecord]],
    *,
    top_k_values: Sequence[int],
    confidence_threshold: float,
) -> RetrievalMetricSlice:
    answerable = [(item, record) for item, record in pairs if item.answerable]
    unanswerable = [(item, record) for item, record in pairs if not item.answerable]
    hit_rate: dict[int, float] = {}
    set_recall: dict[int, float] = {}
    joint_recall: dict[int, float] = {}
    ndcg: dict[int, float] = {}
    for k in top_k_values:
        hits = recalls = joints = gains = 0.0
        for item, record in answerable:
            retrieved = [ranked.card_id for ranked in record.ranked_cards[:k]]
            relevant = set(item.gold_card_ids)
            overlap = relevant.intersection(retrieved)
            hits += float(bool(overlap))
            recalls += len(overlap) / len(relevant)
            joints += float(relevant.issubset(retrieved))
            dcg = sum(
                1.0 / math.log2(rank + 1)
                for rank, card_id in enumerate(retrieved, start=1)
                if card_id in relevant
            )
            ideal_count = min(k, len(relevant))
            idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_count + 1))
            gains += dcg / idcg if idcg else 0.0
        denominator = max(1, len(answerable))
        hit_rate[k] = hits / denominator
        set_recall[k] = recalls / denominator
        joint_recall[k] = joints / denominator
        ndcg[k] = gains / denominator

    reciprocal_ranks = []
    for item, record in answerable:
        relevant = set(item.gold_card_ids)
        first_rank = next(
            (ranked.rank for ranked in record.ranked_cards if ranked.card_id in relevant),
            None,
        )
        reciprocal_ranks.append(0.0 if first_rank is None else 1.0 / first_rank)

    false_positive = true_positive = false_negative = 0
    for item, record in pairs:
        predicted_answerable = bool(record.ranked_cards) and (
            _record_confidence(record) >= confidence_threshold
        )
        if item.answerable and predicted_answerable:
            true_positive += 1
        elif not item.answerable and predicted_answerable:
            false_positive += 1
        elif item.answerable and not predicted_answerable:
            false_negative += 1
    false_retrieval_rate = false_positive / max(1, len(unanswerable))
    latencies = [record.elapsed_milliseconds for _, record in pairs]
    return RetrievalMetricSlice(
        question_count=len(pairs),
        answerable_count=len(answerable),
        unanswerable_count=len(unanswerable),
        hit_rate_at_k=hit_rate,
        set_recall_at_k=set_recall,
        joint_recall_at_k=joint_recall,
        ndcg_at_k=ndcg,
        mean_reciprocal_rank=statistics.fmean(reciprocal_ranks) if reciprocal_ranks else 0.0,
        unanswerable_false_retrieval_rate=false_retrieval_rate,
        answerability_f1=_f1(true_positive, false_positive, false_negative),
        confidence_threshold=confidence_threshold,
        mean_latency_milliseconds=statistics.fmean(latencies) if latencies else 0.0,
        median_latency_milliseconds=statistics.median(latencies) if latencies else 0.0,
        p95_latency_milliseconds=_percentile(latencies, 0.95),
    )


def _aligned_pairs(
    items: Sequence[RagBenchmarkItem],
    records: Sequence[RetrievalRecord],
) -> list[tuple[RagBenchmarkItem, RetrievalRecord]]:
    items_by_id = {item.question_id: item for item in items}
    if len(items_by_id) != len(items):
        raise ValueError("Benchmark item ids must be unique.")
    records_by_id = {record.question_id: record for record in records}
    if len(records_by_id) != len(records):
        raise ValueError("Retrieval record ids must be unique.")
    if set(items_by_id) != set(records_by_id):
        raise ValueError("Retrieval records must match benchmark items exactly.")
    pairs = []
    for question_id in sorted(items_by_id):
        item = items_by_id[question_id]
        record = records_by_id[question_id]
        if record.category != item.category or record.split != item.split:
            raise ValueError(f"Record metadata mismatch for {question_id}.")
        pairs.append((item, record))
    return pairs


def _f1(true_positive: int, false_positive: int, false_negative: int) -> float:
    denominator = 2 * true_positive + false_positive + false_negative
    return 0.0 if denominator == 0 else 2 * true_positive / denominator


def _record_confidence(record: RetrievalRecord) -> float:
    if record.confidence_score is not None:
        return record.confidence_score
    if not record.ranked_cards:
        return -math.inf
    return record.ranked_cards[0].score


def _question_metric(
    item: RagBenchmarkItem,
    record: RetrievalRecord,
    *,
    metric: str,
    k: int,
) -> float:
    if not item.answerable:
        raise ValueError("Retrieval relevance metrics require answerable questions.")
    relevant = set(item.gold_card_ids)
    retrieved = [ranked.card_id for ranked in record.ranked_cards[:k]]
    if metric == "set_recall":
        return len(relevant.intersection(retrieved)) / len(relevant)
    if metric == "joint_recall":
        return float(relevant.issubset(retrieved))
    if metric == "reciprocal_rank":
        first_rank = next(
            (
                rank
                for rank, card_id in enumerate(retrieved, start=1)
                if card_id in relevant
            ),
            None,
        )
        return 0.0 if first_rank is None else 1.0 / first_rank
    if metric == "ndcg":
        dcg = sum(
            1.0 / math.log2(rank + 1)
            for rank, card_id in enumerate(retrieved, start=1)
            if card_id in relevant
        )
        ideal_count = min(k, len(relevant))
        idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_count + 1))
        return dcg / idcg if idcg else 0.0
    raise ValueError(f"Unsupported bootstrap metric: {metric}")


def _percentile(values: Iterable[float], quantile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = max(0, math.ceil(quantile * len(ordered)) - 1)
    return float(ordered[index])


def _quantile_nearest(values: Sequence[float], quantile: float) -> float:
    if not values:
        raise ValueError("Quantile requires at least one value.")
    index = round((len(values) - 1) * quantile)
    return float(values[index])
