from __future__ import annotations

import math
from collections import Counter
from uuid import uuid4

import numpy as np
from sklearn.cluster import AgglomerativeClustering

from . import course_service
from .card_embedding import CardEmbedding
from .card_embedding_store import list_card_embeddings_for_course
from .embedding import cosine_similarity
from .job import utc_now
from .topic import (
    Topic,
    TopicCardMembership,
    TopicSuggestionRequest,
    TopicSuggestionResult,
)
from .topic_service import get_course_map
from .topic_store import (
    clear_suggested_topics_for_course,
    create_topic,
    next_topic_position,
    upsert_topic_membership,
)
from .topic_suggestion_namer import (
    TopicNamingClient,
    TopicNamingError,
    name_topic_clusters,
)


class TopicSuggestionError(Exception):
    pass


def suggest_course_topics(
    course_id: str,
    request: TopicSuggestionRequest,
    *,
    llm_client: TopicNamingClient,
) -> TopicSuggestionResult:
    course = course_service.get_video_course(course_id)
    course_map = get_course_map(course.id)
    unsorted = next((topic for topic in course_map.topics if topic.is_system), None)
    if unsorted is None:
        raise TopicSuggestionError("Course has no Unsorted topic.")
    unsorted_card_ids = {
        membership.card_id
        for membership in course_map.memberships
        if membership.topic_id == unsorted.id
        and membership.role == "primary"
        and membership.status == "accepted"
    }
    cards = [card for card in course_map.cards if card.id in unsorted_card_ids]
    embeddings = _select_compatible_embeddings(
        list_card_embeddings_for_course(course.id),
        {card.id for card in cards},
    )
    cards_by_id = {card.id: card for card in cards}
    eligible_ids = [embedding.card_id for embedding in embeddings if embedding.card_id in cards_by_id]
    if len(eligible_ids) < 2:
        raise TopicSuggestionError(
            "At least two Unsorted cards with compatible embeddings are required."
        )

    target_count = request.target_topic_count or max(
        2,
        min(12, round(math.sqrt(len(eligible_ids) / 2))),
    )
    target_count = min(target_count, len(eligible_ids))
    feature_matrix = _build_feature_matrix(
        [cards_by_id[card_id] for card_id in eligible_ids],
        embeddings,
    )
    labels = AgglomerativeClustering(
        n_clusters=target_count,
        metric="cosine",
        linkage="average",
    ).fit_predict(feature_matrix)
    cluster_card_ids: dict[int, list[str]] = {}
    for card_id, label in zip(eligible_ids, labels, strict=True):
        cluster_card_ids.setdefault(int(label), []).append(card_id)

    names = _fallback_names(cluster_card_ids, cards_by_id)
    naming_method = "embedding_cluster"
    warning = None
    if request.use_local_llm:
        try:
            llm_names = name_topic_clusters(
                _cluster_prompt_payload(cluster_card_ids, cards_by_id),
                llm_client=llm_client,
                model=request.model,
            )
            for cluster_id, topic_name in llm_names.items():
                if cluster_id in names:
                    names[cluster_id] = (topic_name.title, topic_name.summary)
            naming_method = "local_llm"
        except TopicNamingError as exc:
            warning = f"Used deterministic names because local LLM naming failed: {exc}"

    clear_suggested_topics_for_course(course.id)
    now = utc_now()
    suggested_topics: list[Topic] = []
    membership_count = 0
    embeddings_by_id = {embedding.card_id: embedding for embedding in embeddings}
    first_position = next_topic_position(course.id, None)
    for offset, cluster_id in enumerate(sorted(cluster_card_ids)):
        title, summary = names[cluster_id]
        topic = Topic(
            id=uuid4().hex,
            course_id=course.id,
            title=title,
            summary=summary,
            position=first_position + offset,
            depth=0,
            method=naming_method,
            status="suggested",
            created_at=now,
            updated_at=now,
        )
        create_topic(topic)
        suggested_topics.append(topic)
        cluster_embeddings = [
            embeddings_by_id[card_id] for card_id in cluster_card_ids[cluster_id]
        ]
        centroid = np.mean(
            np.asarray([embedding.vector for embedding in cluster_embeddings]),
            axis=0,
        ).tolist()
        for position, card_id in enumerate(cluster_card_ids[cluster_id]):
            confidence = max(
                0.0,
                min(1.0, cosine_similarity(embeddings_by_id[card_id].vector, centroid)),
            )
            upsert_topic_membership(
                TopicCardMembership(
                    id=uuid4().hex,
                    topic_id=topic.id,
                    card_id=card_id,
                    role="primary",
                    position=position,
                    method="embedding_cluster",
                    confidence=confidence,
                    status="suggested",
                    created_at=now,
                    updated_at=now,
                )
            )
            membership_count += 1

    cluster_sizes = [
        len(cluster_card_ids[cluster_id])
        for cluster_id in sorted(cluster_card_ids)
    ]

    return TopicSuggestionResult(
        course_id=course.id,
        eligible_cards=len(eligible_ids),
        suggested_topics=suggested_topics,
        suggested_memberships=membership_count,
        embedding_model=embeddings[0].model,
        naming_method=naming_method,
        warning=warning,
        mean_coherence=_mean_cluster_coherence(
            cluster_card_ids,
            embeddings_by_id,
        ),
        singleton_topic_count=sum(size == 1 for size in cluster_sizes),
        largest_topic_size=max(cluster_sizes, default=0),
        cluster_sizes=cluster_sizes,
    )


def _select_compatible_embeddings(
    embeddings: list[CardEmbedding],
    card_ids: set[str],
) -> list[CardEmbedding]:
    candidates = [embedding for embedding in embeddings if embedding.card_id in card_ids]
    if not candidates:
        return []
    groups = Counter((embedding.model, embedding.dimension) for embedding in candidates)
    model_dimension, _ = groups.most_common(1)[0]
    return [
        embedding
        for embedding in candidates
        if (embedding.model, embedding.dimension) == model_dimension
    ]


def _build_feature_matrix(cards, embeddings: list[CardEmbedding]) -> np.ndarray:
    embeddings_by_id = {embedding.card_id: embedding for embedding in embeddings}
    all_tags = sorted({tag for card in cards for tag in card.tags})[:64]
    tag_index = {tag: index for index, tag in enumerate(all_tags)}
    job_ids = sorted({card.job_id for card in cards})
    job_index = {job_id: index for index, job_id in enumerate(job_ids)}
    max_time_by_job = {
        job_id: max(
            (card.source_start_seconds for card in cards if card.job_id == job_id),
            default=1.0,
        )
        for job_id in job_ids
    }
    rows: list[np.ndarray] = []
    for card in cards:
        semantic = np.asarray(embeddings_by_id[card.id].vector, dtype=np.float64)
        semantic /= max(np.linalg.norm(semantic), 1e-12)
        tags = np.zeros(len(all_tags), dtype=np.float64)
        for tag in card.tags:
            if tag in tag_index:
                tags[tag_index[tag]] = 1.0
        if np.linalg.norm(tags) > 0:
            tags /= np.linalg.norm(tags)
        source = np.zeros(len(job_ids) + 1, dtype=np.float64)
        source[job_index[card.job_id]] = 1.0
        source[-1] = card.source_start_seconds / max(max_time_by_job[card.job_id], 1.0)
        rows.append(np.concatenate([semantic * 0.85, tags * 0.25, source * 0.15]))
    matrix = np.asarray(rows)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


def _fallback_names(cluster_card_ids, cards_by_id) -> dict[int, tuple[str, str]]:
    result: dict[int, tuple[str, str]] = {}
    for cluster_id, card_ids in cluster_card_ids.items():
        cards = [cards_by_id[card_id] for card_id in card_ids]
        tags = Counter(tag for card in cards for tag in card.tags)
        title = tags.most_common(1)[0][0].title() if tags else cards[0].title
        result[cluster_id] = (
            title,
            f"Suggested cluster containing {len(cards)} semantically related cards.",
        )
    return result


def _cluster_prompt_payload(cluster_card_ids, cards_by_id):
    return [
        {
            "cluster_id": cluster_id,
            "cards": [
                {
                    "title": cards_by_id[card_id].title,
                    "summary": cards_by_id[card_id].summary,
                    "tags": cards_by_id[card_id].tags,
                }
                for card_id in card_ids[:20]
            ],
        }
        for cluster_id, card_ids in sorted(cluster_card_ids.items())
    ]


def _mean_cluster_coherence(
    cluster_card_ids: dict[int, list[str]],
    embeddings_by_id: dict[str, CardEmbedding],
) -> float | None:
    scores: list[float] = []
    for card_ids in cluster_card_ids.values():
        for left_index in range(len(card_ids)):
            for right_index in range(left_index + 1, len(card_ids)):
                scores.append(
                    cosine_similarity(
                        embeddings_by_id[card_ids[left_index]].vector,
                        embeddings_by_id[card_ids[right_index]].vector,
                    )
                )
    if not scores:
        return None
    return round(sum(scores) / len(scores), 4)
