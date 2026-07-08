from __future__ import annotations

from uuid import uuid4

from . import course_service
from .card_embedding_store import list_card_embeddings_for_course
from .card_relation import (
    CardRelatedCardsResponse,
    CardRelation,
    CardRelationGraphEdge,
    CardRelationGraphNode,
    CardRelationRecomputeRequest,
    CardRelationRecomputeResult,
    CardRelationUpdate,
    CourseCardRelationsGraph,
    DEFAULT_CARD_RELATION_METHOD,
    DEFAULT_CARD_RELATION_STATUS,
    DEFAULT_CARD_RELATION_TYPE,
    RelatedCard,
)
from .card_relation_store import (
    delete_card_relation,
    get_card_relation,
    list_card_relations_for_course,
    list_related_card_relations,
    replace_suggested_relations_for_course,
    update_card_relation,
)
from .card_similarity import build_similarity_candidates
from .job import utc_now
from .knowledge_card import KnowledgeCard
from .knowledge_card_store import get_card, list_cards_for_course


VISIBLE_RELATION_STATUSES = {"suggested", "accepted"}


class CardRelationServiceError(Exception):
    pass


class CardRelationNotFoundError(CardRelationServiceError):
    pass


class CardRelationCardNotFoundError(CardRelationServiceError):
    pass


class InvalidCardRelationRequestError(CardRelationServiceError):
    pass


def recompute_course_card_relations(
    course_id: str,
    request: CardRelationRecomputeRequest | None = None,
) -> CardRelationRecomputeResult:
    active_request = request or CardRelationRecomputeRequest()
    course = course_service.get_video_course(course_id)
    cards = list_cards_for_course(course.id)
    embeddings = list_card_embeddings_for_course(course.id)
    card_ids = {card.id for card in cards}
    usable_embeddings = [
        embedding
        for embedding in embeddings
        if embedding.card_id in card_ids
    ]

    try:
        candidates = build_similarity_candidates(
            usable_embeddings,
            top_k=active_request.top_k,
            threshold=active_request.threshold,
        )
    except ValueError as exc:
        raise InvalidCardRelationRequestError(str(exc)) from exc

    now = utc_now()
    relations = [
        CardRelation(
            id=uuid4().hex,
            course_id=course.id,
            source_card_id=candidate.source_card_id,
            target_card_id=candidate.target_card_id,
            relation_type=DEFAULT_CARD_RELATION_TYPE,
            score=candidate.score,
            method=DEFAULT_CARD_RELATION_METHOD,
            model=candidate.model,
            status=DEFAULT_CARD_RELATION_STATUS,
            created_at=now,
            updated_at=now,
        )
        for candidate in candidates
    ]

    replace_suggested_relations_for_course(
        course.id,
        relations,
        relation_type=DEFAULT_CARD_RELATION_TYPE,
        method=DEFAULT_CARD_RELATION_METHOD,
    )

    return CardRelationRecomputeResult(
        course_id=course.id,
        total_cards=len(cards),
        embedded_cards=len(usable_embeddings),
        skipped_cards=len(cards) - len(usable_embeddings),
        relations_written=len(relations),
        threshold=active_request.threshold,
        top_k=active_request.top_k,
    )


def get_course_card_relations_graph(
    course_id: str,
) -> CourseCardRelationsGraph:
    course = course_service.get_video_course(course_id)
    cards = list_cards_for_course(course.id)
    cards_by_id = {card.id: card for card in cards}
    relations = [
        relation
        for relation in list_card_relations_for_course(course.id)
        if _is_visible_relation(relation)
        and relation.source_card_id in cards_by_id
        and relation.target_card_id in cards_by_id
    ]

    return CourseCardRelationsGraph(
        course_id=course.id,
        nodes=[
            _card_to_graph_node(card)
            for card in cards
        ],
        edges=[
            _relation_to_graph_edge(relation)
            for relation in relations
        ],
    )


def get_related_cards(card_id: str) -> CardRelatedCardsResponse:
    source_card = get_card(card_id)

    if source_card is None:
        raise CardRelationCardNotFoundError("Knowledge card not found.")

    related: list[RelatedCard] = []

    for relation in list_related_card_relations(source_card.id):
        if not _is_visible_relation(relation):
            continue

        target_card = get_card(relation.target_card_id)

        if target_card is None:
            continue

        related.append(
            RelatedCard(
                relation_id=relation.id,
                card_id=target_card.id,
                job_id=target_card.job_id,
                title=target_card.title,
                summary=target_card.summary,
                tags=target_card.tags,
                review_state=target_card.review_state,
                relation_type=relation.relation_type,
                score=relation.score,
                method=relation.method,
                status=relation.status,
                explanation=relation.explanation,
                source_start_seconds=target_card.source_start_seconds,
                source_end_seconds=target_card.source_end_seconds,
            )
        )

    return CardRelatedCardsResponse(
        card_id=source_card.id,
        related=related,
    )


def update_saved_card_relation(
    relation_id: str,
    request: CardRelationUpdate,
) -> CardRelation:
    relation = get_card_relation(relation_id)

    if relation is None:
        raise CardRelationNotFoundError("Card relation not found.")

    update_data = request.model_dump(exclude_unset=True)

    if not update_data:
        raise InvalidCardRelationRequestError(
            "At least one card relation field is required."
        )

    if "relation_type" in update_data:
        relation.relation_type = request.relation_type or relation.relation_type

    if "explanation" in update_data:
        relation.explanation = request.explanation

    if "status" in update_data:
        relation.status = request.status or relation.status

    relation.updated_at = utc_now()
    update_card_relation(relation)

    updated_relation = get_card_relation(relation.id)

    if updated_relation is None:
        raise CardRelationNotFoundError("Card relation not found.")

    return updated_relation


def delete_saved_card_relation(relation_id: str) -> None:
    relation = get_card_relation(relation_id)

    if relation is None:
        raise CardRelationNotFoundError("Card relation not found.")

    delete_card_relation(relation.id)


def _is_visible_relation(relation: CardRelation) -> bool:
    return relation.status in VISIBLE_RELATION_STATUSES


def _card_to_graph_node(card: KnowledgeCard) -> CardRelationGraphNode:
    return CardRelationGraphNode(
        id=card.id,
        job_id=card.job_id,
        title=card.title,
        summary=card.summary,
        tags=card.tags,
        review_state=card.review_state,
        source_start_seconds=card.source_start_seconds,
        source_end_seconds=card.source_end_seconds,
    )


def _relation_to_graph_edge(
    relation: CardRelation,
) -> CardRelationGraphEdge:
    return CardRelationGraphEdge(
        id=relation.id,
        source=relation.source_card_id,
        target=relation.target_card_id,
        relation_type=relation.relation_type,
        score=relation.score,
        method=relation.method,
        status=relation.status,
        explanation=relation.explanation,
    )
