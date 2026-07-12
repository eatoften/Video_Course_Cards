from __future__ import annotations

from sqlite3 import IntegrityError
from uuid import uuid4

from . import course_service
from .card_embedding_store import list_card_embeddings_for_course
from .card_relation import (
    CardRelatedCardsResponse,
    CardRelation,
    CardRelationClassificationResult,
    CardRelationClassifyRequest,
    CardRelationCreate,
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
from .card_relation_classifier import (
    RelationClassificationError,
    RelationClassificationOutputError,
    RelationClassificationTimeoutError,
    RelationClassifierClient,
    classify_card_relation,
)
from .card_relation_store import (
    create_card_relation,
    delete_card_relation,
    get_card_relation,
    get_card_relation_by_identity,
    list_card_relations_for_course,
    list_related_card_relations,
    replace_suggested_relations_for_course,
    update_card_relation,
    upsert_card_relations,
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


class CardRelationClassificationError(CardRelationServiceError):
    pass


class CardRelationClassificationTimeoutError(
    CardRelationClassificationError
):
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
        if relation.source_card_id in cards_by_id
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


def create_manual_card_relation(
    course_id: str,
    request: CardRelationCreate,
) -> CardRelation:
    course = course_service.get_video_course(course_id)
    cards_by_id = {
        card.id: card
        for card in list_cards_for_course(course.id)
    }
    _require_course_card(request.source_card_id, cards_by_id)
    _require_course_card(request.target_card_id, cards_by_id)
    now = utc_now()
    relation = CardRelation(
        id=uuid4().hex,
        course_id=course.id,
        source_card_id=request.source_card_id,
        target_card_id=request.target_card_id,
        relation_type=request.relation_type,
        score=1.0,
        method="manual",
        explanation=_clean_explanation(request.explanation),
        status=request.status,
        created_at=now,
        updated_at=now,
    )

    try:
        create_card_relation(relation)
    except IntegrityError as exc:
        raise InvalidCardRelationRequestError(
            "This manual card relation already exists."
        ) from exc

    return relation


def classify_saved_card_relation(
    relation_id: str,
    request: CardRelationClassifyRequest,
    *,
    llm_client: RelationClassifierClient,
) -> CardRelationClassificationResult:
    source_relation = get_card_relation(relation_id)

    if source_relation is None:
        raise CardRelationNotFoundError("Card relation not found.")

    source_card = get_card(source_relation.source_card_id)
    target_card = get_card(source_relation.target_card_id)

    if source_card is None or target_card is None:
        raise CardRelationCardNotFoundError(
            "A card referenced by this relation no longer exists."
        )

    try:
        classification = classify_card_relation(
            source_card,
            target_card,
            llm_client=llm_client,
            model=request.model,
        )
    except RelationClassificationTimeoutError as exc:
        raise CardRelationClassificationTimeoutError(str(exc)) from exc
    except (
        RelationClassificationOutputError,
        RelationClassificationError,
    ) as exc:
        raise CardRelationClassificationError(str(exc)) from exc

    selected_model = (
        request.model.strip()
        if request.model and request.model.strip()
        else llm_client.settings.model
    )

    if classification.relation_type == "unclear":
        return CardRelationClassificationResult(
            source_relation_id=source_relation.id,
            classification="unclear",
            explanation=classification.explanation,
            model=selected_model,
        )

    now = utc_now()
    classified_relation = CardRelation(
        id=uuid4().hex,
        course_id=source_relation.course_id,
        source_card_id=source_relation.source_card_id,
        target_card_id=source_relation.target_card_id,
        relation_type=classification.relation_type,
        score=source_relation.score,
        method="local_llm",
        model=selected_model,
        explanation=classification.explanation,
        status="suggested",
        created_at=now,
        updated_at=now,
    )
    upsert_card_relations([classified_relation])
    stored_relation = get_card_relation_by_identity(
        classified_relation.source_card_id,
        classified_relation.target_card_id,
        classified_relation.relation_type,
        classified_relation.method,
    )

    if stored_relation is None:
        raise CardRelationClassificationError(
            "Classified relation could not be saved."
        )

    if source_relation.method == "cosine_similarity":
        source_relation.status = "hidden"
        source_relation.updated_at = now
        update_card_relation(source_relation)

    return CardRelationClassificationResult(
        source_relation_id=source_relation.id,
        classification=classification.relation_type,
        explanation=classification.explanation,
        model=selected_model,
        relation=stored_relation,
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
                content_status=target_card.content_status,
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


def _require_course_card(
    card_id: str,
    cards_by_id: dict[str, KnowledgeCard],
) -> KnowledgeCard:
    card = cards_by_id.get(card_id)

    if card is None:
        raise CardRelationCardNotFoundError(
            "Both cards must exist in the selected course."
        )

    return card


def _clean_explanation(value: str | None) -> str | None:
    if value is None:
        return None

    cleaned = value.strip()
    return cleaned or None


def _card_to_graph_node(card: KnowledgeCard) -> CardRelationGraphNode:
    return CardRelationGraphNode(
        id=card.id,
        job_id=card.job_id,
        title=card.title,
        summary=card.summary,
        tags=card.tags,
        content_status=card.content_status,
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
