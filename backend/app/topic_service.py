from __future__ import annotations

from sqlite3 import IntegrityError
from uuid import uuid4

from . import course_service
from .job import utc_now
from .knowledge_card_store import get_card, list_card_index_for_course
from .learning_coverage_service import build_course_learning_coverage
from .topic import (
    CourseMap,
    SetPrimaryTopicRequest,
    Topic,
    TopicCardMembership,
    TopicCreate,
    TopicRelation,
    TopicRelationCreate,
    TopicMergeRequest,
    TopicSplitRequest,
    TopicUpdate,
)
from .topic_store import (
    create_topic,
    create_topic_relation,
    delete_suggested_topic,
    delete_topic_and_rehome,
    delete_topic_relation,
    get_topic,
    list_memberships_for_course,
    list_memberships_for_topic,
    list_topic_relations_for_course,
    list_topics_for_course,
    next_topic_position,
    update_topic,
    upsert_primary_membership,
)


MAX_TOPIC_DEPTH = 3
UNSORTED_TITLE = "Unsorted"


class TopicServiceError(Exception):
    pass


class TopicNotFoundError(TopicServiceError):
    pass


class TopicCardNotFoundError(TopicServiceError):
    pass


class InvalidTopicError(TopicServiceError):
    pass


def get_course_map(course_id: str) -> CourseMap:
    course = course_service.get_video_course(course_id)
    unsorted = ensure_unsorted_topic(course.id)
    cards = list_card_index_for_course(course.id)
    memberships = list_memberships_for_course(course.id)
    assigned_card_ids = {
        membership.card_id
        for membership in memberships
        if membership.role == "primary" and membership.status == "accepted"
    }

    for card in cards:
        if card.id not in assigned_card_ids:
            _set_primary_membership(
                card.id,
                unsorted,
                position=len(assigned_card_ids),
                method="system",
            )
            assigned_card_ids.add(card.id)

    topics = list_topics_for_course(course.id)
    memberships = list_memberships_for_course(course.id)
    return CourseMap(
        course_id=course.id,
        topics=topics,
        memberships=memberships,
        topic_relations=list_topic_relations_for_course(course.id),
        cards=cards,
        coverage=build_course_learning_coverage(
            course.id,
            topics=topics,
            memberships=memberships,
            cards=cards,
        ),
    )


def ensure_unsorted_topic(course_id: str) -> Topic:
    existing = next(
        (
            topic
            for topic in list_topics_for_course(course_id)
            if topic.is_system and topic.title == UNSORTED_TITLE
        ),
        None,
    )
    if existing is not None:
        return existing

    now = utc_now()
    topic = Topic(
        id=uuid4().hex,
        course_id=course_id,
        title=UNSORTED_TITLE,
        position=0,
        depth=0,
        method="system",
        status="accepted",
        is_system=True,
        created_at=now,
        updated_at=now,
    )
    create_topic(topic)
    return topic


def create_course_topic(course_id: str, request: TopicCreate) -> Topic:
    course = course_service.get_video_course(course_id)
    parent = _optional_parent(course.id, request.parent_topic_id)
    depth = 0 if parent is None else parent.depth + 1
    _validate_depth(depth)
    now = utc_now()
    topic = Topic(
        id=uuid4().hex,
        course_id=course.id,
        parent_topic_id=parent.id if parent else None,
        title=request.title.strip(),
        summary=_clean_optional(request.summary),
        position=(
            request.position
            if request.position is not None
            else next_topic_position(course.id, parent.id if parent else None)
        ),
        depth=depth,
        method="manual",
        status="accepted",
        created_at=now,
        updated_at=now,
    )
    create_topic(topic)
    return topic


def update_course_topic(topic_id: str, request: TopicUpdate) -> Topic:
    topic = _require_topic(topic_id)
    original_depth = topic.depth
    subtree_relative_depth = max(
        (
            descendant.depth - original_depth
            for descendant in _descendants(topic)
        ),
        default=0,
    )
    if topic.is_system and request.parent_topic_id is not None:
        raise InvalidTopicError("The Unsorted topic cannot be nested.")
    data = request.model_dump(exclude_unset=True)
    if not data:
        raise InvalidTopicError("At least one topic field is required.")

    if request.title is not None:
        if topic.is_system and request.title.strip() != UNSORTED_TITLE:
            raise InvalidTopicError("The Unsorted topic cannot be renamed.")
        topic.title = request.title.strip()
    if "summary" in data:
        topic.summary = _clean_optional(request.summary)
    if "parent_topic_id" in data:
        parent = _optional_parent(topic.course_id, request.parent_topic_id)
        if parent is not None:
            if parent.id == topic.id or _is_descendant(parent.id, topic.id):
                raise InvalidTopicError("Topic hierarchy cannot contain a cycle.")
            topic.depth = parent.depth + 1
            topic.parent_topic_id = parent.id
        else:
            topic.depth = 0
            topic.parent_topic_id = None
        _validate_depth(topic.depth + subtree_relative_depth)
    if request.position is not None:
        topic.position = request.position
    if request.status is not None:
        if topic.is_system and request.status != "accepted":
            raise InvalidTopicError("The Unsorted topic must stay visible.")
        topic.status = request.status
    topic.updated_at = utc_now()
    update_topic(topic)
    _refresh_descendant_depths(topic)
    return topic


def delete_course_topic(topic_id: str) -> None:
    topic = _require_topic(topic_id)
    if topic.is_system:
        raise InvalidTopicError("The Unsorted topic cannot be deleted.")
    if topic.status == "suggested":
        delete_suggested_topic(topic.id)
        return
    unsorted = ensure_unsorted_topic(topic.course_id)
    delete_topic_and_rehome(
        topic.id,
        fallback_topic_id=unsorted.id,
        parent_topic_id=topic.parent_topic_id,
    )
    _refresh_all_depths(topic.course_id)


def accept_suggested_topic(topic_id: str) -> Topic:
    topic = _require_topic(topic_id)
    if topic.status != "suggested":
        raise InvalidTopicError("Only suggested topics can be accepted.")
    memberships = list_memberships_for_topic(topic.id)
    now = utc_now()
    for membership in memberships:
        membership.status = "accepted"
        membership.role = "primary"
        membership.updated_at = now
        upsert_primary_membership(membership)
    topic.status = "accepted"
    topic.updated_at = now
    update_topic(topic)
    return topic


def merge_course_topics(
    target_topic_id: str,
    request: TopicMergeRequest,
) -> Topic:
    target = _require_topic(target_topic_id)
    if target.status != "accepted":
        raise InvalidTopicError("Merge target must be an accepted topic.")
    source_ids = list(dict.fromkeys(request.source_topic_ids))
    if target.id in source_ids:
        raise InvalidTopicError("A topic cannot be merged into itself.")
    sources = [_require_course_topic(topic_id, target.course_id) for topic_id in source_ids]
    if any(topic.is_system or topic.status != "accepted" for topic in sources):
        raise InvalidTopicError("Only accepted non-system topics can be merged.")
    if any(_is_descendant(target.id, source.id) for source in sources):
        raise InvalidTopicError("A topic cannot be merged into its descendant.")
    for source in sources:
        for membership in list_memberships_for_topic(source.id):
            if membership.status == "accepted" and membership.role == "primary":
                _set_primary_membership(
                    membership.card_id,
                    target,
                    position=membership.position,
                    method="manual",
                )
        delete_topic_and_rehome(
            source.id,
            fallback_topic_id=target.id,
            parent_topic_id=target.id,
        )
    _refresh_all_depths(target.course_id)
    return _require_topic(target.id)


def split_course_topic(
    source_topic_id: str,
    request: TopicSplitRequest,
) -> Topic:
    source = _require_topic(source_topic_id)
    if source.status != "accepted":
        raise InvalidTopicError("Only accepted topics can be split.")
    memberships = {
        membership.card_id: membership
        for membership in list_memberships_for_topic(source.id)
        if membership.status == "accepted" and membership.role == "primary"
    }
    card_ids = list(dict.fromkeys(request.card_ids))
    if any(card_id not in memberships for card_id in card_ids):
        raise InvalidTopicError("Every split card must belong to the source topic.")
    created = create_course_topic(
        source.course_id,
        TopicCreate(
            title=request.title,
            summary=request.summary,
            parent_topic_id=source.parent_topic_id,
        ),
    )
    for position, card_id in enumerate(card_ids):
        _set_primary_membership(card_id, created, position=position, method="manual")
    return created


def set_card_primary_topic(
    card_id: str,
    request: SetPrimaryTopicRequest,
) -> TopicCardMembership:
    card = get_card(card_id)
    if card is None:
        raise TopicCardNotFoundError("Knowledge card not found.")
    topic = _require_topic(request.topic_id)
    course = course_service.get_video_course(topic.course_id)
    course_card_ids = {card.id for card in list_card_index_for_course(course.id)}
    if card_id not in course_card_ids:
        raise InvalidTopicError("Card and topic must belong to the same course.")
    return _set_primary_membership(
        card_id,
        topic,
        position=request.position,
        method="manual",
    )


def create_course_topic_relation(
    course_id: str,
    request: TopicRelationCreate,
) -> TopicRelation:
    course = course_service.get_video_course(course_id)
    source = _require_course_topic(request.source_topic_id, course.id)
    target = _require_course_topic(request.target_topic_id, course.id)
    now = utc_now()
    relation = TopicRelation(
        id=uuid4().hex,
        course_id=course.id,
        source_topic_id=source.id,
        target_topic_id=target.id,
        relation_type=request.relation_type,
        explanation=_clean_optional(request.explanation),
        method="manual",
        status="accepted",
        created_at=now,
        updated_at=now,
    )
    try:
        create_topic_relation(relation)
    except IntegrityError as exc:
        raise InvalidTopicError("This topic relation already exists.") from exc
    return relation


def delete_course_topic_relation(relation_id: str) -> None:
    if not delete_topic_relation(relation_id):
        raise TopicNotFoundError("Topic relation not found.")


def _set_primary_membership(
    card_id: str,
    topic: Topic,
    *,
    position: int | None,
    method: str,
) -> TopicCardMembership:
    now = utc_now()
    membership = TopicCardMembership(
        id=uuid4().hex,
        topic_id=topic.id,
        card_id=card_id,
        role="primary",
        position=position or 0,
        method=method,
        status="accepted",
        created_at=now,
        updated_at=now,
    )
    upsert_primary_membership(membership)
    return membership


def _require_topic(topic_id: str) -> Topic:
    topic = get_topic(topic_id)
    if topic is None:
        raise TopicNotFoundError("Topic not found.")
    return topic


def _require_course_topic(topic_id: str, course_id: str) -> Topic:
    topic = _require_topic(topic_id)
    if topic.course_id != course_id:
        raise InvalidTopicError("Topic does not belong to this course.")
    return topic


def _optional_parent(course_id: str, topic_id: str | None) -> Topic | None:
    if topic_id is None:
        return None
    return _require_course_topic(topic_id, course_id)


def _is_descendant(candidate_id: str, ancestor_id: str) -> bool:
    current = get_topic(candidate_id)
    while current is not None and current.parent_topic_id is not None:
        if current.parent_topic_id == ancestor_id:
            return True
        current = get_topic(current.parent_topic_id)
    return False


def _validate_depth(depth: int) -> None:
    if depth > MAX_TOPIC_DEPTH:
        raise InvalidTopicError(
            f"Topic depth cannot exceed {MAX_TOPIC_DEPTH}."
        )


def _descendants(topic: Topic) -> list[Topic]:
    topics = list_topics_for_course(topic.course_id)
    descendants: list[Topic] = []
    pending = [topic.id]
    while pending:
        parent_id = pending.pop()
        children = [item for item in topics if item.parent_topic_id == parent_id]
        descendants.extend(children)
        pending.extend(child.id for child in children)
    return descendants


def _refresh_descendant_depths(topic: Topic) -> None:
    topics = list_topics_for_course(topic.course_id)
    pending = [topic]
    while pending:
        parent = pending.pop()
        for child in [item for item in topics if item.parent_topic_id == parent.id]:
            child.depth = parent.depth + 1
            _validate_depth(child.depth)
            child.updated_at = utc_now()
            update_topic(child)
            pending.append(child)


def _refresh_all_depths(course_id: str) -> None:
    topics = list_topics_for_course(course_id)
    for root in [topic for topic in topics if topic.parent_topic_id is None]:
        root.depth = 0
        update_topic(root)
        _refresh_descendant_depths(root)


def _clean_optional(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = value.strip()
    return cleaned or None
