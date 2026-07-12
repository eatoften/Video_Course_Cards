from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from fsrs import Card as FsrsCard
from fsrs import Rating, Scheduler, State

from . import course_service
from .knowledge_card_store import get_card
from .review import (
    ReviewEvent,
    ReviewProgress,
    ReviewQueue,
    ReviewQueueItem,
    ReviewRating,
    ReviewRatingRequest,
    ReviewRatingResult,
)
from .review_item import ReviewItem
from .review_item_store import get_review_item, list_review_items_for_card
from .review_store import (
    create_review_event,
    get_review_progress,
    upsert_review_progress,
)
from .topic_service import get_course_map


RATING_MAP = {
    "again": Rating.Again,
    "hard": Rating.Hard,
    "good": Rating.Good,
    "easy": Rating.Easy,
}


class ReviewServiceError(Exception):
    pass


class ReviewItemNotFoundError(ReviewServiceError):
    pass


class InvalidReviewRequestError(ReviewServiceError):
    pass


def get_course_review_queue(
    course_id: str,
    *,
    topic_id: str | None = None,
    limit: int = 50,
) -> ReviewQueue:
    if limit < 1 or limit > 200:
        raise InvalidReviewRequestError("Review queue limit must be 1 to 200.")
    course = course_service.get_video_course(course_id)
    course_map = get_course_map(course.id)
    cards_by_id = {card.id: card for card in course_map.cards}
    primary_topic_by_card = {
        membership.card_id: membership.topic_id
        for membership in course_map.memberships
        if membership.role == "primary" and membership.status == "accepted"
    }
    topics_by_id = {topic.id: topic for topic in course_map.topics}
    now = datetime.now(timezone.utc)
    all_items: list[ReviewQueueItem] = []

    for card_id, card in cards_by_id.items():
        stored_card = get_card(card_id)
        if stored_card is None:
            continue
        card_topic_id = primary_topic_by_card.get(card_id)
        if topic_id is not None and card_topic_id != topic_id:
            continue
        for item in list_review_items_for_card(card_id):
            if item.status != "active":
                continue
            progress = _get_or_create_progress(item, now=now)
            all_items.append(
                ReviewQueueItem(
                    review_item=item,
                    progress=progress,
                    phase=progress.phase,
                    card_id=card.id,
                    card_title=card.title,
                    card_summary=card.summary,
                    card_kind=card.card_kind,
                    claims=stored_card.claims,
                    topic_id=card_topic_id,
                    topic_title=(
                        topics_by_id[card_topic_id].title
                        if card_topic_id in topics_by_id
                        else None
                    ),
                    source_start_seconds=card.source_start_seconds,
                    source_end_seconds=card.source_end_seconds,
                )
            )

    due_items = [item for item in all_items if item.progress.due_at <= now]
    due_items.sort(
        key=lambda item: (
            0 if item.phase != "new" else 1,
            item.progress.due_at,
            item.card_title,
        )
    )
    return ReviewQueue(
        course_id=course.id,
        topic_id=topic_id,
        due_count=len(due_items),
        new_count=sum(item.phase == "new" for item in all_items),
        learning_count=sum(item.phase == "learning" for item in all_items),
        review_count=sum(item.phase == "review" for item in all_items),
        relearning_count=sum(item.phase == "relearning" for item in all_items),
        items=due_items[:limit],
    )


def rate_review_item(
    review_item_id: str,
    request: ReviewRatingRequest,
) -> ReviewRatingResult:
    item = get_review_item(review_item_id)
    if item is None:
        raise ReviewItemNotFoundError("Review item not found.")
    if item.status != "active":
        raise InvalidReviewRequestError("Disabled review items cannot be rated.")

    now = datetime.now(timezone.utc)
    progress = _get_or_create_progress(item, now=now)
    previous_phase = progress.phase
    due_before = progress.due_at
    fsrs_card = FsrsCard(
        card_id=progress.fsrs_card_id,
        state=State(progress.fsrs_state),
        step=progress.step,
        stability=progress.stability,
        difficulty=progress.fsrs_difficulty,
        due=progress.due_at,
        last_review=progress.last_reviewed_at,
    )
    next_card, _ = Scheduler().review_card(
        fsrs_card,
        RATING_MAP[request.rating],
        review_datetime=now,
        review_duration=request.response_time_ms,
    )
    lapse_increment = int(
        request.rating == "again" and progress.review_count > 0
    )
    next_progress = ReviewProgress(
        review_item_id=item.id,
        fsrs_card_id=progress.fsrs_card_id,
        fsrs_state=next_card.state.value,
        step=next_card.step,
        due_at=next_card.due,
        stability=next_card.stability,
        fsrs_difficulty=next_card.difficulty,
        last_reviewed_at=next_card.last_review,
        review_count=progress.review_count + 1,
        lapse_count=progress.lapse_count + lapse_increment,
        created_at=progress.created_at,
        updated_at=now,
    )
    event = ReviewEvent(
        id=uuid4().hex,
        review_item_id=item.id,
        rating=request.rating,
        reviewed_at=now,
        response_time_ms=request.response_time_ms,
        previous_phase=previous_phase,
        next_phase=next_progress.phase,
        due_before=due_before,
        due_after=next_progress.due_at,
        scheduled_days=max(
            0.0,
            (next_progress.due_at - now).total_seconds() / 86400,
        ),
    )
    upsert_review_progress(next_progress)
    create_review_event(event)
    return ReviewRatingResult(progress=next_progress, event=event)


def _get_or_create_progress(
    item: ReviewItem,
    *,
    now: datetime,
) -> ReviewProgress:
    progress = get_review_progress(item.id)
    if progress is not None:
        return progress
    progress = ReviewProgress(
        review_item_id=item.id,
        fsrs_card_id=_fsrs_card_id(item.id),
        fsrs_state=State.Learning.value,
        step=0,
        due_at=now,
        created_at=now,
        updated_at=now,
    )
    upsert_review_progress(progress)
    return progress


def _fsrs_card_id(item_id: str) -> int:
    try:
        return int(item_id[:15], 16)
    except ValueError:
        return abs(hash(item_id)) % (2**53)
