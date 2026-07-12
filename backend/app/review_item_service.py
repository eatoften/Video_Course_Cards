from __future__ import annotations

from uuid import uuid4

from .job import utc_now
from .knowledge_card_store import get_card
from .review_item import ReviewItem, ReviewItemCreate, ReviewItemUpdate
from .review_item_store import (
    create_review_item,
    delete_review_item,
    get_review_item,
    list_review_items_for_card,
    update_review_item,
)


class ReviewItemServiceError(Exception):
    pass


class ReviewItemNotFoundError(ReviewItemServiceError):
    pass


class ReviewItemCardNotFoundError(ReviewItemServiceError):
    pass


class InvalidReviewItemError(ReviewItemServiceError):
    pass


def list_card_review_items(card_id: str) -> list[ReviewItem]:
    _require_card(card_id)
    return list_review_items_for_card(card_id)


def save_card_review_item(
    card_id: str,
    request: ReviewItemCreate,
) -> ReviewItem:
    _require_card(card_id)
    item = _build_review_item(card_id, request)
    create_review_item(item)
    return item


def save_initial_review_items(
    card_id: str,
    requests: list[ReviewItemCreate],
) -> list[ReviewItem]:
    return [save_card_review_item(card_id, request) for request in requests]


def update_saved_review_item(
    item_id: str,
    request: ReviewItemUpdate,
) -> ReviewItem:
    item = get_review_item(item_id)
    if item is None:
        raise ReviewItemNotFoundError("Review item not found.")

    update_data = request.model_dump(exclude_unset=True)
    if not update_data:
        raise InvalidReviewItemError(
            "At least one review item field is required."
        )

    if request.item_type is not None:
        item.item_type = request.item_type
    if request.prompt is not None:
        item.prompt = request.prompt.strip()
    if request.expected_answer is not None:
        item.expected_answer = request.expected_answer.strip()
    if request.source_claim_ids is not None:
        item.source_claim_ids = _clean_claim_ids(request.source_claim_ids)
    if request.status is not None:
        item.status = request.status
    item.updated_at = utc_now()
    update_review_item(item)
    return item


def delete_saved_review_item(item_id: str) -> None:
    if get_review_item(item_id) is None:
        raise ReviewItemNotFoundError("Review item not found.")
    delete_review_item(item_id)


def _build_review_item(card_id: str, request: ReviewItemCreate) -> ReviewItem:
    now = utc_now()
    return ReviewItem(
        id=uuid4().hex,
        card_id=card_id,
        item_type=request.item_type,
        prompt=request.prompt.strip(),
        expected_answer=request.expected_answer.strip(),
        source_claim_ids=_clean_claim_ids(request.source_claim_ids),
        source=request.source,
        status=request.status,
        created_at=now,
        updated_at=now,
    )


def _require_card(card_id: str) -> None:
    if get_card(card_id) is None:
        raise ReviewItemCardNotFoundError("Knowledge card not found.")


def _clean_claim_ids(values: list[str]) -> list[str]:
    cleaned: list[str] = []
    for value in values:
        normalized = value.strip()
        if normalized and normalized not in cleaned:
            cleaned.append(normalized)
    return cleaned
