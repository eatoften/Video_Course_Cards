from __future__ import annotations

from .knowledge_card import KnowledgeCardIndexItem
from .learning_coverage import CourseLearningCoverage, TopicLearningCoverage
from .learning_document_store import (
    document_counts_by_card,
    document_ids_by_card,
    due_review_counts_by_card,
    list_learning_documents_for_course,
)
from .source_asset_store import list_source_assets_for_course
from .topic import Topic, TopicCardMembership


def build_course_learning_coverage(
    course_id: str,
    *,
    topics: list[Topic],
    memberships: list[TopicCardMembership],
    cards: list[KnowledgeCardIndexItem],
) -> CourseLearningCoverage:
    document_counts = document_counts_by_card(course_id)
    document_ids = document_ids_by_card(course_id)
    due_counts = due_review_counts_by_card(course_id)
    primary_topic_by_card = {
        membership.card_id: membership.topic_id
        for membership in memberships
        if membership.role == "primary" and membership.status == "accepted"
    }
    topic_cards: dict[str, list[KnowledgeCardIndexItem]] = {
        topic.id: [] for topic in topics
    }
    for card in cards:
        topic_id = primary_topic_by_card.get(card.id)
        if topic_id in topic_cards:
            topic_cards[topic_id].append(card)

    topic_coverage = []
    for topic in topics:
        assigned_cards = topic_cards[topic.id]
        topic_coverage.append(
            TopicLearningCoverage(
                topic_id=topic.id,
                card_count=len(assigned_cards),
                cards_with_review_items=sum(
                    card.review_item_count > 0 for card in assigned_cards
                ),
                review_item_count=sum(
                    card.review_item_count for card in assigned_cards
                ),
                due_review_item_count=sum(
                    due_counts.get(card.id, 0) for card in assigned_cards
                ),
                cards_with_learning_documents=sum(
                    document_counts.get(card.id, 0) > 0 for card in assigned_cards
                ),
                learning_document_count=len(
                    set().union(
                        *(document_ids.get(card.id, set()) for card in assigned_cards)
                    )
                    if assigned_cards
                    else set()
                ),
            )
        )

    unsorted_topic_ids = {
        topic.id for topic in topics if topic.is_system
    }
    return CourseLearningCoverage(
        total_cards=len(cards),
        cards_with_review_items=sum(card.review_item_count > 0 for card in cards),
        review_item_count=sum(card.review_item_count for card in cards),
        due_review_item_count=sum(due_counts.values()),
        cards_with_learning_documents=sum(
            document_counts.get(card.id, 0) > 0 for card in cards
        ),
        learning_document_count=len(list_learning_documents_for_course(course_id)),
        source_asset_count=len(list_source_assets_for_course(course_id)),
        unsorted_card_count=sum(
            primary_topic_by_card.get(card.id) in unsorted_topic_ids
            for card in cards
        ),
        topic_coverage=topic_coverage,
    )
