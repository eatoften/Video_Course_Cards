from pydantic import BaseModel, Field


class TopicLearningCoverage(BaseModel):
    topic_id: str
    card_count: int = 0
    cards_with_review_items: int = 0
    review_item_count: int = 0
    due_review_item_count: int = 0
    cards_with_learning_documents: int = 0
    learning_document_count: int = 0


class CourseLearningCoverage(BaseModel):
    total_cards: int = 0
    cards_with_review_items: int = 0
    review_item_count: int = 0
    due_review_item_count: int = 0
    cards_with_learning_documents: int = 0
    learning_document_count: int = 0
    source_asset_count: int = 0
    unsorted_card_count: int = 0
    topic_coverage: list[TopicLearningCoverage] = Field(default_factory=list)
