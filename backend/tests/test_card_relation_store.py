from app.card_relation import CardRelation
from app.card_relation_store import (
    delete_card_relation,
    get_card_relation,
    list_card_relations_for_course,
    list_related_card_relations,
    replace_suggested_relations_for_course,
    update_card_relation,
    upsert_card_relations,
)


def make_relation(
    relation_id: str,
    source_card_id: str,
    target_card_id: str,
    *,
    score: float = 0.8,
    status: str = "suggested",
) -> CardRelation:
    return CardRelation(
        id=relation_id,
        course_id="course-1",
        source_card_id=source_card_id,
        target_card_id=target_card_id,
        relation_type="semantic_similarity",
        score=score,
        method="cosine_similarity",
        model="sentence-transformers/all-MiniLM-L6-v2",
        status=status,
    )


def test_card_relation_store_round_trips_and_updates_relation():
    relation = make_relation("relation-1", "card-a", "card-b")

    upsert_card_relations([relation])

    loaded = get_card_relation(relation.id)

    assert loaded == relation

    relation.status = "accepted"
    relation.explanation = "They cover the same optimization idea."

    update_card_relation(relation)

    updated = get_card_relation(relation.id)

    assert updated is not None
    assert updated.status == "accepted"
    assert updated.explanation == "They cover the same optimization idea."


def test_replace_suggested_relations_preserves_reviewed_relations():
    accepted = make_relation(
        "accepted-relation",
        "card-a",
        "card-b",
        score=0.7,
        status="accepted",
    )
    old_suggested = make_relation(
        "old-suggested",
        "card-a",
        "card-c",
        score=0.6,
    )
    upsert_card_relations([accepted, old_suggested])

    replace_suggested_relations_for_course(
        "course-1",
        [
            make_relation("new-conflict", "card-a", "card-b", score=0.95),
            make_relation("new-suggested", "card-a", "card-d", score=0.9),
        ],
        relation_type="semantic_similarity",
        method="cosine_similarity",
    )

    relations = list_card_relations_for_course("course-1")
    relations_by_id = {
        relation.id: relation
        for relation in relations
    }

    assert "old-suggested" not in relations_by_id
    assert relations_by_id["accepted-relation"].score == 0.7
    assert relations_by_id["accepted-relation"].status == "accepted"
    assert relations_by_id["new-suggested"].score == 0.9


def test_list_related_and_delete_relation():
    first_relation = make_relation("relation-1", "card-a", "card-b")
    second_relation = make_relation("relation-2", "card-b", "card-a")
    upsert_card_relations([first_relation, second_relation])

    related = list_related_card_relations("card-a")

    assert [
        relation.id
        for relation in related
    ] == ["relation-1"]

    delete_card_relation("relation-1")

    assert get_card_relation("relation-1") is None
