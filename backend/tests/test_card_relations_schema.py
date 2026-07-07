import sqlite3

import pytest

from app.db import connect


def test_card_relations_table_has_expected_columns():
    with connect() as conn:
        rows = conn.execute("PRAGMA table_info(card_relations)").fetchall()

    columns = {
        row["name"]: row["type"]
        for row in rows
    }

    assert columns == {
        "id": "TEXT",
        "course_id": "TEXT",
        "source_card_id": "TEXT",
        "target_card_id": "TEXT",
        "relation_type": "TEXT",
        "score": "REAL",
        "method": "TEXT",
        "model": "TEXT",
        "explanation": "TEXT",
        "status": "TEXT",
        "created_at": "TEXT",
        "updated_at": "TEXT",
    }


def test_card_relations_table_has_expected_indexes():
    with connect() as conn:
        indexes = {
            row["name"]: row["unique"]
            for row in conn.execute("PRAGMA index_list(card_relations)")
        }

        unique_index_columns = [
            row["name"]
            for row in conn.execute(
                "PRAGMA index_info(idx_card_relations_unique_pair_type)"
            )
        ]

    assert indexes["idx_card_relations_course_id"] == 0
    assert indexes["idx_card_relations_source_card_id"] == 0
    assert indexes["idx_card_relations_target_card_id"] == 0
    assert indexes["idx_card_relations_unique_pair_type"] == 1
    assert unique_index_columns == [
        "source_card_id",
        "target_card_id",
        "relation_type",
        "method",
    ]


def test_card_relations_unique_pair_type_method_constraint():
    relation = (
        "relation-1",
        "course-1",
        "card-a",
        "card-b",
        "semantic_similarity",
        0.84,
        "cosine_similarity",
        "sentence-transformers/all-MiniLM-L6-v2",
        None,
        "suggested",
        "2026-07-07T00:00:00",
        "2026-07-07T00:00:00",
    )

    with connect() as conn:
        conn.execute(
            """
            INSERT INTO card_relations (
                id,
                course_id,
                source_card_id,
                target_card_id,
                relation_type,
                score,
                method,
                model,
                explanation,
                status,
                created_at,
                updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            relation,
        )

        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                """
                INSERT INTO card_relations (
                    id,
                    course_id,
                    source_card_id,
                    target_card_id,
                    relation_type,
                    score,
                    method,
                    model,
                    explanation,
                    status,
                    created_at,
                    updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "relation-2",
                    *relation[1:],
                ),
            )
