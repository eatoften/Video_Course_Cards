from fastapi.testclient import TestClient

import app.main as main
from app.db import configure_db, connect, init_db
from app.job import VideoJob, VideoJobStatus
from app.job_store import create_job


client = TestClient(main.app)


def test_init_db_drops_legacy_cards_without_claims(tmp_path):
    db_path = tmp_path / "legacy.db"
    configure_db(db_path)

    with connect() as conn:
        conn.execute(
            """
            CREATE TABLE knowledge_cards (
                id TEXT PRIMARY KEY,
                job_id TEXT NOT NULL,
                title TEXT NOT NULL,
                summary TEXT NOT NULL,
                key_points TEXT NOT NULL,
                question TEXT,
                answer TEXT,
                difficulty TEXT NOT NULL,
                source_start_seconds REAL NOT NULL,
                source_end_seconds REAL NOT NULL,
                provider TEXT,
                model TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            INSERT INTO knowledge_cards (
                id,
                job_id,
                title,
                summary,
                key_points,
                difficulty,
                source_start_seconds,
                source_end_seconds,
                created_at,
                updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "legacy-card",
                "job-123",
                "Old Card",
                "No claims.",
                "[]",
                "easy",
                0.0,
                1.0,
                "2026-06-01T00:00:00",
                "2026-06-01T00:00:00",
            ),
        )

    init_db()

    with connect() as conn:
        columns = {
            row["name"]
            for row in conn.execute("PRAGMA table_info(knowledge_cards)")
        }
        rows = conn.execute("SELECT * FROM knowledge_cards").fetchall()

    assert "claims" in columns
    assert "unsupported_terms" in columns
    assert "tags" in columns
    assert "review_state" in columns
    assert rows == []


def test_init_db_adds_card_organization_columns_without_dropping(tmp_path):
    db_path = tmp_path / "cards.db"
    configure_db(db_path)

    with connect() as conn:
        conn.execute(
            """
            CREATE TABLE knowledge_cards (
                id TEXT PRIMARY KEY,
                job_id TEXT NOT NULL,
                title TEXT NOT NULL,
                summary TEXT NOT NULL,
                key_points TEXT NOT NULL,
                claims TEXT NOT NULL,
                unsupported_terms TEXT NOT NULL,
                question TEXT,
                answer TEXT,
                difficulty TEXT NOT NULL,
                source_start_seconds REAL NOT NULL,
                source_end_seconds REAL NOT NULL,
                provider TEXT,
                model TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            INSERT INTO knowledge_cards (
                id,
                job_id,
                title,
                summary,
                key_points,
                claims,
                unsupported_terms,
                difficulty,
                source_start_seconds,
                source_end_seconds,
                created_at,
                updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "card-123",
                "job-123",
                "SVD",
                "A matrix factorization.",
                "[]",
                "[]",
                "[]",
                "medium",
                0.0,
                1.0,
                "2026-06-01T00:00:00",
                "2026-06-01T00:00:00",
            ),
        )

    init_db()

    with connect() as conn:
        row = conn.execute(
            """
            SELECT id, tags, review_state
            FROM knowledge_cards
            WHERE id = ?
            """,
            ("card-123",),
        ).fetchone()

    assert dict(row) == {
        "id": "card-123",
        "tags": "[]",
        "review_state": "draft",
    }


def create_uploaded_job(tmp_path, job_id: str = "job-123") -> VideoJob:
    job = VideoJob(
        id=job_id,
        video_path=tmp_path / "lecture.mp4",
        status=VideoJobStatus.completed,
        original_filename="lecture.mp4",
    )
    create_job(job)

    return job


def card_payload() -> dict:
    return {
        "title": "Linear Algebra",
        "summary": "Linear algebra is one of the course's core subjects.",
        "key_points": [
            "Symmetric matrices",
            "Orthogonal matrices",
        ],
        "claims": [
            {
                "text": (
                    "Linear algebra is one of the course's core subjects."
                ),
                "evidence": [
                    {
                        "quote": "The first big subject is linear algebra.",
                        "segment_start_seconds": 36.44,
                        "segment_end_seconds": 44.0,
                    }
                ],
            }
        ],
        "unsupported_terms": [],
        "question": "What is the first big subject?",
        "answer": "Linear algebra.",
        "difficulty": "easy",
        "source_start_seconds": 36.44,
        "source_end_seconds": 92.68,
        "provider": "ollama",
        "model": "qwen3:4b",
    }


def test_save_and_list_job_cards(tmp_path):
    job = create_uploaded_job(tmp_path)

    create_response = client.post(
        f"/jobs/{job.id}/cards",
        json=card_payload(),
    )

    assert create_response.status_code == 201

    created_card = create_response.json()

    assert created_card["id"]
    assert created_card["job_id"] == job.id
    assert created_card["title"] == "Linear Algebra"
    assert created_card["key_points"] == [
        "Symmetric matrices",
        "Orthogonal matrices",
    ]
    assert created_card["claims"] == card_payload()["claims"]
    assert created_card["unsupported_terms"] == []
    assert created_card["tags"] == []
    assert created_card["review_state"] == "draft"
    assert created_card["created_at"]
    assert created_card["updated_at"]

    list_response = client.get(
        f"/jobs/{job.id}/cards"
    )

    assert list_response.status_code == 200
    assert [
        card["id"]
        for card in list_response.json()
    ] == [created_card["id"]]


def test_get_saved_card(tmp_path):
    job = create_uploaded_job(tmp_path)
    created_card = client.post(
        f"/jobs/{job.id}/cards",
        json=card_payload(),
    ).json()

    response = client.get(f"/cards/{created_card['id']}")

    assert response.status_code == 200
    assert response.json() == created_card


def test_update_saved_card(tmp_path):
    job = create_uploaded_job(tmp_path)
    created_card = client.post(
        f"/jobs/{job.id}/cards",
        json=card_payload(),
    ).json()

    response = client.patch(
        f"/cards/{created_card['id']}",
        json={
            "title": "Singular Value Decomposition",
            "summary": "SVD is a critical matrix factorization.",
            "key_points": [
                "Matrix factorization",
                "Orthogonal matrices",
            ],
            "difficulty": "medium",
            "tags": [
                "SVD",
                "matrix factorization",
                "svd",
            ],
            "review_state": "reviewed",
        },
    )

    assert response.status_code == 200

    updated_card = response.json()

    assert updated_card["id"] == created_card["id"]
    assert updated_card["title"] == "Singular Value Decomposition"
    assert updated_card["summary"] == (
        "SVD is a critical matrix factorization."
    )
    assert updated_card["key_points"] == [
        "Matrix factorization",
        "Orthogonal matrices",
    ]
    assert updated_card["claims"] == created_card["claims"]
    assert updated_card["difficulty"] == "medium"
    assert updated_card["tags"] == [
        "svd",
        "matrix factorization",
    ]
    assert updated_card["review_state"] == "reviewed"
    assert updated_card["updated_at"] >= created_card["updated_at"]


def test_delete_saved_card(tmp_path):
    job = create_uploaded_job(tmp_path)
    created_card = client.post(
        f"/jobs/{job.id}/cards",
        json=card_payload(),
    ).json()

    delete_response = client.delete(
        f"/cards/{created_card['id']}"
    )

    assert delete_response.status_code == 204

    list_response = client.get(
        f"/jobs/{job.id}/cards"
    )

    assert list_response.status_code == 200
    assert list_response.json() == []


def test_save_card_returns_404_for_missing_job():
    response = client.post(
        "/jobs/missing-job/cards",
        json=card_payload(),
    )

    assert response.status_code == 404
    assert response.json() == {
        "detail": "Job not found."
    }


def test_save_card_returns_400_for_invalid_time_range(tmp_path):
    job = create_uploaded_job(tmp_path)
    payload = card_payload()
    payload["source_end_seconds"] = payload["source_start_seconds"]

    response = client.post(
        f"/jobs/{job.id}/cards",
        json=payload,
    )

    assert response.status_code == 400
    assert response.json() == {
        "detail": "Card source end must be greater than start."
    }


def test_save_card_requires_grounded_claim(tmp_path):
    job = create_uploaded_job(tmp_path)
    payload = card_payload()
    payload["claims"] = [
        {
            "text": "   ",
            "evidence": [
                {
                    "quote": "   ",
                    "segment_start_seconds": 1.0,
                    "segment_end_seconds": 2.0,
                }
            ],
        }
    ]

    response = client.post(
        f"/jobs/{job.id}/cards",
        json=payload,
    )

    assert response.status_code == 400
    assert response.json() == {
        "detail": (
            "Knowledge card must include at least one grounded claim."
        )
    }


def test_update_card_returns_404_for_missing_card():
    response = client.patch(
        "/cards/missing-card",
        json={
            "title": "Missing",
        },
    )

    assert response.status_code == 404
    assert response.json() == {
        "detail": "Knowledge card not found."
    }


def test_list_cards_returns_404_for_missing_job():
    response = client.get(
        "/jobs/missing-job/cards"
    )

    assert response.status_code == 404
    assert response.json() == {
        "detail": "Job not found."
    }
