import pytest

from app.db import configure_db, init_db
from app.card_generation_run_store import clear_runs
from app.card_relation_store import clear_card_relations
from app.job_store import clear_jobs
from app.knowledge_card_store import clear_cards
from app.knowledge_card_note_store import clear_notes
from app.transcript_chunk_store import clear_chunks

@pytest.fixture(autouse=True)
def isolated_job_db(tmp_path_factory):
    db_dir = tmp_path_factory.mktemp("job-db")

    configure_db(db_dir / "jobs.db")
    init_db()
    clear_card_relations()
    clear_notes()
    clear_runs()
    clear_cards()
    clear_chunks()
    clear_jobs()

    yield

    clear_card_relations()
    clear_notes()
    clear_runs()
    clear_cards()
    clear_chunks()
    clear_jobs()
