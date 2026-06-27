import pytest

from app.db import configure_db, init_db
from app.job_store import clear_jobs
from app.knowledge_card_store import clear_cards

@pytest.fixture(autouse=True)
def isolated_job_db(tmp_path_factory):
    db_dir = tmp_path_factory.mktemp("job-db")

    configure_db(db_dir / "jobs.db")
    init_db()
    clear_cards()
    clear_jobs()

    yield

    clear_cards()
    clear_jobs()
