import pytest

from app.job import JOB_STORE

@pytest.fixture(autouse=True)
def clear_job_store():
    JOB_STORE.clear()

    yield

    JOB_STORE.clear()