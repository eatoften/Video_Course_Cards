import os
from functools import cache
from pathlib import Path

from pydantic import BaseModel, Field


BACKEND_DIR = Path(__file__).resolve().parent.parent


class LLMSettings(BaseModel):
    provider: str = "ollama"
    base_url: str = "http://localhost:11434/v1"
    model: str = "qwen3:4b"
    api_key: str = "local"
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=8192, ge=1)
    timeout_seconds: float = Field(default=120.0, ge=1.0)


class EmbeddingSettings(BaseModel):
    model: str = "sentence-transformers/all-MiniLM-L6-v2"
    model_path: str | None = None
    batch_size: int = Field(default=32, ge=1)
    local_files_only: bool = True


def _read_env_file() -> dict[str, str]:
    env_path = BACKEND_DIR / ".env"

    if not env_path.is_file():
        return {}

    values: dict[str, str] = {}

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()

        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip("'\"")

    return values


def _env(
    name: str,
    default: str,
    env_file_values: dict[str, str],
) -> str:
    return os.environ.get(name) or env_file_values.get(name, default)


def _env_float(
    name: str,
    default: float,
    env_file_values: dict[str, str],
) -> float:
    raw_value = _env(name, str(default), env_file_values)

    try:
        return float(raw_value)
    except ValueError:
        return default


def _env_int(
    name: str,
    default: int,
    env_file_values: dict[str, str],
) -> int:
    raw_value = _env(name, str(default), env_file_values)

    try:
        return int(raw_value)
    except ValueError:
        return default


def _env_bool(
    name: str,
    default: bool,
    env_file_values: dict[str, str],
) -> bool:
    raw_value = _env(name, str(default), env_file_values).strip().lower()

    if raw_value in {"1", "true", "yes", "on"}:
        return True

    if raw_value in {"0", "false", "no", "off"}:
        return False

    return default


@cache
def get_llm_settings() -> LLMSettings:
    env_file_values = _read_env_file()

    return LLMSettings(
        provider=_env("VCC_LLM_PROVIDER", "ollama", env_file_values),
        base_url=_env(
            "VCC_LLM_BASE_URL",
            "http://localhost:11434/v1",
            env_file_values,
        ),
        model=_env("VCC_LLM_MODEL", "qwen3:4b", env_file_values),
        api_key=_env("VCC_LLM_API_KEY", "local", env_file_values),
        temperature=_env_float(
            "VCC_LLM_TEMPERATURE",
            0.0,
            env_file_values,
        ),
        max_tokens=_env_int(
            "VCC_LLM_MAX_TOKENS",
            8192,
            env_file_values,
        ),
        timeout_seconds=_env_float(
            "VCC_LLM_TIMEOUT_SECONDS",
            120.0,
            env_file_values,
        ),
    )


@cache
def get_embedding_settings() -> EmbeddingSettings:
    env_file_values = _read_env_file()
    model_path = _env("VCC_EMBEDDING_MODEL_PATH", "", env_file_values).strip()

    return EmbeddingSettings(
        model=_env(
            "VCC_EMBEDDING_MODEL",
            "sentence-transformers/all-MiniLM-L6-v2",
            env_file_values,
        ),
        model_path=model_path or None,
        batch_size=_env_int(
            "VCC_EMBEDDING_BATCH_SIZE",
            32,
            env_file_values,
        ),
        local_files_only=_env_bool(
            "VCC_EMBEDDING_LOCAL_FILES_ONLY",
            True,
            env_file_values,
        ),
    )
