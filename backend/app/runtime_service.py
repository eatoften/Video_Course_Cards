import subprocess

from .embedding import (
    EmbeddingError,
    resolve_sentence_transformer_model_source,
)
from .llm_client import LocalLLMClient
from .runtime_status import RuntimeDependencyStatus, RuntimeStatus
from .settings import get_embedding_settings, get_llm_settings


def get_runtime_status() -> RuntimeStatus:
    dependencies = [
        _check_command(
            "ffmpeg",
            install_hint="Install FFmpeg and make sure ffmpeg is on PATH.",
            required_for=["transcription", "audio extraction"],
        ),
        _check_command(
            "ffprobe",
            install_hint="Install FFmpeg and make sure ffprobe is on PATH.",
            required_for=["video validation", "metadata extraction"],
        ),
        _check_ollama_and_model(),
        _check_embedding_model(),
        _check_faster_whisper(),
    ]

    return RuntimeStatus(
        ready=all(
            dependency.available
            for dependency in dependencies
            if dependency.name in {"ffmpeg", "ffprobe"}
        ),
        dependencies=dependencies,
    )


def _check_command(
    name: str,
    *,
    install_hint: str,
    required_for: list[str],
) -> RuntimeDependencyStatus:
    try:
        result = subprocess.run(
            [name, "-version"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return RuntimeDependencyStatus(
            name=name,
            available=False,
            detail=str(exc),
            install_hint=install_hint,
            required_for=required_for,
        )

    first_line = result.stdout.splitlines()[0] if result.stdout else None

    return RuntimeDependencyStatus(
        name=name,
        available=result.returncode == 0,
        version=first_line,
        detail=None if result.returncode == 0 else result.stderr.strip(),
        install_hint=None if result.returncode == 0 else install_hint,
        required_for=required_for,
    )


def _check_ollama_and_model() -> RuntimeDependencyStatus:
    settings = get_llm_settings()
    model_list = LocalLLMClient(
        settings.model_copy(update={"timeout_seconds": 5.0})
    ).list_models()

    if not model_list.available:
        return RuntimeDependencyStatus(
            name="ollama",
            available=False,
            detail=model_list.error_message,
            install_hint=(
                "Install Ollama, start it, then run "
                f"`ollama pull {settings.model}`."
            ),
            required_for=["card generation", "RAG answers"],
        )

    if settings.model not in model_list.models:
        return RuntimeDependencyStatus(
            name="qwen model",
            available=False,
            detail=f"Loaded models: {', '.join(model_list.models) or 'none'}",
            install_hint=f"ollama pull {settings.model}",
            required_for=["card generation", "RAG answers"],
        )

    return RuntimeDependencyStatus(
        name="qwen model",
        available=True,
        version=settings.model,
        detail=f"Ollama available at {settings.base_url}.",
        required_for=["card generation", "RAG answers"],
    )


def _check_embedding_model() -> RuntimeDependencyStatus:
    settings = get_embedding_settings()

    try:
        model_source = resolve_sentence_transformer_model_source(
            settings.model,
            model_path=settings.model_path,
            local_files_only=settings.local_files_only,
        )
    except EmbeddingError as exc:
        return RuntimeDependencyStatus(
            name="embedding model",
            available=False,
            detail=str(exc),
            install_hint=(
                "Download a complete SentenceTransformer snapshot or set "
                "VCC_EMBEDDING_LOCAL_FILES_ONLY=false for development."
            ),
            required_for=["semantic chunking", "card retrieval", "RAG"],
        )

    return RuntimeDependencyStatus(
        name="embedding model",
        available=True,
        version=settings.model,
        detail=str(model_source),
        required_for=["semantic chunking", "card retrieval", "RAG"],
    )


def _check_faster_whisper() -> RuntimeDependencyStatus:
    try:
        import faster_whisper  # noqa: F401
    except ImportError as exc:
        return RuntimeDependencyStatus(
            name="faster-whisper",
            available=False,
            detail=str(exc),
            install_hint=(
                "Install backend dependencies with `uv sync` in the backend "
                "directory."
            ),
            required_for=["transcription"],
        )

    return RuntimeDependencyStatus(
        name="faster-whisper",
        available=True,
        detail="Python package is importable. Model files may be cached on first transcription.",
        required_for=["transcription"],
    )
