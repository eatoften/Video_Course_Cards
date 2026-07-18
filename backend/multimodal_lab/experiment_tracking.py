from __future__ import annotations

import hashlib
import json
import math
import platform
import subprocess
import sys
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Self

import torch
from pydantic import BaseModel, Field, model_validator

from .experiment_protocol import ExperimentRunSpec


class ExperimentRunStatus(str, Enum):
    running = "running"
    completed = "completed"
    failed = "failed"


class RuntimeFingerprint(BaseModel):
    python_version: str = Field(min_length=1)
    python_executable: str = Field(min_length=1)
    platform: str = Field(min_length=1)
    torch_version: str = Field(min_length=1)
    cuda_available: bool
    cuda_runtime_version: str | None = None
    cudnn_version: int | None = None
    requested_device: str = Field(min_length=1)
    device_name: str | None = None


class ArtifactFingerprint(BaseModel):
    path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(ge=0)


class ExperimentRunManifest(BaseModel):
    schema_version: str = "1.0"
    run_id: str = Field(pattern=r"^[0-9]{8}T[0-9]{12}Z-[0-9a-f]{12}$")
    status: ExperimentRunStatus
    spec_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    spec: ExperimentRunSpec
    started_at: datetime
    ended_at: datetime | None = None
    duration_seconds: float | None = Field(default=None, ge=0)
    git_commit: str | None = Field(default=None, pattern=r"^[0-9a-f]{40}$")
    git_dirty: bool
    runtime: RuntimeFingerprint
    metrics: dict[str, float] = Field(default_factory=dict)
    artifacts: dict[str, ArtifactFingerprint] = Field(default_factory=dict)
    failure_type: str | None = None
    failure_message: str | None = None

    @model_validator(mode="after")
    def validate_lifecycle(self) -> Self:
        if self.status is ExperimentRunStatus.running:
            if self.ended_at is not None or self.duration_seconds is not None:
                raise ValueError("A running experiment cannot have an end time.")
            if self.failure_type is not None or self.failure_message is not None:
                raise ValueError("A running experiment cannot contain a failure.")
        elif self.ended_at is None or self.duration_seconds is None:
            raise ValueError("A finished experiment requires timing metadata.")

        if self.status is ExperimentRunStatus.completed:
            if self.failure_type is not None or self.failure_message is not None:
                raise ValueError("A completed experiment cannot contain a failure.")
        if self.status is ExperimentRunStatus.failed:
            if not self.failure_type or not self.failure_message:
                raise ValueError("A failed experiment requires failure details.")
        return self


class ExperimentRunRecorder:
    """Small local run tracker with immutable per-run artifact directories."""

    def __init__(self, run_dir: Path, manifest: ExperimentRunManifest) -> None:
        self.run_dir = run_dir
        self._manifest = manifest

    @property
    def manifest(self) -> ExperimentRunManifest:
        return self._manifest

    @classmethod
    def start(
        cls,
        spec: ExperimentRunSpec,
        *,
        output_root: str | Path,
        code_root: str | Path,
        device: str,
    ) -> ExperimentRunRecorder:
        started_at = datetime.now(UTC)
        spec_sha256 = canonical_model_sha256(spec)
        run_id = (
            started_at.strftime("%Y%m%dT%H%M%S%fZ")
            + "-"
            + spec_sha256[:12]
        )
        run_dir = Path(output_root).resolve() / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        git_commit, git_dirty = _git_fingerprint(Path(code_root).resolve())
        manifest = ExperimentRunManifest(
            run_id=run_id,
            status=ExperimentRunStatus.running,
            spec_sha256=spec_sha256,
            spec=spec,
            started_at=started_at,
            git_commit=git_commit,
            git_dirty=git_dirty,
            runtime=_runtime_fingerprint(device),
        )
        recorder = cls(run_dir, manifest)
        _write_model(run_dir / "experiment_spec.json", spec)
        recorder._write_manifest()
        return recorder

    def complete(
        self,
        *,
        metrics: dict[str, float],
        artifacts: dict[str, str | Path],
    ) -> ExperimentRunManifest:
        self._require_running()
        normalized_metrics = _validate_metrics(metrics)
        artifact_fingerprints = {
            name: _artifact_fingerprint(path, root=self.run_dir)
            for name, path in artifacts.items()
        }
        ended_at = datetime.now(UTC)
        self._manifest = self._manifest.model_copy(
            update={
                "status": ExperimentRunStatus.completed,
                "ended_at": ended_at,
                "duration_seconds": (
                    ended_at - self._manifest.started_at
                ).total_seconds(),
                "metrics": normalized_metrics,
                "artifacts": artifact_fingerprints,
            }
        )
        self._manifest = ExperimentRunManifest.model_validate(
            self._manifest.model_dump()
        )
        self._write_manifest()
        return self._manifest

    def fail(self, error: BaseException) -> ExperimentRunManifest:
        self._require_running()
        ended_at = datetime.now(UTC)
        self._manifest = self._manifest.model_copy(
            update={
                "status": ExperimentRunStatus.failed,
                "ended_at": ended_at,
                "duration_seconds": (
                    ended_at - self._manifest.started_at
                ).total_seconds(),
                "failure_type": type(error).__name__,
                "failure_message": str(error) or repr(error),
            }
        )
        self._manifest = ExperimentRunManifest.model_validate(
            self._manifest.model_dump()
        )
        self._write_manifest()
        return self._manifest

    def _require_running(self) -> None:
        if self._manifest.status is not ExperimentRunStatus.running:
            raise RuntimeError("An experiment run can only be finalized once.")

    def _write_manifest(self) -> None:
        _write_model(self.run_dir / "run_manifest.json", self._manifest)


def canonical_model_sha256(model: BaseModel) -> str:
    payload = json.dumps(
        model.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _runtime_fingerprint(device: str) -> RuntimeFingerprint:
    device_name: str | None = None
    if device.startswith("cuda") and torch.cuda.is_available():
        configured_device = torch.device(device)
        device_index = (
            configured_device.index
            if configured_device.index is not None
            else torch.cuda.current_device()
        )
        device_name = torch.cuda.get_device_name(device_index)
    elif device == "cpu":
        device_name = platform.processor() or None

    return RuntimeFingerprint(
        python_version=platform.python_version(),
        python_executable=sys.executable,
        platform=platform.platform(),
        torch_version=torch.__version__,
        cuda_available=torch.cuda.is_available(),
        cuda_runtime_version=torch.version.cuda,
        cudnn_version=torch.backends.cudnn.version(),
        requested_device=device,
        device_name=device_name,
    )


def _git_fingerprint(code_root: Path) -> tuple[str | None, bool]:
    commit = _run_git(code_root, "rev-parse", "HEAD")
    status = _run_git(code_root, "status", "--porcelain")
    valid_commit = (
        commit
        if commit is not None
        and len(commit) == 40
        and all(character in "0123456789abcdef" for character in commit)
        else None
    )
    return valid_commit, bool(status)


def _run_git(code_root: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=code_root,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def _validate_metrics(metrics: dict[str, float]) -> dict[str, float]:
    normalized: dict[str, float] = {}
    for name, value in metrics.items():
        clean_name = name.strip()
        numeric_value = float(value)
        if not clean_name:
            raise ValueError("Metric names cannot be blank.")
        if not math.isfinite(numeric_value):
            raise ValueError(f"Metric {clean_name} must be finite.")
        normalized[clean_name] = numeric_value
    return normalized


def _artifact_fingerprint(
    path: str | Path,
    *,
    root: Path,
) -> ArtifactFingerprint:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Experiment artifact does not exist: {resolved}")
    try:
        portable_path = resolved.relative_to(root).as_posix()
    except ValueError:
        portable_path = str(resolved)
    return ArtifactFingerprint(
        path=portable_path,
        sha256=_sha256_file(resolved),
        size_bytes=resolved.stat().st_size,
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_model(path: Path, model: BaseModel) -> None:
    content = model.model_dump_json(indent=2) + "\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)
