import json
from pathlib import Path

from multimodal_lab.experiment_protocol import (
    ExperimentPhase,
    ExperimentRunSpec,
    ExperimentTask,
)
from multimodal_lab.experiment_tracking import (
    ExperimentRunManifest,
    ExperimentRunRecorder,
    ExperimentRunStatus,
)


def make_spec() -> ExperimentRunSpec:
    return ExperimentRunSpec(
        experiment_id="ctc-plumbing-gate",
        task=ExperimentTask.ctc_overfit,
        phase=ExperimentPhase.diagnostic,
        model_variant="tiny_cnn_ctc_probe",
        seed=17,
        deterministic_algorithms=False,
        dataset_sha256="d" * 64,
        primary_metric="exact_match_rate",
        parameters={"learning_rate": 0.003},
    )


def test_run_recorder_tracks_spec_runtime_metrics_and_artifacts(tmp_path: Path):
    recorder = ExperimentRunRecorder.start(
        make_spec(),
        output_root=tmp_path / "runs",
        code_root=tmp_path,
        device="cpu",
    )
    artifact = recorder.run_dir / "report.json"
    artifact.write_text('{"passed": true}\n', encoding="utf-8")

    manifest = recorder.complete(
        metrics={"exact_match_rate": 1.0},
        artifacts={"report": artifact},
    )

    saved = ExperimentRunManifest.model_validate_json(
        (recorder.run_dir / "run_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest.status is ExperimentRunStatus.completed
    assert saved.spec == make_spec()
    assert saved.metrics == {"exact_match_rate": 1.0}
    assert saved.artifacts["report"].path == "report.json"
    assert saved.artifacts["report"].size_bytes == artifact.stat().st_size
    assert len(saved.artifacts["report"].sha256) == 64
    assert json.loads(
        (recorder.run_dir / "experiment_spec.json").read_text(encoding="utf-8")
    )["experiment_id"] == "ctc-plumbing-gate"


def test_run_recorder_persists_failures(tmp_path: Path):
    recorder = ExperimentRunRecorder.start(
        make_spec(),
        output_root=tmp_path / "runs",
        code_root=tmp_path,
        device="cpu",
    )

    manifest = recorder.fail(RuntimeError("training exploded"))

    assert manifest.status is ExperimentRunStatus.failed
    assert manifest.failure_type == "RuntimeError"
    assert manifest.failure_message == "training exploded"
