import io
import json

import numpy as np
import pytest
from pydantic import ValidationError

from multimodal_lab.schemas import (
    PixelCrop,
    SlideTransitionAnnotation,
    SlideTransitionPrediction,
    TransitionBaselineConfig,
    TransitionDetectorVariant,
    TransitionEventType,
)
from multimodal_lab.annotation_io import write_jsonl
from multimodal_lab.evaluate_transition_baseline import main as evaluate_main
from multimodal_lab.run_transition_baseline import main as run_main
from multimodal_lab.run_transition_comparison import main as comparison_main
from multimodal_lab.transition_baseline import (
    SampledFrame,
    SceneScoreSample,
    TransitionBaselineError,
    decode_sampled_frames,
    detect_transition_predictions,
    parse_scene_score_metadata,
)


def make_config(**updates) -> TransitionBaselineConfig:
    config = TransitionBaselineConfig(
        profile_name="test-profile",
        slide_crop=PixelCrop(x=0, y=0, width=160, height=90),
    )
    return config.model_copy(update=updates)


def sampled_frame(grayscale, marker_red_ratio=0.9) -> SampledFrame:
    return SampledFrame(
        grayscale=np.asarray(grayscale, dtype=np.uint8),
        marker_red_ratio=marker_red_ratio,
    )


def test_config_rejects_invalid_marker_hysteresis():
    with pytest.raises(ValidationError, match="marker_exit_ratio"):
        TransitionBaselineConfig(
            profile_name="bad-profile",
            slide_crop=PixelCrop(x=0, y=0, width=160, height=90),
            marker_enter_ratio=0.4,
            marker_exit_ratio=0.5,
        )


def test_parse_scene_score_metadata_preserves_absolute_time():
    output = "\n".join(
        [
            "frame:0 pts:0 pts_time:0",
            "lavfi.scene_score=0.000000",
            "frame:1 pts:1 pts_time:0.5",
            "lavfi.scene_score=0.125000",
        ]
    )

    samples = parse_scene_score_metadata(
        output,
        timestamp_offset_seconds=120,
    )

    assert samples == [
        SceneScoreSample(timestamp_seconds=120.0, score=0.0),
        SceneScoreSample(timestamp_seconds=120.5, score=0.125),
    ]


def test_detector_types_slide_state_page_and_content_changes():
    config = make_config()
    camera = np.full((90, 160), 70, dtype=np.uint8)
    base = np.full((90, 160), 220, dtype=np.uint8)
    page = base.copy()
    page[:15, 10:120] = 90
    page[25:45, 20:130] = 170
    built = page.copy()
    built[50:65, 50:120] = 80

    frames = []
    scores = []
    for index in range(17):
        timestamp = index * 0.5
        if index == 0 or index >= 14:
            frame = sampled_frame(camera, marker_red_ratio=0.0)
        elif index < 5:
            frame = sampled_frame(base)
        elif index < 9:
            frame = sampled_frame(page)
        else:
            frame = sampled_frame(built)
        frames.append(frame)
        score = {1: 0.8, 5: 0.2, 9: 0.02, 14: 0.7}.get(index, 0.0)
        scores.append(SceneScoreSample(timestamp, score))

    predictions = detect_transition_predictions(scores, frames, config)

    assert [prediction.event_type for prediction in predictions] == [
        TransitionEventType.enter_slide,
        TransitionEventType.page_change,
        TransitionEventType.content_build,
        TransitionEventType.leave_slide,
    ]
    assert [prediction.timestamp_seconds for prediction in predictions] == [
        0.5,
        2.5,
        4.5,
        7.0,
    ]


def test_registered_variants_add_state_and_spatial_event_types():
    config = make_config()
    camera = np.full((90, 160), 70, dtype=np.uint8)
    slide = np.full((90, 160), 220, dtype=np.uint8)
    built = slide.copy()
    built[40:60, 30:130] = 80
    frames = [
        sampled_frame(camera, marker_red_ratio=0.0),
        sampled_frame(slide),
        sampled_frame(slide),
        sampled_frame(slide),
        sampled_frame(built),
        sampled_frame(built),
        sampled_frame(built),
        sampled_frame(camera, marker_red_ratio=0.0),
        sampled_frame(camera, marker_red_ratio=0.0),
        sampled_frame(camera, marker_red_ratio=0.0),
    ]
    scores = [
        SceneScoreSample(index * 2.0, {1: 0.8, 4: 0.2, 7: 0.7}.get(index, 0.0))
        for index in range(len(frames))
    ]

    scene_only = detect_transition_predictions(
        scores,
        frames,
        config,
        variant=TransitionDetectorVariant.scene_only,
    )
    scene_state = detect_transition_predictions(
        scores,
        frames,
        config,
        variant=TransitionDetectorVariant.scene_state,
    )
    spatial_state = detect_transition_predictions(
        scores,
        frames,
        config,
        variant=TransitionDetectorVariant.spatial_state,
    )

    assert [item.event_type for item in scene_only] == [
        TransitionEventType.page_change,
        TransitionEventType.page_change,
        TransitionEventType.page_change,
    ]
    assert [item.event_type for item in scene_state] == [
        TransitionEventType.enter_slide,
        TransitionEventType.page_change,
        TransitionEventType.leave_slide,
    ]
    assert [item.event_type for item in spatial_state] == [
        TransitionEventType.enter_slide,
        TransitionEventType.content_build,
        TransitionEventType.leave_slide,
    ]


def test_detector_marks_top_only_overlay_as_non_semantic():
    config = make_config()
    base = np.full((90, 160), 220, dtype=np.uint8)
    overlay = base.copy()
    overlay[2:9, 120:158] = 20
    frames = [
        sampled_frame(base),
        sampled_frame(overlay),
        sampled_frame(overlay),
        sampled_frame(overlay),
    ]
    scores = [
        SceneScoreSample(0.0, 0.0),
        SceneScoreSample(0.5, 0.02),
        SceneScoreSample(1.0, 0.0),
        SceneScoreSample(1.5, 0.0),
    ]

    predictions = detect_transition_predictions(scores, frames, config)

    assert len(predictions) == 1
    assert predictions[0].event_type is TransitionEventType.non_semantic_motion


def test_detector_rejects_misaligned_feature_streams():
    frame = sampled_frame(np.zeros((90, 160), dtype=np.uint8))
    scores = [SceneScoreSample(index * 0.5, 0.0) for index in range(4)]

    with pytest.raises(TransitionBaselineError, match="differ"):
        detect_transition_predictions(scores, [frame], make_config())


def test_decode_sampled_frames_parses_streamed_rgb(monkeypatch, tmp_path):
    config = make_config(raw_frame_width=2, raw_frame_height=2)
    red_footer = np.array(
        [
            [[200, 200, 200], [200, 200, 200]],
            [[100, 20, 20], [100, 20, 20]],
        ],
        dtype=np.uint8,
    )

    class FakeProcess:
        def __init__(self):
            self.stdout = io.BytesIO(red_footer.tobytes())
            self.stderr = io.BytesIO()

        def wait(self, timeout=None):
            return 0

        def kill(self):
            return None

    monkeypatch.setattr(
        "multimodal_lab.transition_baseline.subprocess.Popen",
        lambda *args, **kwargs: FakeProcess(),
    )
    video = tmp_path / "video.mp4"
    video.write_bytes(b"fake")

    frames = decode_sampled_frames(
        video,
        start_seconds=0,
        duration_seconds=1,
        config=config,
    )

    assert len(frames) == 1
    assert frames[0].grayscale.shape == (2, 2)
    assert frames[0].marker_red_ratio == 1.0


def test_run_cli_writes_predictions(monkeypatch, tmp_path, capsys):
    config_path = tmp_path / "config.json"
    config_path.write_text(make_config().model_dump_json(), encoding="utf-8")
    video_path = tmp_path / "lecture.mp4"
    video_path.write_bytes(b"fake")
    output_path = tmp_path / "predictions.jsonl"
    prediction = SlideTransitionPrediction(
        timestamp_seconds=12.5,
        event_type=TransitionEventType.enter_slide,
        score=0.9,
    )
    monkeypatch.setattr(
        "multimodal_lab.run_transition_baseline.detect_video_transitions",
        lambda *args, **kwargs: [prediction],
    )

    exit_code = run_main(
        [
            "--video",
            str(video_path),
            "--output",
            str(output_path),
            "--start",
            "0",
            "--end",
            "30",
            "--config",
            str(config_path),
        ]
    )

    assert exit_code == 0
    assert json.loads(output_path.read_text(encoding="utf-8"))["score"] == 0.9
    assert json.loads(capsys.readouterr().out)["prediction_count"] == 1


def test_evaluation_cli_marks_calibration_report(tmp_path, capsys):
    annotation_path = tmp_path / "annotations.jsonl"
    prediction_path = tmp_path / "predictions.jsonl"
    output_path = tmp_path / "report.json"
    annotation = SlideTransitionAnnotation(
        event_id="event-1",
        lecture_id="lecture-1",
        change_start_seconds=12.0,
        stable_at_seconds=12.5,
        from_page=None,
        to_page=1,
        event_type=TransitionEventType.enter_slide,
    )
    prediction = SlideTransitionPrediction(
        timestamp_seconds=12.5,
        event_type=TransitionEventType.enter_slide,
        score=0.9,
    )
    write_jsonl(annotation_path, [annotation])
    write_jsonl(prediction_path, [prediction])

    exit_code = evaluate_main(
        [
            "--annotations",
            str(annotation_path),
            "--predictions",
            str(prediction_path),
            "--duration",
            "30",
            "--output",
            str(output_path),
        ]
    )

    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert report["calibration_only"] is True
    assert report["typed"]["f1"] == 1.0
    assert json.loads(capsys.readouterr().out) == report


def test_comparison_cli_reuses_features_and_reports_ablation(monkeypatch, tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(make_config().model_dump_json(), encoding="utf-8")
    video_path = tmp_path / "lecture.mp4"
    video_path.write_bytes(b"fake")
    annotation_path = tmp_path / "annotations.jsonl"
    write_jsonl(
        annotation_path,
        [
            SlideTransitionAnnotation(
                event_id="event-1",
                lecture_id="lecture-1",
                change_start_seconds=0.5,
                stable_at_seconds=0.5,
                from_page=None,
                to_page=1,
                event_type=TransitionEventType.enter_slide,
            )
        ],
    )
    camera = sampled_frame(np.zeros((90, 160)), marker_red_ratio=0.0)
    slide = sampled_frame(np.full((90, 160), 220))
    scene_scores = [
        SceneScoreSample(0.0, 0.0),
        SceneScoreSample(0.5, 0.8),
        SceneScoreSample(1.0, 0.0),
        SceneScoreSample(1.5, 0.0),
    ]
    frames = [camera, slide, slide, slide]
    calls = {"scene_scores": 0, "sampled_frames": 0}

    def fake_scene_scores(*args, **kwargs):
        calls["scene_scores"] += 1
        return scene_scores

    def fake_sampled_frames(*args, **kwargs):
        calls["sampled_frames"] += 1
        return frames

    monkeypatch.setattr(
        "multimodal_lab.run_transition_comparison.extract_ffmpeg_scene_scores",
        fake_scene_scores,
    )
    monkeypatch.setattr(
        "multimodal_lab.run_transition_comparison.decode_sampled_frames",
        fake_sampled_frames,
    )
    output_dir = tmp_path / "comparison"

    exit_code = comparison_main(
        [
            "--video",
            str(video_path),
            "--annotations",
            str(annotation_path),
            "--output-dir",
            str(output_dir),
            "--start",
            "0",
            "--end",
            "2",
            "--config",
            str(config_path),
            "--held-out",
        ]
    )

    report = json.loads((output_dir / "comparison_report.json").read_text())
    assert exit_code == 0
    assert report["calibration_only"] is False
    assert report["variants"]["scene_only"]["evaluation"]["typed"]["f1"] == 0
    assert report["variants"]["scene_state"]["evaluation"]["typed"]["f1"] == 1
    assert report["variants"]["spatial_state"]["evaluation"]["typed"]["f1"] == 1
    assert calls == {"scene_scores": 1, "sampled_frames": 1}


def test_comparison_cli_checks_frozen_config_before_inference(
    monkeypatch,
    tmp_path,
):
    config_path = tmp_path / "config.json"
    config_path.write_text(make_config().model_dump_json(), encoding="utf-8")
    inference_started = False

    def fail_if_inference_starts(*args, **kwargs):
        nonlocal inference_started
        inference_started = True
        raise AssertionError("Feature extraction must not start.")

    monkeypatch.setattr(
        "multimodal_lab.run_transition_comparison.extract_ffmpeg_scene_scores",
        fail_if_inference_starts,
    )

    with pytest.raises(ValueError, match="preregistered freeze"):
        comparison_main(
            [
                "--video",
                str(tmp_path / "lecture.mp4"),
                "--annotations",
                str(tmp_path / "annotations.jsonl"),
                "--output-dir",
                str(tmp_path / "comparison"),
                "--start",
                "0",
                "--end",
                "2",
                "--config",
                str(config_path),
                "--expected-config-sha256",
                "0" * 64,
            ]
        )

    assert inference_started is False
