from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import numpy as np
from numpy.typing import NDArray

from .schemas import (
    SlideTransitionPrediction,
    TransitionBaselineConfig,
    TransitionDetectorVariant,
    TransitionEventType,
)


_FRAME_TIME_PATTERN = re.compile(r"pts_time:([-+]?[0-9]*\.?[0-9]+)")


class TransitionBaselineError(RuntimeError):
    pass


@dataclass(frozen=True)
class SceneScoreSample:
    timestamp_seconds: float
    score: float


@dataclass(frozen=True)
class SampledFrame:
    grayscale: NDArray[np.uint8]
    marker_red_ratio: float


@dataclass(frozen=True)
class _Candidate:
    timestamp_seconds: float
    score: float
    event_type: TransitionEventType


def detect_video_transitions(
    video_path: str | Path,
    *,
    start_seconds: float,
    end_seconds: float,
    config: TransitionBaselineConfig,
    variant: TransitionDetectorVariant = TransitionDetectorVariant.spatial_state,
    ffmpeg_binary: str = "ffmpeg",
    timeout_seconds: float = 600,
) -> list[SlideTransitionPrediction]:
    path = Path(video_path)
    _validate_detection_request(path, start_seconds, end_seconds)
    duration_seconds = end_seconds - start_seconds

    scene_scores = extract_ffmpeg_scene_scores(
        path,
        start_seconds=start_seconds,
        duration_seconds=duration_seconds,
        config=config,
        ffmpeg_binary=ffmpeg_binary,
        timeout_seconds=timeout_seconds,
    )
    sampled_frames = decode_sampled_frames(
        path,
        start_seconds=start_seconds,
        duration_seconds=duration_seconds,
        config=config,
        ffmpeg_binary=ffmpeg_binary,
        timeout_seconds=timeout_seconds,
    )
    return detect_transition_predictions(
        scene_scores,
        sampled_frames,
        config,
        variant=variant,
    )


def extract_ffmpeg_scene_scores(
    video_path: str | Path,
    *,
    start_seconds: float,
    duration_seconds: float,
    config: TransitionBaselineConfig,
    ffmpeg_binary: str = "ffmpeg",
    timeout_seconds: float = 600,
) -> list[SceneScoreSample]:
    crop = config.slide_crop
    video_filter = (
        f"crop={crop.width}:{crop.height}:{crop.x}:{crop.y},"
        f"fps={config.sample_fps:g},"
        f"scale={config.scene_scale_width}:-2,"
        "select='gte(scene,0)',metadata=print:file=-"
    )
    command = [
        ffmpeg_binary,
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-ss",
        f"{start_seconds:.6f}",
        "-t",
        f"{duration_seconds:.6f}",
        "-i",
        str(video_path),
        "-an",
        "-vf",
        video_filter,
        "-f",
        "null",
        "-",
    ]

    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise TransitionBaselineError(
            f"Could not calculate FFmpeg scene scores: {exc}"
        ) from exc

    if result.returncode != 0:
        message = result.stderr.strip() or "FFmpeg scene-score extraction failed."
        raise TransitionBaselineError(message)

    samples = parse_scene_score_metadata(
        result.stdout,
        timestamp_offset_seconds=start_seconds,
    )
    if not samples:
        raise TransitionBaselineError("FFmpeg returned no scene-score samples.")
    return samples


def parse_scene_score_metadata(
    output: str,
    *,
    timestamp_offset_seconds: float = 0,
) -> list[SceneScoreSample]:
    samples: list[SceneScoreSample] = []
    pending_timestamp: float | None = None

    for line in output.splitlines():
        if line.startswith("frame:"):
            match = _FRAME_TIME_PATTERN.search(line)
            if match is None:
                raise TransitionBaselineError(
                    f"FFmpeg frame metadata has no pts_time: {line}"
                )
            pending_timestamp = float(match.group(1)) + timestamp_offset_seconds
        elif line.startswith("lavfi.scene_score="):
            if pending_timestamp is None:
                raise TransitionBaselineError(
                    "FFmpeg emitted a scene score before its frame timestamp."
                )
            samples.append(
                SceneScoreSample(
                    timestamp_seconds=pending_timestamp,
                    score=float(line.split("=", 1)[1]),
                )
            )
            pending_timestamp = None

    return samples


def decode_sampled_frames(
    video_path: str | Path,
    *,
    start_seconds: float,
    duration_seconds: float,
    config: TransitionBaselineConfig,
    ffmpeg_binary: str = "ffmpeg",
    timeout_seconds: float = 600,
) -> list[SampledFrame]:
    crop = config.slide_crop
    width = config.raw_frame_width
    height = config.raw_frame_height
    frame_size = width * height * 3
    video_filter = (
        f"crop={crop.width}:{crop.height}:{crop.x}:{crop.y},"
        f"fps={config.sample_fps:g},scale={width}:{height}"
    )
    command = [
        ffmpeg_binary,
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-ss",
        f"{start_seconds:.6f}",
        "-t",
        f"{duration_seconds:.6f}",
        "-i",
        str(video_path),
        "-an",
        "-vf",
        video_filter,
        "-pix_fmt",
        "rgb24",
        "-f",
        "rawvideo",
        "-",
    ]

    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=frame_size * 2,
        )
    except OSError as exc:
        raise TransitionBaselineError(
            f"Could not start FFmpeg frame sampling: {exc}"
        ) from exc

    if process.stdout is None or process.stderr is None:
        process.kill()
        raise TransitionBaselineError("FFmpeg pipes were not created.")

    frames: list[SampledFrame] = []
    partial_frame = False
    try:
        while True:
            payload = _read_exact(process.stdout, frame_size)
            if not payload:
                break
            if len(payload) != frame_size:
                partial_frame = True
                break
            frames.append(_sampled_frame_from_rgb(payload, width, height, config))

        stderr = process.stderr.read().decode("utf-8", errors="replace")
        return_code = process.wait(timeout=timeout_seconds)
    except (OSError, subprocess.TimeoutExpired) as exc:
        process.kill()
        process.wait()
        raise TransitionBaselineError(
            f"FFmpeg frame sampling did not complete: {exc}"
        ) from exc

    if return_code != 0:
        raise TransitionBaselineError(
            stderr.strip() or "FFmpeg frame sampling failed."
        )
    if partial_frame:
        raise TransitionBaselineError("FFmpeg returned a truncated raw frame.")
    if not frames:
        raise TransitionBaselineError("FFmpeg returned no sampled frames.")
    return frames


def detect_transition_predictions(
    scene_scores: list[SceneScoreSample],
    sampled_frames: list[SampledFrame],
    config: TransitionBaselineConfig,
    *,
    variant: TransitionDetectorVariant = TransitionDetectorVariant.spatial_state,
) -> list[SlideTransitionPrediction]:
    sample_count = min(len(scene_scores), len(sampled_frames))
    if sample_count == 0:
        return []
    if abs(len(scene_scores) - len(sampled_frames)) > 1:
        raise TransitionBaselineError(
            "Scene-score and sampled-frame counts differ by more than one."
        )

    scores = scene_scores[:sample_count]
    frames = sampled_frames[:sample_count]
    if variant is TransitionDetectorVariant.scene_only:
        raw_candidates = _find_scene_only_candidates(scores, config)
    else:
        states = _stabilize_short_state_runs(
            _slide_presence_states(frames, config),
            minimum_run_samples=config.slide_state_min_run_samples,
        )
        if variant is TransitionDetectorVariant.scene_state:
            raw_candidates = _find_scene_state_candidates(
                scores,
                states,
                config,
            )
        else:
            raw_candidates = _find_raw_candidates(
                scores,
                frames,
                states,
                config,
            )
    grouped = _group_candidate_runs(
        raw_candidates,
        maximum_gap_seconds=config.grouping_gap_seconds,
    )
    selected = _temporal_non_maximum_suppression(
        grouped,
        window_seconds=config.nms_window_seconds,
    )
    return [
        SlideTransitionPrediction(
            timestamp_seconds=candidate.timestamp_seconds,
            event_type=candidate.event_type,
            score=max(0.0, min(1.0, candidate.score)),
        )
        for candidate in sorted(selected, key=lambda item: item.timestamp_seconds)
    ]


def _find_scene_only_candidates(
    scores: list[SceneScoreSample],
    config: TransitionBaselineConfig,
) -> list[_Candidate]:
    return [
        _Candidate(
            timestamp_seconds=sample.timestamp_seconds,
            score=sample.score,
            event_type=TransitionEventType.page_change,
        )
        for sample in scores[1:]
        if sample.score >= config.scene_score_threshold
    ]


def _find_scene_state_candidates(
    scores: list[SceneScoreSample],
    states: list[bool],
    config: TransitionBaselineConfig,
) -> list[_Candidate]:
    candidates: list[_Candidate] = []
    for index in range(1, len(scores)):
        sample = scores[index]
        if states[index] != states[index - 1]:
            candidates.append(
                _Candidate(
                    timestamp_seconds=sample.timestamp_seconds,
                    score=sample.score,
                    event_type=(
                        TransitionEventType.enter_slide
                        if states[index]
                        else TransitionEventType.leave_slide
                    ),
                )
            )
        elif states[index] and sample.score >= config.scene_score_threshold:
            candidates.append(
                _Candidate(
                    timestamp_seconds=sample.timestamp_seconds,
                    score=sample.score,
                    event_type=TransitionEventType.page_change,
                )
            )
    return candidates


def _sampled_frame_from_rgb(
    payload: bytes,
    width: int,
    height: int,
    config: TransitionBaselineConfig,
) -> SampledFrame:
    rgb = np.frombuffer(payload, dtype=np.uint8).reshape(height, width, 3)
    rgb16 = rgb.astype(np.uint16)
    grayscale = (
        (77 * rgb16[..., 0] + 150 * rgb16[..., 1] + 29 * rgb16[..., 2])
        >> 8
    ).astype(np.uint8)

    marker_rows = max(1, round(height * config.marker_footer_fraction))
    footer = rgb[-marker_rows:].astype(np.float32)
    red_pixels = (
        (footer[..., 0] >= config.marker_red_minimum)
        & (
            footer[..., 0]
            >= footer[..., 1] * config.marker_red_to_green_ratio
        )
        & (
            footer[..., 0]
            >= footer[..., 2] * config.marker_red_to_blue_ratio
        )
    )
    return SampledFrame(
        grayscale=grayscale,
        marker_red_ratio=float(red_pixels.mean()),
    )


def _slide_presence_states(
    frames: list[SampledFrame],
    config: TransitionBaselineConfig,
) -> list[bool]:
    states: list[bool] = []
    current = False
    for frame in frames:
        if current:
            current = frame.marker_red_ratio >= config.marker_exit_ratio
        else:
            current = frame.marker_red_ratio >= config.marker_enter_ratio
        states.append(current)
    return states


def _stabilize_short_state_runs(
    states: list[bool],
    *,
    minimum_run_samples: int,
) -> list[bool]:
    if minimum_run_samples <= 1 or len(states) < 3:
        return list(states)

    stabilized = list(states)
    start = 0
    while start < len(states):
        end = start + 1
        while end < len(states) and states[end] == states[start]:
            end += 1
        if (
            end - start < minimum_run_samples
            and start > 0
            and end < len(states)
            and stabilized[start - 1] == states[end]
        ):
            stabilized[start:end] = [states[end]] * (end - start)
        start = end
    return stabilized


def _find_raw_candidates(
    scores: list[SceneScoreSample],
    frames: list[SampledFrame],
    states: list[bool],
    config: TransitionBaselineConfig,
) -> list[_Candidate]:
    lookahead = max(
        1,
        round(config.stable_lookahead_seconds * config.sample_fps),
    )
    height = frames[0].grayscale.shape[0]
    header_end = max(1, round(height * config.header_fraction))
    footer_start = min(
        height - 1,
        height - max(1, round(height * config.footer_fraction)),
    )
    candidates: list[_Candidate] = []

    for index in range(1, len(scores)):
        sample = scores[index]
        if states[index] != states[index - 1]:
            candidates.append(
                _Candidate(
                    timestamp_seconds=sample.timestamp_seconds,
                    score=sample.score,
                    event_type=(
                        TransitionEventType.enter_slide
                        if states[index]
                        else TransitionEventType.leave_slide
                    ),
                )
            )
            continue

        if (
            not states[index]
            or not states[index - 1]
            or sample.score < config.scene_score_threshold
        ):
            continue

        post_index = min(len(frames) - 1, index + lookahead)
        before = frames[index - 1].grayscale.astype(np.int16)
        after = frames[post_index].grayscale.astype(np.int16)
        changed = (
            np.abs(after - before) >= config.changed_pixel_luma_threshold
        )
        overall_changed = float(changed.mean())
        header_changed = float(changed[:header_end].mean())
        body_changed = float(changed[header_end:footer_start].mean())
        footer_changed = float(changed[footer_start:].mean())

        if footer_changed >= config.footer_overlay_changed_ratio:
            event_type = TransitionEventType.non_semantic_motion
        elif body_changed < config.minimum_body_changed_ratio:
            if (
                overall_changed < config.minimum_nonsemantic_changed_ratio
                and header_changed < config.page_change_header_ratio
            ):
                continue
            event_type = TransitionEventType.non_semantic_motion
        elif header_changed >= config.page_change_header_ratio:
            event_type = TransitionEventType.page_change
        else:
            event_type = TransitionEventType.content_build

        candidates.append(
            _Candidate(
                timestamp_seconds=sample.timestamp_seconds,
                score=sample.score,
                event_type=event_type,
            )
        )

    return candidates


def _group_candidate_runs(
    candidates: list[_Candidate],
    *,
    maximum_gap_seconds: float,
) -> list[_Candidate]:
    if not candidates:
        return []
    ordered = sorted(candidates, key=lambda item: item.timestamp_seconds)
    groups: list[list[_Candidate]] = [[ordered[0]]]
    for candidate in ordered[1:]:
        if (
            candidate.timestamp_seconds - groups[-1][-1].timestamp_seconds
            <= maximum_gap_seconds
        ):
            groups[-1].append(candidate)
        else:
            groups.append([candidate])
    return [max(group, key=_candidate_rank) for group in groups]


def _temporal_non_maximum_suppression(
    candidates: list[_Candidate],
    *,
    window_seconds: float,
) -> list[_Candidate]:
    selected: list[_Candidate] = []
    for candidate in sorted(candidates, key=_candidate_rank, reverse=True):
        if all(
            abs(candidate.timestamp_seconds - existing.timestamp_seconds)
            > window_seconds
            for existing in selected
        ):
            selected.append(candidate)
    return selected


def _candidate_rank(candidate: _Candidate) -> tuple[float, int]:
    type_priority = {
        TransitionEventType.enter_slide: 4,
        TransitionEventType.leave_slide: 4,
        TransitionEventType.page_change: 3,
        TransitionEventType.content_build: 2,
        TransitionEventType.non_semantic_motion: 1,
    }
    return candidate.score, type_priority[candidate.event_type]


def _read_exact(stream: BinaryIO, size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        chunk = stream.read(size - len(chunks))
        if not chunk:
            break
        chunks.extend(chunk)
    return bytes(chunks)


def _validate_detection_request(
    video_path: Path,
    start_seconds: float,
    end_seconds: float,
) -> None:
    if not video_path.is_file():
        raise TransitionBaselineError(f"Video does not exist: {video_path}")
    if start_seconds < 0:
        raise TransitionBaselineError("start_seconds cannot be negative.")
    if end_seconds <= start_seconds:
        raise TransitionBaselineError(
            "end_seconds must be greater than start_seconds."
        )
