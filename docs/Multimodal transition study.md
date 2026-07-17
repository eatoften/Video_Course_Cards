# Training-Free Lecture Slide Transition Detection

## Study status

- Protocol registered: 2026-07-17
- Current stage: Assignment 1.1 complete
- Calibration lecture: CS231n Lecture 2, `0-900 s` (legacy 2025 label)
- Held-out lecture: CS231n 2026 Lecture 3, `0-900 s`
- The held-out embargo was lifted only after Lecture 3 gold annotations passed
  validation; results below are from the first frozen-config run.

This document is both the experiment protocol and the append-only result log for
the first multimodal assignment. It deliberately separates decisions made
before held-out evaluation from observations made afterward.

## 1. Research problem

Video Course Cards should read a stable slide only after the displayed visual
state has meaningfully changed. Sampling every video frame wastes computation
and produces duplicate evidence. A useful transition detector must distinguish:

- a new deck page;
- an incremental content build on the same page;
- entering or leaving the slide feed;
- transient player chrome, overlays, fades, and other non-semantic motion.

The immediate research problem is:

> Can a training-free, event-driven detector recover semantically meaningful
> slide transitions from lecture recordings accurately enough to select stable
> frames for downstream page reading?

## 2. Research questions and hypotheses

### RQ1: Event detection

Can the detector recover meaningful visual events within a one-second temporal
tolerance on a lecture that was not used for threshold selection?

**H1:** The complete spatial-state method will achieve higher held-out relaxed
F1 and fewer false positives per hour than a scene-score-only baseline.

### RQ2: Event typing

Can the detector distinguish page changes, content builds, and slide-feed
entry/exit events?

**H2:** Explicit slide-presence state will improve typed F1 over scene score
alone, while persistent spatial-change rules will further improve typed F1 by
separating page changes from content builds and non-semantic overlays.

### RQ3: Timing

Does stability-aware detection locate events soon enough for downstream frame
selection?

**H3:** The complete method will keep mean absolute timing error below `1.0 s`
without increasing duplicate detections.

These hypotheses were written before running any Lecture 3 prediction.

## 3. Data and split policy

| Split | Lecture | Interval | Purpose |
| --- | --- | ---: | --- |
| Calibration | CS231n Lecture 2 (legacy 2025 label) | 0-900 s | Design and threshold selection |
| Held-out test | CS231n 2026 Lecture 3 | 0-900 s | One-shot generalization check |

The official course schedule identifies Lecture 3 as *Regularization and
Optimization*. Its exact 2026 deck contains 121 PDF pages:

- Schedule: <https://cs231n.stanford.edu/schedule>
- Deck: <https://cs231n.stanford.edu/slides/2026/lecture_3.pdf>

### Provenance amendment before held-out inference

The local recordings were initially labeled as 2025 because the first pilot
used the 2025 public Lecture 2 deck. During independent Lecture 3 annotation,
the video showed a `Recall from last time: Distance Metric` page absent from the
2025 Lecture 3 PDF. Searching Stanford's official slide archive identified the
exact page in the 2026 deck. The 2026 title, recap order, and subsequent pages
match the recording, so the held-out manifest was corrected to 2026.

This correction occurred before gold annotation was finalized and before any
Lecture 3 detector variant was run. No numeric detector setting changed. The
legacy configuration filename and SHA-256 are retained to make that chronology
auditable rather than rewriting history.

The split unit is a lecture, not a frame. Frames from the held-out lecture may
not be used to change thresholds before the primary report is generated. Any
later Lecture 3-guided change creates a new configuration version and requires
a different lecture for evaluation.

## 4. Gold annotation protocol

Gold events are produced before and independently from model predictions:

1. Inspect `5 s` overview frames, `1 fps` contact sheets, and the source deck.
2. Use raw `4 fps` FFmpeg scene-score peaks and footer-state runs only as
   navigation aids, then visually scrub every candidate at `0.25 s`
   resolution. These aids expose no detector prediction or event label.
3. Record `change_start_seconds` and the first readable `stable_at_seconds`.
4. Assign one event type and before/after deck page identity.
5. Record non-semantic motion explicitly, even though it is not a positive
   target, so abstentions and false positives can be audited.
6. Run `multimodal_lab.validate_annotations` before exposing model output.

Positive target types are `page_change`, `content_build`, `enter_slide`, and
`leave_slide`. `non_semantic_motion` is excluded from positive-event recall.

The frozen Lecture 3 annotation contains `58` visual events: `54` positive
targets and `4` explicitly labeled non-semantic events. It also contains `14`
stable-page references covering every distinct deck page seen in the interval.
Each reference binds a visually checked video frame to a deck page, manually
cleaned text, technical terms, and concept labels. No registered detector
variant had been run on Lecture 3 when these files were written.

## 5. Methods and ablations

All methods use the same crop, `2 fps` sampling, FFmpeg scene scores, threshold,
grouping gap, and NMS window. This controls the input and temporal postprocessing
so the comparison isolates the added state and spatial reasoning.

### B0: Scene score only

Threshold FFmpeg adjacent-frame scene scores, group nearby threshold crossings,
apply temporal NMS, and label every retained event `page_change`. This baseline
has no knowledge of whether slides are visible.

### B1: Scene score plus slide state

Add the red CS231n footer marker with hysteresis. Marker transitions become
`enter_slide` or `leave_slide`; threshold crossings while a slide is visible
remain `page_change`. This ablation has no persistent spatial mask and cannot
identify content builds or overlays.

### M2: Spatial-state detector

Use slide state plus a one-second lookahead to compare persistent header, body,
and footer changes. Header changes indicate a new page, body-only changes a
content build, and transient top/footer changes a non-semantic event.

This is a course-profile baseline, not a claim of template-independent slide
detection. The red-footer marker is deliberately CS231n-specific.

## 6. Frozen configuration and environment

Primary configuration:
`backend/multimodal_lab/configs/cs231n_2025_web.json` (legacy profile name)

- Configuration SHA-256:
  `3BA8E137EE214B8AB7802F89D522D59A9D983DA76D925DE08C85F90FD8E1ED03`
- Git base before held-out work:
  `c117f8e212dfa21ec14e70bf9814e0adaa857977`
- FFmpeg: `8.1.1-full_build-www.gyan.dev`
- Scene threshold: `0.002`
- Sample rate: `2 fps`
- Stable lookahead: `1.0 s`
- Grouping gap: `0.75 s`
- NMS window: `1.5 s`
- Evaluation tolerance: `1.0 s`

The file hash, not only the human-readable values, is the freeze boundary.

Input video SHA-256 fingerprints:

- Lecture 2 calibration:
  `E6AE3EAD7CF838116A452B4525B585F6A49C96ABE55545B3B48550262655E50A`
- Lecture 3 held-out:
  `BF1EE390D9961BF6B450D79741FDD24182351615F03C39337BE467C4E3226945`

## 7. Evaluation

The evaluator performs an order-preserving one-to-one alignment between gold
and predicted events. It reports:

- relaxed precision, recall, and F1, ignoring event type;
- typed precision, recall, and F1, requiring the same event type;
- mean absolute timing error and signed detection delay;
- duplicate predictions;
- false positives per hour;
- runtime and prediction counts by event type.

The primary endpoint is held-out relaxed F1. Typed F1 is the main diagnostic
endpoint because it measures whether the system selected the right kind of
stable state rather than merely noticing motion.

## 8. Results

### 8.1 Calibration: Lecture 2

| Method | Relaxed P/R/F1 | Typed P/R/F1 | MAE (s) | FP/hour | Duplicates |
| --- | --- | --- | ---: | ---: | ---: |
| B0 scene score | 0.407 / 0.974 / 0.574 | 0.066 / 0.158 / 0.093 | 0.297 | 216.0 | 0 |
| B1 scene + state | 0.507 / 1.000 / 0.673 | 0.413 / 0.816 / 0.549 | 0.303 | 148.0 | 0 |
| M2 spatial-state | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | 0.303 | 0.0 | 0 |

The M2 row is a calibration result and cannot support a generalization claim.
Feature extraction took `40.641 s`; after those shared features were available,
B0, B1, and M2 took `0.005 s`, `0.003 s`, and `0.015 s`, respectively.

### 8.2 Held-out: Lecture 3

| Method | Relaxed P/R/F1 | Typed P/R/F1 | MAE (s) | FP/hour | Duplicates |
| --- | --- | --- | ---: | ---: | ---: |
| B0 scene score | 0.511 / 0.852 / 0.639 | 0.089 / 0.148 / 0.111 | 0.147 | 176.0 | 0 |
| B1 scene + state | 0.628 / 1.000 / 0.771 | 0.616 / 0.981 / 0.757 | 0.148 | 128.0 | 0 |
| M2 spatial-state | 1.000 / 1.000 / 1.000 | 0.963 / 0.963 / 0.963 | 0.148 | 0.0 | 0 |

M2 recovered all `54` target events without a target false positive or
duplicate. Its `56` total outputs comprise those targets plus two explicit
`non_semantic_motion` abstentions. Feature extraction took `38.476 s`; B0, B1,
and M2 then took `0.005 s`, `0.003 s`, and `0.014 s`. The FFmpeg decode and
feature pass therefore dominates runtime, not the decision rules.

### 8.3 Hypothesis outcomes

- **H1 supported within this case study.** M2 raised held-out relaxed F1 from
  B0's `0.639` to `1.000` and reduced target false positives/hour from `176`
  to `0`.
- **H2 supported.** Adding slide state raised typed F1 from `0.111` to `0.757`;
  spatial persistence and region rules raised it further to `0.963`.
- **H3 supported.** M2's held-out timing MAE was `0.148 s`, below `1.0 s`, with
  zero duplicate target detections.

These outcomes apply to a second lecture from the same course and visual
template. They do not establish cross-course or cross-template generalization.

### 8.4 Error analysis

M2 had no relaxed false positives or false negatives. Its two typed errors are
paired misclassifications rather than missed state changes:

| Time | Gold -> prediction | Region evidence | Primary label | Interpretation |
| ---: | --- | --- | --- | --- |
| 292.75 s | `content_build` -> `page_change` | header 0.1188, body 0.2574 | `animation` | PDF page 10 progressively adds the worked example and changes the title to include "Example". The header change exceeds the frozen 0.025 page threshold even though deck identity is unchanged. |
| 711.25 s | `page_change` -> `content_build` | header 0.0021, body 0.0490 | `annotation_ambiguity` | The deck advances from page 13 to 14, but the title is unchanged and only the regularization term appears in the body. Pixels alone make this look like a content build. |

Both outputs still select the correct meaningful state for downstream page
reading. The errors expose a limit of the event taxonomy: PDF page identity and
perceived visual construction are not always identifiable from video pixels
alone. Future alignment may use deck-page similarity to resolve the type, but
the transition trigger itself does not require a learned classifier for these
two cases.

The detector emitted non-semantic abstentions at `26.0 s` (player controls) and
`29.0 s` (course UI pill). It did not emit the opening fade or the control
disappearance as non-semantic labels. Since non-semantic motion is not a target,
this does not affect the primary metrics; it does mean the current study should
not be cited as evaluating exhaustive overlay detection.

## 9. Threats to validity

- Calibration and test use the same course, recording platform, and slide
  template, so this is within-course generalization only.
- One author performs annotation; inter-annotator agreement is not measured.
- Scene-score peaks and footer-state runs assisted candidate navigation. Even
  without model predictions, this can bias annotation toward visually abrupt
  events; uniform frame inspection reduces but does not eliminate that risk.
- The test interval contains correlated sequential events and is too small for
  broad statistical claims.
- The lecture video and stable-frame images remain local, ignored data. Their
  hashes identify the exact inputs, but independent reproduction requires
  lawful access to the same recordings.
- Deck pages can contain animations that are not represented as separate PDF
  pages, making `content_build` boundaries partly judgment-based.
- The footer marker encodes template knowledge and may fail on demos, camera
  views, or another course.
- A single tolerance threshold can hide whether an event is detected before a
  slide is actually readable; timing and stable-frame quality must therefore be
  inspected separately.

The result should be described as a reproducible case study and engineering
baseline, not as state-of-the-art transition detection.

## 10. Reproduction

From `backend/`:

```powershell
uv run python -m multimodal_lab.validate_annotations `
  --manifests data\multimodal_lab\cs231n_2026_lecture_03\dataset_manifest.jsonl `
  --transitions data\multimodal_lab\cs231n_2026_lecture_03\transition_events.jsonl `
  --references data\multimodal_lab\cs231n_2026_lecture_03\stable_page_references.jsonl

uv run python -m multimodal_lab.run_transition_comparison `
  --video data\uploads\524947e169d8423088a865e778c6e1ac.mp4 `
  --annotations data\multimodal_lab\cs231n_2025_lecture_02\transition_events.jsonl `
  --start 0 --end 900 `
  --expected-config-sha256 3ba8e137ee214b8ab7802f89d522d59a9d983da76d925de08c85f90fd8e1ed03 `
  --output-dir data\multimodal_lab\cs231n_2025_lecture_02\comparison

uv run python -m multimodal_lab.run_transition_comparison `
  --video data\uploads\68d4bfa3412f40ce84c7676c9e92df4f.mp4 `
  --annotations data\multimodal_lab\cs231n_2026_lecture_03\transition_events.jsonl `
  --start 0 --end 900 --held-out `
  --expected-config-sha256 3ba8e137ee214b8ab7802f89d522d59a9d983da76d925de08c85f90fd8e1ed03 `
  --output-dir data\multimodal_lab\cs231n_2026_lecture_03\comparison
```

## 11. Decision rule and next assignment

The decision criterion is met: M2 selects all meaningful held-out states, and
the only residual errors concern page-change versus content-build typing. The
next assignment is therefore the page-reading baseline on the `14` frozen
stable references, comparing native PDF text, a training-free OCR baseline,
and later CNN/ViT readers under the same text and concept metrics.

Do not tune M2 on Lecture 3. Any transition-method revision must receive a new
configuration hash and be evaluated on another held-out lecture.
