# Multimodal Upgrade Plan

Last updated: 2026-07-17

Status: planned, not yet implemented

This document is both an engineering roadmap and an assignment sequence. It is
designed to turn the current audio-first VideoCourseCards pipeline into a
traceable multimodal learning system, while using the upgrade to learn computer
vision, representation learning, temporal alignment, evaluation, and production
integration.

The goal is not simply to add screenshots to cards. The goal is to answer a
harder and more useful question:

> At a given moment in a lecture, what is the instructor saying, what source is
> visible, which slide or document unit does it correspond to, and which claims
> can a knowledge card safely derive from those aligned sources?

---

## 1. North Star

The target pipeline is:

```text
video                                      PPTX/PDF/document
  |                                                |
  v                                                v
low-cost transition detector               native source units
  |                                                |
  v                                                |
stable page keyframes                              |
  |                                                |
  +------> CNN reader / ViT reader / OCR ----------+
                         |
                         v
                structured PageContent
                         |
timestamped transcript --+
           |
           v
    multimodal chunks
           |
           v
claim-level grounded knowledge cards
```

This is an event-driven pipeline. FFmpeg/scene detection observes the video at a
low cost, but expensive page reading runs once per stable page event rather than
continuously on every frame.

At the product level, a user should eventually be able to open a card and see:

- the transcript quote and exact video timestamp;
- the matched slide/page and its source file;
- the keyframe visible at that timestamp;
- whether a claim is supported by speech, slide text, or both;
- an explicit low-confidence or unknown state when alignment is uncertain.

This makes audio and visual evidence anchors for later study documents,
knowledge graphs, retrieval, and RAG. It does not replace the existing card
system.

---

## 2. Scope Decisions

### 2.1 What this upgrade will do first

1. Support lectures for which the exact PPTX or PDF deck is available.
2. Preserve every timestamp from the current Faster Whisper result.
3. Detect page/content transitions with a low-cost temporal module.
4. Select one readable stable keyframe after each accepted transition.
5. Establish native-text and mature-OCR baselines before custom models.
6. Implement a CNN reader and a ViT reader with a controlled shared decoder.
7. Compare the readers at text, concept, claim, card, and systems levels.
8. Align page intervals with existing transcript chunks.
9. Generate cards whose claims cite multimodal evidence.

### 2.2 What this upgrade will not do initially

- It will not use CNN/ViT as the primary transition detector. Transition
  detection is a separate temporal stage with deterministic baselines.
- It will not pretend that a CNN or ViT encoder alone can output text. Both
  readers require a shared detector, tokenizer, and sequence decoder.
- It will not assume every frame is a slide. Speaker view, code, whiteboard,
  demos, title screens, breaks, and edited material need an `unknown` state.
- It will not decode every video frame into SQLite. Only sampled frames,
  transition candidates, metadata, and paths belong in persistent storage.
- It will not make a large vision-language model a hard dependency of the
  desktop app.
- It will not claim that a custom model trained on five lectures generalizes to
  arbitrary courses.
- It will not change the knowledge graph until multimodal evidence is reliable.

### 2.3 The most important modeling distinction

There are three different research questions:

**Transition question:** which low-cost method most reliably finds a real page
or content transition and then selects a stable readable frame?

**Reader question:** when the text detector and decoder are held constant, how
do a CNN encoder and ViT encoder differ in recovering text and technical terms
from compressed lecture-video frames?

**Product question:** does better page reading produce more supported, useful
knowledge cards, and is that improvement worth the local compute cost?

The transition detector, page reader, and card generator are evaluated
separately and end to end. Otherwise a good card score could hide weak OCR, or a
low OCR score could be blamed on an unrelated card-generation failure.

---

## 3. Why This Direction Is Technically Meaningful

Educational video is a long-sequence multimodal problem, not just an OCR
problem. Existing work supports several design choices in this roadmap:

- The Lecture Presentations Multimodal Dataset contains more than 180 hours of
  lecture video and over 9,000 aligned slides, with slide transitions, OCR, ASR,
  and figure annotations. It highlights technical vocabulary, cross-modal
  alignment, and long sequences as core challenges.
- SlideSpeech shows that synchronized slide text can improve recognition of
  named entities and rare technical words in lecture speech.
- MaViLS combines OCR, transcript, and image similarities and then uses dynamic
  programming to align video frames to a slide sequence. Its ablations show why
  text, audio, visual, and temporal continuity should be measured separately.
- SliTraNet exposes animations, inserted videos, and fast changes as important
  transition-detection failure cases. This plan uses it as a reference but starts
  with FFmpeg/PySceneDetect and anchor-frame baselines.
- CRNN demonstrates a CNN-based sequence-recognition path; ViTSTR and TrOCR
  demonstrate transformer-based text readers. A controlled experiment must use
  the same text regions, vocabulary, and decoder when isolating encoder effects.
- FActScore motivates atomic-claim support, ALCE separates citation correctness
  and completeness, and QGEval identifies answerability and answer consistency
  as distinct question-quality dimensions.

This project has one especially useful advantage over generic video
understanding: when the source deck is available, slide order and slide content
are known. That turns open-ended visual understanding into constrained retrieval
plus sequence alignment.

---

## 4. Where This Fits the Current Codebase

The current repository already contains most of the non-visual foundation. The
upgrade should reuse it instead of creating a parallel pipeline.

| Existing component | What is already reusable | Multimodal gap |
| --- | --- | --- |
| `backend/app/transcription.py` | `TranscriptSegment` preserves start time, end time, and text | No visual evidence |
| `backend/app/transcript_chunk.py` | Semantic chunks preserve `segment_ids` and time boundaries | Chunks do not reference slides or frames |
| `backend/app/embedding.py` | Local `SentenceTransformerEmbedder`, cosine similarity/distance, model-path validation | Can compare extracted text but does not read page images |
| `backend/app/source_asset.py` | `SourceAsset` supports PPTX/PDF/DOCX and `SourceUnitType` already includes `slide`, `page`, and `video_frame` | No rendered slide or frame artifact model |
| `backend/app/source_asset_parser.py` | PPTX text is extracted slide by slide with `slide_number` locators | Does not render slides, extract speaker notes, or preserve visual layout |
| `backend/app/source_asset_service.py` | Local file import, hashing, ownership, and SQLite orchestration | No visual preprocessing lifecycle |
| `backend/app/card_embedding.py` | Versioned vector metadata and float-vector serialization pattern | Visual embeddings need their own cache/owner contract |
| `backend/app/knowledge_card.py` | Claims and transcript evidence are structured | Evidence is transcript-shaped rather than generic source-shaped |
| `backend/app/card_service.py` | Grounded generation and validation already share one service path | Prompt/context only understand transcript evidence |
| `backend/app/db.py` | SQLite is the primary local database | No frames, alignment observations, or slide intervals |
| Tauri desktop work | Local app data, backend sidecar, and runtime checks exist | Heavy vision dependencies/checkpoints must remain optional and diagnosable |

### 4.1 Existing interfaces that must remain authoritative

- Timestamp authority: `TranscriptSegment.start_seconds/end_seconds`.
- Semantic text grouping authority: `TranscriptChunk.segment_ids` and chunk time
  range.
- Source identity authority: `SourceAsset` and `SourceUnit`.
- Text embedding authority: `SentenceTransformerEmbedder`; do not add another
  Sentence Transformer loader.
- Database authority: SQLite; generated images and model checkpoints are files
  referenced by SQLite, not BLOB-heavy replacements for the filesystem.
- Card generation authority: `card_service.py`; auto and manual generation must
  still converge on the same grounded generation path.

### 4.2 Proposed code boundaries

Production inference and application code should follow the existing flat
module style:

```text
backend/app/
  slide_renderer.py
  slide_transition.py
  slide_transition_detector.py
  stable_page_frame.py
  page_reader.py
  slide_alignment.py
  slide_alignment_store.py
  slide_alignment_service.py
  multimodal_chunk.py
  multimodal_chunk_service.py
```

Training and experiments should not be mixed into FastAPI request handlers:

```text
backend/multimodal_lab/
  schemas.py
  dataset.py
  sampling.py
  metrics.py
  train_cnn.py
  train_vit.py
  evaluate.py
  models/
    cnn.py
    vit.py
```

Unit and integration tests stay with the repository's current test suite:

```text
backend/tests/
  test_multimodal_lab_metrics.py
  test_slide_renderer.py
  test_slide_transition_detector.py
  test_page_reader.py
  test_slide_alignment.py
  test_multimodal_chunk_service.py
```

The Tauri installer should contain inference code and selected checkpoints, not
raw training data, optimizer states, or the whole experiment environment.

---

## 5. Target Data Contracts

These are proposed contracts. Assignment 0 freezes the exact fields before a
database migration is written.

### 5.1 Rendered slide

```python
class RenderedSlide(BaseModel):
    source_unit_id: str
    asset_id: str
    slide_number: int
    image_path: str
    image_sha256: str
    width: int
    height: int
    native_text: str
    speaker_notes: str | None = None
```

The existing `SourceUnit(unit_type="slide")` remains the semantic source unit.
`RenderedSlide` adds a reproducible visual artifact; it does not duplicate slide
ownership.

### 5.2 Transition event

```python
class SlideTransitionEvent(BaseModel):
    id: str
    job_id: str
    change_start_seconds: float
    stable_at_seconds: float
    from_page: int | None
    to_page: int | None
    event_type: Literal[
        "page_change",
        "content_build",
        "enter_slide",
        "leave_slide",
        "non_semantic_motion",
    ]
```

`change_start_seconds` and `stable_at_seconds` are deliberately separate. The
reader must not consume a frame from the unstable transition interval.

### 5.3 Stable page frame

```python
class StablePageFrame(BaseModel):
    id: str
    event_id: str
    job_id: str
    timestamp_seconds: float
    image_path: str
    image_sha256: str
    width: int
    height: int
    sharpness_score: float | None
```

Frames preserve time directly. There is no later attempt to reconstruct time
from image order.

### 5.4 Page content

```python
class PageContent(BaseModel):
    stable_frame_id: str
    reader: str
    reader_version: str
    raw_text: str
    normalized_text: str
    ordered_blocks: list[PageContentBlock]
    technical_terms: list[str]
    source_unit_id: str | None
    confidence: float | None
```

CNN, ViT, mature OCR, and native-text oracle paths all serialize to this same
contract. Downstream card generation must not receive reader-specific objects.

### 5.5 Alignment observation

```python
class SlideAlignmentObservation(BaseModel):
    frame_id: str
    slide_source_unit_id: str | None
    timestamp_seconds: float
    visual_score: float | None
    ocr_score: float | None
    transcript_score: float | None
    fused_score: float
    confidence: float
    method_version: str
```

A null `slide_source_unit_id` represents `unknown/no matching slide`. Raw score
components are retained so failures can be diagnosed rather than hidden behind
one number.

### 5.6 Slide interval

```python
class SlideInterval(BaseModel):
    id: str
    job_id: str
    slide_source_unit_id: str | None
    start_seconds: float
    end_seconds: float
    confidence: float
    supporting_frame_ids: list[str]
    alignment_version: str
```

Consecutive frame predictions are converted to intervals only after temporal
alignment. Intervals make joining with transcript chunks straightforward.

### 5.7 Multimodal chunk

```python
class MultimodalChunk(BaseModel):
    id: str
    transcript_chunk_id: str
    job_id: str
    start_seconds: float
    end_seconds: float
    slide_interval_ids: list[str]
    slide_source_unit_ids: list[str]
    keyframe_ids: list[str]
    alignment_confidence: float
    version: str
```

The transcript chunk remains intact. Multimodal enrichment is a join layer, not
a destructive rewrite of the transcript.

### 5.8 Generic card evidence

The eventual card evidence contract should support:

```python
class KnowledgeCardEvidenceV3(BaseModel):
    source_type: Literal["transcript", "slide", "video_frame", "document"]
    source_id: str
    quote: str | None
    locator: dict[str, object]
    start_seconds: float | None = None
    end_seconds: float | None = None
    confidence: float | None = None
```

Examples of locators are `{"slide_number": 17}`, `{"page_number": 6}`, or a
video timestamp. This migration should happen only after the alignment output is
stable enough to consume.

---

## 6. Database Direction

SQLite remains the primary database. Images and checkpoints remain files under
the configured app data directory.

Proposed tables:

```text
rendered_slides
  source_unit_id PK/FK -> source_units.id
  image_path
  image_sha256
  width
  height
  renderer
  renderer_version
  created_at

slide_transition_runs
  id PK
  job_id FK -> jobs.id
  detector
  detector_version
  config_json
  status
  started_at
  completed_at
  error

slide_transition_events
  id PK
  run_id FK -> slide_transition_runs.id
  job_id FK -> jobs.id
  change_start_seconds
  stable_at_seconds
  from_page nullable
  to_page nullable
  event_type

stable_page_frames
  id PK
  event_id FK -> slide_transition_events.id
  job_id FK -> jobs.id
  timestamp_seconds
  image_path
  image_sha256
  sharpness_score nullable
  created_at

page_contents
  id PK
  stable_frame_id FK -> stable_page_frames.id
  reader
  reader_version
  input_hash
  raw_text
  normalized_text
  ordered_blocks_json
  technical_terms_json
  source_unit_id FK nullable
  confidence nullable
  created_at
  updated_at
  UNIQUE(stable_frame_id, reader, reader_version)

slide_alignment_runs
  id PK
  job_id FK -> jobs.id
  source_asset_id FK -> source_assets.id
  method_version
  config_json
  status
  started_at
  completed_at
  error

slide_alignment_observations
  run_id FK
  frame_id FK
  slide_source_unit_id FK nullable
  visual_score nullable
  ocr_score nullable
  transcript_score nullable
  fused_score
  confidence

slide_intervals
  id PK
  run_id FK
  slide_source_unit_id FK nullable
  start_seconds
  end_seconds
  confidence
  supporting_frame_ids_json

multimodal_chunks
  id PK
  transcript_chunk_id FK -> transcript_chunks.id
  alignment_run_id FK
  slide_interval_ids_json
  slide_source_unit_ids_json
  keyframe_ids_json
  alignment_confidence
  version
```

Important rules:

1. Store model name, input hash, and method version with every derived artifact.
2. A changed slide image, transcript, model, or configuration makes dependent
   output stale; it must not silently appear current.
3. Delete cascades must remove metadata and owned files deliberately.
4. A failed alignment run must not overwrite the last completed run.
5. Every page-reader output conforms to `PageContent`; model-specific tensors
   and optimizer state do not belong in SQLite.

---

## 7. Assignment Method

The structure is inspired by Stanford CS336's implementation-heavy approach:
small interfaces, failing tests, correctness on tiny inputs, explicit resource
accounting, then full experiments.

For every assignment:

1. **Read:** inspect the listed current interfaces and the assigned paper.
2. **Specify:** write data shapes, invariants, and failure states before code.
3. **Implement:** write the core logic in a small module.
4. **Overfit/debug:** prove the model or algorithm can solve a tiny controlled
   case.
5. **Test:** add unit tests and an end-to-end fixture with external tools mocked.
6. **Experiment:** run the frozen protocol and retain configuration plus results.
7. **Write up:** explain failures, not only the best number.

The learning contract for the handwritten model assignments is:

- core CNN, attention, training-loop, and dynamic-programming functions are
  written and understood by the learner;
- library tensor operations are allowed;
- complete model constructors such as `torchvision.models.vit_*` are not used in
  the from-scratch track;
- pretrained libraries are allowed only in the explicitly labeled product
  baseline track;
- tests are written before long GPU runs;
- a 32-example overfit test is required before full training.

---

## 8. Frozen Evaluation Contract

Evaluation design must be decided before CNN/ViT training. Otherwise model
selection will leak into the test set.

### 8.1 Two annotation units

Transition detection uses event annotations rather than labels for every frame:

```text
(lecture_id,
 change_start_seconds,
 stable_at_seconds,
 from_page,
 to_page,
 event_type)
```

`event_type` distinguishes `page_change`, `content_build`, `enter_slide`,
`leave_slide`, and `non_semantic_motion`. The interval from `change_start` to
`stable_at` represents an animation/fade period during which OCR should not run.

Page reading uses one stable keyframe and a corrected reference:

```text
(lecture_id,
 page_event_id,
 stable_frame_timestamp,
 image_path,
 gold_text,
 technical_terms,
 gold_concepts)
```

When the matching PPTX/PDF is available, its native text initializes `gold_text`
but must be manually checked on the evaluation subset for reading order,
decorative text, equations, and missing glyphs.

### 8.2 Split policy

Never randomly split frames. Neighboring frames are near duplicates, so a random
frame split would produce severe leakage.

Use three levels:

- **Development smoke set:** the current five CS231n lectures, used to make the
  pipeline run and reveal annotation/modeling problems.
- **Formal lecture-held-out split:** entire lectures belong to train, validation,
  or test, never more than one.
- **Generalization split:** if enough data is collected, hold out an instructor,
  course, or slide template family.

Five lectures are not enough to make a strong general claim about CNN versus
ViT. A defensible comparison should use several independent lectures in
each split or a public aligned dataset in addition to the personal CS231n test
case. The report must state the sample size and uncertainty.

### 8.3 Three comparison tracks

**Track T: transition baseline**

- FFmpeg `scdet`;
- PySceneDetect `ContentDetector` and `AdaptiveDetector`;
- anchor-frame SSIM/perceptual-hash detector with hysteresis;
- identical annotated event intervals and one-to-one event matching;
- threshold selection on validation lectures only.

This track chooses the low-cost module that decides when a new stable page must
be read. CNN/ViT readers are not part of this comparison.

**Track S: controlled reader experiment**

- same train/validation/test lectures;
- same stable page images, text boxes, and reading order;
- same input resolution and video-corruption augmentations;
- same tokenizer, vocabulary, CTC/sequence decoder, and loss;
- comparable parameter budget;
- comparable optimizer-search budget;
- same early-stopping rule;
- three to five random seeds.

Only the visual encoder changes: CNN versus ViT. This measures inductive bias,
data efficiency, and the ability to preserve text under lecture-video noise.

**Track P: product/pretrained**

- mature local OCR baseline;
- pretrained CRNN-style reader;
- pretrained ViTSTR/TrOCR-style reader;
- native PPTX/PDF text as an oracle upper bound;
- identical downstream card generator and card validator.

This measures actual product value. Results from Track S and Track P must not be
merged into one winner statement.

### 8.4 Transition metrics

- **Event Precision/Recall/F1:** one predicted transition can match at most one
  gold transition within the configured tolerance.
- **F1@0.5s/F1@1s/F1@2s:** sensitivity to timestamp tolerance.
- **False transitions/hour:** operational cost of unnecessary page reads.
- **Missed transitions/hour:** pages never sent to the reader.
- **Duplicate triggers:** extra predictions near an already matched event.
- **Stable-frame success:** selected frame is at or after `stable_at` and before
  the next event.
- **Stable capture delay:** selected time minus `stable_at`.
- **Candidate compression ratio:** accepted page reads divided by sampled frames.

Raw frame accuracy is not a primary metric because almost all sampled frame
pairs are non-transitions; a detector that predicts “no change” everywhere can
look accurate while being useless.

### 8.5 Page-reading metrics

- **CER:** character-level edit operations divided by reference characters.
- **WER:** word-level edit operations divided by reference words.
- **Exact line accuracy:** completely correct recognized lines.
- **Technical-term recall:** fraction of gold technical terms preserved.
- **Title/bullet recall:** structure-sensitive recovery of important blocks.
- **Reading-order accuracy:** whether blocks appear in the expected sequence.
- **Blank-page/abstention precision:** the reader does not invent text where
  there is no readable slide.

Report a macro average over lectures. A frame-weighted average lets one long or
slow-changing lecture dominate the result.

If an exact source deck is available, also report page retrieval Recall@1,
Recall@5, and MRR. This is a separate task from OCR and must not replace text
recognition metrics.

### 8.6 Card-conversion metrics

All readers feed the same transcript, prompt, local Qwen model, validator, and
generation settings. Compare these systems:

```text
audio only
oracle native slide text + audio
CNN reader text + audio
ViT reader text + audio
mature OCR text + audio
```

The oracle isolates the cascade:

```text
oracle -> reader gap = page-reading loss
reader -> card gap   = grounding/generation loss
audio -> multimodal  = value added by visual content
```

Required card metrics:

- **Grounded claim precision:** supported atomic claims / all generated claims.
- **Concept recall:** gold important concepts represented / all gold concepts.
- **Citation correctness:** cited source actually supports the claim.
- **Citation completeness:** support-requiring claims with valid evidence.
- **Technical-term accuracy:** terms remain correct after generation.
- **Question answerability:** active-recall question can be answered from the
  evidence.
- **Answer consistency:** saved answer agrees with the evidence.
- **No-edit acceptance rate:** cards accepted without material correction.
- **Mean edit time:** human time needed to make a card usable.
- **Usable card conversion:** usable cards / eligible gold concepts.

Do not use raw card count as a quality metric. A system can inflate card count by
producing duplicates or low-value cards.

### 8.7 Temporal/end-to-end metrics

- **Frame assignment accuracy:** percentage of sampled timestamps assigned the
  correct slide.
- **Macro frame accuracy:** calculate accuracy per lecture, then average.
- **Boundary precision/recall/F1:** a predicted transition matches one gold
  transition within a tolerance such as 1 second, with one-to-one matching.
- **Temporal interval IoU:** intersection over union between predicted and gold
  slide intervals.
- **Over-segmentation rate:** one gold interval split into too many predicted
  intervals.
- **Under-segmentation rate:** multiple gold intervals merged into one.

The primary product metric is successful page-to-card conversion: a real
transition is detected, a stable page is read, and at least one supported useful
card or source enrichment is produced. Report each upstream component as well so
the end-to-end score cannot hide the failure stage.

### 8.8 Efficiency metrics

- trainable parameter count;
- approximate FLOPs or multiply-accumulate operations;
- checkpoint size;
- peak GPU memory;
- training wall-clock time;
- inference milliseconds per accepted page;
- number of expensive reader calls per lecture hour;
- end-to-end minutes per one-hour lecture;
- CPU fallback behavior.

Measure latency after warm-up, with fixed batch sizes, on the same machine. The
hardware and software versions belong in the report.

### 8.9 Robustness slices

Evaluate each model on labeled subsets:

- blur and motion blur;
- perspective distortion;
- low resolution and video compression;
- subtitles or player controls over the slide;
- instructor camera overlay;
- animations/build slides;
- code demos and whiteboards;
- repeated or near-duplicate slides;
- unseen slide template;
- no-slide/unknown frames.

### 8.10 Statistics

- Run at least three seeds for trained models.
- Report mean and standard deviation.
- Bootstrap confidence intervals by **lecture**, not by frame.
- Use paired lecture-level differences when comparing CNN and ViT.
- Do not tune thresholds, fusion weights, or transition penalties on the test
  set.

---

## 9. Assignment 0: Dataset and Evaluation Harness

Estimated time: 3-5 days

### Learning objectives

- Understand label design, leakage, train/validation/test splits, and evaluation
  units.
- Learn why an ML problem begins with a contract rather than a model.
- Build reproducible manifests and metric tests.

### Read first

- `backend/app/transcription.py`
- `backend/app/transcript_chunk.py`
- `backend/app/source_asset.py`
- Lecture Presentations Multimodal Dataset paper
- MaViLS evaluation setup

### Work

1. Define `dataset_manifest.jsonl` for lecture video, source deck, course,
   instructor, duration, and split.
2. Define `transition_events.jsonl` with `change_start`, `stable_at`, page IDs,
   and event type.
3. Define `page_content_references.jsonl` with corrected text, technical terms,
   and gold concepts for a small stable-page subset.
4. Write an annotation guide covering builds, camera motion, cursor motion,
   repeated slides, demos, fades, and no-slide intervals.
5. Implement one-to-one event Precision/Recall/F1 and timing-error metrics.
6. Implement CER, WER, exact match, and technical-term recall.
7. Implement count-based card metrics for claim support, concept coverage,
   citation quality, no-edit acceptance, and usable conversion.
8. Add synthetic fixtures where every expected metric is hand-computable.

### Annotation strategy

Do not label every 30 fps frame. Label each meaningful transition as an interval
from the first visible change to the first stable readable frame. Generate
non-transition candidates from stable interval interiors. For page reading,
manually correct only a stratified subset of stable pages, including equations,
technical terms, animations, and corrupted frames. The current five lectures are
a smoke/case-study set; expand or use public data before general claims.

### Required tests

- a perfect prediction scores 1.0;
- duplicate predictions around one transition produce one TP and extra FPs;
- event tolerance is inclusive and configurable;
- non-semantic motion is not counted as a positive page transition;
- CER/WER use known edit-distance examples;
- an empty reference fails explicitly rather than creating a misleading score;
- technical-term matching is normalized but does not reward arbitrary
  substrings;
- card conversion rejects a zero eligible-concept denominator;
- repeated occurrences of the same slide are treated as distinct intervals;
- macro averaging gives equal lecture weight;
- train and test lecture IDs cannot overlap.

### Deliverables

- schemas and sample manifests;
- annotation guide;
- tested metric functions;
- a one-page data statement describing source, licensing, split, and limitations.

### Pass gate

No vision model work begins until metrics pass on hand-computed examples and the
split checker proves there is no lecture overlap.

### Write-up questions

1. Why is raw frame accuracy misleading for sparse transition events?
2. Why do `change_start` and `stable_at` need separate timestamps?
3. Why must CNN and ViT share the detector/decoder in an encoder comparison?
4. Why is raw card count not a valid conversion-quality metric?

---

## 10. Assignment 1: Native Slide Ingestion and Rendering

Estimated time: 4-6 days

### Learning objectives

- Separate semantic source parsing from visual rendering.
- Learn adapter/protocol design around external tools.
- Preserve provenance and reproducibility of generated artifacts.

### Current starting point

`source_asset_parser.py` already extracts PPTX text into slide-level
`SourceUnit`s. Reuse that. `python-pptx` reads structure but is not a full slide
renderer, so rendering needs an adapter.

### Work

1. Add a `SlideRenderer` protocol.
2. Implement one local renderer adapter, preferably:
   `PPTX -> headless LibreOffice -> PDF -> page images`.
3. Keep an optional Windows PowerPoint adapter as a later platform-specific
   alternative, not the only implementation.
4. Join rendered images back to existing slide source units by slide number.
5. Store renderer name/version and input/output hashes.
6. Preserve native slide text; optionally extract speaker notes as a distinct
   field.

### Important product constraint

LibreOffice should be detected as an optional runtime dependency. The Tauri app
must show a useful missing-runtime message instead of failing during import.

### Required tests

- fake renderer produces one image for each slide;
- slide count mismatch fails clearly;
- rendering the same input gives stable ordering and hashes;
- a failed render does not delete existing source units;
- paths remain under the configured app data directory;
- the parser can still extract PPTX text without a renderer installed.

### Evaluation

- render success rate;
- slide count parity;
- median rendering time per slide;
- visual inspection grid for fonts, diagrams, equations, and aspect ratios.

### Pass gate

One CS231n deck can be imported, parsed, rendered, reopened after backend
restart, and traced from image back to `SourceAsset` and slide number.

### Write-up questions

1. Why should rendered images not replace `SourceUnit`?
2. Why are hashes and renderer versions needed?
3. Why is native PPT text preferable to OCR when available?

---

## 11. Assignment 2: Transition and Stable-Frame Baseline

Estimated time: 5-7 days

### Learning objectives

- Use FFmpeg as a bounded-memory preprocessing stage.
- Understand event detection, hysteresis, stabilization, and temporal
  tolerances.
- Establish a non-neural baseline.

### Work

1. Add a `VisualFrameExtractor` protocol and FFmpeg implementation.
2. Begin with fixed sampling, for example 2 frames/second, and refine timestamps
   near candidates only when needed.
3. Benchmark FFmpeg `scdet`, PySceneDetect `ContentDetector`, and
   `AdaptiveDetector`.
4. Implement an anchor-frame detector using SSIM/perceptual hash. Compare the
   current sample with the last accepted stable page, not only its adjacent
   frame.
5. Add two-threshold hysteresis: a high threshold enters `changing`; several
   consecutive low-difference samples enter `stable`.
6. Cluster/debounce candidate bursts caused by animations and fades.
7. Select the sharpest readable keyframe after stability.
8. Store only event metadata and selected keyframes, not every sampled frame.

### Why the baseline matters

The reader should run once per meaningful stable page. A useful detector therefore
optimizes both event recall and expensive-call reduction. CNN/ViT OCR results are
not interpretable if the detector sends them blurry transition frames.

### Required tests

- timestamp ordering is strict;
- a synthetic unchanged sequence creates no transition;
- one abrupt image change creates one debounced transition;
- a multi-frame animation produces one event;
- cursor motion does not become a page change in the annotated fixture;
- stable frame selection happens at/after `stable_at`;
- frames retain exact FFmpeg timestamps;
- a one-hour video is processed as a stream/batches rather than loaded fully in
  memory;
- FFmpeg invocation is mocked in unit tests.

### Evaluation

- boundary precision/recall/F1 at 0.5, 1.0, and 2.0 second tolerances;
- false transitions/hour, missed transitions/hour, and duplicate triggers;
- stable-frame success rate and stable-capture delay;
- candidate compression ratio;
- processing time per one-hour lecture;
- number and disk size of retained frames.

### Pass gate

The frame extractor is deterministic and bounded in memory. The selected
transition baseline is evaluated on all five smoke-set lectures and feeds stable
readable keyframes to the next assignment.

### Write-up questions

1. Why can 1 fps miss a fast transition?
2. Why can animation create false boundaries?
3. Which threshold was selected on validation data, and how sensitive is the
   result to it?

---

## 12. Assignment 3: Page-Reading and Oracle Baselines

Estimated time: 1 week

### Learning objectives

- Separate text detection, text recognition, layout ordering, and source-deck
  matching.
- Establish mature OCR and native-text upper bounds.
- Measure how video corruption changes page-reading quality.

### Inputs

For each accepted stable keyframe, produce one common contract:

```text
PageContent
  page_id/page_number (optional)
  ordered text blocks
  full normalized text
  technical terms
  confidence and provenance
```

When the source deck is available, native text is the oracle and production
preference after page alignment. OCR remains necessary for deck-free lectures,
visible annotations, and the CNN/ViT reader experiment.

### Work

1. Define common `TextDetector`, `PageReader`, and `PageContent` protocols.
2. Use one mature local OCR system as the product baseline.
3. Extract native PPTX/PDF text and manually correct a small oracle subset.
4. Normalize Unicode/whitespace for scoring while retaining raw text.
5. Extract a technical-term list and gold concepts for the benchmark subset.
6. If the deck is available, match the keyframe to a source page and report page
   retrieval separately from OCR.
7. Serialize every reader output to the same `PageContent` JSON schema.

### Required tests

- a reader cannot return text without source/keyframe provenance;
- empty reference text is handled explicitly;
- normalization preserves technical symbols in raw output;
- ordered blocks deterministically produce full text;
- cached output becomes stale when image, model, or preprocessing changes;
- native and OCR outputs use the same evaluation interface.

### Evaluation

Report CER, WER, exact-line accuracy, technical-term recall, structure recovery,
latency, and abstention errors. When page matching is used, additionally report
Recall@1, Recall@5, and MRR.

### Pass gate

The oracle and mature OCR paths both produce valid `PageContent`; the OCR
baseline has measured errors on real stable frames; and every failure can be
traced to detection, recognition, ordering, or page matching.

### Write-up questions

1. Why is native deck text an oracle rather than a fair OCR prediction?
2. Why must raw and normalized text both be retained?
3. Why should page retrieval and text recognition have separate metrics?

---

## 13. Assignment 4: Handwritten CNN Page Reader

Estimated time: 1-2 weeks

### Learning objectives

- Understand convolution, receptive fields, pooling, normalization, sequence
  features, CTC alignment, and decoding.
- Build and debug a complete PyTorch training loop.
- Understand why a visual encoder alone cannot emit variable-length text.

### Recommended formulation

Use the common text detector to crop ordered text lines. The CNN reader turns
each crop into a left-to-right feature sequence and uses the shared CTC head:

```text
stable page
-> shared text detector
-> ordered line crops
-> CNN feature map
-> collapse height into a width-wise sequence
-> shared linear vocabulary head
-> CTC loss / CTC decode
-> PageContent
```

The detector, tokenizer, vocabulary, CTC implementation, line ordering, and
postprocessing are frozen for the CNN/ViT comparison. Only the encoder changes.

### Data construction

- Clean training labels come from native slide text and corrected line crops.
- Synthetic corruption adds blur, perspective, rescaling, compression,
  brightness shifts, partial occlusion, subtitles, and speaker overlays.
- Real stable keyframes form validation/test examples and are split by lecture.
- Samples with incorrect or ambiguous reading order are flagged rather than
  silently treated as clean labels.

### Implementation constraints

- Use primitive PyTorch layers such as `Conv2d`, normalization, pooling, and
  linear layers.
- Do not call a complete torchvision CNN model in Track S.
- Use the shared character vocabulary and CTC decoder from the evaluation
  harness; do not give CNN a different language model.
- Write parameter counting and checkpoint save/load utilities.
- Store training config, seed, git commit, and metric history with the
  checkpoint.

### Debug ladder

1. Verify tensor shapes on one batch.
2. Verify all gradients are finite and nonzero where expected.
3. Overfit one short text line.
4. Overfit 32 line crops until decoded text is nearly exact.
5. Run one lecture split.
6. Only then start the full controlled experiment.

### Required tests

- expected `[batch, time, vocabulary]` logit shape;
- deterministic inference in evaluation mode;
- save/load produces the same logits within tolerance;
- a training step changes parameters;
- tiny-batch loss decreases;
- decoded sequences correctly collapse repeats and blanks;
- no lecture ID crosses the split boundary.

### Required experiments

- clean render versus corrupted-video performance;
- depth/width and sequence-resolution ablations;
- case-sensitive and normalized CER/WER;
- technical-term recall by corruption type;
- CNN reader text versus oracle text in the frozen card generator.

### Pass gate

The model overfits the tiny set, completes reproducibly on the formal split, and
is evaluated even if it does not beat the baseline. A negative result with sound
analysis is a valid result; an untested architecture is not.

### Write-up questions

1. How does a 2-D CNN feature map become a 1-D character sequence?
2. Why can CTC train without character-level bounding boxes?
3. Which video corruptions damage technical terms most?

---

## 14. Assignment 5: Handwritten ViT Page Reader

Estimated time: 1-2 weeks

### Learning objectives

- Implement patchification, positional embeddings, multi-head self-attention,
  pre-normalization, residual connections, and MLP blocks.
- Understand why ViTs can be data-hungry.
- Convert patch tokens into the same character vocabulary/CTC contract used by
  the CNN reader.

### Required components

```text
image
-> non-overlapping patches
-> linear patch projection
-> positional embedding
-> N x TransformerBlock
-> ordered visual token sequence
-> shared linear vocabulary head
-> CTC loss / CTC decode
-> PageContent
```

Each transformer block contains:

```text
x = x + MultiHeadSelfAttention(LayerNorm(x))
x = x + MLP(LayerNorm(x))
```

Implement Q/K/V projections, scaled dot-product attention, head reshape, and the
output projection explicitly for the learning track. A library attention layer
may be used later as a parity test, not as the only implementation.

### Fairness controls

- same input resolution as CNN;
- same text detector, line crops, character vocabulary, and decoder;
- comparable parameter budget;
- same examples, augmentation pipeline, loss, and metric code;
- same validation-tuning budget;
- report actual FLOPs and memory even when parameter counts match.

### Debug ladder

1. Test patch count and patch ordering on a toy image.
2. Check that attention rows sum to one after softmax.
3. Check masking and tensor reshapes on tiny dimensions.
4. Overfit one text line and then one batch.
5. Match outputs/gradients against a simple reference attention calculation.
6. Run the same lecture split used by CNN.

### Required tests

- patchification has the expected `[batch, tokens, patch_dim]` shape;
- invalid image/patch sizes fail clearly;
- attention output and gradient shapes are correct;
- positional embeddings affect token order;
- output logits use the shared vocabulary dimension;
- checkpoint round trip is stable;
- tiny-batch loss decreases;
- parameter count falls within the selected comparison band.

### Required experiments

- tiny ViT reader versus parameter-matched CNN reader;
- patch-size ablation;
- augmentation-strength ablation;
- optional DeiT-style distillation as a stretch goal;
- optional pretrained ViTSTR/TrOCR baseline in Track P.

### Pass gate

The ViT passes the same correctness, overfit, split, and evaluation gates as the
CNN. It is not declared worse merely because a small from-scratch dataset favors
CNN inductive bias; that result must be interpreted in context.

### Write-up questions

1. How does patch size change token count, reading resolution, and attention
   cost?
2. Why is a scratch ViT reader versus pretrained OCR comparison invalid?
3. On which text-corruption slices does global attention help or hurt?

---

## 15. Assignment 6: CNN vs ViT Evaluation Report

Estimated time: 1 week

### Learning objectives

- Conduct a controlled model comparison.
- Separate statistical uncertainty, model quality, and system utility.
- Learn ablation and error-analysis practice.

### Mandatory result table

| Reader | Pretraining | Params | FLOPs | CER | WER | Term Recall | Exact Line | ms/page | Peak VRAM |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| oracle native text | none | - | - | 0 | 0 | 1 | 1 | | |
| mature OCR | specified | | | | | | | | |
| custom CNN + shared CTC | none | | | | | | | | |
| custom ViT + shared CTC | none | | | | | | | | |
| pretrained CRNN | specified | | | | | | | | |
| pretrained ViTSTR/TrOCR | specified | | | | | | | | |

Add an end-to-end card table using exactly the same generator:

| Input source | Claim Precision | Concept Recall | Citation Correctness | Answerability | No-edit Rate | Usable Conversion |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| audio only | | | | | | |
| oracle slide text + audio | | | | | | |
| CNN reader + audio | | | | | | |
| ViT reader + audio | | | | | | |
| mature OCR + audio | | | | | | |

### Mandatory plots

- train/validation loss and CER by epoch;
- CER and technical-term recall versus parameter count;
- usable card conversion versus reader latency;
- performance by robustness slice;
- character/term error confusion examples;
- paired per-page CNN-versus-ViT differences;
- upstream CER versus downstream claim/concept quality.

### Required ablations

1. CNN versus ViT with the same data and loss.
2. Scratch versus pretrained.
3. Clean slide render versus real/corrupted stable frames.
4. Audio-only versus oracle multimodal context.
5. Oracle, CNN, ViT, and mature OCR through the same card generator.
6. With and without technical-term-aware augmentation.

### Decision rule

Choose the production encoder using a Pareto decision, not one accuracy number:

- CER/WER and technical-term preservation;
- grounded claim precision and concept recall;
- local latency and memory;
- checkpoint/download size;
- CPU fallback;
- robustness to real lecture artifacts;
- licensing and redistribution.

The custom CNN and ViT remain documented learning artifacts even if production
uses native deck text or a mature OCR reader.

### Pass gate

All models use frozen splits and metric code; every reported number is generated
from a saved config/checkpoint; conclusions include uncertainty and at least 20
categorized failure examples.

---

## 16. Assignment 7: Sequence-Aware Slide Alignment

Estimated time: 1 week

### Learning objectives

- Convert local similarity into a global sequence prediction.
- Implement dynamic programming/Viterbi decoding and backtracking.
- Model temporal continuity without assuming strictly monotonic lectures.

### Problem

An independent argmax can predict:

```text
slide 3 -> slide 4 -> slide 19 -> slide 4 -> slide 5
```

for five adjacent timestamps because one noisy frame resembles slide 19. The
deck order and temporal continuity provide useful structure.

Let `S[i, j]` be the fused emission score between frame `i` and slide state `j`,
including `UNKNOWN`. Let `A[k, j]` be a transition score from previous state `k`
to current state `j`:

```text
D[i, j] = S[i, j] + max_k(D[i - 1, k] + A[k, j])
```

Store the maximizing `k` as a backpointer, then backtrack from the best final
state.

### Transition policy

- staying on the same slide has low/no penalty;
- moving forward one slide has a small penalty;
- skipping several slides is allowed but costs more;
- moving backward is allowed because lecturers revisit slides;
- entering/leaving `UNKNOWN` is allowed;
- penalty may depend on elapsed time between irregular samples;
- no hard-coded assumption that every lecture starts at slide 1 or ends at the
  final slide.

Tune transition weights on validation lectures only. MaViLS is a useful model,
but its exact penalty should not be copied blindly into courses with different
editing and navigation patterns.

### Work

1. Implement greedy decoding as the reference baseline.
2. Implement Viterbi/dynamic-programming decoding with backpointers.
3. Add `UNKNOWN` emissions and transitions.
4. Convert decoded states into `SlideInterval`s.
5. Calculate confidence from score margin, entropy, and/or agreement across
   modalities; document the chosen definition.
6. Persist an immutable alignment run and its configuration.

### Required tests

- a tiny hand-computed matrix returns the expected path;
- a case where greedy fails is corrected by sequence decoding;
- backward navigation is possible;
- `UNKNOWN` spans are preserved;
- repeated slides form separate intervals;
- start/end times exactly cover the sampled timeline without overlap errors;
- failed runs leave the previous completed run intact.

### Evaluation

Compare greedy versus sequence-aware decoding on the same emissions:

- macro frame accuracy;
- boundary F1;
- interval IoU;
- implausible jump count;
- unknown F1;
- added runtime.

### Pass gate

Dynamic programming improves at least one primary temporal metric without a
material unexplained regression in unknown detection, and all transition
behavior is visible in configuration rather than buried in code.

### Write-up questions

1. Why can temporal decoding improve a weak frame-level classifier?
2. When can a strong forward-jump penalty make predictions worse?
3. How should confidence change when visual and transcript evidence disagree?

---

## 17. Assignment 8: Align Transcript Chunks with Slides

Estimated time: 4-6 days

### Learning objectives

- Join two timestamped interval systems.
- Preserve many-to-many provenance.
- Design multimodal chunks without destroying semantic transcript boundaries.

### Alignment rule

For transcript chunk interval `C=[c_start,c_end]` and slide interval
`S=[s_start,s_end]`:

```text
overlap(C, S) = max(0, min(c_end, s_end) - max(c_start, s_start))
coverage(C, S) = overlap(C, S) / duration(C)
```

Attach every slide interval whose overlap exceeds a configured minimum duration
or coverage threshold. A chunk may reference multiple slides when the instructor
changes slides mid-explanation.

Do not split every transcript chunk automatically at every visual transition.
That would damage semantic chunks. Instead:

1. preserve the original transcript chunk;
2. attach all overlapping slide evidence;
3. optionally expose a later `split_on_strong_transition` experiment;
4. record the enrichment algorithm version.

### Work

1. Implement interval overlap as a pure function.
2. Add a sweep-line join for ordered chunks/slide intervals.
3. Select representative keyframes.
4. Calculate aggregate alignment confidence.
5. Persist `MultimodalChunk` rows separately from transcript chunks.
6. Add re-enrichment when a newer alignment run is selected.

### Required tests

- no overlap;
- exact boundary touch has zero duration;
- one chunk maps to one slide;
- one chunk maps to multiple slides;
- one slide maps to multiple chunks;
- unknown intervals do not masquerade as slide evidence;
- the original transcript text, segment IDs, and times are unchanged.

### Pass gate

Every generated multimodal chunk can be traced to original transcript segments,
slide source units, alignment run, and representative frames.

### Write-up questions

1. Why is this a many-to-many relation?
2. What information is lost if chunks are rewritten in place?
3. Which overlap threshold works best, and what errors occur near slide changes?

---

## 18. Assignment 9: Multimodal Grounded Card Generation

Estimated time: 1 week

### Learning objectives

- Extend claim-level grounding across modalities.
- Design prompts and validators around provenance rather than fluent output.
- Evaluate whether visual evidence actually improves cards.

### Generation context

The initial local Qwen path remains text-based. Convert multimodal evidence into
a structured context:

```text
TRANSCRIPT
[12:04.0-12:18.2] exact transcript quote ...

MATCHED SLIDES
[slide 17, confidence 0.91]
title: Singular Value Decomposition
native text: ...

VISUAL LOCATOR
keyframe at 12:10.0, local artifact id ...
```

The first version does not ask a text-only model to invent diagram meaning from
an unseen image. Native slide text is evidence; a keyframe is provenance for the
user. A later vision-language model can add diagram descriptions under a
separate evidence policy.

### Claim policy

- Transcript-grounded claim: must cite an exact transcript quote and timestamp.
- Slide-grounded claim: must cite exact native/OCR slide text and slide number.
- Joint claim: may cite both.
- Diagram-only claim: unsupported until a verified visual-description path
  exists.
- Conflicting modalities: preserve the conflict or abstain; do not silently pick
  the more convenient source.
- Low alignment confidence: fall back to audio-only generation or require user
  review.

### Work

1. Generalize card evidence to source type, ID, locator, and confidence.
2. Build multimodal prompt serialization as a pure/tested function.
3. Extend claim validation by evidence type.
4. Keep manual transcript selection and automatic multimodal chunks on the same
   `card_service.py` generation path.
5. Save alignment version with generated cards.
6. Add frontend source links/thumbnails after backend provenance is stable.

### Evaluation protocol

Compare audio-only and audio+slide generation on the same chunks with a blinded
manual review:

- supported claim rate;
- unsupported claim rate;
- citation correctness;
- slide/timestamp locator correctness;
- technical term correction rate;
- concept coverage;
- duplicate card rate;
- card edit distance or user edit rate;
- latency.

Do not rely only on an LLM judge. Manually label a fixed benchmark subset and
retain disagreements.

### Required tests

- a slide claim without slide evidence is rejected;
- transcript quote validation still works;
- low-confidence alignment triggers the configured fallback;
- source locators survive save/load/edit/export;
- deleting a source does not leave silently valid evidence;
- old audio-only cards remain readable during migration.

### Pass gate

Multimodal generation improves at least one predeclared quality metric without
increasing unsupported claims beyond the chosen tolerance, and every accepted
claim opens a real source locator.

---

## 19. Assignment 10: Product and Desktop Integration

Estimated time: 1 week

### Learning objectives

- Turn a research pipeline into a resumable local product workflow.
- Separate optional runtimes, background execution, persistence, and UI state.

### Backend workflow

```text
POST /jobs/{job_id}/align-slides
-> validate deck and runtime
-> create immutable alignment run
-> background preprocessing/inference
-> persist progress and outputs

GET /jobs/{job_id}/alignments/{run_id}
-> status, progress, config, error

GET /jobs/{job_id}/slide-timeline
-> intervals and confidence

POST /jobs/{job_id}/multimodal-chunks
-> join current transcript chunks with selected alignment run
```

Exact route names can follow current API conventions when implementation begins.

### Frontend workflow

- choose/import a deck for a video;
- start alignment and see real stage progress;
- inspect timeline thumbnails and matched slide numbers;
- see confidence and `unknown` spans;
- correct a wrong interval manually;
- generate cards from approved multimodal chunks;
- open transcript, slide, and keyframe evidence from a card.

The UI should expose the result, not model internals. Detailed similarity matrices
belong in a developer/experiment view.

### Runtime and packaging

- FFmpeg remains required for frame extraction.
- slide renderer and OCR are optional runtimes with checks and instructions.
- CUDA use is optional; CPU inference must fail gracefully or offer a slower
  path.
- checkpoint path, model version, and device appear in diagnostics.
- training dependencies are excluded from the lightweight installer.
- downloaded checkpoints are checksum-verified and stored under app data.
- app shutdown must not corrupt an active alignment run.

### Required tests

- service tests use fake renderer, OCR, and visual encoder;
- API tests cover success, retry, cancellation/failure, and missing runtime;
- restart resumes or clearly fails an incomplete run;
- frontend polling reaches completed and failed terminal states;
- the packaged app can find artifacts outside the repository directory.

### Pass gate

A fresh user can import one video and its deck, run local alignment, inspect and
correct the slide timeline, generate grounded cards, restart the app, and retain
all data.

---

## 20. Capstone Experiment

The capstone is not “train the biggest model.” It is a reproducible answer to:

> Does multimodal alignment improve the usefulness and grounding of course
> knowledge cards enough to justify its local compute and complexity?

### Fixed experimental conditions

- frozen lecture-level split;
- fixed annotation set;
- fixed CPU/GPU hardware;
- fixed input resolution and sampling policy;
- fixed card-generation prompt/model settings;
- all configs and checkpoints retained;
- no test-set threshold tuning.

### Systems to compare

1. Audio-only current system.
2. Audio + oracle native slide text.
3. Audio + mature local OCR text.
4. Audio + custom CNN-reader text.
5. Audio + custom ViT-reader text.
6. Audio + selected pretrained reader text.

### Primary outcomes

- transition F1 and false transitions/hour;
- CER/WER and technical-term recall;
- grounded claim precision and concept recall;
- citation correctness/completeness;
- no-edit acceptance and usable card conversion;
- total processing time per lecture.

### Final report structure

1. Problem and hypothesis.
2. Existing system and exact code boundaries.
3. Dataset, annotation policy, and leakage controls.
4. Transition and stable-frame baselines.
5. Oracle/mature OCR baselines.
6. CNN reader design and training.
7. ViT reader design and training.
8. Controlled reader and card-cascade comparison.
9. Audio-only versus multimodal card evaluation.
10. Failure taxonomy and limitations.
11. Local deployment/resource tradeoffs.
12. Reproducibility instructions.

### Resume-ready evidence

After completion, a defensible summary would be:

> Built a local multimodal lecture-understanding pipeline that detects stable
> slide events, reads page content, and aligns it with timestamped speech.
> Implemented controlled CNN- and ViT-based text readers under lecture-held-out
> splits, propagated source evidence into grounded knowledge cards, and measured
> transition F1, OCR error, technical-term recall, claim support, card conversion,
> latency, and local compute cost.

Only include metrics actually measured by the capstone.

---

## 21. Recommended Execution Order

### Phase A: Make the problem measurable

1. Assignment 0: dataset and evaluation harness.
2. Assignment 1: native slide rendering.
3. Assignment 2: transition and stable-frame baseline.
4. Assignment 3: oracle and mature page-reading baselines.

Exit criterion: a labeled, reproducible baseline detects stable page events,
reads them into `PageContent`, and reports honest component metrics.

### Phase B: Learn and compare page readers

5. Assignment 4: handwritten CNN.
6. Assignment 5: handwritten ViT.
7. Assignment 6: fair CNN/ViT report.

Exit criterion: both models pass correctness and tiny-overfit tests, share the
same detector/tokenizer/decoder, and have a controlled text-to-card comparison.

### Phase C: Turn page events into a timeline

8. Assignment 7: dynamic-programming alignment.
9. Assignment 8: multimodal chunks.

Exit criterion: every transcript chunk can identify overlapping slide intervals
and provenance, including uncertain/unknown regions.

### Phase D: Improve the learning product

10. Assignment 9: multimodal grounded cards.
11. Assignment 10: frontend and Tauri integration.
12. Capstone evaluation.

Exit criterion: multimodal cards demonstrably improve predefined quality metrics
and remain usable in the packaged local app.

### Approximate calendar

| Week | Focus |
| --- | --- |
| 1 | Assignment 0 and annotation smoke set |
| 2 | Assignment 1 slide rendering |
| 3 | Assignment 2 transition/stable-frame baseline |
| 4 | Assignment 3 oracle and mature OCR baselines |
| 5-6 | Assignment 4 CNN page reader |
| 7-8 | Assignment 5 ViT page reader |
| 9 | Assignment 6 controlled comparison |
| 10 | Assignment 7 temporal alignment |
| 11 | Assignment 8 multimodal chunks |
| 12 | Assignment 9 cards and Assignment 10 product integration |

This is a learning schedule, not a deadline. Do not begin a multi-hour training
run while an earlier correctness gate is failing.

---

## 22. Stop/Go Gates

### Gate 1: Is event-driven page reading feasible?

Proceed to custom models only if transition detection selects stable readable
frames and mature OCR/native-text baselines produce meaningful CER/term recall.
If not, inspect sampling, crop, stabilization, labels, and source quality first.

### Gate 2: Is custom training scientifically interpretable?

Proceed to CNN/ViT claims only if there are enough independent held-out lectures.
With five lectures, describe results as a case study or smoke experiment.

### Gate 3: Does temporal structure help?

Proceed to multimodal chunks only if sequence decoding improves temporal metrics
or clearly reduces implausible jumps. Otherwise retain greedy output and analyze
why deck navigation violates the transition model.

### Gate 4: Does multimodality improve cards?

Ship multimodal card generation only if citation correctness and/or technical
concept coverage improves without an unacceptable unsupported-claim increase.

### Gate 5: Is the local cost acceptable?

Ship the chosen encoder only if an hour-long lecture can be processed within the
declared time, disk, RAM/VRAM, and installer constraints.

---

## 23. Failure Taxonomy to Track from Day One

Every incorrect alignment should receive one main label:

- `deck_mismatch`: video and supplied deck differ;
- `sampling_miss`: transition occurs between sampled frames;
- `animation`: incremental build resembles multiple slides;
- `repeated_slide`: same/similar slide appears multiple times;
- `visual_corruption`: blur, crop, compression, overlay;
- `speaker_or_demo`: frame is not a deck slide;
- `ocr_failure`: visible text is not recovered;
- `asr_failure`: transcript is wrong or badly timed;
- `discussion_lead_lag`: speech discusses a different slide than the visible one;
- `encoder_confusion`: representation ranks the wrong candidate;
- `fusion_error`: one modality dominates incorrectly;
- `transition_model_error`: sequence prior suppresses a real jump;
- `annotation_ambiguity`: gold label is uncertain.

This taxonomy connects model work to product behavior and makes later research
questions concrete.

---

## 24. Optional Extensions After the Core Upgrade

These are intentionally deferred:

- OCR correction using slide vocabulary to improve ASR terms;
- diagram/figure region detection;
- vision-language description of figures with evidence validation;
- direct PDF/PPT figure-to-card citations;
- slide-aware semantic chunk boundaries;
- multimodal card embeddings;
- graph edges derived from shared visual/document evidence;
- active learning that asks users to correct only low-confidence intervals;
- weak supervision from deck order and transcript text;
- ImageBind-style shared audio/image/text representations;
- learned transition models or differentiable sequence alignment;
- end-to-end multimodal RAG over card, transcript, slide, and figure evidence.

They should be added only after the evaluation harness can show whether they
help.

---

## 25. Primary Reading List

### Assignment style and systems discipline

- [Stanford CS336: Language Modeling from Scratch](https://cs336.stanford.edu/)
- [CS336 Assignment 1 repository](https://github.com/stanford-cs336/assignment1-basics)

### Lecture video and slide alignment

- [Lecture Presentations Multimodal Dataset, ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/papers/Lee_Lecture_Presentations_Multimodal_Dataset_Towards_Understanding_Multimodality_in_Educational_Videos_ICCV_2023_paper.pdf)
- [MaViLS: Multimodal Video and Slide Alignment, Interspeech 2024](https://www.isca-archive.org/interspeech_2024/anderer24_interspeech.pdf)
- [SliTraNet: Automatic Detection of Slide Transitions in Lecture Videos](https://arxiv.org/abs/2202.03540)
- [SlideSpeech: A Large Scale Slide-Enriched Audio-Visual Corpus](https://arxiv.org/abs/2309.05396)
- [SlideSpeech project and dataset](https://slidespeech.github.io/)

### Transition detection

- [FFmpeg scene-change detection filters](https://ffmpeg.org/ffmpeg-filters.html#scdet)
- [PySceneDetect detector documentation](https://www.scenedetect.com/docs/latest/api/detectors.html)

### Page reading and grounded-card evaluation

- [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)
- [Training Data-Efficient Image Transformers and Distillation through Attention](https://proceedings.mlr.press/v139/touvron21a.html)
- [CRNN: End-to-End Image-Based Sequence Recognition](https://arxiv.org/abs/1507.05717)
- [ViTSTR: Vision Transformer for Scene Text Recognition](https://arxiv.org/abs/2105.08582)
- [TrOCR: Transformer-Based Optical Character Recognition](https://arxiv.org/abs/2109.10282)
- [FActScore: Atomic Factual Precision](https://arxiv.org/abs/2305.14251)
- [ALCE: Citation Evaluation for Grounded Generation](https://arxiv.org/abs/2305.14627)
- [QGEval: Multi-Dimensional Question Generation Evaluation](https://arxiv.org/abs/2406.05707)

---

## 26. Assignment 0 Pilot Result

Assignment 0 is complete for the first 15 minutes of CS231n 2025 Lecture 2.

### Imported source

- Video job: `524947e169d8423088a865e778c6e1ac`
- Official deck: `https://cs231n.stanford.edu/slides/2025/lecture_2.pdf`
- Source asset: `4138d43815a44e6a8dff099c748d1a17`
- Deck pages parsed by the existing source-asset layer: 102
- Pilot interval: `0.0-900.0` seconds

The official deck has updated footer styling, while the recorded video uses an
older CS231n template. The semantic page content and ordering match. Page
identity is therefore based on content, not pixel equality or footer text.

### Gold annotation bundle

Local artifacts live under
`backend/data/multimodal_lab/cs231n_2025_lecture_02/`:

- one dataset manifest;
- 40 visually reviewed events;
- 38 meaningful events (`page_change`, `content_build`, `enter_slide`, or
  `leave_slide`);
- two explicit non-semantic visual changes;
- 16 stable page references covering deck page 1 and pages 5-19;
- cropped stable frames selected after the final content build for each page.

The video revealed three conditions that a naive frame-difference detector
would conflate: hard page changes, progressive slide builds, and switches
between the lecturer camera and the slide feed. Those distinctions are now part
of the schema and evaluation contract.

### Reproducible validation

Run:

```powershell
cd backend
uv run python -m multimodal_lab.validate_annotations `
  --manifests data\multimodal_lab\cs231n_2025_lecture_02\dataset_manifest.jsonl `
  --transitions data\multimodal_lab\cs231n_2025_lecture_02\transition_events.jsonl `
  --references data\multimodal_lab\cs231n_2025_lecture_02\stable_page_references.jsonl
```

The validator checks event/page semantics, pilot bounds, unique identifiers,
reference timing, and agreement between each stable reference page number and
its event's `to_page`.

## 27. Assignment 1: Transition Baseline

The training-free baseline is implemented. It establishes a reproducible lower
bound before CNN/ViT experiments.

### Implemented pipeline

```text
video interval
-> FFmpeg crop + 2 fps sampling
-> FFmpeg adjacent-frame scene_score
-> streamed 160 x 90 RGB frames
-> Stanford footer marker slide-presence state
-> persistent spatial-change mask
-> header/body/footer rule-based event typing
-> temporal grouping
-> non-maximum suppression
-> SlideTransitionPrediction JSONL
```

The profile is explicit and versioned in
`backend/multimodal_lab/configs/cs231n_2025_web.json`. Its main calibrated
values are:

- scene score threshold: `0.002`;
- sampling rate: `2 fps`;
- stable lookahead: `1.0 s`;
- body changed-pixel minimum: `0.004`;
- header changed-pixel threshold for page changes: `0.025`;
- candidate grouping gap: `0.75 s`;
- NMS window: `1.5 s`.

The red CS231n footer acts as a training-free slide-presence marker. A marker
state transition produces `enter_slide` or `leave_slide`. While a slide is
present, a persistent header change is a `page_change`, a body-only change is a
`content_build`, and a transient top/footer overlay is
`non_semantic_motion`.

### Lecture 2 calibration result

For the manually reviewed `0-900 s` pilot:

- runtime: about `25.4 s` on the development machine;
- predictions: 40 total;
- target events: 38;
- relaxed F1: `1.0`;
- event-type-aware F1: `1.0`;
- mean absolute timing error: `0.303 s`;
- duplicate detections: 0;
- target false positives: 0.

The two emitted `non_semantic_motion` abstentions were visually checked: the
`27.0 s` event is player chrome over the slide footer, and the `30.5 s` event
is the course UI pill near the top-right corner. The opening logo-to-camera fade
does not create a slide event.

This is a calibration result, not held-out evidence. The thresholds were chosen
after inspecting this same interval. They must now be frozen before evaluating
a separately annotated lecture interval. A perfect calibration score must not
be presented as model generalization.

### Commands

```powershell
cd backend

uv run python -m multimodal_lab.run_transition_baseline `
  --video data\uploads\524947e169d8423088a865e778c6e1ac.mp4 `
  --start 0 --end 900 `
  --output data\multimodal_lab\cs231n_2025_lecture_02\baseline_predictions.jsonl

uv run python -m multimodal_lab.evaluate_transition_baseline `
  --annotations data\multimodal_lab\cs231n_2025_lecture_02\transition_events.jsonl `
  --predictions data\multimodal_lab\cs231n_2025_lecture_02\baseline_predictions.jsonl `
  --duration 900 --tolerance 1 `
  --output data\multimodal_lab\cs231n_2025_lecture_02\baseline_calibration_report.json
```

### Assignment 1.1: Held-out check (complete)

The frozen profile was evaluated once on the independently annotated first 15
minutes of CS231n 2026 Lecture 3. Before any prediction was run, the gold bundle
was fixed at 58 visual events, including 54 target events and 14 stable-page
references. The configuration hash remained
`3ba8e137ee214b8ab7802f89d522d59a9d983da76d925de08c85f90fd8e1ed03`.

Held-out ablation result:

| Method | Relaxed F1 | Typed F1 | Target FP/hour | Timing MAE |
| --- | ---: | ---: | ---: | ---: |
| Scene score only | 0.639 | 0.111 | 176.0 | 0.147 s |
| Scene score + slide state | 0.771 | 0.757 | 128.0 | 0.148 s |
| Spatial-state detector | 1.000 | 0.963 | 0.0 | 0.148 s |

The complete method detected every target event without duplicates. Its two
typed errors were a same-page animation that also changed the title and a true
PDF page advance that looked exactly like a body-only build. Neither error lost
the stable visual state required by the product.

Decision: keep the transition configuration frozen and proceed to page reading.
Do not train a transition CNN/ViT merely to resolve these two taxonomy-ambiguous
types. A future transition revision requires a new configuration hash and a new
held-out lecture.

The preregistered protocol, ablations, frozen configuration, result tables, and
threats to validity are maintained in
[`Multimodal transition study.md`](Multimodal%20transition%20study.md).
The compact machine-readable result is
[`experiments/assignment_1_transition_results.json`](experiments/assignment_1_transition_results.json).

## 28. Assignment 2: Page-Reading Baseline Result

The training-free page-reading baseline is complete. This implementation-stage
number corresponds to the roadmap's Page-Reading and Oracle Baselines
assignment.

### Implemented contract and readers

Every reader now returns the same provenance-preserving `PageContent` object:

```text
stable page reference
-> image hash + reader/preprocessing versions
-> ordered PageContentBlock values
-> raw text + normalized text
-> confidence, latency, source IDs, and cache key
```

The three reader paths are:

- `GoldReferencePageReader`, the manual evaluation ceiling;
- `NativeSourcePageReader`, the deck-available pypdf baseline;
- `RapidOcrPageReader`, the deck-free local PP-OCRv6/ONNX baseline.

The evaluator refuses to calculate CER/WER from semantic summaries. References
must explicitly use `gold_text_scope=verbatim_content`; this guard was added
after a Lecture 3 diagnostic revealed that summary-shaped gold penalizes correct
extra text.

### Frozen one-shot result

All 16 stable pages from the annotated CS231n 2025 Lecture 2 interval were used.
The reference SHA-256 was frozen before OCR inference:

```text
aecb3a7aa7c4816b6d692a3529123df509fb6c3c7eb7b34f91fd59f26d9c6eb8
```

| Reader | CER | WER | Exact pages | Term recall | Mean reader latency |
| --- | ---: | ---: | ---: | ---: | ---: |
| Manual gold | 0.000 | 0.000 | 16/16 | 1.000 | 0.000009 s |
| Native PDF | 0.272 | 0.316 | 7/16 | 0.912 | 0.000207 s |
| RapidOCR | 0.399 | 0.489 | 10/16 | 0.971 | 2.786 s |

RapidOCR's errors are concentrated in multi-column ordering, dense numerical
illustrations, attribution-filter bypasses, stylized text, and image-source
noise. Native extraction is much faster and has lower aggregate edit error, but
misses rasterized code/labels and can expose a later PDF build state than the
video frame.

Decision: Gate 1 passes. Keep native text and OCR as complementary evidence with
separate provenance. Do not call native text an automatic oracle until page and
build-state alignment is established.

### Next gate: shared reader infrastructure

Before training the handwritten CNN:

1. create a line-crop dataset contract with lecture-level train/validation/test
   splits;
2. render clean source lines and generate versioned video-style corruptions;
3. freeze one character vocabulary, tokenizer, CTC loss, and greedy decoder;
4. add tensor-shape, gradient, checkpoint, and 32-example overfit tests;
5. train the CNN encoder only after those tests pass;
6. reuse exactly the same detector, tokenizer, decoder, data split, and metrics
   for the later ViT encoder.

The full methodology and error analysis are in
[`Multimodal page reading study.md`](Multimodal%20page%20reading%20study.md).
The frozen gold copy and compact result are
[`experiments/assignment_2_page_reading_references.jsonl`](experiments/assignment_2_page_reading_references.jsonl)
and
[`experiments/assignment_2_page_reading_results.json`](experiments/assignment_2_page_reading_results.json).
