# Multimodal Page Reading Baseline Study

Last updated: 2026-07-17

## Abstract

This study tests whether stable frames selected from lecture video can be read
locally and converted into the repository's common `PageContent` contract. It
compares three paths on 16 manually reviewed CS231n pages: a manual gold
reference, native PDF text extraction, and a training-free RapidOCR baseline.

RapidOCR exactly recovered 10 of 16 pages and 97.1% of the registered technical
terms. Native PDF extraction had lower aggregate CER/WER and negligible
per-page latency, but it missed labels and code embedded in images and could
expose a different animation/build state than the video frame. The methods are
therefore complementary: native text is preferred when a correctly aligned
deck is available, while OCR is required for deck-free and visual-only content.

## 1. Research Questions

**RQ1.** Can a small, fully local, training-free OCR system recover useful
instructional text from stable lecture-video frames?

**RQ2.** How does video-frame OCR differ from native PDF text extraction in
accuracy, technical-term coverage, failure mode, and runtime?

**RQ3.** Are the remaining failures attributable to text detection, character
recognition, reading order, irrelevant-text inclusion, or source/video state
mismatch?

The preregistered expectations were:

- RapidOCR should recover most large instructional text but struggle with dense
  layouts, stylized image text, and small attribution text.
- Native extraction should be faster and have lower edit error when its text
  layer is complete, but should miss rasterized content.
- One scalar metric should not select an overall winner because the two readers
  observe different evidence.

## 2. Benchmark Construction

### 2.1 Calibration failure caught before the held-out run

An initial diagnostic used CS231n 2026 Lecture 3. Its reference `gold_text`
contained concise semantic summaries rather than literal visible page content.
Both native extraction and OCR were penalized for correctly returning additional
text, so those CER/WER values are invalid and are not reported as model results.

This exposed a benchmark bug rather than a model bug. The evaluator now rejects
any `StablePageReference` whose `gold_text_scope` is not
`verbatim_content`. Lecture 3 remained the calibration source for generic
footer, watermark, minimum-height, and attribution filtering.

### 2.2 Held-out pages

The one-shot OCR run used all 16 stable references in the first 900 seconds of
CS231n 2025 Lecture 2: page 1 and pages 5-19. No page was removed after seeing
predictions.

The frozen reference SHA-256 is:

```text
aecb3a7aa7c4816b6d692a3529123df509fb6c3c7eb7b34f91fd59f26d9c6eb8
```

The gold scope is the exact visible instructional text in a canonical reading
order. It includes titles, body text, code, formulas, and semantic labels in
figures. It excludes course-template chrome, the Stanford watermark, image
licenses/citations, and illustrative pixel-number matrices. Technical terms
must occur literally in the gold text; the gold reader's term-recall sanity
check must equal 1.0.

The tracked gold copy is
[`experiments/assignment_2_page_reading_references.jsonl`](experiments/assignment_2_page_reading_references.jsonl).
Stable images and full model outputs remain under the ignored local
`backend/data/multimodal_lab/` tree.

### 2.3 Leakage statement

Gold text was completed and hashed before any Lecture 2 RapidOCR output was
generated. OCR preprocessing was fixed from the Lecture 3 diagnostic. Lecture
2 was then run once and retained as-is.

The native PDF cleaner was developed while inspecting parser boilerplate from
the Lecture 2 deck. Its score is therefore a calibrated parser baseline, not a
strict held-out model estimate. This distinction does not affect the one-shot
RapidOCR claim.

## 3. Methods

All readers implement one `PageReader` protocol and return the same
`PageContent` schema. Every output retains the lecture/page ID, stable-frame
timestamp, image path and SHA-256, reader/model/preprocessing versions, ordered
blocks, raw and normalized text, confidence, latency, and a cache key.

### 3.1 Manual gold reference

`GoldReferencePageReader` copies the frozen human transcription into the common
schema. It is an evaluation ceiling and contract sanity check, not a production
reader.

### 3.2 Native source baseline

`NativeSourcePageReader` consumes the source units already parsed by the
application's PDF ingestion layer with pypdf 6.14.2. A deterministic cleaner
removes date-bearing lecture headers, standalone page numbers, URLs, image
licenses, and public-domain notices.

This baseline is only valid after page alignment. The experiment supplies the
known PDF page number; it does not measure page retrieval, Recall@K, or MRR.

### 3.3 Local OCR baseline

`RapidOcrPageReader` uses RapidOCR 3.9.1 and ONNX Runtime 1.27.0 with local
PP-OCRv6 small detection/recognition models and the bundled orientation model:

```text
PP-OCRv6_det_small.onnx
ch_ppocr_mobile_v2.0_cls_mobile.onnx
PP-OCRv6_rec_small.onnx
```

The content filter removes blocks below 1.5% of image height, the bottom 10%
course footer, the bottom-right Stanford watermark, and explicit attribution
patterns. It keeps raw recognized symbols for diagnosis and only normalizes a
separate scoring field.

The run used ONNX Runtime on CPU. Reported per-page latency excludes one-time
model construction. Native latency similarly excludes initial PDF parsing, so
the timing table is reader-call latency rather than full application startup.

## 4. Metrics

- **CER:** character Levenshtein edits divided by reference characters.
- **WER:** word Levenshtein edits divided by reference words.
- **Exact page match:** normalized prediction equals the full normalized gold.
- **Technical-term recall:** registered gold terms found as exact phrases.
- **Abstention count:** pages for which a reader returned no result.
- **Latency:** mean and nearest-rank p95 reader-call time.

Corpus CER/WER sum edits and denominators across pages, so long pages receive
appropriate weight. A page-level CER or WER can exceed 1.0 when a prediction
adds more irrelevant tokens than the reference contains.

Quantitative block-layout recovery is not claimed here because the benchmark
has canonical text lines but no independently annotated bounding boxes or
region hierarchy. Layout is reported in the error taxonomy. Block-level gold
must be frozen before a future numeric structure score is introduced.

## 5. Results

| Reader | CER | WER | Exact pages | Term recall | Mean latency | p95 latency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Manual gold | 0.000 | 0.000 | 16/16 (100.0%) | 100.0% | 0.000009 s | 0.000069 s |
| Native PDF | 0.272 | 0.316 | 7/16 (43.8%) | 91.2% | 0.000207 s | 0.002286 s |
| RapidOCR | 0.399 | 0.489 | 10/16 (62.5%) | 97.1% | 2.786 s | 3.937 s |

RapidOCR has higher aggregate edit error but more exactly correct pages and
higher technical-term recall. Native extraction wins CER/WER because it is
excellent on ordinary PDF text and does not generate visual noise. The result
does not support replacing one with the other.

## 6. Error Analysis

### 6.1 RapidOCR

| Page | Primary error | Observation |
| ---: | --- | --- |
| 5 | Layout ordering | Most content was recognized, but the three columns were interleaved row-wise; `AI` also became `Al`. |
| 8 | Irrelevant dense text | Small numbers from the illustrative pixel matrix passed the filter and dominated edit distance. |
| 13 | Recognition-driven filter bypass | `This image...` became `Ihis image...`, defeating the exact attribution marker. |
| 15 | Figure-label recognition | `Context` became `Coxt`, and the image-source line generated noise. |
| 16 | Stylized text detection miss | `IMAGENET` inside a photo mosaic was not detected. |
| 18 | Irrelevant-text inclusion | The tiny Canny citation was recognized even though citations are outside the target scope. |

The other ten pages were exact after normalization. This concentration explains
why exact-page accuracy and term recall look strong while corpus CER/WER remain
moderate.

### 6.2 Native PDF extraction

- Page 1 retains `Lecture 2:` because it is not a date-bearing template line.
- Page 5 exposes PDF object order rather than visual column reading order.
- Page 6 contains corrupted bullet glyphs.
- Page 7 places the output label before the explanatory label set.
- Pages 15-17 miss figure labels, the mosaic word, and rasterized code.
- Page 18 keeps a citation and contains text-encoding corruption.
- Page 19 exposes later PDF build content absent from the selected video frame.

Page 19 is especially important: a source deck can be semantically correct but
temporally wrong. Native text only becomes trustworthy after page and build-state
alignment.

## 7. Decision

The event-driven page-reading gate passes. Stable frames are readable, local OCR
works without a remote API, and every non-exact output can be assigned to a
specific failure family.

The production direction is a provenance-preserving hybrid:

1. Align a supplied deck to video when available.
2. Prefer native text for ordinary text-layer content.
3. Use OCR for deck-free lectures, annotations, code screenshots, logos, and
   other visual-only text.
4. Keep both observations when they disagree rather than silently overwriting
   one source.
5. Let downstream card generation cite the exact modality and timestamp.

The experiment does not yet justify training a full page-to-text CNN or ViT on
16 pages. The next assignment should first build shared line crops, a character
vocabulary, a CTC decoder, lecture-level splits, synthetic video corruption,
and a 32-sample overfit test. Only the encoder may differ in the controlled CNN
versus ViT comparison.

## 8. Reproduction

From `backend`:

```powershell
$reference = 'data\multimodal_lab\cs231n_2025_lecture_02\page_reading_references.jsonl'
$hash = (Get-FileHash -Algorithm SHA256 -Path $reference).Hash.ToLower()

uv run python -m multimodal_lab.run_page_reading_comparison `
  --references $reference `
  --image-root . `
  --output-dir data\multimodal_lab\cs231n_2025_lecture_02\page_reading_comparison_v2 `
  --source-asset-id 4138d43815a44e6a8dff099c748d1a17 `
  --expected-references-sha256 $hash `
  --held-out
```

The hash guard fails before image loading or model inference if the reference
file changes. The tracked compact result is
[`experiments/assignment_2_page_reading_results.json`](experiments/assignment_2_page_reading_results.json).

## 9. Threats to Validity

- Sixteen pages from one course are a case study, not broad OCR generalization.
- Many pages are large-title slides; the exact-match rate should not be read
  without the complex-page error table.
- Gold scope intentionally excludes attributions and dense illustrative
  numbers; another product may choose a different content contract.
- The official PDF and recorded slide template/build state are not pixel-identical.
- Reader-call latency excludes model initialization and upstream frame extraction.
- Page retrieval and native/video temporal alignment are not evaluated here.

## 10. Sources

- [RapidOCR documentation](https://rapidai.github.io/RapidOCRDocs/)
- [RapidOCR repository](https://github.com/RapidAI/RapidOCR)
- [CRNN: End-to-End Trainable Neural Network for Image-Based Sequence Recognition](https://arxiv.org/abs/1507.05717)
- [ViTSTR: Vision Transformer for Scene Text Recognition](https://arxiv.org/abs/2105.08582)
- [TrOCR: Transformer-Based Optical Character Recognition](https://arxiv.org/abs/2109.10282)
