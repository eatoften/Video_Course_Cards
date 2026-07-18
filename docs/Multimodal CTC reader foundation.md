# Multimodal CTC Reader Foundation

Last updated: 2026-07-18

## 1. Goal

This milestone builds the shared text-recognition infrastructure that both the
handwritten CNN and handwritten ViT readers must use. It deliberately stops
before claiming either encoder is good at unseen OCR.

The completed path is:

```text
stable page image + manually verified page text
-> RapidOCR text polygons
-> exact gold-line matching
-> provenance-preserving RGB line crops
-> shared grayscale/resize/pad transform
-> frozen character tokenizer
-> [batch, time, vocabulary] logits
-> PyTorch CTC loss adapter
-> greedy CTC decoder
```

## 2. Why CTC

A line image has a spatial width, while its label has a shorter character
length. The dataset does not say which horizontal pixel corresponds to each
character. CTC trains over all monotonic alignments that can collapse to the
target string, so character bounding boxes are unnecessary.

The model-facing contract uses `[B, T, V]` logits:

- `B`: batch size;
- `T`: horizontal feature-sequence length;
- `V`: vocabulary size including the CTC blank.

PyTorch `CTCLoss` expects `[T, B, V]` log probabilities. The shared adapter
validates lengths and performs this transpose in exactly one place.

### 2.1 Blank and repeated characters

The shared IDs are fixed:

```text
0 = <blank>
1 = <unk>
2... = ordinary characters
```

Greedy decoding first collapses adjacent duplicate IDs and then removes blank.
For example:

```text
c c blank a blank t t -> cat
```

Repeated target characters need a separating blank. `book` therefore needs at
least five time steps for four characters: `b o blank o k`. The loss adapter
calculates this minimum and fails before training if an image is too narrow.

## 3. Data Contract

Each `LineCropSample` stores:

- lecture, page event, page number, and stable-frame timestamp;
- source image path and SHA-256;
- crop path and SHA-256;
- detector reader, model version, preprocessing version, and cache key;
- final bounding box and source block order;
- raw and normalized one-line label;
- label source (`manual_exact_match`, `human_corrected`, or
  `synthetic_render`).

The sample ID hashes the source image, page-content cache key, page/block ID,
label, cropper version, and final bounding box. A detector or polygon change
therefore invalidates the sample identity instead of silently reusing an old
crop.

Raw crops are stored as RGB PNGs. Model transforms convert them to grayscale,
resize to a shared height without changing aspect ratio, invert them so ink is
positive, and right-pad with zeros. Images that exceed the configured maximum
width fail explicitly rather than being horizontally distorted.

## 4. Label Construction

The first dataset uses the frozen CS231n 2025 Lecture 2 page-reading benchmark.
For each RapidOCR polygon, its normalized recognized text must exactly equal an
unused line in the manual `verbatim_content` gold. The crop receives the manual
gold line as its label.

This produced 62 line crops:

```text
reference SHA-256:
aecb3a7aa7c4816b6d692a3529123df509fb6c3c7eb7b34f91fd59f26d9c6eb8

RapidOCR PageContent SHA-256:
4ed054f7b0aa1c506a854d335b4b4f9da13957cbaa308553fc6daf544a9ecf4c

line-crop dataset SHA-256:
fdcdc086a9ff53931c9ccfa4b0f2d47fd81085ba0c99385633a97b942914a368
```

This selection is intentionally high precision but biased toward lines that
RapidOCR already detected and recognized correctly. It is suitable for a
pipeline overfit test, not for reporting OCR generalization.

## 5. Character Tokenizer

`CharacterTokenizer.fit()` normalizes each training line with Unicode NFKC and
collapses whitespace, then sorts the observed characters into a deterministic
case-sensitive vocabulary. The vocabulary JSON stores its own SHA-256 and is
rejected if characters or contract settings change.

The final CNN/ViT vocabulary must be fit on the formal training lectures only.
Validation/test-only characters map to `<unk>`. The vocabulary produced by the
32-sample overfit run belongs only to that diagnostic and is not the final
research vocabulary.

## 6. Lecture-Level Split

The splitter shuffles sorted lecture IDs with a recorded seed, then assigns
whole lectures to train, validation, and test. It refuses to operate with fewer
than three lectures and validates that the lecture sets are disjoint. Samples
are partitioned only after the lecture manifest is fixed.

No formal real-data split is emitted yet. The current 62 labels all belong to
one lecture, so creating train/validation/test rows from them would be leakage.
The CLI is ready, but it should be run only after line labels exist for at least
three independent lectures.

## 7. The 32-Sample Overfit Gate

### 7.1 Probe architecture

The probe is intentionally small and is not the final CNN experiment:

```text
grayscale line image
-> Conv(1, 32) + GroupNorm + GELU + 2x pooling
-> Conv(32, 64) + GroupNorm + GELU + 2x pooling
-> Conv(64, 96) + GroupNorm + GELU
-> mean over feature height
-> shared linear vocabulary head
-> [B, T, V] logits
```

Configuration:

| Setting | Value |
| --- | ---: |
| Samples | 32 |
| Maximum label length | 20 characters |
| Input height | 24 pixels |
| Maximum input width | 320 pixels |
| Optimizer | Adam |
| Learning rate | 0.003 |
| Seed | 17 |
| Maximum steps | 1200 |
| Decode interval | 20 steps |
| Device | CPU |

### 7.2 Final v2 result

| Metric | Result |
| --- | ---: |
| Parameters | 79,927 |
| Steps to pass | 280 |
| Initial loss | 7.4179 |
| Final loss | 0.0555 |
| Exact lines | 32/32 |
| Exact-match rate | 1.000 |
| Character error rate | 0.000 |
| Training time | 63.7 s |

The gate passed. This establishes that image loading, padding, output-length
calculation, vocabulary IDs, CTC target packing, loss, gradients, and greedy
decoding agree with one another.

It does **not** establish validation performance, compare CNN with ViT, or show
that 79,927 parameters are sufficient for general OCR. Memorizing the tiny set
is the expected behavior of this diagnostic.

## 8. Reproduction

Create the grounded line crops from `backend`:

```powershell
uv run python -m multimodal_lab.run_line_crop_dataset `
  --references data\multimodal_lab\cs231n_2025_lecture_02\page_reading_references.jsonl `
  --page-contents data\multimodal_lab\cs231n_2025_lecture_02\page_reading_comparison_v2\rapidocr_page_contents.jsonl `
  --image-root . `
  --output-dir data\multimodal_lab\cs231n_2025_lecture_02\line_crop_dataset_v2 `
  --expected-references-sha256 aecb3a7aa7c4816b6d692a3529123df509fb6c3c7eb7b34f91fd59f26d9c6eb8 `
  --expected-page-contents-sha256 4ed054f7b0aa1c506a854d335b4b4f9da13957cbaa308553fc6daf544a9ecf4c
```

Run the 32-sample gate:

```powershell
uv run python -m multimodal_lab.run_ctc_overfit `
  --samples data\multimodal_lab\cs231n_2025_lecture_02\line_crop_dataset_v2\line_crop_samples.jsonl `
  --image-root . `
  --output-dir data\multimodal_lab\cs231n_2025_lecture_02\ctc_overfit_v2 `
  --expected-dataset-sha256 fdcdc086a9ff53931c9ccfa4b0f2d47fd81085ba0c99385633a97b942914a368
```

Once at least three lecture datasets exist, create the formal split:

```powershell
uv run python -m multimodal_lab.run_lecture_split `
  --samples path\to\multi_lecture_line_crop_samples.jsonl `
  --output path\to\lecture_split.json `
  --expected-dataset-sha256 <frozen-dataset-hash> `
  --seed 42
```

## 9. Next Assignment

1. Correct or render line labels for at least three lectures.
2. Freeze the lecture split and training-only vocabulary.
3. Add versioned synthetic video corruptions to clean source lines.
4. Turn the probe into the declared handwritten CNN configuration.
5. Add checkpoint parity, deterministic evaluation, and validation CER/WER.
6. Train and evaluate CNN without looking at the test lecture.
7. Reuse the exact dataset, tokenizer, CTC head, and split for ViT.

## 10. Sources

- [PyTorch CTCLoss documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.modules.loss.CTCLoss.html)
- [Graves et al., Connectionist Temporal Classification](https://www.cs.utoronto.ca/~graves/icml_2006.pdf)
- [CRNN: End-to-End Trainable Neural Network for Image-Based Sequence Recognition](https://arxiv.org/abs/1507.05717)
