# Multimodal Formal Reader Pre-Model Gate

Last updated: 2026-07-18

## 1. Research Question

Before comparing a handwritten CNN with a handwritten ViT, can the project
establish a leakage-aware dataset and one model-independent experiment engine?

This stage tests the experimental boundary, not recognition accuracy. It ends
before either formal encoder is implemented.

## 2. Preregistered Stopping Boundary

This stage may create:

- a third independent lecture and immutable input provenance;
- train, validation, and test line-crop components;
- a frozen lecture-level split and train-only vocabulary;
- a formal dataset audit;
- model configuration schemas;
- shared data loading, model-output validation, training, and evaluation code;
- tests that use toy models defined only inside the test suite.

This stage must not create:

- `models/cnn_ctc.py`;
- `models/vit_ctc.py`;
- a production `Conv2d`, transformer block, or attention layer;
- a CNN/ViT checkpoint or benchmark score.

The stopping boundary was respected. There is no production model directory.

## 3. Third Independent Lecture

The held-out lecture is CS231n 2025 Lecture 4, Neural Networks and
Backpropagation. Its identity is supported by the
[official 2025 course schedule](https://cs231n.stanford.edu/2025/schedule.html)
and the
[official Lecture 4 deck](https://cs231n.stanford.edu/slides/2025/lecture_4.pdf).

| Field | Frozen value |
| --- | --- |
| Lecture ID | `cs231n-2025-lecture-04` |
| Product job ID | `574e612a5cf846f69b58e2f508289202` |
| Evaluation interval | `0-900 s` |
| Stable page references | 14 |
| Video SHA-256 | `38e890aca7d6f59725cdd41db33888b71a53863fe1ed937efc27bd2e3de77a12` |
| Transcript SHA-256 | `d377ef8711cb86a135e70cb113114c04b0acd378e0494f6c8f7c984c651fdad0` |
| Deck SHA-256 | `906b09877d36e121705ee89576aff8291ba9d70ea15713753c9a734894ca20b8` |
| Reference SHA-256 | `242a348ad63dbb217495fd918c8996b4b10f0b61fd23d29f2803bb93a75f8530` |

The frozen spatial-state transition profile emitted zero events on this
lecture because the opening camera/layout sequence differs from Lectures 2 and
3. No threshold was tuned on the test lecture. A fixed 30-second overview grid
was used to find candidate regions, followed by stable-frame and official-deck
verification that did not use OCR correctness.

This detector failure is retained as evidence of domain shift. Hiding it by
retuning on the test lecture would weaken both the transition and reader
experiments.

## 4. Label Construction

The formal dataset contains two distinct label mechanisms.

### 4.1 Training labels

Lecture 2 official-deck text is rendered into four deterministic visual
variants per eligible line. These 268 `synthetic_render` samples are used only
for training. Their purpose is to avoid fitting on OCR-selected labels while
providing enough examples for the first handwritten model exercise.

### 4.2 Evaluation labels

Lectures 3 and 4 use `source_aligned` labels. RapidOCR polygons provide crop
geometry, while the final text is aligned against the independently available
official deck and stable video frame. Every polygon receives an explicit
`include`, `exclude`, or pending review record; a pending record blocks dataset
construction.

These records were source-aligned by Codex, not independently corrected by a
human annotator. They are therefore not mislabeled as `human_corrected`. The
audit permits them for local development but emits a warning requiring an
independent human spot-check before any publication claim.

## 5. Frozen Dataset and Split

| Split | Lecture | Label source | Samples |
| --- | --- | --- | ---: |
| Train | CS231n 2025 Lecture 2 | synthetic render | 268 |
| Validation | CS231n 2026 Lecture 3 | source aligned | 96 |
| Test | CS231n 2025 Lecture 4 | source aligned | 67 |
| Total | 3 lectures | mixed | 431 |

Frozen identities:

```text
dataset SHA-256:
e82b00706f07343412f9e7dce40853cd30e4aa06999da5c0980ae7b86f56195d

split SHA-256:
6b0445633d4aa445f6153d592176366e497533f058e99c318e2dd05ee718290a

audit SHA-256:
317740210bc401a4156d3d1c7340e28549a1568bcc23334b5d9dd0d884288143

train-only vocabulary SHA-256:
b0c091d14696d1b8b293c5b4aa6e6470cd593356a542655857d9b8ce678c3b76
```

The split seed is 1. Whole lectures, rather than lines or pages, are assigned
to splits.

## 6. Dataset Audit Result

The formal audit passed with no blocking problems:

- no sample ID duplication;
- no crop hash shared across splits;
- no source-page hash shared across splits;
- every dataset lecture appears in exactly one split;
- the tokenizer is fitted from training text only;
- validation and test use independent label sources;
- the split and audit both target the same dataset hash.

Warnings remain visible:

| Split | Unknown characters | Character rate |
| --- | ---: | ---: |
| Validation | 20 | 1.010% |
| Test | 24 | 1.370% |

The unknown characters are mapped to `<unk>`. They were not inspected and
injected into the training vocabulary because that would leak validation/test
information into the model contract.

## 7. Shared Experiment Engine

The model-independent path is now:

```text
checked-in experiment config
-> verify dataset/split/audit hashes
-> reconstruct and verify train-only tokenizer
-> lecture-partitioned DataLoaders
-> model(images, widths)
-> ReaderModelOutput(logits, input_lengths)
-> shared CTC loss and greedy decoder
-> validation CER/WER/exact match
-> validation-only checkpoint selection
-> one later test evaluation
```

`reader_data.py` refuses changed artifacts before loading images.
`reader_protocol.py` validates `[batch, time, vocabulary]` logits and one valid
input length per sample. `reader_trainer.py` provides AdamW, gradient clipping,
CUDA AMP, validation-based early stopping, and best-checkpoint persistence.
`reader_evaluator.py` refuses train/smoke splits and computes corpus CER, WER,
exact match, CTC loss, predictions, and unknown-reference counts.

The checked-in `reader_cnn_v1.json` freezes the data identities, seed,
optimization policy, selection rule, and intended CNN hyperparameters. It is a
configuration declaration, not a neural-network implementation. Its SHA-256
is:

```text
8b5e657363395cef8c96facdfb78b378ab23bcd4052e1eb78499779e8aa74960
```

## 8. Reproduction

From `backend`:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'

uv run python -m multimodal_lab.run_formal_reader_dataset audit `
  --dataset data\multimodal_lab\formal_reader_v1\line_crop_samples.jsonl `
  --split-output data\multimodal_lab\formal_reader_v1\lecture_split.json `
  --audit-output data\multimodal_lab\formal_reader_v1\dataset_audit.json `
  --seed 1

uv run pytest -q `
  tests/test_experiment_protocol.py `
  tests/test_reader_config.py `
  tests/test_reader_training.py `
  tests/test_formal_reader_dataset.py `
  tests/test_ctc_text.py
```

The audit command was rerun into temporary files and reproduced the frozen
split and audit hashes exactly. The focused test suite reports 23 passing
tests. The complete backend regression suite reports 235 passing tests with
one upstream Starlette `TestClient` deprecation warning.

## 9. Threats to Validity

1. Three lectures provide a valid isolation boundary, but only one lecture per
   split. Results will have high sampling uncertainty.
2. Training is synthetic while evaluation is video-derived, so domain shift is
   deliberately difficult and may dominate architecture differences.
3. Evaluation crop geometry starts from RapidOCR detections. Missed text lines
   are outside this first recognition benchmark, so it does not evaluate text
   detection recall.
4. Source-aligned labels have not received an independent human spot-check.
5. Local course assets are ignored by Git. The tracked hashes and commands
   preserve identity, but a new user must lawfully obtain the source lectures
   and decks.
6. The first config contains one hyperparameter proposal. It has not been
   selected by validation performance because no formal model exists yet.

## 10. Gate Decision

The pre-model gate passes for local development. The data split, audit,
tokenizer boundary, trainer, and evaluator are frozen before model code exists.

The next assignment belongs to the user: handwrite the first CNN so that its
`forward(images, widths)` method returns the shared `ReaderModelOutput`.
Training, test evaluation, and ViT implementation remain intentionally
unstarted.

## 11. Implementation Map

- Dataset/review builders: `backend/multimodal_lab/formal_reader_dataset.py`
- Reproducible dataset CLI: `backend/multimodal_lab/run_formal_reader_dataset.py`
- Split/audit protocol: `backend/multimodal_lab/experiment_protocol.py`
- Frozen config contract: `backend/multimodal_lab/reader_config.py`
- First declaration: `backend/multimodal_lab/configs/reader_cnn_v1.json`
- Shared data path: `backend/multimodal_lab/training/reader_data.py`
- Model output contract: `backend/multimodal_lab/training/reader_protocol.py`
- Shared trainer/evaluator: `backend/multimodal_lab/training/reader_trainer.py`
  and `reader_evaluator.py`
- Machine-readable result:
  `docs/experiments/assignment_4_pre_model_gate_results.json`
