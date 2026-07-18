# Multimodal Lab

This package contains research code for the multimodal upgrade. It is isolated
from the FastAPI product code in `backend/app`: experiments may reuse stable
data contracts, but the application must never depend on a training script.

The lab follows a small-repository version of the organization used by public
research codebases such as VGGT, DINOv2, and VAR: explicit entry points,
versioned configurations, separate training and evaluation, and immutable run
artifacts. It deliberately does not introduce a general experiment platform.

## Current map

| Area | Modules | Responsibility |
| --- | --- | --- |
| Contracts | `schemas.py` | Serialized annotation and result schemas |
| Data | `annotation_io.py`, `line_crop_dataset.py`, `formal_reader_dataset.py`, `run_formal_reader_dataset.py` | Build, review, split, audit, and transform samples |
| Shared text path | `ctc_text.py` | Character vocabulary, CTC loss adapter, decoder |
| Transition study | `transition_baseline.py`, `evaluate_transition_baseline.py`, `run_transition_*.py` | Page-change and stable-frame baselines |
| Reader study | `page_reading.py`, `page_reading_evaluation.py`, `run_page_reading_comparison.py` | Native-text and OCR baselines |
| Diagnostic model | `ctc_overfit.py`, `run_ctc_overfit.py` | Pipeline overfit gate, not a benchmark |
| Experiment contract | `experiment_protocol.py`, `experiment_tracking.py` | Leakage audit and local run provenance |
| Reader configuration | `reader_config.py`, `configs/reader_cnn_v1.json` | Frozen data hashes and declared first-CNN hyperparameters; no model layers |
| Shared reader engine | `training/reader_data.py`, `reader_protocol.py`, `reader_trainer.py`, `reader_evaluator.py` | Model-independent loading, output contract, optimization, and metrics |
| Metrics | `metrics.py` | Reusable metric implementations |

All command-line entry points are named `run_*.py`. A module without that
prefix must be import-safe and must not start work at import time.

## Experiment lifecycle

```text
checked-in config
  -> dataset/split audit
  -> training on train lectures
  -> model selection on validation lectures
  -> one independent test evaluation
  -> compact result committed under docs/experiments
```

A formal reader result is invalid unless all of the following hold:

1. Lectures, source pages, and crops are disjoint across splits.
2. The character vocabulary is fitted on the training split only.
3. Validation and test labels are human-corrected or independently
   source-aligned. Source-aligned labels require a separate human spot-check
   before a publication claim.
4. CNN and ViT use the same split, crops, tokenizer, decoder, metric code, and
   model-selection rule.
5. The test set is not used by a training or checkpoint-selection command.

## Current pre-model boundary

The shared data and experiment engine now exists, but no formal reader model
has been implemented. The current boundary is intentional:

```text
multimodal_lab/
  configs/
    reader_cnn_v1.json       # declaration only
  training/
    reader_data.py
    reader_protocol.py
    reader_trainer.py
    reader_evaluator.py
  models/                    # intentionally absent
```

The next handwritten assignment may add `models/cnn_ctc.py`. The later ViT
must use the same data bundle, tokenizer, output contract, trainer, evaluator,
split, and checkpoint-selection rule. Model-specific branches in the training
loop are a design failure: the controlled experiment changes the encoder, not
the protocol.

Create another package only when it has at least two concrete implementations
or removes real duplication. Do not add registry, plugin, dependency-injection,
or workflow-engine abstractions for hypothetical future experiments.

## Run artifacts

Generated datasets, checkpoints, predictions, and full logs belong under:

```text
backend/data/multimodal_lab/<study>/<run_id>/
```

`backend/data/` is ignored by Git. Every tracked run directory contains a
frozen spec, a manifest with Git/runtime fingerprints, metrics, predictions,
and artifact hashes. Files in a completed run are treated as immutable.

Only compact, reviewed evidence belongs in Git:

```text
docs/experiments/<assignment>_results.json
docs/<assignment study>.md
```

Never commit raw frames, line crops, optimizer state, model checkpoints, or an
entire local run directory.

## Dependency rules

- `backend/app` must not import `multimodal_lab`.
- Model `forward` methods perform tensor computation only; no path, database,
  logging, or serialization work.
- Dataset modules do not know about model classes.
- Trainers do not calculate private model-specific metrics.
- Evaluators never update parameters or select checkpoints on the test split.
- FastAPI, SQLite, Ollama, and Tauri are outside the controlled reader study.
- Add Hydra, MLflow, DVC, or another service only after a measured need appears;
  local JSON configs and manifests are sufficient for the current scale.

## Before merging an experiment change

1. Run the focused unit tests.
2. Run the dataset audit for any formal split.
3. Confirm generated artifacts remain outside Git.
4. Record the exact command, dataset/split hashes, seed, and result.
5. State what the result does **not** prove.
