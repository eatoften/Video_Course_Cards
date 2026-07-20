# Multimodal CNN Reader Study

Last updated: 2026-07-19

Status: complete; the held-out test was opened once after checkpoint freeze.

## 1. Research Question

Can a small handwritten convolutional encoder learn line-level text recognition
from the frozen synthetic training lecture and generalize to independently
source-aligned video crops from another lecture?

This CNN is the first trainable reader baseline. A later ViT must reuse the
same data, split, tokenizer, CTC head, trainer, evaluator, and metrics so that
the encoder is the controlled variable.

## 2. Frozen Inputs

The dataset and split remain unchanged from the pre-model gate:

| Split | Lecture | Samples | Label source |
| --- | --- | ---: | --- |
| Train | CS231n 2025 Lecture 2 | 268 | synthetic render |
| Validation | CS231n 2026 Lecture 3 | 96 | source aligned |
| Test | CS231n 2025 Lecture 4 | 67 | source aligned |

```text
dataset SHA-256:
e82b00706f07343412f9e7dce40853cd30e4aa06999da5c0980ae7b86f56195d

split SHA-256:
6b0445633d4aa445f6153d592176366e497533f058e99c318e2dd05ee718290a

audit SHA-256:
317740210bc401a4156d3d1c7340e28549a1568bcc23334b5d9dd0d884288143
```

## 3. Preflight Shape Audit

At target height 32, the transformed train widths range from 49 to 563 pixels;
validation widths range from 45 to 635. With horizontal stride 4, every train
and validation sample satisfies the CTC alignment lower bound. The smallest
available-minus-required time-step margins are 9 and 7 respectively.

Decision: retain stride 4. Stride 2 would increase CPU cost and sequence memory
without solving an observed feasibility problem; stride 8 would remove useful
horizontal resolution before a baseline exists.

## 4. Architecture Decisions

| Choice | Frozen value | Reason |
| --- | --- | --- |
| Encoder | three 3x3 convolution blocks | Establish a small local-pattern baseline before global attention |
| Channels | 32, 64, 128 | Enough capacity for the 32-sample gate while remaining cheap on CPU |
| Downsampling | 2x2 max-pool after blocks 1 and 2 | Produces exact stride 4 and preserves strong ink responses |
| Normalization | channel-wise LayerNorm at each pixel | Does not mix batch examples or aggregate right-side padding into statistics |
| Activation | GELU | Smooth activation already used by the diagnostic probe and suitable for later ViT parity |
| Height reduction | arithmetic mean | Converts the feature map to a horizontal sequence without a second sequence model |
| Sequence head | shared projection, GELU, dropout, classifier | Can be reused unchanged by the future ViT |
| Dropout | 0.1 | Mild regularization for a small training set |
| Initial blank bias | -1.0 | Reduces early all-blank collapse; the same head policy will apply to ViT |
| Recurrent layer | none | A BiLSTM would confound the CNN-versus-ViT encoder comparison |

The model-facing output remains
`ReaderModelOutput(logits[B,T,V], input_lengths[B])`. Width is the CTC time
axis; height is removed only after local visual features have been computed.

## 5. Runtime Decision

The current environment contains PyTorch `2.12.1+cpu`; CUDA is unavailable to
this virtual environment even though the machine has an NVIDIA GPU. The formal
dataset has only 268 training samples, so the first run will use CPU and record
that runtime. Installing a different PyTorch build now would change both the
model experiment and execution environment at once. CUDA packaging remains a
separate optimization task if measured runtime becomes prohibitive.

## 6. Leakage Boundary

Training code receives a `ReaderTrainingDataBundle` containing only train and
validation loaders. Test evaluation requires a separate
`ReaderTestDataBundle`. Constructing a training bundle cannot accidentally
iterate the test loader because that attribute does not exist.

The training command may select checkpoints only by validation CER, with WER
as tie-breaker. The test command will accept one already frozen checkpoint and
will not select or modify it.

## 7. Gates

1. Shape and output-length tests must pass.
2. Extra right padding must not change logits in valid time steps.
3. CTC loss and every trainable gradient must be finite.
4. The formal CNN must exactly memorize 32 deterministic train samples.
5. Full training may use only train and validation.
6. The best checkpoint hash and validation result must be frozen.
7. Test evaluation may run once, after Gate 6.

## 8. Decision Log

### 2026-07-19: Input and stride audit

The frozen data hashes were reverified. Train and validation widths were
measured after the real image transform, without evaluating test examples.
Stride 4 passed every CTC feasibility check and was retained.

### 2026-07-19: Normalization choice

BatchNorm and GroupNorm both aggregate across padded spatial positions.
Channel-wise LayerNorm normalizes channels independently at each pixel, so a
sample's representation is less sensitive to how much right padding another
sample introduces. This property will be tested directly.

### 2026-07-19: Convolution padding leakage

The first padding-invariance test failed: extending a 100-pixel line to a
128-pixel padded tensor changed 18 of 225 valid logits, with maximum absolute
difference about 0.296. Convolution had propagated boundary activations into
the padded region and a later convolution propagated them back. The encoder
therefore masks columns beyond each sample's current valid width after every
block. This keeps batched output invariant to unrelated right padding rather
than merely shortening the final CTC length.

### 2026-07-19: Formal model preflight

The implemented CNN has 118,307 trainable parameters. Real train and validation
batches produced finite initial CTC losses of about 10.816 and 10.825. The
shared shape, output-length, padding-invariance, loss, and gradient tests pass.

The overfit gate samples 32 distinct normalized labels from the training
lecture, then chooses one deterministic synthetic variant per label. This is
harder and more informative than counting four rendered variants of one label
as four independent memorization targets. Diagnostic optimization uses Adam
without weight decay because this gate tests representational and plumbing
capacity, not regularized generalization.

The gate passed on CPU at step 380: initial/final loss was 11.3583/0.0172,
exact match was 32/32, CER was 0, and model-loop time was 310.6 seconds.
This supports capacity and pipeline correctness only.

### 2026-07-19: Unknown-character scoring

The frozen training vocabulary does not contain 20 validation characters and
24 test characters. Human-readable decoding renders one unknown token as
`<unk>`, but string-level edit distance would incorrectly count that display
label as five characters. CER is therefore computed on collapsed character
token IDs, where unknown occupies one position. Reports retain raw reference,
display prediction, and replacement-character scoring views, and separately
count unknown reference characters.

### 2026-07-19: Checkpoint and command separation

The train command can construct only train and validation loaders. Its best
checkpoint binds model config, dataset, split, experiment-config, and
vocabulary identities. The held-out command constructs the test loader through
a separate API and requires an explicit expected checkpoint SHA-256 before it
runs. It cannot select a checkpoint or update weights.

Formal training prints one JSON progress record per epoch. This was added after
the overfit process continued correctly but appeared silent behind a short
shell timeout; observability is treated as experiment reliability rather than
as a model change.

### 2026-07-19: Formal train and validation selection

The formal run used seed 23, AdamW at learning rate 0.001, weight decay
0.0001, batch size 16, and a maximum of 60 epochs. It ran on CPU because the
installed PyTorch build has no CUDA support. Early stopping ended training at
epoch 45 after ten epochs without a better validation CER. Epoch 35 was frozen
as the selected checkpoint.

The early trajectory initially collapsed to blank-heavy CTC output. The first
validation exact matches appeared at epoch 22, after which CER fell quickly.
Training loss continued to decrease after epoch 35 while validation CER
oscillated, so the validation-only early-stop rule prevented choosing a later,
more overfit checkpoint.

PowerShell buffered the epoch stream in the outer process and briefly made the
run appear silent. The process had completed successfully before termination
was attempted. The completed manifest, duration, per-epoch history, and
checkpoint are authoritative; this was an observability issue, not a failed or
restarted model run.

### 2026-07-19: Held-out test opened once

The validation-selected checkpoint was hashed before test evaluation. The
test command required that exact hash, loaded only the test bundle, created no
optimizer, and performed no checkpoint selection. The test was evaluated once.
No architecture, tokenizer, optimization, or decoding choice was changed after
seeing its result.

## 9. Formal Results

### Experiment identities

| Stage | Run ID | Main artifact |
| --- | --- | --- |
| Capacity gate | `20260719T133927868332Z-c7939af3017a` | 32-line overfit report |
| Train/validation | `20260719T135506719419Z-efdfb3a17097` | frozen best checkpoint |
| Held-out test | `20260719T140046649436Z-8e9f600361b2` | immutable test report |

```text
final config SHA-256:
0402e050e2b9166fc92f36ab0255d65d96dbacea3bb1cd1a0a1adf0a466e2b60

frozen checkpoint SHA-256:
bc3e4ea93e5b5b8723fca416e41573230a27e179bb2358c3a4cd90c011572853
```

The model has 118,307 trainable parameters. The training run took 226.7
seconds on CPU; the isolated test forward pass took 1.15 seconds.

### Capacity and generalization

| Split/gate | Samples | Loss | CER | WER | Exact lines |
| --- | ---: | ---: | ---: | ---: | ---: |
| 32-line train overfit | 32 | 0.0172 | 0.0000 | not used | 32/32 |
| Validation, Lecture 3 | 96 | 1.1051 | 0.1757 | 0.5915 | 13/96 |
| Test, Lecture 4 | 67 | 1.2317 | 0.2723 | 0.8750 | 4/67 |

The gap between perfect memorization and held-out CER is evidence that the
pipeline and model have enough capacity, but synthetic-to-video transfer is the
dominant unresolved problem. It is not evidence that a larger encoder alone
will solve the task.

### Same-crop RapidOCR reference point

RapidOCR's already stored text for the exact 67 included test polygons was
rescored with the same case-sensitive character and word edit definitions. It
achieved CER 0.0417, WER 0.0909, and 62/67 exact lines. The handwritten CNN
therefore remains substantially behind a mature OCR system.

This comparison is deliberately labelled a reference point rather than a fair
end-to-end detector benchmark. Test polygons originated from RapidOCR
detections and were then aligned to source-slide text. Missed detections never
enter this 67-crop recognition set, which favors RapidOCR and says nothing
about page-level detection recall.

### Post-hoc error slices

These slices were computed only after the one test run and did not influence
model selection.

| Reference length | Lines | CER | WER | Exact rate |
| --- | ---: | ---: | ---: | ---: |
| Short, at most 10 characters | 13 | 0.2360 | 0.7143 | 0.3077 |
| Medium, 11-20 characters | 13 | 0.3614 | 0.9167 | 0.0000 |
| Long, over 20 characters | 41 | 0.2621 | 0.8785 | 0.0000 |

Thirteen test lines contain at least one train-vocabulary unknown character.
They reach CER 0.4384 versus 0.2097 on the other 54 lines. This is descriptive,
not causal: the unknown subset also contains long equations, code, digits, and
punctuation. Representative failures include `SGD`, numeric schedules,
`# Vanilla Minibatch Gradient Descent`, and long Python-like code lines. The
model also sometimes emits repeated punctuation on wide code crops.

## 10. Interpretation and Next Decision

The first formal baseline is valid and useful:

1. The CTC data path passes a distinct-label memorization gate.
2. A small local CNN generalizes across lectures, but not well enough to
   replace RapidOCR.
3. The test degradation and error slices point first to domain shift, training
   coverage, and sequence/content complexity rather than an unverified need
   for a larger architecture.
4. CNN v1 is frozen. Any future augmentation, vocabulary, decoder, or crop
   change is a new experiment version and must not retroactively replace this
   result.

The next controlled assignment may implement a ViT encoder against the same
data, shared CTC head, selection rule, and one new predeclared test policy. A
responsible comparison should first add more training/validation lectures or
cross-validation; repeatedly consulting this single test lecture would turn it
into validation data.

## 11. Reproduction

From `backend`:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'

uv run python -m multimodal_lab.run_reader_overfit `
  --config multimodal_lab\configs\reader_cnn_v1.json `
  --output-dir data\multimodal_lab\experiment_runs `
  --device cpu

uv run python -m multimodal_lab.run_train_reader `
  --config multimodal_lab\configs\reader_cnn_v1.json `
  --output-dir data\multimodal_lab\experiment_runs `
  --device cpu

uv run python -m multimodal_lab.run_evaluate_reader `
  --config multimodal_lab\configs\reader_cnn_v1.json `
  --checkpoint data\multimodal_lab\experiment_runs\20260719T135506719419Z-efdfb3a17097\best_reader_checkpoint.pt `
  --expected-checkpoint-sha256 bc3e4ea93e5b5b8723fca416e41573230a27e179bb2358c3a4cd90c011572853 `
  --output-dir data\multimodal_lab\experiment_runs `
  --device cpu
```

The final command is recorded for provenance, not as permission to repeatedly
inspect the same held-out lecture. Full local artifacts remain ignored under
`backend/data`; the reviewed compact result is
`docs/experiments/assignment_4_cnn_reader_results.json`.

## 12. Threats to Validity

1. There is only one lecture in each split, so architecture conclusions have
   high sampling uncertainty.
2. Train images are synthetic while validation and test images are video
   crops. Encoder and data-domain effects are not separable in this result.
3. Source-aligned validation/test labels still need an independent human
   spot-check before a publication claim.
4. The crop set evaluates recognition after RapidOCR detection, not complete
   page reading.
5. The run used a dirty worktree based on commit
   `26421f1055c6ada4df7c6853b9ad8b366b13306d`; exact hashes for model, data,
   trainer, evaluator, tokenizer, config, and artifacts are stored in the run
   manifests and code fingerprint.
6. CPU was adequate for this small baseline, but runtime is not a hardware-
   normalized comparison against other systems.
