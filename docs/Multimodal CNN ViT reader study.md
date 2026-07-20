# Multimodal CNN-ViT Reader Study

Last updated: 2026-07-20

Status: Assignments 5.0-5.4 complete. The one-time Lecture 5 reader comparison
is closed; the downstream card cascade is reported separately as exploratory.

## 1. Research Question

Under the same line-crop data, tokenizer, CTC projection head, optimizer,
checkpoint-selection rule, and compute scale, does a handwritten ViT encoder
improve lecture-slide OCR over the frozen small CNN baseline?

The study also asks a product-facing question: do OCR differences survive the
rest of the pipeline and change the quality of grounded knowledge cards?

## 2. Preregistered Comparison

The controlled variable is the visual encoder:

```text
same image and label
-> same deterministic augmentation
-> CNN v2 or ViT v1
-> same CTC projection head
-> same greedy decoder
-> same line metrics
-> same card-generation protocol
```

CNN v1 is a historical baseline and is not modified. CNN v2 keeps its
architecture but receives the Assignment 5 data and augmentation protocol.
ViT v1 must stay close to CNN's parameter count, so the experiment does not
quietly become a comparison between model size classes.

## 3. Leakage Boundary

| Split | Lectures | Role |
| --- | --- | --- |
| Train | CS231n L1, L2, L3 | Fit weights and tokenizer |
| Validation | CS231n L4 | Select checkpoints and freeze downstream protocol |
| Test | CS231n L5 | Open once after both checkpoints are frozen |

Lecture 4 was evaluated by CNN v1 and therefore cannot remain a new test set.
Lecture 5 had not been used by a reader model when this protocol was frozen.
Its labels may be audited for integrity and vocabulary coverage, but no CNN or
ViT inference is allowed before both model-selection decisions are final.

## 4. Assignment 5.0: Frozen Protocol

CNN v1 remains frozen at commit `4b59b2c`. Its model has 118,307 trainable
parameters and its one-time historical result is retained in
`Multimodal CNN reader study.md`.

The new dataset contains seven independently hashed components:

| Split | Component | Samples | Label source |
| --- | --- | ---: | --- |
| Train | Lecture 1 video crops | 95 | source aligned |
| Train | Lecture 1 deck renders | 328 | synthetic |
| Train | Lecture 2 deck renders | 296 | synthetic |
| Train | Lecture 3 video crops | 96 | source aligned |
| Train | Lecture 3 deck renders | 344 | synthetic |
| Validation | Lecture 4 video crops | 67 | source aligned |
| Test | Lecture 5 video crops | 176 | source aligned |

Frozen identities:

```text
dataset SHA-256:
19c572ca531b6afb7e5f872e28be024a393d7b524824fa768b33104698bfdbb9

split SHA-256:
b11b6a20eb419b57ea69930d0341f2c0dbc9db2d9fa1f8012d8652193d8b9e49

dataset audit SHA-256:
9301a6698e4063ea654e753a8105efc0aab729f6d06cf1df14451305d71c070e
```

The resulting split is 1,159 train, 67 validation, and 176 test lines. No
crop hash or source-page hash crosses a split.

### Data construction decisions

1. Official slide text is used for synthetic labels because it is independent
   of OCR correctness.
2. Real crops are source aligned and each detected polygon receives an
   explicit include/exclude decision.
3. Lecture 2's old OCR-exact-match real crops are excluded from Assignment 5.
   Selecting only crops an OCR system already read correctly would create
   recognition-selection bias.
4. Page selection uses fixed time intervals and slide identity, not model
   predictions.
5. Pure numeric lines are retained in the new synthetic policy because CNN v1
   exposed weak number coverage.

## 5. Assignment 5.1: Fair Data Conditions

Before training either encoder, a content audit partitions every split into
overlapping slices: numeric, punctuation, code/formula, short text, and long
text. These categories are descriptive rather than mutually exclusive. A line
such as `f(x_1) = 42;` belongs to several slices, which mirrors the compound
difficulty seen in lecture slides.

The frozen augmentation must be deterministic per `(sample_id, epoch, seed)`.
This gives CNN and ViT the same transformed sample even if their batch timing
differs. Validation and test receive resizing only.

The content audit is frozen as:

```text
coverage audit SHA-256:
900c9303e016c2b53e3d970e72a257c2c668a9dae69e0b2b7003df913d64624c

train-only vocabulary SHA-256:
30df193be6e3e1562173f64a3921d61c871cf56c087f3dcb6cf973acdde9119b

shared augmentation SHA-256:
a81e496e1ea10adf60c1f7fe3569163fd4db3c185f9c23143df38b2742a1a6e6
```

Training contains 241 numeric, 121 code/formula, 514 punctuation, and 63
long-text lines. Validation contains 16 occurrences of five characters absent
from the train-only vocabulary. Those characters remain unknown rather than
being copied into training after validation inspection.

### Augmentation decision

The policy uses modest rotation, contrast, brightness, blur, and additive
noise. It does not crop text, warp character order, or alter labels. The policy
is intentionally conservative: the observed domain gap is video capture versus
clean deck rendering, so photometric perturbations are justified; aggressive
geometric distortion would create a different recognition task.

### CTC preflight

At height 32, train/validation widths span `45-936` and `55-863`. With stride
4, their minimum available-minus-required CTC margins are 5 and 10; no sample
is infeasible. The historical stride and maximum width are therefore retained.

### CNN v2 capacity gate

The unchanged CNN architecture has 120,629 parameters under the larger
train-only vocabulary. On 32 deterministic, distinct training labels, it
reached 32/32 exact match at step 420. Loss fell from 12.3442 to 0.0188 and CER
was zero. The CPU model loop took 607.2 seconds. This passes the plumbing and
capacity gate; it is not evidence of lecture-level generalization.

```text
run ID:
20260720T043454017958Z-9fc2afedf4d3

gate report SHA-256:
66e775a4720aa5b4b41cd51b8c2192477cf13c17436f6a2f3233b276a8db3ec5
```

### CPU runtime preflight

The first formal invocation was terminated before epoch 1 appeared after
477.2 seconds. It produced no checkpoint or validation metric and is retained
as a failed run (`20260720T044622225963Z-785eed723f0c`). Process inspection
showed active multi-core computation rather than a deadlock; the outer command
was buffering progress output.

A model-only thread benchmark was then run without validation or test access.
On the widest 16-line batch, one training step took 1.332, 0.648, 0.475, 0.440,
0.463, 0.421, and 0.479 seconds at 1, 4, 8, 12, 14, 16, and 20 intra-op
threads. Sixteen threads were frozen for both CNN and ViT formal runs. The
trainer now writes one flushed JSONL record per epoch so process observability
does not depend on PowerShell output buffering.

Results, model choices, and downstream evaluation will be appended only after
their corresponding gates pass.

## 6. Assignment 5.2: ViT-CTC Design

The ViT reads a line as non-overlapping `32 x 4` strip patches. A learned
linear patch projection produces 64-dimensional tokens; learned absolute
position embeddings preserve horizontal order. Two handwritten pre-LayerNorm
encoder blocks each contain explicit Q/K/V projection, scaled dot-product
multi-head attention, residual connections, and a 64-128-64 GELU MLP. Four
heads give 16 features per head.

The padding mask is applied to attention keys and to every padded query output.
This matters for variable-width OCR: valid text logits must not change merely
because another line in the batch is wider. Both encoders emit one time step
per four input pixels and use the same `CtcProjectionHead`.

| Model | Parameters | Relative to CNN |
| --- | ---: | ---: |
| CNN v2 | 120,629 | 100.00% |
| ViT v1 | 111,253 | 92.23% |

ViT config SHA-256:

```text
c6105234fa3a83a5ec3481176c9480d0a3a1a65fbf031b25497ec9813a939b91
```

Shape, output-length, padding-invariance, finite-loss, and finite-gradient
tests pass before any full training. The 32-line overfit gate remains a
separate capacity check and uses the same selected sample IDs as CNN v2.

The ViT gate passed at step 480 with 32/32 exact lines and zero CER. Loss fell
from 12.3136 to 0.0113 in 112.7 seconds on one CPU thread. CNN v2 passed on the
same IDs at step 420. The step comparison describes optimization behavior; the
wall-clock comparison is diagnostic only because convolution and attention use
different CPU kernels.

## 7. Validation-Only Model Selection

CNN v2 completed 51 epochs and stopped after ten epochs without improving the
validation `(CER, WER)` key. Epoch 41 is frozen:

| Metric | CNN v2, Lecture 4 |
| --- | ---: |
| CER | 0.0628 |
| WER | 0.2311 |
| Exact lines | 47/67 |
| Unknown reference characters | 16 |

```text
run ID:
20260720T045724708964Z-3d3a8db19afe

checkpoint SHA-256:
3ed85c65574b13266c641691faa8455a7d68ca2dd99d36a7faa790e72f38ecab
```

Lecture 5 was not loaded or evaluated during this selection.

ViT v1 ran the full 60-epoch budget. Epoch 55 is frozen:

| Metric | ViT v1, Lecture 4 |
| --- | ---: |
| CER | 0.3179 |
| WER | 1.0000 |
| Exact lines | 2/67 |
| Unknown reference characters | 16 |

```text
run ID:
20260720T052714101598Z-8ab94f2b387a

checkpoint SHA-256:
81120eeec3b61ec26055010bf21f8bf9bf44a66d4e6eabe9bec6fec07e49f257
```

The same optimization budget fits CNN's training distribution much faster.
This supports, but does not by itself prove, the hypothesis that convolution's
locality bias is more sample efficient for this small OCR dataset. A separate
ViT-specific optimizer study would be a new experiment, not a retroactive fix.

## 8. Assignment 5.3: Sealed Test Result

Both checkpoints were frozen before the Lecture 5 comparison protocol passed
preflight. Preflight verified data, split, vocabulary, model config,
checkpoint, parameter-count, and RapidOCR-review hashes without loading a test
batch or running inference.

```text
comparison protocol SHA-256:
dba535eae96afc7f7408083b579dcb5a8aab23b1bac0240fdbb448fde1ac2933
```

The one-time command created a test-access ledger before loading Lecture 5 and
now refuses a second run. Latency used two warmups and seven alternating timed
runs on in-memory batches. CER differences used 5,000 paired lecture-line
bootstrap samples with a predeclared 95% interval.

```text
run ID:
20260720T055037227580Z-f42262e6dd2b

comparison report SHA-256:
b14e10675d39e2f3419c62dfc927b4a2191226c5d19d8eeecee592e3e90de985
```

| System | CER | WER | Exact lines | Parameters | Median ms/line |
| --- | ---: | ---: | ---: | ---: | ---: |
| CNN v2 | 0.1150 | 0.3155 | 73/176 | 120,629 | 4.414 |
| ViT v1 | 0.4461 | 0.9442 | 5/176 | 111,253 | 0.687 |
| RapidOCR stored | 0.0071 | 0.0408 | 158/176 | not measured | not measured |

The paired CNN-minus-ViT CER difference is `-0.3311` with a 95% bootstrap
interval of `[-0.4126, -0.2551]`; all 5,000 resamples favored CNN. The result
supports the predeclared small-data locality-bias hypothesis. ViT is much
faster in this CPU forward-only benchmark, but its accuracy is not competitive.
The speed result must not be generalized to training, GPU inference, decoding,
or image I/O.

RapidOCR is a strong practical reference, not an end-to-end detector result.
Its text comes from the same included polygons used to create this benchmark,
so missed detections and detector latency are absent. The comparison answers
"which recognizer reads accepted line crops best," not "which system reads a
whole slide best."

### Error slices

CNN v2 remains weakest on short lines (`CER 0.4643`) and unknown characters
(`CER 0.2165`). Its long-line CER is `0.0285`, suggesting that local visual
evidence plus language regularity is enough when a line contains context. ViT
v1 degrades on long (`0.8387`), numeric (`0.7403`), and code/formula (`0.7346`)
lines. Passing a 32-line memorization gate therefore did not imply useful
lecture-level generalization.

## 9. Assignment 5.4: OCR-to-Card Cascade

### Question

Does lower line-level OCR error produce more concepts, better-grounded claims,
correct citations, and more usable cards when every OCR system feeds the same
local card generator?

The frozen unit is one slide page. Predictions are grouped back into page order
using the preserved `page_event_id`, `source_block_order`, and timestamp. CNN,
ViT, and RapidOCR receive the same 16 pages. Each page requests one card from
the same `qwen3:4b` digest at temperature zero. Page order is cyclically rotated
across systems so no recognizer always runs first or last. Gold concepts are
not included in the generation prompt.

### Infrastructure pilot and protocol revision

Card-cascade v1 exposed an integration failure after five records. The Ollama
OpenAI endpoint was returning long hidden reasoning traces despite a soft
`/no_think` instruction, and terminating the parent shell left child Python
requests running. The five records remain under
`reader_card_cascade_v1` as an aborted infrastructure run.

The revision was based on latency and transport behavior, not card labels or
model ranking. V2 uses Ollama's explicit `reasoning_effort: "none"`, records
the model digest, bypasses OS proxy settings for loopback URLs, and retains an
atomic record after every page/system pair. This follows the official
[Ollama thinking control](https://docs.ollama.com/capabilities/thinking) and
[OpenAI compatibility](https://docs.ollama.com/api/openai-compatibility)
contracts.

Because L5 outputs were visible during this revision, the downstream study is
exploratory rather than a pristine confirmatory test. A future replication
must calibrate the complete cascade on validation lectures and reserve a new
lecture for one-shot downstream evaluation.

```text
v2 protocol SHA-256:
e98806ff24714259b4d58e0a4debe0f0e24a50261b9138ed1ad6c7fd804e0f06

generation records SHA-256:
73137aabcdef105f3f695de0f55e9fe13d28a7aca3f30336fc8cfe8f9bf516a0

source-audit decisions SHA-256:
8b96c4aba2d17b569dbf6cac7b7d5fe90e96e5d6e78540ce5ac80ca3d9b058a4

source-audit records SHA-256:
cf79a65ab531f3609f20d6dcaaa4a683bbe7279ed591b21e947d64b274e38b7b

final report SHA-256:
c43e29c7966ec8ca36c9bb83bfe40f6a15e26b2a7ac34f29b424aba3f0c1a47e
```

### Generation reliability

| System | Generated pages | Failed pages | Mean seconds/page-system |
| --- | ---: | ---: | ---: |
| CNN v2 | 14/16 | 2 | 8.498 |
| ViT v1 | 16/16 | 0 | 8.128 |
| RapidOCR stored | 12/16 | 4 | 8.617 |

All six failures are `NoGroundedClaimsError`. They are not network, timeout, or
JSON errors. In several cases the LLM combined adjacent OCR lines or inserted
a timestamp into an evidence quote. The product's exact-line grounding filter
then rejected an otherwise sensible card. ViT's 16/16 generation rate is not a
quality win: noisy text encouraged generic or hallucinated cards that happened
to quote one noisy line exactly.

### Downstream source audit

One model-assisted source auditor compared every card with the frozen image,
official text, and predeclared concept. Automatic edit similarity was advisory.
A claim passed only when the slide supported it; a citation also had to map to
the slide and support that claim.

| System | Concept recall | Grounded claim precision | Citation correctness | No-edit acceptance | Usable conversion |
| --- | ---: | ---: | ---: | ---: | ---: |
| CNN v2 | 0.7500 | 0.8667 | 0.6250 | 0.2143 | 0.3750 |
| ViT v1 | 0.4375 | 0.4375 | 0.0000 | 0.0000 | 0.0000 |
| RapidOCR stored | 0.7500 | 0.9167 | 0.9231 | 0.8333 | 0.6875 |

For usable-card conversion, paired page bootstrap differences are:

| Pair, A minus B | Difference | 95% interval |
| --- | ---: | ---: |
| CNN v2 - ViT v1 | +0.3750 | `[+0.1250, +0.6250]` |
| CNN v2 - RapidOCR | -0.3125 | `[-0.6250, -0.0625]` |
| ViT v1 - RapidOCR | -0.6875 | `[-0.8750, -0.4375]` |

### Interpretation

1. OCR quality survives downstream. CNN decisively improves cards over this
   ViT, and RapidOCR improves usable cards over CNN.
2. CER is insufficient as a product metric. RapidOCR has near-perfect line
   CER but still loses four pages to exact-quote composition failures.
3. Layout is an independent information channel. On the feature-representation
   slide, all three text-only cascades confuse the diagram direction and claim
   that `f(x)=Wx` creates features; the image shows features entering that
   classifier to produce class scores.
4. Successful generation is not successful learning content. ViT generated a
   card for every page while producing no card that passed the strict usable
   criterion.
5. The next multimodal baseline should preserve page structure or pass the
   slide image to a VLM, then align its claims with audio and OCR evidence.

## 10. Threats to Validity

1. Five lectures are better than the historical three-lecture study but still
   represent one course and one slide style family.
2. Source-aligned labels need independent human spot-checking before a formal
   publication claim.
3. Polygon geometry begins with RapidOCR detections, so the experiment measures
   line recognition after detection, not page-level text-detection recall.
4. Synthetic and real samples are not independent observations when they share
   source-slide wording; all splitting therefore remains lecture level.
5. Lecture 5 vocabulary statistics can be audited without model inference, but
   repeated metric inspection after the final run would invalidate its role as
   a held-out set.
6. The card audit has one model-assisted rater and no inter-rater reliability;
   a human second rater is required before publication-level claims.
7. Sixteen pages give coarse binary rates and wide uncertainty despite paired
   bootstrap intervals.
8. The downstream protocol was revised after an L5 infrastructure failure, so
   its findings are exploratory. The sealed reader comparison is unaffected.
9. Card generation uses slide OCR without lecture audio. It measures visual
   text contribution, not the final multimodal product pipeline.
