# Card Retrieval and Graph RAG Study

Last updated: 2026-07-21

## Status

This is an **exploratory development study**, not a final test result.

The experiment code, frozen hashes, protocols, and compact results are tracked.
The 100-question benchmark and 20-edge graph are still model-assisted candidates
pending independent human review. The 60-question test split remains unopened by
the experiment runner.

## Research Questions

1. How do BM25, dense card retrieval, and reciprocal-rank fusion compare on the
   same course-card corpus?
2. Does one-hop graph expansion improve multi-hop retrieval under the same
   top-k card budget?
3. Does any retrieval improvement survive downstream grounded answer generation?
4. Does graph expansion increase false retrieval or hurt ordinary single-card
   questions?
5. Can a local 4B model abstain on unsupported questions from prompt instructions
   alone?

## Literature Positioning

- [KG2RAG](https://aclanthology.org/2025.naacl-long.449/) starts from dense seed
  chunks and performs knowledge-graph-guided expansion and organization. It is
  the closest published pattern to this experiment.
- [HippoRAG](https://papers.nips.cc/paper_files/paper/2024/hash/6ddc001d07ca4f319af96a3024f6dbd1-Abstract-Conference.html)
  uses a knowledge graph and Personalized PageRank for multi-hop retrieval.
- [G-Retriever](https://papers.nips.cc/paper_files/paper/2024/hash/efaf1c9726648c8ba363a5c927440529-Abstract-Conference.html)
  retrieves textual graphs through a prize-collecting Steiner-tree formulation.
- [Microsoft GraphRAG](https://arxiv.org/abs/2404.16130) focuses on local and
  global query answering over graph communities. Its global summarization path
  is not the present course-level local-QA target.
- [HotpotQA](https://aclanthology.org/D18-1259/) motivates explicit supporting
  evidence for multi-hop questions.
- [KILT](https://aclanthology.org/2021.naacl-main.200/) motivates provenance-aware
  evaluation rather than answer text alone.
- [RAGChecker](https://proceedings.neurips.cc/paper_files/paper/2024/hash/27245589131d17368cccdfa990cbf16e-Abstract-Datasets_and_Benchmarks_Track.html),
  [ARES](https://aclanthology.org/2024.naacl-long.20/), and
  [RAGAS](https://aclanthology.org/2024.eacl-demo.16/) motivate separating
  retrieval, generation, grounding, and abstention failures.
- [Reciprocal Rank Fusion](https://research.google/pubs/reciprocal-rank-fusion-outperforms-condorcet-and-individual-rank-learning-methods/)
  motivates rank-based BM25/dense fusion without mixing incompatible raw scores.

The first graph baseline deliberately stays simpler than HippoRAG or
G-Retriever: dense top-2 anchors, accepted one-hop edges, and the same final
top-5 card budget as every other system.

## Frozen Development Inputs

| Item | Value |
|---|---:|
| Lectures | 5 |
| Cards | 118 |
| Claims | 140 |
| Evidence spans | 150 |
| Product graph rows | 48 directed / 24 reciprocal pairs |
| Candidate reviewed edges | 20 |
| Corpus SHA-256 | `1a00fff655b303d20ae9db6b658a9fff96ae689a6040a74885d4c2cad8666130` |
| Candidate benchmark SHA-256 | `aedca7b92e2d3608d1fe63882eb1cfabc198878aa7621fdce2d45672552245a4` |
| Candidate review SHA-256 | `610180d403fde5078c1660457d7552face349ab36ec53ade8d4b8481ce781fee` |

All 118 card documents were re-encoded from the frozen snapshot using local
`sentence-transformers/all-MiniLM-L6-v2` normalized 384-dimensional embeddings.
This does not mutate SQLite and avoids excluding the 17 lecture-5 cards whose
product embeddings were initially missing.

## R1: Benchmark Construction

The candidate contains 100 questions:

| Category | Development | Test | Total |
|---|---:|---:|---:|
| Factual | 8 | 12 | 20 |
| Concept | 8 | 12 | 20 |
| Comparison | 8 | 12 | 20 |
| Multi-hop | 8 | 12 | 20 |
| Unanswerable | 8 | 12 | 20 |

Qwen only paraphrased question wording. Card IDs, claim IDs, exact evidence
quotes, and timestamps were copied deterministically from the frozen corpus and
then audited for ownership and byte-for-byte equality.

The first candidate produced 66 quality flags over 33 questions: answer-token
leakage, unsupported `why` shape, and yes/no multi-hop wording. A tracked set of
34 revisions produced v2 with zero current heuristic flags. This does not
replace human annotation; the review sheet remains at
`backend/data/rag_lab/r1/human-review-sheet-v2.md`.

## R2: Retrieval Baselines

Configuration:

- BM25: `k1=1.2`, `b=0.75`
- Dense: normalized MiniLM cosine similarity
- Hybrid: BM25 + dense through RRF with `k=60`
- Noisy graph: 24 deduplicated semantic-similarity pairs
- Candidate trusted graph: 20 model-assisted reviewed edges
- Graph expansion: dense top-2 anchors, one hop, weight `0.35`
- Metrics: Recall@1/3/5, MRR, nDCG@5, multi-hop joint Recall@3,
  unanswerable false retrieval rate, and wall-clock latency

Development results (`n=40`):

| System | Recall@1 | Recall@3 | Recall@5 | MRR | nDCG@5 | Multi-hop joint R@3 | Unanswerable FRR | Median ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BM25 | .406 | .812 | .859 | .737 | .746 | .375 | .125 | .80 |
| Dense | .594 | .969 | 1.000 | .901 | .924 | .750 | .000 | 24.19 |
| Hybrid RRF | .609 | .922 | .969 | .891 | .898 | .500 | .375 | 25.30 |
| Dense + noisy graph | .562 | .969 | 1.000 | .875 | .904 | .750 | .000 | 24.97 |
| Dense + candidate trusted graph | .422 | .969 | 1.000 | .805 | .852 | .875 | .000 | 24.57 |

Paired bootstrap (`5,000` samples) for trusted graph minus dense:

| Slice/metric | Difference | 95% CI |
|---|---:|---:|
| Multi-hop joint Recall@3 | +.125 | [.000, .375] |
| Overall nDCG@5 | -.072 | [-.133, -.016] |
| Single-card nDCG@5 | -.163 | [-.255, -.077] |

Interpretation: graph expansion may improve multi-hop coverage, but the small
multi-hop sample does not establish a reliable gain. The ranking damage on
ordinary single-card questions is larger and more stable.

### Confidence Contract Finding

The initial graph implementation reused its reranking score for abstention.
Because its base top score was always at least `1 / 61`, every unsupported query
looked answerable. R2 v3 separates:

```text
ranking_score    = dense rank prior + graph edge boost
confidence_score = original dense anchor cosine
```

Graph ordering can now change without destroying query-answerability
calibration.

## R3: Grounded Answer Generation

Controlled generation uses:

- local `qwen3:4b`, digest
  `359d7dd4bcdab3d86b87d73ac27966f4dbb9f5efdfcc75d34a8764a09474fae7`
- temperature `0`
- top-5 cards and a 6,000-character maximum context
- identical prompt and evidence schema for Dense and Graph
- exact `card_id`, `claim_id`, and `evidence_id` citations

The first answer schema allowed a free answer plus cited claims. It was stopped
after 18 items because 3 responses hit the 500-token limit and the free answer
added unsupported explanations. The replacement schema permits at most two
short cited claims; application code serializes those claims into the final
answer. It completed 80/80 generations successfully.

Prompt-only refusal failed: both systems answered all 8 unsupported questions.
A frozen dense-anchor confidence gate then produced 8/8 correct refusals and one
shared false abstention.

| System | Abstention F1 | Gold-claim citation recall | Citation precision | Reference cosine proxy | Median generation ms |
|---|---:|---:|---:|---:|---:|
| Dense | .984 | .812 | .812 | .855 | 6,884 |
| Dense + candidate trusted graph | .984 | .833 | .816 | .857 | 6,864 |

Reference cosine is not reported as answer correctness. The generated review
sheet requires independent correctness and entailment labels.

## R4: Dense vs Graph Effect

Under the same top-5 and context-character budget:

- Graph changed top-5 order for 19/40 questions.
- Graph changed the top-5 set for 9/40 questions.
- Generated answer text changed for 8/40 questions.
- Gold-claim citation recall: 1 graph win, 39 ties, 0 losses.
- The only win was `comparison-001` about `k` and the selected nearest-neighbor
  result.
- Multi-hop generated citation recall was identical on all 8 development items.

Graph-minus-dense gold-claim citation recall was `+.016` overall with 95% CI
`[.000, .047]`. This is not evidence of a robust downstream gain.

## Current Conclusion

For direct question answering on this corpus, plain dense retrieval is the
strongest default. Candidate one-hop graph expansion improves one retrieval
slice but harms rank quality and does not yet provide meaningful multi-hop
answer gains. A confidence gate is necessary because local Qwen does not
reliably abstain from prompt instructions when irrelevant context is present.

This does not make the graph useless as a knowledge structure. A separate audit
finds that it currently covers only 27.1% of cards, but 45% of its candidate
edges expose at least one endpoint outside the corresponding Dense top-5
neighborhood. The resulting dual-system hypothesis treats Dense as the direct
answer mechanism and Graph as a typed substrate for exploration, review,
global organization, and future learner-conditioned paths. See
`docs/Graph as associative knowledge structure.md`.

## Validity Threats

1. The benchmark and accepted graph are model-assisted and not independently
   human verified.
2. Graph pairs helped define some paired questions, creating possible circular
   curation bias.
3. Only 8 development multi-hop questions are available.
4. Thresholds were selected and evaluated on the same development split.
5. The 60-question test split has not been opened.
6. Citation correctness is not full semantic correctness or claim entailment.
7. The corpus contains generated draft cards, including known ASR and grounding
   errors that were excluded from selected gold claims but remain retrieval
   distractors.

## Next Valid Experiment

1. Independently review all 100 questions and 20 graph edges.
2. Build the accepted graph before inspecting test questions.
3. Freeze development thresholds and every protocol hash.
4. Open the 60-question test split once.
5. Add more independently authored multi-hop questions and relation paths if
   confidence intervals remain too wide.
6. Only then compare richer graph search such as typed traversal, PPR, or a
   learned router.

## Reproduction

The exact commands and artifact conventions live in
`backend/rag_lab/README.md`. Compact tracked results are stored under
`docs/experiments/`; full local artifacts are stored under
`backend/data/rag_lab/` and ignored by Git.
