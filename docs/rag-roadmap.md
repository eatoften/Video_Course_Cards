# RAG Research Roadmap

Last updated: 2026-07-21

## Objective

Build a measured card-level RAG baseline before introducing learned graph
routing or agentic policies. Retrieval, grounding, refusal, and downstream
answer quality must be evaluated separately.

## Completed Development Work

### R0: Product Dense Retrieval

- Card embeddings in SQLite
- Query embedding with local MiniLM
- Cosine top-k retrieval API
- Frontend Ask tab showing retrieved cards

### R1: Candidate Evaluation Dataset

- Frozen 118-card corpus with claim/evidence/timestamp provenance
- 100 questions: factual, concept, comparison, multi-hop, unanswerable
- 40-question development and unopened 60-question test split
- Structural and wording-quality audit
- Human review sheet

Status: **candidate only; independent review pending**.

### R2: Retrieval Baselines

- BM25
- Dense MiniLM retrieval
- BM25 + dense reciprocal-rank fusion
- Dense + noisy one-hop graph
- Dense + candidate reviewed one-hop graph
- Recall@1/3/5, MRR, nDCG, latency, refusal calibration
- Paired bootstrap confidence intervals

Development finding: Dense is the strongest default. Graph expansion improves
one small multi-hop retrieval slice but damages overall and single-card ranking.

### R3: Grounded Generation

- Fixed Qwen model, prompt, top-k, and character budget
- Claim-only structured generation
- Exact card/claim/evidence citations
- Dense-anchor confidence gate
- Resumable JSONL experiment artifacts
- Human answer-review sheet

Development finding: prompt-only refusal fails; calibrated pre-generation gating
is necessary.

### R4: Graph RAG Comparison

- Same top-5 and generation budget for Dense and Graph
- Per-question win/tie/loss analysis
- Bootstrap comparison of citation metrics

Development finding: one graph win, 39 ties, and no multi-hop generation gain.

See `docs/RAG retrieval and graph study.md` for methods and results.

## Required Before Formal Test

1. Independently review all benchmark questions, gold claims, evidence spans,
   timestamps, and answerability labels.
2. Independently curate the accepted graph without looking at test questions.
3. Mark accepted items and graph review as human verified.
4. Freeze every protocol, threshold, model digest, and artifact hash.
5. Open the 60-question test split once.
6. Report confidence intervals and every deviation from protocol.

## After The Baseline Is Valid

### Parallel Track: Graph As Knowledge Substrate

Do not require one retriever to serve every task:

```text
direct question -> Dense -> evidence gate -> answer
explore/review  -> Dense anchor -> typed Graph -> concept trail
```

The first structural audit records 27.1% graph coverage and finds that 45% of
candidate edges contain at least one association outside Dense top-5. This is a
hypothesis-generating result, not evidence of large-scale associative memory.

Measure graph value using relation precision, nonlocal useful discovery, path
quality, community stability, prerequisite violations, and learning outcomes.
See `docs/Graph as associative knowledge structure.md` and the compact artifact
`docs/experiments/rag_graph_organization_audit_v1.json`.

### R5: Harder Multi-hop Benchmark

- More lectures and independently authored paths
- Two- and three-evidence questions
- Hard negatives from neighboring concepts
- Typed relation-path labels

### R6: Stronger Graph Retrieval

- Type-aware expansion
- Relation-specific weights
- Personalized PageRank baseline
- Budgeted path search
- Graph pruning and noise ablation

### R7: Transcript Fallback

- Retrieve transcript evidence only when card coverage is insufficient
- Preserve timestamp provenance
- Compare card-only against card-plus-transcript retrieval

### R8: Learned Router

Only after R1-R7 are measured:

```text
query representation
-> choose dense anchor / graph expansion / transcript fallback / abstain
-> collect reward from retrieval, grounding, and human feedback
```

This is the bridge to the longer-term agentic RL research direction.
