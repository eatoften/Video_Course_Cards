# RAG Roadmap

Last updated: 2026-07-02

## Goal

Build a simple, explainable card-only dense RAG baseline before moving into
graph retrieval, learned routing, or agentic RL.

The baseline should answer this question first:

```text
Given a user question, can the system retrieve the most relevant knowledge
cards from the current course?
```

Only after this works well should the system generate final answers with Qwen.

## Why Start With A Baseline

The long-term project direction includes:

- card similarity graphs
- graph-based retrieval routes
- learned query representations
- learned retrievers or routers
- agentic decision policies over the card graph

Those ideas need a comparison point. A plain dense RAG baseline gives us that
comparison.

Baseline retrieval:

```text
question
-> query embedding
-> cosine similarity against card embeddings
-> top-k cards
```

Future graph retrieval:

```text
question
-> query embedding
-> anchor cards
-> graph route / evidence subgraph
-> answer
```

## Scope Decision

Use only knowledge card embeddings for the first RAG baseline.

Do not add transcript chunk embeddings in this phase.

Reason:

- knowledge cards are already structured and claim-grounded
- card embeddings already exist in SQLite
- card-level retrieval is easier to inspect and debug
- transcript chunks can be added later as fallback evidence if card recall is
  not enough

Current reusable code:

- `backend/app/embedding.py`
  - `SentenceTransformerEmbedder`
  - `cosine_similarity`
- `backend/app/card_embedding.py`
  - `CardEmbedding`
  - vector BLOB serialization
- `backend/app/card_embedding_store.py`
  - card embedding persistence
- `backend/app/card_embedding_service.py`
  - card/job/course embedding generation

Missing reusable capability:

```text
load full card embeddings by course or by job
```

The store currently exposes embedding metadata for course/job status, but dense
retrieval needs full vectors.

## UI Placement

The main frontend is already dense, so do not add a permanent chat box to the
main content area yet.

Use the right rail instead.

Proposed UI:

```text
Right rail
-> Cards tab
-> Ask tab
```

The first Ask tab should only retrieve related cards:

```text
Question: What is SVD?

Top matches:
1. Singular Value Decomposition    score 0.84
2. Orthogonal Matrices             score 0.71
3. Matrix Factorization            score 0.68
```

Do not generate answers in the first UI version. Retrieval quality should be
visible before adding Qwen.

## Milestone 17.1: Card-Only Dense Retrieval

### Step 17.1.1: Store Query Capability

Add full-vector query functions to `card_embedding_store.py`:

```text
list_card_embeddings_for_job(job_id)
list_card_embeddings_for_course(course_id)
```

Return:

```text
list[CardEmbedding]
```

Knowledge to learn:

- SQLite joins
- BLOB to vector decoding
- why retrieval needs full vectors, not only embedding metadata

Tests:

- create two cards
- insert embeddings
- list embeddings by job
- confirm both vectors round-trip correctly

### Step 17.1.2: RAG Schema

Create:

```text
backend/app/rag.py
```

Models:

```text
RagRetrieveRequest
RetrievedCard
RagRetrieveResponse
```

Suggested request:

```json
{
  "question": "What is singular value decomposition?",
  "course_id": "uncategorized",
  "job_id": null,
  "top_k": 5,
  "min_score": 0.25
}
```

Suggested response:

```json
{
  "question": "What is singular value decomposition?",
  "results": [
    {
      "card_id": "...",
      "job_id": "...",
      "title": "Singular Value Decomposition",
      "summary": "...",
      "score": 0.82,
      "source_start_seconds": 724.0,
      "source_end_seconds": 738.0
    }
  ]
}
```

Knowledge to learn:

- Pydantic request models
- Pydantic response models
- validation with `Field`
- why API schema comes before service code

Tests:

- empty question is rejected
- `top_k` must be at least 1
- `min_score` must be within a sensible range

### Step 17.1.3: Pure Ranking Function

Create:

```text
backend/app/rag_retriever.py
```

Add a pure function:

```text
rank_cards_by_similarity(query_vector, candidates, top_k, min_score)
```

Input:

```text
query_vector: list[float]
candidates: list[tuple[KnowledgeCard, CardEmbedding]]
```

Output:

```text
list[RetrievedCard]
```

Knowledge to learn:

- cosine similarity
- sorting by score
- top-k filtering
- pure functions
- why pure functions are easy to test

Tests:

```text
query = [1, 0]
card A = [0.9, 0.1]
card B = [0, 1]
expected: card A ranks first
```

### Step 17.1.4: Retrieval Service

Create:

```text
backend/app/rag_service.py
```

Responsibilities:

```text
1. validate course_id or job_id
2. ensure card embeddings exist
3. embed the user question
4. load card + embedding candidates
5. call rank_cards_by_similarity
```

Important design:

```text
card embeddings are stored ahead of time
query embedding is computed at request time
```

Use:

- `SentenceTransformerEmbedder`
- `card_embedding_service.embed_course_cards`
- `card_embedding_service.embed_job_cards`
- `card_embedding_store.list_card_embeddings_for_course`
- `card_embedding_store.list_card_embeddings_for_job`
- `knowledge_card_store.get_card`

Knowledge to learn:

- service-layer orchestration
- dependency injection with a fake embedder in tests
- why query embedding is not stored
- how stale/missing card embeddings are handled

Tests:

- fake embedder returns a query vector
- two cards exist
- one card is closer to the query
- service returns the closer card first

### Step 17.1.5: Retrieval API

Add to `main.py`:

```text
POST /rag/retrieve
```

This endpoint returns related cards only.

It should not call Qwen yet.

Knowledge to learn:

- FastAPI POST endpoint
- request body parsing
- response models
- mapping service errors to HTTP errors

Tests:

- request with `course_id` returns top cards
- request with `job_id` returns only that job's cards
- missing course/job returns 404

### Step 17.1.6: Frontend Ask Tab

Add an Ask tab to the right rail:

```text
Cards | Ask
```

Ask tab UI:

```text
question input
retrieve button
loading state
error state
top retrieved cards
score display
```

Do not add answer generation yet.

Knowledge to learn:

- React state
- controlled inputs
- `fetch` POST requests
- loading/error states
- conditional rendering

## Milestone 17.2: RAG Answer Generation

After retrieval quality is inspectable, add answer generation.

Flow:

```text
question
-> retrieve top-k cards
-> build grounded prompt
-> Qwen via Ollama
-> answer with citations
```

API:

```text
POST /rag/query
```

Response:

```json
{
  "answer": "...",
  "citations": [
    {
      "card_id": "...",
      "title": "...",
      "score": 0.82,
      "source_start_seconds": 724.0,
      "source_end_seconds": 738.0
    }
  ],
  "retrieved_cards": []
}
```

Prompt rule:

```text
Use only retrieved cards.
If the cards do not contain enough evidence, say that there is not enough
evidence.
```

Knowledge to learn:

- prompt construction
- grounded generation
- citations
- answer abstention
- local Qwen/Ollama API calls

## Milestone 17.3: RAG Query Logs

Store each query for later evaluation.

Potential table:

```text
rag_query_logs
- id
- question
- course_id
- job_id
- top_k
- min_score
- retrieved_card_ids_json
- scores_json
- answer
- model
- latency_ms
- created_at
```

Knowledge to learn:

- experiment logging
- retrieval debugging
- latency measurement
- baseline evaluation

## Milestone 17.4: Baseline Evaluation

Add simple evaluation before graph RAG.

Metrics:

- retrieval hit rate
- average top-1 score
- answer groundedness
- unsupported answer rate
- citation correctness
- latency
- user feedback: good / bad / needs edit

Resume angle:

```text
I built a local RAG baseline and logged retrieval quality, grounding, and
latency so later graph-based retrieval methods could be compared against a
measurable baseline.
```

## Future: Graph RAG Comparison

After ordinary dense RAG works, add graph retrieval.

Future flow:

```text
question
-> dense retrieval anchors
-> card similarity graph expansion
-> evidence subgraph
-> answer with citations
```

Future components:

- `card_similarity_edges`
- relation types:
  - related
  - prerequisite
  - example_of
  - contrast_with
  - part_of
- manual edge editing
- graph visualization
- query-to-anchor evaluation

## Future: Learned Retriever And Router

Only after baseline and graph RAG are measurable, start learning.

Possible path:

```text
query embedding
-> learned projection layer
-> better anchor retrieval
-> graph router
-> selected route
-> answer generation
```

Training data sources:

- user searches
- clicked cards
- saved/deleted cards
- answer feedback
- manual edits
- citation clicks

This is where the project can move toward agentic RL:

```text
state = query + selected evidence so far
action = choose next card / stop / answer
reward = grounded answer quality + user feedback
```

Do not start here. Use the dense RAG baseline first.

