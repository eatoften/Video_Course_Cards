# Video Course Cards Roadmap

Last updated: 2026-07-11

## Product Direction

Video Course Cards is a local-first AI learning workspace. The product goal is
to turn long course videos into grounded knowledge cards, then organize those
cards into a navigable personal knowledge system.

Current shape:

```text
course videos
-> transcripts
-> semantic transcript chunks
-> grounded knowledge cards
-> card embeddings
-> card relations
-> graph / tree / RAG
```

Core design decisions:

- SQLite is the source of truth.
- Markdown is an export snapshot, not the primary database.
- Cards are structured records, not plain text blobs.
- Each important generated claim should be grounded in transcript evidence.
- Similarity edges should be persisted in `card_relations`, not only computed in
  the frontend.
- The UI should evolve from a single crowded workspace into an Obsidian-like
  multi-view learning environment.

## Current State

The project already has a working local demo:

- FastAPI backend.
- React + Vite frontend.
- Tauri Windows desktop shell.
- Packaged FastAPI sidecar for the desktop app.
- GitHub Release installer pipeline.
- SQLite job system.
- Local video upload.
- ffprobe validation.
- FFmpeg audio extraction.
- faster-whisper transcription.
- Transcript timeline UI.
- Transcript semantic chunking.
- Local Ollama/Qwen card generation.
- Claim-level evidence grounding.
- Knowledge card persistence.
- Card editing, deletion, tags, review state, and user notes.
- Card embeddings.
- Basic card-based dense retrieval.
- Persistent card relations generated from cosine similarity.
- Obsidian-like Workspace and Graph views.
- Interactive course graph with relation review and manual editing.
- Local Qwen-assisted relation typing.
- Markdown folder export.

Current product capability:

```text
local course video
-> transcript
-> semantic chunks
-> grounded cards
-> searchable saved cards
-> Markdown export
```

## Completed Milestones

### Milestone 0: Project Foundation

- uv + Python 3.11 backend.
- FastAPI.
- React + TypeScript + Vite.
- pytest.
- Monorepo structure.

### Milestone 1: Video Upload

- `POST /videos`.
- Local upload storage.
- Extension allowlist.
- MIME validation.
- CORS.

### Milestone 2: Media Validation

- ffprobe integration.
- Stream metadata parsing.
- Fake-video detection.
- Mocked ffprobe tests.

### Milestone 3: Audio and ASR

- FFmpeg audio extraction.
- 16 kHz mono PCM WAV conversion.
- faster-whisper integration.
- Timestamped transcript segments.
- Transcript JSON save/load.

### Milestone 4: Processing Pipeline

Pipeline:

```text
probe -> metadata -> audio -> transcribe -> save
```

- `VideoPipeline.process()`.
- Structured processing result.
- Mocked external tools in tests.

### Milestone 5: SQLite Job System

Job state flow:

```text
uploaded
-> probing
-> extracting_audio
-> transcribing
-> completed
```

Any failed stage becomes:

```text
failed
```

Implemented:

- SQLite-backed jobs.
- Created/updated/started/completed timestamps.
- Original filename, stored filename, file size.
- Retry endpoint.
- Job list endpoint.
- Service/store split.

### Milestone 6: Transcript API

- `GET /jobs/{job_id}/transcript`.
- Structured transcript response with language, duration, and timestamped
  segments.

### Milestone 7: Real Frontend Integration

- Upload video from UI.
- Start processing.
- Poll job state.
- Load transcript after completion.
- Display transcript beside workspace.

### Milestone 8: Transcript Timeline

- Video player.
- Transcript list.
- Segment selection.
- Current segment highlighting.
- Fixed transcript overflow panel.

### Milestone 9: Knowledge Context Layer

- Selected transcript spans become card-generation context.
- Context windows preserve source timestamps.

### Milestone 10: Local LLM Integration

- Ollama/OpenAI-compatible local model client.
- Qwen model selection.
- LLM status endpoint.
- LLM model list endpoint.
- `/cards/draft`.

### Milestone 11: Knowledge Card Persistence

- `knowledge_cards` table.
- Save/edit/delete cards.
- Query cards by job and course.
- Delete uploaded videos and associated cards.

### Milestone 12: Claim-Level Grounding

- Cards must include claims.
- Claims must include transcript evidence.
- Evidence quotes are verified against source transcript.
- Verified evidence receives deterministic timestamps.
- Unsupported claims are dropped.

### Milestone 13: Card Generation Reliability

- Generation progress.
- Metadata.
- Timeout handling.
- Cancel generation.
- Better error responses.
- Tests for slow/failing local model paths.

### Milestone 14: Course/Card Workspace

- Course list.
- Video list.
- Course-level card rail.
- Card detail view.
- Tags.
- Review state.
- User notes decoupled from cards.
- Delete all cards for a job or course.

### Milestone 15: Export

- Export one job's cards as Markdown.
- Export all cards as Markdown.
- Obsidian-friendly folder layout.
- Source video and timestamps.
- Claims and evidence.
- Active recall question/answer.
- Local folder export.
- Zip export retained as a portable packaging option.

## Next Phase: Persistent Card Relations And Graph View

The next phase moves the project from a list of cards to a persistent knowledge
structure.

The immediate target is:

```text
card_embeddings
-> cosine similarity
-> card_relations table
-> related cards API
-> Graph view in the frontend
```

This is intentionally not yet full GraphRAG. First we need a durable relation
layer that can be inspected, updated, filtered, and later improved by user
feedback or LLM relation extraction.

## Milestone 16: Card Relations Table (Completed)

Problem:

Cards currently have embeddings, but the relationships between cards are not
stored. If related cards are computed only in memory, the system cannot inspect,
edit, evaluate, or build a graph over those relationships.

Goal:

Add a persistent `card_relations` table as the first real knowledge-graph layer.

Implemented schema:

```sql
CREATE TABLE IF NOT EXISTS card_relations (
    id TEXT PRIMARY KEY,
    course_id TEXT NOT NULL,
    source_card_id TEXT NOT NULL,
    target_card_id TEXT NOT NULL,
    relation_type TEXT NOT NULL,
    score REAL NOT NULL,
    method TEXT NOT NULL,
    model TEXT,
    explanation TEXT,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
```

Initial relation type:

```text
semantic_similarity
```

Future relation types:

```text
prerequisite
related
example_of
contrast_with
part_of
```

Initial method:

```text
cosine_similarity
```

Status values:

```text
suggested
accepted
rejected
hidden
```

Indexes:

```sql
CREATE INDEX IF NOT EXISTS idx_card_relations_course_id
ON card_relations (course_id);

CREATE INDEX IF NOT EXISTS idx_card_relations_source_card_id
ON card_relations (source_card_id);

CREATE INDEX IF NOT EXISTS idx_card_relations_target_card_id
ON card_relations (target_card_id);

CREATE UNIQUE INDEX IF NOT EXISTS idx_card_relations_unique_pair_type
ON card_relations (
    source_card_id,
    target_card_id,
    relation_type,
    method
);
```

Design rules:

- Store directional rows even when similarity is symmetric.
- Normalize pair ordering only if we later decide graph edges should be
  undirected at storage level.
- Keep `score` numeric and model-agnostic.
- Keep `method` so future relations can come from LLMs or manual editing.
- Keep `status` so the user can accept/reject relation suggestions later.
- Do not delete user-reviewed relations during recomputation.

Planned backend files:

```text
backend/app/card_relation.py
backend/app/card_relation_store.py
backend/app/card_relation_service.py
backend/app/card_similarity.py
```

Responsibilities:

```text
card_similarity.py
  cosine similarity and top-k scoring

card_relation_store.py
  SQLite CRUD for card_relations

card_relation_service.py
  recompute relations, list related cards, graph payload assembly

main.py
  HTTP routes only
```

Knowledge to learn:

- Graph edge table design.
- Many-to-many relationships in SQLite.
- Idempotent recomputation.
- Separating algorithm code from persistence code.

## Milestone 17: Relation Generation From Card Embeddings (Completed)

Problem:

The system needs to compute related-card edges from existing card embeddings and
write them into `card_relations`.

Planned algorithm:

```text
load cards for course
load card embeddings for course
for every pair of cards:
    cosine_similarity(a, b)
    if score >= threshold:
        keep candidate
for each source card:
    keep top_k candidates
upsert into card_relations
```

Initial parameters:

```text
threshold = 0.72
top_k = 5
relation_type = semantic_similarity
method = cosine_similarity
status = suggested
```

Important constraints:

- Only compare cards within the same course by default.
- Skip cards without embeddings.
- Skip self-relations.
- Skip duplicate source-target rows.
- Upsert suggested relations when scores change.
- Preserve `accepted` and `rejected` relations if the user has reviewed them.
- Delete or mark stale auto-suggested relations when they fall below threshold.

Planned APIs:

```text
POST /courses/{course_id}/card-relations/recompute
GET  /courses/{course_id}/card-relations
GET  /cards/{card_id}/related
PATCH /card-relations/{relation_id}
DELETE /card-relations/{relation_id}
```

Example `GET /cards/{card_id}/related` response:

```json
{
  "card_id": "...",
  "related": [
    {
      "relation_id": "...",
      "card_id": "...",
      "title": "Gradient Descent",
      "summary": "...",
      "tags": ["optimization"],
      "review_state": "draft",
      "relation_type": "semantic_similarity",
      "score": 0.84,
      "method": "cosine_similarity",
      "status": "suggested",
      "source_start_seconds": 312.4,
      "source_end_seconds": 337.2
    }
  ]
}
```

Example `GET /courses/{course_id}/card-relations` response:

```json
{
  "course_id": "uncategorized",
  "nodes": [
    {
      "id": "...",
      "title": "Backpropagation",
      "tags": ["deep learning"],
      "review_state": "draft"
    }
  ],
  "edges": [
    {
      "id": "...",
      "source": "...",
      "target": "...",
      "relation_type": "semantic_similarity",
      "score": 0.82,
      "status": "suggested"
    }
  ]
}
```

Testing plan:

- cosine similarity unit tests.
- top-k selection tests.
- no self-edge tests.
- threshold tests.
- upsert tests.
- reviewed relation preservation tests.
- API tests for recompute/list/related/update/delete.

Knowledge to learn:

- Cosine similarity over stored BLOB vectors.
- O(n^2) pairwise comparison tradeoffs.
- Threshold tuning.
- Graph edge persistence.
- Relation recomputation policies.

## Milestone 18: Left Sidebar And Multi-View Frontend (Completed)

Problem:

The current UI grew from a single workspace. The right rail now carries too much
responsibility: card list, selected card, ask panel, filters, and navigation.
Before adding a graph, the app needs a stable left-side navigation model.

Goal:

Introduce an Obsidian-like left navigation bar while preserving the current
workspace.

Initial views:

```text
Workspace
Graph
```

Future views:

```text
Cards
Search / RAG
Settings
Runtime
Exports
```

URL state:

```text
?view=workspace&course=uncategorized&card=...
?view=graph&course=uncategorized&card=...
```

Implementation plan:

- Add `AppView` state:

```ts
type AppView = 'workspace' | 'graph'
```

- Parse and write `view` in URL query state.
- Add a persistent left navigation sidebar.
- Move current UI into `WorkspaceView`.
- Add placeholder `GraphView`.
- Keep existing card rail functional during the transition.
- Do not rewrite the entire frontend in one step.

Design direction:

```text
left nav:
  Workspace
  Graph

main area:
  selected view

existing right rail:
  kept for workspace until graph/card views mature
```

Knowledge to learn:

- React state decomposition.
- URL query-state synchronization.
- Multi-view application layout.
- Incremental frontend refactoring.

## Milestone 19: Graph View Version 1 (Completed)

Problem:

Users need to see which cards are related and move through the course memory by
concept similarity.

Goal:

Build a first useful graph view using persisted `card_relations`.

Version 1 should be list-first, graph-second:

```text
selected course
-> selected card
-> related cards ranked by score
-> relation detail
```

Graph view layout:

```text
left panel:
  course card list
  search/filter

center:
  selected card summary
  related cards ranked by score

right panel or lower area:
  relation metadata
  score
  status
  source/target cards
```

Controls:

- Recompute relations.
- Similarity threshold.
- Top-k.
- Filter by relation status.
- Filter by tag.
- Click related card to select it.
- Jump back to workspace card detail.

First version does not require a force-directed canvas. A ranked relation list is
easier to debug and more useful for verifying similarity quality.

Knowledge to learn:

- Rendering graph data without a graph canvas.
- Debugging embedding similarity.
- React list/detail patterns.
- Filtering persisted relationships.

## Milestone 20: Graph Visualization Version 2 (Completed)

Problem:

Once relations are reliable, the user should be able to see the course as a
network of concepts.

Planned work:

- Add a graph visualization library:
  - `react-force-graph-2d`
  - or `vis-network`
  - or D3 force simulation later
- Render:
  - node = card
  - edge = card relation
  - edge weight = similarity score
  - node color = tag or review state
- Interactions:
  - click node to open card
  - hover node to show title/summary
  - threshold slider
  - relation type filter
  - hide rejected edges
  - show accepted edges more strongly

Do not start with a complex graph canvas. The graph should be added after the
relation list is already correct.

Knowledge to learn:

- Graph visualization.
- Force layout tradeoffs.
- Visual encoding for edge weight and relation status.
- Graph UX for learning tools.

## Milestone 21: Relation Review And Manual Editing (Completed)

Problem:

Embedding similarity can suggest relationships, but users should control which
relationships become trusted knowledge structure.

Planned work:

- Accept/reject suggested relations.
- Hide noisy relations.
- Add manual relation creation.
- Allow relation type changes:

```text
semantic_similarity -> prerequisite
semantic_similarity -> example_of
semantic_similarity -> contrast_with
```

- Add optional user explanation.
- Preserve user-reviewed relations during recomputation.
- Add relation history later if needed.

Knowledge to learn:

- Human-in-the-loop data curation.
- Trust states in graph data.
- Separating suggested data from accepted knowledge.

## Milestone 22: LLM-Assisted Relation Typing (Completed)

Problem:

Cosine similarity only says "these cards are close." It does not say why.

Planned work:

- Use high-similarity pairs as candidates.
- Ask local Qwen to classify the relationship:

```text
prerequisite
related
example_of
contrast_with
part_of
unclear
```

- Ask for a short explanation.
- Store result in `card_relations`:
  - `relation_type`
  - `method = local_llm`
  - `model`
  - `explanation`
  - `status = suggested`
- Show explanation in Graph view.
- Let user accept/reject/change relation type.

Knowledge to learn:

- Relation extraction.
- LLM classification prompts.
- Model-generated graph edges.
- Human review of model-suggested structure.

## Milestone 23: Card-Based RAG Assistant

Problem:

Users should be able to ask questions about a course and receive answers
grounded in cards and timestamps.

Planned work:

- Use query embedding to retrieve relevant cards.
- Expand context with accepted/suggested card relations.
- Build a grounded context prompt.
- Use local Qwen to answer.
- Require citations:
  - card title
  - evidence quote
  - source video timestamp
- Say "not enough evidence" when retrieval does not support an answer.

Flow:

```text
question
-> query embedding
-> retrieve top cards
-> expand through card_relations
-> build grounded prompt
-> local LLM answer with citations
```

Knowledge to learn:

- RAG.
- Graph expansion.
- Citation grounding.
- Context construction.
- Retrieval evaluation.

## Milestone 24: Evaluation Layer

Problem:

The project should be evaluated, not only demonstrated.

Planned measurements:

- Unsupported claim rate.
- Grounding pass rate.
- Generation latency.
- Duplicate card rate.
- Retrieval hit rate.
- Relation precision.
- Accepted/rejected relation rate.
- Graph noise rate.
- User edit distance.

Planned tables:

```text
evaluation_runs
card_feedback
relation_feedback
generation_feedback
```

Resume angle:

> I built a local-first AI learning system and designed evaluation metrics for
> grounded card generation, retrieval quality, and graph relation quality.

Knowledge to learn:

- LLM evaluation.
- Product metrics.
- Regression testing for AI features.
- Failure mode analysis.

## Milestone 25: Feedback Dataset And Agentic Learning Loop

Problem:

The system should capture how users improve generated knowledge so future
versions can learn from it.

Planned work:

- Store generated card -> edited card diffs.
- Store save/delete decisions.
- Store accepted/rejected relation suggestions.
- Store evidence clicks.
- Store RAG answer feedback.
- Build preference-style records.
- Prepare data for:
  - prompt optimization
  - reranker training
  - reward modeling
  - agentic policy improvement

Knowledge to learn:

- Feedback data modeling.
- Preference data.
- Human correction logs.
- Agentic learning loop design.

## Immediate Implementation Order

Completed in this phase:

```text
[x] 16.1 Add card_relations schema
[x] 16.2 Add card relation models/store/service
[x] 16.3 Add cosine similarity recompute flow
[x] 16.4 Add related cards and course graph APIs
[x] 16.5 Add tests
[x] 16.6 Add left sidebar view navigation
[x] 16.7 Add Graph view list-first UI
[x] 20 Add interactive force-directed graph visualization
[x] 21 Add relation review and manual editing
[x] 22 Add local Qwen-assisted relation typing
```

Deferred by product decision:

```text
23 Card-based grounded RAG answers
24 Evaluation layer
25 Feedback dataset and agentic learning loop
```

Why this order:

- The database relation layer must exist before the graph UI.
- Persisted relations let us debug and evaluate similarity.
- A list-first graph view is easier to verify than a canvas graph.
- The left sidebar creates room for future views without making the workspace
  even more crowded.

## Long-Term Final Shape

The final system should feel like:

```text
upload a course
-> transcribe videos locally
-> create semantic transcript chunks
-> generate grounded cards
-> embed cards
-> persist card relations
-> review relationship suggestions
-> explore graph/tree views
-> ask grounded questions over course memory
-> export Markdown snapshots
-> collect feedback for future learning loops
```
