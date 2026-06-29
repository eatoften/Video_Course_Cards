# Video Course Cards Roadmap

Last updated: 2026-06-28

## Project Positioning

Video Course Cards is a local-first AI learning workspace that turns long-form
course videos into grounded knowledge cards, searchable transcript memory, and an
evolving personal knowledge graph.

中文定位：

> 一个本地优先的 AI 学习系统：把长课程视频转成可追溯知识卡片、可检索字幕记忆，以及不断演化的个人知识图谱。

## Current State

The project has already completed the core local video-to-knowledge pipeline:

```text
video upload
-> ffprobe validation
-> ffmpeg audio extraction
-> Whisper transcription
-> SQLite job system
-> transcript API
-> React frontend upload/run/transcript UI
-> local Ollama/Qwen integration
-> knowledge_cards SQLite persistence
-> claim-level grounding
```

The current product capability is:

```text
local course video -> transcript -> grounded knowledge cards
```

## Completed Milestones

### Milestone 0: Project Foundation

- Python 3.11 backend managed by uv
- FastAPI backend
- React + Vite frontend
- pytest test suite
- Monorepo layout

### Milestone 1: Video Upload

- `POST /videos`
- Local upload storage under `backend/data/uploads`
- Extension allowlist
- MIME type validation
- FastAPI CORS setup

### Milestone 2: Media Validation

- ffprobe integration
- JSON metadata parsing
- Video/audio stream detection
- Fake-video detection
- `MediaProbeError`
- Mocked ffprobe tests

### Milestone 3: Audio and ASR

- FFmpeg audio extraction
- 16 kHz mono PCM WAV conversion
- faster-whisper integration
- CPU int8 inference
- Timestamped transcript segments
- Structured `TranscriptResult`
- Transcript JSON save/load tests

### Milestone 4: Processing Pipeline

- `VideoPipeline.process()`
- Step flow:

```text
probe -> metadata -> audio -> transcribe -> save
```

- Fake transcriber tests
- Mocked external dependency tests
- `VideoProcessingResult`

### Milestone 5: SQLite Job System

- SQLite-backed jobs table
- Normalized job status flow:

```text
uploaded
-> probing
-> extracting_audio
-> transcribing
-> completed
```

- Any failed stage becomes `failed`
- Job timestamps:
  - `created_at`
  - `updated_at`
  - `started_at`
  - `completed_at`
- Upload metadata:
  - `original_filename`
  - `stored_name`
  - `size_bytes`
- Retry endpoint: `POST /jobs/{job_id}/retry`
- Job list endpoint: `GET /jobs`
- Route layer cleaned up around service/store separation

### Milestone 6: Transcript API

- `GET /jobs/{job_id}/transcript`
- Structured transcript response:

```json
{
  "language": "zh",
  "duration_seconds": 123.4,
  "segments": [
    {
      "start_seconds": 0.0,
      "end_seconds": 4.2,
      "text": "..."
    }
  ]
}
```

### Milestone 7: Real Frontend Integration

- Select video
- Upload via `POST /videos`
- Show job id and job state
- Start processing
- Poll `GET /jobs/{job_id}`
- Load transcript after completion
- Display transcript segments

### Milestone 8: Transcript Timeline

- Video player
- Transcript segment list
- Segment selection
- Current segment highlighting
- Transcript panel scroll area for long transcripts

### Milestone 9: Knowledge Context Layer

- Select transcript range
- Build context window from selected segments
- Send selected context into card generation

### Milestone 10: Local LLM Integration

- Local-first LLM provider configuration
- Ollama OpenAI-compatible endpoint
- Qwen model selection
- LLM status endpoint
- LLM model list endpoint
- `/cards/draft` generation API
- Frontend model selector

### Milestone 11: Knowledge Card Persistence

- `knowledge_cards` SQLite table
- Save generated cards
- Edit saved cards
- Delete saved cards
- Query cards by job
- Delete uploaded videos/jobs and associated cards

### Milestone 12: Claim-Level Grounding

- Each card must include at least one grounded claim
- Each claim must include evidence traced to transcript text
- LLM outputs candidate `claims` and `evidence_quotes`
- Python verifies evidence quotes against transcript segments
- Verified evidence receives deterministic timestamps
- Unsupported claims are dropped
- Cards with no verified claim are rejected
- Frontend displays claims, quotes, and source timestamps

## Next Development Plan

## Phase 1: Stabilize the Current MVP

The goal of this phase is to make the current video-to-card workflow reliable,
transparent, and comfortable to use.

### Milestone 13: Card Generation Reliability

Problem:

Local generation can be slow, and the user currently does not get enough
feedback while the model is working.

Planned work:

- Show generation state in the frontend:

```text
calling local model
validating claims
grounding evidence
done
```

- Add clearer backend error types for `/cards/draft`
- Support canceling a generation request
- Warn users when the selected transcript context is too long
- Record generation metadata:
  - elapsed time
  - selected model
  - approximate input length
  - number of generated cards
  - number of grounded claims
- Shorten and stabilize the Qwen 4B prompt
- Add tests for slow generation and failed generation paths

Knowledge to learn:

- API latency
- Request timeout
- UI loading states
- LLM output robustness
- Local model performance tradeoffs

### Milestone 14: Card Review Workflow

Problem:

Generated cards need a better review and editing workflow before they become a
real study asset.

Planned work:

- Make draft cards and saved cards visually distinct
- Save all generated cards at once
- Jump from evidence timestamp to video playback time
- Edit claims
- Delete claims
- Edit evidence text when needed
- Sort cards by transcript time
- Add optional tags
- Add user notes
- Add difficulty controls

Knowledge to learn:

- CRUD workflow design
- React state management
- Form editing
- Backend data validation
- Human-in-the-loop product design

### Milestone 15: Export

Problem:

The system should produce portable knowledge artifacts, not only in-app data.

Planned work:

- Export one job's cards as Markdown
- Export all cards as Markdown
- Obsidian-friendly vault layout
- Include source video and timestamps
- Include claims and evidence
- Include active recall question/answer

Example export:

```markdown
# Singular Value Decomposition

Summary...

## Claims

- Claim: SVD factors a matrix using orthogonal and diagonal structure.
  Evidence: "..."
  Source: 12:04 - 12:18

## Active Recall

Q: ...
A: ...
```

Knowledge to learn:

- File generation
- Content serialization
- Markdown schema design
- Portable knowledge assets

## Phase 2: Upgrade Cards Into a Knowledge Base

The goal of this phase is to move beyond a list of cards and create a semantic
knowledge system.

### Milestone 16: Embeddings and Similarity

Problem:

Cards are currently isolated. The system should understand when two cards are
semantically related.

Planned work:

- Generate embeddings for each card
- Add a `card_embeddings` SQLite table
- Store card vectors locally
- Implement cosine similarity search
- Show related cards in the frontend
- Detect near-duplicate generated cards
- When generating a new card, compare it with existing cards

Possible local embedding models:

- `nomic-embed-text`
- `bge-small`
- `bge-m3`

Flow:

```text
card text -> embedding model -> vector
query vector vs card vectors -> cosine similarity
```

Knowledge to learn:

- Embeddings
- Cosine similarity
- Vector search
- Semantic deduplication
- SQLite vector storage tradeoffs

### Milestone 17: Knowledge Graph / Knowledge Tree

Problem:

A real learning system should expose relationships between concepts, not only
store independent cards.

Planned work:

- Add card relationship types:

```text
prerequisite
related
example_of
contrast_with
part_of
```

- Recommend relationships using embedding similarity
- Ask the local LLM to explain card relationships
- Add manual relation editing
- Add graph or tree visualization
- Support topic clusters

Knowledge to learn:

- Graph data modeling
- Knowledge graphs
- Relation extraction
- Human-in-the-loop curation
- Concept map product design

### Milestone 18: Local RAG Assistant

Problem:

Users should be able to ask questions about their uploaded course videos.

Planned work:

- User asks a question
- Retrieve related transcript segments and cards
- Build a grounded context prompt
- Use local Qwen to answer
- Require citations with timestamps
- Say "not enough evidence" when retrieval does not support an answer

Flow:

```text
question
-> embedding search
-> retrieve cards/transcript
-> local LLM answer
-> cite evidence
```

Knowledge to learn:

- RAG
- Retrieval
- Context construction
- Citation grounding
- Local LLM assistant design

## Phase 3: Make It Research- and Resume-Ready

The goal of this phase is to make the project measurable, explainable, and easy
for others to run.

### Milestone 19: Evaluation Layer

Problem:

The project should be evaluated, not only demonstrated.

Planned work:

- Measure grounding pass rate
- Measure unsupported claim rate
- Measure generation latency
- Measure duplicate card rate
- Measure retrieval hit rate
- Add user feedback:
  - good
  - bad
  - needs edit
- Store evaluation records
- Add a small benchmark dataset

Knowledge to learn:

- LLM evaluation
- Product metrics
- Error analysis
- Regression testing for AI features

Resume angle:

> I did not only build an LLM app. I designed evaluation metrics for grounded
> knowledge generation and tracked failure modes such as unsupported claims,
> duplicate cards, and retrieval misses.

### Milestone 20: Learning Feedback Dataset

Problem:

The system should capture how users improve generated cards, so future versions
can learn from feedback.

Planned work:

- Store generated card -> edited card diffs
- Store save/delete decisions
- Store evidence clicks
- Store user feedback labels
- Build preference-style training records
- Prepare data for future prompt optimization or reward modeling

Possible future use:

```text
prompt optimization
reward modeling
card generation policy improvement
agentic learning loop
```

Knowledge to learn:

- Feedback data modeling
- Preference data
- Human correction logs
- Agentic learning loop design

### Milestone 21: Packaging, README, and Demo

Problem:

The project should be easy for another person to clone, run, and understand.

Planned work:

- Improve `.env.example`
- Add one-click start scripts
- Add setup guide for:
  - backend
  - frontend
  - Ollama
  - Qwen model
- Add architecture diagram
- Add demo video
- Add sample transcript/card data
- Add troubleshooting guide for ports and local model setup

Knowledge to learn:

- Developer experience
- Documentation
- Project packaging
- Reproducible local setup

## Suggested Next Step

The next recommended milestone is:

```text
Milestone 13: Card Generation Reliability
```

Reason:

The current system already works end-to-end, but real usage has exposed the next
important product problem:

```text
local generation is slow, and the UI does not clearly explain what is happening
```

Before adding embeddings, graph structure, or RAG, the card generation workflow
should become transparent, cancellable, and easier to debug.

## Long-Term Final Shape

The final system should feel like:

```text
local video learning workspace
-> upload course videos
-> transcribe locally
-> generate grounded knowledge cards
-> review and edit cards
-> export to Obsidian
-> discover related concepts
-> build a personal knowledge graph
-> ask grounded questions over the local course memory
-> collect feedback for future agentic improvement
```

