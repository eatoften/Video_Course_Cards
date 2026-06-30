# Video Course Cards Roadmap

Last updated: 2026-06-30

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
-> course/video/card workspace
-> card tags, review state, and user notes
-> Obsidian-friendly Markdown folder export
```

The current product capability is:

```text
local course video -> transcript -> grounded knowledge cards -> Markdown snapshot
```

Design decision:

```text
SQLite is the source of truth.
Markdown files are portable export snapshots.
```

If a user edits Markdown files in Obsidian, those edits do not automatically
sync back to SQLite. Future sync can be designed later, but the current system
keeps data ownership simple and predictable.

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

### Milestone 13: Card Generation Reliability

- Generation progress and metadata
- Local LLM timeout handling
- Context length guardrails
- Cancel generation
- Clearer error responses for failed generation
- Backend tests for slow and failed local model paths

### Milestone 14: Course/Card Workspace and Review

- Course list and video list
- Course-level card rail
- Card detail view
- Saved card edit/delete
- Delete all cards for a job or course
- Tags
- Review state
- User notes stored separately from cards
- Fixed transcript and card panel overflow issues

### Milestone 15: Export

- Export one job's cards as Markdown
- Export all cards as Markdown
- Obsidian-friendly folder layout
- Zip export retained as a portable packaging option
- Local folder export to `C:\Users\12245\Desktop\cards`
- Snapshot manifests:
  - `.vcc-job-export-manifest.json`
  - `.vcc-vault-export-manifest.json`
- Exported Markdown includes:
  - source video
  - timestamps
  - claims
  - evidence quotes
  - active recall question/answer
  - metadata

Obsidian workflow:

```text
export Markdown snapshots
-> open C:\Users\12245\Desktop\cards as an Obsidian vault
```

SQLite remains the main database. Markdown remains a portable snapshot.

## Technical Research Notes

The next phase is guided by a few established ideas:

- Text segmentation can be treated as subtopic boundary detection. The classic
  TextTiling paper frames long text as passages separated by topic shifts.
- Modern semantic chunking usually computes sentence/window embeddings and
  inserts boundaries where adjacent windows become semantically distant.
- Semantic similarity can be measured by embedding texts and computing cosine
  similarity.
- Local embedding models can be served by Ollama, keeping the project
  local-first.
- SQLite can store vectors directly as JSON/BLOB for the MVP. A SQLite vector
  extension such as `sqlite-vec` can be considered later if brute-force search
  becomes too slow.
- RAG should retrieve explicit non-parametric memory, not rely only on the
  model's parameters.
- GraphRAG-style systems use extraction, graph structure, community/topic
  hierarchy, and summarization to answer broader questions over a corpus.

Reference anchors:

- TextTiling: https://aclanthology.org/J97-1003/
- LlamaIndex semantic splitter:
  https://developers.llamaindex.ai/python/examples/node_parsers/semantic_chunking/
- SentenceTransformers semantic textual similarity:
  https://sbert.net/docs/sentence_transformer/usage/semantic_textual_similarity.html
- Ollama embedding models:
  https://ollama.com/blog/embedding-models
- `nomic-embed-text`:
  https://ollama.com/library/nomic-embed-text
- `sqlite-vec`: https://github.com/asg017/sqlite-vec
- RAG paper: https://arxiv.org/abs/2005.11401
- Microsoft GraphRAG:
  https://microsoft.github.io/graphrag/

## Next Development Plan

## Phase 2: Course-Scale Semantic Card Pipeline

The goal of this phase is to move from manual card generation over a selected
transcript range to automated course-scale card generation.

The intended pipeline is:

```text
course videos
-> transcripts
-> semantic transcript chunks
-> automated grounded card generation
-> card/chunk embeddings
-> similarity and deduplication
-> knowledge graph/tree
-> grounded local RAG assistant
```

### Milestone 16: Transcript Semantic Segmentation

Problem:

To build useful card similarity and a knowledge graph, the project first needs
stable semantic units. Raw Whisper segments are too small and arbitrary, while
entire lectures are too large for reliable card generation.

Planned work:

- Add a `transcript_chunks` SQLite table:
  - `id`
  - `course_id`
  - `job_id`
  - `start_seconds`
  - `end_seconds`
  - `text`
  - `segment_ids`
  - `chunk_index`
  - `chunker_version`
  - `created_at`
- Build a transcript chunking service:
  - start from timestamped transcript segments
  - merge tiny Whisper segments into sentence/window candidates
  - compute embeddings for neighboring windows
  - detect topic shifts by cosine distance
  - enforce minimum and maximum chunk length
  - preserve segment ids and timestamps
- Add APIs:
  - `POST /jobs/{job_id}/chunks`
  - `GET /jobs/{job_id}/chunks`
  - `POST /courses/{course_id}/chunks`
- Add a frontend chunk review panel:
  - list chunks for a video
  - show timestamps
  - jump video to chunk start
  - allow regenerating chunks
- Keep a deterministic fallback chunker:
  - duration-based windows
  - max character limit
  - overlap between chunks

Initial algorithm:

```text
transcript segments
-> normalize text
-> sentence/window candidates
-> local embeddings
-> adjacent cosine distance
-> boundaries above threshold
-> merge/split to fit min/max length
-> timestamped transcript_chunks
```

Knowledge to learn:

- Text segmentation
- Semantic chunking
- Embedding-based boundary detection
- Timestamp-preserving NLP pipelines
- Chunk quality debugging

### Milestone 17: Automated Course Card Generation

Problem:

Generating cards manually from selected transcript spans does not scale to an
entire course. The system should automatically create grounded cards from
semantic transcript chunks.

Planned work:

- Generate cards from `transcript_chunks`
- Add a generation run table:
  - `card_generation_runs`
  - chunk/job/course scope
  - model
  - status
  - started/completed timestamps
  - error message
  - generation metadata
- Link cards to source chunks:
  - `source_chunk_id`
  - or `knowledge_card_sources` for many-to-one evidence
- Add batch generation APIs:
  - `POST /jobs/{job_id}/cards/generate-from-chunks`
  - `POST /courses/{course_id}/cards/generate-from-chunks`
- Use existing claim-level grounding for each generated card
- Add retries for failed chunks
- Skip chunks that are too short or low-information
- Add course-level progress UI:
  - chunks pending
  - chunks generating
  - chunks completed
  - chunks failed
- Save generated cards as drafts for human review

Card generation policy:

```text
one semantic chunk
-> ask local LLM for 0-3 cards
-> verify claims against chunk transcript
-> save grounded cards
-> mark weak chunks as no_card
```

Knowledge to learn:

- Batch processing
- Idempotent generation
- Background jobs
- LLM workflow orchestration
- Grounded generation at course scale

### Milestone 18: Embeddings and Similarity

Problem:

After a course has enough cards, the system should understand which concepts are
similar, duplicated, or complementary.

Planned work:

- Generate embeddings for:
  - `knowledge_cards`
  - `transcript_chunks`
- Add embedding tables:
  - `card_embeddings`
  - `transcript_chunk_embeddings`
- Store vectors locally:
  - MVP: JSON or BLOB in SQLite
  - Later: evaluate `sqlite-vec`
- Implement cosine similarity search
- Show related cards in the frontend
- Detect near-duplicate generated cards
- When generating a new card, compare it with existing cards
- Add similarity thresholds:
  - duplicate candidate
  - strongly related
  - weakly related
- Add a small similarity debug page:
  - card text used for embedding
  - top-k neighbors
  - cosine scores
  - source videos and timestamps

Possible local embedding models:

- `nomic-embed-text`
- `nomic-embed-text-v2-moe`
- `bge-small`
- `bge-m3`
- `qwen3-embedding`

Flow:

```text
card/chunk text
-> local embedding model
-> vector
-> SQLite storage
-> cosine similarity
-> related cards / dedupe candidates
```

Knowledge to learn:

- Embeddings
- Cosine similarity
- Vector search
- Semantic deduplication
- SQLite vector storage tradeoffs

### Milestone 19: Knowledge Graph / Knowledge Tree

Problem:

A real learning system should expose relationships between concepts, not only
store independent cards. Similarity is not enough: the system should distinguish
"prerequisite" from "example", "contrast", and "part of".

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
- Store relationship provenance:
  - embedding score
  - LLM explanation
  - user confirmation
- Add topic cluster summaries
- Add prerequisite-path view:

```text
basic concept -> supporting idea -> advanced card
```

Knowledge to learn:

- Graph data modeling
- Knowledge graphs
- Relation extraction
- Human-in-the-loop curation
- Concept map product design

### Milestone 20: Local RAG Assistant

Problem:

Users should be able to ask questions about their uploaded course videos.

Planned work:

- User asks a question
- Retrieve related transcript segments and cards
- Build a grounded context prompt
- Use local Qwen to answer
- Require citations with timestamps
- Say "not enough evidence" when retrieval does not support an answer
- Use both vector retrieval and graph context:
  - similar transcript chunks
  - relevant cards
  - neighboring graph nodes
- Show answer citations:
  - video
  - timestamp
  - card title
  - evidence quote

Flow:

```text
question
-> query embedding
-> retrieve transcript_chunks + cards
-> expand with graph neighbors
-> local LLM answer
-> cite evidence
```

Knowledge to learn:

- RAG
- Retrieval
- Context construction
- Citation grounding
- Local LLM assistant design
- Retrieval evaluation

## Phase 3: Make It Research- and Resume-Ready

The goal of this phase is to make the project measurable, explainable, and easy
for others to run.

### Milestone 21: Evaluation Layer

Problem:

The project should be evaluated, not only demonstrated.

Planned work:

- Measure grounding pass rate
- Measure unsupported claim rate
- Measure generation latency
- Measure duplicate card rate
- Measure retrieval hit rate
- Measure chunk quality:
  - average chunk duration
  - boundary quality sample review
  - chunks with too little information
- Measure graph quality:
  - accepted/rejected relationship suggestions
  - duplicate edge rate
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
> duplicate cards, poor transcript chunks, graph relation errors, and retrieval
> misses.

### Milestone 22: Learning Feedback Dataset

Problem:

The system should capture how users improve generated cards, so future versions
can learn from feedback.

Planned work:

- Store generated card -> edited card diffs
- Store save/delete decisions
- Store evidence clicks
- Store chunk boundary edits
- Store accepted/rejected similarity suggestions
- Store accepted/rejected graph edges
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

### Milestone 23: Packaging, README, and Demo

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
- Add sample course dataset
- Add troubleshooting guide for ports and local model setup

Knowledge to learn:

- Developer experience
- Documentation
- Project packaging
- Reproducible local setup

## Suggested Next Step

The next recommended milestone is:

```text
Milestone 16: Transcript Semantic Segmentation
```

Reason:

The current system can generate grounded cards from selected transcript spans.
To build similarity, graph structure, and RAG, it now needs stable semantic
units across an entire course.

```text
course videos -> transcripts -> semantic chunks -> automated grounded cards
```

Only after the transcript chunking and automated card generation layers exist
will card similarity have enough high-quality material to work with.

## Long-Term Final Shape

The final system should feel like:

```text
local video learning workspace
-> upload a full course worth of videos
-> transcribe locally
-> segment transcripts into semantic chunks
-> automatically generate grounded knowledge cards
-> review and edit cards
-> export Markdown snapshots to Obsidian
-> compute card/chunk similarity
-> detect duplicate and related concepts
-> build a course knowledge graph/tree
-> ask grounded questions over the local course memory
-> collect feedback for future agentic improvement
```
