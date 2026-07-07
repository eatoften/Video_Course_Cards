<h1 align="center">Video Course Cards</h1>

<p align="center">
  Turn long course videos into timestamp-grounded knowledge cards, local card memory, and Obsidian-friendly Markdown.
</p>

<p align="center">
  <a href="https://github.com/eatoften/Video_Course_Cards/releases/latest">Download</a>
  |
  <a href="docs/tauri-desktop.md">Desktop build</a>
  |
  <a href="docs/local-llm.md">Local LLM setup</a>
  |
  <a href="docs/roadmap.md">Roadmap</a>
  |
  <a href="docs/rag-roadmap.md">RAG plan</a>
</p>

<p align="center">
  <img alt="Python 3.11" src="https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white">
  <img alt="FastAPI" src="https://img.shields.io/badge/FastAPI-backend-009688?logo=fastapi&logoColor=white">
  <img alt="React" src="https://img.shields.io/badge/React-TypeScript-61DAFB?logo=react&logoColor=black">
  <img alt="Tauri" src="https://img.shields.io/badge/Tauri-desktop-FFC131?logo=tauri&logoColor=black">
  <img alt="SQLite" src="https://img.shields.io/badge/SQLite-local%20first-003B57?logo=sqlite&logoColor=white">
</p>

## Overview

Video Course Cards is a local-first AI learning workspace for lecture videos. It turns a video into a transcript, cuts the transcript into semantic chunks, drafts grounded knowledge cards with a local LLM, stores everything in SQLite, embeds cards for retrieval, and exports portable Markdown snapshots.

The project is not trying to be another generic "chat with your transcript" demo. The core object is a **claim-grounded knowledge card**: a structured learning unit whose claims point back to transcript evidence and timestamps. And the future plan is to turn these cards in to a graph which can serve as an external world model that a controller can plan over, while a decoder can take the plan and graph as an input to generate answers.

```text
video -> transcript -> semantic chunks -> grounded cards -> card embeddings -> retrieval -> Markdown export
```

SQLite is the source of truth. Markdown is an export format.

## Why This Exists

Long technical lectures contain more than raw transcript text. A useful learning system should preserve where an idea came from, what evidence supports it, how it connects to other cards, and how the user later edits or rejects it.

This repository explores that pipeline as a local desktop application:

- **Grounded generation**: cards keep claims, evidence, and source timestamps.
- **Local-first storage**: videos, transcripts, cards, embeddings, and notes stay on the user's machine.
- **Structured memory**: cards are JSON/SQLite records before they become Markdown.
- **Retrieval baseline**: card embeddings support ordinary dense retrieval before more advanced graph-guided methods.
- **Portable output**: exports are Obsidian-friendly Markdown snapshots.

## Current Demo

The current demo runs on Windows as a Tauri desktop app with a packaged FastAPI sidecar.

It can:

- upload local videos;
- validate media with ffprobe;
- extract audio with FFmpeg;
- transcribe with faster-whisper;
- show transcript segments next to the course workspace;
- create semantic transcript chunks with Sentence Transformer embeddings;
- generate cards manually from selected transcript text or automatically from chunks;
- save, edit, delete, tag, and review cards;
- attach user notes to cards;
- embed cards and run dense card retrieval;
- export one job or all cards as Markdown folders;
- check local runtime dependencies such as FFmpeg, Ollama/Qwen, and embedding models.

Still rough:

- the installer is not code-signed;
- Windows is the only packaged target currently exercised;
- Ollama, Qwen, FFmpeg, and model caches are user-installed dependencies;
- RAG currently retrieves cards, but answer generation with citations is still planned;
- exported Markdown does not sync edits back into SQLite.

## Install

Download the latest Windows installer from:

```text
https://github.com/eatoften/Video_Course_Cards/releases/latest
```

The installer includes:

- Tauri desktop shell;
- React UI;
- packaged FastAPI backend;
- SQLite schema and app code.

The installer does **not** bundle large model assets. Install local AI dependencies separately:

```powershell
ollama pull qwen3:4b
```

Desktop data is stored under:

```text
C:\Users\<user>\AppData\Local\Video Course Cards\
```

See [docs/local-llm.md](docs/local-llm.md) for local model configuration.

## Developer Setup

Clone the repository, then install backend and frontend dependencies.

```powershell
git clone https://github.com/eatoften/Video_Course_Cards.git
cd Video_Course_Cards
```

Run the backend:

```powershell
cd backend
$env:PYTHONUTF8='1'
$env:PYTHONDONTWRITEBYTECODE='1'
uv run python -B -m uvicorn app.main:app --host 127.0.0.1 --port 8001 --reload
```

Run the frontend:

```powershell
cd frontend
npm.cmd install
npm.cmd run dev
```

Open:

```text
http://127.0.0.1:5174
```

## Desktop Build

Tauri requires Rust/Cargo and the Visual Studio C++ build tools on Windows.

Build the backend sidecar:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\build-desktop-backend.ps1
```

Run the desktop app in development:

```powershell
cd frontend
npm.cmd run tauri:dev
```

Build the Windows installer:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\build-windows-installer.ps1
```

Output:

```text
frontend\src-tauri\target\release\bundle\nsis\Video Course Cards_0.1.0_x64-setup.exe
```

GitHub Actions can build and attach the installer to a tag release:

```powershell
git tag v0.1.0
git push origin v0.1.0
```

See [docs/tauri-desktop.md](docs/tauri-desktop.md).

## Architecture

```mermaid
flowchart LR
    subgraph Desktop["Tauri desktop"]
        UI["React / TypeScript"]
        RUST["Rust shell"]
    end

    subgraph Backend["FastAPI sidecar"]
        API["HTTP routes"]
        JOBS["Job service"]
        PIPELINE["Video pipeline"]
        CARDS["Card services"]
        RAG["Retrieval services"]
    end

    subgraph LocalAI["Local AI runtime"]
        FFMPEG["FFmpeg / ffprobe"]
        WHISPER["faster-whisper"]
        EMB["Sentence Transformers"]
        LLM["Ollama / Qwen"]
    end

    subgraph Storage["Local data"]
        DB["SQLite"]
        FILES["uploads / transcripts / exports"]
    end

    UI <--> API
    RUST --> API
    API --> JOBS
    JOBS --> PIPELINE
    PIPELINE --> FFMPEG
    PIPELINE --> WHISPER
    CARDS --> LLM
    RAG --> EMB
    API <--> DB
    API <--> FILES
```

The backend is deliberately split by responsibility:

| Layer | Responsibility |
| --- | --- |
| `main.py` | HTTP routes and response mapping |
| `job_service.py` | video job orchestration |
| `job_store.py` | SQLite CRUD for jobs |
| `video_pipeline.py` | media probe, audio extraction, transcription |
| `transcript_chunker.py` | semantic transcript chunking |
| `knowledge_card_service.py` | card persistence and updates |
| `card_embedding_service.py` | card text -> embedding workflow |
| `rag_service.py` | card retrieval baseline |
| `desktop_server.py` | packaged backend sidecar entrypoint |

## Knowledge Cards

A card is stored as structured data, not just markdown text.

```json
{
  "title": "Singular Value Decomposition",
  "summary": "SVD factors a matrix into orthogonal and diagonal structure.",
  "tags": ["linear algebra", "matrix factorization"],
  "source_start_seconds": 724.0,
  "source_end_seconds": 738.0,
  "claims": [
    {
      "text": "SVD decomposes a matrix using orthogonal and diagonal components.",
      "evidence": [
        {
          "text": "called the singular value decomposition",
          "start_seconds": 724.0,
          "end_seconds": 738.0
        }
      ]
    }
  ],
  "question": "What structure does SVD use to factor a matrix?",
  "answer": "It uses orthogonal matrices and a diagonal matrix."
}
```

This shape makes later work possible: duplicate detection, related-card search, graph edges, citation-aware RAG, and feedback-based evaluation.

## API Surface

Selected endpoints:

| Endpoint | Purpose |
| --- | --- |
| `POST /videos` | upload and register a local video |
| `POST /jobs/{job_id}/run` | run probe -> audio -> transcription |
| `GET /jobs/{job_id}/transcript` | return timestamped transcript segments |
| `POST /jobs/{job_id}/chunks` | generate semantic transcript chunks |
| `POST /jobs/{job_id}/cards/auto-generate` | generate cards from chunks |
| `GET /jobs/{job_id}/cards` | list cards for one video job |
| `PATCH /cards/{card_id}` | edit a saved card |
| `POST /cards/{card_id}/embedding` | embed one card |
| `POST /courses/{course_id}/card-embeddings` | embed all cards in a course |
| `POST /rag/retrieve` | retrieve relevant cards for a question |
| `POST /jobs/{job_id}/cards/export/markdown/folder` | export one job as Markdown |
| `POST /cards/export/markdown/folder` | export all cards as Markdown |
| `GET /runtime/status` | inspect local runtime dependencies |

## Tests

Backend:

```powershell
cd backend
uv run pytest
```

Frontend:

```powershell
cd frontend
npm.cmd run build
```

Tauri:

```powershell
cd frontend\src-tauri
cargo check
```

Sidecar smoke test:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\test-desktop-backend.ps1
```

## Roadmap

Near term:

- improve automatic card generation reliability;
- improve semantic chunk boundary quality;
- detect duplicate or near-duplicate cards with embeddings;
- show related cards in the UI;
- turn card retrieval into a citation-grounded answer assistant;
- add evaluation records for latency, unsupported claims, duplicates, and retrieval misses.

Longer term:

- build a card similarity graph;
- add relation types such as `prerequisite`, `example_of`, `contrast_with`, and `part_of`;
- support human-in-the-loop graph editing;
- compare ordinary dense RAG against graph-guided retrieval;
- use user edits and save/delete decisions as a feedback dataset for future agentic learning loops.

## Project Principles

- Local data should stay local by default.
- Claims should be traceable to evidence.
- SQLite should remain the durable source of truth.
- Markdown should be portable, inspectable, and tool-friendly.
- Advanced AI features should be compared against simple baselines.
- User corrections should become evaluation data before they become training data.

## License

To be determined before the first public release.
