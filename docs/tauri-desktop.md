# Tauri Desktop Packaging Roadmap

This project is moving toward a local-first desktop app:

```text
Tauri desktop window
-> React UI
-> local FastAPI backend
-> local SQLite, uploads, transcripts, embeddings, and Ollama
```

The long-term goal is a normal Windows app where user videos and knowledge data stay on the user's machine.

## Milestone T1: Tauri Shell

Current status: implemented as a shell.

What exists now:

- `frontend/src-tauri/tauri.conf.json`
- `frontend/src-tauri/Cargo.toml`
- `frontend/src-tauri/src/main.rs`
- `frontend/src-tauri/src/lib.rs`
- `frontend/package.json` scripts:
  - `npm run tauri:dev`
  - `npm run tauri:build`

T1 only created the shell. T2 now starts FastAPI automatically through a packaged sidecar.

## T1 Run Commands

Terminal 1:

```powershell
cd C:\Users\12245\Desktop\Video_Course_Cards\backend
$env:PYTHONUTF8='1'
$env:PYTHONDONTWRITEBYTECODE='1'
uv run python -B -m uvicorn app.main:app --host 127.0.0.1 --port 8001
```

Terminal 2:

```powershell
cd C:\Users\12245\Desktop\Video_Course_Cards\frontend
npm.cmd run tauri:dev
```

## Required Tauri Toolchain

Tauri needs native Windows build tools:

- Rust / Cargo through rustup
- Microsoft Visual Studio Build Tools with MSVC and Windows SDK
- WebView2 Runtime
- Node.js and npm

Check local diagnosis with:

```powershell
cd C:\Users\12245\Desktop\Video_Course_Cards\frontend
npm.cmd exec tauri info
```

Install Rust from:

```text
https://rustup.rs/
```

Install Visual Studio Build Tools from:

```text
https://aka.ms/vs/17/release/vs_BuildTools.exe
```

In the Visual Studio installer, select the C++ desktop build tools workload.

After installing, restart PowerShell and run:

```powershell
cd C:\Users\12245\Desktop\Video_Course_Cards\frontend
npm.cmd exec tauri info
npm.cmd run tauri:dev
```

## Milestone T2: Backend Sidecar

### T2.1: Desktop Backend Entry and Build Script

Current status: implemented.

What exists now:

- `backend/app/desktop_server.py`
- `scripts/build-desktop-backend.ps1`
- `frontend/src-tauri/binaries/.gitkeep`
- `pyinstaller` in the backend dev dependency group

The desktop backend entry starts the same FastAPI app used during development:

```text
app.main:app
```

It also checks whether the configured backend is already healthy:

```text
GET http://127.0.0.1:8001/health
```

If the backend is already running, it exits successfully instead of trying to bind the same port again.

Run the desktop backend entry directly:

```powershell
cd C:\Users\12245\Desktop\Video_Course_Cards\backend
uv run python -m app.desktop_server
```

Build the backend executable:

```powershell
cd C:\Users\12245\Desktop\Video_Course_Cards
.\scripts\build-desktop-backend.ps1
```

If PowerShell blocks local scripts, run it with a one-command bypass:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\build-desktop-backend.ps1
```

This build can be slow because the backend imports ML dependencies such as faster-whisper and sentence-transformers.

Expected outputs:

```text
backend/dist/desktop/video-course-cards-backend.exe
frontend/src-tauri/binaries/video-course-cards-backend-x86_64-pc-windows-msvc.exe
```

The second file name follows Tauri's sidecar naming convention. T2.2 will wire Tauri to start this sidecar automatically.

### T2.2: Tauri Sidecar Process Management

Current status: implemented.

What exists now:

- `frontend/src-tauri/src/backend.rs`
- `scripts/start-tauri-frontend-dev.ps1`
- Tauri commands:
  - `ensure_backend`
  - `get_backend_status`
  - `restart_backend`
  - `stop_backend`
- `tauri-plugin-shell` in `frontend/src-tauri/Cargo.toml`
- `externalBin` in `frontend/src-tauri/tauri.conf.json`
- React backend readiness gate in `frontend/src/App.tsx`

Startup flow:

```text
React app starts
-> checks GET http://127.0.0.1:8001/health
-> if healthy, continue into the main UI
-> if not healthy and running inside Tauri, invoke ensure_backend
-> Rust side starts backend sidecar
-> React waits for /health
-> main UI loads courses, jobs, cards, and LLM state
```

Port strategy:

```text
If /health is OK:
  Reuse the existing backend.

If /health is not OK and port 8001 is free:
  Start the packaged backend sidecar.

If /health is not OK and port 8001 is occupied:
  Find the listening PID with netstat.
  Close it with taskkill.
  Start the packaged backend sidecar.
```

Run in desktop dev mode:

```powershell
cd C:\Users\12245\Desktop\Video_Course_Cards
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\build-desktop-backend.ps1

cd C:\Users\12245\Desktop\Video_Course_Cards\frontend
npm.cmd run tauri:dev
```

In T2.2, manually starting FastAPI first is optional. If a compatible backend is already running, the desktop app reuses it. If not, Tauri starts the sidecar.

Manually starting Vite first is also optional. `scripts/start-tauri-frontend-dev.ps1` checks `http://127.0.0.1:5174`; if it is already running, Tauri reuses it. If not, the script starts Vite.

Useful checks:

```powershell
Invoke-WebRequest -UseBasicParsing http://127.0.0.1:8001/health
Get-NetTCPConnection -LocalAddress 127.0.0.1 -LocalPort 8001
```

Implementation responsibility split:

```text
React:
  Shows startup state.
  Calls Tauri ensure_backend when available.
  Waits for /health before loading app data.

Tauri/Rust:
  Owns process management.
  Starts the backend sidecar.
  Reuses healthy existing backend.
  Closes non-compatible process occupying the backend port.

FastAPI:
  Still owns all API, SQLite, transcription, cards, embeddings, and RAG logic.
```

### T2.3: Sidecar Packaging Stabilization

Status: implemented.

Goal:

Make the packaged FastAPI sidecar reliable enough to ship inside a lightweight desktop installer.

Current observation:

```text
frontend/src-tauri/binaries/video-course-cards-backend-x86_64-pc-windows-msvc.exe
```

is about 286 MB after PyInstaller. This is acceptable for an early local AI app, but it should not also bundle large model files.

Lightweight installer boundary:

```text
Bundled:
  Tauri app
  React UI
  FastAPI backend sidecar
  SQLite schema initialization code
  Runtime checks and setup instructions

Not bundled in Strategy A:
  Ollama
  Qwen model files
  Whisper model files if they can be downloaded or cached locally
  sentence-transformer model snapshots
  User videos
  User SQLite database
  User exports
```

Implemented work:

- Add backend sidecar log output.
  - `backend/app/desktop_server.py` accepts `--log-file`.
  - Tauri writes logs under `%LOCALAPPDATA%\Video Course Cards\logs\backend.log`.
- Add a `--desktop` or environment marker for sidecar mode.
  - `--desktop`
  - `VCC_DESKTOP=1`
- Keep model assets external.
  - Do not bundle Qwen.
  - Do not bundle Ollama.
  - Do not bundle local Hugging Face model snapshots in the installer.
- Review PyInstaller hidden imports.
  - Keep only required imports.
  - Avoid accidentally pulling test/dev packages.
  - Confirm core app endpoints work from the sidecar.
- Add a sidecar smoke-test script.
  - `scripts/test-desktop-backend.ps1`
  - Starts the sidecar on port `8765`.
  - Waits for `/health`.
  - Stops the process.
- Confirm packaging outputs.
  - `backend/dist/desktop/video-course-cards-backend.exe`
  - `frontend/src-tauri/binaries/video-course-cards-backend-x86_64-pc-windows-msvc.exe`

Smoke test:

```powershell
cd C:\Users\12245\Desktop\Video_Course_Cards
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\test-desktop-backend.ps1
```

Acceptance criteria:

- Backend sidecar starts without a manually activated Python environment.
- `/health` returns OK from the packaged sidecar.
- Tauri can start the sidecar when no backend is running.
- Tauri reuses a healthy existing backend.
- Sidecar logs are available when startup fails.
- Installer size stays in a reasonable early-demo range by keeping model files external.

## Milestone T3: App Data Directory

Status: implemented.

Goal:

Move all user data out of the repository and into a normal per-user application data directory.

Why this matters:

The current development layout stores important data under:

```text
backend/data/
```

That is fine while developing from the repo, but it is wrong for an installed desktop app. Installed apps should keep mutable user data outside the app installation folder.

Target Windows layout:

```text
C:\Users\<user>\AppData\Local\Video Course Cards\
  data\
    jobs.db
  uploads\
  transcripts\
  exports\
  logs\
```

Implemented work:

- Add `VCC_DATA_DIR`.
  - Environment variable override for development and tests.
  - In desktop mode, defaults to `%LOCALAPPDATA%\Video Course Cards`.
  - In source-development mode, defaults to `backend/data`.
- Add a backend path settings layer.
  - `backend/app/settings.py`
  - Centralizes `data_dir`, `db_path`, `upload_dir`, `transcript_dir`, `export_dir`, and `log_dir`.
- Move SQLite path configuration.
  - `backend/app/db.py` now defaults to the configured DB path.
  - Tests should still be able to inject temporary DB paths.
- Move upload/transcript/export paths.
  - `backend/app/main.py` uses configured upload path.
  - `backend/app/export_service.py` uses configured export path.
- Set `VCC_DATA_DIR` from Tauri sidecar startup.
  - `frontend/src-tauri/src/backend.rs` passes AppData into the sidecar.
- Add migration behavior.
  - First version can be simple: new installed app starts with a clean AppData DB.
  - Later version can offer "Import existing dev data".

Relevant config in `backend/.env.example`:

```env
VCC_DATA_DIR=
VCC_DB_PATH=
VCC_UPLOAD_DIR=
VCC_TRANSCRIPT_DIR=
VCC_EXPORT_DIR=
VCC_LOG_DIR=
VCC_BACKEND_LOG_FILE=
VCC_DESKTOP=false
```

Acceptance criteria:

- Running from source still works.
- Running as Tauri desktop app stores SQLite/uploads/transcripts/exports outside the repo.
- Deleting or moving the repo does not delete installed-user data.
- Tests still use isolated temporary data.

## Milestone T4: Local Runtime Check

Status: implemented.

Goal:

Make first-run setup understandable for normal users without bundling huge model/runtime assets.

Strategy A means the installer stays lightweight. Therefore the app must clearly show which external local tools are missing and how to install them.

Runtime dependencies to check:

```text
FFmpeg / ffprobe
Ollama server
Selected Qwen model
Embedding model path or cache
Whisper/faster-whisper model availability
GPU availability as optional information
```

Implemented backend endpoints:

```text
GET /runtime/status
POST /runtime/check
```

Possible response shape:

```json
{
  "ffmpeg": {
    "available": true,
    "version": "..."
  },
  "ffprobe": {
    "available": true,
    "version": "..."
  },
  "ollama": {
    "available": false,
    "base_url": "http://localhost:11434"
  },
  "llm_model": {
    "available": false,
    "model": "qwen3:4b",
    "install_hint": "ollama pull qwen3:4b"
  },
  "embedding_model": {
    "available": true,
    "model": "sentence-transformers/all-MiniLM-L6-v2",
    "local_files_only": true
  }
}
```

Implemented frontend work:

- Add a runtime setup panel in the existing app layout.
- Show checklist rows:
  - FFmpeg
  - ffprobe
  - Ollama
  - Qwen model
  - Embedding model
  - Whisper model/cache
- For missing dependencies, show copyable commands.
- Do not block the whole app if optional dependencies are missing.
  - Upload/list/export can work without Ollama.
  - Card generation needs Ollama/Qwen.
  - Transcription needs FFmpeg and Whisper runtime.
  - RAG retrieval needs embeddings.
- Add "Re-check" button.

Example user guidance:

```powershell
ollama pull qwen3:4b
```

Acceptance criteria:

- A new user can open the app and understand what is missing.
- Missing Ollama/Qwen produces an actionable setup message, not a vague crash.
- The app distinguishes required vs optional capabilities.
- Runtime status can be refreshed without restarting the app.

## Milestone T5: Lightweight Windows Installer

Status: implemented.

Goal:

Produce a GitHub Release artifact that normal users can download and install like a desktop app.

Expected user flow:

```text
1. Open GitHub Releases.
2. Download Video Course Cards installer.
3. Run installer.
4. Launch Video Course Cards from Start Menu/Desktop.
5. App opens local UI.
6. App starts local FastAPI sidecar automatically.
7. First-run setup checklist explains Ollama/Qwen/FFmpeg/model requirements.
8. User installs missing local tools if needed.
9. User uploads videos and builds local cards.
```

Implemented work:

- Added `scripts/build-windows-installer.ps1`.
- Default installer format: NSIS `.exe`.
- Optional MSI build path remains available through the script parameter.
- Added GitHub Actions workflow:
  - `.github/workflows/windows-desktop-release.yml`
  - `workflow_dispatch` builds an installer artifact manually.
  - pushing a `v*` tag builds and attaches the installer to the GitHub Release.
- Add app icon.
  - `frontend/src-tauri/icons/icon.ico`
- Add app metadata.
  - Product name.
  - Version.
- Ensure sidecar is included.
  - The built backend exe must exist before `tauri build`.
  - Build script should fail early if sidecar is missing.
- Add release script.
  - `scripts/publish-github-release.ps1`
- Add README installation section.
  - "For users"
  - "For developers"
  - "Where your data is stored"

Build command:

```powershell
cd C:\Users\12245\Desktop\Video_Course_Cards
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\build-windows-installer.ps1
```

If the backend sidecar is already freshly built:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\build-windows-installer.ps1 -SkipBackendBuild
```

Current installer output:

```text
frontend\src-tauri\target\release\bundle\nsis\Video Course Cards_0.1.0_x64-setup.exe
```

Publish a draft GitHub Release:

```powershell
cd C:\Users\12245\Desktop\Video_Course_Cards
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\publish-github-release.ps1 -Tag v0.1.0
```

This requires GitHub CLI:

```powershell
gh auth login
```

After publishing the draft release on GitHub, users can download the installer from the repository's Releases page and double-click it like a normal Windows app installer.

Automatic GitHub Release path:

```powershell
git tag v0.1.0
git push origin v0.1.0
```

The `Windows Desktop Release` workflow builds the installer on GitHub's Windows runner and uploads the installer to the tag's Release page.

Acceptance criteria:

- A user can install the app without cloning the repo.
- App launches from the Windows Start Menu.
- App starts the backend sidecar automatically.
- User data is created under AppData, not inside the install directory.
- Missing local AI dependencies are shown in the setup checklist.
- README explains that videos/cards remain local.

## Strategy A Summary

The first distributable version should be a lightweight local-first installer:

```text
Installer includes:
  Desktop app shell
  React UI
  FastAPI sidecar
  SQLite schema code
  Setup checklist

User installs separately:
  Ollama
  Qwen model
  Optional local model caches
```

This avoids a multi-GB installer while preserving the main product promise:

```text
User videos, transcripts, cards, embeddings, and notes stay on the user's machine.
```
