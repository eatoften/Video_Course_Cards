# Local LLM Setup

Video Course Cards talks to local language models through an
OpenAI-compatible HTTP API. The application backend calls the model; the
browser never calls Ollama, LM Studio, or vLLM directly.

```text
React UI -> FastAPI backend -> LLM adapter -> Ollama / LM Studio / vLLM
```

## Default: Ollama

Install Ollama, then pull a Qwen model:

```bash
ollama pull qwen3:4b
```

For better card quality on a stronger machine:

```bash
ollama pull qwen3:8b
```

Copy the backend environment example:

```bash
cd backend
copy .env.example .env
```

Default settings:

```env
VCC_LLM_PROVIDER=ollama
VCC_LLM_BASE_URL=http://localhost:11434/v1
VCC_LLM_MODEL=qwen3:4b
VCC_LLM_API_KEY=local
VCC_LLM_TEMPERATURE=0.2
VCC_LLM_MAX_TOKENS=2048
VCC_LLM_TIMEOUT_SECONDS=120
```

Start the backend:

```bash
uv run python -B -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Check the connection:

```bash
curl http://127.0.0.1:8000/llm/status
```

## LM Studio

In LM Studio:

1. Download a Qwen 4B or 8B GGUF model.
2. Load the model.
3. Open the Local Server panel.
4. Start the OpenAI-compatible server.

Then set:

```env
VCC_LLM_PROVIDER=lmstudio
VCC_LLM_BASE_URL=http://localhost:1234/v1
VCC_LLM_MODEL=your-loaded-model-name
VCC_LLM_API_KEY=local
```

## vLLM

For GPU-oriented local or server deployment:

```bash
vllm serve Qwen/Qwen3-8B --port 8002
```

Then set:

```env
VCC_LLM_PROVIDER=vllm
VCC_LLM_BASE_URL=http://localhost:8002/v1
VCC_LLM_MODEL=Qwen/Qwen3-8B
VCC_LLM_API_KEY=local
```

## API Endpoints

```text
GET  /llm/status
POST /cards/draft
```

`POST /cards/draft` expects a transcript time window and returns structured
knowledge cards. The service asks Qwen to return JSON only, strips possible
`<think>` blocks, extracts JSON from Markdown fences when needed, validates the
result with Pydantic, and retries once with a repair prompt if parsing fails.
