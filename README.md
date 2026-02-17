# Rutoken Docs Mirror + AI Assistant

## Local run

1. Copy `.env.example` to `.env` (or `.env.txt`) and fill keys.
2. Start server:

```powershell
.\run_local.ps1
```

3. Open `http://127.0.0.1:8080`.

## What is included

- Static documentation portal copy (`index.html`).
- Promptline in top header (instead of search input).
- Modal AI assistant chat over page.
- Backend API:
  - `POST /api/assistant` for RAG answers.
  - `GET /health` for healthcheck.

## Railway deploy

Repository is ready for Railway Docker deploy:

- `Dockerfile`
- `.dockerignore`
- `railway.json` (healthcheck: `/health`)

Set these Railway Variables:

- `OPENAI_API_KEY`
- `QDRANT_URL`
- `QDRANT_API_KEY`
- `QDRANT_COLLECTION`

Optional variables:

- `OPENAI_CHAT_MODEL` (default: `gpt-4o-mini`)
- `EMBED_MODEL` (default: `text-embedding-3-large`)
- `RAG_TOP_K` (default: `6`)
- `RAG_SNIPPET_CHARS` (default: `700`)

Railway provides `PORT` automatically. App starts with:

`uvicorn backend.app:app --host 0.0.0.0 --port ${PORT}`
