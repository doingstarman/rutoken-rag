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

