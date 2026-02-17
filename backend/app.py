from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from .rag_service import RagService


ROOT_DIR = Path(__file__).resolve().parents[1]
INDEX_HTML = ROOT_DIR / "index.html"
PORTAL_HTML = ROOT_DIR / "rutoken_portal.html"

app = FastAPI(title="Rutoken Docs AI Assistant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

rag = RagService()


class ChatMessage(BaseModel):
    role: str = Field(pattern=r"^(user|assistant)$")
    content: str = Field(min_length=1, max_length=6000)


class AssistantRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    history: list[ChatMessage] = Field(default_factory=list)
    top_k: int | None = Field(default=None, ge=1, le=12)


class SourceItem(BaseModel):
    title: str
    url: str | None = None
    doc_path: str | None = None
    section: str | None = None
    score: float
    snippet: str


class AssistantResponse(BaseModel):
    answer: str
    sources: list[SourceItem]
    followups: list[str] = Field(default_factory=list)
    answer_id: str


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/")
def root() -> FileResponse:
    return FileResponse(INDEX_HTML)


@app.get("/rutoken_portal.html")
def portal_html() -> FileResponse:
    return FileResponse(PORTAL_HTML)


@app.post("/api/assistant", response_model=AssistantResponse)
def assistant(payload: AssistantRequest) -> AssistantResponse:
    try:
        answer, sources, followups, answer_id = rag.ask(
            question=payload.question.strip(),
            history=[m.model_dump() for m in payload.history],
            top_k=payload.top_k,
        )
        return AssistantResponse(
            answer=answer,
            sources=[SourceItem(**s) for s in sources],
            followups=followups,
            answer_id=answer_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"assistant_failed: {exc}") from exc


@app.post("/api/assistant/stream")
def assistant_stream(payload: AssistantRequest):
    def _event_stream():
        try:
            for ev in rag.stream_answer(
                question=payload.question.strip(),
                history=[m.model_dump() for m in payload.history],
                top_k=payload.top_k,
            ):
                yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
        except Exception as exc:
            err = {"type": "error", "error": f"assistant_stream_failed: {exc}"}
            yield f"data: {json.dumps(err, ensure_ascii=False)}\n\n"

    return StreamingResponse(_event_stream(), media_type="text/event-stream")


class FeedbackRequest(BaseModel):
    answer_id: str = Field(min_length=1, max_length=120)
    vote: str = Field(pattern=r"^(up|down)$")
    question: str | None = Field(default=None, max_length=4000)
    answer: str | None = Field(default=None, max_length=12000)


@app.post("/api/assistant/feedback")
def assistant_feedback(payload: FeedbackRequest) -> dict:
    try:
        rag.save_feedback(
            answer_id=payload.answer_id,
            vote=payload.vote,
            question=payload.question,
            answer=payload.answer,
        )
        return {"ok": True}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"feedback_failed: {exc}") from exc
