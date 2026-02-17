from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient


def _load_env() -> None:
    root = Path(__file__).resolve().parents[1]
    env_file = root / ".env"
    env_txt = root / ".env.txt"
    if env_file.exists():
        load_dotenv(env_file)
    elif env_txt.exists():
        load_dotenv(env_txt)


def _infer_https(url: str) -> bool | None:
    if url.lower().startswith("https://"):
        return True
    if url.lower().startswith("http://"):
        return False
    return None


class RagService:
    def __init__(self) -> None:
        _load_env()
        self.openai_api_key = os.getenv("OPENAI_API_KEY") or ""
        self.openai_chat_model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
        self.openai_embed_model = os.getenv("EMBED_MODEL", "text-embedding-3-large")
        self.qdrant_url = os.getenv("QDRANT_URL") or ""
        self.qdrant_api_key = os.getenv("QDRANT_API_KEY")
        self.qdrant_collection = os.getenv("QDRANT_COLLECTION") or ""
        self.default_top_k = int(os.getenv("RAG_TOP_K", "6"))
        self.max_snippet_chars = int(os.getenv("RAG_SNIPPET_CHARS", "700"))
        self.feedback_store: list[dict[str, Any]] = []

        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required")
        if not self.qdrant_url:
            raise ValueError("QDRANT_URL is required")
        if not self.qdrant_collection:
            raise ValueError("QDRANT_COLLECTION is required")

        self.openai = OpenAI(api_key=self.openai_api_key)
        self.qdrant = QdrantClient(
            url=self.qdrant_url,
            api_key=self.qdrant_api_key,
            https=_infer_https(self.qdrant_url),
            port=None,
            timeout=60,
        )

    def _embed_query(self, question: str) -> list[float]:
        r = self.openai.embeddings.create(model=self.openai_embed_model, input=[question])
        return r.data[0].embedding

    def _search(self, vector: list[float], top_k: int) -> list[Any]:
        if hasattr(self.qdrant, "search"):
            return self.qdrant.search(  # type: ignore[attr-defined]
                collection_name=self.qdrant_collection,
                query_vector=vector,
                limit=top_k,
                with_payload=True,
            )
        qr = self.qdrant.query_points(
            collection_name=self.qdrant_collection,
            query=vector,
            limit=top_k,
            with_payload=True,
        )
        return list(qr.points)

    def _normalize_sources(self, hits: list[Any]) -> list[dict[str, Any]]:
        sources: list[dict[str, Any]] = []
        for h in hits:
            payload = h.payload or {}
            text = str(payload.get("text") or "").strip()
            snippet = text[: self.max_snippet_chars]
            if len(text) > self.max_snippet_chars:
                snippet += "..."
            section = " / ".join(payload.get("header_path") or [])
            sources.append(
                {
                    "title": str(payload.get("title") or "Документация Рутокен"),
                    "url": payload.get("source_url"),
                    "doc_path": payload.get("doc_path"),
                    "section": section or None,
                    "score": float(getattr(h, "score", 0.0) or 0.0),
                    "snippet": snippet,
                }
            )
        return sources

    def _build_context(self, sources: list[dict[str, Any]]) -> str:
        blocks: list[str] = []
        for idx, src in enumerate(sources, start=1):
            label = f"S{idx}"
            title = src["title"]
            section = src.get("section") or "-"
            url = src.get("url") or "-"
            snippet = src.get("snippet") or "-"
            blocks.append(
                f"[{label}] {title}\n"
                f"section: {section}\n"
                f"url: {url}\n"
                f"text: {snippet}"
            )
        return "\n\n".join(blocks)

    def _build_messages(self, question: str, history: list[dict[str, str]], context: str) -> list[dict[str, str]]:
        system = (
            "Ты встроенный AI-помощник портала документации Рутокен. "
            "Отвечай только на основе переданного контекста. "
            "Если данных недостаточно, явно скажи это и предложи, что уточнить. "
            "Пиши кратко и по делу. "
            "Для фактических утверждений добавляй ссылки на источники в формате [S1], [S2]."
        )

        messages: list[dict[str, str]] = [{"role": "system", "content": system}]
        if history:
            messages.extend(history[-8:])
        messages.append(
            {
                "role": "user",
                "content": (
                    f"Вопрос пользователя:\n{question}\n\n"
                    f"Контекст из базы знаний:\n{context}\n\n"
                    "Сформируй ответ на русском языке."
                ),
            }
        )
        return messages

    def _generate_answer(self, question: str, history: list[dict[str, str]], context: str) -> str:
        messages = self._build_messages(question, history, context)
        out = self.openai.chat.completions.create(
            model=self.openai_chat_model,
            temperature=0.2,
            messages=messages,
        )
        return (out.choices[0].message.content or "").strip()

    def _generate_followups(self, question: str, answer: str, sources: list[dict[str, Any]]) -> list[str]:
        src_titles = ", ".join([s.get("title", "") for s in sources[:4] if s.get("title")])
        prompt = (
            "Сгенерируй 4 коротких уточняющих вопроса пользователя по теме ответа. "
            "Ответ верни строго JSON-объектом вида {\"followups\":[\"...\"]} без комментариев. "
            f"Исходный вопрос: {question}\n"
            f"Ответ ассистента: {answer}\n"
            f"Источники: {src_titles}"
        )
        try:
            out = self.openai.chat.completions.create(
                model=self.openai_chat_model,
                temperature=0.4,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
            )
            content = (out.choices[0].message.content or "").strip()
            parsed = json.loads(content)
            items = parsed.get("followups") if isinstance(parsed, dict) else None
            if isinstance(items, list):
                clean = [str(x).strip() for x in items if str(x).strip()]
                return clean[:4]
        except Exception:
            pass
        return [
            "Какие шаги выполнить в Linux по порядку?",
            "Какие есть типичные ошибки и как их исправить?",
            "Покажи минимальный рабочий пример конфигурации.",
            "Какие версии и компоненты должны быть установлены?",
        ]

    def _retrieve(self, question: str, top_k: int | None = None) -> tuple[list[dict[str, Any]], str]:
        vec = self._embed_query(question.strip())
        hits = self._search(vec, top_k or self.default_top_k)
        sources = self._normalize_sources(hits)
        context = self._build_context(sources)
        return sources, context

    def ask(
        self,
        question: str,
        history: list[dict[str, str]] | None = None,
        top_k: int | None = None,
    ) -> tuple[str, list[dict[str, Any]], list[str], str]:
        if not question.strip():
            raise ValueError("question is empty")
        sources, context = self._retrieve(question.strip(), top_k)
        answer = self._generate_answer(question.strip(), history or [], context)
        followups = self._generate_followups(question.strip(), answer, sources)
        answer_id = str(uuid.uuid4())
        return answer, sources, followups, answer_id

    def stream_answer(
        self,
        question: str,
        history: list[dict[str, str]] | None = None,
        top_k: int | None = None,
    ):
        if not question.strip():
            raise ValueError("question is empty")

        sources, context = self._retrieve(question.strip(), top_k)
        messages = self._build_messages(question.strip(), history or [], context)

        answer_chunks: list[str] = []
        stream = self.openai.chat.completions.create(
            model=self.openai_chat_model,
            temperature=0.2,
            messages=messages,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content if chunk.choices else None
            if not delta:
                continue
            answer_chunks.append(delta)
            yield {"type": "delta", "delta": delta}

        answer = "".join(answer_chunks).strip()
        followups = self._generate_followups(question.strip(), answer, sources)
        answer_id = str(uuid.uuid4())
        yield {
            "type": "final",
            "answer": answer,
            "sources": sources,
            "followups": followups,
            "answer_id": answer_id,
        }

    def save_feedback(self, answer_id: str, vote: str, question: str | None = None, answer: str | None = None) -> None:
        self.feedback_store.append(
            {
                "answer_id": answer_id,
                "vote": vote,
                "question": question,
                "answer": answer,
            }
        )
