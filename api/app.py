"""FastAPI backend.  Run: uvicorn api.app:app --reload   (docs at /docs)"""
import logging
from typing import Callable, Literal, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from config import settings
from generation.answer import Answer, answer_question

log = logging.getLogger(__name__)

Answerer = Callable[..., Answer]


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    lang: Optional[Literal["en", "hi"]] = None   # default: detected from the question
    k: Optional[int] = Field(default=None, ge=1, le=15)
    act: Optional[str] = None
    court: Optional[str] = None
    year_from: Optional[int] = None
    year_to: Optional[int] = None


class SourceOut(BaseModel):
    cite: str
    locator: str
    doc_id: str
    legal_status: str
    court: str
    year: int
    text: str


class AskResponse(BaseModel):
    answer: str
    lang: Literal["en", "hi"]
    grounded: bool                  # False: the indexed sources did not answer the question
    citations: list[str]
    sources: list[SourceOut]
    disclaimer: str


def to_response(a: Answer) -> AskResponse:
    return AskResponse(
        answer=a.text, lang=a.lang, grounded=a.grounded, citations=a.citations, disclaimer=a.disclaimer,
        sources=[SourceOut(cite=h.cite, locator=h.locator, doc_id=h.doc_id, legal_status=h.legal_status,
                           court=h.court, year=h.year, text=h.text) for h in a.sources])


def create_app(answerer: Optional[Answerer] = None) -> FastAPI:
    """answerer(question, **options) -> Answer; injected in tests so no model or network is needed."""
    app = FastAPI(title="Nyaya Sahitya", version="0.1")
    app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins, allow_methods=["GET", "POST"],
                       allow_headers=["*"])
    run = answerer or answer_question

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "app": settings.app_name, "llm_configured": bool(settings.anthropic_api_key)}

    @app.post("/ask", response_model=AskResponse)
    def ask(req: AskRequest) -> AskResponse:
        opts = {k: v for k, v in req.model_dump(exclude={"question"}).items() if v is not None}
        try:
            return to_response(run(req.question.strip(), **opts))
        except RuntimeError as e:  # e.g. missing API key: a configuration problem, not a bad request
            log.error("ask failed: %s", e)
            raise HTTPException(503, "The answer service is not configured.") from e
        except Exception as e:
            log.exception("ask failed")
            raise HTTPException(500, "Could not produce an answer.") from e

    return app


app = create_app()
