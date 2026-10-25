"""FastAPI server that hosts the Pipecat voice pipeline.

Routes:
  GET /              → serves the browser client (voice_service/static/index.html)
  GET /chapters      → JSON list of chapters that have a lesson plan built
  POST /webrtc/offer → WebRTC SDP exchange — one call per browser session

Run:
  uvicorn voice_service.server:app --host 0.0.0.0 --port 7860
"""
from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pipecat.pipeline.runner import PipelineRunner
from pipecat.transports.smallwebrtc.connection import SmallWebRTCConnection

from config.logging_config import configure_logging
from config.settings import settings
from conversation.lesson_controller import LessonController, LessonMode
from decision.jev_decisions_client import JevDecisionsClient
from llm.answer_generator import AnswerGenerator
from storage.metadata_repository import MetadataRepository
from voice_service.lesson_processor import LessonSessionConfig
from voice_service.pipeline import build_lesson_task

configure_logging()
logger = logging.getLogger(__name__)


STATIC_DIR = Path(__file__).parent / "static"


def _openrouter_decisions_base() -> str:
    base = settings.openrouter_base_url.rstrip("/")
    if base.endswith("/v1"):
        return base[:-3] + "/alpha"
    return base.rsplit("/", 1)[0] + "/alpha"


# ---- Shared singletons -------------------------------------------------


_repo: MetadataRepository | None = None
_controller: LessonController | None = None


def get_repo() -> MetadataRepository:
    global _repo
    if _repo is None:
        _repo = MetadataRepository(settings.metadata_db_path)
    return _repo


def get_controller() -> LessonController:
    global _controller
    if _controller is None:
        if not settings.openrouter_api_key:
            raise RuntimeError("OPENROUTER_API_KEY is required (Jev gate + verifier)")
        if not settings.llm_api_key:
            raise RuntimeError("LLM api key is required (OPENAI_API_KEY by default)")
        jev = JevDecisionsClient(
            api_key=settings.openrouter_api_key,
            base_url=_openrouter_decisions_base(),
            model=settings.jev_model,
        )
        generator = AnswerGenerator(
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
            model=settings.llm_model,
        )
        _controller = LessonController(
            repo=get_repo(),
            answer_generator=generator,
            decisions_client=jev,
        )
    return _controller


# ---- FastAPI app -------------------------------------------------------


app = FastAPI(title="Textbook Voice Tutor — Lesson voice service")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/chapters")
async def chapters() -> JSONResponse:
    """Return chapters with a built lesson plan (so they're ready for Lesson mode)."""
    out = []
    for ch in get_repo().list_chapters():
        n = get_repo().concept_count(ch.chapter_id)
        if n <= 0:
            continue
        out.append(
            {
                "chapter_id": ch.chapter_id,
                "title": ch.title,
                "subject": ch.subject,
                "page_count": ch.page_count,
                "concept_count": n,
            }
        )
    return JSONResponse(out)


@app.post("/webrtc/offer")
async def webrtc_offer(request: Request) -> JSONResponse:
    """SDP offer/answer exchange. Expects JSON: {sdp, type, chapter_id, mode}."""
    body = await request.json()
    chapter_id = body.get("chapter_id")
    mode_raw = (body.get("mode") or "teach").lower()
    if not chapter_id:
        return JSONResponse({"error": "chapter_id is required"}, status_code=400)
    try:
        mode = LessonMode(mode_raw)
    except ValueError:
        return JSONResponse({"error": f"unknown mode: {mode_raw}"}, status_code=400)

    repo = get_repo()
    if repo.concept_count(chapter_id) == 0:
        return JSONResponse(
            {"error": f"chapter {chapter_id!r} has no lesson plan built"},
            status_code=400,
        )

    # Build a per-session WebRTC connection and launch the pipeline.
    connection = SmallWebRTCConnection()
    await connection.initialize(sdp=body["sdp"], type=body["type"])

    controller = get_controller()
    session = LessonSessionConfig(chapter_id=chapter_id, mode=mode)
    task = build_lesson_task(
        webrtc_connection=connection,
        controller=controller,
        session=session,
        openai_api_key=settings.openai_api_key or "",
        stt_model=settings.openai_stt_model,
        tts_model=settings.openai_tts_model,
        tts_voice=settings.openai_tts_voice,
    )

    runner = PipelineRunner(handle_sigint=False)
    asyncio.create_task(runner.run(task))

    answer = connection.get_answer()
    return JSONResponse(
        {
            "sdp": answer["sdp"],
            "type": answer["type"],
        }
    )
