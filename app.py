"""Streamlit UI for Milestone 3: ingest a chapter PDF, preview OCR output,
and run text-based retrieval against the FAISS index.
"""
from __future__ import annotations

import logging
from pathlib import Path

import streamlit as st

import base64
import hashlib
import time

from streamlit_autorefresh import st_autorefresh

from config.logging_config import configure_logging
from config.settings import settings


def hashlib_md5(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


def _autoplay_once(audio_bytes: bytes, key: str) -> None:
    """Play audio once per unique `key`, with a visible player as fallback.

    Browsers block autoplay until the user has interacted with the page.
    Showing visible <audio controls> means that when autoplay is blocked
    (first load of a lesson, especially on Safari), the student can still
    click Play and hear it.

    We only emit the element on the FIRST call with a given key, so unrelated
    reruns (follow-up mic widget, state changes) don't replay previous audio.
    """
    played: set = st.session_state.setdefault("autoplayed_keys", set())
    if key in played:
        return
    played.add(key)
    b64 = base64.b64encode(audio_bytes).decode()
    st.html(
        f'<audio autoplay controls style="width:100%;height:36px;margin:4px 0;">'
        f'<source src="data:audio/mp3;base64,{b64}" type="audio/mp3">'
        f"Your browser can't play audio inline.</audio>"
    )
from conversation.clarification import combine_question_with_clarification
from conversation.context_manager import ConversationContext
from conversation.controller import ConversationController, TurnStatus
from decision.decision_policy import DecisionPolicy
from decision.jev_client import JevClient
from decision.jev_decisions_client import JevDecisionsClient
from llm.answer_generator import AnswerGenerator
from ingestion.ingest_chapter import ingest_chapter
from ingestion.build_qa_cache import build_qa_cache
from ingestion.build_concepts import build_concepts
from ingestion.concept_extractor import ConceptExtractor
from ingestion.qa_generator import QAGenerator
from conversation.lesson_controller import (
    JudgementTier,
    LessonController,
    LessonMode,
    LessonStage,
)
from retrieval.qa_matcher import QAMatcher
from voice.speech_to_text import SpeechToText, SpeechToTextError
from voice.text_to_speech import TextToSpeech, TextToSpeechError
from ingestion.ocr_processor import tesseract_available
from ingestion.pdf_processor import ExtractionSource, extract_pages
from retrieval.embedding_service import EmbeddingService
from retrieval.retriever import Retriever
from storage.metadata_repository import MetadataRepository
from storage.models import IngestionStatus

configure_logging()
logger = logging.getLogger(__name__)


# ---- cached singletons --------------------------------------------------


@st.cache_resource
def _repo() -> MetadataRepository:
    return MetadataRepository(settings.metadata_db_path)


@st.cache_resource
def _embedding_service() -> EmbeddingService:
    return EmbeddingService(settings.embedding_model)


@st.cache_resource
def _retriever() -> Retriever:
    return Retriever(
        repo=_repo(),
        embedding_service=_embedding_service(),
        vector_store_dir=settings.vector_store_path,
    )


def _openrouter_decisions_base() -> str:
    """Derive the decisions base URL from the OpenRouter chat base.

    e.g. https://openrouter.ai/api/v1 → https://openrouter.ai/api/alpha
    """
    base = settings.openrouter_base_url.rstrip("/")
    if base.endswith("/v1"):
        return base[:-3] + "/alpha"
    return base.rsplit("/", 1)[0] + "/alpha"


@st.cache_resource
def _jev_client() -> JevDecisionsClient | None:
    if not settings.openrouter_api_key:
        return None
    return JevDecisionsClient(
        api_key=settings.openrouter_api_key,
        base_url=_openrouter_decisions_base(),
        model=settings.jev_model,
    )


@st.cache_resource
def _answer_generator() -> AnswerGenerator | None:
    api_key = settings.llm_api_key
    if not api_key:
        return None
    return AnswerGenerator(
        api_key=api_key,
        base_url=settings.llm_base_url,
        model=settings.llm_model,
    )


@st.cache_resource
def _qa_matcher() -> QAMatcher | None:
    if not settings.openrouter_api_key:
        return None
    return QAMatcher(
        repo=_repo(),
        embedding_service=_embedding_service(),
        qa_index_dir=settings.vector_store_path,
        decisions_client=_jev_client(),
    )


@st.cache_resource
def _qa_generator(pairs_per_chunk: int) -> QAGenerator | None:
    if not settings.llm_api_key:
        return None
    return QAGenerator(
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        model=settings.llm_model,
        pairs_per_chunk=pairs_per_chunk,
    )


@st.cache_resource
def _concept_extractor() -> ConceptExtractor | None:
    if not settings.llm_api_key:
        return None
    return ConceptExtractor(
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        model=settings.llm_model,
    )


@st.cache_resource
def _lesson_controller() -> LessonController | None:
    jev = _jev_client()
    gen = _answer_generator()
    if jev is None or gen is None:
        return None
    return LessonController(
        repo=_repo(),
        answer_generator=gen,
        decisions_client=jev,
    )


def _build_controller(verify_answers: bool, use_qa_cache: bool) -> ConversationController | None:
    jev = _jev_client()
    if jev is None:
        return None
    return ConversationController(
        retriever=_retriever(),
        jev_client=jev,
        policy=DecisionPolicy(max_retrieval_attempts=settings.max_retrieval_attempts),
        base_top_k=settings.top_k,
        answer_generator=_answer_generator(),
        verify_answers=verify_answers,
        qa_matcher=_qa_matcher() if use_qa_cache else None,
    )


@st.cache_resource
def _controller(verify_answers: bool, use_qa_cache: bool) -> ConversationController | None:
    return _build_controller(verify_answers=verify_answers, use_qa_cache=use_qa_cache)


def _get_controller(verify_answers: bool, use_qa_cache: bool = True) -> ConversationController | None:
    return _controller(verify_answers=verify_answers, use_qa_cache=use_qa_cache)


# Including a hash-salt lets us bump this cache without restarting Streamlit.
# When class signatures change (new method kwargs etc.), increment _STT_CACHE_SALT.
_STT_CACHE_SALT = 2


@st.cache_resource
def _stt(_salt: int = _STT_CACHE_SALT) -> SpeechToText | None:
    if not settings.openai_api_key:
        return None
    return SpeechToText(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        model=settings.openai_stt_model,
    )


@st.cache_resource
def _tts() -> TextToSpeech | None:
    if not settings.openai_api_key:
        return None
    return TextToSpeech(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        model=settings.openai_tts_model,
        voice=settings.openai_tts_voice,
        audio_format="mp3",
    )


def _synthesize_cached(text: str) -> bytes | None:
    """Cache TTS output per-text so answer re-renders don't re-call the API."""
    ss = st.session_state
    cache = ss.setdefault("tts_cache", {})
    if text in cache:
        return cache[text]
    tts = _tts()
    if tts is None:
        return None
    try:
        audio = tts.synthesize(text)
    except TextToSpeechError as exc:
        logger.warning("TTS failed: %s", exc)
        return None
    cache[text] = audio.audio_bytes
    return audio.audio_bytes


# ---- helpers ------------------------------------------------------------


def _save_upload(uploaded, chapter_id: str) -> Path:
    settings.uploads_path.mkdir(parents=True, exist_ok=True)
    dest = settings.uploads_path / f"{chapter_id}.pdf"
    dest.write_bytes(uploaded.getbuffer())
    return dest


def _page_preview(pdf_path: Path, limit: int = 5) -> None:
    pages = extract_pages(pdf_path)
    st.caption(f"{len(pages)} page(s). Showing up to {limit}.")
    for ext in pages[:limit]:
        with st.expander(
            f"Page {ext.page_number} — {ext.source.value}"
            + ("" if ext.source == ExtractionSource.EMBEDDED else "  (needs OCR)")
        ):
            if ext.text.strip():
                st.text(ext.text[:1200] + ("..." if len(ext.text) > 1200 else ""))
            else:
                st.info("No embedded text on this page. OCR will run during ingestion.")


# ---- UI ------------------------------------------------------------------


def main() -> None:
    st.set_page_config(page_title="Textbook Voice Tutor", page_icon=":book:", layout="wide")
    st.title("Textbook Voice Tutor")
    st.caption("Grounded, Jev-gated, verified answers over your scanned textbook — now with voice.")

    with st.sidebar:
        st.subheader("Environment")
        st.write(
            {
                "embedding_model": settings.embedding_model,
                "jev_model": settings.jev_model,
                "llm_provider": settings.llm_provider,
                "llm_model": settings.llm_model,
                "top_k": settings.top_k,
                "tesseract_installed": tesseract_available(),
            }
        )
        st.divider()
        st.subheader("Chapters")
        chapters = _repo().list_chapters()
        if not chapters:
            st.info("No chapters ingested yet.")
        else:
            for ch in chapters:
                st.markdown(
                    f"**{ch.title}** — _{ch.subject}_  \n"
                    f"`{ch.chapter_id}` · {ch.page_count}p · {ch.ingestion_status.value}"
                )

    ingest_tab, ask_tab = st.tabs(["Ingest a chapter", "Ask a question"])

    with ingest_tab:
        _render_ingest_tab()

    with ask_tab:
        _render_ask_tab()

    # Lesson mode has moved to the real-time voice service — point students there.
    _render_lesson_service_hint()


def _render_ingest_tab() -> None:
    uploaded = st.file_uploader("Chapter PDF", type=["pdf"])
    col1, col2, col3 = st.columns(3)
    with col1:
        chapter_id = st.text_input("Chapter ID", placeholder="ncert-sci-7-ch3")
    with col2:
        title = st.text_input("Title", placeholder="Photosynthesis")
    with col3:
        subject = st.text_input("Subject", placeholder="Science")

    preview_slot = st.container()
    ingest_slot = st.container()

    if uploaded and chapter_id:
        tmp_path = _save_upload(uploaded, chapter_id)
        with preview_slot:
            st.subheader("Pre-ingestion preview")
            _page_preview(tmp_path)

        run = ingest_slot.button(
            "Build index for this chapter",
            type="primary",
            disabled=not (title and subject),
        )
        if run:
            with ingest_slot:
                with st.spinner("Running ingestion..."):
                    try:
                        report = ingest_chapter(
                            pdf_path=tmp_path,
                            chapter_id=chapter_id,
                            title=title,
                            subject=subject,
                            repo=_repo(),
                            embedding_service=_embedding_service(),
                            vector_store_dir=settings.vector_store_path,
                        )
                        _retriever().invalidate(chapter_id)
                    except Exception as exc:
                        logger.exception("ingestion failed")
                        st.error(f"Ingestion failed: {exc}")
                        return
                st.success(
                    f"Indexed {report.chunk_count} chunk(s) from {report.page_count} page(s) "
                    f"in {report.duration_seconds:.1f}s."
                )
                st.json(
                    {
                        "chapter_id": report.chapter_id,
                        "page_count": report.page_count,
                        "embedded_pages": report.embedded_pages,
                        "ocr_pages": report.ocr_pages,
                        "ocr_failed_pages": report.ocr_failed_pages,
                        "chunk_count": report.chunk_count,
                        "duration_seconds": round(report.duration_seconds, 2),
                    }
                )
                if report.ocr_failed_pages:
                    st.warning(
                        f"OCR failed or was unavailable for pages: {report.ocr_failed_pages}. "
                        "Install tesseract or inspect the pages before trusting answers."
                    )

    # Q&A cache corpus — manage separately for each ingested chapter.
    _render_qa_cache_panel()

    # Lesson concepts — same shape as the Q&A panel.
    _render_concepts_panel()


def _render_qa_cache_panel() -> None:
    ready = [c for c in _repo().list_chapters() if c.ingestion_status == IngestionStatus.READY]
    if not ready:
        return
    st.divider()
    st.subheader("Q&A cache corpus (optional)")
    st.caption(
        "Pre-generate common question/answer pairs. At query time the student's "
        "question is matched to the stored Q&A set via Jev; a hit skips the LLM "
        "entirely — big latency + cost win on repeated questions."
    )
    labels = {f"{c.title} ({c.chapter_id})": c.chapter_id for c in ready}
    pick = st.selectbox("Chapter", list(labels), key="qa_cache_chapter")
    chapter_id = labels[pick]
    existing = _repo().qa_pair_count(chapter_id)
    st.caption(f"Current cached pairs for `{chapter_id}`: **{existing}**")
    pairs_per_chunk = st.slider(
        "Pairs per chunk",
        min_value=3, max_value=20, value=10,
        key="qa_cache_pairs_per_chunk",
        help=(
            "How many Q&A pairs to generate from each chapter chunk. More = better "
            "coverage + more cache hits, but higher one-time generation cost. "
            "For a 300–500 token chunk, 8–12 is usually the sweet spot."
        ),
    )
    expected_total = _repo().chunk_count(chapter_id) * pairs_per_chunk
    st.caption(
        f"Will generate up to **{expected_total}** pairs (after dedup, expect ~70–90% of that). "
        f"Estimated cost: ~${expected_total * 0.00006:.3f} at gpt-4o-mini pricing."
    )
    if _qa_generator(pairs_per_chunk) is None:
        st.caption("⚠ LLM api key missing — set OPENAI_API_KEY (or OPENROUTER_API_KEY) to build.")
        return
    if st.button(
        "Build / rebuild Q&A cache",
        type="primary",
        key="build_qa_cache_btn",
    ):
        with st.spinner("Generating Q&A corpus (one-time cost)..."):
            try:
                report = build_qa_cache(
                    chapter_id=chapter_id,
                    repo=_repo(),
                    embedding_service=_embedding_service(),
                    qa_generator=_qa_generator(pairs_per_chunk),
                    qa_index_dir=settings.vector_store_path,
                )
            except Exception as exc:
                logger.exception("Q&A cache build failed")
                st.error(f"Build failed: {exc}")
                return
        if _qa_matcher() is not None:
            _qa_matcher().invalidate(chapter_id)
        st.success(
            f"Built {report.pairs_persisted} Q&A pair(s) from {report.chunks_processed} "
            f"chunk(s) in {report.duration_seconds:.1f}s."
        )
        st.json(
            {
                "chapter_id": report.chapter_id,
                "chunks_processed": report.chunks_processed,
                "pairs_generated": report.pairs_generated,
                "pairs_persisted": report.pairs_persisted,
                "duration_seconds": round(report.duration_seconds, 2),
            }
        )


def _render_lesson_service_hint() -> None:
    """Visible pointer at the bottom of the page toward the voice service."""
    ready = [c for c in _repo().list_chapters() if c.ingestion_status == IngestionStatus.READY]
    if not ready:
        return
    st.divider()
    st.caption(
        "🎙 **Lesson mode is now a separate real-time voice service** "
        "(Pipecat + WebRTC + Silero VAD) with natural interruption. "
        "Build a lesson plan on the Ingest tab, then: "
        "`uvicorn voice_service.server:app --host 0.0.0.0 --port 7860` "
        "and open http://localhost:7860"
    )


def _render_concepts_panel() -> None:
    ready = [c for c in _repo().list_chapters() if c.ingestion_status == IngestionStatus.READY]
    if not ready:
        return
    st.divider()
    st.subheader("Lesson plan (for Lesson mode)")
    st.caption(
        "Extract 3–6 teachable concepts from the chapter so Lesson mode can "
        "walk through them one at a time. One-time cost per chapter."
    )
    labels = {f"{c.title} ({c.chapter_id})": c.chapter_id for c in ready}
    pick = st.selectbox("Chapter", list(labels), key="concepts_chapter")
    chapter_id = labels[pick]
    existing = _repo().concept_count(chapter_id)
    st.caption(f"Current concepts for `{chapter_id}`: **{existing}**")
    if _concept_extractor() is None:
        st.caption("⚠ LLM api key missing — set OPENAI_API_KEY to build.")
        return
    if st.button("Build / rebuild lesson plan", type="primary", key="build_concepts_btn"):
        with st.spinner("Extracting concepts..."):
            try:
                report = build_concepts(
                    chapter_id=chapter_id,
                    repo=_repo(),
                    extractor=_concept_extractor(),
                )
            except Exception as exc:
                logger.exception("concepts build failed")
                st.error(f"Build failed: {exc}")
                return
        st.success(
            f"Built {report.concepts_persisted} concept(s) from "
            f"{report.chunks_processed} chunk(s) in {report.duration_seconds:.1f}s."
        )


def _render_lesson_tab_DEPRECATED() -> None:
    # Replaced by voice_service/ — see _render_lesson_service_hint().
    ready = [c for c in _repo().list_chapters() if c.ingestion_status == IngestionStatus.READY]
    if not ready:
        st.info("Ingest a chapter first.")
        return

    controller = _lesson_controller()
    if controller is None:
        st.warning(
            "Lesson mode needs OPENROUTER_API_KEY (for Jev) and OPENAI_API_KEY "
            "(for the LLM explanation)."
        )
        return

    ss = st.session_state
    chapter_labels = {f"{c.title} ({c.chapter_id})": c.chapter_id for c in ready}

    col1, col2, col3, col4 = st.columns([2, 2, 1, 1])
    with col1:
        label = st.selectbox("Active chapter", list(chapter_labels), key="lesson_chapter")
    with col2:
        mode_label = st.radio(
            "Mode",
            options=["Teach me (tutor explains, then quizzes)", "Reverse (I explain, tutor checks)"],
            index=0, horizontal=False, key="lesson_mode",
        )
    with col3:
        st.session_state["lesson_timeout_sec"] = st.slider(
            "Wait (s)",
            min_value=3, max_value=30, value=10, step=1,
            help=(
                "How long to wait AFTER the audio finishes before revealing the answer. "
                "The timer only starts once the tutor's audio should be done playing."
            ),
            key="lesson_timeout_slider",
        )
    with col4:
        st.caption("")
        if st.button("🔄 Restart", key="lesson_restart"):
            _reset_lesson_state()
            st.rerun()

    chapter_id = chapter_labels[label]
    mode = LessonMode.TEACH if mode_label.startswith("Teach") else LessonMode.REVERSE

    if _repo().concept_count(chapter_id) == 0:
        st.warning(
            "This chapter has no lesson plan yet. Build one on the **Ingest a chapter** tab."
        )
        return

    # Start a lesson if we don't have one going (or the thread key changed).
    thread_key = (chapter_id, mode.value)
    if ss.get("lesson_thread") != thread_key:
        _reset_lesson_state()
        ss.lesson_thread = thread_key
        with st.spinner("Opening the first concept..."):
            try:
                state, step = controller.start(chapter_id=chapter_id, mode=mode)
            except Exception as exc:
                st.error(f"Couldn't start the lesson: {exc}")
                return
        ss.lesson_state = state
        ss.lesson_step = step

    step = ss.get("lesson_step")
    if step is None:
        st.error("Lesson state is missing. Hit Restart.")
        return

    _render_lesson_step(step, controller)


def _render_lesson_step(step, controller) -> None:
    ss = st.session_state
    total = step.concept_total
    idx = step.concept_index + 1 if step.concept_index < total else total

    # progress bar
    st.progress(min(1.0, step.concept_index / total if total else 0),
                text=f"Concept {idx} / {total}")

    stage_colors = {
        LessonStage.PRESENT:         ("🔵", "Teaching"),
        LessonStage.ASK:             ("🟡", "Your turn — listen for the question"),
        LessonStage.LISTEN:          ("🟢", "Your turn — press to answer"),
        LessonStage.CLARIFY:         ("🟠", "Let's try that again"),
        LessonStage.PRAISE_AND_NEXT: ("🟢", "Nice — moving on"),
        LessonStage.LESSON_COMPLETE: ("🏁", "Lesson complete"),
        LessonStage.FAILED:          ("🔴", "Something went wrong"),
    }
    emoji, status = stage_colors.get(step.stage, ("", step.stage.value))
    if step.concept:
        st.markdown(f"**{emoji} {status}** · _{step.concept.title}_")
    else:
        st.markdown(f"**{emoji} {status}**")

    if step.error:
        st.error(step.error)
        if st.button("Try again", key="lesson_retry_err"):
            ss.lesson_state.attempts = 0
            st.rerun()
        return

    if step.stage is LessonStage.LESSON_COMPLETE:
        st.success(step.speech_text or "Lesson complete.")
        _autoplay_once(_synthesize_cached(step.speech_text) or b"", key=f"lesson_done")
        return

    # tutor speech block
    if step.speech_text:
        st.markdown(f"> {step.speech_text}")
        audio_bytes = _synthesize_cached(step.speech_text)
        if audio_bytes:
            _autoplay_once(
                audio_bytes,
                key=f"lesson:{step.concept_index}:{step.stage.value}:{step.attempts_on_concept}",
            )
            st.caption(
                "🔊 If you didn't hear anything, your browser likely blocked autoplay. "
                "Click the ▶ Play button above."
            )
        else:
            st.warning(
                "⚠ Text-to-speech didn't produce audio. "
                "Check that `OPENAI_API_KEY` is set and the TTS model is reachable."
            )

    # If we have a question to ask on this stage, surface it as the prompt.
    if step.prompt_text and step.stage in {LessonStage.PRESENT, LessonStage.ASK, LessonStage.LISTEN, LessonStage.CLARIFY}:
        st.markdown(f"**❓ {step.prompt_text}**")

    if step.judgement:
        j = step.judgement
        emoji = {JudgementTier.CORRECT: "✅", JudgementTier.PARTIAL: "🟡", JudgementTier.WRONG: "🟠"}[j.tier]
        st.caption(
            f"{emoji} Jev judgement — tier: {j.tier.value} · "
            f"noul: {j.probability:.2f} "
            f"(correct ≥ {j.noul_threshold_correct:.2f}, partial ≥ {j.noul_threshold_partial:.2f})"
        )

    # In every stage where the student speaks, the mic is live immediately —
    # no "I'm ready" gate. The tutor's audio plays alongside; the student can
    # answer whenever they're ready.
    if step.stage in {LessonStage.PRESENT, LessonStage.ASK, LessonStage.LISTEN, LessonStage.CLARIFY}:
        _render_lesson_answer_input(controller)
    elif step.stage is LessonStage.PRAISE_AND_NEXT:
        if st.button("Next concept →", type="primary", key="lesson_next"):
            ss.lesson_step = controller.advance(ss.lesson_state)
            st.rerun()


LESSON_SILENCE_TIMEOUT_SECONDS = 10
# Rough TTS pace at the OpenAI `tts-1` default voice. Short sentences run
# faster than this; long paragraphs run slower. We add a 2 s cushion so a
# normal child doesn't get interrupted.
TTS_CHARS_PER_SECOND = 14
TTS_PLAYBACK_CUSHION_SECONDS = 2.0


def _render_lesson_answer_input(controller) -> None:
    """Mic + text input + silence-timeout auto-reveal for Lesson mode.

    Important sequencing:
      1. Tutor's audio plays uninterrupted — no autorefresh ticks, no mic
         countdown, nothing that could stop the DOM audio element.
      2. Once the audio is estimated to have finished (chars / rate + cushion),
         the silence countdown begins and the auto-refresh ticker starts.
      3. When the countdown hits 0, reveal_answer() fires — but only AFTER
         the explanation audio has finished playing, so there's no overlap.
    """
    ss = st.session_state
    step = ss.lesson_step

    # Mark the stage so we can offset the timer by estimated audio length.
    listen_key = f"listen_start:{ss.lesson_state.concept_index}:{ss.lesson_state.attempts}"
    if ss.get("lesson_listen_key") != listen_key:
        ss.lesson_listen_key = listen_key
        now = time.time()
        text_len = len(step.speech_text or "")
        audio_sec = (
            text_len / TTS_CHARS_PER_SECOND + TTS_PLAYBACK_CUSHION_SECONDS
            if text_len else 0.0
        )
        ss.lesson_audio_ends_at = now + audio_sec
        # Silence timer only begins once audio should be done.
        ss.lesson_listen_started = ss.lesson_audio_ends_at

    timeout = int(ss.get("lesson_timeout_sec", LESSON_SILENCE_TIMEOUT_SECONDS))
    now = time.time()
    audio_still_playing = now < ss.lesson_audio_ends_at
    elapsed = max(0, int(now - ss.lesson_listen_started))
    remaining = max(0, timeout - elapsed)

    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        if audio_still_playing:
            left = max(0, int(ss.lesson_audio_ends_at - now))
            st.caption(f"🔊 Listen carefully to the question… ({left}s)")
        elif remaining > 0:
            st.caption(
                f"⏳ Revealing the answer in **{remaining}s** if there's no reply — "
                "take your time, or use the buttons."
            )
        else:
            st.caption("⌛ Revealing the answer…")
    with col2:
        if st.button("More time", key=f"lesson_more_time_{listen_key}"):
            ss.lesson_listen_started = time.time()
            ss.lesson_audio_ends_at = time.time()  # audio has definitely ended by now
            st.rerun()
    with col3:
        reveal_disabled = audio_still_playing
        if st.button("Reveal now", key=f"lesson_reveal_{listen_key}", disabled=reveal_disabled):
            ss.lesson_step = controller.reveal_answer(ss.lesson_state)
            st.rerun()

    # Tick only when there's something changing on screen — during audio
    # playback (to flip to countdown when audio ends), and during the
    # countdown (to decrement). We do NOT tick after audio and after timer,
    # because at that point we're about to reveal.
    if audio_still_playing:
        secs_left = max(1, int(ss.lesson_audio_ends_at - now) + 1)
        st_autorefresh(interval=1000, limit=secs_left + 1, key=f"audio_tick:{listen_key}")
    elif remaining > 0:
        st_autorefresh(interval=1000, limit=remaining + 2, key=f"timer_tick:{listen_key}")

    # Auto-reveal: only AFTER audio has finished AND the timer has expired.
    if not audio_still_playing and remaining <= 0:
        ss.lesson_step = controller.reveal_answer(ss.lesson_state)
        st.rerun()
        return

    st.caption("🎙 Press to answer — or type below if you prefer.")
    key = f"lesson_voice_{ss.lesson_state.concept_index}_{ss.lesson_state.attempts}"
    audio = st.audio_input("Record answer", key=key)
    typed = st.text_input(
        "Or type your answer",
        key=f"lesson_text_{ss.lesson_state.concept_index}_{ss.lesson_state.attempts}",
        placeholder="Type here if you don't want to speak.",
    )

    reply_text: str | None = None
    if audio is not None and audio.getvalue():
        audio_bytes = audio.getvalue()
        hash_key = f"lesson_reply:{len(audio_bytes)}:{hashlib_md5(audio_bytes[:4096])}"
        if ss.get("lesson_last_audio_key") != hash_key:
            ss.lesson_last_audio_key = hash_key
            with st.spinner("Transcribing..."):
                try:
                    transcript = _stt().transcribe(
                        audio_bytes,
                        filename_hint="lesson.wav",
                        content_type=getattr(audio, "type", None) or "audio/wav",
                        language=settings.stt_language,
                    )
                except Exception as exc:
                    st.error(f"Transcription failed: {exc}")
                    return
            reply_text = transcript.text
            st.success(f"Transcribed: {reply_text}")
    elif typed.strip() and st.button("Submit written answer", key="lesson_submit_text"):
        reply_text = typed.strip()

    if reply_text:
        with st.spinner("Checking your answer..."):
            next_step = controller.submit_answer(ss.lesson_state, reply_text)
        ss.lesson_step = next_step
        st.rerun()


def _reset_lesson_state() -> None:
    ss = st.session_state
    for key in (
        "lesson_thread", "lesson_state", "lesson_step", "lesson_last_audio_key",
        "lesson_listen_key", "lesson_listen_started", "lesson_audio_ends_at",
    ):
        ss.pop(key, None)


def _render_ask_tab() -> None:
    chapters = [c for c in _repo().list_chapters() if c.ingestion_status == IngestionStatus.READY]
    if not chapters:
        st.info("Ingest a chapter first.")
        return

    chapter_labels = {f"{c.title} ({c.chapter_id})": c.chapter_id for c in chapters}
    label = st.selectbox("Active chapter", list(chapter_labels))
    chapter_id = chapter_labels[label]
    active_chapter = next(c for c in chapters if c.chapter_id == chapter_id)

    gate_available = _get_controller(False) is not None
    voice_available = _stt() is not None and _tts() is not None

    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a:
        use_gate = st.toggle(
            "Evidence gate (Jev)",
            value=gate_available,
            disabled=not gate_available,
            help="Jev checks clarity / relevance / sufficiency. Requires OPENROUTER_API_KEY.",
        )
    with col_b:
        verify = st.toggle(
            "Verify answer",
            value=settings.enable_answer_verification,
            disabled=not gate_available,
            help="After the LLM answers, Jev re-checks it. Adds latency and cost.",
        )
    with col_c:
        voice_mode = st.toggle(
            "🎙 Voice mode",
            value=False,
            disabled=not voice_available,
            help="Speak your question; hear the answer. Requires OPENAI_API_KEY.",
        )
    with col_d:
        top_k = st.slider("Top K", min_value=1, max_value=10, value=settings.top_k)

    if not gate_available:
        st.caption("⚠ `OPENROUTER_API_KEY` not set — gate is disabled; showing raw retrieval only.")
    if not voice_available:
        st.caption("ℹ Voice mode disabled — set `OPENAI_API_KEY` to enable.")

    # Explicit reset so the student can abandon a stuck clarification loop
    # without reloading the whole page.
    if st.button("🔄 New question", help="Clear the current thread and start over"):
        _reset_ask_state()
        st.rerun()

    stt_prompt = _stt_prompt_for_chapter(active_chapter.title, active_chapter.subject)
    st.session_state["stt_prompt"] = stt_prompt
    question = _collect_question(
        voice_mode=voice_mode and voice_available,
        stt_prompt=stt_prompt,
    )
    if not question:
        return

    if use_gate and gate_available:
        _render_gated_ask(
            chapter_id,
            question,
            verify=verify,
            voice=voice_mode and voice_available,
        )
    else:
        _render_raw_retrieval(chapter_id, question, top_k)


def _stt_prompt_for_chapter(title: str, subject: str) -> str:
    """A short cue to Whisper so it recognises domain vocabulary better."""
    return (
        f"Grade 2 {subject.lower()} tutoring session about {title}. "
        "Expect words like roots, stems, leaves, flowers, climbers, creepers, "
        "trees, shrubs, herbs, photosynthesis, chlorophyll, chloroplast."
    )


def _voice_state() -> tuple[str, str, str]:
    """Return (color, emoji, label) for the current voice-mode state.

    States: green = student should speak; red = app is speaking or working.
    Streamlit blocks during the actual pipeline, so 'processing' is implicit
    (the spinner is already visible); the pill reflects the pre- and post-run state.
    """
    ss = st.session_state
    last = ss.get("last_turn")
    if last is None:
        return ("#1f8f4a", "🟢", "Ready — press the mic to speak")
    status = getattr(last, "status", None)
    if status is TurnStatus.CLARIFY:
        return ("#1f8f4a", "🟢", "Your turn — speak your reply to the follow-up")
    if status is TurnStatus.ANSWERED:
        return ("#b94a4a", "🔴", "Playing the answer — wait, then press mic for your next question")
    if status in {TurnStatus.REFUSE, TurnStatus.FAIL_CLOSED}:
        return ("#1f8f4a", "🟢", "Ready — press the mic to ask something else")
    return ("#777", "⚪", "")


def _render_status_pill() -> None:
    color, emoji, label = _voice_state()
    st.markdown(
        (
            f"<div style='background:{color}22;border:1px solid {color};"
            f"border-radius:10px;padding:14px 18px;text-align:center;"
            f"font-size:17px;font-weight:600;margin-bottom:8px;'>"
            f"{emoji} {label}</div>"
        ),
        unsafe_allow_html=True,
    )


def _reset_ask_state() -> None:
    """Wipe everything tied to the current question thread.

    Keeps settings (chapter, toggles, Top K) and cached singletons; clears the
    conversation context, pending transcripts, audio dedup, and the TTS cache.
    Also bumps `voice_counter` so the st.audio_input widget's key changes and
    Streamlit renders a fresh (empty) recorder instead of the previous audio.
    """
    ss = st.session_state
    for key in (
        "ask_thread",
        "conversation",
        "combined_query",
        "last_turn",
        "clarify_depth",
        "turn_history",
        "pending_transcript",
        "last_audio_key",
        "last_reply_audio_key",
        "tts_cache",
        "played_audio_hashes",
        "autoplayed_keys",
        "replay_counters",
    ):
        ss.pop(key, None)
    # Bump the audio widget key so the previous recording is dropped.
    ss.voice_counter = ss.get("voice_counter", 0) + 1
    # Clear the text input via pending — writing to the widget key directly
    # after it's been instantiated is forbidden.
    ss.pending_transcript = ""


def _collect_question(*, voice_mode: bool, stt_prompt: str = "") -> str:
    """Return the current question, either typed or transcribed from a recording.

    Streamlit forbids writing to session_state[widget_key] after that widget is
    instantiated, so the transcription path stashes the new text under
    `pending_transcript` and the next render promotes it into the text_input's
    key BEFORE the widget is created.
    """
    ss = st.session_state

    # Promote a pending transcript into the widget's key before rendering.
    if "pending_transcript" in ss:
        ss["question_text_input"] = ss.pop("pending_transcript")
    elif "question_text_input" not in ss:
        ss["question_text_input"] = ""

    if voice_mode:
        _render_status_pill()

    typed = st.text_input(
        "Question",
        placeholder="How do plants make their food?",
        key="question_text_input",
    )

    if voice_mode:
        st.caption("🎙 Press to record your question; release to transcribe.")
        # Dynamic key: reset bumps voice_counter, which invalidates the widget
        # so a fresh empty mic is rendered instead of the previous recording.
        audio = st.audio_input(
            "Record",
            key=f"voice_input_widget_{ss.get('voice_counter', 0)}",
        )

        if audio is None:
            st.caption("ℹ Waiting for a recording. Hit the mic, speak, then hit stop.")
        else:
            audio_bytes = audio.getvalue()
            audio_type = getattr(audio, "type", None) or "audio/wav"
            size = len(audio_bytes)
            st.caption(f"ℹ Recording captured: {size} bytes, type `{audio_type}`.")
            key = f"stt:{size}:{hashlib_md5(audio_bytes[:4096])}" if audio_bytes else ""

            if not audio_bytes:
                st.warning("Recording was empty — try again.")
            elif ss.get("last_audio_key") == key:
                # Transcription already ran for this clip. Show a re-run button.
                if st.button("Re-transcribe this recording"):
                    ss.last_audio_key = None
                    st.rerun()
            else:
                with st.spinner("Transcribing..."):
                    stt = _stt()
                    try:
                        transcript = stt.transcribe(
                            audio_bytes,
                            filename_hint="recording.wav",
                            content_type=audio_type,
                            language=settings.stt_language,
                            prompt=stt_prompt or None,
                        )
                    except SpeechToTextError as exc:
                        st.error(f"Transcription failed: {exc}")
                        return typed
                    except Exception as exc:
                        st.error(f"Unexpected transcription error: {type(exc).__name__}: {exc}")
                        logger.exception("STT failed")
                        return typed
                # Mark done ONLY after success, so retries work when STT errors.
                ss.last_audio_key = key
                # Stash under a pending key — promoted into the widget's key
                # on the next rerun, before the widget is instantiated.
                ss.pending_transcript = transcript.text
                ss.ask_thread = None
                st.rerun()

    if typed:
        return typed
    st.caption("Enter or record a question.")
    return ""


MAX_CLARIFY_DEPTH = 2  # how many times we'll invite the student to narrow before refusing


def _render_gated_ask(chapter_id: str, question: str, verify: bool, voice: bool = False) -> None:
    ss = st.session_state

    # Reset the clarification loop whenever the student types a fresh top-level question
    # or switches chapters.
    thread_key = (chapter_id, question.strip())
    if ss.get("ask_thread") != thread_key:
        ss.ask_thread = thread_key
        ss.conversation = ConversationContext()
        ss.combined_query = question
        ss.last_turn = None
        ss.clarify_depth = 0
        ss.turn_history = []  # list of TurnResult, oldest first — for transcript display

    # Run a controller turn when there's no cached result waiting.
    if ss.last_turn is None:
        spinner_label = (
            "Retrieving, gating, generating, and verifying..."
            if verify
            else "Retrieving, gating, and generating..."
        )
        with st.spinner(spinner_label):
            try:
                result = _get_controller(verify).run_turn(
                    chapter_id=chapter_id,
                    question=ss.combined_query,
                    context=ss.conversation,
                )
            except Exception as exc:  # defensive: don't crash the whole app
                st.error(f"Unexpected error: {exc}")
                return

        ss.conversation.add_student(ss.combined_query)
        if result.status == TurnStatus.ANSWERED:
            ss.conversation.add_tutor(result.message)
        elif result.status == TurnStatus.CLARIFY:
            ss.conversation.add_tutor(result.message)
        ss.last_turn = result
        ss.turn_history.append(result)

    # Render the transcript of this question (clarification exchanges included).
    for i, turn in enumerate(ss.turn_history):
        _render_turn_card(turn, index=i, voice=voice, is_latest=(i == len(ss.turn_history) - 1))

    # If the latest turn asked for clarification and we're still under the cap, prompt.
    latest = ss.last_turn
    if latest.status == TurnStatus.CLARIFY and ss.clarify_depth < MAX_CLARIFY_DEPTH:
        _render_followup_input(
            chapter_id,
            verify,
            voice=voice,
            stt_prompt=ss.get("stt_prompt", ""),
        )
    elif latest.status == TurnStatus.CLARIFY:
        st.warning(
            "Still unclear after a couple of tries. Try rephrasing the question in the box above."
        )

    # Voice-mode "Ask another" button — big and primary — after a terminal turn.
    if voice and latest.status in {TurnStatus.ANSWERED, TurnStatus.REFUSE, TurnStatus.FAIL_CLOSED}:
        st.divider()
        if st.button(
            "🎙 Ask another question",
            type="primary",
            use_container_width=True,
            key="voice_ask_another",
        ):
            _reset_ask_state()
            st.rerun()


def _render_turn_card(result, index: int, voice: bool = False, is_latest: bool = False) -> None:
    """Render one controller turn (the answer, refusal, clarify, etc.).

    If `voice` is on, the latest turn also autoplays TTS of the tutor's speech.
    """
    speak_text: str | None = None  # text to synthesize + play
    if result.status == TurnStatus.ANSWERED:
        st.success("Answer")
        st.markdown(result.message)
        cache_hit = getattr(result, "qa_cache_hit", None)
        if cache_hit is not None:
            st.caption(
                f"⚡ Served from Q&A cache (match confidence {cache_hit.confidence:.2f}; "
                f"skipped live LLM call)"
            )
            if cache_hit.qa_pair.page_numbers:
                pages = ", ".join(str(p) for p in cache_hit.qa_pair.page_numbers)
                st.caption(f"📖 Source pages: {pages}")
        elif result.generated_answer and result.generated_answer.supporting_page_numbers:
            pages = ", ".join(str(p) for p in result.generated_answer.supporting_page_numbers)
            st.caption(f"📖 Source pages: {pages}")
        if result.follow_up_question:
            st.caption(f"💭 Would you like to go further? _{result.follow_up_question}_")
        speak_text = result.message
    elif result.status == TurnStatus.ANSWER_READY:
        st.info(
            "Evidence accepted, but no LLM wired. Set OPENROUTER_API_KEY and restart."
        )
    elif result.status == TurnStatus.CLARIFY:
        st.info(f"❓ {result.message}")
        speak_text = result.message
    elif result.status == TurnStatus.REFUSE:
        st.warning(result.message)
        speak_text = result.message
    elif result.status == TurnStatus.FAIL_CLOSED:
        st.error(result.message)
        # Don't speak a fail-closed error — would feel wrong aloud.

    if voice and is_latest and speak_text:
        audio_bytes = _synthesize_cached(speak_text)
        if audio_bytes:
            audio_hash = hashlib_md5(audio_bytes[:4096])
            autoplay_key = f"turn:{index}:{audio_hash}"
            # Autoplay via raw HTML (one-shot per unique key).
            _autoplay_once(audio_bytes, key=autoplay_key)
            # On every render, also offer a Replay button that generates a NEW
            # autoplay key, so clicking it plays the clip fresh.
            replay_counter = st.session_state.setdefault("replay_counters", {})
            counter_key = f"turn_{index}_{audio_hash[:8]}"
            if st.button("🔊 Replay", key=f"replay_{counter_key}"):
                replay_counter[counter_key] = replay_counter.get(counter_key, 0) + 1
                _autoplay_once(
                    audio_bytes,
                    key=f"{autoplay_key}:replay:{replay_counter[counter_key]}",
                )

    if result.decision:
        d = result.decision
        st.caption(
            f"Jev gate — clear: {d.question_clear} · relevant: {d.evidence_relevant} · "
            f"sufficient: {d.evidence_sufficient} · confidence: {d.decision_confidence:.2f} · "
            f"attempts: {result.attempts}"
        )
    if result.verification:
        v = result.verification
        regen_tag = " (after regeneration)" if result.regenerated else ""
        st.caption(
            f"Jev verifier — supported: {v.supported} · confidence: {v.confidence:.2f}{regen_tag}"
        )

    t = getattr(result, "timings", None)
    if t is not None and getattr(t, "total_ms", 0.0) > 0:
        qa_part = f" · QA-cache {t.qa_cache_ms:.0f}" if getattr(t, "qa_cache_ms", 0) else ""
        st.caption(
            f"⏱ Latency — total {t.total_ms:.0f}ms "
            f"(retrieval {t.retrieval_ms:.0f} · gate {t.gate_ms:.0f} · "
            f"LLM {t.llm_ms:.0f} · verify {t.verify_ms:.0f}{qa_part})"
        )

    with st.expander(f"Debug: turn #{index + 1} (Jev + LLM + passages)"):
        if result.decision:
            st.caption("Jev gate raw")
            st.json(result.decision.raw)
        if result.verification:
            st.caption("Jev verifier raw")
            st.json(result.verification.raw)
        if result.generated_answer:
            st.caption("LLM raw output")
            st.json(result.generated_answer.raw)
        if result.passages:
            _show_passages(result.passages)


def _render_followup_input(
    chapter_id: str, verify: bool, voice: bool = False, stt_prompt: str = ""
) -> None:
    """Render the reply box for the student's answer to the follow-up question.

    In voice mode, the reply is captured by mic and transcribed; otherwise a
    text form is shown. Either way the reply is merged into a clean standalone
    question via the LLM rewriter.
    """
    if voice:
        _render_voice_followup(stt_prompt=stt_prompt)
    else:
        _render_text_followup()


def _render_text_followup() -> None:
    ss = st.session_state
    with st.form(f"followup_form_{ss.clarify_depth}", clear_on_submit=True):
        reply = st.text_input(
            "Your reply",
            key=f"reply_input_{ss.clarify_depth}",
            placeholder="Answer the question above, or say what part you want to know about.",
        )
        submitted = st.form_submit_button("Continue")

    if submitted and reply.strip():
        _apply_clarification_reply(reply.strip())


def _render_voice_followup(stt_prompt: str = "") -> None:
    ss = st.session_state
    st.caption("🎙 Record your reply, then press stop.")
    reply_key = f"followup_voice_{ss.clarify_depth}"
    audio = st.audio_input("Record reply", key=reply_key)
    if audio is None:
        return

    audio_bytes = audio.getvalue()
    if not audio_bytes:
        return
    hash_key = f"reply:{len(audio_bytes)}:{hashlib_md5(audio_bytes[:4096])}"
    if ss.get("last_reply_audio_key") == hash_key:
        st.caption("↻ Reply already applied. Record again to try a different reply.")
        return

    with st.spinner("Transcribing reply..."):
        stt = _stt()
        try:
            transcript = stt.transcribe(
                audio_bytes,
                filename_hint="reply.wav",
                content_type=getattr(audio, "type", None) or "audio/wav",
                language=settings.stt_language,
                prompt=stt_prompt or None,
            )
        except SpeechToTextError as exc:
            st.error(f"Transcription failed: {exc}")
            return
        except Exception as exc:
            st.error(f"Unexpected transcription error: {type(exc).__name__}: {exc}")
            logger.exception("STT (reply) failed")
            return
    ss.last_reply_audio_key = hash_key
    reply_text = transcript.text
    st.success(f"Transcribed reply: {reply_text}")
    _apply_clarification_reply(reply_text)
    st.rerun()


def _apply_clarification_reply(reply_text: str) -> None:
    """Merge the reply into a cleaner standalone question and advance the thread."""
    ss = st.session_state
    last_followup = ss.last_turn.message if ss.last_turn else ""
    ss.combined_query = _merge_clarification(
        original=ss.combined_query,
        follow_up=last_followup,
        reply=reply_text,
    )
    ss.conversation.add_student(reply_text)
    ss.clarify_depth += 1
    ss.last_turn = None  # trigger a fresh controller turn on the next rerun


def _merge_clarification(*, original: str, follow_up: str, reply: str) -> str:
    """Produce a single clean question combining the student's reply with the
    original question and the follow-up Jev asked. Uses the LLM to rewrite when
    available; falls back to the parenthetical concat if the LLM isn't wired.
    """
    gen = _answer_generator()
    if gen is None:
        return combine_question_with_clarification(original, reply)
    try:
        return gen.rewrite_question(
            original=original, follow_up=follow_up, reply=reply
        )
    except Exception as exc:
        logger.warning("question rewriter failed, falling back to concat: %s", exc)
        return combine_question_with_clarification(original, reply)


def _render_raw_retrieval(chapter_id: str, question: str, top_k: int) -> None:
    with st.spinner("Retrieving..."):
        hits = _retriever().search(chapter_id, question, top_k=top_k)
    if not hits:
        st.warning("No results.")
        return
    st.caption("Raw retrieval — Jev gate off.")
    _show_passages(hits)


def _show_passages(hits) -> None:
    for i, h in enumerate(hits, start=1):
        header = f"**{i}.** Page {h.page_start}" + (
            f"–{h.page_end}" if h.page_end != h.page_start else ""
        )
        if h.section_title:
            header += f" · _{h.section_title}_"
        header += f" · score {h.score:.3f}"
        st.markdown(header)
        if h.ocr_quality_status.value not in {"not_applicable", "ok"}:
            st.caption(f"⚠ OCR quality: {h.ocr_quality_status.value}")
        st.text(h.chunk_text[:800] + ("..." if len(h.chunk_text) > 800 else ""))
        st.divider()


if __name__ == "__main__":
    main()
