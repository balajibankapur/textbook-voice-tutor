"""Pipecat FrameProcessor that drives Lesson mode as a real-time conversation.

Design:
  - On start: open the first concept and emit a TTSSpeakFrame for the tutor's
    explanation + question. Pipecat's VAD + STT + TTS handles the audio layer.
  - On final transcription: pass the student's answer to LessonController.
    submit_answer(); emit the resulting speech as TTSSpeakFrame.
  - On UserStartedSpeakingFrame while the tutor is talking: Pipecat's TTS
    service handles the barge-in natively (cancels pending audio when the user
    interrupts), so we only need to react to UserStoppedSpeakingFrame + the
    transcript that follows.
  - On idle after the tutor finishes: the pipeline's VAD will emit
    UserStoppedSpeakingFrame naturally when the student eventually speaks.
    We also schedule a silence timeout as a safety net.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

from pipecat.frames.frames import (
    EndFrame,
    Frame,
    StartFrame,
    TranscriptionFrame,
    TTSSpeakFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from conversation.lesson_controller import (
    LessonController,
    LessonMode,
    LessonStage,
    LessonState,
)

logger = logging.getLogger(__name__)


@dataclass
class LessonSessionConfig:
    chapter_id: str
    mode: LessonMode = LessonMode.TEACH
    silence_reveal_seconds: float = 12.0  # auto-reveal after this long with no reply


class LessonProcessor(FrameProcessor):
    """Pipecat processor that runs a Lesson mode conversation.

    This sits between the STT and TTS services in the pipeline. STT emits
    `TranscriptionFrame`; we handle it, call the controller, and push
    `TTSSpeakFrame` back out. The TTS service downstream renders speech.
    """

    def __init__(
        self,
        *,
        controller: LessonController,
        session: LessonSessionConfig,
    ) -> None:
        super().__init__()
        self.controller = controller
        self.session = session
        self.state: LessonState | None = None
        self._last_step = None
        self._silence_task: asyncio.Task | None = None
        self._advance_pending = False

    # ---- Pipecat lifecycle ----------------------------------------------

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)

        # Loud log on every meaningful frame type so we can trace the loop.
        name = type(frame).__name__
        if name not in {"AudioRawFrame", "InputAudioRawFrame", "OutputAudioRawFrame"}:
            logger.info("LessonProcessor got %s (dir=%s)", name, direction)

        if isinstance(frame, StartFrame):
            await self.push_frame(frame, direction)
            await self._open_first_concept()
            return

        if isinstance(frame, EndFrame):
            self._cancel_silence_timer()
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, UserStoppedSpeakingFrame):
            # VAD says the student finished a vocalization. The STT service
            # will shortly emit a TranscriptionFrame with the words. Cancel the
            # silence timer either way.
            self._cancel_silence_timer()
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, TranscriptionFrame):
            text = (frame.text or "").strip()
            if not text:
                await self.push_frame(frame, direction)
                return
            await self._handle_student_answer(text)
            # Don't propagate the transcription frame downstream — we've
            # consumed it.
            return

        await self.push_frame(frame, direction)

    # ---- Lesson flow -----------------------------------------------------

    async def _open_first_concept(self) -> None:
        logger.info(
            "opening lesson: chapter=%s mode=%s",
            self.session.chapter_id, self.session.mode.value,
        )
        try:
            self.state, step = self.controller.start(
                chapter_id=self.session.chapter_id,
                mode=self.session.mode,
            )
        except Exception as exc:
            logger.exception("couldn't open lesson")
            await self._speak(
                "Sorry — I couldn't open the lesson. "
                "Please check the chapter and try again."
            )
            return

        self._last_step = step
        logger.info(
            "lesson open; concept %d/%d title=%r stage=%s",
            step.concept_index + 1, step.concept_total,
            getattr(step.concept, "title", "?"),
            step.stage.value,
        )
        preview = (step.speech_text or "")[:120]
        logger.info("tutor will speak (%d chars): %r...", len(step.speech_text or ""), preview)
        await self._speak(step.speech_text)
        self._arm_silence_timer()

    async def _handle_student_answer(self, text: str) -> None:
        if self.state is None:
            logger.warning("got transcription before lesson started: %r", text)
            return

        logger.info("student said: %r (concept %d)", text, self.state.concept_index)

        # If the previous step was waiting on the student to answer, judge now.
        # If the previous step was PRAISE_AND_NEXT or CLARIFY, treat this as
        # the "I'm ready" signal (any utterance) and advance.
        if self._last_step and self._last_step.stage is LessonStage.PRAISE_AND_NEXT:
            next_step = self.controller.advance(self.state)
            self._last_step = next_step
            await self._emit_step(next_step)
            return

        if self._last_step and self._last_step.stage is LessonStage.CLARIFY:
            # advance after CLARIFY returns to LISTEN; the student's utterance
            # is the actual answer, so judge it.
            self._last_step = self.controller.advance(self.state)
            # Fall through to submit_answer below.

        try:
            step = self.controller.submit_answer(self.state, text)
        except Exception as exc:
            logger.exception("submit_answer failed")
            await self._speak(
                "Sorry — something went wrong checking your answer. "
                "Let's try again."
            )
            self._arm_silence_timer()
            return

        self._last_step = step
        await self._emit_step(step)

    async def _emit_step(self, step) -> None:
        """Speak the step's speech_text and either arm the silence timer or
        auto-advance (for lesson completion)."""
        if step.speech_text:
            await self._speak(step.speech_text)

        if step.stage is LessonStage.LESSON_COMPLETE:
            await self._speak(
                "That's the whole chapter! Great job. "
                "You can start a new one when you're ready."
            )
            self._cancel_silence_timer()
            return

        if step.stage is LessonStage.PRAISE_AND_NEXT:
            # After praise, auto-advance to the next concept without waiting
            # for the student to say "ok" — just open the next concept.
            next_step = self.controller.advance(self.state)
            self._last_step = next_step
            await self._emit_step(next_step)
            return

        # LESSON_PRESENT or CLARIFY → wait for the student to answer.
        self._arm_silence_timer()

    async def _reveal_on_silence(self) -> None:
        """Called by the silence timer when the student doesn't reply."""
        try:
            await asyncio.sleep(self.session.silence_reveal_seconds)
        except asyncio.CancelledError:
            return
        if self.state is None:
            return
        logger.info(
            "silence timeout fired (%.1fs) — revealing answer",
            self.session.silence_reveal_seconds,
        )
        step = self.controller.reveal_answer(self.state)
        self._last_step = step
        await self._emit_step(step)

    def _arm_silence_timer(self) -> None:
        self._cancel_silence_timer()
        self._silence_task = asyncio.create_task(self._reveal_on_silence())

    def _cancel_silence_timer(self) -> None:
        if self._silence_task and not self._silence_task.done():
            self._silence_task.cancel()
        self._silence_task = None

    async def _speak(self, text: str) -> None:
        if not text or not text.strip():
            logger.warning("_speak called with empty text; skipping")
            return
        logger.info("TTSSpeakFrame queued (%d chars)", len(text))
        await self.push_frame(TTSSpeakFrame(text=text), FrameDirection.DOWNSTREAM)
