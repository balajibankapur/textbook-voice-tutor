"""Build the Pipecat pipeline for Lesson mode.

Flow:
  WebRTC audio in
    → VAD (Silero)                     — detects speaking vs silent
    → OpenAI Whisper STT               — transcribes student speech
    → LessonProcessor                  — our brain (uses LessonController)
    → OpenAI TTS                       — speaks the tutor's text
  → WebRTC audio out

Pipecat handles interruption / barge-in natively: when VAD detects the student
has started speaking, the TTS pipeline cancels any pending audio frames.
"""
from __future__ import annotations

import logging

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.services.openai.stt import OpenAISTTService
from pipecat.services.openai.tts import OpenAITTSService
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport
from pipecat.transports.base_transport import TransportParams

from conversation.lesson_controller import LessonController
from voice_service.lesson_processor import LessonProcessor, LessonSessionConfig

logger = logging.getLogger(__name__)


def build_lesson_task(
    *,
    webrtc_connection,
    controller: LessonController,
    session: LessonSessionConfig,
    openai_api_key: str,
    stt_model: str = "whisper-1",
    tts_model: str = "tts-1",
    tts_voice: str = "alloy",
) -> PipelineTask:
    """Assemble the pipeline for a single lesson session and return a task."""
    transport = SmallWebRTCTransport(
        webrtc_connection=webrtc_connection,
        params=TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_analyzer=SileroVADAnalyzer(),
        ),
    )
    stt = OpenAISTTService(api_key=openai_api_key, model=stt_model)
    tts = OpenAITTSService(api_key=openai_api_key, model=tts_model, voice=tts_voice)
    lesson = LessonProcessor(controller=controller, session=session)

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            lesson,
            tts,
            transport.output(),
        ]
    )

    return PipelineTask(
        pipeline,
        params=PipelineParams(
            allow_interruptions=True,
            audio_in_sample_rate=16000,
            audio_out_sample_rate=24000,
        ),
    )
