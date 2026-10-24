"""Live OpenAI STT/TTS smoke tests. Needs OPENAI_API_KEY."""
from __future__ import annotations

import os

import pytest

from voice.speech_to_text import SpeechToText
from voice.text_to_speech import TextToSpeech

pytestmark = pytest.mark.live_openai


@pytest.fixture(scope="module")
def openai_key() -> str:
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        pytest.skip("OPENAI_API_KEY not set")
    return key


def test_live_tts_returns_audio(openai_key: str) -> None:
    tts = TextToSpeech(api_key=openai_key)
    try:
        audio = tts.synthesize("Hello, this is a quick test.")
    finally:
        tts.close()
    assert len(audio.audio_bytes) > 1000  # anything realistic will be bigger
    assert audio.content_type.startswith("audio/")


def test_live_round_trip_tts_then_stt(openai_key: str) -> None:
    """Synthesize a known phrase then feed it back to Whisper and recover roughly the same text."""
    phrase = "Plants have roots, stems, and leaves."
    tts = TextToSpeech(api_key=openai_key, audio_format="mp3")
    stt = SpeechToText(api_key=openai_key)
    try:
        audio = tts.synthesize(phrase)
        transcript = stt.transcribe(
            audio.audio_bytes, filename_hint="roundtrip.mp3", content_type="audio/mp3"
        )
    finally:
        tts.close()
        stt.close()
    text = transcript.text.lower()
    # Must contain the three nouns; Whisper may paraphrase punctuation.
    for word in ("roots", "stems", "leaves"):
        assert word in text, f"transcript missing {word!r}: {transcript.text!r}"
