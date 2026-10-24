"""OpenAI text-to-speech.

Produces mp3 bytes from text; Streamlit's st.audio can autoplay them.
Does not log the input text or audio bytes beyond request-level info.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import httpx

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_SECONDS = 60.0
VALID_FORMATS = {"mp3", "opus", "aac", "flac", "wav", "pcm"}


class TextToSpeechError(RuntimeError):
    """Raised when speech synthesis fails or returns an unusable response."""


@dataclass
class SpeechAudio:
    audio_bytes: bytes
    content_type: str


class TextToSpeech:
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        model: str = "tts-1",
        voice: str = "alloy",
        audio_format: str = "mp3",
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        http_client: httpx.Client | None = None,
    ) -> None:
        if not api_key:
            raise TextToSpeechError("TextToSpeech requires a non-empty API key")
        if audio_format not in VALID_FORMATS:
            raise TextToSpeechError(
                f"audio_format must be one of {VALID_FORMATS}, got {audio_format!r}"
            )
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.voice = voice
        self.audio_format = audio_format
        self.timeout_seconds = timeout_seconds
        self._client = http_client or httpx.Client(timeout=timeout_seconds)
        self._owns_client = http_client is None

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> "TextToSpeech":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def synthesize(self, text: str) -> SpeechAudio:
        """Return audio bytes for the given text."""
        text = text.strip()
        if not text:
            raise TextToSpeechError("empty text")

        url = f"{self.base_url}/audio/speech"
        payload = {
            "model": self.model,
            "voice": self.voice,
            "input": text,
            "response_format": self.audio_format,
        }
        try:
            r = self._client.post(
                url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=self.timeout_seconds,
            )
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            raise TextToSpeechError(f"network error: {exc}") from exc

        if r.status_code != 200:
            raise TextToSpeechError(f"HTTP {r.status_code}: {r.text[:400]}")

        audio = r.content
        if not audio:
            raise TextToSpeechError("TTS returned empty audio")
        content_type = r.headers.get("content-type") or f"audio/{self.audio_format}"
        return SpeechAudio(audio_bytes=audio, content_type=content_type)
