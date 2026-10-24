"""OpenAI Whisper speech-to-text.

Takes recorded audio bytes, returns a transcript. Does not log raw audio
(spec §14, §18).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import httpx

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_SECONDS = 60.0


class SpeechToTextError(RuntimeError):
    """Raised when transcription fails or returns an unusable response."""


@dataclass
class Transcript:
    text: str
    model: str


class SpeechToText:
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        model: str = "whisper-1",
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        http_client: httpx.Client | None = None,
    ) -> None:
        if not api_key:
            raise SpeechToTextError("SpeechToText requires a non-empty API key")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds
        self._client = http_client or httpx.Client(timeout=timeout_seconds)
        self._owns_client = http_client is None

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> "SpeechToText":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def transcribe(
        self,
        audio_bytes: bytes,
        *,
        filename_hint: str = "audio.webm",
        content_type: str = "audio/webm",
        language: str | None = None,
        prompt: str | None = None,
    ) -> Transcript:
        """Transcribe audio bytes. Returns the recognized text.

        `prompt` is a short string that primes the model with expected
        vocabulary (e.g. "Grade 2 science: plants, roots, climbers, creepers").
        Significantly improves recognition of domain terms and acronyms.
        """
        if not audio_bytes:
            raise SpeechToTextError("empty audio buffer")

        url = f"{self.base_url}/audio/transcriptions"
        files = {"file": (filename_hint, audio_bytes, content_type)}
        data: dict[str, str] = {"model": self.model, "response_format": "json"}
        if language:
            data["language"] = language
        if prompt:
            data["prompt"] = prompt

        try:
            r = self._client.post(
                url,
                headers={"Authorization": f"Bearer {self.api_key}"},
                files=files,
                data=data,
                timeout=self.timeout_seconds,
            )
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            raise SpeechToTextError(f"network error: {exc}") from exc

        if r.status_code != 200:
            raise SpeechToTextError(f"HTTP {r.status_code}: {r.text[:400]}")

        try:
            payload = r.json()
            text = str(payload.get("text", "")).strip()
        except Exception as exc:
            raise SpeechToTextError(f"invalid JSON response: {exc}") from exc

        if not text:
            raise SpeechToTextError("Whisper returned an empty transcript")
        return Transcript(text=text, model=self.model)
