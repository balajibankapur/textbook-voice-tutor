"""Fast tests for SpeechToText using httpx MockTransport."""
from __future__ import annotations

import httpx
import pytest

from voice.speech_to_text import SpeechToText, SpeechToTextError, Transcript


def _client(handler) -> SpeechToText:
    transport = httpx.MockTransport(handler)
    http = httpx.Client(transport=transport)
    return SpeechToText(api_key="sk-test", http_client=http)


def test_happy_path_returns_transcript() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"text": "How do plants make food?"})

    stt = _client(handler)
    t = stt.transcribe(b"\x00\x01\x02audio-bytes")
    assert isinstance(t, Transcript)
    assert t.text == "How do plants make food?"
    assert t.model == "whisper-1"


def test_sends_multipart_with_model_and_bearer() -> None:
    seen: dict = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["auth"] = req.headers.get("authorization")
        seen["content_type"] = req.headers.get("content-type", "").split(";")[0]
        body = req.content.decode(errors="replace")
        seen["has_model"] = "whisper-1" in body
        seen["has_file"] = 'name="file"' in body
        return httpx.Response(200, json={"text": "ok"})

    stt = _client(handler)
    stt.transcribe(b"dummy")
    assert seen["auth"] == "Bearer sk-test"
    assert seen["content_type"] == "multipart/form-data"
    assert seen["has_model"] is True
    assert seen["has_file"] is True


def test_rejects_empty_audio() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        pytest.fail("should not be called")

    stt = _client(handler)
    with pytest.raises(SpeechToTextError, match="empty audio"):
        stt.transcribe(b"")


def test_raises_on_401() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(401, text="bad key")

    stt = _client(handler)
    with pytest.raises(SpeechToTextError, match="401"):
        stt.transcribe(b"x")


def test_raises_on_empty_transcript() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"text": "   "})

    stt = _client(handler)
    with pytest.raises(SpeechToTextError, match="empty transcript"):
        stt.transcribe(b"x")


def test_timeout_raises() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("boom")

    stt = _client(handler)
    with pytest.raises(SpeechToTextError, match="network error"):
        stt.transcribe(b"x")


def test_empty_api_key_raises() -> None:
    with pytest.raises(SpeechToTextError, match="API key"):
        SpeechToText(api_key="")


def test_optional_language_passed_through() -> None:
    seen: dict = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["body"] = req.content.decode(errors="replace")
        return httpx.Response(200, json={"text": "ok"})

    stt = _client(handler)
    stt.transcribe(b"x", language="en")
    assert "en" in seen["body"]
