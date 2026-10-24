"""Fast tests for TextToSpeech using httpx MockTransport."""
from __future__ import annotations

import json

import httpx
import pytest

from voice.text_to_speech import SpeechAudio, TextToSpeech, TextToSpeechError


def _client(handler, **kw) -> TextToSpeech:
    transport = httpx.MockTransport(handler)
    http = httpx.Client(transport=transport)
    return TextToSpeech(api_key="sk-test", http_client=http, **kw)


def test_happy_path_returns_audio_bytes() -> None:
    fake_audio = b"ID3\x03mp3-bytes-here"

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=fake_audio, headers={"content-type": "audio/mp3"})

    tts = _client(handler)
    s = tts.synthesize("Hello world.")
    assert isinstance(s, SpeechAudio)
    assert s.audio_bytes == fake_audio
    assert s.content_type == "audio/mp3"


def test_sends_bearer_and_json_body() -> None:
    seen: dict = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["auth"] = req.headers.get("authorization")
        seen["body"] = json.loads(req.content.decode())
        return httpx.Response(200, content=b"mp3")

    tts = _client(handler, model="tts-1-hd", voice="nova")
    tts.synthesize("Plants have roots.")
    assert seen["auth"] == "Bearer sk-test"
    assert seen["body"]["model"] == "tts-1-hd"
    assert seen["body"]["voice"] == "nova"
    assert seen["body"]["input"] == "Plants have roots."
    assert seen["body"]["response_format"] == "mp3"


def test_rejects_empty_text() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        pytest.fail("should not be called")

    tts = _client(handler)
    with pytest.raises(TextToSpeechError, match="empty text"):
        tts.synthesize("   ")


def test_raises_on_401() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(401, text="bad key")

    tts = _client(handler)
    with pytest.raises(TextToSpeechError, match="401"):
        tts.synthesize("hi")


def test_raises_on_empty_body() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"")

    tts = _client(handler)
    with pytest.raises(TextToSpeechError, match="empty audio"):
        tts.synthesize("hi")


def test_timeout_raises() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("boom")

    tts = _client(handler)
    with pytest.raises(TextToSpeechError, match="network error"):
        tts.synthesize("hi")


def test_invalid_format_rejected() -> None:
    with pytest.raises(TextToSpeechError, match="audio_format"):
        TextToSpeech(api_key="sk-test", audio_format="xyz")


def test_empty_api_key_raises() -> None:
    with pytest.raises(TextToSpeechError, match="API key"):
        TextToSpeech(api_key="")
