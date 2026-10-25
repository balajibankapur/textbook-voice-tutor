# Lesson voice service (Pipecat + WebRTC)

Real-time voice tutor built on [Pipecat](https://github.com/pipecat-ai/pipecat).
Replaces the Streamlit Lesson tab with a proper conversational voice loop:

- **WebRTC** audio in/out between browser and server
- **Silero VAD** on each 20ms frame — the server knows exactly when the student is speaking
- **OpenAI Whisper** streaming STT
- **OpenAI TTS** streaming playback
- **Barge-in** built in — if the student starts speaking while the tutor is talking, the tutor's audio cancels immediately
- Reuses the existing `LessonController`, `JevDecisionsClient`, `AnswerGenerator`

No more:
- Time-estimate countdowns
- Autorefresh ticks
- Autoplay fights
- "I'm ready" buttons
- Timer-vs-audio races

## Prerequisites

1. Chapter ingested and a lesson plan built — do this in the main Streamlit app:
   `streamlit run app.py` → **Ingest a chapter** → upload PDF → **Build lesson plan**.
2. `.env` filled in with `OPENAI_API_KEY` (STT + TTS + answer LLM) and
   `OPENROUTER_API_KEY` (Jev gate + verifier).
3. Pipecat installed (already in `requirements.txt`): `pip install -r requirements.txt`.

## Run

```bash
uvicorn voice_service.server:app --host 0.0.0.0 --port 7860
```

Then open **http://localhost:7860** in Chrome or Safari.

1. Pick a chapter from the dropdown (only chapters with a built lesson plan appear).
2. Pick a mode: **Teach me** or **Reverse**.
3. Click **Start lesson** and grant mic permission.
4. Just talk — no buttons during the lesson. The tutor speaks, you reply, it judges with Jev's `noul`, moves on when you're right, re-explains when you're not. If you don't reply at all, it reveals the answer after ~12 s and continues.

## Architecture

```
Browser (static/index.html)
    ↓ WebRTC audio (mic → server, server → speaker)
FastAPI server (voice_service/server.py)
    ↓
Pipecat pipeline (voice_service/pipeline.py):
    SmallWebRTCTransport.input()
      → SileroVADAnalyzer         (speech / silence per frame)
      → OpenAISTTService          (streaming Whisper transcription)
      → LessonProcessor           ← custom (voice_service/lesson_processor.py)
      → OpenAITTSService          (streaming TTS)
      → SmallWebRTCTransport.output()
```

The `LessonProcessor` is the glue: it receives `TranscriptionFrame`s from STT,
calls `LessonController.submit_answer()`, and emits `TTSSpeakFrame`s. Pipecat's
built-in interruption handling cancels any in-flight TTS when the student
starts speaking (barge-in).

## Testing without a client

Any WebRTC-capable browser works. For headless CI we'd need a scripted WebRTC
client (e.g. `aiortc` on the test side) — not shipped here.
