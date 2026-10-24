# I thought I was using Jev as a decision model. I wasn't.

**Then I found the real `/api/alpha/decisions` endpoint — and the gate got 4.3× faster, ~25× cheaper, with the same accuracy.**

Over a weekend I built a voice tutor for my second-grader. She has a scanned NCERT science chapter, can ask questions out loud all day, but can't quite read the chapter cover-to-cover yet. The rule I set for the tutor was simple and strict: **never say anything the chapter doesn't actually say.** No "based on general knowledge." No helpful but ungrounded additions. If the chapter doesn't support it, refuse.

That turned out to need a decision model separate from the LLM that writes the answer — something that could cheaply, confidently say *"is this evidence sufficient?"* over and over. That's where I thought Jev came in.

I was wrong about the easy part. The subtle part is the subject of this post.

---

## The pattern I set up

A standard grounded-RAG pipeline:

1. Student speaks → Whisper transcribes
2. FAISS retrieves top-K passages from the scanned chapter
3. **Jev decides**: question clear? evidence relevant? evidence sufficient?
4. If accepted → GPT-4o-mini writes the answer (grounded, with page citations)
5. **Jev verifies** the answer is supported by the passages
6. If supported → OpenAI TTS speaks it aloud

Any failure anywhere → refuse. The LLM never runs on insufficient evidence. The verifier is structurally different from the writer, so the writer never grades its own homework.

I set `JEV_MODEL=typesafe/jev-router` in my `.env` and wired it to OpenRouter's `/chat/completions` endpoint — the standard OpenAI-compatible path everyone uses. It worked. 100% gate accuracy on a 14-question labelled set.

But the gate took **2.9 seconds on average**. For a yes/no classification. That's bananas.

---

## The first clue: a smoke test

Early on, I pinged `jev-router` with a trivial test — reply with "pong." The response came back fine, but one field caught my eye:

```
model_used: z-ai/glm-5.3-flash
```

**My "Jev" call had been served by Z-AI's GLM-5.3-flash.** Not Jev. `jev-router` is a routing model that dispatches to Jev decision models *or* to arbitrary other chat models depending on the task shape. It's a convenience shim, not the actual Jev API.

So I pinned the model to `typesafe/jev-1.14` to force a real Jev call. The chat API immediately rejected it:

```
HTTP 400: typesafe/jev-1.14 is a decisions model and cannot be used with
the chat/completions endpoint. Use the /api/alpha/decisions endpoint instead.
```

**There was a whole different endpoint I had never seen.**

---

## What the real decisions API looks like

A few blind probes revealed the schema. The real Jev decisions API uses `/api/alpha/decisions` and takes a very different shape from chat completions:

```json
POST /api/alpha/decisions

{
  "model": "typesafe/jev-1.14",
  "state": "<the context — text, record, or array>",
  "questions": {
    "<question_id>": {
      "type": "noul" | "choice" | "score" | "box",
      "instructions": "<what to decide>",
      "criteria": "<optional, depends on type>"
    }
  }
}
```

Four decision primitives, each returning calibrated probabilities straight from the model's output distribution:

- **`noul`** — truth/presence probability. Returns `{"noul": 0.87}`. Use for yes/no gates and verification.
- **`choice`** — pick one of N named options. Returns `{"choice": "a", "probabilities": {...}, "confidence": 0.9}`. Use for Q&A matching and routing.
- **`score`** — weighted criteria. Returns `{"score": 0.72, "legend": {...}}`. Use for ranking.
- **`box`** — spatial bounding box. Not useful for text.

The killer feature: **multiple decisions in a single request.** My three-question gate (clarity + relevance + sufficiency) became three `noul` entries in one payload:

```json
{
  "state": "<student question + retrieved passages>",
  "questions": {
    "clear":      {"type": "noul", "instructions": "The question is specific enough to answer."},
    "relevant":   {"type": "noul", "instructions": "The passages address the question's topic."},
    "sufficient": {"type": "noul", "instructions": "The passages contain enough to answer completely."}
  }
}
```

Response:

```json
{
  "answers": {
    "clear":      {"type": "noul", "noul": 0.92},
    "relevant":   {"type": "noul", "noul": 0.88},
    "sufficient": {"type": "noul", "noul": 0.95}
  },
  "usage": {"input_tokens": 298, "output_tokens": 21, "cost": 0.000012516}
}
```

Those `noul` values are **real calibrated probabilities from a model trained to produce them** — not a chat model guessing a number inside a JSON blob. Thresholding at 0.5 actually means something.

---

## The A/B

Same 14 labelled questions. Same model slug (`jev-1.14`). Only difference: endpoint.

- **Gate accuracy:** 100% → 100% (unchanged)
- **Retrieval recall @5:** 100% → 100% (unchanged)
- **Answer keyword hit:** 86% → 86% (unchanged)
- **Avg gate latency:** 2,931 ms → **675 ms** — **4.3× faster**
- **Avg total pipeline:** 4,092 ms → **1,652 ms** — **2.5× faster**
- **P95 total:** 6,705 ms → **2,854 ms**
- **Cost per gate call:** ~$0.0003 → **~$0.000013** — **~25× cheaper**

**Accuracy didn't move. Latency dropped 2.5× end-to-end. Cost dropped 25×.**

That's the right-tool-for-the-job dividend. Chat completions pay for free-form token generation we weren't using. Decision models are optimized for one thing: structured outputs with calibrated probabilities. Both are the same underlying model class; only the inference path differs.

---

## Where Jev's `choice` finally shone: a Q&A cache

The gate improvement was good, but the dominant cost in a RAG tutor is still the gate+LLM round trip. Classic systems move: **cache the expensive thing.**

Offline, I had the LLM generate ~36 Q&A pairs grounded in the chapter — questions a Grade 2 student might ask, with short grounded answers and page citations. Embedded them, stored them in SQLite, built a second FAISS index.

At query time: embed the student's question, FAISS top-K over the stored *questions* (not passages), then ask Jev a single `choice` decision:

> *"Which of these candidates is semantically equivalent to the student's question — or `none`?"*

Jev returns calibrated probabilities per candidate. Threshold at 0.6. Hit → serve the stored answer. Miss → fall through to the full pipeline.

**This is literally what Jev was built for.** High-volume classification with calibrated outputs. For weeks I'd been shoehorning Jev into "is this evidence sufficient" chat prompts. The `choice` decision is its natural shape.

### Cache A/B (same 14 questions, 50% hit rate on the answerable set)

- **P50 total:** 1,750 ms → **1,345 ms** (−23%)
- **P95 total:** 4,660 ms → **2,396 ms** (−49%)
- **Avg total:** 1,967 ms → **1,434 ms** (−27%)
- **Gate accuracy:** 100% → 100%

**P95 cut in half, zero accuracy loss.** Cache hits are now a single `choice` decision (~500 ms, ~$0.000013) instead of the full gate + LLM round trip (~1,900 ms, ~$0.0008). About **60× cheaper per cache-hit question.**

---

## When Jev earns its keep (and when it doesn't)

**Jev is right when:**

- You need a **calibrated** yes/no or choice, not just "a model's guess at a number inside JSON"
- You have **volume** — decision cost amortizes
- You can structure the task as `noul` / `choice` / `score`
- The failure mode of a wrong decision is expensive (moderation, routing, matching, verification)

**Jev is overkill when:**

- You're generating free-form prose (that's what a general LLM is for)
- You need a textual *reason* for the decision — `noul` doesn't give one; use chat for the user-facing clarification message
- Volume is low and the calibration isn't load-bearing

For my tutor, Jev earns its keep in exactly two places: the three-way gate (now one API call) and the Q&A matcher (literally its canonical use case). Everywhere else — writing the Grade 2 explanation, synthesizing a clarification question — a general LLM is the right tool.

---

## Six lessons I'm taking forward

1. **Read the API docs, not the marketing.** Jev's model page showed only the chat-compatible interface. The real endpoint was documented in a 400 error. Probe the APIs you use.
2. **"Decision model" is a real category, not a style of prompting.** Calibrated outputs from a model trained for decisions are categorically different from a chat model producing JSON. One is prompting; the other is architecture.
3. **Fail closed, always.** The student seeing "I couldn't verify this" is strictly better than hearing something plausible but ungrounded. Especially when the student is seven.
4. **Cache the expensive thing.** The AI version of HTTP caching — pre-generate answers, match-first serve, fall through on miss — produces the same shape of win.
5. **Separate the writer from the judge.** The model that writes the answer shouldn't also decide if the answer is supported.
6. **Measure before you optimize.** My "the gate is slow" instinct turned out to be about which API I was using, not the model itself. A/B eval sets are cheap; opinions are expensive.

---

## The full build

Code, labelled eval set, measurement scripts, Streamlit UI, voice loop, Q&A cache, and the full architecture write-up (with diagrams) are open on GitHub:

**👉 [github.com/balajibankapur/textbook-voice-tutor](https://github.com/balajibankapur/textbook-voice-tutor)**

- 170 passing tests
- `evaluation/evaluate_pipeline.py` reproduces all the numbers above
- `article.md` in the repo is the long-form write-up with the end-to-end architecture, ingestion details, policy state machine, and everything else that didn't fit here

If you build with Jev, use the real decisions API. If you build a RAG tutor, put the gate before the LLM. If you serve the same questions repeatedly, cache them with a calibrated matcher.

My second-grader asked three questions about climber plants on the walk home from school the next day. She knew which page the answer came from.

---

*Balaji Bankapur — [your bio line here: role, what you build, links].*

*The chapter used for development is from the open-access [NCERT](https://ncert.nic.in/) curriculum, used for personal educational purposes. No chapter content is redistributed in the repo.*
