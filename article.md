# Building a Voice Tutor on a Scanned Textbook — and What I Learned About Jev's Real Decision API

> *A weekend project with my second-grader, a scanned NCERT science chapter, and the discovery that the "decision model" I'd been using was actually a chat model in disguise.*

**TL;DR:** I built a voice-enabled tutor that answers Grade 2 questions strictly from a scanned NCERT science chapter. Along the way I thought I was using Jev as a decision model — I wasn't. Switching to Jev's actual `/api/alpha/decisions` endpoint made the gate **4.3× faster**, cut cost **~25×**, and gave me real calibrated probabilities. Then I added a Q&A cache that uses Jev's `choice` decisions to shortcut repeat questions — another **27% latency drop on average, 49% at the P95**. Below is the full flow, the tech stack, the architecture, the pivot, and the numbers.

![Hero image — Streamlit voice tutor UI alongside a scanned NCERT page](docs/images/hero.png)
> *Hero image to replace: screenshot of the Streamlit UI with 🟢 "Ready — press the mic to speak" status alongside your scanned NCERT page. See `docs/images/README.md`.*

---

## Day One

My second-grader has a glossy NCERT science textbook. She can't quite read it yet — not reliably, not for a whole chapter. She can ask questions out loud all day, though. The gap is painful: the knowledge is right there on the page, and her reading isn't where her curiosity is.

So I built the obvious thing: a tutor she can talk to about *that specific chapter*, not the whole internet. Scanned PDF in, voice conversation out, every answer tied to a page she could point to.

The first interesting word in that sentence is "strictly." The tutor should never invent facts — ever. If the chapter doesn't say it, the tutor doesn't say it. "Climber plants need support" is fine if the chapter says that. "Climbing helps them reach sunlight" is **not** fine, even though it's true in the world, because the chapter doesn't actually say it. For a seven-year-old, drawing that line sharply matters more than giving a slightly fuller answer.

That's the whole project. Everything downstream is engineering in service of that one constraint.

---

## The Tech Stack

Everything local except LLM / voice:

| Layer | Tool | Why |
|---|---|---|
| UI | Streamlit | Fast to iterate, built-in `st.audio_input` for push-to-talk |
| PDF | PyMuPDF | Robust page rendering + text extraction |
| OCR | Tesseract | Local, cheap, good enough for scanned pages |
| Chunking | Custom Python | Paragraph-aware, 300–500 tokens with 50-token overlap |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (384-dim) | Local, fast, good enough for topic retrieval |
| Vector search | FAISS CPU (`IndexFlatIP`) | Flat index is fine at textbook scale |
| Metadata | SQLite | Dead simple; `faiss_id == chunk_id` keeps mapping trivial |
| LLM answer | OpenAI `gpt-4o-mini` (direct API) | Fast, cheap, structured JSON output |
| Decision gate | **Jev `typesafe/jev-1.14` via `/api/alpha/decisions`** | The whole subject of this article |
| STT | OpenAI `gpt-4o-mini-transcribe` | Noticeably better than `whisper-1` for a kid's voice |
| TTS | OpenAI `tts-1` | Good enough for a tutor, cheap |

No AWS. No Docker. No hosted vector DB. One Python process, one SQLite file, one directory of FAISS indices.

```
textbook-voice-tutor/
├─ app.py                       # Streamlit UI (ingest · voice ask · Q&A cache)
├─ ingestion/                   # PDF, OCR, chunker, chapter ingest, Q&A corpus
├─ retrieval/                   # embeddings, FAISS store, retriever, Q&A matcher
├─ decision/                    # Jev decisions client, policy state machine
├─ llm/                         # grounded answer generator + question rewriter
├─ voice/                       # OpenAI Whisper STT, OpenAI TTS
├─ conversation/                # controller, context, clarification merging
├─ storage/                     # SQLite schema + repository
├─ evaluation/                  # labelled dataset + CLI evaluator
├─ tests/                       # 170 fast tests, live tests behind markers
└─ docs/images/                 # architecture + A/B charts
```

---

## The Pipeline (Before the Pivot)

### Ingestion (one-time per chapter)

```
Scanned PDF
   ↓
PyMuPDF — try embedded text
   ↓
If scanned → Tesseract OCR
   ↓
Normalize (de-hyphenate, collapse whitespace)
   ↓
Structure-aware chunker
   ↓
MiniLM-L6-v2 → embeddings
   ↓
FAISS index + SQLite metadata
```

Idempotent per `chapter_id`. Each chunk carries `page_start`, `page_end`, `section_title`, and an OCR-quality tag so bad OCR pages can be flagged (not silently accepted).

### Conversation (live)

```
Student speaks
   ↓
Whisper → transcript
   ↓
Embed query → FAISS top-K
   ↓
Jev gate: {question_clear, evidence_relevant, evidence_sufficient}
   ↓
Policy state machine:
   - unclear → ask one clarification question
   - insufficient within budget → widen top-K, re-ask gate
   - accepted → go to LLM
   - error → fail closed (never pretend)
   ↓
GPT-4o-mini generates grounded answer + page citations
   ↓
Jev verifier: "every claim supported by passages?"
   - supported → speak the answer
   - unsupported → regenerate once with the verifier's reason; if still unsupported, refuse
   ↓
OpenAI TTS → audio playback
```

Three design points worth calling out:

1. **The gate runs before the LLM.** The LLM is never asked a question whose evidence has not already been judged sufficient.
2. **Fail closed.** Any network error, invalid JSON, or timeout at the gate → the student is told we couldn't verify. We never silently fall through to "best guess".
3. **The verifier is a separate model** from the writer, so it isn't marking its own homework.

---

## The Jev Discovery

I set `JEV_MODEL=typesafe/jev-router` in `.env` and wired it as a standard OpenAI-compatible chat completion. It worked — 100% gate accuracy across my 14-question labelled eval set. But something nagged:

**Gate latency was 2.9 seconds on average.** For a context-free yes/no decision, that's bananas.

A quick smoke test exposed what was really happening:

```python
# "pong" test on jev-router
model_used: z-ai/glm-5.3-flash
```

**My "Jev" call had been served by Z-AI's GLM-5.3-flash model.** Not Jev at all. `jev-router` is a routing model that dispatches to Jev decision models *or* to arbitrary other chat models depending on the task — a convenience shim, not the actual decisions API.

Pinning to the real `typesafe/jev-1.14` immediately broke everything:

```
HTTP 400: typesafe/jev-1.14 is a decisions model and cannot be used with
the chat/completions endpoint. Use the /api/alpha/decisions endpoint instead.
```

**There was a whole different endpoint I hadn't been using.**

### What the real decisions API actually is

A few probes revealed the schema:

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

Four decision primitives:

| Type | What it does | Response |
|---|---|---|
| `noul` | Truth/presence probability | `{"noul": 0.87}` — calibrated 0–1 |
| `choice` | Pick one named criterion | `{"choice": "climber", "probabilities": {...}, "confidence": 1.0}` |
| `score` | Weighted criteria scoring | `{"score": 0.16, "legend": {...}, "probabilities": {...}}` |
| `box` | Spatial bounding box | `{"box": {"top": 0.05, ...}}` |

**Multiple decisions in one request.** The gate's three questions — clarity, relevance, sufficiency — went from three JSON fields parsed out of one chat response to **three `noul` entries in a single decisions call**:

```json
{
  "state": "<question + passages>",
  "questions": {
    "clear":      {"type": "noul", "instructions": "The student's question is specific enough to answer."},
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

Those `noul` values are **real calibrated probabilities out of a model trained to produce them**, not a chat model guessing a number inside a JSON blob. Thresholding at 0.5 means something.

---

## A/B: Chat API vs. the Real Decisions API

Same 14 labelled questions. Same Jev model slug (`jev-1.14`). Only difference: endpoint.

| | **Chat API via `jev-router`** | **Decisions API via `jev-1.14`** |
|---|---|---|
| Gate accuracy | 100% (14/14) | **100% (14/14)** |
| Retrieval recall @5 | 100% | 100% |
| Answer keyword hit | 86% (6/7) | 86% (6/7) |
| Avg gate latency | 2,931 ms | **675 ms (4.3× faster)** |
| Avg LLM latency | 1,071 ms | 932 ms |
| **Avg total latency** | 4,092 ms | **1,652 ms (2.5× faster)** |
| P50 total | 4,038 ms | **1,970 ms** |
| P95 total | 6,705 ms | **2,854 ms** |
| Cost per gate call | ~$0.0003 | **~$0.000013 (~25× cheaper)** |

**Accuracy stayed exactly the same. Latency dropped 2.5× end-to-end. Cost dropped 25×.**

That's the right-tool-for-the-job dividend. Chat completions are optimized for free-form token generation. Decision models are optimized for one thing: classified outputs with calibrated probabilities. Both are the same underlying model class, but served via different inference paths. The chat path pays for generality we weren't using.

![Chat API vs decisions API — latency and cost](docs/images/ab-latency-cost.svg)

---

## Adding a Q&A Cache

Even at 1.7s/question, the gate dominates — ~40% of total time. For a product serving many students, this adds up fast. Classic systems move: **cache the expensive thing**.

### The Pattern

```
Offline (per chapter, one time):
  LLM generates N question/answer pairs grounded in the chapter
  Embed each question
  Build a second FAISS index: chapter.qa.faiss
  Store pairs + embeddings in SQLite

Live query:
  Student asks a question
  Embed → FAISS top-K over stored questions
  Jev `choice` decision: pick best candidate (or "none")
  If chosen candidate's probability ≥ threshold → serve stored answer
  Else → fall through to the normal retrieve + gate + LLM path
```

The matcher is where Jev finally earns its calibration. The `choice` decision takes the student's question as `state` and the top-K FAISS candidates as named criteria, plus an explicit `none` option:

```json
{
  "state": "What do climber plants do?",
  "questions": {
    "match": {
      "type": "choice",
      "instructions": "Pick the candidate semantically equivalent to the student's question.",
      "criteria": {
        "c0": "What are the parts of a plant?",
        "c1": "How do climbers grow?",
        "c2": "What is photosynthesis?",
        "c3": "Name climber plant examples.",
        "c4": "What are creepers?",
        "none": "None of the candidates is a semantic match."
      }
    }
  }
}
```

Response:

```json
{
  "answers": {
    "match": {
      "type": "choice",
      "choice": "c1",
      "probabilities": {"c0": 0.03, "c1": 0.87, "c2": 0.01, "c3": 0.08, "c4": 0.01, "none": 0.0},
      "confidence": 0.87
    }
  }
}
```

Threshold at 0.6, chosen candidate above threshold → serve `c1`'s stored answer. Below threshold or `none` chosen → fall through.

**This is literally what Jev was built for.** High-volume classification with calibrated probabilities. For months I'd been shoehorning it into a chat-style "is this evidence sufficient" prompt, which it tolerated but wasn't its natural job.

### A/B With and Without Cache

Same 14 questions. Cache corpus: 36 pairs generated at ~$0.03.

| | **Cache OFF** | **Cache ON** |
|---|---|---|
| Gate accuracy | 100% | 100% |
| Cache hit rate (answerable) | — | 4/8 (50%) |
| **P50 total** | 1,750 ms | **1,345 ms** (−23%) |
| **P95 total** | 4,660 ms | **2,396 ms** (−49%) |
| **Avg total** | 1,967 ms | **1,434 ms** (−27%) |
| Avg gate | 1,018 ms | 497 ms (−51%) |
| Avg LLM | 906 ms | 383 ms (−58%) |

**P95 cut in half. Zero accuracy loss.** Cache hits now cost ~$0.000013 (one `choice` decision) instead of ~$0.0003 (gate) + ~$0.0005 (LLM). About **~60× cheaper per answered question** when the cache hits.

![Q&A cache ON vs OFF — P50, P95, average total latency](docs/images/cache-ab.svg)

Hit rate scales with corpus size — bumping pairs-per-chunk from 12 to 20 would push the Plants chapter from 36 to ~60 pairs and probably take hit rate from 50% to 70%+. Generation cost is still cents.

---

## When to Use Jev (And When Not To)

The honest answer, now that I've measured it:

**Jev is right when:**
- You need a **calibrated** yes/no/choice decision, not just "a model's guess at a number"
- You have **volume** — decisions amortize across many requests
- You can structure the decision as `noul`/`choice`/`score`/`box` instead of free-form text
- The failure mode of a wrong decision is expensive — moderation, compliance, agent routing, Q&A matching, evidence verification

**Jev is overkill when:**
- You're generating free-form prose (that's what a general LLM is for)
- You need the "reason" field — `noul` doesn't give one; you'd use chat for the clarification message
- You have low volume and the calibration isn't load-bearing

For my tutor, Jev earns its keep in two places:
1. **The three-way gate** — calibrated probability lets me set honest thresholds for M8-style evaluation
2. **The Q&A matcher** — this is Jev's sweet spot, and the latency numbers prove it

For "write a Grade 2 explanation of why climbers need support," GPT-4o-mini is the right tool. For "which of these six FAQ questions matches the student's query, with calibrated confidence?", Jev is the right tool. The whole project got better once I stopped using Jev to generate text and started using it to decide things.

---

## Architecture Summary

```
                          Student's voice
                                ↓
                      OpenAI Whisper (STT)
                                ↓
                 ┌──────────────────────────────┐
                 │  Q&A Cache Matcher           │
                 │  (Jev choice decision)       │
                 └──────────────────────────────┘
                      ↙ miss              ↘ hit
                                              → stored answer
                     ↓
               FAISS top-K
                     ↓
           ┌──────────────────┐
           │  Jev Gate        │  three noul decisions in one call
           │  clear + relevant + sufficient
           └──────────────────┘
                     ↓ accepted
            GPT-4o-mini (OpenAI)
             grounded answer + pages
                     ↓
           ┌──────────────────┐
           │  Jev Verifier    │  one noul decision
           │  supported?      │
           └──────────────────┘
                     ↓ supported
                OpenAI TTS
                     ↓
                 Answer spoken
```

Every edge that isn't "accepted" / "supported" either:
- Asks the student a clarifying question
- Refuses because the chapter doesn't support it
- Fails closed because the decision couldn't be verified

**No path leads to an unverified answer being spoken.** That's the whole point.

![End-to-end architecture — voice loop, cache-first match, Jev gate + verifier, LLM, TTS](docs/images/architecture.svg)

---

## Lessons

1. **Read the API docs, not the marketing.** I spent weeks using Jev through its chat-compatibility shim because that's what the model page showed. The real endpoint was documented in error messages. Probe the APIs you use.

2. **"Decision model" is a real category, not a style of prompting.** Calibrated outputs out of a model trained to produce them are categorically different from a chat model producing JSON. The latter is prompting; the former is architecture.

3. **Fail closed, always.** Every time I ignored it in a hurry I regretted it. The student seeing "I couldn't verify this" is strictly better than hearing something plausible but ungrounded — especially when the student is seven.

4. **Cache the expensive thing.** This is just computer science. The AI version — pre-generate Q&A, match-first serve, fall-through to the full pipeline on miss — is a direct port of the HTTP-cache pattern and the latency wins are the same shape.

5. **Separate the writer from the judge.** The LLM that generates the answer must not also decide whether its own answer is supported. The verifier being structurally different from the generator is a feature, not a bug.

6. **Measure before you optimize.** My "the gate is slow" instinct turned out to be about which API I was using, not the model itself. A/B eval sets are cheap; opinions are expensive.

---

## What's in the Repo

- `ingestion/` — PDF, OCR, normalization, chunker, chapter ingestion, Q&A corpus generation
- `retrieval/` — embedding service, FAISS vector store, retriever, Q&A matcher
- `decision/` — Jev decisions client (`/api/alpha/decisions`), decision policy state machine, legacy chat-based client kept for A/B comparisons
- `llm/` — grounded answer generator, question rewriter
- `voice/` — Whisper STT, OpenAI TTS
- `conversation/` — controller tying everything together, conversation context, clarification merging
- `evaluation/` — labelled dataset + CLI evaluator with `--qa-cache` and `--verify` flags
- `storage/` — SQLite schema and repository (`chapters`, `chunks`, `qa_pairs`)
- `app.py` — Streamlit UI (ingest, Q&A corpus build, text + voice ask)
- `tests/` — 170 fast tests + live-API tests behind markers

Python, Streamlit, FAISS, SQLite — nothing exotic. The architectural ideas port directly to any stack.

**Source:** [`github.com/<your-handle>/textbook-voice-tutor`](https://github.com/) *(replace with your actual repo URL before publishing)*

---

If you build with Jev, use the real decisions API. If you build a RAG tutor, put the gate before the LLM. If you serve a lot of questions, cache the common ones with a calibrated matcher. And measure every claim.

The experiment was worth it. My second-grader asked three questions about climber plants on the walk home from school the next day. She knew which page the answer came from.

---

### About

*[Your name] — [one-line bio: role, where you work, what you build. e.g., "ML engineer building education products. Previously at ___. Writes about applied AI at [twitter / linkedin / blog]."]*

**Attribution:** The scanned chapter used for this experiment is from the [NCERT Grade 2 EVS / Science](https://ncert.nic.in/) curriculum, used under its open-access terms for personal educational purposes. No chapter content is redistributed in the repo.

**Thanks:** To my daughter, who kept asking "but what does the book say?" until I built this.

---

### Appendix: Reproducing the numbers

```bash
# Build the Q&A cache
python -c "from ingestion.build_qa_cache import build_qa_cache; ..."

# Run the evaluator — chat API baseline (set JEV_MODEL=typesafe/jev-router in .env)
python -m evaluation.evaluate_pipeline --dataset evaluation/datasets/plants.json --out data/eval

# Decisions API (set JEV_MODEL=typesafe/jev-1.14)
python -m evaluation.evaluate_pipeline --dataset evaluation/datasets/plants.json --out data/eval

# With Q&A cache
python -m evaluation.evaluate_pipeline --dataset evaluation/datasets/plants.json --out data/eval --qa-cache
```

Each run writes a per-question CSV and a summary JSON to `data/eval/eval_<timestamp>.*`.
