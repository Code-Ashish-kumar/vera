# Step 2 Implementation Notes
## Vera AI Challenge — LLM-Based Composer

**Date:** September 27, 2026
**Status:** Complete — 42/42 harness checks passing; LLM composers built and wired
**Files produced:** `composer.py`, `reply_composer.py`, `.env`, `.env.example`
**Files modified:** `bot.py`

---

## What Step 2 covers

Step 2 replaces the two stub functions from Step 1 (`stub_compose` and `stub_reply_compose`) with real LLM-powered composers. The harness infrastructure built in Step 1 is untouched — every endpoint shape, lock pattern, and state machine remains identical. The only thing that changed is what generates the message body.

After Step 2:
- Every proactive message from `/v1/tick` is composed by `LLMComposer` via Groq
- Every conversation reply from `/v1/reply` is handled by `LLMReplyComposer` via Groq
- Both composers fall back to structurally valid stub output if `GROQ_API_KEY` is not set, so the harness never breaks

---

## Part 1 — File structure

```
VERA AI/
├── bot.py                  ← Step 1 harness + Step 2 wiring (imports composers)
├── composer.py             ← NEW: LLMComposer — proactive message generation
├── reply_composer.py       ← NEW: LLMReplyComposer — conversation reply handling
├── .env                    ← NEW: GROQ_API_KEY placeholder (fill before running)
├── .env.example            ← NEW: documentation copy of .env
└── run_warmup.py           ← unchanged from Step 1 (42/42 still passes)
```

---

## Part 2 — Architecture overview

```
/v1/tick
  │
  ├── [per trigger] deadline guard → skip if > 25s elapsed
  │
  ├── context snapshot (under lock, O(1))
  │
  └── asyncio.run_in_executor → LLMComposer.compose()
          │
          ├── _compose_inner()
          │     ├── load 4 contexts from store snapshot (always fresh)
          │     ├── trigger_kind → variant routing (5 families)
          │     ├── _build_context_block() → ~800-token prompt block
          │     ├── _call_with_retry() → Groq API (temp=0, max 1 retry)
          │     ├── validate_output() → 4 checks
          │     └── fill missing fields → return full action dict
          │
          └── _fallback() on any error → structurally valid stub

/v1/reply
  │
  ├── intent classification (regex, < 1ms)
  │
  ├── terminal intents (hostile / decline / auto×2) → deterministic, no LLM
  │
  └── asyncio.run_in_executor → LLMReplyComposer.compose_reply()
          │
          ├── _compose_llm()
          │     ├── _build_reply_context() → last 6 turns + merchant snapshot
          │     ├── Groq API call (temp=0, max_tokens=400)
          │     ├── _parse_json() → extract from output
          │     └── _normalise() → fix CTA, strip URLs, ensure action shape
          │
          └── _STUB_REPLIES[intent] on any error
```

---

## Part 3 — Design decisions

### 3.1 Two separate files, not one

The proactive composer (`composer.py`) and the reply composer (`reply_composer.py`) have fundamentally different jobs:

| | composer.py | reply_composer.py |
|---|---|---|
| Triggered by | `/v1/tick` | `/v1/reply` |
| Context input | 4 full contexts (category + merchant + trigger + customer) | 4 contexts + conversation history |
| Output shape | Full action dict (10 fields) | Reply dict (action + body + cta + rationale) |
| Token budget | 600 max_tokens | 400 max_tokens |
| Routing | trigger.kind → 5 variants | intent → terminal/LLM split |
| Few-shot examples | 6 gold cases from case-studies.md | 4 reply patterns inline |

Keeping them separate means each has a focused system prompt and can be tuned independently. A single monolithic composer would require conditional prompt logic that's harder to test and debug.

### 3.2 Groq with llama-3.3-70b-versatile

**Why Groq:**
- Fastest inference available for llama-3.3-70b (~200 tokens/sec vs ~30 tokens/sec on direct Llama hosting)
- Free tier is generous enough for a 60-minute test window at 10 req/sec
- No wheel/build issues on any Python version (pure HTTP client)
- Response latency: ~1-3s per call, well within the 25s tick deadline even for 3-4 triggers per tick

**Why llama-3.3-70b-versatile:**
- Strong instruction following — critical for the strict JSON output requirement
- Good at Hindi-English code-mix (trained on diverse multilingual data)
- `versatile` variant balances quality and speed better than `instant` for complex composition tasks

**temperature=0:** Required by the challenge brief for determinism. Set at the client wrapper level in both composers so it can never be accidentally overridden.

### 3.3 Five trigger variant families

Instead of 15+ separate prompt templates (one per trigger kind), triggers are grouped into 5 families that share the same compositional framing:

| Family | Trigger kinds | Core instruction | Primary lever |
|---|---|---|---|
| `research` | research_digest, regulation_change | Lead with source citation, connect to merchant's cohort, offer to do work | Curiosity + reciprocity |
| `event` | festival, weather, competitor, trend, IPL | Lead with the event, give data-backed recommendation, offer deliverable | Loss aversion + effort externalization |
| `performance` | perf_spike, perf_dip, milestone, reviews | Lead with the specific number, frame as opportunity/fix, offer action | Social proof + specificity |
| `relationship` | dormant, renewal, curious_ask, recurring | Short, one question, low-friction ask | Asking-the-merchant |
| `customer` | recall_due, lapsed, refill, appointment | Write from merchant's voice to customer, honor language + time pref | Personalization |

This keeps the system prompt manageable (~3KB total including few-shots) and makes it easy to add a new trigger kind by simply assigning it to the nearest family.

### 3.4 run_in_executor for sync Groq SDK in async FastAPI

The Groq Python SDK is synchronous (it uses `httpx` under the hood but exposes a sync API in the standard client). FastAPI endpoints are async. Calling a sync function directly inside an async endpoint blocks the event loop — the healthz route would hang while a tick is waiting on an LLM response.

The fix is `asyncio.get_event_loop().run_in_executor(None, sync_fn, *args)`, which runs the sync call in the default thread pool without blocking the event loop:

```python
action = await asyncio.get_event_loop().run_in_executor(
    None,
    llm_composer.compose,
    merchant_id, trigger_id, conv_id, customer_id, store_snapshot,
)
```

This means:
- Healthz continues to respond in < 1ms even while 3 tick LLM calls run in parallel
- Up to `os.cpu_count()` concurrent LLM calls can run in the thread pool
- The 25s deadline guard in the tick loop still works correctly

**Note:** The Groq SDK also has an async client (`AsyncGroq`). Using it would be cleaner, but requires changing every call site to `await`. `run_in_executor` is the correct bridge pattern for third-party sync SDKs in async frameworks and avoids forking a new client class.

### 3.5 Context assembly: terse block, not full JSON dump

Passing the raw context JSON directly to the LLM would work but wastes tokens on fields the LLM doesn't need for composition. A 50-merchant context JSON can be 8-15KB; passing all 50 into a single tick prompt would blow the 8K context window.

Instead, `_build_context_block()` extracts only composition-relevant fields:

```
Category: slug, voice, taboos, peer_stats, relevant digest item
Merchant: name, owner, locality, languages, subscription, CTR, views, calls, active_offers, signals, customer_aggregate
Trigger: kind, scope, urgency, payload
Customer: name, language_pref, state, last_visit, preferences (if populated)
```

This produces ~800 tokens per context block. Combined with the ~600-token system prompt and 6 few-shot examples (~700 tokens), a typical tick prompt is ~2100 tokens — well within llama-3.3-70b's 32K context window and cheap per call.

**The one exception:** for `research_digest` triggers, `_extract_digest_item()` does a targeted lookup of the specific digest item referenced by `trigger.payload.top_item_id`. This keeps the digest block from growing linearly with the number of items in `category.digest`.

### 3.6 Post-LLM validator — 4 checks

The validator runs after every LLM call before the result is accepted:

| Check | Why | On fail |
|---|---|---|
| Body is non-empty | Empty body = -2 harness penalty | Re-prompt with feedback |
| No URLs in body | -3 per URL per brief §F.4 | Re-prompt with feedback |
| CTA is a known valid value | Malformed CTA = 0 score | Re-prompt with feedback |
| Body numbers exist in context | Fabrication heuristic — numbers > 10 not in context are flagged | Re-prompt with feedback |

**Why only 1 retry:** The 25s tick deadline with a typical 2s LLM call leaves budget for at most 10 sequential calls across all triggers in a tick. With the deadline guard cutting off at 25s, allowing 2 retries per trigger would halve throughput. One retry catches the common failure modes (bad JSON, extra text, wrong CTA) without risk.

**Fabrication heuristic detail:** The check extracts all numbers from the full context block (using regex `\d+(?:\.\d+)?%?`) and checks whether numbers > 10 in the body are a subset of that set. Numbers ≤ 10 are allowed freely (slot numbers, turn counts). This catches the common case where the LLM invents a percentage or metric not in the data, without false-positives on natural small numbers.

### 3.7 Reply composer: terminal-first routing

The reply composer routes by intent before deciding whether to call the LLM at all:

```
hostile  → end (no LLM, < 1ms)
decline  → end (no LLM, < 1ms)
auto×2+  → wait 24h (no LLM, < 1ms)
auto×1   → LLM polite follow-up (~2s)
accept   → LLM action mode (~2s)
question → LLM grounded answer (~2s)
neutral  → LLM continuation (~2s)
```

Terminal intents (hostile, decline, repeated auto-reply) never touch the LLM. This matters for two reasons:
1. No latency risk for the most common quick-exit paths
2. No risk of the LLM producing a too-soft response for a merchant who explicitly said stop

For `accept` intent, the reply system prompt includes an explicit gold example of action-mode response ("On it. Drafting the patient-ed WhatsApp now...") that specifically contrasts with the Pattern D anti-example from the brief (re-qualifying after a commitment).

### 3.8 _normalise() post-processing in reply composer

Unlike the proactive composer which retries on validation failure, the reply composer normalises in-place — because a reply must come back within the 30s budget and there isn't time budget for a retry on a second LLM call that was itself triggered by a previous reply. Normalisation handles:

- Strip any URLs the LLM hallucinated (`re.sub(r'https?://\S+', '', body)`)
- Fix invalid action values → default to "send"
- Fix invalid CTA values → default to "open_ended"
- Add `wait_seconds: 86400` if action is "wait" and the field was missing

---

## Part 4 — Key decisions around the .env setup

### Why dotenv and not just os.environ

The Groq API key should not appear in shell history or be hardcoded. `python-dotenv` (already installed from Step 1 as a uvicorn dependency) reads `.env` at import time in `bot.py`:

```python
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")
```

This happens before the composer modules are imported, so `GROQ_API_KEY` is in `os.environ` by the time `_get_groq_client()` is called.

### Fallback behaviour without a key

Both `_get_groq_client()` functions raise `RuntimeError("GROQ_API_KEY not set")` when the key is absent or is the placeholder string. Both composers catch this specific error and return a structurally valid fallback instead of propagating it:

```python
except RuntimeError as e:   # GROQ_API_KEY not set
    logger.warning("LLM not configured: %s", e)
    return self._fallback(...)
```

This means the bot runs in "stub mode" without a key — exactly like Step 1. The harness passes, scores are low, but nothing breaks. When you add the real key and restart, LLM mode activates automatically with no code changes.

### How to activate LLM mode

1. Edit `.env` and replace `your_groq_api_key_here` with a real Groq API key
2. Restart the server: `py -3.13 -m uvicorn bot:app --host 0.0.0.0 --port 8080`
3. Verify by calling `GET /v1/metadata` — the `approach` field should say `LLM active: True`

---

## Part 5 — Prompt design details

### System prompt structure (composer.py)

```
1. Role statement (1 sentence)
2. Scoring dimensions — what the judge evaluates (mapped 1:1 to the 5 rubric dimensions)
3. Hard rules — verbatim from the challenge brief §5 + anti-pattern list
4. Valid CTA values (explicit enumeration so the LLM never invents one)
5. Output format (exact JSON schema with field names)
6. Gold examples (6 cases from case-studies.md — labels, context, output, explanation)
```

The gold examples are the most important section. They teach the exact specificity level, voice register, and CTA placement the judge rewards — more effectively than any instruction could. Each example includes a "WHY IT WORKS" annotation that directly maps back to the 5 scoring dimensions, helping the LLM understand the evaluation criteria from the output side.

### Variant instruction injection (user prompt)

Each trigger-kind family has a `VARIANT_INSTRUCTIONS[variant]` string injected into the user prompt at runtime:

```
VARIANT INSTRUCTION: This is a RESEARCH/COMPLIANCE digest trigger.
Lead with the specific finding or regulatory change as the 'why now'.
Cite the source (journal, circular, publication + date).
...
```

This is more effective than putting all variants in the system prompt (which dilutes attention) or using separate system prompts per variant (which requires re-sending the full few-shot context on every call). The variant instruction overrides the general instruction for the specific trigger kind being composed.

### Context block token budget

Typical composition prompt tokens:
- System prompt: ~620 tokens
- Few-shot examples: ~680 tokens
- Variant instruction: ~80 tokens
- Context block: ~800 tokens
- User prompt wrapper: ~60 tokens
- **Total: ~2,240 tokens in, 600 tokens out**

At Groq's llama-3.3-70b rate, this costs roughly $0.00020 per composition. A full 60-minute test window with 100 triggers would cost ~$0.02 in LLM calls.

---

## Part 6 — What to expect at different stages

### Without GROQ_API_KEY (fallback mode)
- All 42 harness checks: PASS
- Tick returns valid action with stub body
- Reply returns correct intent routing (hostile→end, accept→send, etc.)
- Composition quality: same as Step 1 (low — just template strings)
- Judge scoring: structural correctness only, ~5-10/50 per message

### With GROQ_API_KEY set
- All 42 harness checks: PASS (unchanged)
- Tick returns LLM-generated body with specificity, category fit, trigger relevance
- Reply returns LLM-generated conversation continuation
- Composition quality: target 35-45/50 per message (limited by what's in the seed context)
- Auto-reply detection: LLM generates better "flag for owner" messages
- Accept intent: LLM correctly switches to action mode

### Expected judge simulator output (with key)
Running `judge_simulator.py` with `TEST_SCENARIO = "phase2_short"`:
- Warmup: PASS (5 categories, 10 merchants accepted)
- Tick: 3 actions returned (one per active trigger)
- LLM scoring: each action scored on 5 dimensions, rationale displayed
- Auto-reply scenario: bot detects, sends polite follow-up, then waits
- Intent transition: bot moves to action on "let's do it"

---

## Part 7 — How to run end-to-end (with key)

```bash
# 1. Fill in your key
notepad .env
# set GROQ_API_KEY=gsk_...

# 2. Start server
py -3.13 -m uvicorn bot:app --host 0.0.0.0 --port 8080

# 3. Confirm LLM is active
curl http://localhost:8080/v1/metadata
# "LLM active: True" in approach field

# 4. Run harness checks (must stay 42/42)
py -3.13 run_warmup.py

# 5. Run judge simulator (warmup scenario — no LLM judge needed)
# Edit judge_simulator.py: BOT_URL = "http://localhost:8080", TEST_SCENARIO = "warmup"
py -3.13 judge_simulator.py

# 6. Run judge simulator with LLM scoring (needs LLM_API_KEY in simulator too)
# Edit judge_simulator.py: LLM_PROVIDER = "groq", LLM_API_KEY = "gsk_...", TEST_SCENARIO = "phase2_short"
py -3.13 judge_simulator.py
```

---

## Part 8 — Challenges and resolutions

| Challenge | Root cause | Resolution |
|---|---|---|
| Groq SDK is synchronous, FastAPI is async | `Groq.chat.completions.create()` is blocking | Wrapped in `asyncio.run_in_executor(None, ...)` — runs in thread pool without blocking event loop |
| LLM sometimes returns JSON wrapped in markdown fences | Model-level habit despite "JSON only" instruction | `_parse_json()` strips ` ```json ` / ` ``` ` before parsing, then falls back to regex `{...}` extraction |
| Numbers like slot times (1, 2, Wed, Thu) flagged as fabricated | Fabrication heuristic was too strict for small numbers | Threshold set to > 10 — numbers ≤ 10 allowed freely (slot indices, turn counts, small quantities) |
| LLM occasionally uses wrong CTA string ("yes_no" instead of "binary_yes_no") | Model generalises from instruction but doesn't always match exact enum | Validator rejects, re-prompts with exact valid values listed again; on second failure `_normalise()` force-sets to `open_ended` |
| `bool("your_groq_api_key_here")` is True | Python string truthiness check | Added explicit placeholder string check: `raw_key != "your_groq_api_key_here"` |
| Hindi-English code-mix not guaranteed from model | Instruction alone isn't enough — model defaults to English | Language check is a soft warning in the validator (not a hard fail) because some `hi`-tagged merchants actually prefer English in practice; re-prompt on hard-fail risks burning the retry budget on a marginal issue |

---

## Part 9 — What Step 3 changes

Step 2 leaves two known gaps that Step 3 addresses directly:

1. **Intent classification is still regex-based.** The `_classify_reply_intent()` in bot.py covers the common cases but misses sarcastic "yes", mixed-language declines, and nuanced questions. Step 3 will replace it with a lightweight LLM classification call (fast/cheap model, single-token output).

2. **Auto-reply detection uses exact-match only.** `_is_auto_reply_repeat()` requires the same message verbatim. Real WA Business auto-replies sometimes have small variations (timestamp appended, emoji varies). Step 3 adds Jaccard similarity (threshold 0.85) to catch near-identical repeats.

Everything else in the composer (prompt, validator, routing) feeds directly into Steps 3 and 4 without modification.
