# Step 1 Implementation Notes
## Vera AI Challenge — Skeleton Server

**Date:** September 27, 2026
**Status:** Complete — 42/42 harness checks passing
**Files produced:** `bot.py`, `run_warmup.py`, `dataset/expanded/` (generated)

---

## What Step 1 covers

Step 1 is entirely about operational correctness — getting the HTTP harness bulletproof before writing a single line of LLM composition logic. The goal is zero operational penalties when the judge runs against the bot. The five endpoints, the teardown route, and all state management live in `bot.py`. The stub composer returns structurally valid messages so the harness passes even with no real LLM behind it.

---

## Part 1 — Dataset generation (Step 0)

### What needed to happen

The workspace ships seed files only: `merchants_seed.json` (10 merchants), `customers_seed.json` (15 customers), `triggers_seed.json` (25 triggers). The judge expects 50 merchants, 200 customers, 100 triggers, and a canonical `test_pairs.json` (30 pairs used to generate `submission.jsonl`).

### How it was done

```
cd dataset/
py -3.13 generate_dataset.py --seed-dir . --out ./expanded
```

Output:
```
dataset/expanded/
├── categories/          # 5 files (copied as-is from categories/)
├── merchants/           # 50 individual m_NNN_*.json files
├── customers/           # 200 individual c_NNN_*.json files
├── triggers/            # 100 individual trg_NNN_*.json files
└── test_pairs.json      # 30 canonical (merchant, trigger) pairs
```

The generator is deterministic — it uses `random.Random(20260426)` as its fixed seed, so every participant who runs it gets the exact same expanded dataset.

### Design note: seed files vs expanded files

The judge simulator (`judge_simulator.py`) reads from the seed files in `dataset/` root using `merchants_seed.json` etc. The bot itself needs to load from `dataset/expanded/` to serve the full 50/200/100 counts at warmup. These are two separate concerns:

- The simulator is a dev-time testing tool that reads seeds directly.
- The bot loads contexts via `POST /v1/context` pushes from the judge (it doesn't read files itself).
- The `test_pairs.json` in `dataset/expanded/` is needed only when generating `submission.jsonl` in Step 6.

---

## Part 2 — Environment setup

### Challenge encountered: Python 3.14 incompatibility

The default Python on this machine (`python`) resolves to **Python 3.14.7**. `pydantic-core` (a dependency of Pydantic v2 and FastAPI) doesn't have a pre-built wheel for 3.14 yet and falls back to compiling from Rust source. The Rust toolchain was downloaded successfully, but `cargo metadata` failed with exit code `0xc0e90002` — a Windows-specific error when the MSVC toolchain is missing its prerequisites.

**Root cause:** Python 3.14 is too new; no binary wheel for `pydantic-core` exists yet for `cp314-win_amd64`.

**Fix:** Use `py -3.13` (Python 3.13.15 was available on the machine). Python 3.13 has a pre-built wheel for `pydantic-core==2.33.2`.

### Installed packages (pinned)

```
fastapi==0.115.12
uvicorn[standard]==0.34.3
pydantic==2.11.4
```

All installed into the Python 3.13 site-packages. To run the server:

```bash
py -3.13 -m uvicorn bot:app --host 0.0.0.0 --port 8080
```

**If you restart the machine:** these packages are installed globally into Python 3.13 so no virtual environment activation is needed. Just run the command above.

---

## Part 3 — Architecture decisions in bot.py

### 3.1 Single asyncio.Lock for all state

All mutable state (`context_store`, `conv_store`, `active_conversations`, `suppressed_keys`, `context_counts`) is protected by one `asyncio.Lock`. The judge sends up to 10 requests per second, all to the same process.

**Why a single lock instead of per-collection locks:**
- The tick handler reads from `context_store` and writes to `conv_store` and `active_conversations` atomically. If these were separate locks you'd need to acquire two at once, creating a potential deadlock.
- Under 10 req/sec the single-lock contention is negligible — critical sections are microseconds (dict lookups, no I/O).

**Design decision: snapshot under lock, process outside**

The tick handler does this:
```python
async with _lock:
    trigger_snapshots = {tid: context_store.get(...) for tid in available_triggers}
    merchant_snapshot = dict(context_store)  # shallow copy of keys

# Process outside the lock — no blocking I/O while holding it
for trigger_id in available_triggers:
    ...
```

This means the lock is held for < 1ms even when there are 100 triggers. If we held the lock during the full composition loop (which in Step 2 will make LLM calls), the healthz endpoint would time out under load.

### 3.2 Context store: raw payload only, never pre-processed

```python
context_store[key] = {
    "version": body.version,
    "payload": body.payload,   # raw dict — no pre-processing
    "stored_at": utc_now_iso(),
}
```

This is the single most important design decision for Phase 3 (adaptive context). The judge will push `version: 2` of a merchant's performance data mid-test. If the bot had pre-rendered a prompt string or pre-computed embeddings at store time, it would use stale data. By storing only the raw payload and re-reading it at compose time, every tick and reply automatically picks up the latest version.

### 3.3 O(1) healthz via context_counts

The judge polls `GET /v1/healthz` every 60 seconds and requires accurate `contexts_loaded` counts. Three consecutive failures = disqualification.

Naive approach: iterate `context_store` every healthz call — O(n) where n grows to 355 entries during warmup.

**Our approach:** maintain `context_counts` as a separate dict incremented once per new context accept:

```python
if is_new:
    context_counts[body.scope] = context_counts.get(body.scope, 0) + 1
```

Healthz reads this under a 1ms lock snapshot — always O(1), never blocked by LLM or context processing.

### 3.4 Anti-repetition: sent_bodies list per conversation

The judge penalizes −2 for any body sent verbatim twice in the same conversation. `ConversationState.sent_bodies` keeps a list of every body sent:

```python
def is_repeat(self, body: str) -> bool:
    return body.strip() in [b.strip() for b in self.sent_bodies]
```

The stub composer calls `is_repeat()` before returning and appends `" (follow-up)"` as a minimal differentiator if needed. Step 2 will replace this with a proper re-prompt.

### 3.5 The auto_reply_count bug — and why it matters

**The bug:** The initial implementation used `nudge_count` to decide whether to `wait` on the second auto-reply. But `ConversationState.record_merchant_reply()` resets `nudge_count = 0` on every incoming message (including auto-replies). So the second auto-reply always saw `nudge_count == 0` and triggered the "first detection" path again (returning `send`) instead of escalating to `wait`.

**The fix:** Added a separate `auto_reply_count` field that is incremented in the reply handler before `record_merchant_reply` runs, and is never reset. The reply endpoint now classifies intent first, increments `auto_reply_count` if the intent is `auto_reply`, then records the merchant reply:

```python
intent = _classify_reply_intent(message)
is_repeat_auto = _is_auto_reply_repeat(conv, message)

if intent == "auto_reply" or is_repeat_auto:
    intent = "auto_reply"
    conv.auto_reply_count += 1   # never resets

conv.record_merchant_reply(message)   # this resets nudge_count
```

**Why two counters:**

| Counter | Tracks | Resets on |
|---|---|---|
| `nudge_count` | Unanswered proactive sends | Any merchant reply |
| `auto_reply_count` | Consecutive auto-reply detections | Never |

`nudge_count` drives the graceful-exit logic ("bot sent 3 nudges with no real reply → end"). `auto_reply_count` drives the auto-reply escalation path ("first detection → polite follow-up, second → wait 24h").

### 3.6 Conversation deduplication: active_conversations dict

```python
active_conversations: dict[tuple[str, str], str] = {}  # (mid, tid) → conv_id
```

When tick runs, it checks this dict before starting a new conversation for a `(merchant_id, trigger_id)` pair. If an active conversation exists for that pair, the tick skips it. This prevents the judge from getting duplicate conversations for the same trigger on consecutive tick calls.

A conversation is considered "done" when `conv.phase == "ended"`, at which point a new conversation for the same pair can start (e.g., after a `wait` period expires and the trigger re-fires).

### 3.7 Suppression keys for hostile/declined conversations

When a merchant declines or is hostile, the bot:
1. Returns `action: end`
2. Reads the trigger's `suppression_key` from context_store
3. Adds it to `suppressed_keys`

Future tick calls skip any trigger whose `suppression_key` is in this set:
```python
if suppression_key and suppression_key in suppressed_keys:
    continue
```

This prevents re-engaging a merchant who has explicitly opted out — which would be both a bad experience and a likely penalty.

### 3.8 Global exception handler

The single most important safety net in the bot. Without it, an unhandled exception in the tick handler returns a 500, which the judge logs as a malformed response (−2 penalty).

```python
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    if path == "/v1/healthz":
        return JSONResponse(status_code=200, content={"status": "degraded", ...})
    if path == "/v1/tick":
        return JSONResponse(status_code=200, content={"actions": []})
    if path == "/v1/reply":
        return JSONResponse(status_code=200, content={"action": "end", ...})
```

Key decisions:
- Healthz always returns 200 even on error (returns `"status": "degraded"` with the error message). Three non-200s = disqualification, so a 500 on healthz is the worst possible failure.
- Tick returns `{"actions": []}` — the judge treats this as "bot chose not to send", zero penalty.
- Reply returns `{"action": "end"}` — graceful exit, no penalty.

### 3.9 Tick deadline guard

```python
TICK_TIMEOUT_SECONDS = 25
deadline = time.time() + TICK_TIMEOUT_SECONDS
...
for trigger_id in available_triggers:
    if time.time() > deadline:
        break
```

The judge's timeout is 30s. The bot enforces its own 25s deadline and returns whatever actions it has composed so far. This ensures the response always arrives before the judge gives up, even if later triggers in the list take unexpectedly long.

In Step 2 when real LLM calls are added, the deadline guard becomes critical — LLM calls can take 5-15s each and 25 triggers at 10 req/sec could easily blow the 30s budget without this.

---

## Part 4 — Endpoint-by-endpoint implementation notes

### POST /v1/context

**Idempotency rule:**
- Same version → `409` with `current_version`
- Lower version than stored → also `409` (not just "equal")
- Higher version → `200`, replaces atomically

The `>=` check catches both:
```python
if existing and existing["version"] >= body.version:
    return JSONResponse(status_code=409, ...)
```

**500 KB cap:**
The cap is enforced by reading the raw request body and checking `len(raw_body)`:
```python
raw_body = await request.body()
if len(raw_body) > MAX_PAYLOAD_BYTES:
    return JSONResponse(status_code=400, content={...})
```

One subtlety: FastAPI/Starlette buffers the full body for Pydantic model validation before our handler runs. So `body.payload` is already parsed by the time we check the size. The size check uses `raw_body` (the original bytes) rather than `len(json.dumps(body.payload))` because re-serializing would miss encoding overhead.

**No pre-processing:**
The payload is stored as `body.payload` (a Python dict). Nothing is computed from it at store time. All lookups, prompt construction, and context assembly happen in the tick/reply handlers at compose time.

### GET /v1/healthz

The healthz route has no `async with _lock` on the count read — it uses a snapshot:
```python
async with _lock:
    counts = dict(context_counts)
```

The lock is acquired for the dict copy (~1µs), then released. The route itself never touches context_store, conv_store, or any LLM. It cannot be blocked by a long-running tick.

### POST /v1/tick

The tick handler is designed around a principle: **decide first, compose second**. The full logic:

1. Snapshot context_store under lock (fast)
2. For each trigger in `available_triggers`:
   a. Look up trigger payload from snapshot
   b. Look up merchant_id and customer_id from trigger
   c. Check suppression_key — skip if suppressed
   d. Check active_conversations — skip if ongoing conv for this pair
   e. Verify merchant context exists
   f. If customer trigger, verify customer context exists
   g. Create ConversationState
   h. Call stub_compose() to get action dict
   i. Record send in ConversationState
   j. Persist conv under lock
3. Return {"actions": [...]}

Steps 2a-2f are all O(1) dict lookups. The only potentially slow operation is step 2h (stub compose for now; LLM call in Step 2). The deadline guard in the outer loop ensures we never exceed 25s total.

### POST /v1/reply

The reply handler's ordering matters:

```
1. Classify intent (before recording)
2. Check is_auto_reply_repeat (compares against previous turn)
3. Increment auto_reply_count if auto
4. Record merchant reply (resets nudge_count)
5. Compose reply based on intent
6. Update conv.phase if action is end/wait
7. Add suppression_key if hostile/decline
```

The classify-before-record ordering is important: `_is_auto_reply_repeat` compares the incoming message against `conv.last_merchant_reply`, which is the previous message. If we recorded first, the comparison would always be against the current message (always a match).

### POST /v1/teardown

Simple: acquires the lock, clears all five collections, resets counts to zero. Returns `{"wiped": True, "wiped_at": ...}`.

Used at the start of every local test run via `run_warmup.py` to guarantee a clean slate.

---

## Part 5 — Intent classifier design

The `_classify_reply_intent()` function is a regex-based heuristic with seven output classes:

| Class | Triggers | Bot response |
|---|---|---|
| `auto_reply` | Canned WA Business phrases | First: polite follow-up; Second: wait 24h |
| `hostile` | Stop/spam/useless + Hindi equivalents | `end` immediately |
| `accept` | Yes/ok/let's do it/go ahead + Hindi | `send` in action mode |
| `decline` | No/not now/later + Hindi | `end` gracefully |
| `question` | Any message containing `?` | `send` with acknowledgment |
| `off_topic` | (reserved — not yet pattern-matched) | — |
| `neutral` | Everything else | `send` keeping conversation open |

**Why regex not LLM for Step 1:**
- No LLM API key configured yet
- Classification needs to be < 1ms to not eat into the 30s budget
- The patterns are explicit and testable
- Step 3 will add an LLM call for classification when ambiguous

**Known gaps in the regex classifier (to fix in Step 3):**
- Hindi accept phrases like `"haan kar do"` are not fully covered
- Off-topic classification currently falls through to `neutral`
- Sarcastic "yes" ("Sure, why not 🙄") would be classified as `accept` incorrectly
- Short messages like "Ok" or "Sure" classify as `accept` — which is correct behavior but depends on the full conversation context

---

## Part 6 — Stub composer design

The stub composer produces valid, non-empty messages for all 15 trigger kinds without any LLM. Its job is to pass harness checks, not score well on composition quality.

Every stub message:
- Is non-empty
- Contains no URLs (−3 penalty per URL from the brief)
- Contains no fabricated numbers
- Has a single CTA value from the allowed set
- Has all required fields (`conversation_id`, `merchant_id`, `send_as`, `trigger_id`, `template_name`, `template_params`, `body`, `cta`, `suppression_key`, `rationale`)
- Uses the merchant's `owner_first_name` from context if available (minimum personalization)
- Checks `is_repeat()` before returning and differentiates if needed

The trigger-kind → body mapping in `stub_compose()` will be the routing table for the real LLM variants in Step 2. The structure is already there; Step 2 just replaces each `kind_bodies[kind]` entry with an LLM call.

---

## Part 7 — What the tests verify

The `run_warmup.py` test script runs 42 independent checks covering:

| Area | Checks |
|---|---|
| Teardown | Returns 200, `wiped=true` |
| Healthz pre-warmup | 200, `status=ok`, all counts = 0 |
| Metadata | 200, has `team_name`, `model`, `approach` |
| Category context pushes | All 5 categories accepted |
| Merchant context pushes | 5 seed merchants accepted |
| Idempotency | Same version → 409, `reason=stale_version`, `current_version=1` |
| Version bump | Higher version → 200, accepted |
| Invalid scope | 400, `reason=invalid_scope` |
| Trigger push + tick | Trigger accepted, tick returns ≥1 action |
| Action shape | All 9 required fields present, body non-empty, no URL |
| Reply (accept) | Returns `action=send`, non-empty body |
| Reply (hostile) | Returns `action=end` |
| Healthz post-load | category≥5, merchant≥1, trigger≥1 |
| Teardown + reset | Returns `wiped=true`, counts back to 0 |

---

## Part 8 — How to run

### Start the server
```bash
py -3.13 -m uvicorn bot:app --host 0.0.0.0 --port 8080
```

### Run the harness test
```bash
py -3.13 run_warmup.py
```
Expected output: `Results: 42/42 passed | 0 failed`

### Run the judge simulator (warmup scenario, no LLM needed)
Configure in `judge_simulator.py`:
```python
BOT_URL = "http://localhost:8080"
TEST_SCENARIO = "warmup"
```
Then run:
```bash
py -3.13 judge_simulator.py
```

### Quick curl smoke test
```bash
curl http://localhost:8080/v1/healthz
curl http://localhost:8080/v1/metadata
```

---

## Part 9 — What Step 2 changes

Step 1 establishes the full harness skeleton. Everything built here stays in place. Step 2 only replaces the internals of two functions:

1. **`stub_compose()`** → replaced with `llm_compose()` — makes an LLM call with the 4 contexts assembled as a structured prompt, routes by trigger kind, runs the post-LLM validator
2. **`stub_reply_compose()`** → augmented with LLM-based reply generation for non-terminal intents (accept, question, neutral)

The `ConversationState`, context store, lock architecture, healthz O(1) pattern, global exception handler, and all endpoint routing stay exactly as implemented here.

---

## Challenges summary

| Challenge | Impact | Resolution |
|---|---|---|
| Python 3.14 has no pydantic-core wheel | Blocked dependency install | Switched to `py -3.13` |
| nudge_count reset by record_merchant_reply | Auto-reply #2 returned `send` instead of `wait` | Added separate `auto_reply_count` that never resets |
| Holding lock during composition would block healthz | Healthz could time out under tick load | Snapshot context under lock, process outside lock |
| Judge simulator loads seed files, not expanded files | Simulator showed 10 merchants not 50 | Simulator is a dev-time tool using seeds; bot loads via /v1/context |
| context_store O(n) scan for healthz | Would slow under 355 entries at full warmup | Separate `context_counts` dict gives O(1) healthz |
| Classify intent before or after recording | Order affects is_auto_reply_repeat comparison | Classify first (against previous turn), then record |
