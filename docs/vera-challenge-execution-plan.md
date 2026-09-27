# magicpin "Vera" AI Challenge — Execution Plan

## 0. What's actually being judged (read this first)

This isn't "write a good chatbot." It's a **message-composition + conversation-policy** problem with a hard technical contract:

- A `compose(category, merchant, trigger, customer?) → message` function, exposed behind **5 HTTP endpoints** the judge calls.
- Scored on **5 dimensions × 10 = 50 points per message**: Specificity, Category fit, Merchant fit, Trigger relevance, Engagement compulsion.
- Plus **operational penalties** (timeouts, healthz failures, malformed JSON, repetition) that can silently tank a great composer — max −20 points total.
- Plus a **live-adaptation bonus** (new context injected mid-run, not seen during dev) and, for the top 10, a **3-scenario multi-turn replay** worth up to +30.

So the plan has two tracks that must both be solid: **(A) the composer** (does it write great messages) and **(B) the harness** (does the server behave correctly under the judge's protocol). Teams lose more points to (B) than to (A) — most entrants will have a decent LLM prompt; few will nail idempotent context storage, sub-30s ticks, and graceful auto-reply/intent-transition handling.

---

## 1. Scoring model — reverse-engineered target

| Source | Max points | What it actually rewards |
|---|---:|---|
| Phase 2 base rubric (30 test pairs × 5 dims × 10) | 1,500 pts total (50/message) | Concrete numbers, category-correct voice, personalization to *this* merchant, explicit "why now," a reply-worthy hook |
| Phase 3 adaptation bonus | +5/dimension per adapted message | Using injected digest/perf/trigger updates without hallucinating; not sending stale copy |
| Phase 4 replay (top 10 only) | +30 total | Auto-reply detection, intent-transition handling, staying on-mission under hostility |
| Operational penalties | −20 max | Timeouts (−1 each), healthz failures (−10 for 3× consecutive), malformed JSON (−2 each), empty body (−2), verbatim repeats (−2 each) |

**Implication for prioritization:** get operational correctness to zero-penalty first (it's binary and easy to lose points on for silly reasons), then spend the rest of the time on composer quality, then on the two conversation-policy challenges (auto-reply detection, intent transition) since those are explicitly tested in Phase 4.

---

## 2. Architecture

```
                     ┌───────────────────────────────────────────┐
                     │              Your Bot (public URL)         │
Judge ──HTTP/JSON──► │  FastAPI/Express app                       │
                     │  ┌───────────────────────────────────────┐ │
                     │  │ Context Store (in-memory dict is fine, │ │
                     │  │ keyed by (scope, context_id) → version,│ │
                     │  │ payload)                                │ │
                     │  │ NEVER pre-render prompts at store time: │ │
                     │  │ always re-read raw payload at compose   │ │
                     │  │ time so Phase 3 updates are picked up   │ │
                     │  └───────────────────────────────────────┘ │
                     │  ┌───────────────────────────────────────┐ │
                     │  │ Conversation Store                     │ │
                     │  │ conversation_id → {turns[], state,     │ │
                     │  │ last_bodies[] (anti-repeat), phase}    │ │
                     │  └───────────────────────────────────────┘ │
                     │  ┌───────────────────────────────────────┐ │
                     │  │ Composer (LLM call)                    │ │
                     │  │ - trigger-kind → prompt-variant router │ │
                     │  │ - retrieval over category.digest       │ │
                     │  │ - post-LLM validator (CTA shape, len,  │ │
                     │  │   language match, no-repeat, no-URL-   │ │
                     │  │   fabrication)                         │ │
                     │  │ - re-prompt-on-fail (max 1 retry)      │ │
                     │  │ - temperature=0 (determinism required) │ │
                     │  └───────────────────────────────────────┘ │
                     │  ┌───────────────────────────────────────┐ │
                     │  │ Conversation Policy Layer               │ │
                     │  │ - auto-reply detector (verbatim-repeat  │ │
                     │  │   heuristic + canned-phrase classifier) │ │
                     │  │ - intent classifier (accept/decline/    │ │
                     │  │   ask-later/curveball)                  │ │
                     │  │ - turn budget (max nudges before end)   │ │
                     │  └───────────────────────────────────────┘ │
                     └───────────────────────────────────────────┘
```

**Stack recommendation:** the testing brief ships a Python/FastAPI skeleton, so Python is the path of least friction (async, 30s timeout budget is easy to manage with `asyncio`). Given your MERN background, an Express/Node version is equally viable if you'd rather move fast in familiar territory — the contract is just JSON over HTTP, nothing Python-specific is required. Pick whichever you can make *reliable* fastest; this is not a place to learn a new stack under time pressure.

**LLM choice:** any frontier model works (Claude/GPT/Gemini/DeepSeek). Recommendation:
- **One fast, cheap model** for classification/routing (auto-reply detection, intent classification, trigger-variant selection). Keep this call async so it doesn't add serial latency.
- **One stronger model** for the actual composition call.

Both calls together must stay inside the 30s tick budget, including network round-trips and any provider rate-limit back-off. If you can parallelize the classification and a speculative composition call, do it — start composing with your best guess at the variant while the classifier is in flight.

Set **`temperature=0`** on both. The brief requires determinism. Bake it into your LLM client wrapper so it can't be forgotten.

---

## 3. Build sequence (do these in order)

### Step 0 — Generate the full dataset (prerequisite, do before anything else)

The workspace ships *seed* files only (`merchants_seed.json`, `customers_seed.json`, `triggers_seed.json`). Your bot needs the full expanded dataset — 50 merchants, 200 customers, 100 triggers, individual JSON files, and the 30-pair `test_pairs.json` used to generate `submission.jsonl`.

Run the generator from the `dataset/` directory:

```bash
cd dataset
python generate_dataset.py --seed-dir . --out ./expanded
```

This produces:
```
dataset/expanded/
├── categories/         # 5 files (copied as-is from categories/)
├── merchants/          # 50 individual m_NNN_*.json files
├── customers/          # 200 individual c_NNN_*.json files
├── triggers/           # 100 individual trg_NNN_*.json files
└── test_pairs.json     # 30 canonical (merchant, trigger) pairs
```

The generator is **deterministic** (fixed seed `20260426`) so your output matches every other participant's. Run it once, commit the output, point your bot at `dataset/expanded/`.

> **Do not skip this step.** Steps 1–6 all assume the expanded directory structure exists.

---

### Step 1 — Skeleton + operational correctness (do this before any LLM work)

Get a passing harness score with zero penalties using stubbed responses. Every subsequent step builds on this foundation.

**Implement all 5 endpoints** exactly per `challenge-testing-brief.md` §2:

- **`POST /v1/context`** — idempotent by `(scope, context_id, version)`:
  - Higher version replaces prior version atomically.
  - Return `409` with `current_version` on stale push (same or lower version).
  - Enforce the 500 KB payload cap — return `400` gracefully, don't crash.
  - Store raw payload only; **never pre-render or embed at store time** (would cause stale context in Phase 3).

- **`GET /v1/healthz`** — must reflect real `contexts_loaded` counts after warmup:
  ```json
  { "status": "ok", "uptime_seconds": ..., "contexts_loaded": { "category": 5, "merchant": 50, "customer": 200, "trigger": 100 } }
  ```
  Three consecutive failures = disqualification for that test slot. Make this route trivially fast: O(1) counter increment on every `/v1/context` accept, never dependent on an LLM call.

- **`GET /v1/metadata`** — static; keep `version`/`approach` accurate. The judge reads `approach` when interpreting edge cases.

- **`POST /v1/tick`** — **must return within 30s even with nothing to send** → `{"actions": []}`. Never block on a slow LLM call inside tick. If composition is taking too long, bail with empty actions rather than timing out. The judge does not penalize restraint; it penalizes timeouts.

- **`POST /v1/reply`** — same 30s budget; must return `send` / `wait` / `end`. Always return one of the three valid action values; never return an unexpected shape.

- **`POST /v1/teardown`** (optional but required per testing-brief §11) — wipe all in-memory state. The judge may call this at the end of the test. If you don't implement it, the judge's cleanup step silently fails which can cause state bleed into re-runs. Two lines to implement; worth doing.

**Anti-repetition:** store every `body` you've ever sent per `conversation_id`. Before returning any action, check against this list. Never resend verbatim (−2 penalty each time, explicitly tested in Phase 2 and Phase 4).

**Rate limits:** the judge sends ≤10 req/sec. Make sure your context store isn't doing anything blocking/synchronous that would serialize requests under load. In-memory dict with an asyncio lock is sufficient.

**Test this step with `judge_simulator.py` before writing a single line of real composer logic.** Configure `BOT_URL`, `LLM_PROVIDER`, and `LLM_API_KEY` in the simulator and run the `warmup` scenario. Get non-zero, non-penalized scores on the harness alone using stubbed template responses before moving to Step 2.

---

### Step 2 — Category-aware composer (core of the score)

This is where most of the 1,500 base points live.

**One prompt template** that ingests all 4 contexts as structured JSON and outputs the 5 required fields (`body`, `cta`, `send_as`, `suppression_key`, `rationale`).

**Trigger-kind routing** (§13 of the challenge brief recommends this explicitly): a `research_digest` trigger, a `perf_dip` trigger, and a `recall_due` trigger want different framings and different compulsion levers. Build one prompt variant per trigger kind — there are ~15 kinds total. Don't use one generic prompt for all of them.

**Hard constraints from §5 of the brief** — bake these into the prompt as explicit rules, not hopes:
- Single binary CTA (`YES/STOP`) for action triggers; no CTA for pure-information triggers; multi-choice allowed only for booking flows.
- No fabrication — if a number, citation, or competitor name isn't in the provided JSON, it must not appear in the output.
- Specificity requirement — every message must anchor on at least one verifiable fact from the contexts (number, date, headline, source citation).
- Voice and taboo list per category — inject `category.voice.tone`, `category.voice.vocab_allowed`, and `category.voice.taboos` directly into the system prompt.
- Language match — if `merchant.identity.languages` includes `hi`, compose in Hindi-English code-mix.
- `temperature=0` — set in your LLM client wrapper, not as a per-call parameter. It must never be accidentally omitted.

**Retrieval for `research_digest` triggers:** the `category.digest` array is included in the full category payload. For digest triggers, don't dump the entire digest array into the prompt — look up the specific item referenced in `trigger.payload.top_item_id` (or the top-ranked item by relevance to the merchant's signals). This keeps the prompt tight and makes Phase 3 adaptation natural: a new digest version simply has new items in the array, and your lookup picks them up automatically.

**Post-LLM validator** (deterministic, no LLM needed — fast and cheap):

| Check | Action on fail |
|---|---|
| Multiple CTAs present | Re-prompt (max 1 retry), then return empty action |
| Body contains a number or source string not present in the provided context JSON | Flag as fabrication, re-prompt |
| Language doesn't match `identity.languages` | Re-prompt with language instruction emphasized |
| Body is empty | Re-prompt |
| Body matches a prior send in this `conversation_id` (exact or >0.9 similarity) | Re-prompt with explicit instruction to vary |
| More than 1 URL in body not traceable to context | Re-prompt |

**Few-shot anchoring:** include the Appendix A and B gold examples from `challenge-brief.md` and the 10 case studies from `examples/case-studies.md` as few-shot references in your system prompt. They show the exact tone, structure, and specificity level the judge rewards with 10/10s — including where near-misses lose a point (e.g., the multi-choice CTA in Case Study 2 that cost 1 point on engagement compulsion). Don't copy them verbatim; use them to calibrate the style.

---

### Step 3 — Conversation policy layer (targets Phase 4 directly)

Three behaviors the brief explicitly calls out as differentiators over production Vera. Build these deliberately; don't leave them to prompt luck. Phase 4 tests all three in isolation.

**1. Auto-reply detection**

If ≥2 of the last N merchant messages in a conversation are near-identical to each other or to a known canned-auto-reply phrase, treat the conversation as auto-reply. Suggested threshold: Jaccard similarity >0.85 between any two recent turns counts as a match.

Common canned patterns to seed your classifier (from the brief's Pattern B example and real WA Business canned replies):
- "Thank you for contacting us. Our team will respond shortly."
- "Aapki jaankari ke liye bahut-bahut shukriya…"
- "I am an automated assistant…"
- "We will get back to you within 24 hours."

**Policy on detection:**
1. First detection: send one graceful human-sounding follow-up ("Samajh gayi — ek aur check kar lungi…").
2. Second detection in same conversation: return `{"action": "end"}` with a polite rationale. Do not re-ask or burn additional turns.

This directly fixes Vera's #1 documented pain point.

**2. Intent transition**

Classify each incoming reply into one of: `{accept, decline, ask_for_time, question, off_topic, hostile, auto_reply}`.

On `accept` (any variant of "yes", "let's do it", "go ahead", "chalte hain", "theek hai kar do"):
- **Immediately route to action** — generate the action-mode response, not another qualifying question.
- Pattern D in the brief is the explicit anti-example: merchant said "I want to join" and Vera re-asked a qualifying question. This costs points and is tested directly in Phase 4.

You can use your fast/cheap classification model here. A simple prompt: "Classify this merchant reply as one of: accept / decline / ask_for_time / question / off_topic / hostile / auto_reply. Reply with one word."

**3. Graceful exit (turn budget)**

Track `nudge_count` per `conversation_id` — the number of proactive sends you've made without a substantive merchant reply. After 3 unanswered nudges or an explicit `decline`/`hostile`, return `{"action": "end"}` with a polite closing rationale. Don't keep pinging.

**4. Hostile / off-topic handling** (Phase 4 scenario 3)

If classified `hostile` (abuse, strong refusal): acknowledge briefly and gracefully end. Don't mirror hostility.

If classified `off_topic` (e.g., "can you help with GST filing?"):
- Acknowledge the question in one sentence.
- Redirect to mission without pretending you can help with the unrelated request.
- Keep responding — this is not a conversation-ending event unless followed by hostility.

---

### Step 4 — Adaptive context handling (targets Phase 3)

Phase 3 injects new/updated contexts mid-test that the bot didn't see during development. Bots that pick up these updates in subsequent compositions score the Phase 3 adaptation bonus.

**The key rule: always re-read current context state at compose time.** If you stored a pre-processed version of the merchant context at `/v1/context` time (e.g., a rendered string, an embedding, a summarized version), throw it away and re-derive from the raw stored payload on every `/v1/tick` and `/v1/reply`. This is the single most common Phase 3 failure mode.

**Explicitly test this before submission:**
1. Run `/v1/tick` for a merchant → note the numbers used in the output.
2. Push a `v2` `performance` context for the same merchant (with different `views`, `ctr`, etc.).
3. Run `/v1/tick` again → confirm the next send reflects the new numbers, not the warmup numbers.

**Phase 3 also injects 5 new customer contexts** paired with `recall_due` triggers 2 minutes later. Your `customer is not None` branch in the composer must be exercised — test it explicitly before submission.

**Guard against hallucination on sparse context:** if no digest item is relevant to the current trigger yet, or if the merchant has no signals worth acting on, return `{"actions": []}`. An empty tick response never loses points; a hallucinated fabrication caps every dimension at 5/10. Restraint beats fabrication.

---

### Step 5 — Multi-turn cadence + customer-facing sends (secondary, tiebreaker)

`conversation_handlers.py` is a tiebreaker per §7.4 of the brief, not required. Build it only after Steps 1–4 are solid.

**Customer-facing sends** (`send_as: "merchant_on_behalf"`):
- Stricter voice rules apply — no medical claims, no guarantees for regulated categories (dentists, pharmacies). Reuse the same validator with a `customer_facing=True` flag that adds those extra taboo checks.
- Language preference must match `customer.identity.language_pref`, not `merchant.identity.languages`.
- 5 of the 30 test pairs get a customer context injected mid-test (Phase 3). Make sure your composer branches correctly on `customer is not None` — test this path explicitly.

**24-hour session window rule** (§5 of the challenge brief):
- The *first* outbound to a new `conversation_id` must use a template structure with `{{1}}/{{2}}/…` parameters.
- Subsequent messages within 24 hours of a merchant reply can be free-form.
- Track `first_send_at` per `conversation_id`; switch to free-form mode after the first merchant reply within that window.

---

### Step 6 — README + submission.jsonl (last, reflects final deployed state)

**`submission.jsonl`** — 30 lines, one per test pair from `dataset/expanded/test_pairs.json`:

```json
{"test_id": "T01", "body": "...", "cta": "open_ended", "send_as": "vera", "suppression_key": "...", "rationale": "..."}
```

Generate these by calling your *actual deployed composer* against each of the 30 test pairs — not a separate one-off script. What you submit must match what the judge independently re-derives by hitting your URL. Any discrepancy between `submission.jsonl` and a live re-run is a red flag the judge will investigate.

```python
# generate_submission.py  (quick helper — adapt to your stack)
import json
from pathlib import Path

test_pairs = json.loads(Path("dataset/expanded/test_pairs.json").read_text())["pairs"]
results = []
for pair in test_pairs:
    # load contexts from dataset/expanded/
    # call your compose() function exactly as the bot would
    # append result to results
    ...
Path("submission.jsonl").write_text("\n".join(json.dumps(r) for r in results))
```

**`README.md`** (1 page max): approach, tradeoffs, what additional context would have helped. Be honest — the judge reads the README alongside the rationale fields and a clear-eyed tradeoffs section reads better than false confidence.

---

## 4. Testing plan

1. **Local unit tests** for the validator (CTA shape, language match, repetition, fabrication heuristics) — fast, no LLM calls, run on every change.
2. **`judge_simulator.py` runs** after every meaningful change to Steps 1–4. Set `LLM_PROVIDER`, `LLM_API_KEY`, `BOT_URL` in it per the instructions at the top of the file; run with `TEST_SCENARIO = "all"` to hit warmup, auto-reply, intent transition, and hostile in one go.
3. **Manual adversarial tests** before submission:
   - Send the same canned auto-reply 4× — confirm `end` after ≤2 attempts.
   - Send "yes let's do it" mid-qualification — confirm immediate action routing, no re-qualifying question.
   - Send abuse then an off-topic question — confirm polite redirect; bot does not engage with either.
   - Push a `v1` then `v2` `performance` context — confirm next `/v1/tick` reflects v2 numbers.
   - Push a stale (`v1` again after `v2` exists) context — confirm `409` response, state unchanged.
   - Hit `/v1/tick` with an empty `available_triggers` list — confirm fast `{"actions": []}`, not a timeout.
   - Call `POST /v1/teardown` — confirm state is wiped and subsequent `/v1/healthz` shows `contexts_loaded: 0`.
4. **Load / latency check:** simulate 10 req/sec against your deployed URL for 5 minutes. Confirm no request exceeds 30s and `/v1/healthz` never dips under load. The health route must never depend on LLM calls or block on the context store lock.
5. **Full 60-minute dry run** using the simulator if it supports a full-window mode. Catches state leaks (memory growth, stale caching) that only show up over time.

---

## 5. Deployment checklist

- [ ] Ran `generate_dataset.py` and confirmed `dataset/expanded/` has 50 merchants / 200 customers / 100 triggers / `test_pairs.json`
- [ ] Public HTTPS URL reachable from outside your network (test from a phone on mobile data, not just localhost)
- [ ] All 5 endpoints match exact request/response schemas in `challenge-testing-brief.md` §2–3
- [ ] `POST /v1/teardown` implemented (wipes all state; returns `{"wiped": true}`)
- [ ] `/v1/context` idempotent + version-ordered; 500 KB payload cap returns `400`, not a crash
- [ ] `/v1/tick` and `/v1/reply` both return within 30s worst-case — even under LLM provider latency spikes; fail soft to `{"actions": []}` / `{"action": "end"}` rather than timing out
- [ ] `GET /v1/healthz` is never blocked by LLM calls or context store writes; reflects live `contexts_loaded` counts
- [ ] Bot persists context in-memory across calls; **no restarts during the test window** — check your hosting platform's idle-restart / cold-start policy (common silent failure on serverless free tiers)
- [ ] `temperature=0` confirmed set in your LLM client wrapper (not just "usually" set)
- [ ] Phase 3 adaptive test passed: push a `v2` context mid-run → next composition reflects new data
- [ ] `customer is not None` branch tested with at least one customer-facing composition
- [ ] `judge_simulator.py` passes locally with non-zero, non-degrading scores across all scenarios
- [ ] LLM API quota/budget sized for a 60-min run at up to 10 req/sec bursts
- [ ] Anti-repetition check confirmed: same body never sent twice in same `conversation_id`
- [ ] No calls to non-LLM external APIs with merchant/customer payload data (privacy rule per testing-brief §11)
- [ ] `submission.jsonl` generated from the actual deployed composer, not a standalone script
- [ ] `README.md` written (1 page: approach, tradeoffs, what additional context would have helped)
- [ ] Submitted URL + `README.md` + `submission.jsonl` (+ optional `conversation_handlers.py`) via the portal

---

## 6. Suggested time allocation

Adjust to your actual deadline. These are rough effort weights, not clock hours.

| Phase | Focus | Rough weight | Why this order |
|---|---|---:|---|
| 0 | Run `generate_dataset.py`, inspect output, verify `test_pairs.json` | 1× | Everything else depends on the expanded dataset existing |
| 1 | All 5 endpoints + `/v1/teardown` + context store + healthz/metadata + stub composer | 2× | Get zero-penalty harness score first; operational penalties are easy points to lose for silly reasons |
| 2 | Core composer + trigger-variant routing + validator + few-shot anchoring | 5× | Most of the 1,500 base points live here |
| 3 | Auto-reply detection + intent transition + graceful exit + hostile handling | 2× | Directly targeted by Phase 4 (top-10 replay), cheap relative to payoff, explicit "beat production Vera" ask |
| 4 | Adaptive context handling + customer-facing branch + teardown test | 1× | Targets Phase 3 bonus and the 5 customer-context test pairs |
| 5 | Adversarial testing pass + deployment hardening + load check | 2× | Where operational penalties get eliminated |
| 6 | README + `submission.jsonl` + final `judge_simulator.py` pass | 1× | Last — must reflect final deployed behavior |

---

## 7. Common failure modes to actively avoid

- **Generic offers** ("Flat 30% off") instead of service+price ("Haircut @ ₹99") — bake the `offer_catalog` format into the prompt; don't let the LLM improvise discount language.
- **Multiple CTAs** in one message — validator must hard-reject this.
- **Buried CTA** — prompt must explicitly say "the ask lands in the last sentence."
- **Promotional tone for clinical categories** (dentists, pharmacies) — `category.voice.taboos` must be enforced, not just suggested. Inject them into the system prompt as a hard-deny list.
- **Hallucinated citations or competitors** — validator must check that any cited number or source string appears in the actual context JSON provided to the prompt. If the number can't be traced, re-prompt.
- **Long preambles or re-introductions** — track `turn_count` per `conversation_id`. After turn 1, suppress intro language ("Hi, I'm Vera…") explicitly in the prompt.
- **Ignoring language preference** — hard rule: if `identity.languages` includes `hi`, compose in Hindi-English code-mix. Enforce in validator.
- **Verbatim repeats** — tracked and penalized explicitly (−2 each); trivial de-dup check to implement.
- **Stale context after a Phase 3 push** — never pre-process or cache derived versions of context payloads; always re-read raw payload at compose time.
- **`temperature` not set to 0** — required for determinism. Easy to forget under time pressure; put it in your client wrapper, not per-call.
- **Cold-start restarts on serverless** — in-memory state is wiped on restart; use a platform that keeps the process alive or implement a persistence layer (SQLite is fine).

---

## Source files referenced (already in your workspace)

`challenge-brief.md`, `challenge-testing-brief.md`, `examples/api-call-examples.md`, `examples/case-studies.md`, `dataset/generate_dataset.py`, `dataset/*_seed.json`, `dataset/categories/*.json`, `judge_simulator.py`, `engagement-design.md`, `engagement-research.md`.

To load this plan alongside the challenge brief in a fresh coding session:
```
load: challenge-brief.md, challenge-testing-brief.md, docs/vera-challenge-execution-plan.md
```
