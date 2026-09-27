# Vera AI Challenge — Submission README

## Approach

The bot is a stateful FastAPI server that exposes the five required endpoints plus `/v1/teardown`. The core is a four-layer stack:

**1. Composer** (`composer.py`)
LLM-powered proactive message generator. Each tick reads the raw context payloads directly from the in-memory store (never pre-processed, so Phase 3 v2 updates are picked up automatically) and routes the trigger kind to one of five prompt variant families: `research`, `event`, `performance`, `relationship`, and `customer`. Six gold-standard case studies from the challenge brief are embedded as few-shot examples in the system prompt to anchor tone, specificity level, and CTA placement.

A post-LLM validator runs after every compose call and rejects: empty bodies, URLs, invalid CTA values, and numbers not traceable to the provided context (fabrication heuristic). One retry with explicit feedback is attempted before falling back to a structural stub. For customer-facing triggers, the validator additionally enforces `send_as=merchant_on_behalf` and checks category voice taboos.

**2. Conversation policy** (`conversation_policy.py`)
Two-stage intent classifier: a fast regex heuristic for unambiguous cases (hostile, accept, decline, auto-reply patterns), followed by an LLM call with the 8B model for ambiguous messages. Auto-reply detection uses token-level Jaccard similarity (threshold 0.85, last 3 merchant turns) to catch near-duplicate canned replies even when phrasing varies slightly. Three-step auto-reply escalation: polite follow-up on count 1, 24-hour wait on count 2, hard exit on count 3. Turn budget: 3 unanswered proactive nudges without a substantive reply triggers a graceful exit.

**3. Reply composer** (`reply_composer.py`)
Handles conversation continuations. Terminal intents (hostile, decline, repeated auto-reply) are resolved deterministically — no LLM token cost, no timeout risk. Non-terminal intents (accept, question, off-topic, ask-for-time, neutral) go to the LLM with the full conversation history and merchant context block. On `accept`, the system prompt explicitly instructs "move to action mode — never re-ask a qualifying question after a commitment" (Pattern D anti-example from the brief).

**4. State management** (`bot.py`)
In-memory context store keyed by `(scope, context_id)`. Version ordering enforced at write time: same or lower version returns 409 with `current_version`. Healthz count maintained as an O(1) counter so the liveness probe never blocks under load. All LLM calls run in a thread pool via `asyncio.run_in_executor` so the async event loop (and healthz) are never blocked.

Model: `qwen/qwen3.8-27b` via Groq at `temperature=0`.

---

## Tradeoffs

**Single model for both composer and classifier.** Ideally the classifier uses a fast sub-1B model and the composer uses a 70B model running in parallel. On this Groq account only `qwen/qwen3.8-27b` was available for text generation; using it for both serialises the calls and increases tick latency. On an account with `llama-3.1-8b-instant` + `llama-3.3-70b-versatile` the classifier would run in ~200ms while composition runs concurrently, fitting comfortably inside the 30s tick budget even with 5-10 triggers per tick.

**Fabrication heuristic is imprecise.** The validator checks that numbers > 10 in the output exist in the raw context string. This catches the most common hallucination patterns but has two known failure modes: (a) numbers that legitimately derive from arithmetic on context values (e.g., `views / calls = implied conversion rate`) are flagged even when correct; (b) numbers present in the context as decimals (`0.45`) but rendered as percentages (`45%`) in the output also trigger false flags. Both cases currently cause a retry and sometimes keep the imprecise output if the retry also fails. A tighter approach would be a dedicated fact-checking pass after composition.

**Stub fallbacks for sparse context.** Research triggers with no digest items return `{"actions": []}` rather than composing a generic message. This is the correct behaviour per the brief ("restraint beats fabrication") but means lower tick throughput when digest content hasn't been pushed yet. A future improvement would be routing these to the `relationship` variant (a curious-ask instead of a research anchor) rather than skipping entirely.

**In-memory state only.** The server stores all context and conversation state in Python dicts. A cold restart wipes everything. For the 60-minute judge test window this is fine — but any deployment platform that restarts the process on idle (Render free tier, Replit, etc.) would break Phase 2. Deployment requires a persistent-process platform (Render Starter, Fly, Railway).

---

## What additional context would have helped most

**1. Merchant conversation history with engagement tags.** The dataset includes `conversation_history` but it's empty for most expanded merchants. Knowing which trigger kinds a merchant has already engaged with, and what phrasing drove the highest reply rate for them historically, would allow personalising the compulsion lever selection per merchant rather than relying on category-level defaults.

**2. Real-time slot availability for booking flows.** Customer-facing recall and appointment triggers compose messages offering specific slots, but the dataset has no actual availability data. The bot has to reference slots generically ("let me know a time that works") rather than offering concrete options like "Wed 6pm or Thu 5pm" — which is the gold-standard pattern shown in Case Study 2 (49/50). A lightweight slots API or a pre-populated slot list per merchant would directly improve customer-facing scores.

**3. Locality-scoped peer benchmarks.** The peer stats in the category context are scoped at the city or segment level (`delhi_solo_practices`). Composing "your CTR is below the peer median" is less compelling than "3 other clinics in Lajpat Nagar have a CTR of 3.2% vs your 2.1%." Locality-level benchmarks would make the social proof lever significantly sharper and more verifiable.
