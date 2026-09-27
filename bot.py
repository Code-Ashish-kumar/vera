#!/usr/bin/env python3
"""
magicpin Vera AI Challenge — Candidate Bot
==========================================
Step 4: Adaptive context handling + customer-facing branch hardened.
- composer.compose() returns None on sparse context (restraint > fabrication)
- Customer-facing validator: enforces send_as, checks category taboos
- send_as forcibly set to merchant_on_behalf (LLM cannot override)
- Phase 3 no-digest guard: research triggers skip if no digest items yet
- bot.py tick skips action when compose() returns None

Run with Python 3.13:
    py -3.13 -m uvicorn bot:app --host 0.0.0.0 --port 8080 --reload

Configuration (set in .env or environment):
    GEMINI_API_KEY          — required for LLM composition + intent classification
    GEMINI_MODEL            — optional, defaults to gemini-2.0-flash
"""

import asyncio
import logging
import os
import time
import uuid
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

# Load .env before importing composer modules so GEMINI_API_KEY is set
try:
    from dotenv import load_dotenv
    _env_path = Path(__file__).parent / ".env"
    if _env_path.exists():
        load_dotenv(_env_path)
        logging.info("Loaded .env from %s", _env_path)
except ImportError:
    pass  # python-dotenv not installed; rely on shell environment

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Import LLM composers and conversation policy (after env is loaded)
from composer import composer as llm_composer
from reply_composer import reply_composer as llm_reply_composer
from conversation_policy import (
    classify_intent,
    is_near_duplicate,
    should_exit_on_budget,
    auto_reply_action,
    MAX_NUDGES_BEFORE_EXIT,
)

# =============================================================================
# APP + STARTUP
# =============================================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Vera AI Bot", version="4.0.0")
START_TIME = time.time()

# =============================================================================
# IN-MEMORY STATE
# =============================================================================
# All state is protected by a single asyncio.Lock so concurrent requests from
# the judge (up to 10 req/sec) never produce a torn read or double-write.
#
# context_store: (scope, context_id) → {"version": int, "payload": dict}
# conv_store:    conversation_id → ConversationState
# context_counts: scope → int  (kept in sync for O(1) healthz)

_lock = asyncio.Lock()

context_store: dict[tuple[str, str], dict] = {}
context_counts: dict[str, int] = {
    "category": 0,
    "merchant": 0,
    "customer": 0,
    "trigger": 0,
}

# =============================================================================
# CONVERSATION STATE
# =============================================================================

class ConversationState:
    """Tracks everything that matters about one merchant/customer conversation."""

    def __init__(self, conversation_id: str, merchant_id: str,
                 customer_id: Optional[str], trigger_id: str):
        self.conversation_id = conversation_id
        self.merchant_id = merchant_id
        self.customer_id = customer_id
        self.trigger_id = trigger_id
        self.turns: list[dict] = []          # {"from": "vera"|"merchant", "body": str}
        self.sent_bodies: list[str] = []     # all bodies ever sent — anti-repeat
        self.nudge_count: int = 0            # proactive sends without a substantive reply
        self.auto_reply_count: int = 0       # consecutive auto-replies detected (never resets)
        self.phase: str = "active"           # "active" | "waiting" | "ended"
        self.first_send_at: Optional[float] = None  # epoch — for 24h session window
        self.last_merchant_reply: Optional[str] = None

    def record_send(self, body: str):
        self.turns.append({"from": "vera", "body": body})
        self.sent_bodies.append(body)
        self.nudge_count += 1
        if self.first_send_at is None:
            self.first_send_at = time.time()

    def record_merchant_reply(self, body: str):
        self.turns.append({"from": "merchant", "body": body})
        self.last_merchant_reply = body
        self.nudge_count = 0  # reset nudge counter on any substantive reply

    def is_repeat(self, body: str) -> bool:
        """Exact match check — anti-repetition guard (−2 penalty per repeat)."""
        return body.strip() in [b.strip() for b in self.sent_bodies]

    def turn_count(self) -> int:
        return len([t for t in self.turns if t["from"] == "vera"])


conv_store: dict[str, ConversationState] = {}

# Tracks which (merchant_id, trigger_id) pairs have already started a
# conversation — prevents duplicate proactive sends per trigger.
active_conversations: dict[tuple[str, str], str] = {}  # (mid, tid) → conv_id

# Ended conversations — suppress future ticks for these triggers.
suppressed_keys: set[str] = set()

# =============================================================================
# PYDANTIC REQUEST MODELS
# =============================================================================

class ContextBody(BaseModel):
    scope: str
    context_id: str
    version: int
    payload: dict[str, Any]
    delivered_at: str = ""

class TickBody(BaseModel):
    now: str
    available_triggers: list[str] = []

class ReplyBody(BaseModel):
    conversation_id: str
    merchant_id: Optional[str] = None
    customer_id: Optional[str] = None
    from_role: str = "merchant"
    message: str
    received_at: str = ""
    turn_number: int = 1

# =============================================================================
# HELPER UTILITIES
# =============================================================================

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

def new_ack_id(context_id: str, version: int) -> str:
    return f"ack_{context_id}_v{version}"

def new_conv_id(merchant_id: str, trigger_id: str) -> str:
    """Generate a meaningful, decodable conversation ID."""
    short_mid = merchant_id.split("_")[1] if "_" in merchant_id else merchant_id[:8]
    short_tid = trigger_id.split("_")[2] if trigger_id.count("_") >= 2 else trigger_id[:8]
    return f"conv_{short_mid}_{short_tid}_{uuid.uuid4().hex[:6]}"

def get_context(scope: str, context_id: str) -> Optional[dict]:
    """Retrieve the payload for a stored context, or None."""
    entry = context_store.get((scope, context_id))
    return entry["payload"] if entry else None

def _classify_reply_intent(message: str, conv: "ConversationState") -> str:
    """
    Two-stage intent classifier (Step 3).
    Stage 1: Jaccard near-duplicate + fast heuristic patterns.
    Stage 2: LLM classification for ambiguous messages.
    Delegates entirely to conversation_policy.classify_intent().
    """
    prior_turns = [t["body"] for t in conv.turns if t["from"] == "merchant"]
    # Build a brief context summary for the LLM classifier
    context_summary = ""
    if conv.trigger_id != "unknown":
        trigger = get_context("trigger", conv.trigger_id)
        if trigger:
            context_summary = f"trigger_kind={trigger.get('kind', '')}"
    return classify_intent(message, prior_turns, context_summary)


def _should_force_exit(conv: "ConversationState") -> bool:
    """
    Returns True when turn budget is exhausted — bot must end the conversation
    regardless of what the next merchant message says.
    """
    return should_exit_on_budget(conv.nudge_count, "neutral")

# =============================================================================
# STUB COMPOSERS — REMOVED IN STEP 2
# =============================================================================
# stub_compose() and stub_reply_compose() have been replaced by:
#   - composer.py  → LLMComposer (proactive messages from /v1/tick)
#   - reply_composer.py → LLMReplyComposer (conversation replies from /v1/reply)
# Both fall back to stub-shaped output if GEMINI_API_KEY is not set.

# =============================================================================
# ENDPOINT: GET /v1/healthz
# =============================================================================

@app.get("/v1/healthz")
async def healthz():
    """
    Liveness probe. Must:
    - Respond in < 2s at all times
    - Never depend on LLM or blocking I/O
    - Reflect accurate context_counts (updated atomically on each /v1/context accept)
    """
    async with _lock:
        counts = dict(context_counts)

    return {
        "status": "ok",
        "uptime_seconds": int(time.time() - START_TIME),
        "contexts_loaded": counts,
    }

# =============================================================================
# ENDPOINT: GET /v1/metadata
# =============================================================================

@app.get("/v1/metadata")
async def metadata():
    """Static bot identity. Update team_name / model before submission."""
    model = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
    raw_key = os.environ.get("GEMINI_API_KEY", "")
    key_set = bool(raw_key) and raw_key != "your_gemini_api_key_here"
    return {
        "team_name": "Vera Challenger",
        "team_members": ["Candidate"],
        "model": model,
        "approach": (
            "4-layer stack: LLM composer (gemini-2.0-flash, 5 trigger-variant families, "
            "post-LLM validator, few-shot anchors from case studies), "
            "2-stage intent classifier (Jaccard + Gemini), "
            "3-step auto-reply escalation, turn-budget enforcer (3 nudges), "
            "adaptive raw-payload context store. temperature=0."
        ),
        "contact_email": "candidate@example.com",
        "version": "4.0.0",
        "submitted_at": "2026-09-27T00:00:00Z",
    }

# =============================================================================
# ENDPOINT: POST /v1/context
# =============================================================================

VALID_SCOPES = {"category", "merchant", "customer", "trigger"}
MAX_PAYLOAD_BYTES = 500 * 1024  # 500 KB

@app.post("/v1/context")
async def push_context(body: ContextBody, request: Request):
    """
    Receive a context push from the judge.

    Behaviour:
    - Idempotent by (context_id, version): same version is a 409.
    - Higher version atomically replaces the stored payload.
    - Payload size capped at 500 KB — returns 400 on violation.
    - Never pre-processes or renders the payload; stores raw dict only.
      This ensures Phase 3 adaptive context updates are always picked up
      at compose time, not stale cached derivatives.
    """
    # Scope validation
    if body.scope not in VALID_SCOPES:
        return JSONResponse(status_code=400, content={
            "accepted": False,
            "reason": "invalid_scope",
            "details": f"scope must be one of {sorted(VALID_SCOPES)}",
        })

    # Payload size guard
    raw_body = await request.body()
    if len(raw_body) > MAX_PAYLOAD_BYTES:
        return JSONResponse(status_code=400, content={
            "accepted": False,
            "reason": "payload_too_large",
            "details": f"Payload exceeds 500 KB limit ({len(raw_body)} bytes received)",
        })

    key = (body.scope, body.context_id)

    async with _lock:
        existing = context_store.get(key)

        # Version conflict: we already have this version or newer
        if existing and existing["version"] >= body.version:
            return JSONResponse(status_code=409, content={
                "accepted": False,
                "reason": "stale_version",
                "current_version": existing["version"],
            })

        # Accept: store raw payload only (never pre-process)
        is_new = existing is None
        context_store[key] = {
            "version": body.version,
            "payload": body.payload,
            "stored_at": utc_now_iso(),
        }

        # Keep counts in sync for O(1) healthz
        if is_new:
            context_counts[body.scope] = context_counts.get(body.scope, 0) + 1

    ack_id = new_ack_id(body.context_id, body.version)
    stored_at = utc_now_iso()

    return {
        "accepted": True,
        "ack_id": ack_id,
        "stored_at": stored_at,
    }

# =============================================================================
# ENDPOINT: POST /v1/tick
# =============================================================================

TICK_TIMEOUT_SECONDS = 25  # hard ceiling; return empty actions before judge's 30s

@app.post("/v1/tick")
async def tick(body: TickBody):
    """
    Periodic wake-up. Judge calls this every ~5 simulated minutes.

    For each available trigger:
    1. Look up the trigger payload from context_store.
    2. Look up the merchant (and optionally customer) from context_store.
    3. Skip if already have an active conversation for this (merchant, trigger) pair.
    4. Skip if trigger's suppression_key is in suppressed_keys.
    5. Compose a stub message and record a new conversation.

    Must return within 30s (we cut off at 25s to be safe).
    Returns {"actions": []} if nothing is worth sending or time runs short.
    """
    deadline = time.time() + TICK_TIMEOUT_SECONDS
    actions = []

    async with _lock:
        available_triggers = list(body.available_triggers)
        # Snapshot store references under lock, process outside
        trigger_snapshots = {
            tid: context_store.get(("trigger", tid), {}).get("payload")
            for tid in available_triggers
        }
        merchant_snapshot = dict(context_store)  # shallow copy of keys

    for trigger_id in available_triggers:
        # Safety: bail out if we're approaching the deadline
        if time.time() > deadline:
            break

        trigger = trigger_snapshots.get(trigger_id)
        if not trigger:
            continue  # trigger context not yet pushed — skip silently

        merchant_id = trigger.get("merchant_id")
        customer_id = trigger.get("customer_id")
        suppression_key = trigger.get("suppression_key", "")

        if not merchant_id:
            continue

        # Skip if suppressed (merchant explicitly opted out)
        if suppression_key and suppression_key in suppressed_keys:
            continue

        # Skip if we already have an active conversation for this pair
        pair_key = (merchant_id, trigger_id)
        if pair_key in active_conversations:
            existing_conv_id = active_conversations[pair_key]
            existing_conv = conv_store.get(existing_conv_id)
            if existing_conv and existing_conv.phase != "ended":
                continue  # ongoing conversation — don't start a second one

        # Look up merchant context
        merchant_entry = merchant_snapshot.get(("merchant", merchant_id))
        if not merchant_entry:
            continue  # merchant context not pushed yet

        # Look up customer context if this is a customer-scoped trigger
        if customer_id:
            customer_entry = merchant_snapshot.get(("customer", customer_id))
            if not customer_entry:
                continue  # customer context not pushed yet

        # Create a new conversation state
        conv_id = new_conv_id(merchant_id, trigger_id)
        conv = ConversationState(conv_id, merchant_id, customer_id, trigger_id)

        # Compose via LLM (falls back to stub if LLM unavailable)
        async with _lock:
            store_snapshot = dict(context_store)  # fresh snapshot for composer

        action = await asyncio.get_event_loop().run_in_executor(
            None,
            llm_composer.compose,
            merchant_id, trigger_id, conv_id, customer_id, store_snapshot,
        )

        # compose() returns None when it deliberately chooses not to send
        # (e.g., no digest items — restraint beats fabrication)
        if action is None:
            logger.info("Composer chose not to send for trigger %s — skipping", trigger_id)
            continue

        # Record the send in conversation state
        conv.record_send(action["body"])

        # Persist conversation
        async with _lock:
            conv_store[conv_id] = conv
            active_conversations[pair_key] = conv_id

        actions.append(action)

    return {"actions": actions}

# =============================================================================
# ENDPOINT: POST /v1/reply
# =============================================================================

@app.post("/v1/reply")
async def reply(body: ReplyBody):
    """
    Receive a reply from the simulated merchant or customer.

    Returns one of three action shapes:
    - {"action": "send", "body": ..., "cta": ..., "rationale": ...}
    - {"action": "wait", "wait_seconds": ..., "rationale": ...}
    - {"action": "end", "rationale": ...}

    Must return within 30s. Never raises an unhandled exception.
    """
    conv_id = body.conversation_id
    merchant_id = body.merchant_id
    message = body.message.strip()

    async with _lock:
        conv = conv_store.get(conv_id)

    # Unknown conversation — create a minimal one to handle gracefully
    if not conv:
        if not merchant_id:
            return {"action": "end", "rationale": "Unknown conversation and no merchant_id provided."}
        conv = ConversationState(conv_id, merchant_id, body.customer_id, "unknown")
        async with _lock:
            conv_store[conv_id] = conv

    # Already ended — don't respond
    if conv.phase == "ended":
        return {
            "action": "end",
            "rationale": "Conversation already ended; no further engagement.",
        }

    # ── Turn-budget check ────────────────────────────────────────────────────
    # If the bot has already sent MAX_NUDGES_BEFORE_EXIT proactive messages
    # without a substantive merchant reply, close gracefully.
    if _should_force_exit(conv):
        conv.phase = "ended"
        return {
            "action": "end",
            "rationale": (
                f"Turn budget exhausted ({conv.nudge_count} unanswered nudges). "
                "Closing conversation gracefully to avoid spamming the merchant."
            ),
        }

    # ── Intent classification (Stage 1 heuristic + Stage 2 LLM) ────────────
    # classify_intent() checks Jaccard near-duplicate first, then heuristics,
    # then LLM for ambiguous messages. Runs synchronously in the thread pool
    # (via run_in_executor in the reply handler — see below).
    intent = _classify_reply_intent(message, conv)

    # ── Auto-reply counter ───────────────────────────────────────────────────
    # Increment BEFORE recording so the escalation policy sees the updated count
    if intent == "auto_reply":
        conv.auto_reply_count += 1

    # Record the incoming reply
    conv.record_merchant_reply(message)

    # Compose reply via LLM (falls back to stub if LLM unavailable)
    async with _lock:
        store_snapshot = dict(context_store)

    result = await asyncio.get_event_loop().run_in_executor(
        None,
        llm_reply_composer.compose_reply,
        conv.turns,
        intent,
        message,
        conv.merchant_id,
        conv.trigger_id,
        conv.auto_reply_count,
        store_snapshot,
    )
    # Post-compose state updates
    action = result.get("action", "send")

    if action == "send":
        reply_body = result.get("body", "")
        conv.record_send(reply_body)
    elif action in ("end", "wait"):
        conv.phase = "ended" if action == "end" else "waiting"
        # Suppress further ticks for hostile/declined conversations
        if intent in ("hostile", "decline"):
            trigger = get_context("trigger", conv.trigger_id)
            if trigger:
                sk = trigger.get("suppression_key", "")
                if sk:
                    async with _lock:
                        suppressed_keys.add(sk)

    return result

# =============================================================================
# ENDPOINT: POST /v1/teardown
# =============================================================================

@app.post("/v1/teardown")
async def teardown():
    """
    Wipe all in-memory state. Called by the judge at the end of the test window
    (testing-brief §11). Also useful between local test runs to start clean.
    """
    async with _lock:
        context_store.clear()
        conv_store.clear()
        active_conversations.clear()
        suppressed_keys.clear()
        context_counts.update({
            "category": 0,
            "merchant": 0,
            "customer": 0,
            "trigger": 0,
        })

    return {"wiped": True, "wiped_at": utc_now_iso()}

# =============================================================================
# GLOBAL EXCEPTION HANDLER
# =============================================================================
# Ensures the bot never returns a 500 with an unexpected traceback —
# which the judge would treat as a malformed response (−2 penalty).

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    path = request.url.path
    # healthz must never fail
    if path == "/v1/healthz":
        return JSONResponse(status_code=200, content={
            "status": "degraded",
            "uptime_seconds": int(time.time() - START_TIME),
            "contexts_loaded": dict(context_counts),
            "error": str(exc)[:200],
        })
    # tick and reply: return safe fallback shapes
    if path == "/v1/tick":
        return JSONResponse(status_code=200, content={"actions": []})
    if path == "/v1/reply":
        return JSONResponse(status_code=200, content={
            "action": "end",
            "rationale": f"Internal error; ending conversation safely. ({type(exc).__name__})",
        })
    # All others: return 500 with structured error (not a raw traceback)
    return JSONResponse(status_code=500, content={
        "error": type(exc).__name__,
        "detail": str(exc)[:500],
    })

# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("bot:app", host="0.0.0.0", port=8080, reload=False)
