"""
conversation_policy.py — Conversation Policy Layer (Step 3)
============================================================

Replaces the regex-only classifiers in bot.py with a proper two-stage pipeline:

  Stage 1 — Fast heuristic (< 1ms, no LLM)
    Handles the unambiguous cases deterministically:
    - Known canned auto-reply phrases  → "auto_reply"
    - Jaccard similarity ≥ 0.85 with prior merchant turn → "auto_reply"
    - Explicit hostile opt-out patterns → "hostile"
    - Clear accept patterns (yes/ok/go/kar do) → "accept"
    - Clear decline patterns (no/nahi/not now) → "decline"
    - Explicit ask-for-time patterns → "ask_for_time"

  Stage 2 — LLM classifier (called only when Stage 1 returns "ambiguous")
    Single-token output: one of the 7 intent classes.
    Uses llama-3.1-8b-instant (Groq's smallest/fastest model) —
    classification is cheap and doesn't need the 70B model's quality.

  Turn-budget enforcer:
    MAX_NUDGES_BEFORE_EXIT = 3
    After 3 proactive sends without a substantive merchant reply,
    the bot returns action="end" regardless of what the merchant says next.

  Auto-reply escalation policy:
    count=1 → send one polite follow-up
    count=2 → wait 24h
    count≥3 → end the conversation

  All public functions return a string intent label:
    "accept" | "decline" | "ask_for_time" | "question" |
    "off_topic" | "hostile" | "auto_reply" | "neutral"
"""

from __future__ import annotations

import logging
import os
import re
from typing import Optional

from groq import Groq

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

# Groq's fast 8B model — enough for single-token classification, much faster
# than 70B. Falls back gracefully if not available.
CLASSIFIER_MODEL = os.environ.get("GROQ_CLASSIFIER_MODEL", "llama-3.1-8b-instant")
LLM_TEMPERATURE  = 0

# Jaccard similarity threshold for near-duplicate auto-reply detection
JACCARD_THRESHOLD = 0.85

# After this many unanswered proactive sends, close the conversation
MAX_NUDGES_BEFORE_EXIT = 3

# ---------------------------------------------------------------------------
# KNOWN CANNED AUTO-REPLY PATTERNS
# From Phase 4 test scenarios + real WA Business defaults
# ---------------------------------------------------------------------------

_CANNED_AUTO_REPLY_PATTERNS: list[str] = [
    r"thank you for contacting",
    r"our team will (respond|get back)",
    r"i (am|'m) an automated( assistant)?",
    r"this is an auto.?reply",
    r"we will get back to you",
    r"aapki (jaankari|madad) ke liye.*shukriya",
    r"bahut.{0,10}shukriya.*team",
    r"main ek automated",
    r"sorry (i|we) (missed|couldn't answer)",
    r"currently (unavailable|busy|away)",
    r"office hours are",
    r"we('re| are) currently closed",
]

# Compiled once at module load
_AUTO_REPLY_RE = re.compile(
    "|".join(_CANNED_AUTO_REPLY_PATTERNS),
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# HOSTILE / EXPLICIT OPT-OUT PATTERNS
# ---------------------------------------------------------------------------

_HOSTILE_PATTERNS: list[str] = [
    r"\bstop\b",
    r"\bquit\b",
    r"\bunsubscribe\b",
    r"\bblock\b",
    r"don'?t (message|contact|call|text|bother|ping)",
    r"(useless|spam|annoying|irritating|bothering)",
    r"\bnot interested\b",
    r"chodo\b",
    r"band karo",
    r"mat karo",
    r"mat bhejo",
    r"hatao",
    r"remove (me|mujhe)",
]
_HOSTILE_RE = re.compile("|".join(_HOSTILE_PATTERNS), re.IGNORECASE)

# ---------------------------------------------------------------------------
# ACCEPT / COMMIT PATTERNS
# ---------------------------------------------------------------------------

_ACCEPT_PATTERNS: list[str] = [
    r"^(yes|yeah|yep|yup|haan|ha)\b",
    r"^(ok|okay|sure|fine|alright)\b",
    r"^(go ahead|proceed|confirm|done)\b",
    r"\blet'?s do (it|this)\b",
    r"\bchalta hai\b",
    r"\bchalte hain\b",
    r"\bkar do\b",
    r"\btheek hai\b",
    r"\bbhej do\b",
    r"\bsend karo\b",
    r"what'?s next",
    r"please (send|do it|go ahead|proceed)",
    r"\bsend (it|them|the|karo)\b",
    r"\bi (want|would like|need) (to|this|that)\b",
]
_ACCEPT_RE = re.compile("|".join(_ACCEPT_PATTERNS), re.IGNORECASE)

# ---------------------------------------------------------------------------
# DECLINE PATTERNS
# ---------------------------------------------------------------------------

_DECLINE_PATTERNS: list[str] = [
    r"^(no|nope|nahi|nahin|nahi chahiye)\b",
    r"not (now|today|right now|at this time)",
    r"(busy|later|baad mein|abhi nahi)",
    r"don'?t (want|need) (this|it|that)",
    r"not (interested|required|needed)",
    r"mujhe nahi chahiye",
]
_DECLINE_RE = re.compile("|".join(_DECLINE_PATTERNS), re.IGNORECASE)

# ---------------------------------------------------------------------------
# ASK-FOR-TIME PATTERNS
# ---------------------------------------------------------------------------

_ASK_TIME_PATTERNS: list[str] = [
    r"(call|contact|reach|message) (me )?(later|tomorrow|evening|morning)",
    r"(will|i'?ll) get back",
    r"(give|dedo) (me )?(some )?(time|thoda)",
    r"(not now|baad mein|kal|tomorrow|tonight)",
    r"(busy|in a meeting|unavailable) (right now|at the moment)",
    r"(check|dekh) (karke|later|baad)",
]
_ASK_TIME_RE = re.compile("|".join(_ASK_TIME_PATTERNS), re.IGNORECASE)

# ---------------------------------------------------------------------------
# JACCARD SIMILARITY
# ---------------------------------------------------------------------------

def _jaccard(a: str, b: str) -> float:
    """
    Token-level Jaccard similarity between two strings.
    Tokenises by splitting on whitespace + punctuation.
    Returns 0.0 to 1.0.
    """
    def tokens(s: str) -> set[str]:
        return set(re.split(r"[\s\W]+", s.lower()))

    t_a = tokens(a)
    t_b = tokens(b)
    if not t_a or not t_b:
        return 0.0
    intersection = len(t_a & t_b)
    union = len(t_a | t_b)
    return intersection / union if union > 0 else 0.0


def is_near_duplicate(message: str, prior_merchant_turns: list[str]) -> bool:
    """
    Returns True if `message` is Jaccard-similar (≥ 0.85) to any of the
    last 3 merchant turns — catches near-duplicate auto-replies even when
    the text has minor variations (emoji, timestamp appended, etc.).
    """
    for prior in prior_merchant_turns[-3:]:
        if _jaccard(message, prior) >= JACCARD_THRESHOLD:
            return True
    return False

# ---------------------------------------------------------------------------
# FAST HEURISTIC CLASSIFIER (Stage 1)
# ---------------------------------------------------------------------------

def classify_heuristic(message: str) -> Optional[str]:
    """
    Returns an intent string if the message matches an unambiguous pattern,
    or None if the message needs LLM classification.
    """
    msg = message.strip()

    # Auto-reply (canned phrase)
    if _AUTO_REPLY_RE.search(msg):
        return "auto_reply"

    # Hostile / explicit opt-out
    if _HOSTILE_RE.search(msg):
        return "hostile"

    # Accept — check against multiple patterns
    if _ACCEPT_RE.search(msg):
        # Guard against sarcastic "ok whatever" — if message is very long,
        # let LLM decide (genuine accepts tend to be short)
        if len(msg) < 120:
            return "accept"

    # Decline
    if _DECLINE_RE.search(msg.lower()):
        return "decline"

    # Ask for time
    if _ASK_TIME_RE.search(msg):
        return "ask_for_time"

    # Question mark is weak evidence — let LLM confirm
    # (many Hindi messages end with ? for emphasis, not actually a question)

    return None  # ambiguous — needs LLM

# ---------------------------------------------------------------------------
# LLM INTENT CLASSIFIER (Stage 2)
# ---------------------------------------------------------------------------

_CLASSIFIER_SYSTEM = """You classify merchant WhatsApp replies into exactly one intent category.

Categories:
- accept      — merchant agrees, commits, says yes, wants to proceed
- decline     — merchant says no, not interested, not now
- ask_for_time — merchant asks you to follow up later / says they're busy
- question    — merchant asks a specific question about their business
- off_topic   — merchant asks about something unrelated to Vera's mission (GST, staff issues, etc.)
- hostile     — merchant expresses frustration, abuse, or strong opt-out
- auto_reply  — this looks like a WhatsApp Business canned auto-reply
- neutral     — none of the above (generic acknowledgment, conversation filler)

RULES:
- Reply with EXACTLY ONE WORD from the list above.
- No explanation, no punctuation, no other text.
- If unsure between accept and neutral, choose neutral.
- If message is a question AND an accept (e.g., "Yes, when can you send it?"), choose accept.
"""

_groq_classifier_client: Optional[Groq] = None

def _get_classifier_client() -> Groq:
    global _groq_classifier_client
    if _groq_classifier_client is None:
        api_key = os.environ.get("GROQ_API_KEY", "")
        if not api_key or api_key == "your_groq_api_key_here":
            raise RuntimeError("GROQ_API_KEY not set")
        _groq_classifier_client = Groq(api_key=api_key)
    return _groq_classifier_client


_VALID_INTENTS = frozenset(
    ["accept", "decline", "ask_for_time", "question", "off_topic",
     "hostile", "auto_reply", "neutral"]
)

def classify_llm(message: str, conversation_context: str = "") -> str:
    """
    LLM-based intent classifier using the fast 8B model.
    Returns one of the 8 valid intent strings.
    Falls back to heuristic neutral if LLM is unavailable.
    """
    try:
        client = _get_classifier_client()
        user_content = message
        if conversation_context:
            user_content = f"[Context: {conversation_context[:300]}]\n\nMessage: {message}"

        response = client.chat.completions.create(
            model=CLASSIFIER_MODEL,
            temperature=LLM_TEMPERATURE,
            max_tokens=10,   # single word only
            messages=[
                {"role": "system", "content": _CLASSIFIER_SYSTEM},
                {"role": "user",   "content": user_content},
            ],
        )
        raw = (response.choices[0].message.content or "").strip().lower()
        # Extract first word in case model adds punctuation
        word = re.split(r"\W+", raw)[0] if raw else ""
        if word in _VALID_INTENTS:
            return word
        logger.warning("LLM classifier returned unknown intent '%s'; defaulting to neutral", raw)
        return "neutral"

    except RuntimeError:
        # No API key — fall back to simple heuristic
        return _classify_heuristic_fallback(message)
    except Exception as e:
        logger.warning("LLM classifier error: %s; falling back to heuristic", e)
        return _classify_heuristic_fallback(message)


def _classify_heuristic_fallback(message: str) -> str:
    """Minimal fallback when LLM is unavailable."""
    if "?" in message:
        return "question"
    return "neutral"

# ---------------------------------------------------------------------------
# COMBINED CLASSIFIER (public API)
# ---------------------------------------------------------------------------

def classify_intent(
    message: str,
    prior_merchant_turns: list[str],
    context_summary: str = "",
) -> str:
    """
    Main entry point. Returns an intent string for use in bot.py.

    Process:
    1. Check for Jaccard near-duplicate → "auto_reply"
    2. Run fast heuristic classifier → return if unambiguous
    3. Run LLM classifier for ambiguous messages

    Always returns one of: accept | decline | ask_for_time | question |
                           off_topic | hostile | auto_reply | neutral
    """
    # Step 1: near-duplicate detection (catches auto-reply variants with emoji etc.)
    if is_near_duplicate(message, prior_merchant_turns):
        logger.debug("Near-duplicate detected (Jaccard) → auto_reply")
        return "auto_reply"

    # Step 2: fast heuristic
    heuristic_result = classify_heuristic(message)
    if heuristic_result is not None:
        logger.debug("Heuristic classified '%s...' → %s", message[:40], heuristic_result)
        return heuristic_result

    # Step 3: LLM (ambiguous message)
    logger.debug("Ambiguous message — calling LLM classifier")
    return classify_llm(message, context_summary)

# ---------------------------------------------------------------------------
# TURN BUDGET ENFORCER
# ---------------------------------------------------------------------------

def should_exit_on_budget(nudge_count: int, intent: str) -> bool:
    """
    Returns True if the conversation should be closed based on turn budget rules.

    Rules:
    - If nudge_count ≥ MAX_NUDGES_BEFORE_EXIT (3) with no substantive reply → close
    - nudge_count is reset to 0 on any non-auto-reply merchant message in ConversationState
    - This enforces "don't keep pinging an unresponsive merchant"

    Note: hostile and decline already return "end" before this is called.
    This handles the silent-ignore case.
    """
    return nudge_count >= MAX_NUDGES_BEFORE_EXIT

# ---------------------------------------------------------------------------
# AUTO-REPLY ESCALATION POLICY
# ---------------------------------------------------------------------------

def auto_reply_action(auto_reply_count: int) -> str:
    """
    Maps auto_reply_count to the appropriate action.
    
    count=1 → "send_followup"  (send one polite follow-up)
    count=2 → "wait"           (back off 24h)
    count≥3 → "end"            (close conversation)
    """
    if auto_reply_count == 1:
        return "send_followup"
    elif auto_reply_count == 2:
        return "wait"
    else:
        return "end"
