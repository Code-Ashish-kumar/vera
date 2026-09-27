"""
reply_composer.py — LLM-based conversation reply composer (Step 3)
===================================================================

Step 3 additions vs Step 2:
  - off_topic intent branch: acknowledge + redirect (not just "neutral")
  - ask_for_time intent branch: schedule a follow-up, don't push
  - Auto-reply escalation updated to 3-step:
      count=1 → send polite follow-up (LLM)
      count=2 → wait 24h (deterministic)
      count≥3 → end (deterministic)
  - _handle_auto_reply_end() added for count≥3 case
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Optional

from groq import Groq

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# RE-USE THE SAME CLIENT CONFIG AS composer.py
# ---------------------------------------------------------------------------

GROQ_MODEL   = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
LLM_TEMPERATURE = 0

def _get_groq_client() -> Groq:
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not set")
    return Groq(api_key=api_key)

# ---------------------------------------------------------------------------
# SYSTEM PROMPT FOR REPLY COMPOSER
# ---------------------------------------------------------------------------

REPLY_SYSTEM_PROMPT = """You are Vera, magicpin's merchant AI assistant, in the middle of a WhatsApp conversation.

YOUR JOB:
- Continue the conversation in a way that advances the merchant toward the intended action.
- Match the merchant's language (Hindi-English mix if they write in it).
- Be concise — max 3-4 sentences for conversational replies.
- When the merchant has committed ("yes / ok / let's do it / kar do"), immediately move to ACTION — draft the thing, send the thing, confirm the thing. NEVER ask another qualifying question after a commitment.
- When the merchant asks a question, answer it using facts from the context provided. Never fabricate.
- When the topic goes off-mission (GST, staffing, unrelated business), acknowledge briefly in ONE sentence, then redirect back to the original task.
- When the merchant asks for more time, acknowledge gracefully and offer a specific callback time.

HARD RULES:
- No URLs in body (−3 penalty).
- No fabricated numbers — only use figures present in the context.
- Single CTA, placed in the last sentence.
- No re-introducing yourself.

VALID CTA VALUES: open_ended | binary_yes_no | binary_confirm_cancel | multi_choice_slot | none

OUTPUT FORMAT — JSON only, no markdown:
{
  "action": "send",
  "body": "<reply message>",
  "cta": "<cta value>",
  "rationale": "<1 sentence: what intent was detected and what lever was used>"
}

GOLD PATTERNS:

ACCEPT — switch from qualifying to doing:
User: "Ok let's do it. What's next?"
Response: {"action":"send","body":"On it. Drafting the patient-ed WhatsApp now — I'll have a ready-to-send version in 2 minutes. Want me to also schedule a Google post for tomorrow 10am?","cta":"binary_yes_no","rationale":"Merchant committed; moving to action, delivering the promised artifact immediately."}

AUTO-REPLY (first detection):
User: "Thank you for contacting us! Our team will respond shortly."
Response: {"action":"send","body":"Looks like an auto-reply — when you're free, just reply here and I'll pick up from where we left off.","cta":"none","rationale":"Detected canned auto-reply; one polite flag for the owner."}

QUESTION — answer from context:
User: "What is my current CTR compared to others?"
Response: {"action":"send","body":"Your CTR last month was 2.1% — the peer median for solo practices in your area is 3.0%, so you're about 30% below. The fastest fix is usually adding a few more photos and updating business hours. Want me to draft those changes?","cta":"binary_yes_no","rationale":"Answered with real numbers from context; offered next action."}

OFF-TOPIC — acknowledge one sentence + redirect:
User: "Can you also help me with GST filing this month?"
Response: {"action":"send","body":"That's outside what I can help with directly — your CA would be the right person for GST. Coming back to the JIDA piece — want me to draft the patient post or send the abstract first?","cta":"open_ended","rationale":"Off-scope acknowledged in one sentence; redirected to original mission."}

ASK-FOR-TIME — respect the request, offer a callback:
User: "I'm busy right now, call me later."
Response: {"action":"send","body":"No problem — I'll check back tomorrow morning. Just reply here whenever you're ready and I'll pick up exactly where we left off.","cta":"none","rationale":"Merchant asked for time; acknowledging gracefully without pushing."}
"""

# ---------------------------------------------------------------------------
# CONVERSATION CONTEXT BUILDER
# ---------------------------------------------------------------------------

def _build_reply_context(
    conv_turns: list[dict],
    merchant_id: str,
    trigger_id: str,
    context_store: dict,
    message: str,
    intent: str,
) -> str:
    """
    Build a terse context block for the reply prompt.
    Includes: merchant snapshot, last N turns, current message, intent.
    """
    merchant_entry = context_store.get(("merchant", merchant_id), {})
    merchant = merchant_entry.get("payload", {})
    identity = merchant.get("identity", {})
    perf = merchant.get("performance", {})
    offers = [o["title"] for o in merchant.get("offers", []) if o.get("status") == "active"]
    signals = merchant.get("signals", [])

    trigger_entry = context_store.get(("trigger", trigger_id), {})
    trigger = trigger_entry.get("payload", {})
    trigger_kind = trigger.get("kind", "unknown")

    category_slug = merchant.get("category_slug", "")
    category_entry = context_store.get(("category", category_slug), {})
    category = category_entry.get("payload", {})
    voice = category.get("voice", {})

    # Last 6 turns for context window efficiency
    recent_turns = conv_turns[-6:] if len(conv_turns) > 6 else conv_turns
    turns_block = "\n".join(
        f"  {'VERA' if t['from'] == 'vera' else 'MERCHANT'}: {t['body'][:200]}"
        for t in recent_turns
    )

    return f"""=== MERCHANT ===
Name: {identity.get('name', 'unknown')} | Owner: {identity.get('owner_first_name', 'there')}
Languages: {identity.get('languages', ['en'])} | Locality: {identity.get('locality', 'N/A')}, {identity.get('city', 'N/A')}
CTR: {perf.get('ctr', 'N/A')} | Views(30d): {perf.get('views', 'N/A')} | Calls: {perf.get('calls', 'N/A')}
Active offers: {offers if offers else 'none'} | Signals: {signals}
Category: {category_slug} | Voice: {voice.get('tone', 'professional')}
Peer avg_ctr: {category.get('peer_stats', {}).get('avg_ctr', 'N/A')}

=== TRIGGER CONTEXT ===
Kind: {trigger_kind} | Scope: {trigger.get('scope', 'merchant')}
Payload: {json.dumps(trigger.get('payload', {}), ensure_ascii=False)[:300]}

=== CONVERSATION HISTORY (last {len(recent_turns)} turns) ===
{turns_block if turns_block else '  (no prior turns)'}

=== CURRENT MERCHANT MESSAGE ===
"{message}"

=== DETECTED INTENT ===
{intent}

Compose the reply now."""

# ---------------------------------------------------------------------------
# DETERMINISTIC TERMINAL HANDLERS
# ---------------------------------------------------------------------------
# These never call the LLM — no token cost, no latency, no failure mode.

def _handle_hostile() -> dict:
    return {
        "action": "end",
        "rationale": (
            "Merchant expressed frustration or explicit opt-out. "
            "Closing conversation without further engagement."
        ),
    }

def _handle_decline() -> dict:
    return {
        "action": "end",
        "rationale": "Merchant declined. Closing conversation gracefully.",
    }

def _handle_auto_reply_wait() -> dict:
    return {
        "action": "wait",
        "wait_seconds": 86400,
        "rationale": (
            "Auto-reply detected for the second time. "
            "Backing off 24h before re-engaging."
        ),
    }

def _handle_auto_reply_end() -> dict:
    """Three or more consecutive auto-replies — close the conversation."""
    return {
        "action": "end",
        "rationale": (
            "Auto-reply detected three or more times in a row. "
            "No real engagement signal present; closing conversation."
        ),
    }

# ---------------------------------------------------------------------------
# STUB FALLBACKS (used when LLM unavailable or errors)
# ---------------------------------------------------------------------------

_STUB_REPLIES: dict[str, dict] = {
    "accept": {
        "action": "send",
        "body": "On it — I'll have that ready for you shortly.",
        "cta": "none",
        "rationale": "Merchant committed; moving to action mode. (fallback)",
    },
    "auto_reply": {
        "action": "send",
        "body": (
            "Looks like an auto-reply — when you're free, just reply here "
            "and I'll pick up from where we left off."
        ),
        "cta": "none",
        "rationale": "Detected auto-reply; one polite prompt for the owner. (fallback)",
    },
    "question": {
        "action": "send",
        "body": "Good question — let me look into that and get back to you shortly.",
        "cta": "open_ended",
        "rationale": "Merchant asked a question; acknowledging. (fallback)",
    },
    "off_topic": {
        "action": "send",
        "body": (
            "That's outside what I can help with directly. "
            "Coming back to where we were — want to continue?"
        ),
        "cta": "open_ended",
        "rationale": "Off-topic acknowledged; redirected. (fallback)",
    },
    "ask_for_time": {
        "action": "send",
        "body": "No problem — I'll check back tomorrow. Just reply here whenever you're ready.",
        "cta": "none",
        "rationale": "Merchant asked for time; acknowledging gracefully. (fallback)",
    },
    "neutral": {
        "action": "send",
        "body": "Got it. Is there anything specific you'd like me to help with next?",
        "cta": "open_ended",
        "rationale": "Neutral reply; keeping conversation open. (fallback)",
    },
}

# ---------------------------------------------------------------------------
# MAIN REPLY COMPOSER CLASS
# ---------------------------------------------------------------------------

class LLMReplyComposer:
    """
    Stateless reply composer for /v1/reply.
    Handles the full intent spectrum — LLM only called for non-terminal intents.
    """

    def __init__(self):
        self._client: Optional[Groq] = None

    @property
    def client(self) -> Groq:
        if self._client is None:
            self._client = _get_groq_client()
        return self._client

    def compose_reply(
        self,
        conv_turns: list[dict],
        intent: str,
        message: str,
        merchant_id: str,
        trigger_id: str,
        auto_reply_count: int,
        context_store: dict,
    ) -> dict:
        """
        Main entry point — never raises, always returns a valid action dict.

        Intent routing (Step 3 — 3-step auto-reply escalation):
          hostile                   → deterministic end
          decline                   → deterministic end
          auto_reply, count == 1    → LLM polite follow-up
          auto_reply, count == 2    → deterministic wait 24h
          auto_reply, count >= 3    → deterministic end
          ask_for_time              → LLM graceful acknowledgment
          accept                    → LLM action mode
          question                  → LLM grounded answer
          off_topic                 → LLM acknowledge + redirect
          neutral                   → LLM continuation
        """
        # --- Deterministic terminal intents (no LLM needed) ---
        if intent == "hostile":
            return _handle_hostile()
        if intent == "decline":
            return _handle_decline()
        if intent == "auto_reply":
            if auto_reply_count == 2:
                return _handle_auto_reply_wait()
            elif auto_reply_count >= 3:
                return _handle_auto_reply_end()
            # auto_reply_count == 1 → fall through to LLM

        # --- LLM-powered intents ---
        try:
            return self._compose_llm(
                conv_turns, intent, message,
                merchant_id, trigger_id, context_store
            )
        except RuntimeError as e:
            logger.warning("Reply LLM not configured: %s", e)
            return _STUB_REPLIES.get(intent, _STUB_REPLIES["neutral"])
        except Exception as e:
            logger.error("Reply composer error: %s", e, exc_info=True)
            return _STUB_REPLIES.get(intent, _STUB_REPLIES["neutral"])

    def _compose_llm(
        self,
        conv_turns: list[dict],
        intent: str,
        message: str,
        merchant_id: str,
        trigger_id: str,
        context_store: dict,
    ) -> dict:
        context_block = _build_reply_context(
            conv_turns, merchant_id, trigger_id,
            context_store, message, intent
        )

        response = self.client.chat.completions.create(
            model=GROQ_MODEL,
            temperature=LLM_TEMPERATURE,
            max_tokens=400,
            messages=[
                {"role": "system", "content": REPLY_SYSTEM_PROMPT},
                {"role": "user",   "content": context_block},
            ],
        )
        raw = response.choices[0].message.content or ""
        result = self._parse_json(raw)

        if result is None:
            logger.warning("Reply composer: failed to parse JSON — using fallback")
            return _STUB_REPLIES.get(intent, _STUB_REPLIES["neutral"])

        # Validate and normalise
        result = self._normalise(result, intent)
        return result

    def _parse_json(self, raw: str) -> Optional[dict]:
        """Extract JSON from LLM output, stripping markdown fences if present."""
        raw = re.sub(r"```(?:json)?", "", raw).strip().strip("`").strip()
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass
        match = re.search(r'\{[\s\S]*\}', raw)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        return None

    def _normalise(self, result: dict, intent: str) -> dict:
        """
        Ensure the reply dict has the correct shape and valid values.
        Fixes common LLM mistakes without re-prompting.
        """
        action = result.get("action", "send")
        if action not in ("send", "wait", "end"):
            action = "send"
            result["action"] = action

        valid_ctas = {"open_ended", "binary_yes_no", "binary_confirm_cancel",
                      "multi_choice_slot", "none"}

        if action == "send":
            body = result.get("body", "").strip()
            # Remove any URLs the LLM may have hallucinated
            body = re.sub(r'https?://\S+', '', body).strip()
            result["body"] = body if body else _STUB_REPLIES.get(intent, _STUB_REPLIES["neutral"])["body"]

            # Always force-write cta so the field is never absent from a send response
            cta = result.get("cta", "open_ended")
            if cta not in valid_ctas:
                cta = "open_ended"
            result["cta"] = cta

        # Ensure rationale is always present
        if not result.get("rationale"):
            result["rationale"] = f"LLM reply for intent '{intent}'."

        # wait action needs wait_seconds
        if action == "wait" and "wait_seconds" not in result:
            result["wait_seconds"] = 86400

        return result


# ---------------------------------------------------------------------------
# MODULE-LEVEL SINGLETON
# ---------------------------------------------------------------------------

reply_composer = LLMReplyComposer()
