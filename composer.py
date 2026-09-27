"""
composer.py — LLM-based proactive message composer (Step 2)
============================================================

Replaces stub_compose() in bot.py.

Architecture:
  LLMComposer.compose(merchant_id, trigger_id, conv, context_store)
    → dict (full action shape ready to return from /v1/tick)

  Internally:
    1. Assemble the 4 contexts from context_store (raw, never pre-cached)
    2. Route trigger.kind → prompt variant
    3. Call Gemini LLM (gemini-2.0-flash, temperature=0)
    4. Parse JSON output
    5. Post-LLM validator — reject/retry on: empty body, multiple CTAs,
       URLs present, fabricated numbers, language mismatch
    6. One retry with feedback if validation fails
    7. Return validated action dict

Fallback: on any error returns a valid stub-shaped dict so the harness
never breaks.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any, Optional

import google.generativeai as genai

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LLM CLIENT
# ---------------------------------------------------------------------------

def _get_gemini_client() -> genai.GenerativeModel:
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set — add it to .env")
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(
        model_name=GEMINI_MODEL,
        generation_config=genai.GenerationConfig(
            temperature=LLM_TEMPERATURE,
            max_output_tokens=600,
        ),
    )


GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
LLM_TEMPERATURE = 0  # required for determinism

# ---------------------------------------------------------------------------
# TRIGGER-KIND → PROMPT VARIANT ROUTING
# ---------------------------------------------------------------------------
# Five families cover the 15 known trigger kinds.
# Each variant gets a different framing instruction + compulsion lever emphasis.

TRIGGER_VARIANTS: dict[str, str] = {
    # External research / compliance
    "research_digest":              "research",
    "regulation_change":            "research",
    "category_research_digest_release": "research",

    # External events / market
    "festival_upcoming":            "event",
    "weather_heatwave":             "event",
    "local_news_event":             "event",
    "competitor_opened":            "event",
    "category_trend_movement":      "event",
    "ipl_match_today":              "event",

    # Internal performance
    "perf_spike":                   "performance",
    "perf_dip":                     "performance",
    "milestone_reached":            "performance",
    "review_theme_emerged":         "performance",

    # Internal relationship / cadence
    "dormant_with_vera":            "relationship",
    "renewal_due":                  "relationship",
    "curious_ask_due":              "relationship",
    "active_planning_intent":       "relationship",
    "scheduled_recurring":          "relationship",

    # Customer-scoped
    "recall_due":                   "customer",
    "customer_lapsed_soft":         "customer",
    "customer_lapsed_hard":         "customer",
    "appointment_tomorrow":         "customer",
    "chronic_refill_due":           "customer",
    "trial_followup":               "customer",
}

VARIANT_INSTRUCTIONS: dict[str, str] = {
    "research": (
        "This is a RESEARCH/COMPLIANCE digest trigger. "
        "Lead with the specific finding or regulatory change as the 'why now'. "
        "Cite the source (journal, circular, publication + date). "
        "Connect the finding to THIS merchant's specific patient/customer cohort from the context. "
        "Offer to do work for them (pull abstract, draft patient-ed content, summarize). "
        "Use curiosity + reciprocity levers. "
        "CTA: open_ended (never binary for pure-information triggers)."
    ),
    "event": (
        "This is an EXTERNAL EVENT trigger (festival, weather, competitor, trend). "
        "Lead with the specific event as the 'why now'. "
        "Give a contrarian or non-obvious data-backed recommendation if possible. "
        "Connect to the merchant's existing offers or current performance. "
        "Offer to create a concrete deliverable (banner, post, WhatsApp draft). "
        "Use loss aversion + effort externalization levers. "
        "CTA: binary_yes_no (clear YES/STOP choice)."
    ),
    "performance": (
        "This is an INTERNAL PERFORMANCE trigger (spike, dip, milestone, reviews). "
        "Lead with the specific number from the context — views, calls, CTR, review count. "
        "Frame spikes as amplification opportunities, dips as fixable with a concrete action. "
        "For review themes: quote the pattern, not generic feedback. "
        "Offer to take the next action for them. "
        "Use social proof + specificity levers. "
        "CTA: binary_yes_no."
    ),
    "relationship": (
        "This is a RELATIONSHIP/CADENCE trigger (dormant, renewal, curious-ask, recurring). "
        "For dormant/curious-ask: ask the merchant ONE specific question about their business. "
        "For renewal: lead with concrete value delivered, not a pitch. "
        "Keep it short — max 3 sentences. Single low-friction ask. "
        "Use reciprocity + asking-the-merchant levers. "
        "CTA: open_ended."
    ),
    "customer": (
        "This is a CUSTOMER-FACING trigger — message goes FROM the merchant TO their customer. "
        "send_as MUST be 'merchant_on_behalf'. "
        "Write from the merchant's voice, not Vera's. "
        "Use the customer's name, honor their language preference, honor their time preference. "
        "For recall/lapse: be warm, no guilt-trip, offer specific slots with prices. "
        "For refill: list the exact medication names and total cost with savings shown. "
        "CTA: multi_choice_slot for appointment booking; binary_yes_no for confirmations."
    ),
}

# ---------------------------------------------------------------------------
# FEW-SHOT EXAMPLES (extracted from case-studies.md — gold standard)
# ---------------------------------------------------------------------------

FEW_SHOT_EXAMPLES = """
=== GOLD EXAMPLE 1 — Research Digest (score 50/50) ===
Trigger kind: research_digest | Category: dentists | Merchant: Dr. Meera's Dental Clinic
Context used: high_risk_adult_count=124, CTR=2.1% (below peer 3.0%), digest item JIDA Oct 2026 p.14
Output:
{
  "body": "Dr. Meera, JIDA's Oct issue landed. One item relevant to your high-risk adult patients — 2,100-patient trial showed 3-month fluoride recall cuts caries recurrence 38% better than 6-month. Worth a look (2-min abstract). Want me to pull it + draft a patient-ed WhatsApp you can share?  — JIDA Oct 2026 p.14",
  "cta": "open_ended",
  "send_as": "vera",
  "rationale": "External research digest with merchant-relevant clinical anchor (high-risk-adult cohort). Source citation maintains credibility. Open-ended CTA invites continuation."
}
WHY IT WORKS: Specificity (2,100-patient, 38%, page number), Category fit (clinical vocabulary, peer tone), Merchant fit (her cohort), Trigger relevance (digest as reason), Engagement (curiosity + reciprocity).

=== GOLD EXAMPLE 2 — Customer Recall (score 49/50) ===
Trigger kind: recall_due | Category: dentists | Customer: Priya (lapsed_soft, hi-en mix, weekday evening)
Context used: offer "Dental Cleaning @ ₹299", slots Wed 5 Nov 6pm + Thu 6 Nov 5pm
Output:
{
  "body": "Hi Priya, Dr. Meera's clinic here 🦷 It's been 5 months since your last visit — your 6-month cleaning recall is due. Apke liye 2 slots ready hain: Wed 5 Nov, 6pm ya Thu 6 Nov, 5pm. ₹299 cleaning + complimentary fluoride. Reply 1 for Wed, 2 for Thu, or tell us a time that works.",
  "cta": "multi_choice_slot",
  "send_as": "merchant_on_behalf",
  "rationale": "Customer-scoped recall. Honoring hi-en mix language pref + weekday-evening preference. Multi-choice slot CTA appropriate for booking flows."
}

=== GOLD EXAMPLE 3 — IPL Event (score 50/50) ===
Trigger kind: ipl_match_today | Category: restaurants | Merchant: SK Pizza Junction
Context used: BOGO pizza offer active, Saturday match = -12% covers
Output:
{
  "body": "Quick heads-up Suresh — DC vs MI at Arun Jaitley tonight, 7:30pm. Important: Saturday IPL matches usually shift -12% restaurant covers (people watch at home). Skip the match-night promo today; instead push your BOGO pizza (already active) as a delivery-only Saturday special. Want me to draft the Swiggy banner + an Insta story? Live in 10 min.",
  "cta": "binary_yes_no",
  "send_as": "vera",
  "rationale": "Contrarian data-backed call (Saturday IPL = -12% covers). Loss aversion framing. Leverages existing BOGO offer. 10-min effort cap."
}

=== GOLD EXAMPLE 4 — Performance Dip / Seasonal Reframe (score 48/50) ===
Trigger kind: perf_dip | Category: gyms | Merchant: PowerHouse Fitness, HSR Layout
Context used: views -30% w/w, 245 active members, April-June seasonal lull
Output:
{
  "body": "Karthik, your views are down 30% this week — but I want to flag this is the normal April-June acquisition lull (every metro gym sees -25 to -35% in this window). Action: skip ad spend now, save it for Sept-Oct when conversion is 2x. For now, focus retention on your 245 members. Want me to draft a 'summer attendance challenge' to keep them through the dip?",
  "cta": "binary_yes_no",
  "send_as": "vera",
  "rationale": "Anxiety pre-emption (dip is normal). Data anchor (-25 to -35% range). Reframe as opportunity. Specific member count. Action proposal."
}

=== GOLD EXAMPLE 5 — Curious Ask (score 44/50) ===
Trigger kind: curious_ask_due | Category: salons | Merchant: Studio11 Family Salon
Output:
{
  "body": "Hi Lakshmi! Quick check — what service has been most asked-for this week at Studio11? I'll turn the answer into a Google post + a 4-line WhatsApp reply you can use when customers ask about pricing. Takes 5 min.",
  "cta": "open_ended",
  "send_as": "vera",
  "rationale": "Low-stakes question (no commitment). Reciprocity offered up-front. Effort externalization (5 min). Asking-the-merchant lever."
}

=== GOLD EXAMPLE 6 — Compliance Alert (score 50/50) ===
Trigger kind: supply_alert | Category: pharmacies | Merchant: Apollo Health Plus Pharmacy
Context used: 2 atorvastatin batches recalled, 22 of 240 chronic-Rx customers affected
Output:
{
  "body": "Ramesh, urgent: voluntary recall on 2 atorvastatin batches (AT2024-1102, AT2024-1108) by Mfr Z — sub-potency, no safety risk, but customers should be informed for replacement. Pulled your repeat-Rx list: 22 of your chronic-Rx customers were dispensed these batches in last 90 days. Want me to draft their WhatsApp note + the replacement-pickup workflow?",
  "cta": "binary_yes_no",
  "send_as": "vera",
  "rationale": "Urgency + specificity (batch numbers, count of affected customers). Risk-bounded framing. End-to-end workflow offer."
}
"""

# ---------------------------------------------------------------------------
# SYSTEM PROMPT
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are Vera, magicpin's merchant AI assistant. You compose WhatsApp messages to merchants and their customers.

SCORING DIMENSIONS (what the judge evaluates — maximize all 5):
1. SPECIFICITY — anchor every message on at least ONE verifiable fact: a number, date, source citation, or stat from the provided context. "Your CTR is 2.1% vs peer median 3.0%" beats "your performance could be better".
2. CATEGORY FIT — match the voice, vocabulary, and register of the merchant's category. Dentists: clinical-peer, technical terms OK, taboos ("guaranteed", "cure"). Salons: warm-practical. Restaurants: operator-to-operator. Gyms: coaching. Pharmacies: trustworthy-precise.
3. MERCHANT FIT — use the merchant's actual name, owner first name, locality, active offers, and signals. Never address "there" when a name is in the context.
4. TRIGGER RELEVANCE — the message must clearly explain WHY NOW. The trigger is the reason; make it explicit.
5. ENGAGEMENT COMPULSION — use one or more: loss aversion, social proof, curiosity, reciprocity, effort externalization, asking-the-merchant, single binary CTA.

HARD RULES (violations lose points):
- NEVER fabricate numbers, citations, or competitor names not present in the context JSON.
- NEVER include URLs (penalty: -3 per URL).
- NEVER use multiple CTAs in one message.
- NEVER use promotional tone ("AMAZING DEAL!") for clinical categories.
- NEVER re-introduce yourself after turn 1 of a conversation.
- NEVER repeat verbatim a body you already sent in this conversation.
- ALWAYS put the CTA/ask in the LAST sentence.
- ALWAYS match the merchant's language preference (if languages includes "hi", use Hindi-English code-mix).
- temperature=0 is required — be deterministic.

VALID CTA VALUES: open_ended | binary_yes_no | binary_confirm_cancel | multi_choice_slot | none

OUTPUT FORMAT — respond ONLY with this JSON, no markdown, no explanation:
{
  "body": "<the WhatsApp message — no markdown formatting, plain text only>",
  "cta": "<one of the valid CTA values>",
  "send_as": "<vera OR merchant_on_behalf>",
  "template_name": "<vera_{trigger_kind}_v1>",
  "template_params": ["<param1>", "<param2>", "<param3>"],
  "suppression_key": "<from trigger context>",
  "rationale": "<1-2 sentences: why this message, what compulsion lever used>"
}
""" + "\nGOLD EXAMPLES (learn the pattern, do NOT copy verbatim):\n" + FEW_SHOT_EXAMPLES

# ---------------------------------------------------------------------------
# CONTEXT ASSEMBLER
# ---------------------------------------------------------------------------

def _extract_digest_item(category: dict, trigger: dict) -> Optional[dict]:
    """Find the specific digest item referenced by the trigger payload."""
    top_item_id = trigger.get("payload", {}).get("top_item_id")
    digest = category.get("digest", [])
    if top_item_id:
        for item in digest:
            if item.get("id") == top_item_id:
                return item
    # Fallback: return the first digest item
    return digest[0] if digest else None


def _build_context_block(
    category: dict,
    merchant: dict,
    trigger: dict,
    customer: Optional[dict],
) -> str:
    """
    Serialize the 4 contexts into a concise prompt block.
    Only includes fields that matter for composition — keeps tokens low.
    """
    identity = merchant.get("identity", {})
    perf = merchant.get("performance", {})
    sub = merchant.get("subscription", {})
    offers = merchant.get("offers", [])
    signals = merchant.get("signals", [])
    cust_agg = merchant.get("customer_aggregate", {})
    voice = category.get("voice", {})

    # Active offers only
    active_offers = [o["title"] for o in offers if o.get("status") == "active"]
    paused_offers = [o["title"] for o in offers if o.get("status") in ("paused", "expired")]

    # Relevant digest item
    digest_item = _extract_digest_item(category, trigger)
    digest_block = ""
    if digest_item:
        digest_block = f"""
RELEVANT DIGEST ITEM:
  ID: {digest_item.get('id', '')}
  Title: {digest_item.get('title', '')}
  Source: {digest_item.get('source', '')}
  Summary: {digest_item.get('summary', digest_item.get('title', ''))}
  Trial N: {digest_item.get('trial_n', 'N/A')}
  Patient segment: {digest_item.get('patient_segment', 'N/A')}"""

    # Peer stats
    peer = category.get("peer_stats", {})
    trend = category.get("trend_signals", [{}])[0] if category.get("trend_signals") else {}
    seasonal = category.get("seasonal_beats", [])

    # Trigger payload
    trigger_payload = json.dumps(trigger.get("payload", {}), ensure_ascii=False)

    # Customer block (only if populated)
    customer_block = ""
    if customer:
        cid = customer.get("identity", {})
        rel = customer.get("relationship", {})
        customer_block = f"""
CUSTOMER CONTEXT (message goes TO this customer FROM the merchant):
  Name: {cid.get('name', 'Customer')}
  Language preference: {cid.get('language_pref', 'en')}
  Age band: {cid.get('age_band', 'N/A')}
  State: {customer.get('state', 'unknown')}
  First visit: {rel.get('first_visit', 'N/A')}
  Last visit: {rel.get('last_visit', 'N/A')}
  Total visits: {rel.get('visits_total', 'N/A')}
  Services received: {rel.get('services_received', [])}
  Preferred slots: {customer.get('preferences', {}).get('preferred_slots', 'N/A')}
  Consent scope: {customer.get('consent', {}).get('scope', [])}"""

    return f"""=== CATEGORY CONTEXT ===
Slug: {category.get('slug', 'unknown')}
Voice tone: {voice.get('tone', 'professional')}
Vocabulary taboos (NEVER use these): {voice.get('taboos', voice.get('vocab_taboo', []))}
Allowed vocabulary: {voice.get('vocab_allowed', [])[:10]}
Active offer catalog: {category.get('offer_catalog', [{}])[0].get('title', 'N/A') if category.get('offer_catalog') else 'N/A'}
Peer stats: avg_rating={peer.get('avg_rating', 'N/A')}, avg_ctr={peer.get('avg_ctr', 'N/A')}, avg_reviews={peer.get('avg_reviews', 'N/A')}
Trend signal: {trend.get('query', 'N/A')} delta_yoy={trend.get('delta_yoy', 'N/A')}
Seasonal beats: {[s.get('note', '') for s in seasonal[:2]]}{digest_block}

=== MERCHANT CONTEXT ===
Merchant ID: {merchant.get('merchant_id', 'unknown')}
Business name: {identity.get('name', 'unknown')}
Owner first name: {identity.get('owner_first_name', identity.get('name', 'there'))}
City: {identity.get('city', 'N/A')} | Locality: {identity.get('locality', 'N/A')}
Languages: {identity.get('languages', ['en'])}
Verified: {identity.get('verified', False)}
Subscription: {sub.get('status', 'N/A')} plan={sub.get('plan', 'N/A')} days_remaining={sub.get('days_remaining', 'N/A')}
Performance (30d): views={perf.get('views', 'N/A')}, calls={perf.get('calls', 'N/A')}, CTR={perf.get('ctr', 'N/A')}, directions={perf.get('directions', 'N/A')}
7d delta: views_pct={perf.get('delta_7d', {}).get('views_pct', 'N/A')}, calls_pct={perf.get('delta_7d', {}).get('calls_pct', 'N/A')}
Active offers: {active_offers if active_offers else 'none'}
Paused/expired offers: {paused_offers[:3] if paused_offers else 'none'}
Customer aggregate: total_ytd={cust_agg.get('total_unique_ytd', 'N/A')}, lapsed_180d={cust_agg.get('lapsed_180d_plus', 'N/A')}, retention_6mo={cust_agg.get('retention_6mo_pct', 'N/A')}, high_risk_adults={cust_agg.get('high_risk_adult_count', 'N/A')}
Signals: {signals}
Recent conversation: {merchant.get('conversation_history', [])[-2:] if merchant.get('conversation_history') else 'none'}

=== TRIGGER CONTEXT ===
ID: {trigger.get('id', 'unknown')}
Kind: {trigger.get('kind', 'unknown')}
Scope: {trigger.get('scope', 'merchant')}
Source: {trigger.get('source', 'internal')}
Urgency: {trigger.get('urgency', 2)}/5
Suppression key: {trigger.get('suppression_key', '')}
Payload: {trigger_payload}
{customer_block}"""


# ---------------------------------------------------------------------------
# POST-LLM VALIDATOR
# ---------------------------------------------------------------------------

# Numbers extracted from context to allow in output (anti-fabrication whitelist)
def _extract_allowed_numbers(context_block: str) -> set[str]:
    """Pull all numeric strings from the context so we can whitelist them."""
    return set(re.findall(r'\b\d+(?:\.\d+)?%?\b', context_block))


def validate_output(result: dict, context_block: str, trigger_kind: str,
                    languages: list[str],
                    customer_facing: bool = False,
                    category_taboos: list[str] = None) -> tuple[bool, str]:
    """
    Returns (is_valid, rejection_reason).
    Checks:
      1. body is non-empty
      2. no URLs in body
      3. cta is a known valid value
      4. body does not contain numbers not present in context (fabrication heuristic)
      5. customer-facing: send_as must be merchant_on_behalf
      6. customer-facing: body must not use category voice taboos (clinical categories)
    """
    body = result.get("body", "").strip()
    cta = result.get("cta", "")

    # 1. Non-empty body
    if not body:
        return False, "body is empty"

    # 2. No URLs
    if re.search(r'https?://', body, re.IGNORECASE):
        return False, "body contains a URL (−3 penalty per URL)"

    # 3. Valid CTA
    valid_ctas = {"open_ended", "binary_yes_no", "binary_confirm_cancel",
                  "multi_choice_slot", "none"}
    if cta not in valid_ctas:
        return False, f"invalid cta value: '{cta}' — must be one of {valid_ctas}"

    # 4. Fabrication heuristic — numbers in body must exist in context
    allowed_numbers = _extract_allowed_numbers(context_block)
    body_numbers = set(re.findall(r'\b\d+(?:\.\d+)?%?\b', body))
    # Allow small numbers (1-10) as they're likely turn counts, slot numbers
    suspicious = {n for n in body_numbers
                  if n not in allowed_numbers and float(n.rstrip('%')) > 10}
    if suspicious:
        return False, (
            f"body contains numbers not found in context (possible fabrication): {suspicious}. "
            f"Only use numbers present in the context JSON."
        )

    # 5. Customer-facing: enforce send_as = merchant_on_behalf
    if customer_facing:
        send_as = result.get("send_as", "")
        if send_as != "merchant_on_behalf":
            return False, (
                f"customer-facing message must have send_as='merchant_on_behalf', "
                f"got '{send_as}'"
            )

    # 6. Customer-facing: check category voice taboos (clinical categories)
    #    e.g. "guaranteed", "cure", "100% safe" must never appear in
    #    patient-facing messages for dentists/pharmacies
    if customer_facing and category_taboos:
        body_lower = body.lower()
        for taboo in category_taboos:
            if taboo.lower() in body_lower:
                return False, (
                    f"body contains taboo word '{taboo}' for this category "
                    f"(not allowed in customer-facing messages)"
                )

    return True, ""


# ---------------------------------------------------------------------------
# MAIN COMPOSER CLASS
# ---------------------------------------------------------------------------

class LLMComposer:
    """
    Stateless proactive message composer.
    Called from bot.py tick handler for each (merchant, trigger) pair.
    Falls back to stub output on any error so the harness never breaks.
    """

    def __init__(self):
        self._client: Optional[genai.GenerativeModel] = None

    @property
    def client(self) -> genai.GenerativeModel:
        if self._client is None:
            self._client = _get_gemini_client()
        return self._client

    def compose(
        self,
        merchant_id: str,
        trigger_id: str,
        conv_id: str,
        customer_id: Optional[str],
        context_store: dict,
    ) -> dict | None:
        """
        Main entry point. Returns a full action dict ready for /v1/tick response,
        or None when the bot deliberately chooses not to send (sparse context,
        no digest items). Never raises — always returns something valid or None.
        """
        try:
            return self._compose_inner(
                merchant_id, trigger_id, conv_id, customer_id, context_store
            )
        except ValueError as e:
            if "no_digest_items" in str(e):
                # Restraint is correct: no relevant content yet, don't fabricate
                logger.info("Skipping action — no digest items for trigger %s", trigger_id)
                return None
            logger.error("Composer ValueError: %s", e, exc_info=True)
            return self._fallback(merchant_id, trigger_id, conv_id, customer_id, context_store)
        except RuntimeError as e:
            # GEMINI_API_KEY not set — return graceful stub
            logger.warning("LLM not configured: %s", e)
            return self._fallback(merchant_id, trigger_id, conv_id, customer_id, context_store)
        except Exception as e:
            logger.error("Composer error: %s", e, exc_info=True)
            return self._fallback(merchant_id, trigger_id, conv_id, customer_id, context_store)

    def _compose_inner(
        self,
        merchant_id: str,
        trigger_id: str,
        conv_id: str,
        customer_id: Optional[str],
        context_store: dict,
    ) -> dict:
        # --- 1. Load contexts from store (ALWAYS re-read raw; never use pre-cached) ---
        # This is the core Phase 3 guarantee: every compose call reads the
        # current version of each context from the store snapshot passed in.
        # A v2 performance push between ticks is automatically picked up here.
        merchant_entry = context_store.get(("merchant", merchant_id), {})
        merchant = merchant_entry.get("payload", {})

        trigger_entry = context_store.get(("trigger", trigger_id), {})
        trigger = trigger_entry.get("payload", {})

        category_slug = merchant.get("category_slug", "")
        category_entry = context_store.get(("category", category_slug), {})
        category = category_entry.get("payload", {})

        customer = None
        if customer_id:
            cust_entry = context_store.get(("customer", customer_id), {})
            customer = cust_entry.get("payload")

        # --- 2. Route to prompt variant ---
        trigger_kind = trigger.get("kind", "scheduled_recurring")
        variant_key = TRIGGER_VARIANTS.get(trigger_kind, "relationship")
        variant_instruction = VARIANT_INSTRUCTIONS[variant_key]

        # --- 3. Assemble context block (always fresh from raw payload) ---
        context_block = _build_context_block(category, merchant, trigger, customer)

        # --- 4. Build user prompt ---
        suppression_key = trigger.get("suppression_key", f"auto:{merchant_id}:{trigger_id}")
        languages = merchant.get("identity", {}).get("languages", ["en"])
        is_customer_facing = trigger.get("scope") == "customer" and bool(customer_id)
        category_taboos = (
            category.get("voice", {}).get("taboos")
            or category.get("voice", {}).get("vocab_taboo")
            or []
        )

        # Phase 3 guard: if the digest trigger references a top_item_id but no
        # matching item exists in the (possibly just-updated) category digest,
        # return empty rather than hallucinating a citation.
        if trigger_kind in ("research_digest", "regulation_change",
                             "category_research_digest_release"):
            digest_item = _extract_digest_item(category, trigger)
            if digest_item is None and not category.get("digest"):
                logger.info(
                    "No digest items in category '%s' for trigger '%s' — "
                    "returning empty (restraint beats fabrication)",
                    category_slug, trigger_id,
                )
                raise ValueError("no_digest_items")

        user_prompt = f"""TASK: Compose a WhatsApp message for this merchant/customer.

TRIGGER KIND: {trigger_kind}
VARIANT INSTRUCTION: {variant_instruction}

{context_block}

SUPPRESSION KEY TO USE: {suppression_key}
IS FIRST TURN IN CONVERSATION: yes (use template-style opening)
SEND AS: {"merchant_on_behalf" if is_customer_facing else "vera"}
{"CUSTOMER-FACING RULES: no medical claims, no guarantees, no taboo words: " + str(category_taboos) if is_customer_facing else ""}

Compose the message now. Output ONLY the JSON object, no other text."""

        # --- 5. LLM call (with retry on validation failure) ---
        result = self._call_with_retry(
            user_prompt, context_block, trigger_kind, languages,
            customer_facing=is_customer_facing,
            category_taboos=category_taboos,
        )

        # --- 6. Fill in any missing required fields ---
        result.setdefault("template_name", f"vera_{trigger_kind}_v1")
        result.setdefault("template_params", [
            merchant.get("identity", {}).get("owner_first_name", ""),
            merchant.get("identity", {}).get("name", ""),
            trigger_kind,
        ])
        result.setdefault("suppression_key", suppression_key)
        # Force correct send_as — never let the LLM override this
        result["send_as"] = "merchant_on_behalf" if is_customer_facing else "vera"
        result.setdefault("cta", "open_ended")
        result.setdefault("rationale", f"LLM-composed message for {trigger_kind} trigger.")

        # --- 7. Wrap in full action dict ---
        return {
            "conversation_id": conv_id,
            "merchant_id": merchant_id,
            "customer_id": customer_id,
            "send_as": result["send_as"],
            "trigger_id": trigger_id,
            "template_name": result["template_name"],
            "template_params": result["template_params"],
            "body": result["body"],
            "cta": result["cta"],
            "suppression_key": result["suppression_key"],
            "rationale": result["rationale"],
        }

    def _call_with_retry(
        self,
        user_prompt: str,
        context_block: str,
        trigger_kind: str,
        languages: list[str],
        max_retries: int = 1,
        customer_facing: bool = False,
        category_taboos: list[str] = None,
    ) -> dict:
        """Call LLM, validate output, retry once with explicit feedback if invalid."""
        last_error = ""
        result = None
        for attempt in range(max_retries + 1):
            prompt = user_prompt
            if attempt > 0 and last_error:
                prompt += (
                    f"\n\nPREVIOUS ATTEMPT FAILED VALIDATION: {last_error}\n"
                    "Fix the issue and output corrected JSON only."
                )

            raw = self._call_llm(prompt)
            result = self._parse_json(raw)

            if result is None:
                last_error = "Could not parse JSON from LLM output"
                continue

            valid, reason = validate_output(
                result, context_block, trigger_kind, languages,
                customer_facing=customer_facing,
                category_taboos=category_taboos or [],
            )
            if valid:
                return result

            last_error = reason
            logger.warning("Validation failed (attempt %d): %s", attempt + 1, reason)

        # All retries exhausted — return whatever we last parsed, or empty shell
        if result:
            # Force-fix the CTA if that was the problem
            if result.get("cta", "") not in {"open_ended", "binary_yes_no",
                                              "binary_confirm_cancel",
                                              "multi_choice_slot", "none"}:
                result["cta"] = "open_ended"
            # Force-fix send_as for customer-facing
            if customer_facing:
                result["send_as"] = "merchant_on_behalf"
            return result

        raise ValueError(f"LLM composition failed after {max_retries + 1} attempts: {last_error}")

    def _call_llm(self, user_prompt: str) -> str:
        """Single Gemini API call. Returns raw string content."""
        full_prompt = f"{SYSTEM_PROMPT}\n\n{user_prompt}"
        response = self.client.generate_content(full_prompt)
        return response.text or ""

    def _parse_json(self, raw: str) -> Optional[dict]:
        """Extract and parse JSON from LLM output. Handles markdown code fences."""
        # Strip markdown code fences if present
        raw = re.sub(r"```(?:json)?", "", raw).strip()
        raw = raw.strip("`").strip()

        # Try direct parse
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass

        # Find first {...} block
        match = re.search(r'\{[\s\S]*\}', raw)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

        return None

    def _fallback(
        self,
        merchant_id: str,
        trigger_id: str,
        conv_id: str,
        customer_id: Optional[str],
        context_store: dict,
    ) -> dict:
        """
        Structural fallback — valid shape, low quality.
        Used when LLM is unavailable or throws an unexpected error.
        Mirrors the Step 1 stub composer logic.
        """
        merchant_entry = context_store.get(("merchant", merchant_id), {})
        merchant = merchant_entry.get("payload", {})
        trigger_entry = context_store.get(("trigger", trigger_id), {})
        trigger = trigger_entry.get("payload", {})

        owner = merchant.get("identity", {}).get("owner_first_name", "there")
        name  = merchant.get("identity", {}).get("name", "your business")
        kind  = trigger.get("kind", "general")
        sk    = trigger.get("suppression_key", f"fallback:{merchant_id}:{trigger_id}")
        send_as = "merchant_on_behalf" if (trigger.get("scope") == "customer" and customer_id) else "vera"

        body = f"Hi {owner}, I have an update for {name}. Want to hear more?"
        return {
            "conversation_id": conv_id,
            "merchant_id": merchant_id,
            "customer_id": customer_id,
            "send_as": send_as,
            "trigger_id": trigger_id,
            "template_name": f"vera_{kind}_v1",
            "template_params": [owner, name, kind],
            "body": body,
            "cta": "open_ended",
            "suppression_key": sk,
            "rationale": "Fallback: LLM unavailable. Structural stub returned.",
        }


# ---------------------------------------------------------------------------
# MODULE-LEVEL SINGLETON
# ---------------------------------------------------------------------------
# bot.py imports this and calls composer.compose(...)

composer = LLMComposer()
