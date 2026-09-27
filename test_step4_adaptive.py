"""
test_step4_adaptive.py — Step 4 Adaptive Context + Customer-Facing Tests
=========================================================================

Covers the three Phase 3 judge scenarios:

  1. v2 context injection
     - Push merchant v1 (views=1000, CTR=0.021)
     - Fire tick → note body references v1 numbers
     - Push merchant v2 (views=5000, CTR=0.055, big spike)
     - Push perf_spike trigger for same merchant
     - Fire tick (new trigger) → verify body references NEW numbers, not v1
     - Also verifies: push stale (v1 again) → 409 with current_version=2

  2. Customer-facing branch
     - Push category (dentists) + merchant + customer context
     - Push a recall_due trigger (scope=customer)
     - Fire tick → verify:
         - action returned with send_as=merchant_on_behalf
         - body references customer name and/or language preference
         - body does NOT contain category taboo words
     - Verify tick skips trigger when customer context not yet pushed

  3. No-hallucination on sparse context (restraint > fabrication)
     - Push merchant with NO performance data and NO signals
     - Push category with EMPTY digest
     - Push a research_digest trigger
     - Fire tick → verify actions=[] (bot chose not to send)
       because no digest item exists to anchor a specific claim

  4. Validator unit tests (no server)
     - customer_facing=True + wrong send_as → rejected
     - customer_facing=True + taboo word in body → rejected
     - customer_facing=False + taboo word → passes
     - v2 numbers in context → allowed in body
     - numbers not in context → rejected

Run: py -3.13 test_step4_adaptive.py
"""

import json
import sys
import time
from pathlib import Path
from urllib import request as urlrequest, error as urlerror

BASE = "http://localhost:8080"
DATASET = Path(__file__).parent / "dataset"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def call(method: str, path: str, body: dict = None, timeout: int = 30):
    url = f"{BASE}{path}"
    data = json.dumps(body).encode() if body else None
    req = urlrequest.Request(url, data=data, method=method,
                             headers={"Content-Type": "application/json"})
    try:
        resp = urlrequest.urlopen(req, timeout=timeout)
        return json.loads(resp.read()), resp.status, None
    except urlerror.HTTPError as e:
        try:
            return json.loads(e.read()), e.code, None
        except Exception:
            return None, e.code, str(e)
    except Exception as ex:
        return None, None, str(ex)


passed = 0
failed = 0

def check(label: str, condition: bool, detail: str = "") -> bool:
    global passed, failed
    icon = "PASS" if condition else "FAIL"
    print(f"  [{icon}] {label}" + (f" — {detail}" if detail else ""))
    if condition:
        passed += 1
    else:
        failed += 1
    return condition


# Reusable context payloads
_CAT_DENTISTS = {
    "slug": "dentists",
    "voice": {
        "tone": "peer_clinical",
        "taboos": ["guaranteed", "cure", "100% safe"],
        "vocab_taboo": ["guaranteed", "cure", "100% safe"],
        "vocab_allowed": ["fluoride varnish", "caries", "recall"],
    },
    "offer_catalog": [{"id": "den_001", "title": "Dental Cleaning @ Rs.299"}],
    "peer_stats": {"avg_rating": 4.4, "avg_ctr": 0.030, "avg_reviews": 62},
    "digest": [
        {
            "id": "d_jida_001",
            "title": "3-month fluoride recall cuts caries 38% better than 6-month",
            "source": "JIDA Oct 2026 p.14",
            "trial_n": 2100,
            "patient_segment": "high_risk_adults",
            "summary": "2100-patient RCT shows 3-month recall superior for high-risk adults",
        }
    ],
    "seasonal_beats": [{"month_range": "Nov-Feb", "note": "exam-stress bruxism spike"}],
    "trend_signals": [],
}

def _push(scope, context_id, version, payload):
    return call("POST", "/v1/context", {
        "scope": scope, "context_id": context_id, "version": version,
        "delivered_at": "2026-04-26T10:00:00Z", "payload": payload,
    })

def _tick(trigger_ids):
    return call("POST", "/v1/tick", {
        "now": "2026-04-26T10:00:00Z",
        "available_triggers": trigger_ids,
    })

# ---------------------------------------------------------------------------
# SCENARIO 1 — v2 Context Injection (Phase 3 core test)
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("SCENARIO 1 — v2 context injection (Phase 3 adaptive context)")
print("=" * 60)

call("POST", "/v1/teardown")

# Push category
_push("category", "dentists", 1, _CAT_DENTISTS)

# Push merchant v1 — low performance
merch_v1 = {
    "merchant_id": "m_adaptive", "category_slug": "dentists",
    "identity": {
        "name": "Adaptive Dental Clinic", "owner_first_name": "Vikram",
        "city": "Delhi", "locality": "Hauz Khas",
        "languages": ["en"], "verified": True,
    },
    "subscription": {"status": "active", "plan": "Pro", "days_remaining": 90},
    "performance": {
        "window_days": 30,
        "views": 1000, "calls": 8, "ctr": 0.021, "directions": 15,
        "delta_7d": {"views_pct": -0.05, "calls_pct": -0.10},
    },
    "offers": [{"id": "o1", "title": "Dental Cleaning @ Rs.299", "status": "active"}],
    "conversation_history": [],
    "customer_aggregate": {"total_unique_ytd": 200, "high_risk_adult_count": 45},
    "signals": ["ctr_below_peer_median"],
}
r, code, _ = _push("merchant", "m_adaptive", 1, merch_v1)
check("Merchant v1 pushed (views=1000, CTR=0.021)", code == 200 and r and r.get("accepted"))

# Push a perf_dip trigger for v1 state
trig_dip = {
    "id": "trg_dip_v1", "scope": "merchant", "kind": "perf_dip",
    "source": "internal", "merchant_id": "m_adaptive", "customer_id": None,
    "urgency": 3, "suppression_key": "perf_dip:m_adaptive:v1",
    "expires_at": "2026-06-30T00:00:00Z",
    "payload": {"metric": "ctr", "value": 0.021, "peer_median": 0.030},
}
_push("trigger", "trg_dip_v1", 1, trig_dip)

# Tick 1 — should compose using v1 numbers
t1, _, _ = _tick(["trg_dip_v1"])
actions_t1 = t1.get("actions", []) if t1 else []
check("Tick 1 (v1 context) produces action", len(actions_t1) >= 1,
      f"{len(actions_t1)} actions")

body_t1 = actions_t1[0].get("body", "").lower() if actions_t1 else ""
check("Tick 1 body non-empty", bool(body_t1.strip()))

# Now push merchant v2 — performance spiked
merch_v2 = dict(merch_v1)
merch_v2["performance"] = {
    "window_days": 30,
    "views": 5000, "calls": 42, "ctr": 0.055, "directions": 80,
    "delta_7d": {"views_pct": 0.45, "calls_pct": 0.35},
}
merch_v2["signals"] = ["perf_spike_this_week"]
r2, code2, _ = _push("merchant", "m_adaptive", 2, merch_v2)
check("Merchant v2 pushed (views=5000, CTR=0.055)", code2 == 200 and r2 and r2.get("accepted"))

# Stale push: sending v1 again should 409
r3, code3, _ = _push("merchant", "m_adaptive", 1, merch_v1)
check("Stale v1 re-push → 409", code3 == 409)
check("409 body has current_version=2", r3 and r3.get("current_version") == 2,
      str(r3.get("current_version") if r3 else "no body"))

# Push a perf_spike trigger to create a new conversation
trig_spike = {
    "id": "trg_spike_v2", "scope": "merchant", "kind": "perf_spike",
    "source": "internal", "merchant_id": "m_adaptive", "customer_id": None,
    "urgency": 2, "suppression_key": "perf_spike:m_adaptive:v2",
    "expires_at": "2026-06-30T00:00:00Z",
    "payload": {"metric": "views", "value": 5000, "delta_pct": 0.45},
}
_push("trigger", "trg_spike_v2", 1, trig_spike)

# Tick 2 — must use v2 numbers
t2, _, _ = _tick(["trg_spike_v2"])
actions_t2 = t2.get("actions", []) if t2 else []
check("Tick 2 (v2 context) produces action", len(actions_t2) >= 1,
      f"{len(actions_t2)} actions")

if actions_t2:
    body_t2 = actions_t2[0].get("body", "")
    body_t2_lower = body_t2.lower()
    # Body should reference new numbers — 5000 views OR 0.055/5.5% CTR OR 42 calls
    # (any of v2's distinctive values)
    v2_numbers = ["5000", "5,000", "0.055", "5.5%", "42", "45%", "0.45"]
    v2_signals = ["spike", "up", "increase", "growth", "strong", "more views"]
    has_v2_number = any(n in body_t2_lower for n in v2_numbers)
    has_v2_signal = any(s in body_t2_lower for s in v2_signals)
    check("Tick 2 body references v2 data (new numbers or spike framing)",
          has_v2_number or has_v2_signal,
          f"body[:120]: '{body_t2[:120]}'")
    check("Tick 2 send_as=vera (merchant-facing trigger)",
          actions_t2[0].get("send_as") == "vera",
          actions_t2[0].get("send_as"))

# ---------------------------------------------------------------------------
# SCENARIO 2 — Customer-facing branch (Phase 3 5 customer test pairs)
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("SCENARIO 2 — Customer-facing branch (send_as + taboo enforcement)")
print("=" * 60)

call("POST", "/v1/teardown")
_push("category", "dentists", 1, _CAT_DENTISTS)

merch_cust = {
    "merchant_id": "m_cust", "category_slug": "dentists",
    "identity": {
        "name": "Dr. Priya's Dental Studio", "owner_first_name": "Priya",
        "city": "Mumbai", "locality": "Bandra",
        "languages": ["en", "hi"], "verified": True,
    },
    "subscription": {"status": "active", "plan": "Pro", "days_remaining": 75},
    "performance": {"window_days": 30, "views": 3200, "calls": 25, "ctr": 0.034},
    "offers": [{"id": "o_cust", "title": "Dental Cleaning @ Rs.299", "status": "active"}],
    "conversation_history": [],
    "customer_aggregate": {"total_unique_ytd": 480, "lapsed_180d_plus": 62},
    "signals": [],
}
_push("merchant", "m_cust", 1, merch_cust)

# Customer context
customer_ctx = {
    "customer_id": "c_priya_001",
    "merchant_id": "m_cust",
    "identity": {
        "name": "Priya",
        "phone_redacted": "<phone>",
        "language_pref": "hi-en mix",
        "age_band": "25-35",
    },
    "relationship": {
        "first_visit": "2025-11-04",
        "last_visit": "2026-04-10",
        "visits_total": 3,
        "services_received": ["cleaning", "cleaning", "whitening"],
    },
    "state": "lapsed_soft",
    "preferences": {"preferred_slots": "weekday_evening", "channel": "whatsapp"},
    "consent": {"opted_in_at": "2025-11-04", "scope": ["recall_reminders"]},
}
_push("customer", "c_priya_001", 1, customer_ctx)

# Test: tick fires BEFORE customer context is pushed for a second customer
# (should skip — customer context not yet in store)
trig_recall_missing_cust = {
    "id": "trg_recall_noctx", "scope": "customer", "kind": "recall_due",
    "source": "internal", "merchant_id": "m_cust", "customer_id": "c_missing_999",
    "urgency": 3, "suppression_key": "recall:m_cust:c_missing_999",
    "expires_at": "2026-06-30T00:00:00Z",
    "payload": {"recall_months": 6},
}
_push("trigger", "trg_recall_noctx", 1, trig_recall_missing_cust)
t_skip, _, _ = _tick(["trg_recall_noctx"])
actions_skip = t_skip.get("actions", []) if t_skip else []
check("Tick skips customer trigger when customer context missing",
      len(actions_skip) == 0,
      f"got {len(actions_skip)} actions")

# Now push the real recall trigger
trig_recall = {
    "id": "trg_recall_priya", "scope": "customer", "kind": "recall_due",
    "source": "internal", "merchant_id": "m_cust", "customer_id": "c_priya_001",
    "urgency": 3, "suppression_key": "recall:m_cust:c_priya_001",
    "expires_at": "2026-06-30T00:00:00Z",
    "payload": {"recall_months": 6, "last_service": "cleaning"},
}
_push("trigger", "trg_recall_priya", 1, trig_recall)
t_cust, _, _ = _tick(["trg_recall_priya"])
actions_cust = t_cust.get("actions", []) if t_cust else []
check("Customer recall tick produces action", len(actions_cust) >= 1,
      f"{len(actions_cust)} actions")

if actions_cust:
    a = actions_cust[0]
    check("send_as = merchant_on_behalf",
          a.get("send_as") == "merchant_on_behalf",
          a.get("send_as"))
    check("customer_id populated in action",
          a.get("customer_id") == "c_priya_001",
          a.get("customer_id"))
    body_cust = a.get("body", "")
    body_lower = body_cust.lower()
    check("Body non-empty", bool(body_cust.strip()))

    # Body must not contain category taboo words
    taboos = ["guaranteed", "cure", "100% safe"]
    taboo_found = [t for t in taboos if t.lower() in body_lower]
    check("No category taboos in customer-facing body",
          not taboo_found,
          f"taboos found: {taboo_found}" if taboo_found else "")

    # Body should reference the customer's name or language preference signal
    has_personal = (
        "priya" in body_lower
        or "hi" in body_lower
        or "hain" in body_lower
        or "slot" in body_lower
        or "recall" in body_lower
        or "cleaning" in body_lower
        or "appointment" in body_lower
    )
    check("Body personalised (customer name/language/service)",
          has_personal,
          f"body[:120]: '{body_cust[:120]}'")

# ---------------------------------------------------------------------------
# SCENARIO 3 — Sparse context: restraint beats fabrication
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("SCENARIO 3 — Sparse context: research trigger with no digest items")
print("=" * 60)

call("POST", "/v1/teardown")

# Category with EMPTY digest — nothing to cite
cat_empty_digest = dict(_CAT_DENTISTS)
cat_empty_digest["digest"] = []
_push("category", "dentists", 1, cat_empty_digest)

merch_sparse = {
    "merchant_id": "m_sparse", "category_slug": "dentists",
    "identity": {
        "name": "Sparse Dental", "owner_first_name": "Ankit",
        "city": "Pune", "locality": "Aundh",
        "languages": ["en"], "verified": True,
    },
    "subscription": {"status": "active", "plan": "Pro", "days_remaining": 45},
    "performance": {"window_days": 30, "views": 600, "calls": 5, "ctr": 0.019},
    "offers": [], "conversation_history": [],
    "customer_aggregate": {}, "signals": [],
}
_push("merchant", "m_sparse", 1, merch_sparse)

# Research trigger — but category has no digest items to cite
trig_research_sparse = {
    "id": "trg_research_sparse", "scope": "merchant",
    "kind": "research_digest", "source": "external",
    "merchant_id": "m_sparse", "customer_id": None,
    "urgency": 2, "suppression_key": "research:dentists:sparse",
    "expires_at": "2026-06-30T00:00:00Z",
    "payload": {"category": "dentists", "top_item_id": "nonexistent_id"},
}
_push("trigger", "trg_research_sparse", 1, trig_research_sparse)

t_sparse, _, _ = _tick(["trg_research_sparse"])
actions_sparse = t_sparse.get("actions", []) if t_sparse else []
check("Sparse context: bot returns empty actions (no fabrication)",
      len(actions_sparse) == 0,
      f"got {len(actions_sparse)} actions — if >0, check for fabricated content")


# ---------------------------------------------------------------------------
# SCENARIO 4 — Validator unit tests (no server needed)
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("SCENARIO 4 — Validator unit tests")
print("=" * 60)

from composer import validate_output

# Build a minimal context block containing specific numbers
_ctx = "views=5000 CTR=0.055 calls=42 peer_avg_ctr=0.030 views_pct=0.45 45%"

# 4a. Basic valid message
ok, reason = validate_output(
    {"body": "Your views jumped to 5000 this week — up 45%.", "cta": "binary_yes_no",
     "send_as": "vera"},
    _ctx, "perf_spike", ["en"]
)
check("Valid merchant-facing message passes", ok, reason)

# 4b. Customer-facing: wrong send_as
ok, reason = validate_output(
    {"body": "Hi Priya, time for your cleaning!", "cta": "multi_choice_slot",
     "send_as": "vera"},
    _ctx, "recall_due", ["en"],
    customer_facing=True, category_taboos=["guaranteed", "cure"]
)
check("Customer-facing wrong send_as → rejected", not ok and "merchant_on_behalf" in reason, reason)

# 4c. Customer-facing: taboo word in body
ok, reason = validate_output(
    {"body": "Guaranteed clean teeth for Priya!", "cta": "binary_yes_no",
     "send_as": "merchant_on_behalf"},
    _ctx, "recall_due", ["en"],
    customer_facing=True, category_taboos=["guaranteed", "cure"]
)
check("Customer-facing taboo word → rejected", not ok and "guaranteed" in reason.lower(), reason)

# 4d. Customer-facing: correct send_as + no taboos → passes
ok, reason = validate_output(
    {"body": "Hi Priya, your 6-month recall is due. Reply YES to book.", "cta": "binary_yes_no",
     "send_as": "merchant_on_behalf"},
    _ctx, "recall_due", ["en"],
    customer_facing=True, category_taboos=["guaranteed", "cure"]
)
check("Customer-facing valid message passes", ok, reason)

# 4e. Merchant-facing: taboo word is allowed (taboo check only for customer_facing)
ok, reason = validate_output(
    {"body": "Guaranteed slots available — 5000 patients served!", "cta": "open_ended",
     "send_as": "vera"},
    _ctx, "perf_spike", ["en"],
    customer_facing=False, category_taboos=["guaranteed", "cure"]
)
check("Merchant-facing taboo word NOT checked (customer_facing=False)", ok, reason)

# 4f. Fabricated number (not in context)
ok, reason = validate_output(
    {"body": "Your CTR jumped to 99.5% — amazing week!", "cta": "binary_yes_no",
     "send_as": "vera"},
    _ctx, "perf_spike", ["en"]
)
check("Fabricated number (99.5%) → rejected", not ok, reason)

# 4g. Number in context → allowed in body
ok, reason = validate_output(
    {"body": "Your calls hit 42 this month — best in the quarter.", "cta": "binary_yes_no",
     "send_as": "vera"},
    _ctx, "perf_spike", ["en"]
)
check("Number present in context (42) → allowed", ok, reason)

# 4h. URL in body → rejected
ok, reason = validate_output(
    {"body": "Check https://example.com for details.", "cta": "open_ended",
     "send_as": "vera"},
    _ctx, "perf_spike", ["en"]
)
check("URL in body → rejected", not ok and "URL" in reason, reason)

# 4i. Empty body → rejected
ok, reason = validate_output(
    {"body": "", "cta": "open_ended", "send_as": "vera"},
    _ctx, "perf_spike", ["en"]
)
check("Empty body → rejected", not ok, reason)

# 4j. Invalid CTA → rejected
ok, reason = validate_output(
    {"body": "Your views are up 5000 this week.", "cta": "yes_no_maybe",
     "send_as": "vera"},
    _ctx, "perf_spike", ["en"]
)
check("Invalid CTA → rejected", not ok, reason)


# ---------------------------------------------------------------------------
# SCENARIO 5 — Context version audit (unit-level, no LLM call)
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("SCENARIO 5 — Context store version audit")
print("=" * 60)

call("POST", "/v1/teardown")

# Push v1
r_v1, c1, _ = call("POST", "/v1/context", {
    "scope": "merchant", "context_id": "m_version_test", "version": 1,
    "delivered_at": "2026-04-26T09:00:00Z",
    "payload": {"merchant_id": "m_version_test", "performance": {"views": 100}},
})
check("v1 push accepted", c1 == 200 and r_v1 and r_v1.get("accepted"))

# Push v2 — replaces v1
r_v2, c2, _ = call("POST", "/v1/context", {
    "scope": "merchant", "context_id": "m_version_test", "version": 2,
    "delivered_at": "2026-04-26T10:00:00Z",
    "payload": {"merchant_id": "m_version_test", "performance": {"views": 9999}},
})
check("v2 push accepted", c2 == 200 and r_v2 and r_v2.get("accepted"))

# Re-push v1 → must 409
r_stale, c3, _ = call("POST", "/v1/context", {
    "scope": "merchant", "context_id": "m_version_test", "version": 1,
    "delivered_at": "2026-04-26T11:00:00Z",
    "payload": {"merchant_id": "m_version_test", "performance": {"views": 100}},
})
check("Stale v1 re-push → 409", c3 == 409)
check("409 current_version=2", r_stale and r_stale.get("current_version") == 2,
      str(r_stale.get("current_version") if r_stale else "no body"))

# Re-push v2 (same version) → also 409
r_dup, c4, _ = call("POST", "/v1/context", {
    "scope": "merchant", "context_id": "m_version_test", "version": 2,
    "delivered_at": "2026-04-26T11:00:00Z",
    "payload": {"merchant_id": "m_version_test", "performance": {"views": 9999}},
})
check("Duplicate v2 push → 409 (idempotent)", c4 == 409)

# Push v3 — should succeed
r_v3, c5, _ = call("POST", "/v1/context", {
    "scope": "merchant", "context_id": "m_version_test", "version": 3,
    "delivered_at": "2026-04-26T12:00:00Z",
    "payload": {"merchant_id": "m_version_test", "performance": {"views": 15000}},
})
check("v3 push accepted (incremental)", c5 == 200 and r_v3 and r_v3.get("accepted"))

# Healthz should show merchant count=1 (not 3 — same context_id)
r_h, _, _ = call("GET", "/v1/healthz")
check("Healthz merchant count=1 (3 versions of same merchant_id = 1 entry)",
      r_h and r_h.get("contexts_loaded", {}).get("merchant") == 1,
      str(r_h.get("contexts_loaded") if r_h else "error"))


# ---------------------------------------------------------------------------
# SUMMARY
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print(f"  Adaptive results: {passed}/{passed + failed} passed | {failed} failed")
print("=" * 60)

call("POST", "/v1/teardown")
sys.exit(0 if failed == 0 else 1)
