"""
generate_submission.py — Produces submission.jsonl from the 30 canonical test pairs.
=====================================================================================

Usage:
    py -3.13 generate_submission.py

Output:
    submission.jsonl  — 30 lines, one per test pair, in the root workspace folder.

Design rules (from execution plan Step 6):
  - Uses the SAME composer.py code path as the deployed bot (not a standalone script).
  - Loads .env before importing composer so GROQ_API_KEY is set identically.
  - Reads all 4 contexts from dataset/expanded/ — the same files the judge will use.
  - Calls composer.compose() with a real context_store dict, exactly as bot.py tick does.
  - Falls back to stub for any pair where composition fails, logs the failure, and
    continues — the file must always contain exactly 30 lines.
  - Validates each output before writing (no empty bodies, no URLs, valid CTA).
  - Saves incrementally to submission_progress.json so runs can resume after interruption.
  - Prints a per-pair summary and a final pass/fail count.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path

# ── Bootstrap: load .env before importing composer ────────────────────────────
try:
    from dotenv import load_dotenv
    _env = Path(__file__).parent / ".env"
    if _env.exists():
        load_dotenv(_env)
except ImportError:
    pass

# ── Workspace paths ──────────────────────────────────────────────────────────
ROOT      = Path(__file__).parent
EXPANDED  = ROOT / "dataset" / "expanded"
CATS_DIR  = EXPANDED / "categories"
MERCH_DIR = EXPANDED / "merchants"
CUST_DIR  = EXPANDED / "customers"
TRIG_DIR  = EXPANDED / "triggers"
PAIRS_FILE = EXPANDED / "test_pairs.json"
OUT_FILE      = ROOT / "submission.jsonl"
PROGRESS_FILE = ROOT / "submission_progress.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("generate_submission")

# ── Import composer (after .env is loaded) ───────────────────────────────────
from composer import LLMComposer, validate_output

COMPOSER = LLMComposer()

# ── Helpers ──────────────────────────────────────────────────────────────────

def load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_context_store(
    merchant_id: str,
    trigger_id: str,
    customer_id: str | None,
) -> dict:
    """
    Build a context_store dict identical in shape to bot.py's in-memory store.
    Keys are (scope, context_id) tuples; values are {"version": int, "payload": dict}.
    Always re-reads from disk — guarantees fresh data, same as the deployed bot.
    """
    store: dict = {}

    # Load merchant
    merch_path = MERCH_DIR / f"{merchant_id}.json"
    if not merch_path.exists():
        raise FileNotFoundError(f"Merchant file missing: {merch_path}")
    merchant = load_json(merch_path)
    store[("merchant", merchant_id)] = {"version": 1, "payload": merchant}

    # Load category from merchant's category_slug
    cat_slug = merchant.get("category_slug", "")
    cat_path = CATS_DIR / f"{cat_slug}.json"
    if not cat_path.exists():
        raise FileNotFoundError(f"Category file missing: {cat_path}")
    category = load_json(cat_path)
    store[("category", cat_slug)] = {"version": 1, "payload": category}

    # Load trigger
    trig_path = TRIG_DIR / f"{trigger_id}.json"
    if not trig_path.exists():
        raise FileNotFoundError(f"Trigger file missing: {trig_path}")
    trigger = load_json(trig_path)
    store[("trigger", trigger_id)] = {"version": 1, "payload": trigger}

    # Load customer (optional)
    if customer_id:
        cust_path = CUST_DIR / f"{customer_id}.json"
        if not cust_path.exists():
            raise FileNotFoundError(f"Customer file missing: {cust_path}")
        customer = load_json(cust_path)
        store[("customer", customer_id)] = {"version": 1, "payload": customer}

    return store


def stub_result(test_id: str, merchant_id: str, trigger_id: str,
                customer_id: str | None, reason: str) -> dict:
    """Structurally valid fallback for any pair where composition fails."""
    return {
        "test_id": test_id,
        "body": f"Hi, I have an update for you. Want to hear more? (stub: {reason[:60]})",
        "cta": "open_ended",
        "send_as": "merchant_on_behalf" if customer_id else "vera",
        "suppression_key": f"stub:{merchant_id}:{trigger_id}",
        "rationale": f"Fallback stub — composition failed: {reason[:120]}",
    }


# ── Main generation loop ──────────────────────────────────────────────────────

def main():
    log.info("Reading test pairs from %s", PAIRS_FILE)
    pairs = load_json(PAIRS_FILE)["pairs"]
    assert len(pairs) == 30, f"Expected 30 pairs, got {len(pairs)}"
    log.info("Found %d pairs", len(pairs))

    # Load any previous progress so we can resume without re-running completed pairs
    progress: dict[str, dict] = {}
    if PROGRESS_FILE.exists():
        progress = load_json(PROGRESS_FILE)
        log.info("Resuming: %d/%d pairs already done", len(progress), len(pairs))

    results: list[dict] = []
    n_ok = 0
    n_stub = 0
    n_warn = 0

    for i, pair in enumerate(pairs, 1):
        test_id     = pair["test_id"]
        merchant_id = pair["merchant_id"]
        trigger_id  = pair["trigger_id"]
        customer_id = pair.get("customer_id")

        # Use cached result if already done in a previous run
        if test_id in progress:
            row = progress[test_id]
            results.append(row)
            log.info("[%s] CACHED  cta=%-20s  %s", test_id, row["cta"], row["body"][:60])
            n_ok += 1
            continue

        # Pace: 3s between pairs to stay under Groq free-tier rate limits
        if i > 1:
            time.sleep(3)

        t0 = time.time()
        try:
            store = build_context_store(merchant_id, trigger_id, customer_id)

            # Use a stable conv_id derived from the test_id so it's reproducible
            conv_id = f"conv_submission_{test_id.lower()}"

            action = COMPOSER.compose(
                merchant_id  = merchant_id,
                trigger_id   = trigger_id,
                conv_id      = conv_id,
                customer_id  = customer_id,
                context_store = store,
            )

            elapsed = (time.time() - t0) * 1000

            if action is None:
                # Composer chose not to send (sparse context) — use stub
                log.warning("[%s] Composer returned None (sparse context) — using stub", test_id)
                row = stub_result(test_id, merchant_id, trigger_id, customer_id,
                                  "sparse context / no digest items")
                n_stub += 1
            else:
                # Build the submission row from the action dict
                row = {
                    "test_id":        test_id,
                    "body":           action["body"],
                    "cta":            action["cta"],
                    "send_as":        action["send_as"],
                    "suppression_key": action["suppression_key"],
                    "rationale":      action["rationale"],
                }

                # Validate — log warnings but don't block output
                trigger_payload = store[("trigger", trigger_id)]["payload"]
                trigger_kind    = trigger_payload.get("kind", "unknown")
                merchant_payload = store[("merchant", merchant_id)]["payload"]
                languages       = merchant_payload.get("identity", {}).get("languages", ["en"])
                category_payload = store[("category",
                                          merchant_payload.get("category_slug", ""))].get("payload", {})
                taboos = (
                    category_payload.get("voice", {}).get("taboos")
                    or category_payload.get("voice", {}).get("vocab_taboo")
                    or []
                )
                is_customer_facing = bool(customer_id)

                # Build context block text for fabrication whitelist
                from composer import _build_context_block
                customer_payload = (
                    store[("customer", customer_id)]["payload"] if customer_id else None
                )
                ctx_block = _build_context_block(
                    category_payload, merchant_payload, trigger_payload, customer_payload
                )

                valid, reason = validate_output(
                    {"body": row["body"], "cta": row["cta"], "send_as": row["send_as"]},
                    ctx_block, trigger_kind, languages,
                    customer_facing=is_customer_facing,
                    category_taboos=taboos,
                )
                if not valid:
                    log.warning("[%s] Validation warning: %s (keeping output)", test_id, reason)
                    n_warn += 1

                n_ok += 1

        except Exception as e:
            elapsed = (time.time() - t0) * 1000
            log.error("[%s] Composition error: %s — using stub", test_id, e)
            row = stub_result(test_id, merchant_id, trigger_id, customer_id, str(e))
            n_stub += 1

        results.append(row)

        # Save progress after every pair so we can resume if interrupted
        progress[test_id] = row
        with open(PROGRESS_FILE, "w", encoding="utf-8") as pf:
            json.dump(progress, pf, ensure_ascii=False, indent=2)

        # Progress log
        status = "STUB" if "stub:" in row.get("rationale","") else "OK"
        body_preview = row["body"][:70].replace("\n", " ")
        log.info("[%s] %s  (%dms)  cta=%-20s  %s",
                 test_id, status.upper(), elapsed, row["cta"], body_preview)

    # ── Write output ──────────────────────────────────────────────────────────
    assert len(results) == 30, f"Expected 30 results, got {len(results)}"

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    log.info("Wrote %d lines to %s", len(results), OUT_FILE)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"  Submission generation complete")
    print(f"  Composed:   {n_ok:2d} / 30")
    print(f"  Stubs:      {n_stub:2d} / 30  (sparse context or errors)")
    print(f"  Warnings:   {n_warn:2d} / 30  (validation issues, still included)")
    print(f"  Output:     {OUT_FILE}")
    print("=" * 60)

    # Verify output file
    with open(OUT_FILE, encoding="utf-8") as f:
        lines = f.readlines()

    print(f"\n  Verifying {OUT_FILE}...")
    errors = []
    for j, line in enumerate(lines, 1):
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as e:
            errors.append(f"Line {j}: invalid JSON — {e}")
            continue
        required = ["test_id", "body", "cta", "send_as", "suppression_key", "rationale"]
        missing = [k for k in required if not obj.get(k)]
        if missing:
            errors.append(f"Line {j} ({obj.get('test_id','?')}): missing fields {missing}")
        if "http" in obj.get("body", "").lower():
            errors.append(f"Line {j} ({obj.get('test_id','?')}): URL in body")
        if obj.get("cta") not in {"open_ended", "binary_yes_no",
                                   "binary_confirm_cancel", "multi_choice_slot", "none"}:
            errors.append(f"Line {j} ({obj.get('test_id','?')}): invalid cta '{obj.get('cta')}'")

    if errors:
        print("  ERRORS:")
        for err in errors:
            print(f"    [FAIL] {err}")
        sys.exit(1)
    else:
        print(f"  [PASS] All 30 lines valid JSON with required fields")
        print(f"  [PASS] No URLs in any body")
        print(f"  [PASS] All CTA values valid")


if __name__ == "__main__":
    main()
