"""
Quick warmup harness test — no LLM needed.
Run: py -3.13 run_warmup.py
"""
import sys
import json
from pathlib import Path
from urllib import request as urlrequest, error as urlerror

BOT_URL = "http://localhost:8080"
DATASET_DIR = Path(__file__).parent / "dataset"

def call(method, path, body=None, timeout=10):
    url = f"{BOT_URL}{path}"
    data = json.dumps(body).encode() if body else None
    req = urlrequest.Request(url, data=data, method=method,
                             headers={"Content-Type": "application/json"})
    try:
        resp = urlrequest.urlopen(req, timeout=timeout)
        return json.loads(resp.read()), resp.status, None
    except urlerror.HTTPError as e:
        try:
            return json.loads(e.read()), e.code, None
        except:
            return None, e.code, str(e)
    except Exception as ex:
        return None, None, str(ex)

def check(label, condition, detail=""):
    icon = "PASS" if condition else "FAIL"
    print(f"  [{icon}] {label}" + (f" — {detail}" if detail else ""))
    return condition

passed = 0
failed = 0

# ── Teardown first ────────────────────────────────────────────────────────────
print("\n── Teardown (clean slate) ──")
r, code, err = call("POST", "/v1/teardown")
ok = check("POST /v1/teardown → 200", code == 200)
passed += ok; failed += (not ok)

# ── Healthz (empty) ───────────────────────────────────────────────────────────
print("\n── Healthz (pre-warmup) ──")
r, code, err = call("GET", "/v1/healthz")
ok = check("GET /v1/healthz → 200", code == 200)
ok2 = check("status=ok", r and r.get("status") == "ok")
ok3 = check("all counts zero", r and all(v == 0 for v in r["contexts_loaded"].values()),
            str(r.get("contexts_loaded") if r else err))
for o in [ok, ok2, ok3]:
    passed += o; failed += (not o)

# ── Metadata ──────────────────────────────────────────────────────────────────
print("\n── Metadata ──")
r, code, err = call("GET", "/v1/metadata")
ok = check("GET /v1/metadata → 200", code == 200)
ok2 = check("has team_name", r and "team_name" in r)
ok3 = check("has model", r and "model" in r)
ok4 = check("has approach", r and "approach" in r)
for o in [ok, ok2, ok3, ok4]:
    passed += o; failed += (not o)

# ── Context pushes ────────────────────────────────────────────────────────────
print("\n── Context: category pushes ──")
cat_dir = DATASET_DIR / "categories"
cat_count = 0
for f in sorted(cat_dir.glob("*.json")):
    payload = json.loads(f.read_text(encoding="utf-8"))
    slug = payload.get("slug", f.stem)
    r, code, err = call("POST", "/v1/context", {
        "scope": "category", "context_id": slug, "version": 1,
        "delivered_at": "2026-04-26T09:45:00Z", "payload": payload
    })
    ok = check(f"category/{slug}", code == 200 and r and r.get("accepted"),
               r.get("reason", err) if r else err)
    passed += ok; failed += (not ok)
    cat_count += ok

print("\n── Context: merchant pushes (seed) ──")
seeds = json.loads((DATASET_DIR / "merchants_seed.json").read_text())["merchants"]
merch_count = 0
for m in seeds[:5]:
    mid = m["merchant_id"]
    r, code, err = call("POST", "/v1/context", {
        "scope": "merchant", "context_id": mid, "version": 1,
        "delivered_at": "2026-04-26T09:45:30Z", "payload": m
    })
    ok = check(f"merchant/{mid[:20]}", code == 200 and r and r.get("accepted"),
               r.get("reason", err) if r else err)
    passed += ok; failed += (not ok)
    merch_count += ok

print("\n── Idempotency: same version → 409 ──")
m0 = seeds[0]
r, code, err = call("POST", "/v1/context", {
    "scope": "merchant", "context_id": m0["merchant_id"], "version": 1,
    "delivered_at": "2026-04-26T09:45:30Z", "payload": m0
})
ok = check("same version → HTTP 409", code == 409)
ok2 = check("reason=stale_version", r and r.get("reason") == "stale_version")
ok3 = check("current_version=1 present", r and r.get("current_version") == 1)
for o in [ok, ok2, ok3]:
    passed += o; failed += (not o)

print("\n── Version bump → 200 ──")
m0_v2 = dict(m0); m0_v2["performance"] = {"views": 9999}
r, code, err = call("POST", "/v1/context", {
    "scope": "merchant", "context_id": m0["merchant_id"], "version": 2,
    "delivered_at": "2026-04-26T10:30:00Z", "payload": m0_v2
})
ok = check("version 2 accepted", code == 200 and r and r.get("accepted"))
passed += ok; failed += (not ok)

print("\n── Invalid scope → 400 ──")
r, code, err = call("POST", "/v1/context", {
    "scope": "invalid_scope", "context_id": "x", "version": 1,
    "delivered_at": "2026-04-26T09:45:00Z", "payload": {}
})
ok = check("invalid scope → HTTP 400", code == 400)
ok2 = check("reason=invalid_scope", r and r.get("reason") == "invalid_scope")
for o in [ok, ok2]:
    passed += o; failed += (not o)

print("\n── Tick: push a trigger, get an action ──")
tseeds = json.loads((DATASET_DIR / "triggers_seed.json").read_text())["triggers"]
# Find a trigger for one of the 5 merchants we already pushed
pushed_mids = {m["merchant_id"] for m in seeds[:5]}
t0 = next((t for t in tseeds if t.get("merchant_id") in pushed_mids), tseeds[0])
r, code, err = call("POST", "/v1/context", {
    "scope": "trigger", "context_id": t0["id"], "version": 1,
    "delivered_at": "2026-04-26T10:00:00Z", "payload": t0
})
ok = check(f"trigger/{t0['id'][:25]} pushed", code == 200 and r and r.get("accepted"))
passed += ok; failed += (not ok)

r, code, err = call("POST", "/v1/tick", {
    "now": "2026-04-26T10:35:00Z", "available_triggers": [t0["id"]]
})
ok = check("POST /v1/tick → 200", code == 200)
ok2 = check("response has 'actions' key", r and "actions" in r)
actions = r.get("actions", []) if r else []
ok3 = check("at least 1 action returned", len(actions) >= 1,
            f"{len(actions)} actions")
passed += ok; failed += (not ok)
passed += ok2; failed += (not ok2)
passed += ok3; failed += (not ok3)

if actions:
    a = actions[0]
    required = ["conversation_id", "merchant_id", "send_as", "trigger_id",
                "template_name", "template_params", "body", "cta",
                "suppression_key", "rationale"]
    missing = [f for f in required if f not in a]
    ok = check("action has all required fields", not missing,
               f"missing: {missing}" if missing else "")
    ok2 = check("body is non-empty", bool(a.get("body", "").strip()))
    ok3 = check("no URL in body", "http" not in a.get("body", "").lower())
    conv_id = a["conversation_id"]
    for o in [ok, ok2, ok3]:
        passed += o; failed += (not o)

    print("\n── Reply: accept intent ──")
    r, code, err = call("POST", "/v1/reply", {
        "conversation_id": conv_id,
        "merchant_id": a["merchant_id"],
        "from_role": "merchant",
        "message": "Yes, let's do it",
        "received_at": "2026-04-26T10:42:00Z",
        "turn_number": 2
    })
    ok = check("POST /v1/reply → 200", code == 200)
    ok2 = check("action field present", r and "action" in r)
    ok3 = check("action=send (accept intent)", r and r.get("action") == "send")
    ok4 = check("body non-empty on send", r and bool(r.get("body", "").strip()))
    for o in [ok, ok2, ok3, ok4]:
        passed += o; failed += (not o)

    print("\n── Reply: hostile intent → end ──")
    # Need a new conversation since this one just sent
    r2, code2, _ = call("POST", "/v1/reply", {
        "conversation_id": conv_id,
        "merchant_id": a["merchant_id"],
        "from_role": "merchant",
        "message": "Stop messaging me. This is spam.",
        "received_at": "2026-04-26T10:45:00Z",
        "turn_number": 3
    })
    ok = check("hostile reply → action=end", r2 and r2.get("action") == "end",
               r2.get("action") if r2 else "error")
    passed += ok; failed += (not ok)

print("\n── Healthz after load ──")
r, code, err = call("GET", "/v1/healthz")
ok = check("category count ≥ 5", r and r["contexts_loaded"]["category"] >= 5,
           str(r["contexts_loaded"]["category"] if r else err))
ok2 = check("merchant count ≥ 1", r and r["contexts_loaded"]["merchant"] >= 1)
ok3 = check("trigger count ≥ 1", r and r["contexts_loaded"]["trigger"] >= 1)
for o in [ok, ok2, ok3]:
    passed += o; failed += (not o)

print("\n── Teardown: wipe state ──")
r, code, err = call("POST", "/v1/teardown")
ok = check("POST /v1/teardown → 200", code == 200)
ok2 = check("wiped=true", r and r.get("wiped") is True)
passed += ok; failed += (not ok)
passed += ok2; failed += (not ok2)
r, code, err = call("GET", "/v1/healthz")
ok3 = check("all counts zero after teardown",
            r and all(v == 0 for v in r["contexts_loaded"].values()),
            str(r.get("contexts_loaded") if r else err))
passed += ok3; failed += (not ok3)

# ── Summary ───────────────────────────────────────────────────────────────────
total = passed + failed
print(f"\n{'='*50}")
print(f"  Results: {passed}/{total} passed | {failed} failed")
print(f"{'='*50}")
sys.exit(0 if failed == 0 else 1)
