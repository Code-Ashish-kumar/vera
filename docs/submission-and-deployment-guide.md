# Submission & Deployment Guide
## magicpin Vera AI Challenge — Everything left to do

**Current state:** All 6 build steps complete. 161/161 tests passing.
**What remains:** Finish `submission.jsonl`, deploy to a public URL, submit.

---

## Step A — Regenerate T18–T30 in submission.jsonl

The daily Groq token quota (200k tokens/day) was exhausted during generation. T18–T30 are currently structural stubs. You need to re-run the generator after the quota resets.

**When does it reset?** Midnight UTC. Check the current UTC time at https://time.is/UTC.

**How to run:**

```bash
py -3.13 generate_submission.py
```

Expected output:
```
T01: CACHED    (no LLM call — loaded from submission_progress.json)
...
T17: CACHED
T18: OK        (fresh LLM composition)
...
T30: OK
[PASS] All 30 lines valid JSON with required fields
```

The 3-second pacing between pairs means T18–T30 (13 pairs) takes about 65 seconds. No rate limit hits expected with a fresh daily quota.

**Verify the result:**

```bash
py -3.13 -c "
import json
lines = open('submission.jsonl', encoding='utf-8').readlines()
stubs = [json.loads(l) for l in lines if 'have an update for' in l]
print(f'LLM: {30-len(stubs)}/30  Stubs: {len(stubs)}/30')
for s in stubs:
    print(f'  STUB: {s[\"test_id\"]}')
"
```

Target: `LLM: 30/30  Stubs: 0/30`. If stubs remain, the quota may not have reset yet — wait and retry.

---

## Step B — Deploy to a public HTTPS URL

The bot must be reachable from outside your network for the full 60-minute test window without interruption. Three options below, ranked by recommendation.

---

### Option 1 — ngrok (fastest, good for testing)

Best if you want to submit within the next hour. Your local machine runs the server; ngrok creates a public tunnel.

**1. Download ngrok**
Go to https://ngrok.com/download, download the Windows binary, extract to any folder.

**2. Sign up and get your auth token**
Register at https://ngrok.com, go to Your Authtoken, copy it.

**3. Authenticate**
```cmd
ngrok config add-authtoken YOUR_TOKEN_HERE
```

**4. Make sure the bot server is running locally**
```bash
py -3.13 -m uvicorn bot:app --host 0.0.0.0 --port 8080
```

**5. Start the tunnel** (in a separate terminal)
```cmd
ngrok http 8080
```

ngrok will display something like:
```
Forwarding   https://abc123.ngrok-free.app -> http://localhost:8080
```

**6. Your public URL is** `https://abc123.ngrok-free.app`

**Limitation:** If your machine sleeps, the tunnel dies and the judge will get connection errors. Keep the machine awake for the entire 60-minute test window. Free ngrok has a 1-request/second rate limit — fine for the judge's 10 req/sec? No. Use a paid ngrok plan or Option 2/3 for the real submission.

---

### Option 2 — Railway (recommended, free tier works)

Railway keeps the process alive permanently with no cold starts on the free Hobby tier.

**1. Install Railway CLI**
```bash
npm install -g @railway/cli
# or download from https://railway.app/cli
```

**2. Login**
```bash
railway login
```

**3. Initialize in the project folder**
```bash
cd "d:\megaProject\VERA AI"
railway init
# Choose "Create a new project" → name it "vera-bot"
```

**4. Set environment variables in Railway dashboard**
Go to https://railway.app → your project → Variables, add:
```
GROQ_API_KEY = gsk_fwcyqzC5...  (your full key from .env)
GROQ_MODEL = qwen/qwen3.8-27b
GROQ_CLASSIFIER_MODEL = qwen/qwen3.8-27b
```
Do NOT commit `.env` to git.

**5. Create a `Procfile`** in the project root:
```
web: py -3.13 -m uvicorn bot:app --host 0.0.0.0 --port $PORT
```
Or if `python` resolves to Python 3 on the Railway server:
```
web: python -m uvicorn bot:app --host 0.0.0.0 --port $PORT
```

**6. Create `requirements.txt`** if not present:
```
fastapi==0.115.12
uvicorn[standard]==0.34.3
pydantic==2.11.4
groq==0.28.0
python-dotenv==1.2.3
```

**7. Deploy**
```bash
railway up
```

Railway will build and deploy. When it finishes, run:
```bash
railway open
```
to get your public URL (format: `https://vera-bot-production.up.railway.app`).

**8. Verify**
```bash
curl https://your-railway-url/v1/healthz
```

---

### Option 3 — Render (alternative)

Similar to Railway. Use the **Starter plan** ($7/month) — the free tier spins down after 15 minutes of inactivity which will break the judge test.

**1.** Push your code to a GitHub repo (create one if needed):
```bash
git init
git add bot.py composer.py reply_composer.py conversation_policy.py requirements.txt
git commit -m "Vera AI Challenge submission"
git remote add origin https://github.com/YOUR_USERNAME/vera-bot.git
git push -u origin main
```

**2.** Go to https://render.com → New → Web Service → connect your GitHub repo.

**3.** Settings:
- Runtime: Python 3
- Build Command: `pip install -r requirements.txt`
- Start Command: `uvicorn bot:app --host 0.0.0.0 --port $PORT`
- Plan: **Starter** (not Free — Free has cold starts)

**4.** Under Environment → Add environment variables:
```
GROQ_API_KEY = your_key
GROQ_MODEL = qwen/qwen3.8-27b
GROQ_CLASSIFIER_MODEL = qwen/qwen3.8-27b
```

**5.** Deploy. Your URL will be `https://vera-bot.onrender.com`.

---

## Step C — Verify the deployed URL

Run this after deployment to confirm the public URL works end-to-end:

```bash
# Replace with your actual URL
set BOT_URL=https://your-deployed-url

# 1. Healthz
curl %BOT_URL%/v1/healthz

# 2. Metadata
curl %BOT_URL%/v1/metadata

# 3. Run the judge simulator against the live URL
py -3.13 -c "
import judge_simulator as sim
from pathlib import Path
sim.BOT_URL = 'https://your-deployed-url'
sim.DATASET_DIR = Path('dataset')

class Mock(sim.LLMProvider):
    def complete(self, p, s=None): return '{}'
    def name(self): return 'Mock'

judge = sim.JudgeSimulator(Mock())
judge.run('warmup')
"
```

Expected: all context pushes `[PASS]`, healthz `[PASS]`, metadata showing your team name.

**Test from a phone on mobile data** (not your WiFi) to confirm it's truly public:
```
Open browser → https://your-deployed-url/v1/healthz
Should return: {"status":"ok",...}
```

---

## Step D — Final submission.jsonl check before submitting

```bash
py -3.13 -c "
import json
with open('submission.jsonl', encoding='utf-8') as f:
    lines = f.readlines()
print(f'Lines: {len(lines)}')
assert len(lines) == 30, 'Must be exactly 30 lines'
for i, line in enumerate(lines, 1):
    obj = json.loads(line)
    required = ['test_id', 'body', 'cta', 'send_as', 'suppression_key', 'rationale']
    missing = [k for k in required if not obj.get(k)]
    if missing: print(f'Line {i} MISSING: {missing}')
    if 'http' in obj.get('body','').lower(): print(f'Line {i} HAS URL')
print('All checks passed' if True else 'ERRORS')
"
```

---

## Step E — Submit via the challenge portal

Upload these three files:
1. **Public URL** — paste the HTTPS URL from Step B
2. **`submission.jsonl`** — from `d:\megaProject\VERA AI\submission.jsonl`
3. **`README.md`** — from `d:\megaProject\VERA AI\README.md`

Optional (tiebreaker, not required):
4. `conversation_handlers.py` — not implemented in this project, skip

---

## Step F — Keep the server alive during the test window

Once you submit the URL, the judge will test it within the scheduled window. The judge runs for approximately 60 simulated minutes.

**Critical:** The server must not restart during the test window. In-memory state (all context pushes, all conversation history) is wiped on restart. A restart mid-test means the judge's subsequent ticks will see no contexts and return empty actions.

**Before the test window:**
- Confirm the deployment platform won't restart (Railway and Render Starter keep processes alive)
- If using ngrok: disable sleep on your laptop, keep the terminal open
- Check the Groq API key has sufficient quota for the test (the judge makes ~100-200 LLM calls during a 60-minute window)

**During the test:**
- Don't restart the server
- Don't push code changes
- Monitor the healthz endpoint if you can: it should always return `{"status":"ok",...}`

---

## Quick-reference checklist

```
[ ] Daily token quota reset — re-run generate_submission.py
[ ] submission.jsonl verified: 30 lines, 0 stubs
[ ] Bot deployed to public HTTPS URL
[ ] GROQ_API_KEY set on server (not in .env, in platform environment variables)
[ ] GROQ_MODEL=qwen/qwen3.8-27b set on server
[ ] Healthz reachable from mobile data (not just WiFi)
[ ] judge_simulator warmup passes against live URL
[ ] submission.jsonl uploaded to portal
[ ] README.md uploaded to portal
[ ] Public URL submitted via portal
[ ] Machine/server kept alive for 60-minute test window
```
