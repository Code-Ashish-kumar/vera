"""
run_simulator.py — Runs judge_simulator.py for all key scenarios.
Usage: py -3.13 run_simulator.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import judge_simulator as sim

sim.BOT_URL = "http://localhost:8080"
sim.DATASET_DIR = Path(__file__).parent / "dataset"


class MockProvider(sim.LLMProvider):
    """No-LLM provider — structural tests only (no scoring)."""
    def complete(self, prompt, system=None):
        return "{}"
    def name(self):
        return "Mock (no scoring)"


SCENARIOS = ["warmup", "auto_reply_hell", "intent_transition", "hostile"]

results = []
for scenario in SCENARIOS:
    print(f"\n{'='*70}")
    print(f"  Running scenario: {scenario}")
    print(f"{'='*70}")
    judge = sim.JudgeSimulator(MockProvider())
    ok = judge.run(scenario)
    results.append((scenario, ok))

print("\n" + "=" * 70)
print("  Judge Simulator Results")
print("=" * 70)
for scenario, ok in results:
    icon = "PASS" if ok else "FAIL"
    print(f"  [{icon}] {scenario}")

all_ok = all(ok for _, ok in results)
print(f"\n  Overall: {'ALL PASS' if all_ok else 'SOME FAILED'}")
sys.exit(0 if all_ok else 1)
