#!/usr/bin/env python3
"""Reproducible learning-curve demo.

Runs ONE instruction N times inside a single process against ONE shared mock
GitHub repo and ONE memory DB, so the agent's improvement is measurable:

    APIA_LLM_PROVIDER=replay python scripts/replay_demo.py 2

This forces mock + replay regardless of your .env, needs no API keys, and
prints a per-run table plus the run-1-vs-last delta. Pick the instruction with
the integer arg (1, 2 or 3 — the DEMO.md instructions). Default: 2.
"""
import os
import sys
import tempfile
from pathlib import Path

os.environ["APIA_GITHUB_MODE"] = "mock"
os.environ["APIA_LLM_PROVIDER"] = "replay"

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from apia.agent import Agent
from apia.memory import Memory
from apia.platform import MockTransport
from run import DEMO

RUNS = 5


def main():
    which = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    instruction = DEMO[which - 1]

    db = Path(tempfile.gettempdir()) / "apia_replay_demo.db"
    if db.exists():
        db.unlink()
    memory = Memory(db)
    transport = MockTransport()                      # one shared repo across all runs
    agent = Agent(memory=memory, verbose=False, transport=transport)

    print(f"\nInstruction: {instruction}\n")
    rows = []
    for i in range(1, RUNS + 1):
        r = agent.run(instruction)
        m = r.metrics
        plan_src = "reused" if i > 1 and not r.failed else ("fresh" if i == 1 else "reused")
        rows.append((i, m["llm_calls"], m["api_calls"], m["failures"],
                     m["wall_ms"], r.status, ", ".join(r.constraints_learned) or "-"))

    hdr = f"{'run':>3} {'llm':>4} {'api':>4} {'fail':>5} {'ms':>6}  status   constraint-learned"
    print(hdr)
    print("-" * len(hdr))
    for (i, llm, api, fail, ms, status, con) in rows:
        print(f"{i:>3} {llm:>4} {api:>4} {fail:>5} {ms:>6}  {status:<8} {con}")

    first, last = rows[0], rows[-1]
    print("\nLearning delta (run 1 → run %d):" % RUNS)
    print(f"  LLM calls : {first[1]:>3}  →  {last[1]:<3}")
    print(f"  API calls : {first[2]:>3}  →  {last[2]:<3}")
    print(f"  failures  : {first[3]:>3}  →  {last[3]:<3}")
    print(f"  wall (ms) : {first[4]:>3}  →  {last[4]:<3}")
    agent.close()


if __name__ == "__main__":
    main()
