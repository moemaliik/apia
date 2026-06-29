#!/usr/bin/env python3
"""APIA CLI.

  python run.py "create an issue titled 'Login fails' labeled bug"
  python run.py --demo            # run the three DEMO.md instructions in order
  python run.py --rollback 12     # undo everything done by instruction id 12
  python run.py --reset           # wipe the memory DB (for a clean slate ONLY)
"""
from __future__ import annotations

import sys
from pathlib import Path

# Force UTF-8 so the report's ✓/✗ glyphs print on Windows (cp1252) consoles.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

sys.path.insert(0, str(Path(__file__).resolve().parent))

from apia import config
from apia.agent import Agent
from apia.rollback import rollback


DEMO = [
    "Create an issue titled 'Login times out after 30 seconds' and label it bug",
    "Find all open issues with no assignee and add the 'needs-triage' label to each",
    "Find all open unlabeled issues, group them by age (this week / this month / older), "
    "and create one triage summary issue with a checklist grouped by bucket",
]


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 0

    if argv[0] == "--reset":
        if config.DB_PATH.exists():
            config.DB_PATH.unlink()
        print(f"memory wiped: {config.DB_PATH}")
        return 0

    if argv[0] == "--rollback":
        for line in rollback(int(argv[1])):
            print(line)
        return 0

    agent = Agent()
    try:
        if argv[0] == "--demo":
            for i, instr in enumerate(DEMO, 1):
                print(f"\n########## DEMO {i}/3 ##########")
                print(agent.run(instr).pretty())
        else:
            print(agent.run(" ".join(argv)).pretty())
    finally:
        agent.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
