#!/usr/bin/env python3
"""Learning dashboard — renders, from the persistent memory DB, the numbers that
prove the agent improves: the capability library (with status + success rate)
and, per instruction signature, the run-1-vs-latest deltas.

    python dashboard.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Force UTF-8 so the '→' in the delta line prints on Windows (cp1252) consoles.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

sys.path.insert(0, str(Path(__file__).resolve().parent))

from rich.console import Console
from rich.table import Table
from rich import box

from apia.memory import Memory

console = Console()


def main():
    mem = Memory()

    caps = mem.list_capabilities()
    ct = Table(title="Capability library", box=box.SIMPLE_HEAVY, header_style="bold")
    for c in ("name", "status", "uses", "success rate", "synthesised for"):
        ct.add_column(c)
    if not caps:
        ct.add_row("—", "—", "—", "—", "no synthesised capabilities yet")
    for c in caps:
        total = c["success_count"] + c["failure_count"]
        rate = f"{(c['success_count']/total*100):.0f}%" if total else "—"
        spec = (json.loads(c["provenance_json"] or "{}").get("synthesised_for") or "")[:40]
        status_color = {"trusted": "green", "experimental": "yellow",
                        "deprecated": "red"}.get(c["status"], "white")
        ct.add_row(c["name"], f"[{status_color}]{c['status']}[/]",
                   str(total), rate, spec)
    console.print(ct)

    for sig in mem.signatures():
        runs = mem.runs_for(sig)
        if not runs:
            continue
        t = Table(title=f"Learning curve — '{sig}'", box=box.SIMPLE_HEAVY, header_style="bold")
        for c in ("run", "plan", "llm calls", "api calls", "failures", "wall ms"):
            t.add_column(c, justify="right")
        for r in runs:
            t.add_row(str(r["run_number"]), r["plan_source"], str(r["llm_calls"]),
                      str(r["api_calls"]), str(r["failures"]), str(r["wall_ms"]))
        console.print(t)
        first, last = runs[0], runs[-1]
        if len(runs) > 1:
            console.print(
                f"   [bold]delta[/] run1→run{last['run_number']}:  "
                f"llm {first['llm_calls']}→{last['llm_calls']}   "
                f"api {first['api_calls']}→{last['api_calls']}   "
                f"failures {first['failures']}→{last['failures']}   "
                f"wall {first['wall_ms']}→{last['wall_ms']}ms\n")
    mem.close()


if __name__ == "__main__":
    main()
