#!/usr/bin/env python3
"""APIA test suite — runnable with `python tests/test_apia.py` (no pytest needed,
though `pytest tests/` also works). Forces mock + replay so it needs no keys and
spends nothing.
"""
import os
import sys
import tempfile
from pathlib import Path

os.environ["APIA_GITHUB_MODE"] = "mock"
os.environ["APIA_LLM_PROVIDER"] = "replay"

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from apia.planner import PlannerAgent
from apia.sandbox import compile_capability, SandboxError
from apia.agent import Agent
from apia.memory import Memory
from apia.platform import MockTransport
from run import DEMO


def _fresh_agent():
    db = Path(tempfile.mkdtemp()) / "test.db"
    mem = Memory(db)
    return Agent(memory=mem, verbose=False, transport=MockTransport()), mem


def test_signature_collapses_literals():
    a = PlannerAgent.signature("Create an issue titled 'Foo' and label it bug")
    b = PlannerAgent.signature('Create an issue titled "Bar baz" and label it bug')
    assert a == b, (a, b)
    # different intent must NOT collide
    c = PlannerAgent.signature("Delete all closed issues")
    assert c != a


def test_sandbox_rejects_dangerous_code():
    for src in [
        "def capability(ctx, **k):\n    import os\n    return 1",
        "def capability(ctx, **k):\n    return open('/etc/passwd').read()",
        "def capability(ctx, **k):\n    return ctx.__class__.__bases__",
        "def capability(ctx, **k):\n    return eval('1')",
    ]:
        try:
            compile_capability(src)
            assert False, f"sandbox accepted dangerous code: {src!r}"
        except SandboxError:
            pass


def test_sandbox_accepts_clean_code():
    fn = compile_capability(
        "def capability(ctx, **k):\n    return {'ok': sorted([3, 1, 2])}")
    assert fn(None) == {"ok": [1, 2, 3]}


def test_end_to_end_demo_runs_clean():
    agent, mem = _fresh_agent()
    for instr in DEMO:
        rep = agent.run(instr)
        assert rep.status == "success", (instr, rep.status, rep.failed)
    # the two synthesised filters/labelers should be registered
    names = {c["name"] for c in mem.list_capabilities()}
    assert {"filter_unassigned", "label_issues", "filter_unlabeled",
            "bucket_issues_by_age"} <= names, names
    agent.close()


def test_learning_curve_reduces_llm_calls():
    agent, mem = _fresh_agent()
    instr = DEMO[1]
    first = agent.run(instr).metrics
    for _ in range(3):
        last = agent.run(instr).metrics
    assert first["llm_calls"] > last["llm_calls"], (first, last)
    assert last["llm_calls"] == 0, last
    # plan was reused on later runs
    runs = mem.runs_for(PlannerAgent.signature(instr))
    assert runs[0]["plan_source"] == "fresh"
    assert runs[-1]["plan_source"] == "reused"
    agent.close()


def test_constraint_learned_then_prechecked():
    agent, mem = _fresh_agent()
    instr = DEMO[1]
    r1 = agent.run(instr)        # creates needs-triage
    r2 = agent.run(instr)        # 422 -> learns constraint
    assert any("create_label" in c for c in r2.constraints_learned), r2.constraints_learned
    cons = mem.constraints_for("create_label")
    assert cons and cons[0]["rule"].get("precheck") == "label_exists"
    agent.close()


def test_capability_promoted_to_trusted():
    agent, mem = _fresh_agent()
    for _ in range(3):
        agent.run(DEMO[1])
    cap = mem.get_capability("filter_unassigned")
    assert cap["status"] == "trusted", cap["status"]
    agent.close()


def _run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}: {e}")
        except Exception as e:
            print(f"  ERROR {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{passed}/{len(tests)} passed")
    return passed == len(tests)


if __name__ == "__main__":
    raise SystemExit(0 if _run_all() else 1)
