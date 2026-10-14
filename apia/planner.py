"""PlannerAgent — decomposes a natural-language instruction into a step DAG.

Memory makes this cheaper over time:
  - It computes a normalised *signature* for the instruction.
  - If a previously SUCCESSFUL plan exists for that signature, it reuses the
    decomposition verbatim (0 planning LLM calls) — this is the measurable
    'decomposition quality / fewer LLM calls' learning signal.
  - Otherwise it asks the LLM, and injects any reflection lessons learned on
    past runs of this signature so the new plan avoids known pitfalls.
"""
from __future__ import annotations

import json
import re
from typing import Optional

from .llm import LLM
from .platform.base_capabilities import BUILTIN_SIGNATURES
from .schemas import Plan, Step

PLAN_SYSTEM = """You are the planner for a GitHub automation agent.
Decompose the instruction into ordered steps. Output JSON: a list of steps.
Each step: {"description": str, "capability": str, "args": {...}, "depends_on": [int]}.

`capability` is either a BUILT-IN or a NEW capability you name (snake_case) that
the agent will synthesise at runtime. Prefer built-ins for atomic CRUD; invent a
new capability ONLY for compound logic (filtering by a predicate, grouping/
bucketing, batch operations over many issues, rendering a summary, computing age).

Steps are numbered by their 0-based position in the array: the first step is 0,
the second is 1, and so on. Use that index in `depends_on` and in `$step`.

To pass data between steps, set an arg to {"$step": N} for the whole result of
step N, or {"$step": N, "get": "key"} for one key of it. NEVER hardcode an id
(issue number, etc.) that an earlier step produces — always thread it via $step.

Example — "create an issue titled 'X' and label it bug":
[
  {"description": "Create the issue", "capability": "create_issue",
   "args": {"title": "X", "labels": ["bug"]}, "depends_on": []}
]
create_issue accepts labels directly, so one step suffices. If you instead label
in a separate step, thread the number:
  {"capability": "add_labels_to_issue",
   "args": {"number": {"$step": 0, "get": "number"}, "labels": ["bug"]},
   "depends_on": [0]}

FAN-OUT (acting on MANY issues, e.g. "add label L to EACH/EVERY issue with no
assignee"): a step that produces a LIST cannot be threaded into a single-issue
capability. NEVER do {"$step": N, "get": "number"} when step N returns a list —
that yields one bogus id and 404s. Use a batch capability that takes the whole
list. Example:
[
  {"description": "List open issues", "capability": "list_open_issues",
   "args": {}, "depends_on": []},
  {"description": "Filter issues with no assignee", "capability": "filter_unassigned",
   "args": {"issues": {"$step": 0, "get": "issues"}}, "depends_on": [0]},
  {"description": "Ensure label L exists", "capability": "create_label",
   "args": {"name": "L"}, "depends_on": []},
  {"description": "Add L to each filtered issue", "capability": "batch_add_labels",
   "args": {"issues": {"$step": 1, "get": "issues"}, "label": "L"},
   "depends_on": [1, 2]}
]

BUILT-INS:
%s

Return ONLY the JSON array.""" % "\n".join(
    f"  {k}{v}" for k, v in BUILTIN_SIGNATURES.items())


# a quoted value in an instruction; group 1 is the value without its quotes
_LITERAL = r"['\"`](.*?)['\"`]"
# quoted values shorter than this are only replaced when they are a whole arg,
# never as a substring (avoids rewriting e.g. 'x' inside unrelated text)
_MIN_SUBSTR = 3


class PlannerAgent:
    def __init__(self, llm: LLM, memory):
        self.llm = llm
        self.memory = memory

    @staticmethod
    def signature(instruction: str) -> str:
        """Normalised intent key: lowercase, strip numbers/quotes/punctuation and
        collapse whitespace so 'create issue X' and 'create issue Y' collide.

        The WHOLE instruction is kept. An earlier version truncated to the first
        12 tokens, which made instructions that only differ at the end (e.g.
        '...group by age and create a summary' vs '...group by age and close
        every one') share a signature and therefore a reused plan."""
        s = instruction.lower()
        s = re.sub(_LITERAL, " qstr ", s)
        s = re.sub(r"\d+", " qnum ", s)
        s = re.sub(r"[^a-z ]", " ", s)
        return re.sub(r"\s+", " ", s).strip()

    # ---- plan parameters ------------------------------------------------
    # The signature erases literals ('Login fails' -> qstr) so that plans can be
    # reused across instructions that differ only in their values. That means a
    # saved plan must not carry the OLD values either: they are lifted out into
    # numbered parameters when the plan is saved and re-bound from the NEW
    # instruction when it is reused.

    @staticmethod
    def literals(instruction: str) -> list[str]:
        """Quoted strings and bare numbers, in the order the signature erases them."""
        out = []
        for m in re.finditer(_LITERAL + r"|(\d+)", instruction):
            out.append(m.group(1) if m.group(1) is not None else m.group(2))
        return out

    @staticmethod
    def parameterise(steps: list[dict], literals: list[str]) -> dict:
        """Replace values that came from the instruction with {"$param": i}
        (whole value) or {"$tmpl": "...{{$i}}..."} (value embedded in a string)."""
        quoted = sorted(((i, l) for i, l in enumerate(literals)
                         if not l.isdigit() and len(l) >= _MIN_SUBSTR),
                        key=lambda p: -len(p[1]))

        def walk(v):
            if isinstance(v, dict):
                return {k: walk(x) for k, x in v.items()}
            if isinstance(v, list):
                return [walk(x) for x in v]
            if isinstance(v, bool):
                return v
            if isinstance(v, int):
                for i, l in enumerate(literals):
                    if l.isdigit() and int(l) == v:
                        return {"$param": i, "type": "int"}
                return v
            if isinstance(v, str):
                for i, l in enumerate(literals):
                    if v == l:
                        return {"$param": i}
                t = v
                for i, l in quoted:
                    t = t.replace(l, "{{$%d}}" % i)
                return {"$tmpl": t} if t != v else v
            return v

        return {"literals": list(literals),
                "steps": [{**s, "args": walk(s.get("args", {})),
                           "description": walk(s.get("description", ""))}
                          for s in steps]}

    @staticmethod
    def bind(saved, literals: list[str]) -> Optional[list[dict]]:
        """Fill a saved plan's parameters from the new instruction's literals.
        Returns None when the plan can't be reused SAFELY, so the caller replans:
          - a legacy (un-parameterised) plan and the instruction has literals,
          - the literal count differs,
          - an old value would survive into the new plan's args."""
        if isinstance(saved, list):                      # pre-parameterisation format
            return saved if not literals else None
        old, steps = saved.get("literals", []), saved.get("steps", [])
        if len(old) != len(literals):
            return None

        def fill(v):
            if isinstance(v, dict):
                if "$param" in v:
                    x = literals[v["$param"]]
                    return int(x) if v.get("type") == "int" else x
                if "$tmpl" in v:
                    return re.sub(r"\{\{\$(\d+)\}\}",
                                  lambda m: literals[int(m.group(1))], v["$tmpl"])
                return {k: fill(x) for k, x in v.items()}
            if isinstance(v, list):
                return [fill(x) for x in v]
            return v

        bound = [{**s, "args": fill(s.get("args", {})),
                  "description": fill(s.get("description", ""))} for s in steps]
        stale = {l for l in old if len(l) >= _MIN_SUBSTR and not l.isdigit()} - set(literals)
        blob = json.dumps([s["args"] for s in bound])
        if any(json.dumps(l)[1:-1] in blob for l in stale):
            return None
        return bound

    def plan(self, instruction: str) -> Plan:
        sig = self.signature(instruction)
        saved = self.memory.find_reusable_plan(sig)
        reused = self.bind(saved, self.literals(instruction)) if saved else None
        if reused:
            steps = [Step(**{k: v for k, v in s.items()
                            if k in Step.__dataclass_fields__}) for s in reused]
            # reset runtime fields
            for st in steps:
                st.status, st.error, st.attempts = "pending", None, 0
                st.api_calls, st.wall_ms, st.result = 0, 0, None
            return Plan(instruction=instruction, signature=sig, steps=steps, source="reused")

        lessons = self.memory.lessons_for(sig)
        user = f"Instruction: {instruction}"
        if lessons:
            user += "\n\nLessons from past runs of similar instructions:\n" + \
                    "\n".join(f"- {l}" for l in lessons)
        data = self.llm.complete_json(PLAN_SYSTEM, user, cache_key=f"plan::{sig}")
        steps = []
        for i, raw in enumerate(data):
            steps.append(Step(
                idx=i,
                description=raw.get("description", raw.get("capability", f"step {i}")),
                capability=raw["capability"],
                args=raw.get("args", {}),
                depends_on=raw.get("depends_on", []),
            ))
        return Plan(instruction=instruction, signature=sig, steps=steps, source="fresh")
