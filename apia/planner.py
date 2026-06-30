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

import re

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


class PlannerAgent:
    def __init__(self, llm: LLM, memory):
        self.llm = llm
        self.memory = memory

    @staticmethod
    def signature(instruction: str) -> str:
        """Normalised intent key: lowercase, strip numbers/quotes/punctuation and
        collapse whitespace so 'create issue X' and 'create issue Y' collide."""
        s = instruction.lower()
        s = re.sub(r"['\"`].*?['\"`]", " qstr ", s)
        s = re.sub(r"\d+", " qnum ", s)
        s = re.sub(r"[^a-z ]", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
        # keep the salient verbs/nouns only — first 12 tokens is plenty for intent
        return " ".join(s.split()[:12])

    def plan(self, instruction: str) -> Plan:
        sig = self.signature(instruction)
        reused = self.memory.find_reusable_plan(sig)
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
