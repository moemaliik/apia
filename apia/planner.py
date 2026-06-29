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

To pass data between steps, set an arg to {"$step": N} for the whole result of
step N, or {"$step": N, "get": "key"} for one key of it.

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
