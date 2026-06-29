"""ReflectionAgent — Reflexion-style self-critique after each run.

It looks at the plan, what failed and what was learned, and writes 0-3 short,
actionable lessons keyed to the instruction signature. The planner injects these
into future plans for the same signature, closing a second learning loop on top
of plan-reuse and constraint-learning.

When the LLM is unavailable (replay misses), it falls back to deterministic
lessons derived from the run facts so the loop still functions offline.
"""
from __future__ import annotations

from .llm import LLM

REFLECT_SYSTEM = """You are a critic for a GitHub automation agent. Given a run
summary, output JSON: a list of 0-3 short imperative lessons (<=140 chars each)
that would make the NEXT run of a similar instruction faster or more reliable.
Focus on ordering, prechecks, and avoiding known failures. Return ONLY JSON."""


class ReflectionAgent:
    def __init__(self, llm: LLM, memory):
        self.llm = llm
        self.memory = memory

    def reflect(self, instruction_id: int, signature: str, plan, report_facts: dict) -> list[str]:
        lessons: list[str] = []
        try:
            data = self.llm.complete_json(
                REFLECT_SYSTEM,
                f"Instruction: {plan.instruction}\nFacts: {report_facts}",
                cache_key=f"reflect::{signature}",
                max_tokens=400,
            )
            if isinstance(data, list):
                lessons = [str(x)[:140] for x in data][:3]
        except Exception:
            # any LLM/parse failure falls back to deterministic lessons
            lessons = self._fallback(report_facts)
        for l in lessons:
            self.memory.add_lesson(instruction_id, signature, "reflection", l)
        return lessons

    @staticmethod
    def _fallback(facts: dict) -> list[str]:
        out = []
        for c in facts.get("constraints_learned", []):
            out.append(f"Precheck before this op: {c}")
        if facts.get("synthesised"):
            out.append("Reuse synthesised capability " +
                       ", ".join(facts["synthesised"]) + " instead of re-deriving.")
        return out[:3]
