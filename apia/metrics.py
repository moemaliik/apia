"""A Meter is created per instruction run and threaded into the GitHub and LLM
clients. Every real API call and every LLM call increments it, so the numbers in
the execution report and learning dashboard are measured, not estimated.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class Meter:
    api_calls: int = 0
    llm_calls: int = 0
    tokens: int = 0
    failures: int = 0
    _t0: float = field(default_factory=time.time)

    def api(self, n: int = 1) -> None:
        self.api_calls += n

    def llm(self, tokens: int = 0) -> None:
        self.llm_calls += 1
        self.tokens += tokens

    def fail(self, n: int = 1) -> None:
        self.failures += n

    @property
    def wall_ms(self) -> int:
        return int((time.time() - self._t0) * 1000)

    def snapshot(self) -> dict:
        return {
            "api_calls": self.api_calls,
            "llm_calls": self.llm_calls,
            "tokens": self.tokens,
            "failures": self.failures,
            "wall_ms": self.wall_ms,
        }
