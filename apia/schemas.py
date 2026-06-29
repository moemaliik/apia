"""Typed structures passed between the planner, executor and report layer.

Plain dataclasses (stdlib) instead of pydantic — one less dependency to defend,
and we don't need runtime coercion of external input here.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from typing import Any, Optional


@dataclass
class Step:
    idx: int
    description: str
    capability: str                       # name of a built-in OR a to-be-synthesised capability
    args: dict[str, Any] = field(default_factory=dict)
    depends_on: list[int] = field(default_factory=list)
    # runtime
    status: str = "pending"               # pending|done|failed|blocked|skipped
    error: Optional[str] = None
    attempts: int = 0
    api_calls: int = 0
    wall_ms: int = 0
    result: Any = None
    synthesised: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Plan:
    instruction: str
    signature: str
    steps: list[Step]
    source: str = "fresh"                  # fresh|reused

    def to_dict(self) -> dict:
        return {
            "instruction": self.instruction,
            "signature": self.signature,
            "source": self.source,
            "steps": [s.to_dict() for s in self.steps],
        }


@dataclass
class ExecutionReport:
    instruction: str
    instruction_id: int
    status: str                            # success|partial|failed
    confidence: float
    done: list[str] = field(default_factory=list)
    failed: list[dict] = field(default_factory=list)
    blocked: list[str] = field(default_factory=list)
    decisions: list[str] = field(default_factory=list)
    synthesised: list[str] = field(default_factory=list)
    constraints_learned: list[str] = field(default_factory=list)
    lessons: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    rollback_token: Optional[int] = None

    def to_dict(self) -> dict:
        return asdict(self)

    def pretty(self) -> str:
        import json
        lines = [
            "",
            "=" * 64,
            f"EXECUTION REPORT  —  {self.status.upper()}  (confidence {self.confidence:.0%})",
            "=" * 64,
            f"Instruction: {self.instruction}",
            "",
        ]
        if self.done:
            lines.append("Done:")
            lines += [f"  ✓ {d}" for d in self.done]
        if self.failed:
            lines.append("Failed:")
            for f in self.failed:
                lines.append(f"  ✗ {f['step']} — {f['why']}  → {f['decision']}")
        if self.blocked:
            lines.append("Blocked (upstream failure):")
            lines += [f"  ⊘ {b}" for b in self.blocked]
        if self.synthesised:
            lines.append(f"Synthesised capabilities: {', '.join(self.synthesised)}")
        if self.constraints_learned:
            lines.append("Constraints learned this run:")
            lines += [f"  • {c}" for c in self.constraints_learned]
        if self.lessons:
            lines.append("Reflection lessons:")
            lines += [f"  • {l}" for l in self.lessons]
        lines.append("")
        lines.append("Metrics: " + json.dumps(self.metrics))
        if self.rollback_token is not None:
            lines.append(f"Rollback: python run.py --rollback {self.rollback_token}")
        lines.append("=" * 64)
        return "\n".join(lines)


def now_ms() -> int:
    return int(time.time() * 1000)
