"""ExecutionAgent — runs a plan as a dependency graph.

Responsibilities:
  - topological execution honouring depends_on; pass results between steps
  - resolve each step's capability: built-in, else delegate to SynthesisAgent
  - CONSTRAINT PRECHECKS: consult the learned-constraint ledger before a call
    and skip calls that memory knows will fail (this is the API-call/failure
    learning signal you can point at with numbers)
  - PARTIAL FAILURE: a failed step is recorded with cause + decision; its
    dependents are marked blocked; independent steps still run. Never a silent
    half-completion.
  - record operation stats + capability lifecycle (promote/demote) + undo log
  - compute a confidence score for the run
"""
from __future__ import annotations

import time
from typing import Any, Optional

from . import config
from .platform.base_capabilities import BUILTINS
from .platform import GitHubError
from .runctx import Ctx
from .schemas import Step, Plan, ExecutionReport
from .synthesis import SynthesisAgent

# capability name -> underlying mutating op (for constraint prechecks)
OP_OF = {"create_label": "create_label"}


class ExecutionAgent:
    def __init__(self, gh, llm, memory, instruction_id: int, log):
        self.gh = gh
        self.memory = memory
        self.instruction_id = instruction_id
        self.log = log
        self.synth = SynthesisAgent(llm, memory)
        self.ctx = Ctx(gh, memory, instruction_id, log)
        self._label_cache: Optional[list[str]] = None

    # ---- public ----------------------------------------------------------
    def run(self, plan: Plan) -> ExecutionReport:
        report = ExecutionReport(instruction=plan.instruction,
                                 instruction_id=self.instruction_id,
                                 status="success", confidence=1.0)
        results: dict[int, Any] = {}
        order = self._toposort(plan.steps)
        by_idx = {s.idx: s for s in plan.steps}

        for idx in order:
            step = by_idx[idx]
            # block if any dependency failed/blocked
            if any(by_idx[d].status in ("failed", "blocked") for d in step.depends_on):
                step.status = "blocked"
                report.blocked.append(step.description)
                continue
            self._execute_step(step, results, plan, report)
            if step.status == "done":
                report.done.append(step.description)

        # finalise status
        if report.failed and report.done:
            report.status = "partial"
        elif report.failed and not report.done:
            report.status = "failed"
        report.confidence = self._confidence(plan, report)
        report.rollback_token = self.instruction_id
        report.metrics = self.gh.meter.snapshot()
        # persist steps
        for s in plan.steps:
            self.memory.save_step(self.instruction_id, s.to_dict())
        return report

    # ---- per-step --------------------------------------------------------
    def _execute_step(self, step: Step, results: dict, plan: Plan, report: ExecutionReport):
        t0 = time.time()
        op = OP_OF.get(step.capability)

        # CONSTRAINT PRECHECK — skip calls memory knows will fail
        if op and self._precheck_skips(op, step):
            step.status = "done"
            step.wall_ms = int((time.time() - t0) * 1000)
            report.done.append(step.description + " (precheck: already satisfied)")
            report.decisions.append(
                f"Skipped `{step.capability}` — learned constraint says it would fail; "
                f"end-state already satisfied.")
            return

        # resolve capability
        fn, synthesised = self._resolve(step, report)
        if fn is None:
            step.status = "failed"
            step.error = "capability unavailable (synthesis failed)"
            report.failed.append({"step": step.description, "why": step.error,
                                  "decision": "reported gap; downstream steps blocked"})
            self.gh.meter.fail()
            return
        step.synthesised = synthesised

        args = self._resolve_args(step.args, results)
        retries = config.MAX_STEP_RETRIES
        for attempt in range(1, retries + 2):
            step.attempts = attempt
            api_before = self.gh.meter.api_calls
            try:
                step.result = fn(self.ctx, **args)
                step.status = "done"
                step.api_calls = self.gh.meter.api_calls - api_before
                step.wall_ms = int((time.time() - t0) * 1000)
                results[step.idx] = step.result
                self.memory.record_operation(step.capability, True, step.wall_ms)
                self.memory.record_capability_run(step.capability, self.instruction_id,
                                                  True, step.wall_ms, None)
                self._bump_lifecycle(step.capability, True, report)
                return
            except GitHubError as e:
                step.api_calls = self.gh.meter.api_calls - api_before
                handled = self._handle_github_error(e, step, op, report)
                if handled == "satisfied":
                    step.status = "done"
                    step.wall_ms = int((time.time() - t0) * 1000)
                    results[step.idx] = {"satisfied": True}
                    return
                step.error = e.message
                if attempt <= retries and handled == "retry":
                    continue
                break
            except Exception as e:  # synthesised code or arg error
                step.error = f"{type(e).__name__}: {e}"
                break

        # exhausted
        step.status = "failed"
        step.wall_ms = int((time.time() - t0) * 1000)
        self.gh.meter.fail()
        self.memory.record_operation(step.capability, False, step.wall_ms)
        self.memory.record_capability_run(step.capability, self.instruction_id,
                                          False, step.wall_ms, step.error)
        self._bump_lifecycle(step.capability, False, report)
        report.failed.append({"step": step.description, "why": step.error,
                              "decision": "retries exhausted; dependents blocked"})

    # ---- helpers ---------------------------------------------------------
    def _resolve(self, step: Step, report: ExecutionReport):
        if step.capability in BUILTINS:
            return BUILTINS[step.capability], False
        # need synthesis
        cached = self.memory.get_capability(step.capability)
        spec = step.description if not cached else cached["spec"]
        res = self.synth.ensure(step.capability, spec, self.ctx)
        if res.ok:
            if res.attempts > 0:  # genuinely synthesised (not pure cache hit)
                report.synthesised.append(step.capability)
                report.decisions.append(
                    f"Synthesised `{step.capability}` in {res.attempts} attempt(s) and "
                    f"registered it (experimental).")
            return res.fn, res.attempts > 0
        report.decisions.append(
            f"Could not synthesise `{step.capability}` after {res.attempts} attempts: {res.error}")
        return None, False

    def _precheck_skips(self, op: str, step: Step) -> bool:
        constraints = self.memory.constraints_for(op)
        if not any(c["rule"].get("precheck") == "label_exists" for c in constraints):
            return False
        name = step.args.get("name")
        if name is None:
            return False
        if self._label_cache is None:
            self._label_cache = self.ctx.known_labels()  # one fetch, then cached
        return name in self._label_cache

    @staticmethod
    def _is_already_exists(e: GitHubError) -> bool:
        """True if a 422 means 'the resource already exists' — works whether the
        signal is in the message text (mock) or in errors[].code (real GitHub)."""
        if "already_exists" in (e.message or "").lower():
            return True
        payload = e.payload if isinstance(e.payload, dict) else {}
        return any(isinstance(err, dict) and err.get("code") == "already_exists"
                   for err in (payload.get("errors") or []))

    def _handle_github_error(self, e: GitHubError, step: Step, op: Optional[str],
                             report: ExecutionReport) -> str:
        msg = (e.message or "").lower()
        if e.status == 422 and op == "create_label" and self._is_already_exists(e):
            new = self.memory.add_constraint(
                op, {"precheck": "label_exists", "key": "name"},
                evidence=f"422 already_exists for {step.args.get('name')}")
            if new:
                report.constraints_learned.append(
                    "create_label fails (422) if the label exists → precheck first")
                report.decisions.append(
                    "Learned: label already existed; treating create as satisfied.")
            if self._label_cache is not None and step.args.get("name"):
                self._label_cache.append(step.args["name"])
            return "satisfied"
        if e.status in (502, 503, 429):
            return "retry"
        return "fail"

    def _bump_lifecycle(self, capability: str, success: bool, report: ExecutionReport):
        if capability in BUILTINS:
            return
        before = self.memory.get_capability(capability)
        if not before:
            return
        new_status = self.memory.bump_capability(capability, success)
        if before["status"] != new_status:
            report.decisions.append(
                f"Capability `{capability}` {before['status']} → {new_status}.")

    @staticmethod
    def _resolve_args(args: dict, results: dict) -> dict:
        out = {}
        for k, v in (args or {}).items():
            if isinstance(v, dict) and "$step" in v:
                base = results.get(v["$step"])
                out[k] = base.get(v["get"]) if (isinstance(base, dict) and "get" in v) else base
            else:
                out[k] = v
        return out

    @staticmethod
    def _toposort(steps: list[Step]) -> list[int]:
        idxs = {s.idx for s in steps}
        deps = {s.idx: [d for d in s.depends_on if d in idxs] for s in steps}
        order, seen = [], set()

        def visit(n, stack):
            if n in seen:
                return
            if n in stack:  # cycle guard
                return
            stack.add(n)
            for d in deps.get(n, []):
                visit(d, stack)
            stack.discard(n)
            seen.add(n)
            order.append(n)

        for s in sorted(steps, key=lambda x: x.idx):
            visit(s.idx, set())
        return order

    @staticmethod
    def _confidence(plan: Plan, report: ExecutionReport) -> float:
        c = 1.0
        if plan.source == "fresh":
            c -= 0.10
        c -= 0.20 * len(report.failed)
        c -= 0.08 * len(report.synthesised)   # experimental code in the loop
        return max(0.05, round(c, 2))
