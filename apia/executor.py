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

import json
import time
from typing import Any, Optional

from . import config
from .platform.base_capabilities import BUILTINS, BUILTIN_SIGNATURES
from .platform import GitHubError
from .runctx import Ctx
from .schemas import Step, Plan, ExecutionReport
from .synthesis import SynthesisAgent

# capability name -> underlying mutating op (for constraint prechecks)
OP_OF = {"create_label": "create_label"}

# Built-in params GitHub requires to be plain strings. If an upstream step's
# whole result (a dict/list) got threaded into one of these — the classic
# planner slip of {"$step": N} instead of {"$step": N, "get": "..."} — we unwrap
# it to a string at the call boundary rather than letting the API reject it.
STRING_PARAMS = {
    "create_issue": ("title", "body"),
    "update_issue": ("title", "body"),
    "add_comment": ("body",),
    "create_label": ("name", "color"),
}
# keys a wrapper dict commonly hides its text under, tried in order
_TEXT_KEYS = ("body", "text", "content", "markdown", "md", "checklist",
              "summary", "message", "value", "result", "output")

_ARG_REPAIR_SYSTEM = """You fix the ARGUMENTS of one failed step in a GitHub
automation agent. You are given the capability and its signature, the arguments
that failed, the results of the upstream steps it depends on, and the platform
error. Return ONLY JSON: {"args": { ...corrected arguments... }}.
Rules:
- Keep the SAME capability; change only the arguments.
- A parameter typed `str` (e.g. title, body, name) must be a STRING, never an
  object or array. If the value is wrapped in an object, inline the inner string.
- Use ids/values that actually appear in the upstream results; never invent an
  issue number or a value that isn't there.
- If you cannot determine a valid fix, return {"args": null}."""


def _coerce_str(value):
    """Turn a dict/list accidentally threaded into a string-typed param into a
    string. Strings/None pass through untouched, so this is a safe no-op when
    the arg was already correct."""
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, dict):
        for k in _TEXT_KEYS:
            if isinstance(value.get(k), str):
                return value[k]
        strs = [v for v in value.values() if isinstance(v, str)]
        if len(strs) == 1:
            return strs[0]
        return json.dumps(value, indent=2, default=str)
    if isinstance(value, list):
        if all(isinstance(x, str) for x in value):
            return "\n".join(value)
        return json.dumps(value, indent=2, default=str)
    return str(value)


class ExecutionAgent:
    def __init__(self, gh, llm, memory, instruction_id: int, log):
        self.gh = gh
        self.llm = llm
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
                # Surface any GitHub URL the step produced so the user gets a
                # direct link instead of hunting in GitHub's eventually-consistent
                # Issues list (which can lag a created issue by a minute or two).
                if isinstance(step.result, dict):
                    url = step.result.get("html_url")
                    if isinstance(url, str) and url not in report.links:
                        report.links.append(url)

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
                                  "decision": f"reported gap{self._dependents_note(step, plan)}"})
            self.gh.meter.fail()
            return
        step.synthesised = synthesised

        args = self._resolve_args(step.args, results)
        # Pre-emptively unwrap a whole-result dict/list threaded into a string
        # param (zero-cost fix for the most common planner slip).
        args = self._coerce_builtin_args(step.capability, args)
        spec = self._spec_for(step)
        gh_retries = config.MAX_STEP_RETRIES
        gh_attempt = 0
        arg_repairs_left = config.MAX_ARG_REPAIRS
        # Only synthesised capabilities can be self-healed: a runtime error in one
        # is a defect the read-only selftest could not catch, so we feed it back to
        # the synthesiser and retry the regenerated code. Built-in failures are our
        # bug, not the model's, so they are never "repaired" this way.
        repairs_left = 0 if step.capability in BUILTINS else config.MAX_RUNTIME_REPAIRS
        while True:
            step.attempts += 1
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
                # A 4xx validation/type rejection is usually a bad PAYLOAD, not a
                # transient fault — try to self-heal the arguments and retry.
                if handled == "fail" and e.status in (400, 422) and arg_repairs_left > 0:
                    new_args = self._repair_args(step, args, e.message, results, report)
                    if new_args is not None:
                        arg_repairs_left -= 1
                        args = new_args
                        continue
                gh_attempt += 1
                if handled == "retry" and gh_attempt <= gh_retries:
                    continue
                break
            except Exception as e:  # bug in synthesised code or bad args
                step.error = f"{type(e).__name__}: {e}"
                new_fn = (self._repair_capability(step, spec, e, args, report)
                          if repairs_left > 0 else None)
                if new_fn is not None:
                    repairs_left -= 1
                    fn = new_fn
                    step.synthesised = True
                    continue
                # Arg-level self-heal works for built-ins too: a TypeError/KeyError
                # from a bad payload shape is fixable by fixing the args, not code.
                if arg_repairs_left > 0:
                    new_args = self._repair_args(step, args, step.error, results, report)
                    if new_args is not None:
                        arg_repairs_left -= 1
                        args = new_args
                        continue
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
                              "decision": f"retries exhausted{self._dependents_note(step, plan)}"})

    # ---- helpers ---------------------------------------------------------
    def _spec_for(self, step: Step) -> str:
        cached = self.memory.get_capability(step.capability)
        return cached["spec"] if cached else step.description

    def _repair_capability(self, step: Step, spec: str, exc: Exception,
                           args: dict, report: ExecutionReport):
        """Self-heal: hand the runtime error back to the synthesiser, which
        regenerates the capability. Returns the new callable, or None if repair
        failed. The regenerated version must still pass selftest before it runs."""
        err = f"{type(exc).__name__}: {exc}"
        res = self.synth.resynthesize(step.capability, spec, self.ctx,
                                      runtime_error=err, args=args)
        if res.ok:
            if step.capability not in report.synthesised:
                report.synthesised.append(step.capability)
            report.decisions.append(
                f"Self-healed `{step.capability}`: runtime {type(exc).__name__} fed back "
                f"to the synthesiser; regenerated code passed selftest.")
            return res.fn
        report.decisions.append(
            f"Could not repair `{step.capability}` after runtime {type(exc).__name__}: {res.error}")
        return None

    @staticmethod
    def _dependents_note(step: Step, plan: Plan) -> str:
        """Accurate suffix for a failed step's decision: only claim downstream
        steps are blocked when something actually depends on this one."""
        n = sum(1 for s in plan.steps if step.idx in s.depends_on)
        if n == 0:
            return "; no dependent steps"
        return f"; {n} dependent step{'s' if n > 1 else ''} blocked"

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
    def _coerce_builtin_args(capability: str, args: dict) -> dict:
        """Unwrap dict/list values that landed in a string-typed built-in param.
        No-op for synthesised capabilities and for already-correct args."""
        params = STRING_PARAMS.get(capability)
        if not params:
            return args
        out = dict(args)
        for p in params:
            if p in out:
                out[p] = _coerce_str(out[p])
        return out

    def _repair_args(self, step: Step, args: dict, error: str,
                     results: dict, report: ExecutionReport) -> Optional[dict]:
        """Self-heal a step by CORRECTING ITS ARGUMENTS — works for built-ins and
        synthesised capabilities alike, since the fix is the payload, not the
        code. Hands the model the signature/spec, the failing args, a digest of
        the upstream results it can draw on, and the error. Never raises: any LLM
        or parse failure degrades to None so the run still finishes."""
        sig = BUILTIN_SIGNATURES.get(step.capability) or f"  # {self._spec_for(step)}"
        upstream = self._upstream_digest(step, results)
        user = (f"Capability: {step.capability}{sig}\n"
                f"Arguments passed (FAILED): {json.dumps(args, default=str)[:1200]}\n"
                f"Results of the steps it depends on (draw correct values from these):\n"
                f"{json.dumps(upstream, default=str)[:1500]}\n"
                f"Platform error: {error}\n\n"
                "Return the corrected arguments as JSON.")
        try:
            data = self.llm.complete_json(_ARG_REPAIR_SYSTEM, user, max_tokens=700)
        except Exception as e:
            report.decisions.append(
                f"Arg-repair for `{step.capability}` unavailable ({type(e).__name__}); "
                "leaving step failed.")
            return None
        new = data.get("args") if isinstance(data, dict) else None
        if not isinstance(new, dict) or new == args:
            return None
        new = self._coerce_builtin_args(step.capability, new)
        msg = f"Self-healed args for `{step.capability}` after error: {error[:80]}"
        report.self_healed.append(msg)
        report.decisions.append(msg)
        return new

    @staticmethod
    def _upstream_digest(step: Step, results: dict) -> dict:
        return {d: ExecutionAgent._shape(results.get(d)) for d in step.depends_on}

    @staticmethod
    def _shape(v: Any) -> Any:
        """A compact, size-bounded view of a value so the repair model can see
        what keys/values upstream produced without flooding the prompt."""
        if isinstance(v, dict):
            return {k: ExecutionAgent._shape(x) for k, x in list(v.items())[:12]}
        if isinstance(v, list):
            if not v:
                return []
            head = [ExecutionAgent._shape(v[0])]
            return head + [f"...+{len(v) - 1} more"] if len(v) > 1 else head
        if isinstance(v, str):
            return v if len(v) <= 200 else v[:200] + "…"
        return v

    @staticmethod
    def _resolve_args(args: dict, results: dict) -> dict:
        out = {}
        for k, v in (args or {}).items():
            if isinstance(v, dict) and "$step" in v:
                base = results.get(v["$step"])
                if isinstance(base, dict) and "get" in v:
                    key = v["get"]
                    # If the planner guessed a key the upstream result doesn't
                    # actually have (common for synthesised capabilities whose
                    # output shape it can't see), don't silently yield None and
                    # write an empty payload — fall back to the whole result so
                    # downstream coercion can still pull the right value out.
                    out[k] = base[key] if key in base else base
                else:
                    out[k] = base
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
