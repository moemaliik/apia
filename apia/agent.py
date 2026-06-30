"""Agent — the orchestrator (multi-agent: a planner hands a step DAG to the
execution agent, which delegates novel steps to the synthesis agent; a
reflection agent critiques the run afterwards).

Per instruction it:
  1. plans (reuse-aware),
  2. executes,
  3. records the run's measured metrics against the instruction signature
     (this is the learning curve the dashboard reads),
  4. saves the plan as reusable iff it succeeded,
  5. reflects to produce lessons (only when it could change future behaviour),
  6. opportunistically compacts old execution detail.
"""
from __future__ import annotations

from . import config
from .executor import ExecutionAgent
from .llm import LLM
from .memory import Memory
from .metrics import Meter
from .planner import PlannerAgent
from .platform import GitHubClient
from .reflection import ReflectionAgent
from .schemas import ExecutionReport


class Agent:
    def __init__(self, memory: Memory | None = None, verbose: bool = True, transport=None):
        self.memory = memory or Memory()
        self.verbose = verbose
        self.transport = transport  # if set (e.g. a shared MockTransport), reused across runs

    def _log(self, msg: str):
        if self.verbose:
            print(f"  · {msg}")

    def run(self, instruction: str) -> ExecutionReport:
        meter = Meter()
        gh = GitHubClient(meter, transport=self.transport)
        llm = LLM(meter)
        planner = PlannerAgent(llm, self.memory)

        sig = PlannerAgent.signature(instruction)
        run_no = self.memory.next_run_number(sig)
        iid = self.memory.start_instruction(instruction, sig)
        self._log(f"signature='{sig}'  run #{run_no}")

        try:
            plan = planner.plan(instruction)
        except LLMError as e:
            # Planning needs the LLM; if it's unreachable (quota/network) we can't
            # decompose. Fail cleanly with a report instead of crashing the run.
            self._log(f"planning failed: {e}")
            report = ExecutionReport(instruction=instruction, instruction_id=iid,
                                     status="failed", confidence=0.05)
            report.failed.append({"step": "Plan the instruction",
                                  "why": f"LLM unavailable: {e}",
                                  "decision": "aborted before execution"})
            report.metrics = meter.snapshot()
            report.rollback_token = iid
            self.memory.finish_instruction(iid, "failed", report.metrics)
            self.memory.record_run_metric(sig, run_no, report.metrics, "fresh")
            return report
        self._log(f"plan source = {plan.source.upper()}  ({len(plan.steps)} steps, "
                  f"{'0 planning LLM calls' if plan.source == 'reused' else '1 planning LLM call'})")

        ex = ExecutionAgent(gh, llm, self.memory, iid, self._log)
        report = ex.run(plan)

        # save reusable plan only if the whole thing succeeded
        self.memory.save_plan(sig, iid, plan.to_dict()["steps"],
                              succeeded=(report.status == "success"))

        # reflect only when it could change future behaviour
        if report.failed or report.synthesised or report.constraints_learned or plan.source == "fresh":
            reflector = ReflectionAgent(llm, self.memory)
            report.lessons = reflector.reflect(iid, sig, plan, {
                "status": report.status,
                "failed": report.failed,
                "synthesised": report.synthesised,
                "constraints_learned": report.constraints_learned,
            })

        m = meter.snapshot()
        self.memory.finish_instruction(iid, report.status, m)
        self.memory.record_run_metric(sig, run_no, m, plan.source)

        # opportunistic compaction
        if self.memory.step_detail_count(sig) > config.COMPACT_AFTER:
            n = self.memory.compact_signature(
                sig, f"Compacted older runs of '{sig}'. Stable plan + learned constraints retained.")
            if n:
                self._log(f"compacted {n} old step rows for this signature")

        report.metrics = m
        return report

    def close(self):
        self.memory.close()
