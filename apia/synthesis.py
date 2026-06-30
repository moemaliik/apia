"""SynthesisAgent — the 'real version' of capability synthesis.

When the executor needs a capability that neither the built-ins nor capability
memory provide, this agent:
  1. reasons about what the capability must do, given the available primitives,
  2. generates a Python function `capability(ctx, **kwargs)` plus a `selftest`,
  3. statically validates + compiles it in the sandbox,
  4. runs `selftest(ctx)` against the live platform (read-only probe),
  5. on success registers it into capability memory (status=experimental) with
     full provenance; on failure feeds the error back and retries up to N;
  6. after N failures, returns a structured failure the executor reports.

Synthesis happens at runtime and persists — re-running a similar instruction
later reuses the stored code with zero LLM calls.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional

from . import config
from .llm import LLM
from .platform.base_capabilities import BUILTIN_SIGNATURES
from .sandbox import compile_capability, SandboxError

SYNTH_SYSTEM = """You are a capability synthesiser for a GitHub automation agent.
You write a single Python function that performs ONE compound operation by
composing low-level primitives. You also write a selftest.

HARD RULES
- Define exactly: def capability(ctx, **kwargs): ...  returning a dict.
- Define: def selftest(ctx): ... It MUST return the literal boolean True on
  success (never a dict, never the capability's output). selftest MUST be
  read-only (no create/update/delete) — exercise parsing/grouping/filtering
  logic on data fetched read-only (or on a small in-line sample).
- selftest MUST exercise the capability against the REAL platform. NEVER
  reassign, monkeypatch, wrap, or otherwise replace ctx.gh / ctx.gh.request
  (e.g. `ctx.gh.request = mock` is forbidden). A selftest that stubs the
  platform proves nothing — it only confirms your own assumptions about the
  data shape, so a wrong assumption passes the test and then crashes for real.
  Either call the capability directly (it will hit the live read-only API) or
  test pure helper logic on an in-line literal sample WITHOUT touching ctx.gh.
- The ONLY way to touch GitHub is ctx.gh.request(method, path, json_body=None, params=None).
  Paths use the literal token {repo}, e.g. "/repos/{repo}/issues" — write {repo}
  verbatim; ctx.gh expands it. Do NOT build paths from ctx.repo — prefer the
  {repo} token. ctx.repo is the full "owner/repo" string; ctx.owner and
  ctx.repo_name are its two halves if you need them (e.g. to build a browser
  URL like https://github.com/{owner}/{repo_name}/issues/N). NEVER use ctx.gh.repo,
  and do NOT invent other ctx attributes — only ctx.gh, ctx.repo, ctx.owner,
  ctx.repo_name, ctx.known_labels(), ctx.log() exist.
- ctx.gh.request returns the RAW GitHub JSON, NOT the wrapped shape the
  built-ins return. In particular:
    * GET /repos/{repo}/issues       -> a LIST of issue dicts (NOT {"issues":[...]})
    * GET /repos/{repo}/labels       -> a LIST of label dicts
    * GET /repos/{repo}/issues/{n}   -> a single issue dict
    * GET /repos/{repo}              -> a repo dict
  Each issue dict has keys like number, title, state, labels (list of {"name":..}),
  assignee (a dict or None), assignees (list). The {"issues":[...], "count":N}
  / {"labels":[...], "names":[...]} shapes below belong to the BUILT-IN wrappers
  only — do NOT expect ctx.gh.request to return them. If you want the wrapped
  shape, the simplest path is to NOT re-fetch and instead consume the upstream
  step's result via kwargs.
- You MAY import only: json, re, datetime, collections, math.
- No file, network (other than ctx.gh), os, subprocess, eval/exec/open.
- Keep it small and defensive. Handle empty lists. Never raise on normal data.

AVAILABLE PRIMITIVES (you can call ctx.gh.request directly OR rely on these patterns).
NOTE: the shapes shown here are the BUILT-IN return shapes; ctx.gh.request
returns raw JSON as described above, not these wrappers:
%s

You may also read ctx.known_labels() -> list[str].
""" % "\n".join(f"  {k}{v}" for k, v in BUILTIN_SIGNATURES.items())


@dataclass
class SynthesisResult:
    ok: bool
    name: str
    source: Optional[str] = None
    fn: Optional[object] = None
    attempts: int = 0
    error: Optional[str] = None
    provenance: Optional[dict] = None


class SynthesisAgent:
    def __init__(self, llm: LLM, memory):
        self.llm = llm
        self.memory = memory

    def ensure(self, name: str, spec: str, ctx) -> SynthesisResult:
        # 1) reuse from capability memory if present and not deprecated
        cached = self.memory.get_capability(name)
        if cached and cached["status"] != "deprecated":
            try:
                fn = compile_capability(cached["source_code"])
                return SynthesisResult(ok=True, name=name, source=cached["source_code"],
                                       fn=fn, attempts=0,
                                       provenance=json.loads(cached["provenance_json"] or "{}"))
            except SandboxError:
                pass  # corrupted; fall through and re-synthesise

        # 2) synthesise fresh
        return self._synthesize(name, spec, ctx)

    def resynthesize(self, name: str, spec: str, ctx, runtime_error: str,
                     args: Optional[dict] = None) -> SynthesisResult:
        """Repair a capability that PASSED selftest but failed at RUNTIME.

        The real execution error is the strongest possible signal — the selftest
        provably did not cover the failing path — so we feed it (plus the call
        args and the current broken source) back to the model and regenerate,
        bypassing the cache. This is what lets a cold-memory run self-heal a buggy
        first generation instead of dying on it.
        """
        cached = self.memory.get_capability(name)
        prior = cached["source_code"] if cached else "(unavailable)"
        seed = (f"At RUNTIME (not in selftest) the capability was called as "
                f"capability(ctx, **{args!r}) and raised:\n  {runtime_error}\n"
                "The selftest passed, so it did NOT exercise the failing path. Fix "
                "the capability so this cannot happen and make it defensive about the "
                "inputs shown. Write a selftest that actually CALLS capability(ctx, ...) "
                "rather than re-implementing its logic, so the real path is covered.\n"
                f"--- current (broken) code ---\n{prior}\n--- end ---")
        return self._synthesize(name, spec, ctx, seed_err=seed)

    def _synthesize(self, name: str, spec: str, ctx,
                    seed_err: Optional[str] = None) -> SynthesisResult:
        last_err = seed_err
        for attempt in range(1, config.MAX_SYNTH_ATTEMPTS + 1):
            user = self._prompt(name, spec, last_err)
            raw = self.llm.complete(SYNTH_SYSTEM, user,
                                    cache_key=f"synthesize::{name}", max_tokens=1800)
            source = _extract_code(raw)
            try:
                module_fn = compile_capability(source, "capability")
                selftest = compile_capability(source, "selftest")
            except SandboxError as e:
                last_err = f"sandbox rejected code: {e}"
                continue
            try:
                if selftest(ctx) is not True:
                    last_err = "selftest returned non-True"
                    continue
            except Exception as e:
                last_err = f"selftest raised: {e}"
                continue
            provenance = {
                "synthesised_for": spec,
                "model": self.llm.model,
                "attempts": attempt,
                "composed_from": "ctx.gh primitives",
                "repaired_from_runtime_error": seed_err is not None,
            }
            self.memory.save_capability(name, spec, source, provenance)
            return SynthesisResult(ok=True, name=name, source=source, fn=module_fn,
                                   attempts=attempt, provenance=provenance)
        return SynthesisResult(ok=False, name=name, attempts=config.MAX_SYNTH_ATTEMPTS,
                               error=last_err)

    def _prompt(self, name: str, spec: str, last_err: Optional[str]) -> str:
        p = (f"Synthesise a capability named `{name}`.\n"
             f"It must: {spec}\n\n"
             "Return ONLY a Python code block containing both `capability` and `selftest`.")
        if last_err:
            p += f"\n\nYour previous attempt failed with: {last_err}\nFix it."
        return p


def _extract_code(raw: str) -> str:
    raw = raw.strip()
    if "```" in raw:
        parts = raw.split("```")
        for i in range(1, len(parts), 2):
            block = parts[i]
            if block.startswith("python"):
                block = block[len("python"):]
            if "def capability" in block:
                return block.strip()
    return raw
