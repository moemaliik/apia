# Architecture

APIA is a small multi-agent loop over one platform (GitHub). A **planner** turns
an instruction into a step DAG; an **executor** runs it, delegating any step it
can't satisfy with a built-in primitive to a **synthesis** agent; a
**reflection** agent critiques the run. All four read and write one persistent
SQLite memory. The single low-level primitive is `GitHubClient.request()` — every
capability, built-in or synthesised, goes through it, which is what makes the
API-call counts in the report real.

### 1. What does memory store, and why that and not something else?

Memory stores **structured knowledge**, not a transcript of past prompts, in
normalised SQLite tables, because the agent has to *query* it at decision time —
similarity search over old conversations couldn't drive the decisions below.

- **Execution layer** — `instructions`, `plans` (with a normalised signature and
  a `succeeded` flag), `steps`, `run_metrics`, `lessons`.
- **Capability layer** — `capabilities` (source code + lifecycle status +
  provenance), `capability_runs`, `operation_stats` (per-op success rate),
  `constraints` (learned platform rules).

Each table exists because a specific decision reads it: the planner reads
`plans` to reuse a decomposition; the executor reads `constraints` to skip a
call it knows will 422; synthesis reads `capabilities` to reuse code instead of
regenerating; capability selection reads `status` to prefer `trusted` over
`experimental`; the planner reads `lessons` to avoid past mistakes. Memory
persists across runs and is never wiped between them.

### 2. How does runtime capability synthesis work, and how is it bounded?

When the executor hits a step naming a capability that is neither a built-in nor
in capability memory, the synthesis agent: (1) prompts the LLM with the
available primitives to generate a `capability(ctx, **kwargs)` plus a read-only
`selftest`; (2) **statically validates** the code with the `ast` module — no
imports outside `{json, re, datetime, collections, math}`, no dunder access, no
`open`/`eval`/`exec`/`__import__`, running under a curated builtins dict; (3)
runs `selftest(ctx)` against the live platform read-only; (4) on success
registers it as `experimental` with provenance (instruction, model, attempts);
on failure feeds the error back and retries up to `MAX_SYNTH_ATTEMPTS`. The
sandbox is defence-in-depth against an LLM accidentally reaching the filesystem
or network — the only I/O a capability can do is through `ctx.gh`. Synthesised
code persists, so the next similar instruction reuses it with zero LLM calls.

A capability earns `trusted` after `PROMOTE_AFTER` successes and is `deprecated`
after `DEPRECATE_AFTER` consecutive failures (and can earn its way back), so the
library curates itself rather than accumulating dead code.

### 3. What is the learning signal, and how does run N differ from run 1?

The signal is **measured, not asserted**: every run records `llm_calls`,
`api_calls`, `failures` and `wall_ms` against the instruction signature
(`run_metrics`), and `dashboard.py` shows the run-1-vs-latest delta. Three
mechanisms move those numbers:

1. **Plan reuse** — a matching successful signature means the decomposition is
   reused verbatim: planning LLM calls go to 0.
2. **Capability reuse + promotion** — synthesised code is cached and reused
   (synthesis LLM calls go to 0) and promoted to `trusted`.
3. **Constraint learning** — a 422 `already_exists` on `create_label` is written
   to the constraint ledger; subsequent runs precheck and skip the doomed write,
   and the agent only reflects when there's something new to learn, so once the
   task is stable the LLM cost is zero.

Measured on five runs of *"add needs-triage to all unassigned issues"*: LLM
calls **4 → 0**, API calls **10 → 6**, failures **0 → 0**, wall time falls with
the LLM calls. Run 1 plans from scratch, synthesises two capabilities, and
reflects; run 5 reuses a trusted plan and trusted capabilities, prechecks the
label, and skips writes to already-labelled issues.

### Trade-offs

Signature matching is a normalised-text key, not embeddings — transparent and
dependency-free, at the cost of not generalising across very differently-worded
but equivalent instructions. The sandbox is an allowlist suitable for
semi-trusted LLM output, not a hostile-code boundary (no seccomp/containers).
The mock transport enforces the constraints we rely on (422 on duplicate label,
404 on missing issue) but is not a full GitHub emulation.
