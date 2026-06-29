# APIA — Autonomous Platform Intelligence Agent

APIA takes a natural-language instruction, decomposes it, and executes it against
a real platform — **GitHub** — through real REST calls. It has persistent,
structured memory; it synthesises new capabilities at runtime when its built-in
toolset is insufficient; and it measurably gets cheaper and more reliable each
time it sees a similar instruction.

The agent's built-in toolset is deliberately just atomic CRUD primitives. Every
compound behaviour (filtering by a predicate, grouping issues by age, batch
labelling, rendering a summary) is **synthesised at runtime** and persisted, so
synthesis is load-bearing rather than decorative.

## Run it in 30 seconds, no keys, no risk

The agent ships with an in-memory mock GitHub repo and a recorded-LLM replay
mode, so the whole thing runs end-to-end with zero API spend and no token:

```bash
pip install -r requirements.txt

# run the three demo instructions against the mock repo with the replay LLM
APIA_GITHUB_MODE=mock APIA_LLM_PROVIDER=replay python run.py --demo

# watch the agent get cheaper across 5 runs of the same instruction
APIA_LLM_PROVIDER=replay python scripts/replay_demo.py 2

# see the learning curve + capability library the agent has accumulated
python dashboard.py
```

The `replay_demo.py` output is the headline result — same instruction, five
times, in one process:

```
run  llm  api  fail     ms  status   constraint-learned
  1    4   10     0     72  success  -
  2    1    6     0     55  success  create_label fails (422) if the label exists → precheck first
  3    0    6     0     46  success  -
  4    0    6     0     44  success  -
  5    0    6     0     52  success  -
```

LLM calls collapse (the plan and the synthesised capabilities are reused; the
agent stops reflecting once there's nothing new to learn); API calls drop once
it has learned the label already exists and that the issues are already
labelled.

## Run it for real against GitHub

```bash
cp .env.example .env        # then fill in ANTHROPIC_API_KEY, GITHUB_TOKEN, GITHUB_REPO
python run.py "Create an issue titled 'Login times out' and label it bug"
python run.py --demo
python run.py --rollback 3  # undo everything instruction #3 changed
```

Use a throwaway sandbox repo. A fine-grained PAT with issues read/write is enough.

## What's inside

| file | role |
|---|---|
| `apia/planner.py` | decomposes the instruction; **reuses** a past successful plan when the normalised signature matches (0 planning LLM calls) |
| `apia/synthesis.py` + `apia/sandbox.py` | generates a capability, **AST-validates** it, self-tests it read-only, registers it with provenance |
| `apia/executor.py` | runs the step DAG; constraint **prechecks**; partial-failure handling; capability lifecycle |
| `apia/memory/__init__.py` | SQLite structured memory: plans, steps, capabilities, constraints, operation stats, run metrics |
| `apia/reflection.py` | Reflexion-style self-critique → lessons fed into future plans |
| `apia/agent.py` | the orchestrator tying the specialist agents together |
| `dashboard.py` | the real before/after numbers |

See `ARCHITECTURE.md` for the design and `DEMO.md` for the three instructions.

## Dependencies

Two, both easy to defend: `requests` (HTTP for GitHub + the LLM — no vendor SDK)
and `rich` (the dashboard only; the agent core is pure standard library, incl.
`sqlite3` for memory and `ast` for the sandbox).
