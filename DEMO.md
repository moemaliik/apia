# Demo

Three instructions of increasing complexity. Each shows a different part of the
system. Run them against the mock repo with the replay LLM (no keys, no risk):

```bash
APIA_GITHUB_MODE=mock APIA_LLM_PROVIDER=replay python run.py --demo
```

…or against a real sandbox repo once your `.env` is filled in (`python run.py --demo`).

---

## 1. "Create an issue titled 'Login times out after 30 seconds' and label it bug"

**What it exercises:** the basic plan → execute → report loop, plus immediate
constraint learning. The plan is two built-in steps: ensure the `bug` label
exists, then create the issue.

**What to watch:** `bug` already exists, so `create_label` returns 422
`already_exists`. The agent does **not** treat this as a failure — it records the
constraint *"create_label fails (422) if the label exists → precheck first"*,
treats the end-state as satisfied, and proceeds to create the issue. Report:
`SUCCESS`, one constraint learned. On any later run of this shape the agent
prechecks and skips the label call entirely.

## 2. "Find all open issues with no assignee and add the 'needs-triage' label to each"

**What it exercises:** runtime synthesis + batch work + data passed between
steps. No built-in filters by assignee or batches a label across many issues, so
the agent synthesises two capabilities — `filter_unassigned` and `label_issues`
(idempotent: it GETs each issue and only PATCHes the ones missing the label) —
validates them in the sandbox, self-tests them read-only, and registers them.

**What to watch:** first run synthesises both capabilities (you'll see them in
"Synthesised capabilities" and in `dashboard.py` as `experimental`). After three
successful uses they're promoted to `trusted`. This is the instruction used for
the learning-curve harness below.

## 3. "Find all open unlabeled issues, group them by age (this week / this month / older), and create one triage summary issue with a checklist grouped by bucket"

**What it exercises:** the most compound case — a novel instruction type with no
matching primitive at all. The agent synthesises `filter_unlabeled` and
`bucket_issues_by_age` (which computes age from `created_at` and renders a
grouped markdown checklist), then composes them with the built-in `create_issue`
to post the summary. Demonstrates that "intelligence" lives in synthesised,
composable capabilities rather than hardcoded handlers.

---

## Seeing the learning loop (the measurable part)

The mock repo resets each **process**, so to see API-side savings accumulate,
run the same instruction several times in **one** process:

```bash
APIA_LLM_PROVIDER=replay python scripts/replay_demo.py 2     # 1, 2 or 3
```

Expected (instruction 2):

```
run  llm  api  fail     ms  status   constraint-learned
  1    4   10     0     72  success  -
  2    1    6     0     55  success  create_label fails (422) if the label exists → precheck first
  3    0    6     0     46  success  -
  4    0    6     0     44  success  -
  5    0    6     0     52  success  -
```

`llm 4 → 0`, `api 10 → 6`. Note that **LLM** savings (plan + capability reuse)
persist even across separate `python run.py` invocations because they live in
the SQLite memory; the **API** savings (constraint precheck, skipping
already-labelled issues) need persistent platform state, which you get either in
a single process (the harness above) or against a real GitHub repo where the
label and labels genuinely persist server-side.

View everything the agent has accumulated:

```bash
python dashboard.py
```

## Rollback

Every mutating built-in records its inverse, so a whole instruction can be undone:

```bash
python run.py --rollback 3
```
