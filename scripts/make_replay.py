"""Generates apia/replay/responses.json — the recorded LLM outputs that let the
agent run end-to-end (and reproducibly) with APIA_LLM_PROVIDER=replay. Run once:
    python scripts/make_replay.py
"""
import json
from pathlib import Path

SIG1 = "create an issue titled qstr and label it bug"
SIG2 = "find all open issues with no assignee and add the qstr label"
SIG3 = "find all open unlabeled issues group them by age this week this"

plan1 = [
    {"description": "Ensure the 'bug' label exists", "capability": "create_label",
     "args": {"name": "bug", "color": "d73a4a"}, "depends_on": []},
    {"description": "Create the issue", "capability": "create_issue",
     "args": {"title": "Login times out after 30 seconds",
              "body": "Reported via APIA.", "labels": ["bug"]}, "depends_on": [0]},
]
plan2 = [
    {"description": "List open issues", "capability": "list_open_issues",
     "args": {}, "depends_on": []},
    {"description": "Filter issues with no assignee", "capability": "filter_unassigned",
     "args": {"issues": {"$step": 0, "get": "issues"}}, "depends_on": [0]},
    {"description": "Ensure the 'needs-triage' label exists", "capability": "create_label",
     "args": {"name": "needs-triage", "color": "fbca04"}, "depends_on": []},
    {"description": "Add 'needs-triage' to each unassigned issue", "capability": "label_issues",
     "args": {"numbers": {"$step": 1, "get": "numbers"}, "label": "needs-triage"},
     "depends_on": [1, 2]},
]
plan3 = [
    {"description": "List open issues", "capability": "list_open_issues",
     "args": {}, "depends_on": []},
    {"description": "Filter unlabeled issues", "capability": "filter_unlabeled",
     "args": {"issues": {"$step": 0, "get": "issues"}}, "depends_on": [0]},
    {"description": "Bucket issues by age", "capability": "bucket_issues_by_age",
     "args": {"issues": {"$step": 1, "get": "issues"}}, "depends_on": [1]},
    {"description": "Create the triage summary issue", "capability": "create_issue",
     "args": {"title": "Weekly triage summary",
              "body": {"$step": 2, "get": "markdown"}, "labels": []}, "depends_on": [2]},
]

filter_unassigned = '''```python
def capability(ctx, **kwargs):
    issues = kwargs.get("issues") or []
    nums = []
    for it in issues:
        if not it.get("assignee") and not (it.get("assignees") or []):
            nums.append(it.get("number"))
    return {"numbers": nums, "count": len(nums)}

def selftest(ctx):
    out = capability(ctx, issues=[])
    return isinstance(out, dict) and "numbers" in out
```'''

label_issues = '''```python
def capability(ctx, **kwargs):
    numbers = kwargs.get("numbers") or []
    label = kwargs.get("label")
    updated = []
    skipped = []
    for n in numbers:
        issue = ctx.gh.request("GET", "/repos/{repo}/issues/" + str(n))
        existing = [l.get("name") for l in (issue.get("labels") or [])]
        if label in existing:
            skipped.append(n)
            continue
        merged = sorted(set(existing) | {label})
        ctx.gh.request("PATCH", "/repos/{repo}/issues/" + str(n), json_body={"labels": merged})
        updated.append(n)
    return {"updated": updated, "skipped": skipped, "count": len(updated)}

def selftest(ctx):
    out = capability(ctx, numbers=[], label="x")
    return isinstance(out, dict) and out.get("count") == 0
```'''

filter_unlabeled = '''```python
def capability(ctx, **kwargs):
    issues = kwargs.get("issues") or []
    unl = []
    for it in issues:
        if not (it.get("labels") or []):
            unl.append(it)
    return {"numbers": [it.get("number") for it in unl], "issues": unl, "count": len(unl)}

def selftest(ctx):
    out = capability(ctx, issues=[])
    return isinstance(out, dict) and out.get("count") == 0
```'''

bucket_by_age = '''```python
def capability(ctx, **kwargs):
    issues = kwargs.get("issues") or []
    now = datetime.datetime.now(datetime.timezone.utc)
    buckets = {"this week": [], "this month": [], "older": []}
    for it in issues:
        created = it.get("created_at") or ""
        age = 99999
        if len(created) >= 10:
            try:
                y = int(created[0:4])
                mo = int(created[5:7])
                d = int(created[8:10])
                dt = datetime.datetime(y, mo, d, tzinfo=datetime.timezone.utc)
                age = (now - dt).days
            except ValueError:
                age = 99999
        if age <= 7:
            buckets["this week"].append(it)
        elif age <= 31:
            buckets["this month"].append(it)
        else:
            buckets["older"].append(it)
    lines = ["## Triage summary", ""]
    for name in ["this week", "this month", "older"]:
        items = buckets[name]
        lines.append("### " + name + " (" + str(len(items)) + ")")
        for it in items:
            lines.append("- [ ] #" + str(it.get("number")) + " " + str(it.get("title") or ""))
        lines.append("")
    counts = {}
    for name in buckets:
        counts[name] = len(buckets[name])
    return {"buckets": counts, "markdown": "\\n".join(lines)}

def selftest(ctx):
    sample = [{"number": 0, "title": "x", "created_at": "2020-01-01T00:00:00Z", "labels": []}]
    out = capability(ctx, issues=sample)
    return isinstance(out, dict) and "markdown" in out and out["buckets"]["older"] == 1
```'''

responses = {
    f"plan::{SIG1}": json.dumps(plan1),
    f"plan::{SIG2}": json.dumps(plan2),
    f"plan::{SIG3}": json.dumps(plan3),
    "synthesize::filter_unassigned": filter_unassigned,
    "synthesize::label_issues": label_issues,
    "synthesize::filter_unlabeled": filter_unlabeled,
    "synthesize::bucket_issues_by_age": bucket_by_age,
    f"reflect::{SIG1}": json.dumps([
        "The 'bug' label already exists; precheck labels before create_label to avoid a 422.",
        "Reuse this plan: ensure-label then create_issue."]),
    f"reflect::{SIG2}": json.dumps([
        "Precheck the needs-triage label before create_label.",
        "Reuse filter_unassigned + label_issues; skip issues already labelled."]),
    f"reflect::{SIG3}": json.dumps([
        "Reuse filter_unlabeled + bucket_issues_by_age for triage summaries."]),
}

out = Path(__file__).resolve().parent.parent / "apia" / "replay" / "responses.json"
out.write_text(json.dumps(responses, indent=2))
print(f"wrote {out} ({len(responses)} keys)")
