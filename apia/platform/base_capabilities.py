"""The built-in capability set is deliberately small and atomic — CRUD only.

Anything compound (filter by predicate, group/bucket, batch over many issues,
render a summary, compute age) is intentionally absent so that capability
synthesis is genuinely exercised rather than decorative. Each built-in also
declares an `undo` so rollback works for the common mutations.
"""
from __future__ import annotations

from typing import Any, Callable


# Each capability: fn(ctx, **args) -> dict.  ctx.gh is the GitHubClient.
def get_repo_info(ctx, **_):
    return ctx.gh.request("GET", "/repos/{repo}")


def list_open_issues(ctx, **_):
    issues = ctx.gh.request("GET", "/repos/{repo}/issues", params={"state": "open"})
    return {"issues": issues, "count": len(issues)}


def get_issue(ctx, number: int, **_):
    return ctx.gh.request("GET", f"/repos/{{repo}}/issues/{number}")


def create_issue(ctx, title: str, body: str = "", labels: list | None = None, **_):
    issue = ctx.gh.request("POST", "/repos/{repo}/issues",
                           json_body={"title": title, "body": body, "labels": labels or []})
    ctx.record_undo({"op": "close_issue", "number": issue["number"]})
    return issue


def update_issue(ctx, number: int, **fields):
    before = ctx.gh.request("GET", f"/repos/{{repo}}/issues/{number}")
    out = ctx.gh.request("PATCH", f"/repos/{{repo}}/issues/{number}", json_body=fields)
    ctx.record_undo({"op": "restore_issue", "number": number,
                     "fields": {k: before.get(k) for k in fields}})
    return out


def add_labels_to_issue(ctx, number: int, labels: list, **_):
    issue = ctx.gh.request("GET", f"/repos/{{repo}}/issues/{number}")
    existing = [l["name"] for l in issue.get("labels", [])]
    merged = sorted(set(existing) | set(labels))
    out = ctx.gh.request("PATCH", f"/repos/{{repo}}/issues/{number}",
                         json_body={"labels": merged})
    ctx.record_undo({"op": "set_labels", "number": number, "labels": existing})
    return out


def list_labels(ctx, **_):
    labels = ctx.gh.request("GET", "/repos/{repo}/labels")
    return {"labels": labels, "names": [l["name"] for l in labels]}


def create_label(ctx, name: str, color: str = "ededed", **_):
    out = ctx.gh.request("POST", "/repos/{repo}/labels",
                         json_body={"name": name, "color": color})
    ctx.record_undo({"op": "noop", "note": f"label {name} created"})
    return out


def add_comment(ctx, number: int, body: str, **_):
    return ctx.gh.request("POST", f"/repos/{{repo}}/issues/{number}/comments",
                          json_body={"body": body})


BUILTINS: dict[str, Callable[..., Any]] = {
    "get_repo_info": get_repo_info,
    "list_open_issues": list_open_issues,
    "get_issue": get_issue,
    "create_issue": create_issue,
    "update_issue": update_issue,
    "add_labels_to_issue": add_labels_to_issue,
    "list_labels": list_labels,
    "create_label": create_label,
    "add_comment": add_comment,
}

# Compact signatures shown to the planner & synthesiser so they know the toolbox.
BUILTIN_SIGNATURES = {
    "get_repo_info": "() -> repo metadata",
    "list_open_issues": "() -> {issues:[...], count:int}",
    "get_issue": "(number:int) -> issue",
    "create_issue": "(title:str, body:str='', labels:list=[]) -> issue",
    "update_issue": "(number:int, **fields) -> issue   # title/body/state",
    "add_labels_to_issue": "(number:int, labels:list) -> issue   # merges, idempotent",
    "list_labels": "() -> {labels:[...], names:[...]}",
    "create_label": "(name:str, color:str='ededed') -> label   # 422 if already exists",
    "add_comment": "(number:int, body:str) -> comment",
}
