"""Rollback — replays the inverse operations recorded in the undo_log for a given
instruction, newest-first. Built-ins record their own inverse at execution time
(create_issue -> close, update_issue -> restore fields, add_labels -> restore
prior label set), so undo is structural, not guesswork.
"""
from __future__ import annotations

import json

from .memory import Memory
from .metrics import Meter
from .platform import GitHubClient, GitHubError


def rollback(instruction_id: int, memory: Memory | None = None) -> list[str]:
    memory = memory or Memory()
    gh = GitHubClient(Meter())
    log: list[str] = []
    for undo in memory.undos_for(instruction_id):
        inv = json.loads(undo["inverse_json"])
        try:
            _apply(gh, inv)
            memory.mark_undo_applied(undo["id"])
            log.append(f"undone: {inv}")
        except GitHubError as e:
            log.append(f"failed to undo {inv}: {e}")
    return log


def _apply(gh: GitHubClient, inv: dict) -> None:
    op = inv["op"]
    if op == "close_issue":
        gh.request("PATCH", f"/repos/{{repo}}/issues/{inv['number']}",
                   json_body={"state": "closed"})
    elif op == "restore_issue":
        gh.request("PATCH", f"/repos/{{repo}}/issues/{inv['number']}",
                   json_body=inv["fields"])
    elif op == "set_labels":
        gh.request("PATCH", f"/repos/{{repo}}/issues/{inv['number']}",
                   json_body={"labels": inv["labels"]})
    elif op == "noop":
        pass
