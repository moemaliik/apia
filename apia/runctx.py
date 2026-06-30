"""The Ctx object handed to every capability (built-in or synthesised).

It exposes exactly what a capability is allowed to touch: the GitHub client,
read-only views of memory, a logger, and an undo recorder. Synthesised code
receives the same Ctx, so it can compose GitHub calls but cannot reach anything
the sandbox didn't inject.
"""
from __future__ import annotations

from typing import Any, Callable


class Ctx:
    def __init__(self, gh, memory, instruction_id: int, logger: Callable[[str], None]):
        self.gh = gh
        # The bare "owner/repo" string. Synthesised code may use ctx.repo, though
        # the literal {repo} token in a path is preferred (gh.request expands it).
        # owner/repo_name are the two halves, exposed because synthesised code
        # often reaches for them (e.g. to build a browser URL); missing them was a
        # common source of AttributeError in otherwise-correct capabilities.
        self.repo = getattr(gh, "repo", None)
        self.owner = self.repo.split("/")[0] if self.repo and "/" in self.repo else None
        self.repo_name = self.repo.split("/", 1)[1] if self.repo and "/" in self.repo else None
        self.memory = memory
        self.instruction_id = instruction_id
        self._log = logger
        self._undos: list[dict] = []

    def log(self, msg: str) -> None:
        self._log(msg)

    def record_undo(self, inverse: dict) -> None:
        self._undos.append(inverse)
        self.memory.push_undo(self.instruction_id, inverse)

    # read-only helpers synthesised code may use
    def known_labels(self) -> list[str]:
        try:
            return [l["name"] for l in self.gh.request("GET", "/repos/{repo}/labels")]
        except Exception:
            return []
