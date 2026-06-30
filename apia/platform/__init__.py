"""GitHub access.

`GitHubClient.request()` is the single low-level primitive every capability —
built-in or synthesised — must go through. It increments the run meter on every
call, so API-call counts in the report are real.

Two transports:
  RealTransport  -> https://api.github.com (needs GITHUB_TOKEN + GITHUB_REPO)
  MockTransport  -> in-memory repo that enforces realistic constraints
                    (422 on duplicate label, 404 on missing issue, simple
                    rate-limit headers). Lets the agent run end-to-end and learn
                    constraints with no token and no risk to a real repo.
"""
from __future__ import annotations

import re
import time
from typing import Any

import requests

from .. import config
from ..metrics import Meter


class GitHubError(RuntimeError):
    def __init__(self, status: int, message: str, payload: Any = None):
        super().__init__(f"{status}: {message}")
        self.status = status
        self.message = message
        self.payload = payload


# --------------------------------------------------------------------------
class RealTransport:
    def __init__(self):
        if not config.GITHUB_TOKEN:
            raise GitHubError(0, "GITHUB_TOKEN is not set")
        if not config.GITHUB_REPO:
            raise GitHubError(0, "GITHUB_REPO is not set (expected 'owner/repo')")
        self.repo = config.GITHUB_REPO

    def call(self, method: str, path: str, json_body=None, params=None):
        url = path if path.startswith("http") else f"{config.GITHUB_API}{path}"
        url = url.replace("{repo}", self.repo)
        r = requests.request(
            method, url,
            headers={
                "Authorization": f"Bearer {config.GITHUB_TOKEN}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            },
            json=json_body, params=params, timeout=60,
        )
        if r.status_code >= 400:
            payload = None
            try:
                payload = r.json() if r.content else None
            except Exception:
                payload = None
            if isinstance(payload, dict):
                msg = payload.get("message", r.text)
                # GitHub buries the actionable code (e.g. "already_exists") in
                # errors[].code — fold it into the message so handlers that match
                # on the message text behave the same against real and mock.
                errs = payload.get("errors") or []
                detail = "; ".join(
                    " ".join(str(e.get(k)) for k in ("resource", "field", "code") if e.get(k))
                    if isinstance(e, dict) else str(e)
                    for e in errs).strip()
                if detail:
                    msg = f"{msg} ({detail})"
            else:
                msg = r.text
            raise GitHubError(r.status_code, msg, payload)
        return r.json() if r.content else {}


# --------------------------------------------------------------------------
class MockTransport:
    """Minimal but constraint-faithful GitHub stand-in."""

    def __init__(self, seed: bool = True):
        self.repo = config.GITHUB_REPO or "demo/repo"
        self.issues: dict[int, dict] = {}
        self.labels: dict[str, dict] = {}
        self._next = 1
        if seed:
            self._seed()

    def _seed(self):
        for name in ("bug", "enhancement"):
            self.labels[name] = {"name": name, "color": "d73a4a"}
        seeds = [
            ("Login times out after 30 seconds", ["bug"], "octocat", 40),
            ("Add dark mode", ["enhancement"], None, 5),
            ("Crash on empty search", [], None, 2),
            ("Typo in onboarding copy", [], None, 20),
            ("Slow dashboard load", [], "octocat", 70),
            ("Export to CSV fails", [], None, 1),
        ]
        for title, labels, assignee, age_days in seeds:
            n = self._next
            self._next += 1
            self.issues[n] = {
                "number": n, "title": title, "body": "", "state": "open",
                "labels": [{"name": l} for l in labels],
                "assignee": ({"login": assignee} if assignee else None),
                "assignees": ([{"login": assignee}] if assignee else []),
                "created_at": time.strftime(
                    "%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() - age_days * 86400)),
                "comments": [],
            }

    def call(self, method: str, path: str, json_body=None, params=None):
        method = method.upper()
        # /repos/{repo}/labels
        if re.search(r"/labels/?$", path):
            if method == "GET":
                return list(self.labels.values())
            if method == "POST":
                name = json_body["name"]
                if name in self.labels:                       # the constraint we learn
                    raise GitHubError(422, f"Validation Failed: label {name} already_exists")
                self.labels[name] = {"name": name, "color": json_body.get("color", "ededed")}
                return self.labels[name]
        # /repos/{repo}/issues
        if re.search(r"/issues/?$", path):
            if method == "GET":
                items = [i for i in self.issues.values()
                         if i["state"] == (params or {}).get("state", "open")]
                return items
            if method == "POST":
                n = self._next
                self._next += 1
                issue = {
                    "number": n, "title": json_body.get("title", ""),
                    "body": json_body.get("body", ""), "state": "open",
                    "labels": [{"name": l} for l in json_body.get("labels", [])],
                    "assignee": None, "assignees": [], "comments": [],
                }
                self.issues[n] = issue
                return issue
        # /repos/{repo}/issues/{n}
        m = re.search(r"/issues/(\d+)/?$", path)
        if m:
            n = int(m.group(1))
            if n not in self.issues:
                raise GitHubError(404, "Not Found")
            if method == "GET":
                return self.issues[n]
            if method == "PATCH":
                self.issues[n].update({k: v for k, v in (json_body or {}).items()
                                       if k in ("title", "body", "state")})
                if "labels" in (json_body or {}):
                    self.issues[n]["labels"] = [{"name": l} for l in json_body["labels"]]
                return self.issues[n]
        # /repos/{repo}/issues/{n}/comments
        m = re.search(r"/issues/(\d+)/comments/?$", path)
        if m and method == "POST":
            n = int(m.group(1))
            if n not in self.issues:
                raise GitHubError(404, "Not Found")
            c = {"id": len(self.issues[n]["comments"]) + 1, "body": json_body.get("body", "")}
            self.issues[n]["comments"].append(c)
            return c
        # /repos/{repo}
        if re.search(r"/repos/[^/]+/[^/]+/?$", path) and method == "GET":
            return {"full_name": self.repo, "open_issues_count":
                    sum(1 for i in self.issues.values() if i["state"] == "open")}
        raise GitHubError(404, f"mock has no route for {method} {path}")


# --------------------------------------------------------------------------
class GitHubClient:
    def __init__(self, meter: Meter, transport=None):
        self.meter = meter
        if transport is not None:
            self.transport = transport
        elif config.GITHUB_MODE == "mock":
            self.transport = MockTransport()
        else:
            self.transport = RealTransport()
        self.repo = self.transport.repo

    def request(self, method: str, path: str, json_body=None, params=None):
        """The single primitive. Every capability calls through here."""
        self.meter.api()
        return self.transport.call(method, path, json_body, params)
