"""Central configuration. All knobs live here; ``.env`` provides their values.

We deliberately avoid python-dotenv: a 15-line loader keeps the dependency
list to two libraries (requests, rich), which is easier to justify in review.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv(path: Path) -> None:
    """Load ``path`` into the environment, overriding any existing values.

    The committed ``.env`` is the source of truth: a value in the file wins
    over a variable already set in the shell. This avoids the surprising case
    where a stale session variable (e.g. a leftover ``APIA_LLM_PROVIDER`` from
    an earlier experiment) silently shadows what ``.env`` plainly says.
    """
    if not path.exists():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip()
        if val[:1] in ("'", '"'):                 # quoted value: take what's inside the quotes
            q = val[0]
            end = val.find(q, 1)
            val = val[1:end] if end != -1 else val[1:]
        else:                                      # unquoted: drop a trailing ' # comment'
            val = val.split(" #", 1)[0].strip()
        os.environ[key] = val


_load_dotenv(ROOT / ".env")


def _get(key: str, default: str) -> str:
    return os.environ.get(key, default)


# --- LLM -------------------------------------------------------------------
LLM_PROVIDER = _get("APIA_LLM_PROVIDER", "anthropic")  # anthropic | openai | gemini | replay
LLM_MODEL = _get("APIA_LLM_MODEL", "claude-sonnet-4-6")
ANTHROPIC_API_KEY = _get("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY = _get("OPENAI_API_KEY", "")
ANTHROPIC_VERSION = _get("ANTHROPIC_VERSION", "2023-06-01")
GEMINI_API_KEY = _get("GEMINI_API_KEY", "")

# --- GitHub ----------------------------------------------------------------
def _normalize_repo(raw: str) -> str:
    """Coerce a repo identifier to the bare ``owner/repo`` the GitHub API wants.

    Accepts and strips common copy-paste forms: full ``https://github.com/...``
    or ``git@github.com:...`` URLs, a trailing ``.git``, surrounding quotes, and
    leading/trailing slashes or whitespace. A bare ``owner/repo`` passes through
    unchanged.
    """
    repo = raw.strip().strip("'\"").strip()
    if not repo:
        return ""
    # git@github.com:owner/repo(.git)
    if repo.startswith("git@"):
        repo = repo.split(":", 1)[-1]
    # any scheme://host/owner/repo  (http, https, ssh, git, ...)
    elif "://" in repo:
        repo = repo.split("://", 1)[1].split("/", 1)[-1] if "/" in repo.split("://", 1)[1] else ""
    repo = repo.strip("/")
    if repo.endswith(".git"):
        repo = repo[:-4]
    # keep only owner/repo even if extra path segments (e.g. /issues) were pasted
    parts = [p for p in repo.split("/") if p]
    return "/".join(parts[:2])


GITHUB_MODE = _get("APIA_GITHUB_MODE", "real")  # real | mock
GITHUB_TOKEN = _get("GITHUB_TOKEN", "")
GITHUB_REPO = _normalize_repo(_get("GITHUB_REPO", ""))  # always bare "owner/repo"
GITHUB_API = "https://api.github.com"

# --- Memory ----------------------------------------------------------------
DB_PATH = Path(_get("APIA_DB_PATH", str(ROOT / "apia_memory.db")))

# --- Behaviour knobs -------------------------------------------------------
MAX_SYNTH_ATTEMPTS = int(_get("APIA_MAX_SYNTH_ATTEMPTS", "3"))
MAX_STEP_RETRIES = int(_get("APIA_MAX_STEP_RETRIES", "2"))
# how many times a synthesised capability may be regenerated from its own
# RUNTIME error (the self-healing loop) before the step is declared failed
MAX_RUNTIME_REPAIRS = int(_get("APIA_MAX_RUNTIME_REPAIRS", "2"))
PROMOTE_AFTER = int(_get("APIA_PROMOTE_AFTER", "3"))   # successes to reach 'trusted'
DEPRECATE_AFTER = int(_get("APIA_DEPRECATE_AFTER", "2"))  # consecutive failures to deprecate
COMPACT_AFTER = int(_get("APIA_COMPACT_AFTER", "8"))  # detail rows per signature before compaction
MAX_API_CALLS_BUDGET = int(_get("APIA_MAX_API_CALLS", "40"))
STRICT_BUDGET = _get("APIA_STRICT_BUDGET", "0") == "1"
