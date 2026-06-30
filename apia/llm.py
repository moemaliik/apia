"""LLM access over plain HTTP — no vendor SDK, so swapping providers is an env
change, not a refactor, and there's no SDK version to defend in review.

Four providers:
  anthropic  -> POST https://api.anthropic.com/v1/messages
  openai     -> POST https://api.openai.com/v1/chat/completions
  gemini     -> POST https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent
  replay     -> returns pre-recorded responses keyed by `cache_key`

`replay` exists so the learning curve is reproducible on the walkthrough call
and so the test-suite runs with zero API spend. Real providers ignore cache_key.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Optional

import requests

from . import config
from .metrics import Meter

_REPLAY_FILE = Path(__file__).resolve().parent / "replay" / "responses.json"


class LLMError(RuntimeError):
    pass


class LLM:
    def __init__(self, meter: Meter, provider: str | None = None, model: str | None = None):
        self.meter = meter
        self.provider = provider or config.LLM_PROVIDER
        self.model = model or config.LLM_MODEL
        self._replay = None
        if self.provider == "replay":
            self._replay = json.loads(_REPLAY_FILE.read_text()) if _REPLAY_FILE.exists() else {}

    def complete(self, system: str, user: str, *, cache_key: Optional[str] = None,
                 max_tokens: int = 1500, temperature: float = 0.0) -> str:
        self.meter.llm()
        if self.provider == "replay":
            if cache_key is None or cache_key not in self._replay:
                raise LLMError(f"replay: no recorded response for cache_key={cache_key!r}")
            resp = self._replay[cache_key]
            self.meter.tokens += len(resp) // 4
            return resp
        if self.provider == "anthropic":
            return self._anthropic(system, user, max_tokens, temperature)
        if self.provider == "openai":
            return self._openai(system, user, max_tokens, temperature)
        if self.provider == "gemini":
            return self._gemini(system, user, max_tokens, temperature)
        raise LLMError(f"unknown provider {self.provider!r}")

    def complete_json(self, system: str, user: str, *, cache_key: Optional[str] = None,
                      max_tokens: int = 1500, temperature: float = 0.0,
                      repair_attempts: int = 2) -> dict | list:
        sys = system + "\n\nRespond with ONLY valid JSON, no prose, no code fences."
        raw = self.complete(sys, user, cache_key=cache_key,
                            max_tokens=max_tokens, temperature=temperature)
        try:
            return _parse_json(raw)
        except (json.JSONDecodeError, ValueError) as err:
            # Even a capable model occasionally emits invalid JSON (an unescaped
            # quote or a raw newline inside a long string). Rather than crash the
            # whole run, show the model its own broken output plus the parser
            # error and ask for strict JSON. This recovers the common case; if it
            # still can't, we raise LLMError so callers (e.g. agent.run) can fail
            # gracefully instead of surfacing a raw JSONDecodeError traceback.
            for _ in range(repair_attempts):
                repair = (f"{user}\n\nYour previous reply was NOT valid JSON and failed to "
                          f"parse with: {err}\n--- your reply ---\n{raw}\n--- end ---\n"
                          "Return the SAME content as STRICTLY valid JSON only. Escape every "
                          "quote and newline inside string values. No prose, no code fences.")
                raw = self.complete(sys, repair, cache_key=None,
                                    max_tokens=max_tokens, temperature=temperature)
                try:
                    return _parse_json(raw)
                except (json.JSONDecodeError, ValueError) as e2:
                    err = e2
            raise LLMError(f"model did not return valid JSON after {repair_attempts} "
                           f"repair attempt(s): {err}")

    # -- providers ---------------------------------------------------------
    def _anthropic(self, system, user, max_tokens, temperature) -> str:
        if not config.ANTHROPIC_API_KEY:
            raise LLMError("ANTHROPIC_API_KEY is not set")
        r = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": config.ANTHROPIC_API_KEY,
                "anthropic-version": config.ANTHROPIC_VERSION,
                "content-type": "application/json",
            },
            json={
                "model": self.model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "system": system,
                "messages": [{"role": "user", "content": user}],
            },
            timeout=120,
        )
        if r.status_code >= 400:
            raise LLMError(f"anthropic {r.status_code}: {r.text[:300]}")
        data = r.json()
        usage = data.get("usage", {})
        self.meter.tokens += usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
        return "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")

    def _openai(self, system, user, max_tokens, temperature) -> str:
        if not config.OPENAI_API_KEY:
            raise LLMError("OPENAI_API_KEY is not set")
        r = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {config.OPENAI_API_KEY}",
                     "content-type": "application/json"},
            json={
                "model": self.model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": [{"role": "system", "content": system},
                             {"role": "user", "content": user}],
            },
            timeout=120,
        )
        if r.status_code >= 400:
            raise LLMError(f"openai {r.status_code}: {r.text[:300]}")
        data = r.json()
        self.meter.tokens += data.get("usage", {}).get("total_tokens", 0)
        return data["choices"][0]["message"]["content"]

    def _gemini(self, system, user, max_tokens, temperature) -> str:
        if not config.GEMINI_API_KEY:
            raise LLMError("GEMINI_API_KEY is not set")
        # Retry transient errors (429/500/503) with exponential backoff.
        for attempt in range(4):
            try:
                r = requests.post(
                    f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent",
                    headers={"content-type": "application/json"},
                    params={"key": config.GEMINI_API_KEY},
                    json={
                        "systemInstruction": {"parts": [{"text": system}]},
                        "contents": [{"role": "user", "parts": [{"text": user}]}],
                        "generationConfig": {"maxOutputTokens": max_tokens,
                                             "temperature": temperature},
                    },
                    timeout=(10, 60),
                )
            except requests.exceptions.RequestException as e:
                raise LLMError(f"gemini network error: {e}") from None
            if r.status_code in (429, 500, 503) and attempt < 3:
                time.sleep(2 ** attempt)
                continue
            break
        if r.status_code >= 400:
            raise LLMError(f"gemini {r.status_code}: {r.text[:300]}")
        data = r.json()
        usage = data.get("usageMetadata", {})
        self.meter.tokens += (usage.get("promptTokenCount", 0)
                              + usage.get("candidatesTokenCount", 0))
        cands = data.get("candidates", [])
        if not cands:
            raise LLMError(f"gemini: no candidates returned ({data})")
        parts = cands[0].get("content", {}).get("parts", [])
        return "".join(p.get("text", "") for p in parts)


def _parse_json(raw: str) -> dict | list:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```", 2)[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start = min((raw.find("{") if "{" in raw else len(raw)),
                    (raw.find("[") if "[" in raw else len(raw)))
        end = max(raw.rfind("}"), raw.rfind("]"))
        if start >= end:
            raise
        candidate = raw[start:end + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            # Cheap local repair for the most common LLM slip — a trailing comma
            # before a closing ] or } — so we don't burn an LLM round-trip on it.
            return json.loads(re.sub(r",(\s*[}\]])", r"\1", candidate))