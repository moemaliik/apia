"""Persistent, structured memory backed by SQLite (stdlib).

This is intentionally NOT a vector store of past prompts. It stores *structured
knowledge* in normalised tables and is queried at decision time:

  EXECUTION LAYER   instructions, plans, steps, run_metrics, lessons
  CAPABILITY LAYER  capabilities, capability_runs, operation_stats, constraints

The planner queries `plans` to reuse a decomposition; the executor queries
`constraints` to skip doomed calls; the synthesiser queries `capabilities` to
reuse code; capability selection prefers `trusted` over `experimental`. Memory
changing behaviour is the whole point, so every read path is exercised below.
"""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Optional

from .. import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS instructions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    text TEXT NOT NULL,
    signature TEXT NOT NULL,
    status TEXT,
    created_at REAL,
    api_calls INTEGER DEFAULT 0,
    llm_calls INTEGER DEFAULT 0,
    tokens INTEGER DEFAULT 0,
    wall_ms INTEGER DEFAULT 0,
    failures INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS plans (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    signature TEXT NOT NULL,
    instruction_id INTEGER,
    steps_json TEXT NOT NULL,
    succeeded INTEGER DEFAULT 0,
    created_at REAL
);
CREATE TABLE IF NOT EXISTS steps (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    instruction_id INTEGER,
    idx INTEGER,
    description TEXT,
    capability TEXT,
    status TEXT,
    error TEXT,
    attempts INTEGER,
    api_calls INTEGER,
    wall_ms INTEGER
);
CREATE TABLE IF NOT EXISTS lessons (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    instruction_id INTEGER,
    signature TEXT,
    kind TEXT,
    text TEXT,
    created_at REAL
);
CREATE TABLE IF NOT EXISTS capabilities (
    name TEXT PRIMARY KEY,
    spec TEXT,
    source_code TEXT,
    status TEXT,                 -- experimental | trusted | deprecated
    version INTEGER DEFAULT 1,
    success_count INTEGER DEFAULT 0,
    failure_count INTEGER DEFAULT 0,
    consecutive_failures INTEGER DEFAULT 0,
    provenance_json TEXT,
    created_at REAL,
    last_used_at REAL
);
CREATE TABLE IF NOT EXISTS capability_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    capability TEXT,
    instruction_id INTEGER,
    success INTEGER,
    wall_ms INTEGER,
    error TEXT,
    created_at REAL
);
CREATE TABLE IF NOT EXISTS operation_stats (
    op TEXT PRIMARY KEY,
    attempts INTEGER DEFAULT 0,
    successes INTEGER DEFAULT 0,
    failures INTEGER DEFAULT 0,
    total_wall_ms INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS constraints (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    op TEXT,
    rule_json TEXT,
    evidence TEXT,
    hits INTEGER DEFAULT 0,
    created_at REAL
);
CREATE TABLE IF NOT EXISTS run_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    signature TEXT,
    run_number INTEGER,
    api_calls INTEGER,
    llm_calls INTEGER,
    tokens INTEGER,
    wall_ms INTEGER,
    failures INTEGER,
    plan_source TEXT,
    created_at REAL
);
CREATE TABLE IF NOT EXISTS undo_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    instruction_id INTEGER,
    inverse_json TEXT,
    applied INTEGER DEFAULT 0,
    created_at REAL
);
"""


class Memory:
    def __init__(self, path: Path | str = config.DB_PATH):
        self.path = str(path)
        self.db = sqlite3.connect(self.path)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)
        self.db.commit()

    # ---- execution layer -------------------------------------------------
    def start_instruction(self, text: str, signature: str) -> int:
        cur = self.db.execute(
            "INSERT INTO instructions (text, signature, status, created_at) VALUES (?,?,?,?)",
            (text, signature, "running", time.time()),
        )
        self.db.commit()
        return cur.lastrowid

    def finish_instruction(self, iid: int, status: str, m: dict) -> None:
        self.db.execute(
            "UPDATE instructions SET status=?, api_calls=?, llm_calls=?, tokens=?, "
            "wall_ms=?, failures=? WHERE id=?",
            (status, m["api_calls"], m["llm_calls"], m["tokens"],
             m["wall_ms"], m["failures"], iid),
        )
        self.db.commit()

    def find_reusable_plan(self, signature: str) -> Optional[list[dict]]:
        """Return the most recent *successful* decomposition for this signature."""
        row = self.db.execute(
            "SELECT steps_json FROM plans WHERE signature=? AND succeeded=1 "
            "ORDER BY id DESC LIMIT 1",
            (signature,),
        ).fetchone()
        return json.loads(row["steps_json"]) if row else None

    def save_plan(self, signature: str, iid: int, steps: list[dict], succeeded: bool) -> None:
        self.db.execute(
            "INSERT INTO plans (signature, instruction_id, steps_json, succeeded, created_at) "
            "VALUES (?,?,?,?,?)",
            (signature, iid, json.dumps(steps), int(succeeded), time.time()),
        )
        self.db.commit()

    def save_step(self, iid: int, s: dict) -> None:
        self.db.execute(
            "INSERT INTO steps (instruction_id, idx, description, capability, status, "
            "error, attempts, api_calls, wall_ms) VALUES (?,?,?,?,?,?,?,?,?)",
            (iid, s["idx"], s["description"], s["capability"], s["status"],
             s.get("error"), s["attempts"], s["api_calls"], s["wall_ms"]),
        )
        self.db.commit()

    def add_lesson(self, iid: int, signature: str, kind: str, text: str) -> None:
        self.db.execute(
            "INSERT INTO lessons (instruction_id, signature, kind, text, created_at) "
            "VALUES (?,?,?,?,?)",
            (iid, signature, kind, text, time.time()),
        )
        self.db.commit()

    def lessons_for(self, signature: str, limit: int = 5) -> list[str]:
        rows = self.db.execute(
            "SELECT text FROM lessons WHERE signature=? ORDER BY id DESC LIMIT ?",
            (signature, limit),
        ).fetchall()
        return [r["text"] for r in rows]

    # ---- capability layer ------------------------------------------------
    def get_capability(self, name: str) -> Optional[dict]:
        row = self.db.execute("SELECT * FROM capabilities WHERE name=?", (name,)).fetchone()
        return dict(row) if row else None

    def save_capability(self, name: str, spec: str, source_code: str, provenance: dict) -> None:
        self.db.execute(
            "INSERT OR REPLACE INTO capabilities (name, spec, source_code, status, version, "
            "success_count, failure_count, consecutive_failures, provenance_json, created_at, "
            "last_used_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (name, spec, source_code, "experimental", 1, 0, 0, 0,
             json.dumps(provenance), time.time(), time.time()),
        )
        self.db.commit()

    def bump_capability(self, name: str, success: bool) -> str:
        """Update counters and apply the lifecycle policy. Returns the new status."""
        cap = self.get_capability(name)
        if not cap:
            return "unknown"
        sc, fc, cf, status = (cap["success_count"], cap["failure_count"],
                              cap["consecutive_failures"], cap["status"])
        if success:
            sc += 1
            cf = 0
            if status == "experimental" and sc >= config.PROMOTE_AFTER:
                status = "trusted"
            elif status == "deprecated":
                status = "experimental"  # earned its way back
        else:
            fc += 1
            cf += 1
            if cf >= config.DEPRECATE_AFTER:
                status = "deprecated"
        self.db.execute(
            "UPDATE capabilities SET success_count=?, failure_count=?, consecutive_failures=?, "
            "status=?, last_used_at=? WHERE name=?",
            (sc, fc, cf, status, time.time(), name),
        )
        self.db.commit()
        return status

    def record_capability_run(self, name: str, iid: int, success: bool,
                              wall_ms: int, error: str | None) -> None:
        self.db.execute(
            "INSERT INTO capability_runs (capability, instruction_id, success, wall_ms, error, "
            "created_at) VALUES (?,?,?,?,?,?)",
            (name, iid, int(success), wall_ms, error, time.time()),
        )
        self.db.commit()

    def list_capabilities(self) -> list[dict]:
        return [dict(r) for r in self.db.execute(
            "SELECT * FROM capabilities ORDER BY status, name").fetchall()]

    def record_operation(self, op: str, success: bool, wall_ms: int) -> None:
        self.db.execute("INSERT OR IGNORE INTO operation_stats (op) VALUES (?)", (op,))
        self.db.execute(
            "UPDATE operation_stats SET attempts=attempts+1, successes=successes+?, "
            "failures=failures+?, total_wall_ms=total_wall_ms+? WHERE op=?",
            (int(success), int(not success), wall_ms, op),
        )
        self.db.commit()

    def operation_success_rate(self, op: str) -> Optional[float]:
        row = self.db.execute("SELECT * FROM operation_stats WHERE op=?", (op,)).fetchone()
        if not row or row["attempts"] == 0:
            return None
        return row["successes"] / row["attempts"]

    # ---- constraint ledger ----------------------------------------------
    def add_constraint(self, op: str, rule: dict, evidence: str) -> bool:
        """Record a learned constraint. Returns True if it was new."""
        existing = self.db.execute(
            "SELECT id FROM constraints WHERE op=? AND rule_json=?",
            (op, json.dumps(rule, sort_keys=True)),
        ).fetchone()
        if existing:
            self.db.execute("UPDATE constraints SET hits=hits+1 WHERE id=?", (existing["id"],))
            self.db.commit()
            return False
        self.db.execute(
            "INSERT INTO constraints (op, rule_json, evidence, hits, created_at) VALUES (?,?,?,?,?)",
            (op, json.dumps(rule, sort_keys=True), evidence, 1, time.time()),
        )
        self.db.commit()
        return True

    def constraints_for(self, op: str) -> list[dict]:
        rows = self.db.execute("SELECT * FROM constraints WHERE op=?", (op,)).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["rule"] = json.loads(d["rule_json"])
            out.append(d)
        return out

    # ---- run metrics & the learning curve --------------------------------
    def next_run_number(self, signature: str) -> int:
        row = self.db.execute(
            "SELECT COUNT(*) AS c FROM run_metrics WHERE signature=?", (signature,)).fetchone()
        return row["c"] + 1

    def record_run_metric(self, signature: str, run_number: int, m: dict, plan_source: str) -> None:
        self.db.execute(
            "INSERT INTO run_metrics (signature, run_number, api_calls, llm_calls, tokens, "
            "wall_ms, failures, plan_source, created_at) VALUES (?,?,?,?,?,?,?,?,?)",
            (signature, run_number, m["api_calls"], m["llm_calls"], m["tokens"],
             m["wall_ms"], m["failures"], plan_source, time.time()),
        )
        self.db.commit()

    def runs_for(self, signature: str) -> list[dict]:
        return [dict(r) for r in self.db.execute(
            "SELECT * FROM run_metrics WHERE signature=? ORDER BY run_number",
            (signature,)).fetchall()]

    def signatures(self) -> list[str]:
        rows = self.db.execute(
            "SELECT DISTINCT signature FROM run_metrics ORDER BY signature").fetchall()
        return [r["signature"] for r in rows]

    # ---- rollback --------------------------------------------------------
    def push_undo(self, iid: int, inverse: dict) -> None:
        self.db.execute(
            "INSERT INTO undo_log (instruction_id, inverse_json, created_at) VALUES (?,?,?)",
            (iid, json.dumps(inverse), time.time()),
        )
        self.db.commit()

    def undos_for(self, iid: int) -> list[dict]:
        rows = self.db.execute(
            "SELECT * FROM undo_log WHERE instruction_id=? AND applied=0 ORDER BY id DESC",
            (iid,)).fetchall()
        return [dict(r) for r in rows]

    def mark_undo_applied(self, undo_id: int) -> None:
        self.db.execute("UPDATE undo_log SET applied=1 WHERE id=?", (undo_id,))
        self.db.commit()

    # ---- compaction ------------------------------------------------------
    def step_detail_count(self, signature: str) -> int:
        row = self.db.execute(
            "SELECT COUNT(*) AS c FROM steps s JOIN instructions i ON s.instruction_id=i.id "
            "WHERE i.signature=?", (signature,)).fetchone()
        return row["c"]

    def compact_signature(self, signature: str, summary: str) -> int:
        """Replace fine-grained step rows for a signature with one summary lesson.
        Run-level metrics (the learning curve) are preserved untouched."""
        ids = [r["id"] for r in self.db.execute(
            "SELECT id FROM instructions WHERE signature=?", (signature,)).fetchall()]
        if not ids:
            return 0
        qmarks = ",".join("?" * len(ids))
        deleted = self.db.execute(
            f"DELETE FROM steps WHERE instruction_id IN ({qmarks})", ids).rowcount
        self.add_lesson(ids[-1], signature, "compaction", summary)
        self.db.commit()
        return deleted

    def close(self) -> None:
        self.db.close()
