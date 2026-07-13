from __future__ import annotations

import json
import sqlite3
import threading
from datetime import UTC, datetime
from typing import Any, cast


def now() -> str:
    return datetime.now(UTC).isoformat()


class EventStore:
    """Append-only audit store; updates are represented by additional events.

    ``runs`` and ``approvals`` hold mutable current-state rows (status, resolution);
    only the ``events`` table is append-only and is never updated or deleted.
    """

    def __init__(self, database: str = "trustmesh.db") -> None:
        self.connection = sqlite3.connect(database, check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA foreign_keys=ON")
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS runs (id TEXT PRIMARY KEY, task TEXT NOT NULL, status TEXT NOT NULL,
              trace_id TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
              cost_usd REAL NOT NULL DEFAULT 0, latency_ms INTEGER NOT NULL DEFAULT 0);
            CREATE TABLE IF NOT EXISTS events (id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT NOT NULL
              REFERENCES runs(id), at TEXT NOT NULL, kind TEXT NOT NULL, payload TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS approvals (id TEXT PRIMARY KEY, run_id TEXT NOT NULL REFERENCES runs(id),
              action TEXT NOT NULL, arguments TEXT NOT NULL, status TEXT NOT NULL, created_at TEXT NOT NULL,
              resolved_at TEXT, reviewer TEXT);
            """
        )
        self.connection.commit()

    def close(self) -> None:
        with self._lock:
            self.connection.close()

    def create_run(self, run_id: str, task: str, trace_id: str) -> None:
        with self._lock:
            self.connection.execute(
                "INSERT INTO runs VALUES (?, ?, 'running', ?, ?, ?, 0, 0)", (run_id, task, trace_id, now(), now())
            )
            self._event_locked(run_id, "run.started", {"task": task, "trace_id": trace_id})
            self.connection.commit()

    def event(self, run_id: str, kind: str, payload: dict[str, Any]) -> None:
        with self._lock:
            self._event_locked(run_id, kind, payload)
            self.connection.commit()

    def _event_locked(self, run_id: str, kind: str, payload: dict[str, Any]) -> None:
        self.connection.execute(
            "INSERT INTO events(run_id, at, kind, payload) VALUES (?, ?, ?, ?)",
            (run_id, now(), kind, json.dumps(payload, sort_keys=True)),
        )

    def set_status(self, run_id: str, status: str, *, cost: float = 0, latency: int = 0) -> None:
        with self._lock:
            self.connection.execute(
                "UPDATE runs SET status=?, updated_at=?, cost_usd=cost_usd+?, latency_ms=latency_ms+? WHERE id=?",
                (status, now(), cost, latency, run_id),
            )
            self.connection.commit()

    def add_approval(self, approval_id: str, run_id: str, action: str, arguments: dict[str, str]) -> None:
        with self._lock:
            self.connection.execute(
                "INSERT INTO approvals VALUES (?, ?, ?, ?, 'pending', ?, NULL, NULL)",
                (approval_id, run_id, action, json.dumps(arguments, sort_keys=True), now()),
            )
            self.connection.commit()

    def resolve_approval(self, approval_id: str, status: str, reviewer: str) -> sqlite3.Row | None:
        """Atomically resolve a pending approval.

        The status transition and the pending-check happen in a single UPDATE so two
        concurrent callers cannot both observe 'pending' and both proceed (a prior
        SELECT-then-UPDATE implementation had that race).
        """
        with self._lock:
            cursor = self.connection.execute(
                "UPDATE approvals SET status=?, resolved_at=?, reviewer=? WHERE id=? AND status='pending'",
                (status, now(), reviewer, approval_id),
            )
            if cursor.rowcount == 0:
                self.connection.rollback()
                return None
            row = self.connection.execute("SELECT * FROM approvals WHERE id=?", (approval_id,)).fetchone()
            self.connection.commit()
            return cast("sqlite3.Row | None", row)

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self.connection.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
            if row is None:
                return None
            data = dict(row)
            data["events"] = [
                {**dict(event), "payload": json.loads(event["payload"])}
                for event in self.connection.execute("SELECT * FROM events WHERE run_id=? ORDER BY id", (run_id,))
            ]
            data["approvals"] = [dict(x) for x in self.connection.execute("SELECT * FROM approvals WHERE run_id=?", (run_id,))]
            return data

    def dashboard(self) -> dict[str, Any]:
        with self._lock:
            runs = [dict(x) for x in self.connection.execute("SELECT * FROM runs ORDER BY created_at DESC LIMIT 20")]
            pending = [dict(x) for x in self.connection.execute("SELECT * FROM approvals WHERE status='pending'")]
            runs_total = self.connection.execute("SELECT count(*) FROM runs").fetchone()[0]
            blocked_actions = self.connection.execute("SELECT count(*) FROM events WHERE kind='policy.denied'").fetchone()[0]
            return {
                "runs": runs,
                "pending": pending,
                "metrics": {"runs_total": runs_total, "pending_approvals": len(pending), "blocked_actions": blocked_actions},
            }
