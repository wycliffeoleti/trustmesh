"""Regression tests written before fixing real bugs found in code review.

Each test documents the concrete defect it guards against.
"""
from __future__ import annotations

import json
import subprocess
import sqlite3
import sys
import tempfile
import threading
from pathlib import Path

import pytest

from trustmesh.service import ControlPlane
from trustmesh.store import EventStore

ROOT = Path(__file__).resolve().parent.parent


def make_plane(path: str) -> ControlPlane:
    return ControlPlane(EventStore(path))


def test_concurrent_approval_only_resolves_once() -> None:
    """Two threads racing to approve the same pending action must not both succeed.

    The original resolve_approval() did a SELECT to check status=='pending' and only
    later issued an UPDATE, leaving a window where two threads could both observe
    'pending' and both execute the sensitive action.
    """
    with tempfile.NamedTemporaryFile(suffix=".db") as file:
        plane = make_plane(file.name)
        run = plane.submit("Send an email update")
        approval_id = run["approvals"][0]["id"]

        results: list[dict | None] = []
        lock = threading.Lock()
        barrier = threading.Barrier(16)

        def attempt() -> None:
            barrier.wait()
            result = plane.resolve(approval_id, True, "racer")
            with lock:
                results.append(result)

        threads = [threading.Thread(target=attempt) for _ in range(16)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        successes = [r for r in results if r is not None]
        run_after = plane.store.get_run(run["id"])
        assert run_after is not None
        executed_events = [e for e in run_after["events"] if e["kind"] == "tool.executed" and e["payload"]["tool"] == "send_email"]
        assert len(executed_events) <= 1, "send_email must not execute more than once"
        assert len(successes) <= 1 or all(s["id"] == successes[0]["id"] for s in successes)


def test_approval_resumes_exact_proposed_action_arguments() -> None:
    """Approving a paused action must resume with the same arguments that were proposed,
    not just the tool name (arguments were previously discarded by add_approval).
    """
    with tempfile.NamedTemporaryFile(suffix=".db") as file:
        plane = make_plane(file.name)
        run = plane.submit("Send an email update")
        proposed_payload = next(
            e["payload"] for e in run["events"] if e["kind"] == "approval.requested"
        )
        approval_id = run["approvals"][0]["id"]

        resumed = plane.resolve(approval_id, True, "alice")
        assert resumed is not None
        executed = next(e for e in resumed["events"] if e["kind"] == "tool.executed" and e["payload"]["tool"] == "send_email")
        assert executed["payload"]["arguments"] == proposed_payload["arguments"]


def test_runs_total_metric_is_not_capped_by_display_limit() -> None:
    """dashboard()['metrics']['runs_total'] must reflect the true run count, not the
    LIMIT 20 used for the displayed recent-runs list.
    """
    with tempfile.NamedTemporaryFile(suffix=".db") as file:
        plane = make_plane(file.name)
        for _ in range(23):
            plane.submit("Summarise the incident runbook")
        metrics = plane.store.dashboard()["metrics"]
        assert metrics["runs_total"] == 23


def test_events_require_an_existing_run_via_foreign_key() -> None:
    """events.run_id and approvals.run_id must be enforced as foreign keys referencing
    runs.id so orphaned audit rows cannot be inserted.
    """
    with tempfile.NamedTemporaryFile(suffix=".db") as file:
        store = EventStore(file.name)
        with pytest.raises(sqlite3.IntegrityError):
            store.event("nonexistent-run-id", "run.started", {})


def test_documented_eval_command_runs_from_repo_root() -> None:
    """`uv run python evals/run_eval.py`, exactly as README/Makefile/CI invoke it, must
    succeed. A prior version only worked as `python -m evals.run_eval` because plain
    script invocation does not put the repo root on sys.path.
    """
    result = subprocess.run(
        [sys.executable, "evals/run_eval.py"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["passed"] == report["total"]


def test_documented_email_classifier_eval_command_runs_from_repo_root(tmp_path: Path) -> None:
    """`uv run python evals/run_email_classifier_eval.py`, exactly as README/Makefile/CI
    invoke it, must succeed from the repo root and print a structured headline status.
    """
    result = subprocess.run(
        [sys.executable, "evals/run_email_classifier_eval.py", "--output-dir", str(tmp_path)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    alert = json.loads(result.stdout)
    assert alert["dataset_status"] == "staged_draft"
    assert alert["headline_status"] in {"pass", "warn", "fail"}
    assert (tmp_path / "report.json").exists()
    assert (tmp_path / "report.html").exists()
