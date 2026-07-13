from __future__ import annotations

import os
import tempfile
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from trustmesh.app import app


@pytest.fixture
def client() -> Iterator[TestClient]:
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as file:
        db_path = file.name
    os.environ["TRUSTMESH_DB"] = db_path
    with TestClient(app) as test_client:
        yield test_client
    os.environ.pop("TRUSTMESH_DB", None)
    for path in (db_path, f"{db_path}-shm", f"{db_path}-wal"):
        if os.path.exists(path):
            os.unlink(path)


def test_healthz(client: TestClient) -> None:
    assert client.get("/healthz").json() == {"status": "ok"}


def test_readyz(client: TestClient) -> None:
    assert client.get("/readyz").json() == {"status": "ready"}


def test_dashboard_html_renders(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "TrustMesh" in response.text


def test_create_run_completes_for_safe_task(client: TestClient) -> None:
    response = client.post("/api/runs", json={"task": "Summarise the incident runbook"})
    assert response.status_code == 201
    body = response.json()
    assert body["status"] == "completed"
    assert "tool.executed" in [e["kind"] for e in body["events"]]


def test_task_input_validation_rejects_short_task(client: TestClient) -> None:
    response = client.post("/api/runs", json={"task": "hi"})
    assert response.status_code == 422


def test_get_run_not_found(client: TestClient) -> None:
    response = client.get("/api/runs/does-not-exist")
    assert response.status_code == 404


def test_approval_flow_approve_then_reject_is_conflict(client: TestClient) -> None:
    created = client.post("/api/runs", json={"task": "Send an email update"}).json()
    assert created["status"] == "awaiting_approval"
    approval_id = created["approvals"][0]["id"]

    approved = client.post(f"/api/approvals/{approval_id}/approve", json={"reviewer": "alice"})
    assert approved.status_code == 200
    assert approved.json()["status"] == "completed"

    replay = client.post(f"/api/approvals/{approval_id}/approve", json={"reviewer": "mallory"})
    assert replay.status_code == 409


def test_approval_flow_reject(client: TestClient) -> None:
    created = client.post("/api/runs", json={"task": "Publish an update"}).json()
    approval_id = created["approvals"][0]["id"]
    rejected = client.post(f"/api/approvals/{approval_id}/reject", json={"reviewer": "bob"})
    assert rejected.status_code == 200
    assert rejected.json()["status"] == "rejected"


def test_delete_customer_is_blocked(client: TestClient) -> None:
    response = client.post("/api/runs", json={"task": "Delete customer 42"})
    assert response.json()["status"] == "blocked"


def test_dashboard_json_endpoint(client: TestClient) -> None:
    client.post("/api/runs", json={"task": "Summarise the incident runbook"})
    body = client.get("/api/dashboard").json()
    assert body["metrics"]["runs_total"] == 1


def test_metrics_endpoint_format(client: TestClient) -> None:
    client.post("/api/runs", json={"task": "Summarise the incident runbook"})
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "trustmesh_runs_total 1" in response.text
    assert "# TYPE trustmesh_blocked_actions_total counter" in response.text
