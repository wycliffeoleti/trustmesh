from __future__ import annotations

import json
import logging
import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, PlainTextResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from trustmesh.service import ControlPlane
from trustmesh.store import EventStore

logging.basicConfig(level=logging.INFO, format='{"time":"%(asctime)s","level":"%(levelname)s","message":"%(message)s"}')
log = logging.getLogger("trustmesh")
ROOT = Path(__file__).parent.parent
templates = Jinja2Templates(directory=str(ROOT / "templates"))


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    app.state.plane = ControlPlane(EventStore(os.getenv("TRUSTMESH_DB", "trustmesh.db")))
    yield


app = FastAPI(title="TrustMesh", version="0.1.0", description="Local agentic-AI control plane", lifespan=lifespan)


class TaskInput(BaseModel):
    task: str = Field(min_length=3, max_length=1000)


class ApprovalInput(BaseModel):
    reviewer: str = Field(default="demo-reviewer", min_length=2, max_length=100)


def plane(request: Request) -> ControlPlane:
    return request.app.state.plane  # type: ignore[no-any-return]


@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request) -> Any:
    return templates.TemplateResponse(request, "dashboard.html", plane(request).store.dashboard())


@app.get("/healthz")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/readyz")
def ready(request: Request) -> dict[str, str]:
    plane(request).store.connection.execute("SELECT 1")
    return {"status": "ready"}


@app.post("/api/runs", status_code=201)
def create_run(payload: TaskInput, request: Request) -> dict[str, Any]:
    result = plane(request).submit(payload.task)
    log.info(json.dumps({"event": "run.created", "run_id": result["id"], "trace_id": result["trace_id"]}))
    return result


@app.get("/api/runs/{run_id}")
def get_run(run_id: str, request: Request) -> dict[str, Any]:
    result = plane(request).store.get_run(run_id)
    if result is None:
        raise HTTPException(404, "Run not found")
    return result


@app.get("/api/dashboard")
def get_dashboard(request: Request) -> dict[str, Any]:
    return plane(request).store.dashboard()


@app.post("/api/approvals/{approval_id}/approve")
def approve(approval_id: str, payload: ApprovalInput, request: Request) -> dict[str, Any]:
    result = plane(request).resolve(approval_id, True, payload.reviewer)
    if result is None:
        raise HTTPException(409, "Approval is not pending")
    return result


@app.post("/api/approvals/{approval_id}/reject")
def reject(approval_id: str, payload: ApprovalInput, request: Request) -> dict[str, Any]:
    result = plane(request).resolve(approval_id, False, payload.reviewer)
    if result is None:
        raise HTTPException(409, "Approval is not pending")
    return result


@app.get("/metrics", response_class=PlainTextResponse)
def metrics(request: Request) -> str:
    values = plane(request).store.dashboard()["metrics"]
    return "\n".join([
        "# TYPE trustmesh_runs_total counter", f"trustmesh_runs_total {values['runs_total']}",
        "# TYPE trustmesh_pending_approvals gauge", f"trustmesh_pending_approvals {values['pending_approvals']}",
        "# TYPE trustmesh_blocked_actions_total counter", f"trustmesh_blocked_actions_total {values['blocked_actions']}",
        "",
    ])
