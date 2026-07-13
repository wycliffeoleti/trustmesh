from __future__ import annotations

import json
import re
import time
import uuid
from typing import Any

from trustmesh.domain import Action, PolicyGateway
from trustmesh.store import EventStore


def redact(value: str) -> str:
    return re.sub(r'(?i)(api[_-]?key|password|token|secret)\s*[:=]\s*("[^"]*"|\'[^\']*\'|\S+)', r"\1=[REDACTED]", value)


class ControlPlane:
    def __init__(self, store: EventStore, policy: PolicyGateway | None = None) -> None:
        self.store, self.policy = store, policy or PolicyGateway()

    def plan(self, task: str) -> list[Action]:
        task_l = task.lower()
        if "delete" in task_l:
            return [Action("delete_customer", {"id": "c-42"})]
        if "send" in task_l or "email" in task_l or "publish" in task_l:
            tool = "publish_update" if "publish" in task_l else "send_email"
            return [
                Action("search_knowledge", {"query": redact(task)}),
                Action(tool, {"to": "stakeholders@example.eu", "body": redact(task)}),
            ]
        return [Action("search_knowledge", {"query": redact(task)})]

    def submit(self, task: str) -> dict[str, Any]:
        run_id, trace_id = str(uuid.uuid4()), uuid.uuid4().hex
        self.store.create_run(run_id, redact(task), trace_id)
        self._execute(run_id, self.plan(task))
        return self.store.get_run(run_id) or {}

    def _execute(self, run_id: str, actions: list[Action]) -> None:
        initial = time.perf_counter()
        for action in actions:
            decision = self.policy.decide(action)
            payload = {
                "tool": action.tool,
                "arguments": action.arguments,
                "risk": decision.risk,
                "outcome": decision.outcome,
                "reason": decision.reason,
            }
            self.store.event(run_id, "policy.decided", payload)
            latency = int((time.perf_counter() - initial) * 1000)
            if decision.outcome == "deny":
                self.store.event(run_id, "policy.denied", payload)
                self.store.set_status(run_id, "blocked", latency=latency)
                return
            if decision.outcome == "require_approval":
                approval_id = str(uuid.uuid4())
                self.store.add_approval(approval_id, run_id, action.tool, action.arguments)
                self.store.event(run_id, "approval.requested", {"approval_id": approval_id, **payload})
                self.store.set_status(run_id, "awaiting_approval", latency=latency)
                return
            self.store.event(
                run_id, "tool.executed", {"tool": action.tool, "result": "Local knowledge search returned 3 approved records."}
            )
        self.store.event(run_id, "run.completed", {"success": True})
        self.store.set_status(run_id, "completed", cost=0.0002, latency=int((time.perf_counter() - initial) * 1000))

    def resolve(self, approval_id: str, approve: bool, reviewer: str) -> dict[str, Any] | None:
        approval = self.store.resolve_approval(approval_id, "approved" if approve else "rejected", reviewer)
        if approval is None:
            return None
        run_id = approval["run_id"]
        arguments = json.loads(approval["arguments"])
        self.store.event(
            run_id,
            "approval.resolved",
            {"approval_id": approval_id, "outcome": "approved" if approve else "rejected", "reviewer": reviewer},
        )
        if approve:
            self.store.event(
                run_id,
                "tool.executed",
                {
                    "tool": approval["action"],
                    "arguments": arguments,
                    "result": f"{approval['action']} executed after human approval.",
                },
            )
            self.store.event(run_id, "run.completed", {"success": True})
            self.store.set_status(run_id, "completed", cost=0.0005)
        else:
            self.store.set_status(run_id, "rejected")
        return self.store.get_run(run_id)
