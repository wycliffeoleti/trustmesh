"""Inbound/outbound contracts: concrete adapters may be swapped without changing policy logic."""
from __future__ import annotations

from typing import Any, Protocol

from trustmesh.domain import Action, Decision


class ModelProvider(Protocol):
    def plan(self, task: str) -> list[Action]: ...


class Tool(Protocol):
    name: str
    def execute(self, arguments: dict[str, str]) -> dict[str, Any]: ...


class PolicyEngine(Protocol):
    def decide(self, action: Action) -> Decision: ...


class AuditEventStore(Protocol):
    def event(self, run_id: str, kind: str, payload: dict[str, Any]) -> None: ...


class Evaluator(Protocol):
    def evaluate(self, task: str, expected_status: str) -> dict[str, Any]: ...
