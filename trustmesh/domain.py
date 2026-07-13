from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml

Outcome = Literal["allow", "deny", "require_approval"]


@dataclass(frozen=True)
class Action:
    tool: str
    arguments: dict[str, str]


@dataclass(frozen=True)
class Decision:
    outcome: Outcome
    risk: str
    reason: str
    scopes: tuple[str, ...] = ()


class PolicyGateway:
    """Deterministic least-privilege policy decision point."""

    def __init__(self, policy_path: Path | None = None) -> None:
        path = policy_path or Path(__file__).parent.parent / "config" / "policy.yaml"
        self.policy: dict[str, object] = yaml.safe_load(path.read_text())

    def decide(self, action: Action) -> Decision:
        if any(token in str(action.arguments).lower() for token in ("ignore previous", "system prompt", "api_key")):
            return Decision("deny", "critical", "Prompt-injection or secret-exfiltration marker detected.")
        tools = self.policy["tools"]
        rule = tools.get(action.tool) if isinstance(tools, dict) else None
        if not isinstance(rule, dict):
            return Decision("deny", "medium", f"Tool '{action.tool}' is not registered.")
        return Decision(
            rule["outcome"],
            str(rule["risk"]),
            str(rule.get("reason", f"Policy {rule['outcome']} for {action.tool}.")),
            tuple(rule.get("scopes", [])),
        )
