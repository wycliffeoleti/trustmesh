from trustmesh.domain import Action, PolicyGateway


def test_sensitive_action_requires_human_approval() -> None:
    decision = PolicyGateway().decide(Action(tool="send_email", arguments={"to": "ops@example.eu"}))
    assert decision.outcome == "require_approval"
    assert decision.risk == "high"


def test_forbidden_delete_is_blocked() -> None:
    decision = PolicyGateway().decide(Action(tool="delete_customer", arguments={"id": "c-1"}))
    assert decision.outcome == "deny"
