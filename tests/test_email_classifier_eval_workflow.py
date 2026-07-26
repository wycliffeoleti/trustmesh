"""Static validation of the dedicated email classifier evaluation GitHub Actions
workflow. This never executes the workflow (no `act`/network dependency here); it
parses the checked-in YAML and asserts the structural properties Phase 5 requires:
relevant-path triggers, the offline evaluator command, artifact upload, and the
explicit absence of any secret/credential/branch-protection configuration.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
WORKFLOW_PATH = ROOT / ".github" / "workflows" / "email-classifier-eval.yml"


def _load_workflow() -> dict[str, Any]:
    return yaml.safe_load(WORKFLOW_PATH.read_text())


def test_workflow_file_exists() -> None:
    assert WORKFLOW_PATH.exists()


def test_workflow_triggers_on_relevant_prompt_evaluator_and_data_paths() -> None:
    workflow = _load_workflow()
    # YAML parses the bare key `on:` as the boolean True.
    triggers = workflow[True]

    for event in ("push", "pull_request"):
        assert event in triggers
        paths = triggers[event]["paths"]
        assert any("prompts/email_classifier" in p for p in paths)
        assert any("evals/email_classifier_staged" in p for p in paths)
        assert any("evals/baselines" in p for p in paths)
        assert any("trustmesh/email_classifier" in p for p in paths)
        assert any("evals/run_email_classifier_eval.py" in p for p in paths)


def test_workflow_does_not_trigger_on_unrelated_paths_only() -> None:
    workflow = _load_workflow()
    triggers = workflow[True]
    for event in ("push", "pull_request"):
        paths = triggers[event]["paths"]
        assert not any(p in {"**", "*"} for p in paths)


def test_workflow_runs_the_documented_offline_evaluator_command() -> None:
    workflow = _load_workflow()
    steps = workflow["jobs"]["evaluate"]["steps"]
    run_commands = [step["run"] for step in steps if "run" in step]
    assert any("evals/run_email_classifier_eval.py" in cmd for cmd in run_commands)


def test_workflow_uploads_report_artifacts() -> None:
    workflow = _load_workflow()
    steps = workflow["jobs"]["evaluate"]["steps"]
    upload_steps = [step for step in steps if str(step.get("uses", "")).startswith(
        "actions/upload-artifact"
    )]
    assert upload_steps, "expected an actions/upload-artifact step"
    assert upload_steps[0].get("if") == "always()"


def test_workflow_contains_no_secrets_credentials_or_branch_protection_config() -> None:
    raw = WORKFLOW_PATH.read_text().lower()
    forbidden = [
        "secrets.",
        "slack",
        "webhook",
        "openai",
        "anthropic",
        "api_key",
        "required_status_checks",
        "branch_protection",
        "pull-requests: write",
        "issues: write",
    ]
    for term in forbidden:
        assert term not in raw, f"workflow unexpectedly references {term!r}"


def test_workflow_does_not_comment_on_pull_requests() -> None:
    workflow = _load_workflow()
    steps = workflow["jobs"]["evaluate"]["steps"]
    uses = [str(step.get("uses", "")) for step in steps]
    assert not any("github-script" in u or "comment" in u.lower() for u in uses)
