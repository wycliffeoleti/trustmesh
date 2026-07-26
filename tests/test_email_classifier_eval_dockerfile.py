"""Static validation of the dedicated evaluator Dockerfile. This never builds or runs
the image (no Docker dependency here); it asserts the checked-in Dockerfile text is a
deterministic, offline, credential-free container definition distinct from the
application Dockerfile, matching Phase 5's constraints.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EVALUATOR_DOCKERFILE = ROOT / "Dockerfile.evaluator"
APP_DOCKERFILE = ROOT / "Dockerfile"


def test_evaluator_dockerfile_exists_and_is_distinct_from_app_dockerfile() -> None:
    assert EVALUATOR_DOCKERFILE.exists()
    assert EVALUATOR_DOCKERFILE.read_text() != APP_DOCKERFILE.read_text()


def test_app_dockerfile_still_serves_the_uvicorn_application() -> None:
    text = APP_DOCKERFILE.read_text()
    assert "uvicorn" in text
    assert "trustmesh.app:app" in text


def test_evaluator_dockerfile_default_command_runs_the_offline_evaluator() -> None:
    text = EVALUATOR_DOCKERFILE.read_text()
    assert "evals/run_email_classifier_eval.py" in text
    assert "uvicorn" not in text


def test_evaluator_dockerfile_declares_no_environment_variables() -> None:
    lines = EVALUATOR_DOCKERFILE.read_text().splitlines()
    assert not any(line.strip().upper().startswith("ENV") for line in lines)


def test_evaluator_dockerfile_exposes_no_network_port() -> None:
    text = EVALUATOR_DOCKERFILE.read_text()
    assert "EXPOSE" not in text


def test_evaluator_dockerfile_references_no_credentials_or_paid_providers() -> None:
    lowered = EVALUATOR_DOCKERFILE.read_text().lower()
    forbidden = [
        "api_key", "apikey", "secret", "token", "openai", "anthropic", "slack", "webhook",
    ]
    for term in forbidden:
        assert term not in lowered, f"Dockerfile.evaluator unexpectedly references {term!r}"
