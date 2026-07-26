"""Static validation of the README's email classifier evaluation onboarding section.
This never renders or publishes the README; it asserts the checked-in text covers the
concrete points Phase 5 requires (purpose, local command, provenance warning, baseline
change procedure, thresholds, and the slow-drift rationale) using exact, honest
language rather than a production/paid-provider claim.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
README = ROOT / "README.md"


def _readme_text() -> str:
    return README.read_text()


def test_readme_documents_the_local_offline_evaluator_command() -> None:
    text = _readme_text()
    assert "uv run python evals/run_email_classifier_eval.py" in text


def test_readme_states_purpose_as_offline_deterministic_regression_check() -> None:
    text = _readme_text().lower()
    assert "email classifier" in text
    assert "baseline" in text
    assert "offline" in text
    assert "no paid" in text or "no model/provider" in text or "makes no model" in text


def test_readme_carries_staged_draft_not_golden_provenance_warning() -> None:
    text = _readme_text()
    assert "staged draft" in text.lower() or "staged_draft" in text
    assert "not golden" in text.lower() or "not a golden dataset" in text.lower()
    assert "not production" in text.lower() or "not production evidence" in text.lower()


def test_readme_documents_baseline_change_procedure() -> None:
    text = _readme_text()
    assert "evals/baselines/email_classifier_staged_v1.json" in text
    lowered = text.lower()
    assert "review" in lowered
    assert "regenerat" in lowered  # covers "regenerate"/"regenerating"


def test_readme_documents_warning_and_critical_thresholds() -> None:
    text = _readme_text()
    assert "3" in text and "8" in text
    lowered = text.lower()
    assert "warning" in lowered and "critical" in lowered


def test_readme_explains_why_slow_drift_exists() -> None:
    lowered = _readme_text().lower()
    assert "drift" in lowered
    assert "seven" in lowered or "7-run" in lowered or "moving average" in lowered


def test_readme_does_not_claim_production_or_paid_provider_use() -> None:
    lowered = _readme_text().lower()
    forbidden = ["openai", "paid provider", "slack webhook", "production traffic"]
    for term in forbidden:
        assert term not in lowered, f"README unexpectedly claims {term!r}"
