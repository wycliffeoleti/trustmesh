from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

import pytest

from trustmesh.email_classifier import EmailCategory
from trustmesh.email_classifier_eval_alert import (
    EmailClassifierEvalAlertPayload,
    build_email_classifier_eval_alert_payload,
)
from trustmesh.email_classifier_eval_comparison import (
    CategoryAccuracyDelta,
    EmailClassifierRunComparison,
    RegressionThresholds,
)
from trustmesh.email_classifier_eval_engine import (
    UNSUPPORTED_OFFLINE,
    EmailClassifierCaseResult,
    EmailClassifierRunResult,
)
from trustmesh.email_classifier_staged_dataset import StagedDatasetProvenance

PROVENANCE = StagedDatasetProvenance(
    authored_by="assistant",
    reviewed_by=None,
    statement=(
        "This dataset was authored entirely by an AI assistant as synthetic seed data. "
        "It has not been reviewed by a human and must not be treated as a golden dataset "
        "until a human reviewer completes review."
    ),
)


def make_case_result(case_id: str, category: EmailCategory, match: bool) -> EmailClassifierCaseResult:
    return EmailClassifierCaseResult(
        case_id=case_id,
        expected_category=category,
        actual_category=category if match else "general",
        category_match=match,
        actual_summary="Synthetic summary.",
        summary_relevance=UNSUPPORTED_OFFLINE,
        token_usage=UNSUPPORTED_OFFLINE,
        latency_ms=1.0,
    )


def make_run_result(case_results: list[EmailClassifierCaseResult]) -> EmailClassifierRunResult:
    match_count = sum(r.category_match for r in case_results)
    return EmailClassifierRunResult(
        prompt_version="v1",
        prompt_created_at=datetime(2026, 1, 1, tzinfo=UTC),
        dataset_schema_version="1",
        dataset_status="staged_draft",
        dataset_provenance=PROVENANCE,
        case_results=case_results,
        category_pass_rate=match_count / len(case_results),
        summary_relevance=UNSUPPORTED_OFFLINE,
        token_usage=UNSUPPORTED_OFFLINE,
    )


def make_comparison(
    status: Literal["ok", "warning", "critical"],
    regressed_case_ids: list[str],
    improved_case_ids: list[str],
) -> EmailClassifierRunComparison:
    return EmailClassifierRunComparison(
        previous_pass_rate=0.9,
        current_pass_rate=0.8,
        overall_delta_percentage_points=-10.0,
        per_category_deltas=[
            CategoryAccuracyDelta(
                category="billing",
                previous_accuracy=0.9,
                current_accuracy=0.8,
                delta_percentage_points=-10.0,
            )
        ],
        regressed_case_ids=regressed_case_ids,
        improved_case_ids=improved_case_ids,
        thresholds=RegressionThresholds(),
        status=status,
    )


# --- Cycle 1: headline status mapping and headline numbers ---


@pytest.mark.parametrize(
    ("comparison_status", "expected_headline"),
    [("ok", "pass"), ("warning", "warn"), ("critical", "fail")],
)
def test_build_alert_payload_maps_comparison_status_to_headline_status(
    comparison_status: Literal["ok", "warning", "critical"], expected_headline: str
) -> None:
    current = make_run_result([make_case_result("a", "billing", True)])
    comparison = make_comparison(comparison_status, regressed_case_ids=[], improved_case_ids=[])

    payload = build_email_classifier_eval_alert_payload(
        report_id="report-1", current=current, comparison=comparison
    )

    assert payload.headline_status == expected_headline


def test_build_alert_payload_carries_headline_numbers_and_report_identifier() -> None:
    current = make_run_result([make_case_result("a", "billing", True)])
    comparison = make_comparison("critical", regressed_case_ids=["a", "b"], improved_case_ids=["c"])

    payload = build_email_classifier_eval_alert_payload(
        report_id="report-42", current=current, comparison=comparison
    )

    assert payload.report_id == "report-42"
    assert payload.dataset_status == "staged_draft"
    assert payload.previous_pass_rate == pytest.approx(0.9)
    assert payload.current_pass_rate == pytest.approx(0.8)
    assert payload.overall_delta_percentage_points == pytest.approx(-10.0)
    assert payload.regressed_case_count == 2
    assert payload.improved_case_count == 1
    assert isinstance(payload, EmailClassifierEvalAlertPayload)
