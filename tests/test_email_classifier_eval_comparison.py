from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from trustmesh.email_classifier_eval_comparison import (
    CategoryAccuracyDelta,
    EmailClassifierRunComparison,
    RegressionThresholds,
    compare_email_classifier_runs,
)
from trustmesh.email_classifier import EmailCategory
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


def make_case_result(
    case_id: str, expected_category: EmailCategory, category_match: bool
) -> EmailClassifierCaseResult:
    actual_category = expected_category if category_match else "general"
    return EmailClassifierCaseResult(
        case_id=case_id,
        expected_category=expected_category,
        actual_category=actual_category,
        category_match=category_match,
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


# --- RegressionThresholds ---


def test_regression_thresholds_default_values_are_transparent() -> None:
    thresholds = RegressionThresholds()
    assert thresholds.warning_percentage_points == 3.0
    assert thresholds.critical_percentage_points == 8.0


def test_regression_thresholds_accepts_custom_values() -> None:
    thresholds = RegressionThresholds(warning_percentage_points=5.0, critical_percentage_points=10.0)
    assert thresholds.warning_percentage_points == 5.0
    assert thresholds.critical_percentage_points == 10.0


def test_regression_thresholds_rejects_warning_at_or_above_critical() -> None:
    with pytest.raises(ValidationError):
        RegressionThresholds(warning_percentage_points=8.0, critical_percentage_points=8.0)


# --- compare_email_classifier_runs: overall pass-rate delta ---


def test_compare_runs_computes_overall_pass_rate_delta_in_percentage_points() -> None:
    previous = make_run_result([
        make_case_result("a", "billing", True),
        make_case_result("b", "billing", True),
        make_case_result("c", "billing", False),
        make_case_result("d", "billing", False),
    ])
    current = make_run_result([
        make_case_result("a", "billing", True),
        make_case_result("b", "billing", True),
        make_case_result("c", "billing", True),
        make_case_result("d", "billing", False),
    ])

    comparison = compare_email_classifier_runs(current, previous)

    assert comparison.previous_pass_rate == pytest.approx(0.5)
    assert comparison.current_pass_rate == pytest.approx(0.75)
    assert comparison.overall_delta_percentage_points == pytest.approx(25.0)


def test_compare_runs_reports_negative_delta_on_decline() -> None:
    previous = make_run_result([
        make_case_result("a", "billing", True),
        make_case_result("b", "billing", True),
    ])
    current = make_run_result([
        make_case_result("a", "billing", True),
        make_case_result("b", "billing", False),
    ])

    comparison = compare_email_classifier_runs(current, previous)

    assert comparison.overall_delta_percentage_points == pytest.approx(-50.0)


# --- per-category accuracy deltas ---


def test_compare_runs_computes_per_category_accuracy_deltas() -> None:
    previous = make_run_result([
        make_case_result("b1", "billing", True),
        make_case_result("b2", "billing", False),
        make_case_result("t1", "technical", True),
        make_case_result("t2", "technical", True),
    ])
    current = make_run_result([
        make_case_result("b1", "billing", True),
        make_case_result("b2", "billing", True),
        make_case_result("t1", "technical", True),
        make_case_result("t2", "technical", False),
    ])

    comparison = compare_email_classifier_runs(current, previous)
    by_category = {delta.category: delta for delta in comparison.per_category_deltas}

    assert by_category["billing"].previous_accuracy == pytest.approx(0.5)
    assert by_category["billing"].current_accuracy == pytest.approx(1.0)
    assert by_category["billing"].delta_percentage_points == pytest.approx(50.0)

    assert by_category["technical"].previous_accuracy == pytest.approx(1.0)
    assert by_category["technical"].current_accuracy == pytest.approx(0.5)
    assert by_category["technical"].delta_percentage_points == pytest.approx(-50.0)


def test_compare_runs_omits_category_absent_from_either_run() -> None:
    previous = make_run_result([make_case_result("b1", "billing", True)])
    current = make_run_result([make_case_result("b1", "billing", True)])

    comparison = compare_email_classifier_runs(current, previous)

    categories = {delta.category for delta in comparison.per_category_deltas}
    assert categories == {"billing"}


def test_category_accuracy_delta_is_a_typed_model() -> None:
    delta = CategoryAccuracyDelta(
        category="billing",
        previous_accuracy=0.5,
        current_accuracy=1.0,
        delta_percentage_points=50.0,
    )
    assert delta.category == "billing"


# --- pass-to-fail / fail-to-pass case tracking ---


def test_compare_runs_identifies_pass_to_fail_regression_case_ids() -> None:
    previous = make_run_result([
        make_case_result("a", "billing", True),
        make_case_result("b", "billing", True),
    ])
    current = make_run_result([
        make_case_result("a", "billing", False),
        make_case_result("b", "billing", True),
    ])

    comparison = compare_email_classifier_runs(current, previous)

    assert comparison.regressed_case_ids == ["a"]
    assert comparison.improved_case_ids == []


def test_compare_runs_identifies_fail_to_pass_improvement_case_ids() -> None:
    previous = make_run_result([
        make_case_result("a", "billing", False),
        make_case_result("b", "billing", True),
    ])
    current = make_run_result([
        make_case_result("a", "billing", True),
        make_case_result("b", "billing", True),
    ])

    comparison = compare_email_classifier_runs(current, previous)

    assert comparison.improved_case_ids == ["a"]
    assert comparison.regressed_case_ids == []


def test_compare_runs_ignores_case_ids_not_shared_between_runs() -> None:
    previous = make_run_result([make_case_result("a", "billing", True)])
    current = make_run_result([make_case_result("z", "billing", False)])

    comparison = compare_email_classifier_runs(current, previous)

    assert comparison.regressed_case_ids == []
    assert comparison.improved_case_ids == []


# --- threshold classification ---


def _run_with_pass_rate(pass_count: int, total: int) -> EmailClassifierRunResult:
    cases = [make_case_result(f"c{i}", "billing", i < pass_count) for i in range(total)]
    return make_run_result(cases)


def test_compare_runs_status_is_ok_when_pass_rate_is_unchanged() -> None:
    previous = _run_with_pass_rate(80, 100)
    current = _run_with_pass_rate(80, 100)

    comparison = compare_email_classifier_runs(current, previous)

    assert comparison.status == "ok"


def test_compare_runs_status_is_ok_when_pass_rate_improves() -> None:
    previous = _run_with_pass_rate(70, 100)
    current = _run_with_pass_rate(90, 100)

    comparison = compare_email_classifier_runs(current, previous)

    assert comparison.status == "ok"


def test_compare_runs_status_is_warning_when_decline_exceeds_default_warning_threshold() -> None:
    previous = _run_with_pass_rate(90, 100)
    current = _run_with_pass_rate(85, 100)

    comparison = compare_email_classifier_runs(current, previous)

    assert comparison.status == "warning"


def test_compare_runs_status_is_ok_at_exactly_the_warning_boundary() -> None:
    previous = _run_with_pass_rate(90, 100)
    current = _run_with_pass_rate(87, 100)

    comparison = compare_email_classifier_runs(current, previous)

    assert comparison.status == "ok"


def test_compare_runs_status_is_critical_when_decline_exceeds_default_critical_threshold() -> None:
    previous = _run_with_pass_rate(90, 100)
    current = _run_with_pass_rate(80, 100)

    comparison = compare_email_classifier_runs(current, previous)

    assert comparison.status == "critical"


def test_compare_runs_status_is_warning_at_exactly_the_critical_boundary() -> None:
    previous = _run_with_pass_rate(90, 100)
    current = _run_with_pass_rate(82, 100)

    comparison = compare_email_classifier_runs(current, previous)

    assert comparison.status == "warning"


def test_compare_runs_accepts_custom_thresholds() -> None:
    previous = _run_with_pass_rate(90, 100)
    current = _run_with_pass_rate(85, 100)

    default_comparison = compare_email_classifier_runs(current, previous)
    strict_comparison = compare_email_classifier_runs(
        current, previous, thresholds=RegressionThresholds(
            warning_percentage_points=1.0, critical_percentage_points=2.0
        )
    )

    assert default_comparison.status == "warning"
    assert strict_comparison.status == "critical"


def test_compare_runs_result_carries_the_thresholds_used() -> None:
    previous = _run_with_pass_rate(90, 100)
    current = _run_with_pass_rate(85, 100)
    thresholds = RegressionThresholds(warning_percentage_points=1.0, critical_percentage_points=2.0)

    comparison = compare_email_classifier_runs(current, previous, thresholds=thresholds)

    assert comparison.thresholds == thresholds


def test_compare_email_classifier_runs_is_pure_and_deterministic() -> None:
    previous = _run_with_pass_rate(90, 100)
    current = _run_with_pass_rate(85, 100)

    first = compare_email_classifier_runs(current, previous)
    second = compare_email_classifier_runs(current, previous)

    assert first == second


def test_email_classifier_run_comparison_is_a_typed_model() -> None:
    comparison = compare_email_classifier_runs(
        _run_with_pass_rate(90, 100), _run_with_pass_rate(90, 100)
    )
    assert isinstance(comparison, EmailClassifierRunComparison)
