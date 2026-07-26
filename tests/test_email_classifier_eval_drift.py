from __future__ import annotations

import pytest
from pydantic import ValidationError

from trustmesh.email_classifier_eval_comparison import (
    CategoryAccuracyDelta,
    EmailClassifierRunComparison,
    RegressionThresholds,
)
from trustmesh.email_classifier_eval_drift import (
    ScoreTrendPoint,
    SlowDriftThresholds,
    detect_slow_drift,
)
from typing import Literal


def make_comparison(status: Literal["ok", "warning", "critical"]) -> EmailClassifierRunComparison:
    return EmailClassifierRunComparison(
        previous_pass_rate=0.9,
        current_pass_rate=0.85,
        overall_delta_percentage_points=-5.0,
        per_category_deltas=[
            CategoryAccuracyDelta(
                category="billing",
                previous_accuracy=0.9,
                current_accuracy=0.85,
                delta_percentage_points=-5.0,
            )
        ],
        regressed_case_ids=[],
        improved_case_ids=[],
        thresholds=RegressionThresholds(),
        status=status,
    )


def make_history(pass_rates: list[float]) -> list[ScoreTrendPoint]:
    return [
        ScoreTrendPoint(run_label=f"run-{i}", category_pass_rate=rate)
        for i, rate in enumerate(pass_rates)
    ]


# --- Cycle 1: insufficient history ---


def test_detect_slow_drift_reports_insufficient_history_below_seven_runs() -> None:
    result = detect_slow_drift(
        comparison=make_comparison("ok"),
        history=make_history([0.9, 0.9, 0.9, 0.9, 0.9, 0.9]),
        reference_pass_rate=0.9,
    )

    assert result.has_sufficient_history is False
    assert result.moving_average_pass_rate is None
    assert result.is_slow_drift_warning is False


# --- Cycle 2: moving average uses only the most recent seven runs ---


def test_detect_slow_drift_averages_only_the_most_recent_seven_runs() -> None:
    # An old, much-lower run outside the seven-run window must not pull the average down.
    history = make_history([0.10, 0.90, 0.90, 0.90, 0.90, 0.90, 0.90, 0.90])

    result = detect_slow_drift(
        comparison=make_comparison("ok"),
        history=history,
        reference_pass_rate=0.90,
    )

    assert result.has_sufficient_history is True
    assert result.moving_average_pass_rate == pytest.approx(0.90)
    assert result.decline_percentage_points == pytest.approx(0.0)
    assert result.is_slow_drift_warning is False


# --- Cycle 3: warning fires only when decline exceeds threshold AND status is ok ---


def test_detect_slow_drift_warns_when_decline_exceeds_threshold_and_status_is_ok() -> None:
    history = make_history([0.80] * 7)

    result = detect_slow_drift(
        comparison=make_comparison("ok"),
        history=history,
        reference_pass_rate=0.90,
        thresholds=SlowDriftThresholds(decline_percentage_points=5.0),
    )

    assert result.moving_average_pass_rate == pytest.approx(0.80)
    assert result.decline_percentage_points == pytest.approx(10.0)
    assert result.is_slow_drift_warning is True


def test_detect_slow_drift_does_not_warn_when_decline_is_within_threshold() -> None:
    history = make_history([0.88] * 7)

    result = detect_slow_drift(
        comparison=make_comparison("ok"),
        history=history,
        reference_pass_rate=0.90,
        thresholds=SlowDriftThresholds(decline_percentage_points=5.0),
    )

    assert result.is_slow_drift_warning is False


def test_detect_slow_drift_does_not_warn_when_comparison_status_is_not_ok() -> None:
    # A per-run threshold breach is already surfaced by EmailClassifierRunComparison.status;
    # slow-drift must not double-report it.
    history = make_history([0.80] * 7)

    result = detect_slow_drift(
        comparison=make_comparison("critical"),
        history=history,
        reference_pass_rate=0.90,
        thresholds=SlowDriftThresholds(decline_percentage_points=5.0),
    )

    assert result.decline_percentage_points == pytest.approx(10.0)
    assert result.is_slow_drift_warning is False


def test_detect_slow_drift_is_pure_and_deterministic() -> None:
    history = make_history([0.80] * 7)
    comparison = make_comparison("ok")

    first = detect_slow_drift(comparison, history, reference_pass_rate=0.90)
    second = detect_slow_drift(comparison, history, reference_pass_rate=0.90)

    assert first == second


def test_score_trend_point_rejects_pass_rate_outside_unit_interval() -> None:
    with pytest.raises(ValidationError):
        ScoreTrendPoint(run_label="run-0", category_pass_rate=1.5)
