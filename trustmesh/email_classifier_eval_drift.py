"""Pure seven-run moving-average slow-drift detection for email classifier eval history.

This never reads a history store, clock, or filesystem; the caller supplies the ordered
run history, the reference pass rate to compare against, and the decline threshold. A
slow-drift warning fires only when the moving average has declined past the configured
threshold *and* the caller-supplied `EmailClassifierRunComparison.status` is `"ok"` — a
run that already breached the per-run warning/critical threshold is reported through that
mechanism, not this one.
"""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from trustmesh.email_classifier_eval_comparison import EmailClassifierRunComparison

_WINDOW_SIZE = 7


class ScoreTrendPoint(BaseModel):
    model_config = ConfigDict(frozen=True)

    run_label: str = Field(min_length=1)
    category_pass_rate: float = Field(ge=0.0, le=1.0)


class SlowDriftThresholds(BaseModel):
    model_config = ConfigDict(frozen=True)

    decline_percentage_points: float = 3.0


class SlowDriftResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    has_sufficient_history: bool
    moving_average_pass_rate: float | None
    decline_percentage_points: float | None
    is_slow_drift_warning: bool


def detect_slow_drift(
    comparison: EmailClassifierRunComparison,
    history: list[ScoreTrendPoint],
    reference_pass_rate: float,
    thresholds: SlowDriftThresholds = SlowDriftThresholds(),
) -> SlowDriftResult:
    if len(history) < _WINDOW_SIZE:
        return SlowDriftResult(
            has_sufficient_history=False,
            moving_average_pass_rate=None,
            decline_percentage_points=None,
            is_slow_drift_warning=False,
        )

    window = history[-_WINDOW_SIZE:]
    moving_average = sum(point.category_pass_rate for point in window) / _WINDOW_SIZE
    decline = round((reference_pass_rate - moving_average) * 100, 9)
    is_warning = decline > thresholds.decline_percentage_points and comparison.status == "ok"

    return SlowDriftResult(
        has_sufficient_history=True,
        moving_average_pass_rate=moving_average,
        decline_percentage_points=decline,
        is_slow_drift_warning=is_warning,
    )
