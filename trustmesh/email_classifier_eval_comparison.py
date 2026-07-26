"""Pure comparison between two email classifier evaluation runs.

This module never classifies an email, calls a model, or touches the filesystem; it only
reads two already-computed `EmailClassifierRunResult` values (see
`trustmesh.email_classifier_eval_engine`) and derives deltas. There is no baseline
persistence here — callers are responsible for supplying whatever "previous" run they
want compared. Warning/critical thresholds mirror the BASWE guide's suggested 3/8
percentage-point defaults, but are ordinary configurable values, not hidden constants.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator

from trustmesh.email_classifier import EmailCategory
from trustmesh.email_classifier_eval_engine import EmailClassifierRunResult

_CATEGORY_ORDER: tuple[EmailCategory, ...] = ("billing", "technical", "account", "general")


class RegressionThresholds(BaseModel):
    model_config = ConfigDict(frozen=True)

    warning_percentage_points: float = 3.0
    critical_percentage_points: float = 8.0

    @model_validator(mode="after")
    def _warning_below_critical(self) -> RegressionThresholds:
        if self.warning_percentage_points >= self.critical_percentage_points:
            raise ValueError(
                "warning_percentage_points must be lower than critical_percentage_points"
            )
        return self


class CategoryAccuracyDelta(BaseModel):
    model_config = ConfigDict(frozen=True)

    category: EmailCategory
    previous_accuracy: float
    current_accuracy: float
    delta_percentage_points: float


class EmailClassifierRunComparison(BaseModel):
    model_config = ConfigDict(frozen=True)

    previous_pass_rate: float
    current_pass_rate: float
    overall_delta_percentage_points: float
    per_category_deltas: list[CategoryAccuracyDelta]
    regressed_case_ids: list[str]
    improved_case_ids: list[str]
    thresholds: RegressionThresholds
    status: Literal["ok", "warning", "critical"]


def _category_accuracy(run: EmailClassifierRunResult, category: EmailCategory) -> float | None:
    matches = [r.category_match for r in run.case_results if r.expected_category == category]
    if not matches:
        return None
    return sum(matches) / len(matches)


def compare_email_classifier_runs(
    current: EmailClassifierRunResult,
    previous: EmailClassifierRunResult,
    thresholds: RegressionThresholds = RegressionThresholds(),
) -> EmailClassifierRunComparison:
    # Rounded to avoid float-representation noise (e.g. 3.0000000000000027) turning an
    # exact threshold boundary into a spurious warning/critical classification.
    overall_delta = round((current.category_pass_rate - previous.category_pass_rate) * 100, 9)

    per_category_deltas: list[CategoryAccuracyDelta] = []
    for category in _CATEGORY_ORDER:
        previous_accuracy = _category_accuracy(previous, category)
        current_accuracy = _category_accuracy(current, category)
        if previous_accuracy is None or current_accuracy is None:
            continue
        per_category_deltas.append(
            CategoryAccuracyDelta(
                category=category,
                previous_accuracy=previous_accuracy,
                current_accuracy=current_accuracy,
                delta_percentage_points=round((current_accuracy - previous_accuracy) * 100, 9),
            )
        )

    previous_by_id = {r.case_id: r.category_match for r in previous.case_results}
    current_by_id = {r.case_id: r.category_match for r in current.case_results}
    shared_ids = previous_by_id.keys() & current_by_id.keys()

    regressed_case_ids = sorted(
        case_id for case_id in shared_ids if previous_by_id[case_id] and not current_by_id[case_id]
    )
    improved_case_ids = sorted(
        case_id for case_id in shared_ids if not previous_by_id[case_id] and current_by_id[case_id]
    )

    decline = -overall_delta
    status: Literal["ok", "warning", "critical"]
    if decline > thresholds.critical_percentage_points:
        status = "critical"
    elif decline > thresholds.warning_percentage_points:
        status = "warning"
    else:
        status = "ok"

    return EmailClassifierRunComparison(
        previous_pass_rate=previous.category_pass_rate,
        current_pass_rate=current.category_pass_rate,
        overall_delta_percentage_points=overall_delta,
        per_category_deltas=per_category_deltas,
        regressed_case_ids=regressed_case_ids,
        improved_case_ids=improved_case_ids,
        thresholds=thresholds,
        status=status,
    )
