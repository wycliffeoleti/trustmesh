"""Structured offline alert payload for an email classifier evaluation run.

This builds a typed pass/warn/fail summary plus headline numbers and a report identifier
from an already-computed `EmailClassifierRunResult` and `EmailClassifierRunComparison`
(see `trustmesh.email_classifier_eval_engine` / `email_classifier_eval_comparison`). It is
an in-memory payload only: there is no webhook client, no environment variable read, no
network request, and no Slack (or any other transport) dependency here. Delivering this
payload anywhere is a separate, later, explicitly-approved concern.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from trustmesh.email_classifier_eval_comparison import EmailClassifierRunComparison
from trustmesh.email_classifier_eval_engine import EmailClassifierRunResult

HeadlineStatus = Literal["pass", "warn", "fail"]

_HEADLINE_STATUS_BY_COMPARISON_STATUS: dict[str, HeadlineStatus] = {
    "ok": "pass",
    "warning": "warn",
    "critical": "fail",
}


class EmailClassifierEvalAlertPayload(BaseModel):
    model_config = ConfigDict(frozen=True)

    report_id: str = Field(min_length=1)
    dataset_status: Literal["staged_draft"]
    headline_status: HeadlineStatus
    previous_pass_rate: float
    current_pass_rate: float
    overall_delta_percentage_points: float
    regressed_case_count: int
    improved_case_count: int


def build_email_classifier_eval_alert_payload(
    report_id: str,
    current: EmailClassifierRunResult,
    comparison: EmailClassifierRunComparison,
) -> EmailClassifierEvalAlertPayload:
    return EmailClassifierEvalAlertPayload(
        report_id=report_id,
        dataset_status=current.dataset_status,
        headline_status=_HEADLINE_STATUS_BY_COMPARISON_STATUS[comparison.status],
        previous_pass_rate=comparison.previous_pass_rate,
        current_pass_rate=comparison.current_pass_rate,
        overall_delta_percentage_points=comparison.overall_delta_percentage_points,
        regressed_case_count=len(comparison.regressed_case_ids),
        improved_case_count=len(comparison.improved_case_ids),
    )
