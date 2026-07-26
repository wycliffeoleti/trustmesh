"""Static, offline JSON/HTML reporting for email classifier evaluation runs.

This never calls a model, reads a history store, or touches the network; it only turns an
already-computed `EmailClassifierRunResult` pair and their `EmailClassifierRunComparison`
(see `trustmesh.email_classifier_eval_engine` / `email_classifier_eval_comparison`) into a
typed report, plus static JSON/HTML renderings of it. Every rendering carries the run's
`dataset_status`/`dataset_provenance` forward so a staged draft can never read as a golden,
model-evaluated, or production result.
"""
from __future__ import annotations

import html
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from trustmesh.email_classifier import EmailCategory
from trustmesh.email_classifier_eval_comparison import (
    CategoryAccuracyDelta,
    EmailClassifierRunComparison,
    RegressionThresholds,
)
from trustmesh.email_classifier_eval_drift import ScoreTrendPoint
from trustmesh.email_classifier_eval_engine import (
    EmailClassifierRunResult,
    UnsupportedOfflineDimension,
)
from trustmesh.email_classifier_staged_dataset import StagedDatasetProvenance


class RegressionDetail(BaseModel):
    model_config = ConfigDict(frozen=True)

    case_id: str
    expected_category: EmailCategory
    previous_category: EmailCategory
    previous_summary: str
    current_category: EmailCategory
    current_summary: str


class EmailClassifierEvalReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    report_id: str = Field(min_length=1)
    dataset_status: Literal["staged_draft"]
    dataset_provenance: StagedDatasetProvenance
    dataset_schema_version: str
    prompt_version: str
    prompt_created_at: datetime
    previous_pass_rate: float
    current_pass_rate: float
    overall_delta_percentage_points: float
    per_category_deltas: list[CategoryAccuracyDelta]
    thresholds: RegressionThresholds
    status: Literal["ok", "warning", "critical"]
    summary_relevance: UnsupportedOfflineDimension
    token_usage: UnsupportedOfflineDimension
    regressions: list[RegressionDetail]
    score_trend: list[ScoreTrendPoint]


def build_email_classifier_eval_report(
    report_id: str,
    current: EmailClassifierRunResult,
    previous: EmailClassifierRunResult,
    comparison: EmailClassifierRunComparison,
    score_trend: list[ScoreTrendPoint],
) -> EmailClassifierEvalReport:
    if comparison.current_pass_rate != current.category_pass_rate:
        raise ValueError("comparison.current_pass_rate does not match current.category_pass_rate")
    if comparison.previous_pass_rate != previous.category_pass_rate:
        raise ValueError("comparison.previous_pass_rate does not match previous.category_pass_rate")

    previous_by_id = {r.case_id: r for r in previous.case_results}
    current_by_id = {r.case_id: r for r in current.case_results}

    regressions = [
        RegressionDetail(
            case_id=case_id,
            expected_category=current_by_id[case_id].expected_category,
            previous_category=previous_by_id[case_id].actual_category,
            previous_summary=previous_by_id[case_id].actual_summary,
            current_category=current_by_id[case_id].actual_category,
            current_summary=current_by_id[case_id].actual_summary,
        )
        for case_id in comparison.regressed_case_ids
    ]

    return EmailClassifierEvalReport(
        report_id=report_id,
        dataset_status=current.dataset_status,
        dataset_provenance=current.dataset_provenance,
        dataset_schema_version=current.dataset_schema_version,
        prompt_version=current.prompt_version,
        prompt_created_at=current.prompt_created_at,
        previous_pass_rate=comparison.previous_pass_rate,
        current_pass_rate=comparison.current_pass_rate,
        overall_delta_percentage_points=comparison.overall_delta_percentage_points,
        per_category_deltas=comparison.per_category_deltas,
        thresholds=comparison.thresholds,
        status=comparison.status,
        summary_relevance=current.summary_relevance,
        token_usage=current.token_usage,
        regressions=regressions,
        score_trend=score_trend,
    )


def render_email_classifier_eval_report_json(report: EmailClassifierEvalReport) -> str:
    return report.model_dump_json(indent=2)


def _e(value: object) -> str:
    """Escape a value for HTML text/attribute content — the sole guard against a data
    field (a case summary, a provenance statement) turning into markup or a script."""
    return html.escape(str(value))


def _render_scorecard_rows(report: EmailClassifierEvalReport) -> str:
    rows = [
        "<tr><th scope=\"row\">Previous pass rate</th>"
        f"<td>{_e(f'{report.previous_pass_rate:.1%}')}</td></tr>",
        "<tr><th scope=\"row\">Current pass rate</th>"
        f"<td>{_e(f'{report.current_pass_rate:.1%}')}</td></tr>",
        "<tr><th scope=\"row\">Overall delta</th>"
        f"<td>{_e(f'{report.overall_delta_percentage_points:+.2f} pp')}</td></tr>",
    ]
    for delta in report.per_category_deltas:
        rows.append(
            f"<tr><th scope=\"row\">{_e(delta.category)} accuracy</th>"
            f"<td>{_e(f'{delta.previous_accuracy:.1%}')} &rarr; "
            f"{_e(f'{delta.current_accuracy:.1%}')} "
            f"({_e(f'{delta.delta_percentage_points:+.2f} pp')})</td></tr>"
        )
    return "\n".join(rows)


def _render_threshold_status(report: EmailClassifierEvalReport) -> str:
    return (
        f"<p>Status: <strong>{_e(report.status)}</strong> "
        f"(warning above {_e(report.thresholds.warning_percentage_points)} pp decline, "
        f"critical above {_e(report.thresholds.critical_percentage_points)} pp decline)</p>"
    )


def _render_regressions(report: EmailClassifierEvalReport) -> str:
    if not report.regressions:
        return "<p>No regressed cases.</p>"
    rows = "\n".join(
        "<tr>"
        f"<td>{_e(r.case_id)}</td>"
        f"<td>{_e(r.expected_category)}</td>"
        f"<td>{_e(r.previous_category)}</td>"
        f"<td>{_e(r.previous_summary)}</td>"
        f"<td>{_e(r.current_category)}</td>"
        f"<td>{_e(r.current_summary)}</td>"
        "</tr>"
        for r in report.regressions
    )
    return (
        "<table>\n"
        "<caption>Regressed cases (old vs. new)</caption>\n"
        "<thead><tr>"
        "<th scope=\"col\">Case ID</th><th scope=\"col\">Expected category</th>"
        "<th scope=\"col\">Previous category</th><th scope=\"col\">Previous summary</th>"
        "<th scope=\"col\">Current category</th><th scope=\"col\">Current summary</th>"
        "</tr></thead>\n"
        f"<tbody>\n{rows}\n</tbody>\n"
        "</table>"
    )


def _render_score_trend(report: EmailClassifierEvalReport) -> str:
    if not report.score_trend:
        return "<p>No run history supplied for the score trend.</p>"
    rows = "\n".join(
        "<tr>"
        f"<td>{_e(point.run_label)}</td>"
        "<td>"
        f"<meter min=\"0\" max=\"1\" value=\"{_e(point.category_pass_rate)}\" "
        f"aria-label=\"Category pass rate for {_e(point.run_label)}: "
        f"{_e(f'{point.category_pass_rate:.1%}')}\">"
        f"{_e(f'{point.category_pass_rate:.1%}')}"
        "</meter>"
        "</td>"
        "</tr>"
        for point in report.score_trend
    )
    return (
        "<table>\n"
        "<caption>Score trend over supplied run history</caption>\n"
        "<thead><tr><th scope=\"col\">Run</th>"
        "<th scope=\"col\">Category pass rate</th></tr></thead>\n"
        f"<tbody>\n{rows}\n</tbody>\n"
        "</table>"
    )


def render_email_classifier_eval_report_html(report: EmailClassifierEvalReport) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Email classifier evaluation report {_e(report.report_id)}</title>
<style>
table {{ border-collapse: collapse; }}
th, td {{ border: 1px solid #888; padding: 0.25rem 0.5rem; text-align: left; }}
</style>
</head>
<body>
<h1>Email classifier evaluation report</h1>
<p role="alert"><strong>STAGED DRAFT &mdash; NOT GOLDEN &mdash; NOT PRODUCTION EVIDENCE</strong></p>
<section>
<h2>Provenance</h2>
<p>Dataset status: <strong>{_e(report.dataset_status)}</strong></p>
<p>Authored by: {_e(report.dataset_provenance.authored_by)}</p>
<p>{_e(report.dataset_provenance.statement)}</p>
</section>
<section>
<h2>Run metadata</h2>
<p>Report ID: {_e(report.report_id)}</p>
<p>Prompt version: {_e(report.prompt_version)} (created {_e(report.prompt_created_at.isoformat())})</p>
<p>Dataset schema version: {_e(report.dataset_schema_version)}</p>
<p>Summary relevance: {_e(report.summary_relevance)}</p>
<p>Token usage: {_e(report.token_usage)}</p>
</section>
<section>
<h2>Scorecard</h2>
<table>
<caption>Baseline vs. current</caption>
<tbody>
{_render_scorecard_rows(report)}
</tbody>
</table>
{_render_threshold_status(report)}
</section>
<section>
<h2>Regressions</h2>
{_render_regressions(report)}
</section>
<section>
<h2>Score trend</h2>
{_render_score_trend(report)}
</section>
</body>
</html>
"""
