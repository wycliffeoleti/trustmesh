from __future__ import annotations

from datetime import UTC, datetime

import pytest

from trustmesh.email_classifier import EmailCategory
from trustmesh.email_classifier_eval_comparison import (
    RegressionThresholds,
    compare_email_classifier_runs,
)
from trustmesh.email_classifier_eval_drift import ScoreTrendPoint
from trustmesh.email_classifier_eval_engine import (
    UNSUPPORTED_OFFLINE,
    EmailClassifierCaseResult,
    EmailClassifierRunResult,
)
from trustmesh.email_classifier_eval_report import (
    build_email_classifier_eval_report,
    render_email_classifier_eval_report_html,
    render_email_classifier_eval_report_json,
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
    case_id: str,
    expected_category: EmailCategory,
    actual_category: EmailCategory,
    summary: str = "Synthetic summary.",
) -> EmailClassifierCaseResult:
    return EmailClassifierCaseResult(
        case_id=case_id,
        expected_category=expected_category,
        actual_category=actual_category,
        category_match=actual_category == expected_category,
        actual_summary=summary,
        summary_relevance=UNSUPPORTED_OFFLINE,
        token_usage=UNSUPPORTED_OFFLINE,
        latency_ms=1.0,
    )


def make_run_result(
    case_results: list[EmailClassifierCaseResult],
    prompt_version: str = "v1",
) -> EmailClassifierRunResult:
    match_count = sum(r.category_match for r in case_results)
    return EmailClassifierRunResult(
        prompt_version=prompt_version,
        prompt_created_at=datetime(2026, 1, 1, tzinfo=UTC),
        dataset_schema_version="1",
        dataset_status="staged_draft",
        dataset_provenance=PROVENANCE,
        case_results=case_results,
        category_pass_rate=match_count / len(case_results),
        summary_relevance=UNSUPPORTED_OFFLINE,
        token_usage=UNSUPPORTED_OFFLINE,
    )


# --- Cycle 1: report exposes provenance, prompt/dataset metadata, and scorecard ---


def test_build_report_exposes_provenance_prompt_and_scorecard_metadata() -> None:
    previous = make_run_result([
        make_case_result("a", "billing", "billing"),
        make_case_result("b", "billing", "billing"),
    ])
    current = make_run_result([
        make_case_result("a", "billing", "general"),
        make_case_result("b", "billing", "billing"),
    ])
    comparison = compare_email_classifier_runs(current, previous)

    report = build_email_classifier_eval_report(
        report_id="report-1",
        current=current,
        previous=previous,
        comparison=comparison,
        score_trend=[],
    )

    assert report.report_id == "report-1"
    assert report.dataset_status == "staged_draft"
    assert report.dataset_provenance == PROVENANCE
    assert report.prompt_version == "v1"
    assert report.prompt_created_at == datetime(2026, 1, 1, tzinfo=UTC)
    assert report.dataset_schema_version == "1"
    assert report.previous_pass_rate == pytest.approx(comparison.previous_pass_rate)
    assert report.current_pass_rate == pytest.approx(comparison.current_pass_rate)
    assert report.overall_delta_percentage_points == pytest.approx(
        comparison.overall_delta_percentage_points
    )
    assert report.thresholds == RegressionThresholds()
    assert report.status == comparison.status
    assert report.summary_relevance == UNSUPPORTED_OFFLINE
    assert report.token_usage == UNSUPPORTED_OFFLINE


# --- Cycle 2: regressions carry old/new category and summary side-by-side ---


def test_build_report_lists_every_regression_with_old_and_new_category_and_summary() -> None:
    previous = make_run_result([
        make_case_result("a", "billing", "billing", summary="Old billing summary."),
        make_case_result("b", "billing", "billing", summary="Stable summary."),
    ])
    current = make_run_result([
        make_case_result("a", "billing", "general", summary="New general summary."),
        make_case_result("b", "billing", "billing", summary="Stable summary."),
    ])
    comparison = compare_email_classifier_runs(current, previous)

    report = build_email_classifier_eval_report(
        report_id="report-1",
        current=current,
        previous=previous,
        comparison=comparison,
        score_trend=[],
    )

    assert comparison.regressed_case_ids == ["a"]
    assert len(report.regressions) == 1
    regression = report.regressions[0]
    assert regression.case_id == "a"
    assert regression.expected_category == "billing"
    assert regression.previous_category == "billing"
    assert regression.previous_summary == "Old billing summary."
    assert regression.current_category == "general"
    assert regression.current_summary == "New general summary."


def test_build_report_carries_supplied_score_trend_through_unchanged() -> None:
    previous = make_run_result([make_case_result("a", "billing", "billing")])
    current = make_run_result([make_case_result("a", "billing", "billing")])
    comparison = compare_email_classifier_runs(current, previous)
    trend = [
        ScoreTrendPoint(run_label="run-1", category_pass_rate=0.8),
        ScoreTrendPoint(run_label="run-2", category_pass_rate=0.9),
    ]

    report = build_email_classifier_eval_report(
        report_id="report-1",
        current=current,
        previous=previous,
        comparison=comparison,
        score_trend=trend,
    )

    assert report.score_trend == trend


# --- Cycle 3: mismatched comparison/run inputs are rejected ---


def test_build_report_rejects_comparison_that_does_not_match_supplied_runs() -> None:
    previous = make_run_result([make_case_result("a", "billing", "billing")])
    current = make_run_result([make_case_result("a", "billing", "billing")])
    other_current = make_run_result([make_case_result("a", "billing", "general")])
    mismatched_comparison = compare_email_classifier_runs(other_current, previous)

    with pytest.raises(ValueError, match="does not match"):
        build_email_classifier_eval_report(
            report_id="report-1",
            current=current,
            previous=previous,
            comparison=mismatched_comparison,
            score_trend=[],
        )


# --- Cycle 4: JSON rendering ---


def _sample_report() -> object:
    previous = make_run_result([
        make_case_result("a", "billing", "billing", summary="Old billing summary."),
    ])
    current = make_run_result([
        make_case_result("a", "billing", "general", summary="New general summary."),
    ])
    comparison = compare_email_classifier_runs(current, previous)
    return build_email_classifier_eval_report(
        report_id="report-1",
        current=current,
        previous=previous,
        comparison=comparison,
        score_trend=[ScoreTrendPoint(run_label="run-1", category_pass_rate=0.8)],
    )


def test_render_report_json_round_trips_report_fields() -> None:
    import json

    report = _sample_report()

    payload = json.loads(render_email_classifier_eval_report_json(report))

    assert payload["report_id"] == "report-1"
    assert payload["dataset_status"] == "staged_draft"
    assert payload["regressions"][0]["case_id"] == "a"
    assert payload["regressions"][0]["previous_category"] == "billing"
    assert payload["regressions"][0]["current_category"] == "general"
    assert payload["score_trend"][0]["run_label"] == "run-1"


# --- Cycle 5: HTML rendering never presents staged-draft data as golden/production ---


def test_render_report_html_shows_staged_draft_disclaimer_and_provenance() -> None:
    report = _sample_report()

    output = render_email_classifier_eval_report_html(report)

    assert "STAGED DRAFT" in output
    assert "NOT GOLDEN" in output
    assert "NOT PRODUCTION" in output
    assert "staged_draft" in output
    assert PROVENANCE.statement in output
    assert "unsupported_offline" in output


def test_render_report_html_shows_scorecard_threshold_status_and_metadata() -> None:
    report = _sample_report()

    output = render_email_classifier_eval_report_html(report)

    assert "report-1" in output
    assert "v1" in output
    assert report.status in output
    assert "3.0" in output  # default warning threshold
    assert "8.0" in output  # default critical threshold


def test_render_report_html_shows_every_regression_old_and_new_side_by_side() -> None:
    report = _sample_report()

    output = render_email_classifier_eval_report_html(report)

    assert "Old billing summary." in output
    assert "New general summary." in output
    assert "billing" in output
    assert "general" in output


def test_render_report_html_shows_accessible_score_trend_with_aria_labels() -> None:
    report = _sample_report()

    output = render_email_classifier_eval_report_html(report)

    assert "<meter" in output
    assert "aria-label=" in output
    assert "run-1" in output


def test_render_report_html_escapes_data_that_looks_like_markup_or_a_script() -> None:
    previous = make_run_result([
        make_case_result("a", "billing", "billing", summary="<script>alert(1)</script>"),
    ])
    current = make_run_result([
        make_case_result("a", "billing", "general", summary="<img src=x onerror=alert(1)>"),
    ])
    comparison = compare_email_classifier_runs(current, previous)
    report = build_email_classifier_eval_report(
        report_id="report-1",
        current=current,
        previous=previous,
        comparison=comparison,
        score_trend=[],
    )

    output = render_email_classifier_eval_report_html(report)

    assert "<script>" not in output
    assert "<img" not in output
    assert "&lt;script&gt;" in output


def test_render_report_html_contains_no_scripts_or_network_references() -> None:
    report = _sample_report()

    output = render_email_classifier_eval_report_html(report)

    lowered = output.lower()
    assert "<script" not in lowered
    assert "javascript:" not in lowered
    assert "http://" not in lowered
    assert "https://" not in lowered
    assert "<link" not in lowered
    assert "fetch(" not in lowered
