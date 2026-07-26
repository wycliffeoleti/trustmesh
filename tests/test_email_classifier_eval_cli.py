from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from trustmesh.email_classifier import EmailCategory
from trustmesh.email_classifier_eval_cli import (
    DEFAULT_BASELINE_PATH,
    DEFAULT_DATASET_PATH,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PROMPT_PATH,
    evaluate_against_baseline,
    exit_code_for_status,
    main,
    write_reports,
)
from trustmesh.email_classifier_eval_engine import (
    UNSUPPORTED_OFFLINE,
    EmailClassifierCaseResult,
    EmailClassifierRunResult,
)
from trustmesh.email_classifier_staged_dataset import StagedDatasetProvenance

ROOT = Path(__file__).resolve().parent.parent

PROVENANCE = StagedDatasetProvenance(
    authored_by="assistant",
    reviewed_by=None,
    statement=(
        "This dataset was authored entirely by an AI assistant as synthetic seed data. "
        "It has not been reviewed by a human and must not be treated as a golden dataset "
        "until a human reviewer completes review."
    ),
)

# The 16 checked-in staged-dataset case ids/expected categories, reused here so synthetic
# baseline fixtures line up (by case id) with whatever the real deterministic v1 run
# produces, without needing to duplicate the real case input text.
_CASE_CATEGORIES: list[tuple[str, EmailCategory]] = [
    ("billing-001", "billing"),
    ("billing-002", "billing"),
    ("billing-003", "billing"),
    ("billing-004", "billing"),
    ("technical-001", "technical"),
    ("technical-002", "technical"),
    ("technical-003", "technical"),
    ("technical-004", "technical"),
    ("account-001", "account"),
    ("account-002", "account"),
    ("account-003", "account"),
    ("account-004", "account"),
    ("general-001", "general"),
    ("general-002", "general"),
    ("general-003", "general"),
    ("general-004", "general"),
]


def make_synthetic_baseline(passing_case_ids: set[str]) -> EmailClassifierRunResult:
    case_results = [
        EmailClassifierCaseResult(
            case_id=case_id,
            expected_category=category,
            actual_category=category if case_id in passing_case_ids else "general",
            category_match=case_id in passing_case_ids,
            actual_summary="Synthetic baseline summary.",
            summary_relevance=UNSUPPORTED_OFFLINE,
            token_usage=UNSUPPORTED_OFFLINE,
            latency_ms=0.0,
        )
        for case_id, category in _CASE_CATEGORIES
    ]
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


def write_baseline(tmp_path: Path, passing_case_ids: set[str]) -> Path:
    baseline = make_synthetic_baseline(passing_case_ids)
    path = tmp_path / "baseline.json"
    path.write_text(baseline.model_dump_json())
    return path


# --- Cycle 1: default paths point at real checked-in files ---


def test_default_paths_point_at_real_checked_in_files() -> None:
    assert DEFAULT_PROMPT_PATH.exists()
    assert DEFAULT_DATASET_PATH.exists()
    assert DEFAULT_BASELINE_PATH.exists()
    assert DEFAULT_PROMPT_PATH == ROOT / "prompts" / "email_classifier" / "v1.yaml"
    assert DEFAULT_DATASET_PATH == ROOT / "evals" / "email_classifier_staged" / "v1.json"


# --- Cycle 2: exit_code_for_status is nonzero only for "critical" ---


@pytest.mark.parametrize(
    ("status", "expected_exit_code"),
    [("ok", 0), ("warning", 0), ("critical", 1)],
)
def test_exit_code_for_status_is_nonzero_only_for_critical(
    status: str, expected_exit_code: int
) -> None:
    assert exit_code_for_status(status) == expected_exit_code


# --- Cycle 3: evaluate_against_baseline against the real checked-in baseline is "ok" ---


def test_evaluate_against_checked_in_baseline_is_deterministic_and_ok() -> None:
    report, alert = evaluate_against_baseline(
        DEFAULT_PROMPT_PATH, DEFAULT_DATASET_PATH, DEFAULT_BASELINE_PATH
    )

    assert report.status == "ok"
    assert report.overall_delta_percentage_points == pytest.approx(0.0)
    assert report.dataset_status == "staged_draft"
    assert report.prompt_version == "v1"
    assert alert.headline_status == "pass"
    assert alert.dataset_status == "staged_draft"


def test_evaluate_against_baseline_is_pure_and_deterministic_across_calls() -> None:
    first_report, first_alert = evaluate_against_baseline(
        DEFAULT_PROMPT_PATH, DEFAULT_DATASET_PATH, DEFAULT_BASELINE_PATH
    )
    second_report, second_alert = evaluate_against_baseline(
        DEFAULT_PROMPT_PATH, DEFAULT_DATASET_PATH, DEFAULT_BASELINE_PATH
    )

    assert first_report == second_report
    assert first_alert == second_alert


# --- Cycle 4: a deliberately degraded baseline fixture classifies as critical ---


def test_evaluate_against_deliberately_healthier_baseline_is_critical(tmp_path: Path) -> None:
    # Baseline where every case "passed" is far healthier than the real deterministic
    # v1 run (10/16 correct) — this must classify as a critical regression.
    baseline_path = write_baseline(tmp_path, passing_case_ids={cid for cid, _ in _CASE_CATEGORIES})

    report, alert = evaluate_against_baseline(
        DEFAULT_PROMPT_PATH, DEFAULT_DATASET_PATH, baseline_path
    )

    assert report.status == "critical"
    assert alert.headline_status == "fail"
    assert exit_code_for_status(report.status) == 1


def test_evaluate_against_mildly_healthier_baseline_is_warning_not_critical(
    tmp_path: Path,
) -> None:
    # One case flip on a 16-case dataset is a 6.25pp decline: above the 3pp warning
    # threshold but at/below the 8pp critical threshold.
    real_report, _ = evaluate_against_baseline(
        DEFAULT_PROMPT_PATH, DEFAULT_DATASET_PATH, DEFAULT_BASELINE_PATH
    )
    real_passing_ids = {
        cid
        for cid, category in _CASE_CATEGORIES
        if cid not in {"billing-004", "technical-002", "technical-003", "technical-004",
                        "account-002", "account-003"}
    }
    assert len(real_passing_ids) == 10  # matches the real deterministic pass count (10/16)

    one_extra_pass = set(real_passing_ids) | {"billing-004"}
    baseline_path = write_baseline(tmp_path, passing_case_ids=one_extra_pass)

    report, alert = evaluate_against_baseline(
        DEFAULT_PROMPT_PATH, DEFAULT_DATASET_PATH, baseline_path
    )

    assert report.status == "warning"
    assert alert.headline_status == "warn"
    assert exit_code_for_status(report.status) == 0
    assert real_report.status == "ok"  # sanity: unrelated to this fixture


# --- Cycle 5: reports preserve staged-draft/not-golden provenance ---


def test_evaluate_against_baseline_report_never_drops_staged_draft_provenance(
    tmp_path: Path,
) -> None:
    baseline_path = write_baseline(tmp_path, passing_case_ids={cid for cid, _ in _CASE_CATEGORIES})

    report, alert = evaluate_against_baseline(
        DEFAULT_PROMPT_PATH, DEFAULT_DATASET_PATH, baseline_path
    )

    assert report.dataset_status == "staged_draft"
    assert report.dataset_provenance.authored_by == "assistant"
    assert alert.dataset_status == "staged_draft"


# --- Cycle 6: write_reports writes JSON/HTML/alert artifacts that keep provenance ---


def test_write_reports_creates_json_html_and_alert_files_with_provenance(
    tmp_path: Path,
) -> None:
    report, alert = evaluate_against_baseline(
        DEFAULT_PROMPT_PATH, DEFAULT_DATASET_PATH, DEFAULT_BASELINE_PATH
    )
    output_dir = tmp_path / "out"

    write_reports(output_dir, report, alert)

    report_json = (output_dir / "report.json").read_text()
    report_html = (output_dir / "report.html").read_text()
    alert_json = (output_dir / "alert.json").read_text()

    assert json.loads(report_json)["dataset_status"] == "staged_draft"
    assert "STAGED DRAFT" in report_html
    assert "NOT GOLDEN" in report_html
    assert json.loads(alert_json)["dataset_status"] == "staged_draft"
    assert json.loads(alert_json)["headline_status"] == "pass"


def test_write_reports_creates_output_directory_if_missing(tmp_path: Path) -> None:
    report, alert = evaluate_against_baseline(
        DEFAULT_PROMPT_PATH, DEFAULT_DATASET_PATH, DEFAULT_BASELINE_PATH
    )
    output_dir = tmp_path / "nested" / "does" / "not" / "exist"

    write_reports(output_dir, report, alert)

    assert (output_dir / "report.json").exists()
    assert (output_dir / "report.html").exists()
    assert (output_dir / "alert.json").exists()


# --- Cycle 7: main() end-to-end CLI wiring ---


def test_main_with_default_paths_exits_zero_and_writes_reports_and_prints_headline(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output_dir = tmp_path / "cli-out"

    exit_code = main(["--output-dir", str(output_dir)])

    assert exit_code == 0
    assert (output_dir / "report.json").exists()
    assert (output_dir / "report.html").exists()
    assert (output_dir / "alert.json").exists()
    printed = json.loads(capsys.readouterr().out)
    assert printed["headline_status"] == "pass"
    assert printed["dataset_status"] == "staged_draft"


def test_main_with_deliberately_healthier_baseline_exits_nonzero(
    tmp_path: Path,
) -> None:
    baseline_path = write_baseline(tmp_path, passing_case_ids={cid for cid, _ in _CASE_CATEGORIES})
    output_dir = tmp_path / "cli-out"

    exit_code = main([
        "--baseline-path", str(baseline_path),
        "--output-dir", str(output_dir),
    ])

    assert exit_code == 1


def test_main_default_output_dir_is_not_inside_tracked_docs_verification() -> None:
    assert "docs" not in DEFAULT_OUTPUT_DIR.parts


def test_main_is_fully_deterministic_across_two_independent_runs(tmp_path: Path) -> None:
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"

    first_exit = main(["--output-dir", str(first_dir)])
    second_exit = main(["--output-dir", str(second_dir)])

    assert first_exit == second_exit == 0
    assert (first_dir / "report.json").read_text() == (second_dir / "report.json").read_text()
    assert (first_dir / "report.html").read_text() == (second_dir / "report.html").read_text()
    assert (first_dir / "alert.json").read_text() == (second_dir / "alert.json").read_text()
