"""Deterministic, offline end-to-end CLI for the email classifier evaluation.

Loads the checked-in v1 prompt config and staged draft dataset, runs the async
evaluation engine (`trustmesh.email_classifier_eval_engine`), compares the result
against a checked-in deterministic baseline run, and writes JSON/HTML reports plus a
structured headline-status payload to an output directory. All configuration is via
explicit function arguments or CLI flags with checked-in-file defaults — this module
never reads an environment variable, a secret, or makes a network request (see the
offline-guardrails test for this module). The process exits nonzero only when the
comparison status is `"critical"`; `"ok"` and `"warning"` both exit zero so a CI
workflow can run in review/observe mode before any merge-blocking behavior is
separately authorized.
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from trustmesh.email_classifier import load_prompt_config
from trustmesh.email_classifier_eval_alert import (
    EmailClassifierEvalAlertPayload,
    build_email_classifier_eval_alert_payload,
)
from trustmesh.email_classifier_eval_comparison import (
    RegressionThresholds,
    compare_email_classifier_runs,
)
from trustmesh.email_classifier_eval_engine import (
    EmailClassifierRunResult,
    run_email_classifier_evaluation,
)
from trustmesh.email_classifier_eval_report import (
    EmailClassifierEvalReport,
    build_email_classifier_eval_report,
    render_email_classifier_eval_report_html,
    render_email_classifier_eval_report_json,
)
from trustmesh.email_classifier_staged_dataset import load_staged_dataset

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PROMPT_PATH = ROOT / "prompts" / "email_classifier" / "v1.yaml"
DEFAULT_DATASET_PATH = ROOT / "evals" / "email_classifier_staged" / "v1.json"
DEFAULT_BASELINE_PATH = ROOT / "evals" / "baselines" / "email_classifier_staged_v1.json"
DEFAULT_OUTPUT_DIR = ROOT / "eval-artifacts" / "email_classifier"


def load_baseline(path: Path) -> EmailClassifierRunResult:
    return EmailClassifierRunResult.model_validate_json(path.read_text())


def build_report_id(current: EmailClassifierRunResult) -> str:
    return f"{current.prompt_version}__{current.dataset_schema_version}"


def evaluate_against_baseline(
    prompt_path: Path,
    dataset_path: Path,
    baseline_path: Path,
    thresholds: RegressionThresholds = RegressionThresholds(),
) -> tuple[EmailClassifierEvalReport, EmailClassifierEvalAlertPayload]:
    prompt_config = load_prompt_config(prompt_path)
    dataset = load_staged_dataset(dataset_path)
    baseline = load_baseline(baseline_path)

    current = asyncio.run(run_email_classifier_evaluation(dataset, prompt_config))
    comparison = compare_email_classifier_runs(current, baseline, thresholds=thresholds)
    report_id = build_report_id(current)

    report = build_email_classifier_eval_report(
        report_id=report_id,
        current=current,
        previous=baseline,
        comparison=comparison,
        score_trend=[],
    )
    alert = build_email_classifier_eval_alert_payload(
        report_id=report_id, current=current, comparison=comparison
    )
    return report, alert


def exit_code_for_status(status: str) -> int:
    return 1 if status == "critical" else 0


def write_reports(
    output_dir: Path,
    report: EmailClassifierEvalReport,
    alert: EmailClassifierEvalAlertPayload,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.json").write_text(render_email_classifier_eval_report_json(report) + "\n")
    (output_dir / "report.html").write_text(render_email_classifier_eval_report_html(report))
    (output_dir / "alert.json").write_text(alert.model_dump_json(indent=2) + "\n")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Deterministic offline evaluation of the staged-draft email classifier "
            "against a checked-in baseline. Makes no model/provider or network call."
        )
    )
    parser.add_argument("--prompt-path", type=Path, default=DEFAULT_PROMPT_PATH)
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--baseline-path", type=Path, default=DEFAULT_BASELINE_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)

    report, alert = evaluate_against_baseline(
        args.prompt_path, args.dataset_path, args.baseline_path
    )
    write_reports(args.output_dir, report, alert)
    print(alert.model_dump_json(indent=2))
    return exit_code_for_status(report.status)


if __name__ == "__main__":
    sys.exit(main())
