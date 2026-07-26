from __future__ import annotations

import asyncio
import threading
import time
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from trustmesh.email_classifier import EmailCategory, EmailClassification, PromptConfig
from trustmesh.email_classifier_eval_engine import (
    UNSUPPORTED_OFFLINE,
    EmailClassifierCaseResult,
    EmailClassifierRunResult,
    run_email_classifier_evaluation,
)
from trustmesh.email_classifier_staged_dataset import StagedEmailDataset

VALID_STATEMENT = (
    "This dataset was authored entirely by an AI assistant as synthetic seed data. "
    "It has not been reviewed by a human and must not be treated as a golden dataset "
    "until a human reviewer completes review."
)

VALID_EXPANSION_CRITERION = (
    "This staged draft dataset must not be promoted or expanded toward the BASWE 50-100-case "
    "target until a named human reviewer has reviewed and approved all 16 draft cases and "
    "records the added human-curated cases."
)


def make_prompt_config(**overrides: object) -> PromptConfig:
    payload: dict[str, object] = {
        "version": "test-eval-1",
        "created_at": datetime(2026, 1, 1, tzinfo=UTC),
        "system_prompt": "Classify the email and summarise it in one sentence.",
        "examples": [
            {"input": "example", "category": "general", "summary": "An example email."},
        ],
        "category_keywords": {
            "billing": ["invoice", "charge", "refund"],
            "technical": ["crash", "bug", "error"],
            "account": ["password", "login", "locked out"],
        },
        "default_category": "general",
    }
    payload.update(overrides)
    return PromptConfig.model_validate(payload)


def make_case(case_id: str, input_text: str, expected_category: EmailCategory) -> dict[str, object]:
    return {
        "id": case_id,
        "input": input_text,
        "expected_category": expected_category,
        "ideal_summary": "Synthetic summary for testing.",
        "expected_difficulty": "normal",
        "notes": "Synthetic case built for the eval engine test suite.",
        "review_status": "draft",
    }


def make_all_correct_dataset() -> StagedEmailDataset:
    cases = [
        make_case("billing-001", "invoice charge issue one", "billing"),
        make_case("billing-002", "invoice charge issue two", "billing"),
        make_case("billing-003", "invoice charge issue three", "billing"),
        make_case("billing-004", "invoice charge issue four", "billing"),
        make_case("technical-001", "app crash bug one", "technical"),
        make_case("technical-002", "app crash bug two", "technical"),
        make_case("technical-003", "app crash bug three", "technical"),
        make_case("technical-004", "app crash bug four", "technical"),
        make_case("account-001", "password login issue one", "account"),
        make_case("account-002", "password login issue two", "account"),
        make_case("account-003", "password login issue three", "account"),
        make_case("account-004", "password login issue four", "account"),
        make_case("general-001", "just saying hello one", "general"),
        make_case("general-002", "just saying hello two", "general"),
        make_case("general-003", "just saying hello three", "general"),
        make_case("general-004", "just saying hello four", "general"),
    ]
    return StagedEmailDataset.model_validate({
        "schema_version": "test-1",
        "dataset_status": "staged_draft",
        "provenance": {
            "authored_by": "assistant",
            "reviewed_by": None,
            "statement": VALID_STATEMENT,
        },
        "expansion_criterion": VALID_EXPANSION_CRITERION,
        "cases": cases,
    })


def make_one_mismatch_dataset() -> StagedEmailDataset:
    dataset = make_all_correct_dataset()
    cases = list(dataset.cases)
    mismatched = cases[0].model_copy(update={"input": "just saying hello mismatched"})
    cases[0] = mismatched
    return dataset.model_copy(update={"cases": cases})


# --- Cycle 1: per-case and run-level result schemas ---


def test_case_result_records_true_category_match_when_actual_equals_expected() -> None:
    result = EmailClassifierCaseResult(
        case_id="billing-001",
        expected_category="billing",
        actual_category="billing",
        category_match=True,
        actual_summary="Customer billing issue.",
        summary_relevance=UNSUPPORTED_OFFLINE,
        token_usage=UNSUPPORTED_OFFLINE,
        latency_ms=0.5,
    )
    assert result.category_match is True


def test_case_result_records_false_category_match_when_actual_differs_from_expected() -> None:
    result = EmailClassifierCaseResult(
        case_id="billing-001",
        expected_category="billing",
        actual_category="general",
        category_match=False,
        actual_summary="Customer billing issue.",
        summary_relevance=UNSUPPORTED_OFFLINE,
        token_usage=UNSUPPORTED_OFFLINE,
        latency_ms=0.5,
    )
    assert result.category_match is False


@pytest.mark.parametrize("value", [UNSUPPORTED_OFFLINE, None])
def test_case_result_accepts_unsupported_offline_or_none_for_summary_relevance(
    value: str | None,
) -> None:
    result = EmailClassifierCaseResult(
        case_id="billing-001",
        expected_category="billing",
        actual_category="billing",
        category_match=True,
        actual_summary="Customer billing issue.",
        summary_relevance=value,
        token_usage=UNSUPPORTED_OFFLINE,
        latency_ms=0.5,
    )
    assert result.summary_relevance == value


def test_case_result_rejects_fabricated_numeric_summary_relevance_score() -> None:
    with pytest.raises(ValidationError):
        EmailClassifierCaseResult(
            case_id="billing-001",
            expected_category="billing",
            actual_category="billing",
            category_match=True,
            actual_summary="Customer billing issue.",
            summary_relevance=4,  # type: ignore[arg-type]
            token_usage=UNSUPPORTED_OFFLINE,
            latency_ms=0.5,
        )


def test_case_result_rejects_fabricated_numeric_token_usage() -> None:
    with pytest.raises(ValidationError):
        EmailClassifierCaseResult(
            case_id="billing-001",
            expected_category="billing",
            actual_category="billing",
            category_match=True,
            actual_summary="Customer billing issue.",
            summary_relevance=UNSUPPORTED_OFFLINE,
            token_usage=123,  # type: ignore[arg-type]
            latency_ms=0.5,
        )


def test_run_result_holds_prompt_and_dataset_metadata_and_case_results() -> None:
    dataset = make_all_correct_dataset()
    prompt_config = make_prompt_config()
    case_result = EmailClassifierCaseResult(
        case_id="billing-001",
        expected_category="billing",
        actual_category="billing",
        category_match=True,
        actual_summary="Customer billing issue.",
        summary_relevance=UNSUPPORTED_OFFLINE,
        token_usage=UNSUPPORTED_OFFLINE,
        latency_ms=0.5,
    )
    run_result = EmailClassifierRunResult(
        prompt_version=prompt_config.version,
        prompt_created_at=prompt_config.created_at,
        dataset_schema_version=dataset.schema_version,
        dataset_status=dataset.dataset_status,
        dataset_provenance=dataset.provenance,
        case_results=[case_result],
        category_pass_rate=1.0,
        summary_relevance=UNSUPPORTED_OFFLINE,
        token_usage=UNSUPPORTED_OFFLINE,
    )
    assert run_result.dataset_status == "staged_draft"
    assert run_result.dataset_provenance == dataset.provenance
    assert run_result.prompt_version == "test-eval-1"
    assert run_result.case_results == [case_result]


def test_run_result_rejects_non_staged_draft_dataset_status() -> None:
    dataset = make_all_correct_dataset()
    prompt_config = make_prompt_config()
    with pytest.raises(ValidationError):
        EmailClassifierRunResult(
            prompt_version=prompt_config.version,
            prompt_created_at=prompt_config.created_at,
            dataset_schema_version=dataset.schema_version,
            dataset_status="golden",  # type: ignore[arg-type]
            dataset_provenance=dataset.provenance,
            case_results=[],
            category_pass_rate=1.0,
            summary_relevance=UNSUPPORTED_OFFLINE,
            token_usage=UNSUPPORTED_OFFLINE,
        )


# --- Cycle 2: async batched runner ---


def test_run_email_classifier_evaluation_is_a_coroutine_function() -> None:
    assert asyncio.iscoroutinefunction(run_email_classifier_evaluation)


def test_run_email_classifier_evaluation_returns_one_case_result_per_dataset_case() -> None:
    dataset = make_all_correct_dataset()
    prompt_config = make_prompt_config()

    result = asyncio.run(run_email_classifier_evaluation(dataset, prompt_config))

    assert len(result.case_results) == len(dataset.cases)
    assert [r.case_id for r in result.case_results] == [c.id for c in dataset.cases]


def test_run_email_classifier_evaluation_records_actual_category_and_summary_and_latency() -> None:
    dataset = make_all_correct_dataset()
    prompt_config = make_prompt_config()

    result = asyncio.run(run_email_classifier_evaluation(dataset, prompt_config))

    first = result.case_results[0]
    assert first.actual_category == "billing"
    assert first.category_match is True
    assert first.actual_summary
    assert first.latency_ms >= 0.0


def test_run_email_classifier_evaluation_marks_mismatched_case_as_category_match_false() -> None:
    dataset = make_one_mismatch_dataset()
    prompt_config = make_prompt_config()

    result = asyncio.run(run_email_classifier_evaluation(dataset, prompt_config))

    mismatched = next(r for r in result.case_results if r.case_id == "billing-001")
    assert mismatched.category_match is False
    assert mismatched.actual_category == "general"


def test_run_email_classifier_evaluation_computes_overall_category_pass_rate() -> None:
    dataset = make_one_mismatch_dataset()
    prompt_config = make_prompt_config()

    result = asyncio.run(run_email_classifier_evaluation(dataset, prompt_config))

    assert result.category_pass_rate == pytest.approx(15 / 16)


def test_run_email_classifier_evaluation_run_level_dimensions_are_unsupported_offline() -> None:
    dataset = make_all_correct_dataset()
    prompt_config = make_prompt_config()

    result = asyncio.run(run_email_classifier_evaluation(dataset, prompt_config))

    assert result.summary_relevance == UNSUPPORTED_OFFLINE
    assert result.token_usage == UNSUPPORTED_OFFLINE
    for case_result in result.case_results:
        assert case_result.summary_relevance == UNSUPPORTED_OFFLINE
        assert case_result.token_usage == UNSUPPORTED_OFFLINE


def test_run_email_classifier_evaluation_preserves_dataset_and_prompt_metadata() -> None:
    dataset = make_all_correct_dataset()
    prompt_config = make_prompt_config()

    result = asyncio.run(run_email_classifier_evaluation(dataset, prompt_config))

    assert result.dataset_status == "staged_draft"
    assert result.dataset_provenance == dataset.provenance
    assert result.dataset_schema_version == dataset.schema_version
    assert result.prompt_version == prompt_config.version
    assert result.prompt_created_at == prompt_config.created_at


def test_run_email_classifier_evaluation_rejects_non_positive_max_concurrency() -> None:
    dataset = make_all_correct_dataset()
    prompt_config = make_prompt_config()

    with pytest.raises(ValueError, match="max_concurrency"):
        asyncio.run(
            run_email_classifier_evaluation(dataset, prompt_config, max_concurrency=0)
        )


def test_run_email_classifier_evaluation_respects_bounded_concurrency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = make_all_correct_dataset()
    prompt_config = make_prompt_config()

    lock = threading.Lock()
    state = {"current": 0, "max": 0}

    def fake_classify_email(email_text: str, config: PromptConfig) -> EmailClassification:
        with lock:
            state["current"] += 1
            state["max"] = max(state["max"], state["current"])
        time.sleep(0.05)
        with lock:
            state["current"] -= 1
        return EmailClassification("general", "Stub summary.")

    monkeypatch.setattr(
        "trustmesh.email_classifier_eval_engine.classify_email", fake_classify_email
    )

    result = asyncio.run(
        run_email_classifier_evaluation(dataset, prompt_config, max_concurrency=4)
    )

    assert len(result.case_results) == 16
    assert state["max"] <= 4
    assert state["max"] > 1


def test_checked_in_staged_dataset_and_v1_prompt_run_end_to_end() -> None:
    from pathlib import Path

    from trustmesh.email_classifier import load_prompt_config
    from trustmesh.email_classifier_staged_dataset import load_staged_dataset

    root = Path(__file__).resolve().parent.parent
    dataset = load_staged_dataset(root / "evals" / "email_classifier_staged" / "v1.json")
    prompt_config = load_prompt_config(root / "prompts" / "email_classifier" / "v1.yaml")

    result = asyncio.run(run_email_classifier_evaluation(dataset, prompt_config))

    assert len(result.case_results) == 16
    assert 0.0 <= result.category_pass_rate <= 1.0
    assert result.dataset_status == "staged_draft"
