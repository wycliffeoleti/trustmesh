from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from trustmesh.email_classifier_staged_dataset import (
    StagedEmailCase,
    StagedEmailDataset,
    load_staged_dataset,
)

ROOT = Path(__file__).resolve().parent.parent
STAGED_DATASET_PATH = ROOT / "evals" / "email_classifier_staged" / "v1.json"

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

REQUIRED_CASE_FIELDS = [
    "id", "input", "expected_category", "ideal_summary", "expected_difficulty", "notes",
    "review_status",
]


def make_case(**overrides: object) -> dict:
    case = {
        "id": "billing-001",
        "input": "I was charged twice for my subscription, please refund me.",
        "expected_category": "billing",
        "ideal_summary": "Customer was double-charged and wants a refund.",
        "expected_difficulty": "normal",
        "notes": "Baseline unambiguous billing case.",
        "review_status": "draft",
    }
    case.update(overrides)
    return case


def make_cases() -> list[dict]:
    cases = []
    for category in ["billing", "technical", "account", "general"]:
        for i in range(4):
            cases.append(make_case(id=f"{category}-{i:03d}", expected_category=category))
    return cases


def make_dataset(**overrides: object) -> dict:
    dataset = {
        "schema_version": "1",
        "dataset_status": "staged_draft",
        "provenance": {
            "authored_by": "assistant",
            "reviewed_by": None,
            "statement": VALID_STATEMENT,
        },
        "expansion_criterion": VALID_EXPANSION_CRITERION,
        "cases": make_cases(),
    }
    dataset.update(overrides)
    return dataset


# --- case-level schema ---

@pytest.mark.parametrize("missing_field", REQUIRED_CASE_FIELDS)
def test_staged_case_rejects_missing_required_field(missing_field: str) -> None:
    payload = make_case()
    del payload[missing_field]
    with pytest.raises(ValidationError):
        StagedEmailCase.model_validate(payload)


def test_staged_case_rejects_unknown_expected_category() -> None:
    with pytest.raises(ValidationError):
        StagedEmailCase.model_validate(make_case(expected_category="urgent"))


def test_staged_case_rejects_unknown_expected_difficulty() -> None:
    with pytest.raises(ValidationError):
        StagedEmailCase.model_validate(make_case(expected_difficulty="hard"))


def test_staged_case_rejects_non_draft_review_status() -> None:
    with pytest.raises(ValidationError):
        StagedEmailCase.model_validate(make_case(review_status="approved"))


def test_staged_case_rejects_multi_sentence_ideal_summary() -> None:
    with pytest.raises(ValidationError):
        StagedEmailCase.model_validate(
            make_case(ideal_summary="Customer is upset. They want a refund.")
        )


def test_staged_case_accepts_valid_case() -> None:
    case = StagedEmailCase.model_validate(make_case())
    assert case.review_status == "draft"
    assert case.expected_category == "billing"


# --- dataset-level schema ---

def test_staged_dataset_rejects_reviewer_identity_in_provenance() -> None:
    payload = make_dataset()
    payload["provenance"]["reviewed_by"] = "Wicky"
    with pytest.raises(ValidationError):
        StagedEmailDataset.model_validate(payload)


def test_staged_dataset_rejects_non_draft_dataset_status() -> None:
    with pytest.raises(ValidationError):
        StagedEmailDataset.model_validate(make_dataset(dataset_status="golden"))


def test_staged_dataset_rejects_provenance_statement_missing_assistant_authorship() -> None:
    payload = make_dataset()
    payload["provenance"]["statement"] = "This is a great dataset, use it as golden data."
    with pytest.raises(ValidationError):
        StagedEmailDataset.model_validate(payload)


def test_staged_dataset_rejects_wrong_case_count() -> None:
    payload = make_dataset()
    payload["cases"] = payload["cases"][:15]
    with pytest.raises(ValidationError):
        StagedEmailDataset.model_validate(payload)


def test_staged_dataset_rejects_duplicate_case_ids() -> None:
    payload = make_dataset()
    payload["cases"][1] = {**payload["cases"][1], "id": payload["cases"][0]["id"]}
    with pytest.raises(ValidationError):
        StagedEmailDataset.model_validate(payload)


def test_staged_dataset_rejects_unbalanced_categories() -> None:
    payload = make_dataset()
    payload["cases"][0] = {**payload["cases"][0], "expected_category": "technical"}
    with pytest.raises(ValidationError):
        StagedEmailDataset.model_validate(payload)


def test_staged_dataset_accepts_valid_dataset() -> None:
    dataset = StagedEmailDataset.model_validate(make_dataset())
    assert len(dataset.cases) == 16
    assert dataset.provenance.reviewed_by is None


# --- expansion criterion ---

INVALID_EXPANSION_CRITERIA = [
    pytest.param(
        "This dataset may be expanded to the BASWE 50-100-case target once all 16 draft cases "
        "look good and someone records the added human-curated cases.",
        id="missing_named_human_reviewer",
    ),
    pytest.param(
        "This staged draft dataset must not be promoted or expanded toward the BASWE 50-100-case "
        "target until a named human reviewer has reviewed and approved the draft cases and "
        "records the added human-curated cases.",
        id="missing_case_count",
    ),
    pytest.param(
        "This staged draft dataset must not be promoted or expanded until a named human reviewer "
        "has reviewed and approved all 16 draft cases and records the added human-curated cases.",
        id="missing_target_scale",
    ),
    pytest.param(
        "This staged draft dataset must not be promoted or expanded toward the BASWE 50-100-case "
        "target until a named human reviewer has reviewed and approved all 16 draft cases.",
        id="missing_recorded_curation",
    ),
]


def test_staged_dataset_rejects_missing_expansion_criterion() -> None:
    payload = make_dataset()
    del payload["expansion_criterion"]
    with pytest.raises(ValidationError):
        StagedEmailDataset.model_validate(payload)


@pytest.mark.parametrize("statement", INVALID_EXPANSION_CRITERIA)
def test_staged_dataset_rejects_expansion_criterion_missing_required_language(statement: str) -> None:
    payload = make_dataset(expansion_criterion=statement)
    with pytest.raises(ValidationError):
        StagedEmailDataset.model_validate(payload)


def test_staged_dataset_accepts_valid_expansion_criterion() -> None:
    dataset = StagedEmailDataset.model_validate(make_dataset())
    lowered = dataset.expansion_criterion.lower()
    assert "named human reviewer" in lowered
    assert "16" in lowered
    assert "50" in lowered and "100" in lowered


def test_load_staged_dataset_reads_json_from_disk(tmp_path: Path) -> None:
    path = tmp_path / "staged.json"
    path.write_text(json.dumps(make_dataset()))
    dataset = load_staged_dataset(path)
    assert dataset.dataset_status == "staged_draft"


def test_checked_in_staged_draft_dataset_loads_with_expected_shape() -> None:
    dataset = load_staged_dataset(STAGED_DATASET_PATH)
    assert dataset.dataset_status == "staged_draft"
    assert dataset.provenance.reviewed_by is None
    assert len(dataset.cases) == 16

    ids = [case.id for case in dataset.cases]
    assert len(set(ids)) == 16

    difficulties = {case.expected_difficulty for case in dataset.cases}
    assert difficulties == {"normal", "ambiguous", "short", "typo", "mixed_language", "sarcastic"}

    categories = {case.expected_category for case in dataset.cases}
    assert categories == {"billing", "technical", "account", "general"}
    counts: dict[str, int] = {}
    for category in [case.expected_category for case in dataset.cases]:
        counts[category] = counts.get(category, 0) + 1
    assert counts == {"billing": 4, "technical": 4, "account": 4, "general": 4}


def test_checked_in_staged_draft_dataset_declares_expansion_criterion() -> None:
    dataset = load_staged_dataset(STAGED_DATASET_PATH)
    lowered = dataset.expansion_criterion.lower()
    assert "named human reviewer" in lowered
    assert "16" in lowered
    assert "50" in lowered and "100" in lowered
    assert "record" in lowered
