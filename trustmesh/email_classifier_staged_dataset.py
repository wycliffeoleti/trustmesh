"""Typed contract for the email classifier's staged draft dataset.

This is a small, versioned Pydantic schema for a *staged draft* — assistant-authored
synthetic seed data, not a human-curated golden dataset (see `StagedDatasetProvenance`).
It must not be used as a basis for production, human-reviewed, or model-evaluated claims
until a named human reviewer completes review and the dataset is explicitly promoted.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from trustmesh.email_classifier import EmailCategory

CaseDifficulty = Literal["normal", "ambiguous", "short", "typo", "mixed_language", "sarcastic"]

_EXPECTED_CASE_COUNT = 16
_CASES_PER_CATEGORY = 4
_CATEGORIES: tuple[EmailCategory, ...] = ("billing", "technical", "account", "general")
_SENTENCE_RE = re.compile(r"[^.!?]*[.!?]")


def _is_single_sentence(value: str) -> bool:
    stripped = value.strip()
    match = _SENTENCE_RE.match(stripped)
    return match is None or not stripped[match.end():].strip()


class StagedEmailCase(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str = Field(min_length=1)
    input: str = Field(min_length=1)
    expected_category: EmailCategory
    ideal_summary: str = Field(min_length=1)
    expected_difficulty: CaseDifficulty
    notes: str = Field(min_length=1)
    review_status: Literal["draft"]

    @field_validator("ideal_summary")
    @classmethod
    def _ideal_summary_is_a_single_sentence(cls, value: str) -> str:
        if not _is_single_sentence(value):
            raise ValueError("ideal_summary must be a single sentence, not multiple")
        return value


class StagedDatasetProvenance(BaseModel):
    model_config = ConfigDict(frozen=True)

    authored_by: Literal["assistant"]
    reviewed_by: None = None
    statement: str = Field(min_length=1)

    @field_validator("statement")
    @classmethod
    def _statement_declares_assistant_authorship_and_review_requirement(cls, value: str) -> str:
        lowered = value.lower()
        if "assistant" not in lowered or "human review" not in lowered or "golden" not in lowered:
            raise ValueError(
                "provenance statement must declare assistant authorship, that human review is "
                "required, and that this is not a golden dataset"
            )
        return value


class StagedEmailDataset(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_version: str = Field(min_length=1)
    dataset_status: Literal["staged_draft"]
    provenance: StagedDatasetProvenance
    expansion_criterion: str = Field(min_length=1)
    cases: list[StagedEmailCase]

    @field_validator("expansion_criterion")
    @classmethod
    def _expansion_criterion_requires_named_reviewer_and_full_case_review(cls, value: str) -> str:
        lowered = value.lower()
        required_phrases = ("named human reviewer", "16", "50", "100", "record", "human-curated")
        if not all(phrase in lowered for phrase in required_phrases):
            raise ValueError(
                "expansion_criterion must declare that promotion or expansion toward the BASWE "
                "50-100-case target requires a named human reviewer to review and approve all 16 "
                "draft cases, and that added human-curated cases are recorded"
            )
        return value

    @field_validator("cases")
    @classmethod
    def _cases_are_16_with_unique_ids_and_balanced_categories(
        cls, value: list[StagedEmailCase]
    ) -> list[StagedEmailCase]:
        if len(value) != _EXPECTED_CASE_COUNT:
            raise ValueError(f"staged dataset must contain exactly {_EXPECTED_CASE_COUNT} cases")

        ids = [case.id for case in value]
        if len(set(ids)) != len(ids):
            raise ValueError("staged dataset case ids must be unique")

        counts: dict[str, int] = {}
        for case in value:
            counts[case.expected_category] = counts.get(case.expected_category, 0) + 1
        for category in _CATEGORIES:
            if counts.get(category, 0) != _CASES_PER_CATEGORY:
                raise ValueError(
                    f"staged dataset must contain exactly {_CASES_PER_CATEGORY} cases for "
                    f"category {category!r}"
                )
        return value


def load_staged_dataset(path: Path) -> StagedEmailDataset:
    raw = json.loads(path.read_text())
    return StagedEmailDataset.model_validate(raw)
