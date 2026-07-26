"""Async batched evaluation engine for the email classifier over the staged dataset.

This runs the deterministic, provider-free `classify_email` (see `trustmesh.email_classifier`)
against every case in a `StagedEmailDataset` and measures real local latency and exact
category match only. Summary relevance and token usage are always recorded as the explicit
sentinel `"unsupported_offline"` rather than a fabricated LLM-as-judge score or a fabricated
token count, because this offline harness has no model provider (see `trustmesh.ports`).
Run-level metadata always carries the dataset's `staged_draft` status and provenance forward
so downstream reporting cannot present these results as golden-dataset or model-evaluated
evidence.
"""
from __future__ import annotations

import asyncio
import time
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict

from trustmesh.email_classifier import EmailCategory, PromptConfig, classify_email
from trustmesh.email_classifier_staged_dataset import (
    StagedDatasetProvenance,
    StagedEmailCase,
    StagedEmailDataset,
)

UnsupportedOfflineDimension = Literal["unsupported_offline"] | None

UNSUPPORTED_OFFLINE: Literal["unsupported_offline"] = "unsupported_offline"


class EmailClassifierCaseResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    case_id: str
    expected_category: EmailCategory
    actual_category: EmailCategory
    category_match: bool
    actual_summary: str
    summary_relevance: UnsupportedOfflineDimension
    token_usage: UnsupportedOfflineDimension
    latency_ms: float


class EmailClassifierRunResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    prompt_version: str
    prompt_created_at: datetime
    dataset_schema_version: str
    dataset_status: Literal["staged_draft"]
    dataset_provenance: StagedDatasetProvenance
    case_results: list[EmailClassifierCaseResult]
    category_pass_rate: float
    summary_relevance: UnsupportedOfflineDimension
    token_usage: UnsupportedOfflineDimension


async def _run_case(
    case: StagedEmailCase,
    prompt_config: PromptConfig,
    semaphore: asyncio.Semaphore,
) -> EmailClassifierCaseResult:
    async with semaphore:
        start = time.perf_counter()
        classification = await asyncio.to_thread(classify_email, case.input, prompt_config)
        latency_ms = (time.perf_counter() - start) * 1000
    return EmailClassifierCaseResult(
        case_id=case.id,
        expected_category=case.expected_category,
        actual_category=classification.category,
        category_match=classification.category == case.expected_category,
        actual_summary=classification.summary,
        summary_relevance=UNSUPPORTED_OFFLINE,
        token_usage=UNSUPPORTED_OFFLINE,
        latency_ms=latency_ms,
    )


async def run_email_classifier_evaluation(
    dataset: StagedEmailDataset,
    prompt_config: PromptConfig,
    *,
    max_concurrency: int = 8,
) -> EmailClassifierRunResult:
    if max_concurrency < 1:
        raise ValueError("max_concurrency must be at least 1")

    semaphore = asyncio.Semaphore(max_concurrency)
    case_results = await asyncio.gather(
        *(_run_case(case, prompt_config, semaphore) for case in dataset.cases)
    )

    match_count = sum(result.category_match for result in case_results)
    category_pass_rate = match_count / len(case_results)

    return EmailClassifierRunResult(
        prompt_version=prompt_config.version,
        prompt_created_at=prompt_config.created_at,
        dataset_schema_version=dataset.schema_version,
        dataset_status=dataset.dataset_status,
        dataset_provenance=dataset.provenance,
        case_results=list(case_results),
        category_pass_rate=category_pass_rate,
        summary_relevance=UNSUPPORTED_OFFLINE,
        token_usage=UNSUPPORTED_OFFLINE,
    )
