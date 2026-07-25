"""Customer-support email classifier: one function driven by a versioned, configurable prompt.

This is a safe local deterministic implementation (a test double) — it makes no model/provider
calls and does not semantically execute `system_prompt` as an LLM would; `system_prompt` and
`version` are validated prompt metadata only. An exact match against a configured few-shot example
returns that example's category and summary; anything else falls back to the deterministic
keyword rules and default category declared in the same config (`category_keywords`,
`default_category`), so a config change changes fallback classification.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

EmailCategory = Literal["billing", "technical", "account", "general"]

_SENTENCE_RE = re.compile(r"[^.!?]*[.!?]")


class PromptExample(BaseModel):
    model_config = ConfigDict(frozen=True)

    input: str = Field(min_length=1)
    category: EmailCategory
    summary: str = Field(min_length=1)

    @field_validator("summary")
    @classmethod
    def _summary_is_a_single_sentence(cls, value: str) -> str:
        stripped = value.strip()
        match = _SENTENCE_RE.match(stripped)
        # No terminal punctuation at all is still a single sentence; a match followed
        # by more non-whitespace content means a second sentence follows.
        if match is not None and stripped[match.end():].strip():
            raise ValueError("summary must be a single sentence, not multiple")
        return value


class PromptConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    version: str = Field(min_length=1)
    created_at: datetime
    system_prompt: str = Field(min_length=1)
    examples: list[PromptExample] = Field(min_length=1)
    # The fallback checks categories in this declared order; multi-category emails
    # resolve to the first match. Keys are validated against EmailCategory.
    category_keywords: dict[EmailCategory, list[str]] = Field(min_length=1)
    default_category: EmailCategory

    @field_validator("category_keywords")
    @classmethod
    def _category_keywords_are_non_empty(
        cls, value: dict[EmailCategory, list[str]]
    ) -> dict[EmailCategory, list[str]]:
        for category, keywords in value.items():
            if not keywords or any(not keyword.strip() for keyword in keywords):
                raise ValueError(f"category_keywords for {category!r} must contain non-empty keywords")
        return value


@dataclass(frozen=True)
class EmailClassification:
    category: EmailCategory
    summary: str


def load_prompt_config(path: Path) -> PromptConfig:
    raw = yaml.safe_load(path.read_text())
    return PromptConfig.model_validate(raw)


def _summarize(email_text: str) -> str:
    stripped = email_text.strip()
    if not stripped:
        return "No content provided."
    match = _SENTENCE_RE.match(stripped)
    sentence = match.group(0).strip() if match else stripped
    return sentence if sentence[-1] in ".!?" else f"{sentence}."


def classify_email(email_text: str, config: PromptConfig) -> EmailClassification:
    for example in config.examples:
        if example.input == email_text:
            return EmailClassification(example.category, example.summary)
    text_l = email_text.lower()
    for category, keywords in config.category_keywords.items():
        if any(keyword in text_l for keyword in keywords):
            return EmailClassification(category, _summarize(email_text))
    return EmailClassification(config.default_category, _summarize(email_text))
