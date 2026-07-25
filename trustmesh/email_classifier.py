"""Customer-support email classifier: one function driven by a versioned, configurable prompt.

This is a safe local deterministic implementation (a test double) — it makes no model/provider
calls. An exact match against a configured few-shot example returns that example's category and
summary; anything else falls back to fixed keyword rules.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

EmailCategory = Literal["billing", "technical", "account", "general"]

# The fallback checks categories in this declared order; multi-category emails resolve to the first match.
_CATEGORY_KEYWORDS: dict[EmailCategory, tuple[str, ...]] = {
    "billing": ("invoice", "charge", "charged", "payment", "refund", "billing", "subscription"),
    "technical": ("crash", "crashes", "bug", "error", "broken", "not working", "install"),
    "account": ("password", "log in", "login", "locked out", "username"),
}
_SENTENCE_RE = re.compile(r"[^.!?]*[.!?]")


class PromptExample(BaseModel):
    model_config = ConfigDict(frozen=True)

    input: str = Field(min_length=1)
    category: EmailCategory
    summary: str = Field(min_length=1)


class PromptConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    version: str = Field(min_length=1)
    created_at: datetime
    system_prompt: str = Field(min_length=1)
    examples: list[PromptExample] = Field(min_length=1)


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
    for category, keywords in _CATEGORY_KEYWORDS.items():
        if any(keyword in text_l for keyword in keywords):
            return EmailClassification(category, _summarize(email_text))
    return EmailClassification("general", _summarize(email_text))
