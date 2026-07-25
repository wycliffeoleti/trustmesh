from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from trustmesh.email_classifier import PromptConfig, classify_email, load_prompt_config

ROOT = Path(__file__).resolve().parent.parent
V1_PROMPT_PATH = ROOT / "prompts" / "email_classifier" / "v1.yaml"


def make_config() -> PromptConfig:
    return PromptConfig.model_validate({
        "version": "test-1",
        "created_at": datetime(2026, 1, 1, tzinfo=UTC),
        "system_prompt": "Classify the email and summarise it in one sentence.",
        "examples": [
            {"input": "example", "category": "general", "summary": "An example email."},
        ],
    })


def test_prompt_config_rejects_missing_system_prompt() -> None:
    payload = {
        "version": "test-1",
        "created_at": "2026-01-01T00:00:00Z",
        "examples": [{"input": "hi", "category": "general", "summary": "Hi."}],
    }
    with pytest.raises(ValidationError):
        PromptConfig.model_validate(payload)


def test_prompt_config_rejects_unknown_category() -> None:
    payload = {
        "version": "test-1",
        "created_at": "2026-01-01T00:00:00Z",
        "system_prompt": "Classify the email.",
        "examples": [{"input": "hi", "category": "urgent", "summary": "Hi."}],
    }
    with pytest.raises(ValidationError):
        PromptConfig.model_validate(payload)


def test_prompt_config_rejects_empty_examples() -> None:
    payload = {
        "version": "test-1",
        "created_at": "2026-01-01T00:00:00Z",
        "system_prompt": "Classify the email.",
        "examples": [],
    }
    with pytest.raises(ValidationError):
        PromptConfig.model_validate(payload)


def test_load_prompt_config_reads_yaml_from_disk(tmp_path: Path) -> None:
    yaml_path = tmp_path / "v1.yaml"
    yaml_path.write_text(
        "version: '2026.1'\n"
        "created_at: 2026-01-01T00:00:00Z\n"
        "system_prompt: Classify the email.\n"
        "examples:\n"
        "  - input: hi\n"
        "    category: general\n"
        "    summary: Hi there.\n"
    )
    config = load_prompt_config(yaml_path)
    assert config.version == "2026.1"
    assert config.examples[0].category == "general"


def test_checked_in_v1_prompt_config_loads_and_covers_all_categories() -> None:
    config = load_prompt_config(V1_PROMPT_PATH)
    # Pins the internal version to the filename it's checked in under (v1.yaml),
    # so the two identities can't silently drift apart.
    assert config.version == "v1"
    assert config.system_prompt
    categories = {example.category for example in config.examples}
    assert categories == {"billing", "technical", "account", "general"}


@pytest.mark.parametrize(
    ("email_text", "expected_category"),
    [
        ("I was charged twice for the same invoice, please refund me.", "billing"),
        ("The app crashes every time I try to upload a photo.", "technical"),
        ("I can't log in to my account after resetting my password.", "account"),
        ("What are your support hours on weekends?", "general"),
    ],
)
def test_classify_email_returns_expected_category(email_text: str, expected_category: str) -> None:
    result = classify_email(email_text, make_config())
    assert result.category == expected_category


def test_classify_email_summary_is_first_sentence_of_multi_sentence_input() -> None:
    result = classify_email("My invoice is wrong. Please help. Thanks.", make_config())
    assert result.summary == "My invoice is wrong."


def test_classify_email_summary_adds_period_when_input_has_no_terminal_punctuation() -> None:
    result = classify_email("please refund my invoice", make_config())
    assert result.summary == "please refund my invoice."


def test_classify_email_summary_handles_empty_input() -> None:
    result = classify_email("   ", make_config())
    assert result.summary == "No content provided."


def test_classify_email_uses_configured_example_to_override_keyword_fallback() -> None:
    # This text matches no fixed keyword rule, so a config-blind implementation
    # always falls back to "general" regardless of which config is passed in.
    text = "My gizmo transmogrifier needs a checkup before the launch."

    config_a = PromptConfig.model_validate({
        "version": "test-a",
        "created_at": datetime(2026, 1, 1, tzinfo=UTC),
        "system_prompt": "Classify the email and summarise it in one sentence.",
        "examples": [
            {"input": text, "category": "technical", "summary": "Gizmo transmogrifier needs servicing."},
        ],
    })
    config_b = PromptConfig.model_validate({
        "version": "test-b",
        "created_at": datetime(2026, 1, 1, tzinfo=UTC),
        "system_prompt": "Classify the email and summarise it in one sentence.",
        "examples": [
            {"input": text, "category": "billing", "summary": "Gizmo transmogrifier billing issue."},
        ],
    })

    result_a = classify_email(text, config_a)
    result_b = classify_email(text, config_b)

    assert result_a.category == "technical"
    assert result_a.summary == "Gizmo transmogrifier needs servicing."
    assert result_b.category == "billing"
    assert result_b.summary == "Gizmo transmogrifier billing issue."


def test_classify_email_exact_configured_example_wins_over_multi_category_keyword_match() -> None:
    # This text contains both a billing keyword ("invoice") and a technical keyword
    # ("crashes"), so keyword-only fallback would resolve it to "billing" (the first
    # match in _CATEGORY_KEYWORDS iteration order). An exact configured example must
    # still take precedence over that keyword match, not just over a no-keyword-match text.
    text = "My invoice payment app crashes constantly, please help."

    config = PromptConfig.model_validate({
        "version": "test-c",
        "created_at": datetime(2026, 1, 1, tzinfo=UTC),
        "system_prompt": "Classify the email and summarise it in one sentence.",
        "examples": [
            {"input": text, "category": "technical", "summary": "App crash on invoice payment."},
        ],
    })

    result = classify_email(text, config)

    assert result.category == "technical"
    assert result.summary == "App crash on invoice payment."
