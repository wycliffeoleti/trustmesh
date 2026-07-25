from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from trustmesh.email_classifier import PromptConfig, PromptExample, classify_email, load_prompt_config

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
        "category_keywords": {
            "billing": ["invoice", "charge", "charged", "payment", "refund", "billing", "subscription"],
            "technical": ["crash", "crashes", "bug", "error", "broken", "not working", "install"],
            "account": ["password", "log in", "login", "locked out", "username"],
        },
        "default_category": "general",
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


def test_prompt_config_rejects_missing_category_keywords() -> None:
    payload = {
        "version": "test-1",
        "created_at": "2026-01-01T00:00:00Z",
        "system_prompt": "Classify the email.",
        "examples": [{"input": "hi", "category": "general", "summary": "Hi."}],
        "default_category": "general",
    }
    with pytest.raises(ValidationError):
        PromptConfig.model_validate(payload)


def test_prompt_config_rejects_missing_default_category() -> None:
    payload = {
        "version": "test-1",
        "created_at": "2026-01-01T00:00:00Z",
        "system_prompt": "Classify the email.",
        "examples": [{"input": "hi", "category": "general", "summary": "Hi."}],
        "category_keywords": {"billing": ["invoice"]},
    }
    with pytest.raises(ValidationError):
        PromptConfig.model_validate(payload)


def test_prompt_config_rejects_empty_category_keywords() -> None:
    payload = {
        "version": "test-1",
        "created_at": "2026-01-01T00:00:00Z",
        "system_prompt": "Classify the email.",
        "examples": [{"input": "hi", "category": "general", "summary": "Hi."}],
        "category_keywords": {},
        "default_category": "general",
    }
    with pytest.raises(ValidationError):
        PromptConfig.model_validate(payload)


def test_prompt_config_rejects_category_keywords_with_empty_keyword_list() -> None:
    payload = {
        "version": "test-1",
        "created_at": "2026-01-01T00:00:00Z",
        "system_prompt": "Classify the email.",
        "examples": [{"input": "hi", "category": "general", "summary": "Hi."}],
        "category_keywords": {"billing": []},
        "default_category": "general",
    }
    with pytest.raises(ValidationError):
        PromptConfig.model_validate(payload)


def test_prompt_config_rejects_unknown_category_keyword_key() -> None:
    payload = {
        "version": "test-1",
        "created_at": "2026-01-01T00:00:00Z",
        "system_prompt": "Classify the email.",
        "examples": [{"input": "hi", "category": "general", "summary": "Hi."}],
        "category_keywords": {"urgent": ["asap"]},
        "default_category": "general",
    }
    with pytest.raises(ValidationError):
        PromptConfig.model_validate(payload)


def test_prompt_example_rejects_multi_sentence_summary() -> None:
    payload = {
        "input": "My invoice is wrong. Please fix it.",
        "category": "billing",
        "summary": "The invoice is wrong. Please refund it.",
    }
    with pytest.raises(ValidationError):
        PromptExample.model_validate(payload)


def test_prompt_example_rejects_summary_with_trailing_text_after_terminal_punctuation() -> None:
    payload = {
        "input": "My invoice is wrong.",
        "category": "billing",
        "summary": "The invoice is wrong! thanks",
    }
    with pytest.raises(ValidationError):
        PromptExample.model_validate(payload)


def test_prompt_example_accepts_single_sentence_summary_without_terminal_punctuation() -> None:
    example = PromptExample.model_validate(
        {"input": "hi", "category": "general", "summary": "Hi there"}
    )
    assert example.summary == "Hi there"


def test_prompt_config_rejects_multi_sentence_example_summary() -> None:
    payload = {
        "version": "test-1",
        "created_at": "2026-01-01T00:00:00Z",
        "system_prompt": "Classify the email.",
        "examples": [
            {"input": "hi", "category": "general", "summary": "First sentence. Second sentence."},
        ],
        "category_keywords": {"billing": ["invoice"]},
        "default_category": "general",
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
        "category_keywords:\n"
        "  billing: [invoice]\n"
        "default_category: general\n"
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
        "category_keywords": {"billing": ["invoice"]},
        "default_category": "general",
    })
    config_b = PromptConfig.model_validate({
        "version": "test-b",
        "created_at": datetime(2026, 1, 1, tzinfo=UTC),
        "system_prompt": "Classify the email and summarise it in one sentence.",
        "examples": [
            {"input": text, "category": "billing", "summary": "Gizmo transmogrifier billing issue."},
        ],
        "category_keywords": {"billing": ["invoice"]},
        "default_category": "general",
    })

    result_a = classify_email(text, config_a)
    result_b = classify_email(text, config_b)

    assert result_a.category == "technical"
    assert result_a.summary == "Gizmo transmogrifier needs servicing."
    assert result_b.category == "billing"
    assert result_b.summary == "Gizmo transmogrifier billing issue."


def test_classify_email_category_keywords_change_fallback_classification() -> None:
    # Non-exact input (no example matches it), so classification is driven purely by
    # the configured fallback keyword rules. A config-blind implementation would ignore
    # config.category_keywords and produce the same category regardless of config.
    text = "Please help, my widget subscription is acting up."

    config_a = PromptConfig.model_validate({
        "version": "test-d",
        "created_at": datetime(2026, 1, 1, tzinfo=UTC),
        "system_prompt": "Classify the email and summarise it in one sentence.",
        "examples": [{"input": "example", "category": "general", "summary": "An example email."}],
        "category_keywords": {"billing": ["widget"]},
        "default_category": "general",
    })
    config_b = PromptConfig.model_validate({
        "version": "test-e",
        "created_at": datetime(2026, 1, 1, tzinfo=UTC),
        "system_prompt": "Classify the email and summarise it in one sentence.",
        "examples": [{"input": "example", "category": "general", "summary": "An example email."}],
        "category_keywords": {"technical": ["widget"]},
        "default_category": "general",
    })

    result_a = classify_email(text, config_a)
    result_b = classify_email(text, config_b)

    assert result_a.category == "billing"
    assert result_b.category == "technical"


def test_classify_email_default_category_is_configurable() -> None:
    # No example matches and no keyword matches, so the configured default_category
    # must be used verbatim rather than a code-level "general" fallback.
    text = "Just saying hello, nothing specific."

    config = PromptConfig.model_validate({
        "version": "test-f",
        "created_at": datetime(2026, 1, 1, tzinfo=UTC),
        "system_prompt": "Classify the email and summarise it in one sentence.",
        "examples": [{"input": "example", "category": "general", "summary": "An example email."}],
        "category_keywords": {"billing": ["invoice"]},
        "default_category": "account",
    })

    result = classify_email(text, config)

    assert result.category == "account"


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
        "category_keywords": {"billing": ["invoice"], "technical": ["crashes"]},
        "default_category": "general",
    })

    result = classify_email(text, config)

    assert result.category == "technical"
    assert result.summary == "App crash on invoice payment."
