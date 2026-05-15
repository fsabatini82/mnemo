"""Tests for the runtime-config classification + validation."""

from __future__ import annotations

import pytest

from mnemo.runtime_config import (
    CATEGORY,
    ConfigCategory,
    InvalidConfigValue,
    UnknownConfigKey,
    all_keys_by_category,
    classify,
    is_known,
    validate_value,
)


# ---------------------------------------------------------------------------
# classify() / is_known()
# ---------------------------------------------------------------------------


def test_known_keys_are_in_category_map() -> None:
    for key, cat in CATEGORY.items():
        assert is_known(key)
        assert classify(key) == cat


def test_unknown_key_raises() -> None:
    with pytest.raises(UnknownConfigKey):
        classify("NOT_A_REAL_KEY")


@pytest.mark.parametrize(
    "key,expected",
    [
        ("MNEMO_GHMODELS_MODEL", ConfigCategory.EDITABLE_NOW),
        ("MNEMO_FAMILY_GPT", ConfigCategory.EDITABLE_NOW),
        ("MNEMO_TOP_K", ConfigCategory.EDITABLE_NOW),
        ("MNEMO_CHUNK_SIZE", ConfigCategory.EDITABLE_WITH_RESTART),
        ("MNEMO_STORE", ConfigCategory.EDITABLE_WITH_RESTART),
        ("MNEMO_PROJECT", ConfigCategory.READONLY),
        ("MNEMO_PERSIST_DIR", ConfigCategory.READONLY),
        ("MNEMO_COPILOT_BIN", ConfigCategory.READONLY),
        ("MNEMO_GHMODELS_ENDPOINT", ConfigCategory.READONLY),
        ("MNEMO_GHMODELS_TOKEN", ConfigCategory.HIDDEN),
        ("GH_TOKEN", ConfigCategory.HIDDEN),
        ("GITHUB_TOKEN", ConfigCategory.HIDDEN),
    ],
)
def test_classification_examples(key: str, expected: ConfigCategory) -> None:
    assert classify(key) == expected


def test_all_categories_have_entries() -> None:
    grouped = all_keys_by_category()
    for cat in ConfigCategory:
        assert grouped[cat], f"Category {cat} is empty — should have at least one key"


# ---------------------------------------------------------------------------
# validate_value()
# ---------------------------------------------------------------------------


def test_validate_hidden_raises() -> None:
    with pytest.raises(InvalidConfigValue, match="Token"):
        validate_value("MNEMO_GHMODELS_TOKEN", "any-token")


def test_validate_readonly_raises() -> None:
    with pytest.raises(InvalidConfigValue, match="read-only"):
        validate_value("MNEMO_PROJECT", "alpha")


def test_validate_unknown_raises() -> None:
    with pytest.raises(UnknownConfigKey):
        validate_value("NOT_A_REAL_KEY", "value")


def test_validate_non_string_raises() -> None:
    with pytest.raises(InvalidConfigValue):
        validate_value("MNEMO_TOP_K", 5)  # type: ignore[arg-type]


# --- Settings-backed validation -------------------------------------------


def test_validate_top_k_accepts_positive_int_string() -> None:
    validate_value("MNEMO_TOP_K", "10")  # no exception


def test_validate_top_k_rejects_zero() -> None:
    with pytest.raises(InvalidConfigValue):
        validate_value("MNEMO_TOP_K", "0")


def test_validate_top_k_rejects_non_int() -> None:
    with pytest.raises(InvalidConfigValue):
        validate_value("MNEMO_TOP_K", "abc")


def test_validate_chunk_size_accepts() -> None:
    validate_value("MNEMO_CHUNK_SIZE", "512")


def test_validate_chunk_size_rejects_negative() -> None:
    with pytest.raises(InvalidConfigValue):
        validate_value("MNEMO_CHUNK_SIZE", "-1")


def test_validate_chunk_overlap_accepts_zero() -> None:
    """chunk_overlap allows 0 (`ge=0`), unlike chunk_size."""
    validate_value("MNEMO_CHUNK_OVERLAP", "0")


def test_validate_reasoning_effort_accepts_canonical() -> None:
    for v in ("minimal", "low", "medium", "high"):
        validate_value("MNEMO_GHMODELS_REASONING_EFFORT", v)


def test_validate_reasoning_effort_rejects_other() -> None:
    with pytest.raises(InvalidConfigValue):
        validate_value("MNEMO_GHMODELS_REASONING_EFFORT", "ultra")


def test_validate_family_alias_accepts_any_string() -> None:
    # Family aliases are str without enum constraint — any non-empty string ok.
    validate_value("MNEMO_FAMILY_GPT", "gpt-5")
    validate_value("MNEMO_FAMILY_GPT", "openai/gpt-5.4")


# --- Non-Settings env-only ------------------------------------------------


def test_validate_copilot_timeout_positive() -> None:
    validate_value("MNEMO_COPILOT_TIMEOUT", "60")


def test_validate_copilot_timeout_rejects_zero() -> None:
    with pytest.raises(InvalidConfigValue, match="> 0"):
        validate_value("MNEMO_COPILOT_TIMEOUT", "0")


def test_validate_copilot_timeout_rejects_non_int() -> None:
    with pytest.raises(InvalidConfigValue, match="integer"):
        validate_value("MNEMO_COPILOT_TIMEOUT", "abc")


def test_validate_audit_file_budget_accepts() -> None:
    validate_value("MNEMO_AUDIT_FILE_BUDGET", "10000")


def test_validate_audit_file_budget_rejects_below_minimum() -> None:
    with pytest.raises(InvalidConfigValue, match="at least 500"):
        validate_value("MNEMO_AUDIT_FILE_BUDGET", "100")
