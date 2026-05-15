"""Tests for the model resolver + catalog access."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch
from urllib.error import HTTPError, URLError

import pytest

from mnemo.config import Settings
from mnemo.models_catalog import (
    COPILOT_SUBPROCESS_MARKER,
    FAMILY_ALIASES,
    FAMILY_TO_SETTING,
    MODEL_SHORTNAMES,
    ModelResolutionError,
    fetch_catalog_models,
    is_family_alias,
    is_subprocess_marker,
    list_shortnames,
    resolve_model,
    resolve_token_from_env,
    _extract_model_ids,
)


# ---------------------------------------------------------------------------
# Constants / membership
# ---------------------------------------------------------------------------


def test_family_aliases_are_the_canonical_four() -> None:
    assert set(FAMILY_ALIASES) == {"gpt", "claude", "sonnet", "opus"}


def test_family_to_setting_keys_match_aliases() -> None:
    assert set(FAMILY_TO_SETTING.keys()) == set(FAMILY_ALIASES)


def test_subprocess_marker_is_copilot() -> None:
    assert COPILOT_SUBPROCESS_MARKER == "copilot"
    assert is_subprocess_marker("copilot")
    assert not is_subprocess_marker("gpt-5-mini")


def test_shortnames_include_expected_basics() -> None:
    names = set(list_shortnames())
    # Always at least these for user expectations.
    for required in (
        "gpt-5-mini",
        "gpt-5",
        "claude-sonnet-4-6",
        "claude-opus-4-6",
    ):
        assert required in names, f"missing required short-name: {required}"


# ---------------------------------------------------------------------------
# resolve_model — happy paths
# ---------------------------------------------------------------------------


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(persist_dir=tmp_path / "data")


def test_resolve_short_name(settings: Settings) -> None:
    assert resolve_model("gpt-5-mini", settings) == "openai/gpt-5-mini"
    assert resolve_model("claude-opus-4-6", settings) == "anthropic/claude-opus-4-6"


def test_resolve_full_id_passthrough(settings: Settings) -> None:
    assert resolve_model("openai/gpt-5.4", settings) == "openai/gpt-5.4"
    assert resolve_model("anthropic/claude-opus-4-7", settings) == "anthropic/claude-opus-4-7"


def test_resolve_family_uses_settings(settings: Settings) -> None:
    # Defaults
    assert resolve_model("gpt", settings) == "openai/gpt-5-mini"
    assert resolve_model("opus", settings) == "anthropic/claude-opus-4-6"


def test_resolve_family_with_override(tmp_path) -> None:
    s = Settings(
        persist_dir=tmp_path / "data",
        family_gpt="gpt-5",
        family_opus="claude-opus-4-6",
    )
    assert resolve_model("gpt", s) == "openai/gpt-5"
    assert resolve_model("opus", s) == "anthropic/claude-opus-4-6"


def test_resolve_family_can_point_to_full_id(tmp_path) -> None:
    """If the family setting is a full id, resolution short-circuits to it."""
    s = Settings(
        persist_dir=tmp_path / "data",
        family_gpt="openai/gpt-5-custom",
    )
    assert resolve_model("gpt", s) == "openai/gpt-5-custom"


def test_resolve_trims_whitespace(settings: Settings) -> None:
    assert resolve_model("  gpt-5-mini  ", settings) == "openai/gpt-5-mini"


# ---------------------------------------------------------------------------
# resolve_model — error paths
# ---------------------------------------------------------------------------


def test_resolve_rejects_copilot_marker(settings: Settings) -> None:
    """Callers must dispatch on `copilot` BEFORE calling the resolver."""
    with pytest.raises(ModelResolutionError, match="copilot"):
        resolve_model("copilot", settings)


def test_resolve_rejects_unknown_shortname(settings: Settings) -> None:
    with pytest.raises(ModelResolutionError, match="Unknown model"):
        resolve_model("totally-fake-model", settings)


@pytest.mark.parametrize("bad", ["", "   ", None])
def test_resolve_rejects_empty(settings: Settings, bad: Any) -> None:
    with pytest.raises(ModelResolutionError):
        resolve_model(bad, settings)


def test_resolve_family_with_empty_setting_raises(tmp_path) -> None:
    s = Settings(persist_dir=tmp_path / "data", family_gpt="   ")
    with pytest.raises(ModelResolutionError, match="empty"):
        resolve_model("gpt", s)


def test_is_family_alias() -> None:
    assert is_family_alias("gpt")
    assert is_family_alias("opus")
    assert not is_family_alias("gpt-5")


# ---------------------------------------------------------------------------
# resolve_token_from_env
# ---------------------------------------------------------------------------


def test_token_resolution_priority(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MNEMO_GHMODELS_TOKEN", "mnemo-pat")
    monkeypatch.setenv("GH_TOKEN", "gh-pat")
    monkeypatch.setenv("GITHUB_TOKEN", "gha-pat")
    assert resolve_token_from_env() == "mnemo-pat"


def test_token_resolution_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MNEMO_GHMODELS_TOKEN", raising=False)
    monkeypatch.setenv("GH_TOKEN", "gh-pat")
    monkeypatch.setenv("GITHUB_TOKEN", "gha-pat")
    assert resolve_token_from_env() == "gh-pat"


def test_token_resolution_falls_to_last(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MNEMO_GHMODELS_TOKEN", raising=False)
    monkeypatch.delenv("GH_TOKEN", raising=False)
    monkeypatch.setenv("GITHUB_TOKEN", "gha-pat")
    assert resolve_token_from_env() == "gha-pat"


def test_token_resolution_returns_none_when_absent() -> None:
    # conftest already strips MNEMO_* vars; ensure non-MNEMO ones are also unset
    # The other two might be set in the dev's real env; use monkeypatch here
    with patch.dict("os.environ", {}, clear=True):
        assert resolve_token_from_env() is None


# ---------------------------------------------------------------------------
# Catalog fetch (mocked HTTP)
# ---------------------------------------------------------------------------


def test_extract_model_ids_from_list_shape() -> None:
    data = [
        {"id": "openai/gpt-5", "name": "GPT-5"},
        {"id": "anthropic/claude-opus-4-6"},
        {"name": "junk-without-slash"},  # filtered
    ]
    ids = _extract_model_ids(data)
    assert ids == ["openai/gpt-5", "anthropic/claude-opus-4-6"]


def test_extract_model_ids_from_models_wrapper() -> None:
    data = {"models": [{"id": "openai/gpt-5"}]}
    assert _extract_model_ids(data) == ["openai/gpt-5"]


def test_extract_model_ids_from_data_wrapper() -> None:
    data = {"data": [{"id": "openai/gpt-5"}]}
    assert _extract_model_ids(data) == ["openai/gpt-5"]


def test_extract_model_ids_returns_empty_for_unexpected_shape() -> None:
    assert _extract_model_ids("not a list or dict") == []
    assert _extract_model_ids(None) == []


def _make_urlopen_response(status: int, body: bytes):
    """Build a fake context-manager-friendly urlopen response."""
    mock_resp = MagicMock()
    mock_resp.read.return_value = body
    mock_resp.status = status
    mock_resp.__enter__ = lambda self: mock_resp
    mock_resp.__exit__ = lambda self, *args: None
    return mock_resp


def test_fetch_catalog_models_success() -> None:
    body = json.dumps([{"id": "openai/gpt-5"}, {"id": "openai/gpt-5-mini"}]).encode()
    with patch("urllib.request.urlopen", return_value=_make_urlopen_response(200, body)):
        ids = fetch_catalog_models("test-token", "https://example.com/catalog")
    assert ids == ["openai/gpt-5", "openai/gpt-5-mini"]


def test_fetch_catalog_models_http_error_returns_empty() -> None:
    err = HTTPError("https://example.com", 401, "Unauthorized", {}, None)
    with patch("urllib.request.urlopen", side_effect=err):
        ids = fetch_catalog_models("bad-token", "https://example.com/catalog")
    assert ids == []


def test_fetch_catalog_models_network_error_returns_empty() -> None:
    err = URLError("connection refused")
    with patch("urllib.request.urlopen", side_effect=err):
        ids = fetch_catalog_models("token", "https://example.com/catalog")
    assert ids == []


def test_fetch_catalog_models_bad_json_returns_empty() -> None:
    with patch(
        "urllib.request.urlopen",
        return_value=_make_urlopen_response(200, b"not valid json"),
    ):
        ids = fetch_catalog_models("token", "https://example.com/catalog")
    assert ids == []
