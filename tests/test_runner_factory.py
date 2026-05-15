"""Tests for the runner factory (CopilotRunner vs GitHubModelsRunner dispatch)."""

from __future__ import annotations

import pytest

from mnemo.config import Settings
from mnemo.ingestion.agents.copilot.runner import CopilotRunner
from mnemo.ingestion.agents.gh_models.runner import GitHubModelsRunner
from mnemo.ingestion.agents.runner_factory import RunnerBuildError, build_runner


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(persist_dir=tmp_path / "data")


def test_no_agentic_raises(settings: Settings) -> None:
    with pytest.raises(RunnerBuildError, match="deterministic"):
        build_runner(None, settings)


def test_copilot_marker_returns_subprocess_runner(settings: Settings) -> None:
    r = build_runner("copilot", settings)
    assert isinstance(r, CopilotRunner)


def test_shortname_returns_ghmodels(settings: Settings) -> None:
    r = build_runner("gpt-5-mini", settings)
    assert isinstance(r, GitHubModelsRunner)
    assert "openai/gpt-5-mini" in r.describe()


def test_family_alias_resolves_via_settings(tmp_path) -> None:
    s = Settings(persist_dir=tmp_path / "data", family_opus="claude-opus-4-6")
    r = build_runner("opus", s)
    assert isinstance(r, GitHubModelsRunner)
    assert "anthropic/claude-opus-4-6" in r.describe()


def test_full_id_passthrough(settings: Settings) -> None:
    r = build_runner("openai/gpt-5.4", settings)
    assert isinstance(r, GitHubModelsRunner)
    assert "openai/gpt-5.4" in r.describe()


def test_unknown_shortname_raises(settings: Settings) -> None:
    with pytest.raises(RunnerBuildError, match="Unknown model"):
        build_runner("totally-fake-name", settings)


def test_reasoning_effort_override(settings: Settings) -> None:
    r = build_runner("gpt-5-mini", settings, reasoning_effort="low")
    assert "reasoning_effort='low'" in r.describe()


def test_reasoning_effort_defaults_from_settings(tmp_path) -> None:
    s = Settings(persist_dir=tmp_path / "data", ghmodels_reasoning_effort="high")
    r = build_runner("gpt-5-mini", s)
    assert "reasoning_effort='high'" in r.describe()


def test_max_completion_tokens_from_settings(tmp_path) -> None:
    s = Settings(
        persist_dir=tmp_path / "data",
        ghmodels_max_completion_tokens=12000,
    )
    r = build_runner("gpt-5-mini", s)
    assert "max_completion_tokens=12000" in r.describe()
