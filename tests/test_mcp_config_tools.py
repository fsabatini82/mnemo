"""Tests for the MCP config tools (list_runtime_config + set_runtime_config).

These tools live in `mnemo.mcp_server` and assume `_settings` and
`_system` are already populated (normally done by `main()`). We bypass
`main()` by injecting fakes directly into the module.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

import mnemo.mcp_server as mcp_server
from mnemo.config import Settings
from mnemo.factory import MnemoSystem


# ---------------------------------------------------------------------------
# Bootstrap — populate _settings / _system without running main()
# ---------------------------------------------------------------------------


@pytest.fixture
def configured_server(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Settings, MnemoSystem]:
    """Provide a server-like context with synthetic Settings and a stub system."""
    # Ensure tests run with the .env file isolated to tmp_path.
    monkeypatch.chdir(tmp_path)

    settings = Settings(
        persist_dir=tmp_path / "data",
        project="alpha",
        environment="dev",
    )

    class _StubPipeline:
        def ingest(self, *_a, **_kw): pass
        def query(self, *_a, **_kw): pass

    system = MnemoSystem(
        specs=_StubPipeline(),
        bugs=_StubPipeline(),
        settings=settings,
        project_id="001",
        environment="dev",
    )

    monkeypatch.setattr(mcp_server, "_settings", settings, raising=False)
    monkeypatch.setattr(mcp_server, "_system", system, raising=False)

    return settings, system


# ---------------------------------------------------------------------------
# list_runtime_config
# ---------------------------------------------------------------------------


def _call_tool(name: str, **kwargs: Any) -> Any:
    """Invoke an MCP tool function directly by going through its wrapper."""
    # FastMCP wraps the function; the underlying callable is .fn
    tool = getattr(mcp_server, name)
    if hasattr(tool, "fn"):
        return tool.fn(**kwargs)
    return tool(**kwargs)


def test_list_returns_four_buckets(configured_server) -> None:
    result = _call_tool("list_runtime_config")
    assert set(result.keys()) == {
        "editable_now", "editable_with_restart", "readonly", "hidden",
    }
    assert isinstance(result["editable_now"], dict)
    assert isinstance(result["hidden"], list)


def test_list_includes_known_editable_now_keys(configured_server) -> None:
    result = _call_tool("list_runtime_config")
    assert "MNEMO_GHMODELS_MODEL" in result["editable_now"]
    assert "MNEMO_TOP_K" in result["editable_now"]
    assert "MNEMO_FAMILY_GPT" in result["editable_now"]


def test_list_includes_readonly_keys_with_values(configured_server) -> None:
    settings, _ = configured_server
    result = _call_tool("list_runtime_config")
    assert "MNEMO_PROJECT" in result["readonly"]
    # Value is exposed for readonly (visibility is OK, just not editable).
    assert result["readonly"]["MNEMO_PROJECT"] == "alpha"


def test_list_hidden_lists_only_when_token_present(
    configured_server, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MNEMO_GHMODELS_TOKEN", raising=False)
    monkeypatch.delenv("GH_TOKEN", raising=False)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    result = _call_tool("list_runtime_config")
    assert result["hidden"] == []  # no tokens set

    monkeypatch.setenv("GH_TOKEN", "secret-value")
    result = _call_tool("list_runtime_config")
    assert "GH_TOKEN" in result["hidden"]
    # Value must NEVER appear anywhere in the output.
    flattened = str(result)
    assert "secret-value" not in flattened


# ---------------------------------------------------------------------------
# set_runtime_config — happy paths
# ---------------------------------------------------------------------------


def test_set_editable_now_writes_env_and_updates_process(
    configured_server, tmp_path: Path,
) -> None:
    result = _call_tool(
        "set_runtime_config",
        key="MNEMO_TOP_K", value="42",
    )
    assert "error" not in result
    assert result["category"] == "editable_now"
    assert result["takes_effect"] == "immediately"
    assert result["new_value"] == "42"
    # Live process env updated.
    assert os.environ["MNEMO_TOP_K"] == "42"
    # .env file written with the new key.
    env_file = tmp_path / ".env"
    assert env_file.is_file()
    text = env_file.read_text(encoding="utf-8")
    assert "MNEMO_TOP_K=42" in text


def test_set_editable_with_restart_requires_confirm(configured_server) -> None:
    result = _call_tool(
        "set_runtime_config",
        key="MNEMO_CHUNK_SIZE", value="1024",
    )
    assert "error" in result
    assert "confirm_structural" in result["error"]


def test_set_editable_with_restart_with_confirm_succeeds(
    configured_server, tmp_path: Path,
) -> None:
    result = _call_tool(
        "set_runtime_config",
        key="MNEMO_CHUNK_SIZE", value="1024", confirm_structural=True,
    )
    assert "error" not in result
    assert result["takes_effect"] == "on_restart"
    assert os.environ["MNEMO_CHUNK_SIZE"] == "1024"


# ---------------------------------------------------------------------------
# set_runtime_config — refusal paths
# ---------------------------------------------------------------------------


def test_set_unknown_key_refused(configured_server) -> None:
    result = _call_tool(
        "set_runtime_config",
        key="NOT_REAL", value="x",
    )
    assert "error" in result
    assert "Unknown" in result["error"]


def test_set_readonly_refused(configured_server) -> None:
    result = _call_tool(
        "set_runtime_config",
        key="MNEMO_PROJECT", value="bravo",
    )
    assert "error" in result
    assert "read-only" in result["error"]
    assert result["category"] == "readonly"


def test_set_hidden_refused(configured_server) -> None:
    result = _call_tool(
        "set_runtime_config",
        key="MNEMO_GHMODELS_TOKEN", value="secret-pat",
    )
    assert "error" in result
    assert "token" in result["error"].lower()


def test_set_invalid_value_refused(configured_server) -> None:
    result = _call_tool(
        "set_runtime_config",
        key="MNEMO_TOP_K", value="not-an-int",
    )
    assert "error" in result


def test_set_invalid_reasoning_effort_refused(configured_server) -> None:
    result = _call_tool(
        "set_runtime_config",
        key="MNEMO_GHMODELS_REASONING_EFFORT", value="ultra",
    )
    assert "error" in result


# ---------------------------------------------------------------------------
# Audit comment present in .env after change
# ---------------------------------------------------------------------------


def test_audit_comment_in_env_after_change(
    configured_server, tmp_path: Path,
) -> None:
    _call_tool("set_runtime_config", key="MNEMO_TOP_K", value="7")
    text = (tmp_path / ".env").read_text(encoding="utf-8")
    assert "# changed via MCP at " in text


def test_value_change_reflected_in_subsequent_call(
    configured_server, tmp_path: Path,
) -> None:
    """First set bumps the value; second set sees the previous as old_value."""
    first = _call_tool("set_runtime_config", key="MNEMO_TOP_K", value="11")
    second = _call_tool("set_runtime_config", key="MNEMO_TOP_K", value="12")
    assert first["new_value"] == "11"
    assert second["old_value"] == "11"
    assert second["new_value"] == "12"
