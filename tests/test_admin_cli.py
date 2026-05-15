"""Tests for the mnemo-admin CLI."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import mnemo.admin_cli as admin_cli
from mnemo.registry import open_registry


@pytest.fixture
def isolated_data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    persist = tmp_path / "data"
    persist.mkdir()
    monkeypatch.setenv("MNEMO_PERSIST_DIR", str(persist))
    return persist


def _seed_registry(persist_dir: Path) -> None:
    """Pre-populate the registry without touching any vector store."""
    reg = open_registry(persist_dir)
    reg.ensure("alpha")
    reg.add_environment("alpha", "dev")
    reg.add_environment("alpha", "prd")
    reg.ensure("beta")
    reg.add_environment("beta", "dev")
    reg.save()


def test_list_empty(isolated_data: Path, capsys: pytest.CaptureFixture) -> None:
    rc = admin_cli.main(["list-projects"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "No projects registered" in out


def test_list_shows_records(isolated_data: Path, capsys: pytest.CaptureFixture) -> None:
    _seed_registry(isolated_data)
    rc = admin_cli.main(["list-projects"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "001" in out and "alpha" in out
    assert "002" in out and "beta" in out
    assert "dev,prd" in out


def test_rename_preserves_id(isolated_data: Path, capsys: pytest.CaptureFixture) -> None:
    _seed_registry(isolated_data)
    rc = admin_cli.main(["rename-project", "alpha", "alpha-platform"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "id=001" in out

    reg = open_registry(isolated_data)
    assert reg.get("alpha") is None
    renamed = reg.get("alpha-platform")
    assert renamed is not None and renamed.id == "001"


def test_rename_invalid_slug_argparse_error(isolated_data: Path) -> None:
    _seed_registry(isolated_data)
    with pytest.raises(SystemExit):
        admin_cli.main(["rename-project", "alpha", "Bad-Name"])


def test_show_collection_names(isolated_data: Path, capsys: pytest.CaptureFixture) -> None:
    _seed_registry(isolated_data)
    rc = admin_cli.main(["show-collection-names", "alpha", "--env", "prd"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "001_prd_specs" in out
    assert "001_prd_bug_memory" in out


def test_show_unknown_project_returns_error(
    isolated_data: Path, capsys: pytest.CaptureFixture,
) -> None:
    rc = admin_cli.main(["show-collection-names", "ghost"])
    assert rc == 2


def test_drop_specific_env_keeps_other(
    isolated_data: Path, capsys: pytest.CaptureFixture,
) -> None:
    _seed_registry(isolated_data)
    rc = admin_cli.main(["drop-project", "alpha", "--env", "dev"])
    assert rc == 0
    reg = open_registry(isolated_data)
    record = reg.get("alpha")
    # alpha still exists with `prd` environment only.
    assert record is not None
    assert record.environments == ["prd"]


def test_drop_all_envs_removes_project(
    isolated_data: Path, capsys: pytest.CaptureFixture,
) -> None:
    _seed_registry(isolated_data)
    rc = admin_cli.main(["drop-project", "alpha"])
    assert rc == 0
    reg = open_registry(isolated_data)
    assert reg.get("alpha") is None
    # beta still registered.
    assert reg.get("beta") is not None


def test_drop_unknown_project_returns_error(isolated_data: Path) -> None:
    rc = admin_cli.main(["drop-project", "ghost"])
    assert rc == 2


def test_subcommand_required(isolated_data: Path) -> None:
    with pytest.raises(SystemExit):
        admin_cli.main([])
