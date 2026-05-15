"""Tests for the project registry."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mnemo.registry import (
    ENVIRONMENTS,
    ProjectRegistry,
    RegistryError,
    collection_name,
    default_slug,
    open_registry,
    validate_environment,
    validate_slug,
)


# ---------------------------------------------------------------------------
# Validation primitives
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "good",
    ["a", "alpha", "alpha-beta", "p1", "demo-project", "a1b2c3", "x" * 32],
)
def test_validate_slug_accepts_good(good: str) -> None:
    assert validate_slug(good) == good


@pytest.mark.parametrize(
    "bad",
    ["", "Alpha", "1alpha", "alpha_beta", "alpha.beta", "alpha/beta",
     "x" * 33, " alpha", "alpha ", "-alpha"],
)
def test_validate_slug_rejects_bad(bad: str) -> None:
    with pytest.raises(RegistryError):
        validate_slug(bad)


@pytest.mark.parametrize("env", list(ENVIRONMENTS))
def test_validate_environment_accepts_enum(env: str) -> None:
    assert validate_environment(env) == env


@pytest.mark.parametrize("bad", ["development", "DEV", "uat", "test", ""])
def test_validate_environment_rejects_other(bad: str) -> None:
    with pytest.raises(RegistryError):
        validate_environment(bad)


def test_collection_name_format() -> None:
    assert collection_name("001", "dev", "specs") == "001_dev_specs"
    assert collection_name("042", "prd", "bug_memory") == "042_prd_bug_memory"


@pytest.mark.parametrize(
    "args",
    [
        ("1", "dev", "specs"),       # 1-digit id
        ("9999", "dev", "specs"),    # 4-digit id
        ("001", "DEV", "specs"),     # uppercase env
        ("001", "test", "specs"),    # unknown env
        ("001", "dev", "Spec"),      # uppercase axis
        ("001", "dev", "spec-1"),    # dash in axis
    ],
)
def test_collection_name_rejects_bad(args: tuple[str, str, str]) -> None:
    with pytest.raises(RegistryError):
        collection_name(*args)


def test_default_slug() -> None:
    assert default_slug() == "demo-project"


# ---------------------------------------------------------------------------
# Registry lifecycle
# ---------------------------------------------------------------------------


def test_open_registry_empty_when_no_file(tmp_path: Path) -> None:
    reg = open_registry(tmp_path)
    assert len(reg) == 0
    assert reg.projects() == []


def test_ensure_registers_first_project(tmp_path: Path) -> None:
    reg = open_registry(tmp_path)
    rec = reg.ensure("alpha")
    assert rec.id == "001"
    assert rec.slug == "alpha"
    assert rec.environments == []  # populated by add_environment later
    assert "alpha" in reg


def test_ensure_is_idempotent(tmp_path: Path) -> None:
    reg = open_registry(tmp_path)
    a1 = reg.ensure("alpha")
    a2 = reg.ensure("alpha")
    assert a1.id == a2.id


def test_ensure_assigns_sequential_ids(tmp_path: Path) -> None:
    reg = open_registry(tmp_path)
    assert reg.ensure("alpha").id == "001"
    assert reg.ensure("beta").id == "002"
    assert reg.ensure("gamma").id == "003"


def test_save_and_reload_preserves_state(tmp_path: Path) -> None:
    reg = open_registry(tmp_path)
    reg.ensure("alpha")
    reg.add_environment("alpha", "dev")
    reg.add_environment("alpha", "prd")
    reg.ensure("beta")
    reg.save()

    reg2 = open_registry(tmp_path)
    assert reg2.get("alpha").id == "001"
    assert reg2.get("alpha").environments == ["dev", "prd"]
    assert reg2.get("beta").id == "002"


def test_save_writes_atomically(tmp_path: Path) -> None:
    """The temp-file rename should leave no .tmp behind."""
    reg = open_registry(tmp_path)
    reg.ensure("alpha")
    reg.save()
    files = list(tmp_path.iterdir())
    assert (tmp_path / "projects.json") in files
    assert not any(f.name.endswith(".tmp") for f in files)


def test_load_rejects_corrupted_json(tmp_path: Path) -> None:
    (tmp_path / "projects.json").write_text("this is not json", encoding="utf-8")
    with pytest.raises(RegistryError, match="corrupted"):
        open_registry(tmp_path)


def test_load_rejects_wrong_version(tmp_path: Path) -> None:
    (tmp_path / "projects.json").write_text(
        json.dumps({"version": 99, "projects": {}}), encoding="utf-8",
    )
    with pytest.raises(RegistryError, match="version"):
        open_registry(tmp_path)


def test_add_environment_dedup(tmp_path: Path) -> None:
    reg = open_registry(tmp_path)
    reg.ensure("alpha")
    reg.add_environment("alpha", "dev")
    reg.add_environment("alpha", "dev")
    reg.add_environment("alpha", "prd")
    assert reg.get("alpha").environments == ["dev", "prd"]


def test_add_environment_on_unknown_project_raises(tmp_path: Path) -> None:
    reg = open_registry(tmp_path)
    with pytest.raises(RegistryError):
        reg.add_environment("alpha", "dev")


def test_rename_preserves_id(tmp_path: Path) -> None:
    reg = open_registry(tmp_path)
    rec = reg.ensure("alpha")
    original_id = rec.id
    reg.rename("alpha", "alpha-platform")
    assert reg.get("alpha") is None
    renamed = reg.get("alpha-platform")
    assert renamed is not None
    assert renamed.id == original_id  # ← the whole point of numeric IDs


def test_rename_collision_raises(tmp_path: Path) -> None:
    reg = open_registry(tmp_path)
    reg.ensure("alpha")
    reg.ensure("beta")
    with pytest.raises(RegistryError, match="already exists"):
        reg.rename("alpha", "beta")


def test_rename_unknown_source_raises(tmp_path: Path) -> None:
    reg = open_registry(tmp_path)
    with pytest.raises(RegistryError, match="not registered"):
        reg.rename("alpha", "beta")


def test_drop_removes_entry(tmp_path: Path) -> None:
    reg = open_registry(tmp_path)
    reg.ensure("alpha")
    reg.drop("alpha")
    assert reg.get("alpha") is None
    assert len(reg) == 0


def test_drop_unknown_raises(tmp_path: Path) -> None:
    reg = open_registry(tmp_path)
    with pytest.raises(RegistryError, match="not registered"):
        reg.drop("alpha")


def test_id_reuse_after_drop(tmp_path: Path) -> None:
    """Dropping frees the slot; next ensure can grab it."""
    reg = open_registry(tmp_path)
    reg.ensure("alpha")     # 001
    reg.ensure("beta")      # 002
    reg.drop("alpha")
    reg.ensure("gamma")     # should take 001 (lowest free)
    assert reg.get("gamma").id == "001"


def test_id_exhaustion(tmp_path: Path) -> None:
    """The 1000-project ceiling triggers a clear error."""
    reg = ProjectRegistry(tmp_path)
    # Pre-populate the registry with 999 entries (we skip the load path).
    for n in range(1, 1000):
        reg._projects[f"slug-{n}"] = type(reg).__dict__  # any non-None placeholder
    # The "any non-None" hack above doesn't carry ProjectRecord shape, so
    # build proper records:
    reg._projects.clear()
    from mnemo.registry import ProjectRecord
    for n in range(1, 1000):
        slug = f"a{n}"
        reg._projects[slug] = ProjectRecord(
            id=f"{n:03d}", slug=slug, created="2026-01-01", environments=[],
        )
    with pytest.raises(RegistryError, match="exhausted"):
        reg.ensure("one-too-many")
