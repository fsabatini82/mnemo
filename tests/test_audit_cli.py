"""Tests for the mnemo-audit CLI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import mnemo.audit_cli as audit_cli
from mnemo.audit import AuditEngine, DriftIssue, DriftReport
from mnemo.core.models import Hit, QueryResult
from mnemo.factory import MnemoSystem


class _RecordingPipeline:
    def __init__(self, hits: list[Hit] | None = None) -> None:
        self._hits = hits or []

    def ingest(self, documents) -> None:  # pragma: no cover
        pass

    def query(self, question: str, *, k: int = 5, where=None) -> QueryResult:
        if question == "*":
            return QueryResult(question=question, hits=self._hits[:k])
        return QueryResult(
            question=question,
            hits=[h for h in self._hits if h.doc_id == question][:k],
        )


@pytest.fixture
def patched_system(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> tuple[_RecordingPipeline, Path]:
    """Patch build_system so the CLI sees a controllable pipeline."""
    code_root = tmp_path / "code"
    code_root.mkdir()
    (code_root / "src").mkdir()
    (code_root / "src" / "tasks.py").write_text("# US-102\n", encoding="utf-8")
    (code_root / "src" / "missing.py").write_text("# nothing\n", encoding="utf-8")

    hits = [
        Hit(
            chunk_id="US-102::chunk-0000",
            doc_id="US-102",
            text="...",
            score=1.0,
            metadata={
                "kind": "story",
                "lifecycle": "implemented",
                "related_files": "src/tasks.py",
                "template_compliance": "full",
            },
        ),
        Hit(
            chunk_id="US-MISSING::chunk-0000",
            doc_id="US-MISSING",
            text="...",
            score=1.0,
            metadata={
                "kind": "story",
                "lifecycle": "implemented",
                "related_files": "src/totally_missing.py",
                "template_compliance": "full",
            },
        ),
    ]
    pipeline = _RecordingPipeline(hits)

    def fake_build(settings: Any, **kwargs: Any) -> MnemoSystem:
        return MnemoSystem(
            specs=pipeline,
            bugs=pipeline,
            settings=settings,
            project_id="999",
            environment=settings.environment,
        )

    monkeypatch.setattr("mnemo.audit_cli.build_system", fake_build)
    monkeypatch.setenv("MNEMO_PERSIST_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("MNEMO_CODE_ROOT", str(code_root))
    return pipeline, code_root


def test_drift_finds_no_issues_when_files_present(
    patched_system: tuple[_RecordingPipeline, Path],
    capsys: pytest.CaptureFixture,
) -> None:
    rc = audit_cli.main(["drift", "--spec", "US-102"])
    out = capsys.readouterr().out
    # No high-severity → exit 0.
    assert rc == 0
    assert "US-102" in out


def test_drift_high_severity_exit_code(
    patched_system: tuple[_RecordingPipeline, Path],
) -> None:
    rc = audit_cli.main(["drift", "--spec", "US-MISSING"])
    # Missing related_files → high → exit 1 (CI-friendly).
    assert rc == 1


def test_drift_unknown_spec_returns_2(
    patched_system: tuple[_RecordingPipeline, Path],
) -> None:
    rc = audit_cli.main(["drift", "--spec", "US-DOES-NOT-EXIST"])
    assert rc == 2


def test_drift_writes_json_output(
    patched_system: tuple[_RecordingPipeline, Path],
    tmp_path: Path,
) -> None:
    out_file = tmp_path / "report.json"
    audit_cli.main(["drift", "-o", str(out_file)])
    assert out_file.is_file()
    payload = json.loads(out_file.read_text(encoding="utf-8"))
    assert payload["total_specs_audited"] == 2
    assert {r["spec_id"] for r in payload["reports"]} == {"US-102", "US-MISSING"}


def test_drift_lifecycle_filter(
    patched_system: tuple[_RecordingPipeline, Path],
    tmp_path: Path,
) -> None:
    out_file = tmp_path / "r.json"
    audit_cli.main(["drift", "--lifecycle", "implemented", "-o", str(out_file)])
    payload = json.loads(out_file.read_text(encoding="utf-8"))
    assert payload["lifecycle_filter"] == "implemented"
    assert payload["total_specs_audited"] == 2


def test_drift_rejects_invalid_lifecycle(
    patched_system: tuple[_RecordingPipeline, Path],
) -> None:
    with pytest.raises(SystemExit):
        audit_cli.main(["drift", "--lifecycle", "totally-made-up"])


def test_drift_requires_subcommand() -> None:
    with pytest.raises(SystemExit):
        audit_cli.main([])


# ---------------------------------------------------------------------------
# --deep flag (F5)
# ---------------------------------------------------------------------------


def test_drift_deep_unavailable_returns_2(
    patched_system: tuple[_RecordingPipeline, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If Copilot CLI is missing, --deep must error out (rc=2)."""
    monkeypatch.setenv("MNEMO_COPILOT_BIN", "/definitely/not/installed/copilot")
    rc = audit_cli.main(["drift", "--spec", "US-102", "--deep"])
    assert rc == 2


def test_drift_deep_merges_issues(
    patched_system: tuple[_RecordingPipeline, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When deep is available and returns drift, cheap + deep merge into one report."""
    from mnemo.audit import DriftIssue

    class _FakeDeepEngine:
        def __init__(self, pipeline, *, code_root, **_kw): pass
        def is_available(self): return True
        def audit_spec(self, spec_id):
            if spec_id == "US-102":
                return [DriftIssue(
                    type="behavior", severity="high",
                    description="spec says X, code does Y",
                    suggested_action="patch",
                )]
            return []

    # Patch the engine + the runner factory so we never hit a real runner.
    monkeypatch.setattr("mnemo.audit_cli.DeepAuditEngine", _FakeDeepEngine)
    monkeypatch.setattr(
        "mnemo.audit_cli.build_runner",
        lambda *a, **kw: type("FR", (), {
            "is_available": lambda self: True,
            "describe": lambda self: "FakeRunner()",
        })(),
    )
    out_file = tmp_path / "r.json"
    rc = audit_cli.main(["drift", "--spec", "US-102", "--deep", "-o", str(out_file)])
    assert rc == 1
    payload = json.loads(out_file.read_text(encoding="utf-8"))
    issue_types = {i["type"] for r in payload["reports"] for i in r["issues"]}
    assert "behavior" in issue_types


def test_drift_agentic_implies_deep(
    patched_system: tuple[_RecordingPipeline, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """--agentic on audit should activate --deep automatically."""
    seen: dict[str, bool] = {"deep_engine_used": False}

    class _FakeDeepEngine:
        def __init__(self, pipeline, *, code_root, **_kw):
            seen["deep_engine_used"] = True
        def is_available(self): return True
        def audit_spec(self, spec_id): return []

    monkeypatch.setattr("mnemo.audit_cli.DeepAuditEngine", _FakeDeepEngine)
    monkeypatch.setattr(
        "mnemo.audit_cli.build_runner",
        lambda *a, **kw: type("FR", (), {
            "is_available": lambda self: True,
            "describe": lambda self: "FakeRunner()",
        })(),
    )
    audit_cli.main(["drift", "--spec", "US-102", "--agentic", "gpt-5-mini"])
    assert seen["deep_engine_used"] is True


def test_drift_invalid_reasoning_effort_argparse_error(
    patched_system: tuple[_RecordingPipeline, Path],
) -> None:
    with pytest.raises(SystemExit):
        audit_cli.main(["drift", "--deep", "--reasoning-effort", "wow"])
