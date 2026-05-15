"""Tests for the GitHub Models HTTP runner."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch
from urllib.error import HTTPError, URLError

import pytest

from mnemo.ingestion.agents.gh_models.runner import GitHubModelsRunner, RunnerError


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def runner() -> GitHubModelsRunner:
    return GitHubModelsRunner(
        model="openai/gpt-5-mini",
        endpoint="https://example.com/inference/chat/completions",
        reasoning_effort="medium",
        max_completion_tokens=2000,
        timeout=30,
        token="fake-test-token",
    )


def _make_urlopen_response(payload: dict, status: int = 200):
    body = json.dumps(payload).encode("utf-8")
    mock_resp = MagicMock()
    mock_resp.read.return_value = body
    mock_resp.status = status
    mock_resp.__enter__ = lambda self: mock_resp
    mock_resp.__exit__ = lambda self, *args: None
    return mock_resp


def _ok_response(content: str, *, finish_reason: str = "stop") -> dict:
    return {
        "choices": [{
            "message": {"role": "assistant", "content": content},
            "finish_reason": finish_reason,
        }],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "completion_tokens_details": {"reasoning_tokens": 10},
        },
    }


# ---------------------------------------------------------------------------
# Availability + describe
# ---------------------------------------------------------------------------


def test_describe_contains_key_fields(runner: GitHubModelsRunner) -> None:
    desc = runner.describe()
    assert "openai/gpt-5-mini" in desc
    assert "reasoning_effort='medium'" in desc
    assert "token_present=True" in desc


def test_is_available_with_token(runner: GitHubModelsRunner) -> None:
    assert runner.is_available() is True


def test_is_available_without_token_falls_back_to_env(monkeypatch: pytest.MonkeyPatch) -> None:
    r = GitHubModelsRunner(
        model="openai/gpt-5-mini",
        endpoint="https://example.com",
    )
    # conftest strips MNEMO_*, but GH_TOKEN/GITHUB_TOKEN may still leak from
    # the developer's env. Force them off.
    monkeypatch.delenv("GH_TOKEN", raising=False)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    assert r.is_available() is False

    monkeypatch.setenv("GH_TOKEN", "env-fallback-token")
    assert r.is_available() is True


# ---------------------------------------------------------------------------
# run() — success path
# ---------------------------------------------------------------------------


def test_run_returns_assistant_content(runner: GitHubModelsRunner) -> None:
    payload = _ok_response("hello world")
    with patch("urllib.request.urlopen", return_value=_make_urlopen_response(payload)):
        content = runner.run("test prompt")
    assert content == "hello world"


def test_run_builds_correct_request_body(runner: GitHubModelsRunner) -> None:
    """Verify the JSON body shape sent to the API."""
    captured: dict = {}

    def capture(req, **_kwargs):
        captured["body"] = req.data.decode("utf-8")
        captured["headers"] = dict(req.headers)
        return _make_urlopen_response(_ok_response("ok"))

    with patch("urllib.request.urlopen", side_effect=capture):
        runner.run("system instructions\n---\nuser question")

    body = json.loads(captured["body"])
    assert body["model"] == "openai/gpt-5-mini"
    assert body["max_completion_tokens"] == 2000
    assert body["reasoning_effort"] == "medium"
    assert body["messages"] == [
        {"role": "system", "content": "system instructions"},
        {"role": "user", "content": "user question"},
    ]


def test_run_no_separator_means_user_only(runner: GitHubModelsRunner) -> None:
    captured: dict = {}

    def capture(req, **_kwargs):
        captured["body"] = req.data.decode("utf-8")
        return _make_urlopen_response(_ok_response("ok"))

    with patch("urllib.request.urlopen", side_effect=capture):
        runner.run("just a user question with no separator")

    body = json.loads(captured["body"])
    assert body["messages"] == [
        {"role": "user", "content": "just a user question with no separator"},
    ]


def test_run_sets_authorization_header(runner: GitHubModelsRunner) -> None:
    captured: dict = {}

    def capture(req, **_kwargs):
        # urllib lowercases header keys
        captured["auth"] = req.headers.get("Authorization")
        return _make_urlopen_response(_ok_response("ok"))

    with patch("urllib.request.urlopen", side_effect=capture):
        runner.run("hi")
    assert captured["auth"] == "Bearer fake-test-token"


# ---------------------------------------------------------------------------
# run() — error paths
# ---------------------------------------------------------------------------


def test_run_missing_token_raises() -> None:
    r = GitHubModelsRunner(
        model="openai/gpt-5-mini",
        endpoint="https://example.com",
    )
    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(RunnerError, match="token not found"):
            r.run("hi")


def test_run_http_400_raises_with_body(runner: GitHubModelsRunner) -> None:
    fake_err = HTTPError(
        "https://example.com", 400, "Bad Request", {},
        MagicMock(read=lambda: b'{"error":"model not found"}'),
    )
    with patch("urllib.request.urlopen", side_effect=fake_err):
        with pytest.raises(RunnerError, match="HTTP 400"):
            runner.run("hi")


def test_run_unreachable_raises(runner: GitHubModelsRunner) -> None:
    with patch("urllib.request.urlopen", side_effect=URLError("connection refused")):
        with pytest.raises(RunnerError, match="unreachable"):
            runner.run("hi")


def test_run_timeout_raises(runner: GitHubModelsRunner) -> None:
    with patch("urllib.request.urlopen", side_effect=TimeoutError()):
        with pytest.raises(RunnerError, match="timed out"):
            runner.run("hi")


def test_run_non_json_response_raises(runner: GitHubModelsRunner) -> None:
    mock = MagicMock()
    mock.read.return_value = b"not json"
    mock.status = 200
    mock.__enter__ = lambda self: mock
    mock.__exit__ = lambda self, *args: None
    with patch("urllib.request.urlopen", return_value=mock):
        with pytest.raises(RunnerError, match="non-JSON"):
            runner.run("hi")


def test_run_no_choices_raises(runner: GitHubModelsRunner) -> None:
    with patch("urllib.request.urlopen",
               return_value=_make_urlopen_response({"choices": []})):
        with pytest.raises(RunnerError, match="no choices"):
            runner.run("hi")


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def test_run_logs_warning_on_truncation(
    runner: GitHubModelsRunner, caplog: pytest.LogCaptureFixture,
) -> None:
    payload = _ok_response("partial response", finish_reason="length")
    with patch("urllib.request.urlopen", return_value=_make_urlopen_response(payload)):
        import logging
        with caplog.at_level(logging.WARNING):
            runner.run("hi")
    assert any("truncated" in rec.message.lower() for rec in caplog.records)


def test_run_logs_warning_on_near_empty_response(
    runner: GitHubModelsRunner, caplog: pytest.LogCaptureFixture,
) -> None:
    payload = _ok_response("ok")  # short content
    with patch("urllib.request.urlopen", return_value=_make_urlopen_response(payload)):
        import logging
        with caplog.at_level(logging.WARNING):
            runner.run("hi")
    assert any("empty" in rec.message.lower() for rec in caplog.records)


# ---------------------------------------------------------------------------
# run_json — JSON extraction
# ---------------------------------------------------------------------------


def test_run_json_extracts_object(runner: GitHubModelsRunner) -> None:
    payload = _ok_response('{"key": "value"}')
    with patch("urllib.request.urlopen", return_value=_make_urlopen_response(payload)):
        assert runner.run_json("hi") == {"key": "value"}


def test_run_json_returns_none_on_garbage(runner: GitHubModelsRunner) -> None:
    payload = _ok_response("plain text no json")
    with patch("urllib.request.urlopen", return_value=_make_urlopen_response(payload)):
        assert runner.run_json("hi") is None


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------


def test_invalid_reasoning_effort_coerced_to_medium() -> None:
    r = GitHubModelsRunner(
        model="m", endpoint="https://x", reasoning_effort="ultra-mega-high",
        token="t",
    )
    # describe shows what we used
    assert "reasoning_effort='medium'" in r.describe()
