"""GitHub Models runner — direct HTTP-based curl-mode invocation.

Shape-compatible with `CopilotRunner` (same public surface: `is_available()`,
`run()`, `run_json()`, `describe()`) so agents stay runtime-agnostic.

Unlike the subprocess Copilot CLI runtime, this lets the caller select
the exact model (e.g. `openai/gpt-5`, `anthropic/claude-opus-4-6`),
tune `reasoning_effort`, and bound `max_completion_tokens` — exactly
the controls needed to keep token costs predictable.

Uses `urllib.request` from the stdlib (no new dependency). Always sends
`reasoning_effort` in the body; non-reasoning models silently ignore it.
"""

from __future__ import annotations

import json
import logging
import os
import shlex
import time
import urllib.error
import urllib.request
from collections.abc import Sequence
from typing import Any

from mnemo.ingestion.agents.copilot.runner import (
    CopilotRunnerError,
    extract_json,
)
from mnemo.models_catalog import resolve_token_from_env

logger = logging.getLogger(__name__)


# We reuse `CopilotRunnerError` as the public error type so agents and
# the audit engine can keep their existing `except CopilotRunnerError`
# blocks without caring about the runner backend. Conceptually it's a
# "runner failure" — the name is historical.
RunnerError = CopilotRunnerError


# Reasoning-effort values accepted by GPT-5 family. Non-reasoning models
# ignore the field silently when sent.
_VALID_EFFORTS = ("minimal", "low", "medium", "high")


class GitHubModelsRunner:
    """Direct HTTP runner against the GitHub Models inference endpoint."""

    def __init__(
        self,
        *,
        model: str,
        endpoint: str,
        reasoning_effort: str = "medium",
        max_completion_tokens: int = 6000,
        timeout: int = 180,
        token: str | None = None,
    ) -> None:
        self._model = model
        self._endpoint = endpoint
        self._reasoning_effort = (
            reasoning_effort if reasoning_effort in _VALID_EFFORTS else "medium"
        )
        self._max_completion_tokens = int(max_completion_tokens)
        self._timeout = int(timeout)
        # Token can be passed explicitly (for tests) or pulled from env.
        self._token: str | None = token

    # ------------------------------------------------------------------ checks

    def _resolved_token(self) -> str | None:
        return self._token or resolve_token_from_env()

    def is_available(self) -> bool:
        """True when both the endpoint URL and an auth token are present."""
        return bool(self._endpoint) and self._resolved_token() is not None

    def describe(self) -> str:
        token_present = self._resolved_token() is not None
        return (
            f"GitHubModelsRunner(model={self._model!r}, "
            f"endpoint={self._endpoint!r}, "
            f"reasoning_effort={self._reasoning_effort!r}, "
            f"max_completion_tokens={self._max_completion_tokens}, "
            f"timeout={self._timeout}s, "
            f"token_present={token_present})"
        )

    # ------------------------------------------------------------------ exec

    def run(self, prompt: str) -> str:
        token = self._resolved_token()
        if not token:
            raise RunnerError(
                "GitHub Models token not found. Set MNEMO_GHMODELS_TOKEN, "
                "GH_TOKEN, or GITHUB_TOKEN in your shell environment "
                "(scope `models:read` required)."
            )

        body = self._build_body(prompt)
        encoded = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            self._endpoint,
            data=encoded,
            method="POST",
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "mnemo-gh-models/1.0",
            },
        )

        started = time.monotonic()
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                raw = resp.read()
                status = resp.status
        except urllib.error.HTTPError as exc:
            body_excerpt = _safe_read_body(exc)
            raise RunnerError(
                f"GitHub Models API returned HTTP {exc.code}: {body_excerpt}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RunnerError(
                f"GitHub Models endpoint unreachable: {exc.reason}"
            ) from exc
        except TimeoutError as exc:
            raise RunnerError(
                f"GitHub Models request timed out after {self._timeout}s"
            ) from exc

        elapsed = time.monotonic() - started

        try:
            payload: dict[str, Any] = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RunnerError(
                f"GitHub Models returned non-JSON response: {exc}"
            ) from exc

        return self._extract_and_log(payload, status=status, elapsed=elapsed)

    def run_json(self, prompt: str) -> Any:
        """Convenience: run + parse the model's content as JSON."""
        return extract_json(self.run(prompt))

    # ------------------------------------------------------------------ helpers

    def _build_body(self, prompt: str) -> dict[str, Any]:
        """Build the OpenAI-compatible request body.

        We split the system prompt from the user prompt by looking for
        the `\\n---\\n` separator that all Mnemo agent prompts use. If
        the separator is absent, the whole prompt goes as user message.
        """
        system, user = self._split_prompt(prompt)
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})

        # `reasoning_effort` is sent unconditionally; non-reasoning models
        # ignore it. Same for `max_completion_tokens` which is the
        # canonical field name in the GH Models / OpenAI-compat schema.
        return {
            "model": self._model,
            "messages": messages,
            "max_completion_tokens": self._max_completion_tokens,
            "reasoning_effort": self._reasoning_effort,
        }

    @staticmethod
    def _split_prompt(prompt: str) -> tuple[str, str]:
        marker = "\n---\n"
        if marker in prompt:
            head, _, tail = prompt.partition(marker)
            return head.strip(), tail.strip()
        return "", prompt

    def _extract_and_log(
        self,
        payload: dict[str, Any],
        *,
        status: int,
        elapsed: float,
    ) -> str:
        choices = payload.get("choices") or []
        if not choices or not isinstance(choices, list):
            raise RunnerError("GitHub Models response had no choices.")
        first = choices[0] if isinstance(choices[0], dict) else {}
        message = first.get("message") if isinstance(first, dict) else None
        content = (message or {}).get("content") if isinstance(message, dict) else None
        finish_reason = first.get("finish_reason", "unknown") if isinstance(first, dict) else "unknown"

        usage = payload.get("usage") or {}
        completion_tokens = usage.get("completion_tokens", 0)
        prompt_tokens = usage.get("prompt_tokens", 0)
        reasoning_tokens = (
            (usage.get("completion_tokens_details") or {}).get("reasoning_tokens", 0)
        )

        logger.info(
            "GH Models call: model=%s elapsed=%.2fs status=%s finish=%s "
            "tokens(prompt=%s, completion=%s, reasoning=%s)",
            self._model, elapsed, status, finish_reason,
            prompt_tokens, completion_tokens, reasoning_tokens,
        )

        if finish_reason == "length":
            logger.warning(
                "Response truncated by token cap. "
                "Increase max_completion_tokens (currently %s) or lower reasoning_effort.",
                self._max_completion_tokens,
            )

        if not isinstance(content, str):
            content = ""
        if len(content.strip()) < 100:
            logger.warning(
                "Near-empty response (%d chars). Likely cause: reasoning tokens "
                "(%s) consumed the budget. Bump max_completion_tokens or lower "
                "reasoning_effort.",
                len(content), reasoning_tokens,
            )
        return content


def _safe_read_body(exc: urllib.error.HTTPError, limit: int = 500) -> str:
    """Read the response body from a failed request, capped at `limit` chars."""
    try:
        raw = exc.read()
    except Exception:  # noqa: BLE001
        return "<no body>"
    text = raw.decode("utf-8", errors="replace")[:limit]
    return text.strip() or "<empty body>"
