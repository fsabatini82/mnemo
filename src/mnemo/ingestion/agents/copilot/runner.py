"""Headless wrapper around the GitHub Copilot CLI.

The exact binary and argument layout varies across Copilot CLI flavors
(`gh copilot`, the standalone `copilot` command, future replacements).
We expose a small surface — `run(prompt)` and `run_json(prompt)` —
and let the binary + args be fully configured via env vars so the
runner adapts without code changes.

Defaults match the standalone agentic CLI: `copilot --no-stream` with
the prompt fed on stdin. Override via:

    MNEMO_COPILOT_BIN      = "copilot"      (path to executable)
    MNEMO_COPILOT_ARGS     = "--no-stream"  (space-separated extra args)
    MNEMO_COPILOT_STDIN    = "1"            ("1" passes prompt via stdin,
                                            "0" passes as final positional arg)
    MNEMO_COPILOT_TIMEOUT  = "180"          (seconds per single call)
"""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
from collections.abc import Sequence


class CopilotRunnerError(RuntimeError):
    """Raised when the Copilot CLI invocation fails or returns no usable output."""


class CopilotRunner:
    def __init__(
        self,
        binary: str | None = None,
        extra_args: Sequence[str] | None = None,
        use_stdin: bool | None = None,
        timeout: int | None = None,
    ) -> None:
        self._binary = binary or os.environ.get("MNEMO_COPILOT_BIN", "copilot")
        if extra_args is None:
            raw_args = os.environ.get("MNEMO_COPILOT_ARGS", "--no-stream")
            extra_args = shlex.split(raw_args) if raw_args else []
        self._extra_args = list(extra_args)
        if use_stdin is None:
            use_stdin = os.environ.get("MNEMO_COPILOT_STDIN", "1") == "1"
        self._use_stdin = use_stdin
        if timeout is None:
            timeout = int(os.environ.get("MNEMO_COPILOT_TIMEOUT", "180"))
        self._timeout = timeout

    # ------------------------------------------------------------------ checks

    def is_available(self) -> bool:
        """True if the configured binary is on PATH or is a valid file."""
        return shutil.which(self._binary) is not None or os.path.isfile(self._binary)

    def describe(self) -> str:
        return (
            f"CopilotRunner(binary={self._binary!r}, "
            f"extra_args={self._extra_args!r}, use_stdin={self._use_stdin}, "
            f"timeout={self._timeout}s)"
        )

    # ------------------------------------------------------------------ exec

    def run(self, prompt: str) -> str:
        """Invoke the CLI with `prompt`, return stdout as text."""
        if not self.is_available():
            raise CopilotRunnerError(
                f"Copilot CLI binary not found: {self._binary!r}. "
                "Set MNEMO_COPILOT_BIN to the correct path, or install the CLI."
            )

        cmd = [self._binary, *self._extra_args]
        try:
            if self._use_stdin:
                result = subprocess.run(
                    cmd,
                    input=prompt,
                    capture_output=True,
                    text=True,
                    timeout=self._timeout,
                    encoding="utf-8",
                )
            else:
                result = subprocess.run(
                    [*cmd, prompt],
                    capture_output=True,
                    text=True,
                    timeout=self._timeout,
                    encoding="utf-8",
                )
        except subprocess.TimeoutExpired as exc:
            raise CopilotRunnerError(
                f"Copilot CLI timed out after {self._timeout}s"
            ) from exc
        except FileNotFoundError as exc:
            raise CopilotRunnerError(f"Failed to spawn {self._binary!r}: {exc}") from exc

        if result.returncode != 0:
            stderr_excerpt = (result.stderr or "").strip()[:500]
            raise CopilotRunnerError(
                f"Copilot CLI exited with {result.returncode}: {stderr_excerpt}"
            )
        return result.stdout

    def run_json(self, prompt: str) -> dict | list | None:
        """Invoke the CLI, parse the first JSON value found in stdout.

        Returns None if the output contains no parseable JSON (treat as
        "model declined to extract"). Useful when the model is asked to
        classify items as indexable vs noise.
        """
        raw = self.run(prompt)
        return extract_json(raw)


# ---------------------------------------------------------------------------
# JSON extraction helpers
# ---------------------------------------------------------------------------


_FENCE_RE = re.compile(r"```(?:json)?\s*\n(.*?)\n```", re.DOTALL | re.IGNORECASE)


def extract_json(text: str) -> dict | list | None:
    """Best-effort JSON extraction from LLM output.

    Order of attempts:
      1. Parse the whole string as JSON.
      2. Look for a ```json ... ``` fenced block and parse its content.
      3. Slice from the first '{' or '[' to its matching close, parse.

    Returns None if nothing parses or if the parsed value is JSON null.
    """
    text = text.strip()
    if not text:
        return None

    # 1) whole string
    try:
        parsed = json.loads(text)
        return parsed if parsed is not None else None
    except json.JSONDecodeError:
        pass

    # 2) fenced block
    if match := _FENCE_RE.search(text):
        try:
            parsed = json.loads(match.group(1).strip())
            return parsed if parsed is not None else None
        except json.JSONDecodeError:
            pass

    # 3) brace/bracket slicing
    for open_char, close_char in (("{", "}"), ("[", "]")):
        start = text.find(open_char)
        end = text.rfind(close_char)
        if 0 <= start < end:
            try:
                parsed = json.loads(text[start : end + 1])
                return parsed if parsed is not None else None
            except json.JSONDecodeError:
                continue

    return None
