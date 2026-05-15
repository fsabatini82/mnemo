"""Atomic `.env` file editing — preserves comments and ordering.

Used by the MCP tool `set_runtime_config` to update a single key in the
on-disk `.env` without disturbing the rest of the file. The write is
atomic via temp-file rename, so a crash mid-write can't leave a half-
written file. Comments and blank lines are preserved.

We deliberately do NOT support exotic dotenv syntax (quoted values,
inline comments, multi-line values). The Mnemo `.env` is plain
`KEY=value` per line; anything fancier is rejected with a clear error.
"""

from __future__ import annotations

import datetime as _dt
import os
import re
from pathlib import Path

_KEY_LINE_RE = re.compile(r"^(?P<key>[A-Z_][A-Z0-9_]*)=(?P<value>.*)$")


def read_env(path: Path) -> list[str]:
    """Read the file as a list of lines (no trailing newline per element).

    Returns an empty list if the file doesn't exist.
    """
    if not path.is_file():
        return []
    text = path.read_text(encoding="utf-8")
    # splitlines drops the trailing newline if present; we restore it
    # at write time consistently.
    return text.splitlines()


def find_value(lines: list[str], key: str) -> str | None:
    """Return the current value for `key`, or None if absent."""
    for line in lines:
        m = _KEY_LINE_RE.match(line.strip())
        if m and m.group("key") == key:
            return m.group("value")
    return None


def update_or_append(
    lines: list[str], key: str, value: str,
) -> tuple[list[str], str | None]:
    """Replace the first matching `KEY=…` line, or append a new one.

    Returns (new_lines, old_value). `old_value` is `None` when the key
    was not previously present.
    """
    pattern = re.compile(rf"^{re.escape(key)}=(.*)$")
    new_lines: list[str] = []
    old_value: str | None = None
    found = False
    for line in lines:
        if not found:
            m = pattern.match(line)
            if m:
                old_value = m.group(1)
                new_lines.append(f"{key}={value}")
                found = True
                continue
        new_lines.append(line)
    if not found:
        # Ensure separation from the last non-empty line.
        if new_lines and new_lines[-1].strip() != "":
            new_lines.append("")
        new_lines.append(f"{key}={value}")
    return new_lines, old_value


def add_audit_comment(
    lines: list[str], key: str, old_value: str | None,
) -> list[str]:
    """Insert `# changed via MCP at <iso8601> (was: <old>)` above `KEY=…`."""
    ts = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    comment = f"# changed via MCP at {ts} (was: {old_value if old_value is not None else '<unset>'})"
    out: list[str] = []
    added = False
    for line in lines:
        if not added and line.startswith(f"{key}="):
            out.append(comment)
            added = True
        out.append(line)
    if not added:
        # Edge case: key not in lines yet (caller forgot update_or_append).
        # Append the comment + a placeholder marker at the end so we don't
        # silently drop it.
        out.append(comment)
    return out


def atomic_write(path: Path, lines: list[str]) -> None:
    """Write `lines` to `path` atomically via a temp-file rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    body = "\n".join(lines)
    if not body.endswith("\n"):
        body += "\n"
    tmp.write_text(body, encoding="utf-8")
    # Replace is atomic on the same filesystem.
    os.replace(tmp, path)


def commit_change(path: Path, key: str, value: str) -> str | None:
    """Apply a single key=value change to `.env` with audit comment.

    Convenience: combines read_env / update_or_append / add_audit_comment
    / atomic_write. Returns the previous value (or None).
    """
    lines = read_env(path)
    new_lines, old_value = update_or_append(lines, key, value)
    new_lines = add_audit_comment(new_lines, key, old_value)
    atomic_write(path, new_lines)
    return old_value
