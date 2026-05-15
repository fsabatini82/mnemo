"""Runtime configuration classification + validation.

The MCP tools `list_runtime_config` / `set_runtime_config` use this
module to decide:

- which env vars can be changed at runtime from chat (`editable_now`)
- which can be changed but require a server restart (`editable_with_restart`)
- which must NEVER be changed from chat (`readonly`: scope / security)
- which must never have their **values** exposed (`hidden`: tokens)

The category map is the single source of truth — there is no implicit
"editable by default". Unknown keys are rejected.
"""

from __future__ import annotations

from enum import Enum

from mnemo.config import Settings


class ConfigCategory(str, Enum):
    EDITABLE_NOW = "editable_now"
    EDITABLE_WITH_RESTART = "editable_with_restart"
    READONLY = "readonly"
    HIDDEN = "hidden"


# Canonical map: env var name → category. New env vars added by future
# features must be classified here explicitly — there is no default.
CATEGORY: dict[str, ConfigCategory] = {
    # --- editable now (runtime LLM / retrieval choices) ---------------
    "MNEMO_GHMODELS_MODEL":                ConfigCategory.EDITABLE_NOW,
    "MNEMO_GHMODELS_REASONING_EFFORT":     ConfigCategory.EDITABLE_NOW,
    "MNEMO_GHMODELS_MAX_COMPLETION_TOKENS": ConfigCategory.EDITABLE_NOW,
    "MNEMO_GHMODELS_TIMEOUT_SECONDS":      ConfigCategory.EDITABLE_NOW,
    "MNEMO_FAMILY_GPT":                    ConfigCategory.EDITABLE_NOW,
    "MNEMO_FAMILY_CLAUDE":                 ConfigCategory.EDITABLE_NOW,
    "MNEMO_FAMILY_SONNET":                 ConfigCategory.EDITABLE_NOW,
    "MNEMO_FAMILY_OPUS":                   ConfigCategory.EDITABLE_NOW,
    "MNEMO_TOP_K":                         ConfigCategory.EDITABLE_NOW,
    "MNEMO_COPILOT_TIMEOUT":               ConfigCategory.EDITABLE_NOW,
    "MNEMO_AUDIT_FILE_BUDGET":             ConfigCategory.EDITABLE_NOW,

    # --- editable with restart (structural rebuild required) ----------
    "MNEMO_CHUNK_SIZE":   ConfigCategory.EDITABLE_WITH_RESTART,
    "MNEMO_CHUNK_OVERLAP": ConfigCategory.EDITABLE_WITH_RESTART,
    "MNEMO_STORE":        ConfigCategory.EDITABLE_WITH_RESTART,
    "MNEMO_PIPELINE":     ConfigCategory.EDITABLE_WITH_RESTART,
    "MNEMO_EMBED_MODEL":  ConfigCategory.EDITABLE_WITH_RESTART,

    # --- readonly via chat (scope-changing or security-sensitive) -----
    # Scope: changing these mid-session would silently swap which RAG
    # the agent is reasoning over.
    "MNEMO_PROJECT":       ConfigCategory.READONLY,
    "MNEMO_ENVIRONMENT":   ConfigCategory.READONLY,
    # Storage / path: a malicious prompt redirecting a path could exfil
    # or destroy data.
    "MNEMO_PERSIST_DIR":        ConfigCategory.READONLY,
    "MNEMO_CODE_ROOT":          ConfigCategory.READONLY,
    "MNEMO_SPECS_COLLECTION":   ConfigCategory.READONLY,
    "MNEMO_BUGS_COLLECTION":    ConfigCategory.READONLY,
    "MNEMO_SPECS_SOURCE_DIR":   ConfigCategory.READONLY,
    "MNEMO_BUGS_SOURCE_DIR":    ConfigCategory.READONLY,
    # Endpoints + binaries: a malicious change could redirect inference
    # traffic or swap the Copilot CLI executable.
    "MNEMO_GHMODELS_ENDPOINT":         ConfigCategory.READONLY,
    "MNEMO_GHMODELS_CATALOG_ENDPOINT": ConfigCategory.READONLY,
    "MNEMO_COPILOT_BIN":   ConfigCategory.READONLY,
    "MNEMO_COPILOT_ARGS":  ConfigCategory.READONLY,
    "MNEMO_COPILOT_STDIN": ConfigCategory.READONLY,

    # --- hidden (token-shaped, never exposed by value) ----------------
    "MNEMO_GHMODELS_TOKEN": ConfigCategory.HIDDEN,
    "GH_TOKEN":             ConfigCategory.HIDDEN,
    "GITHUB_TOKEN":         ConfigCategory.HIDDEN,
}


class UnknownConfigKey(KeyError):
    """Raised when the caller references a key not in CATEGORY."""


class InvalidConfigValue(ValueError):
    """Raised when validate_value rejects a candidate value."""


def classify(key: str) -> ConfigCategory:
    """Return the category for `key`, raising if it's not in the map."""
    if key not in CATEGORY:
        raise UnknownConfigKey(
            f"Unknown config key {key!r}. Run list_runtime_config() to "
            "see the known keys."
        )
    return CATEGORY[key]


def is_known(key: str) -> bool:
    return key in CATEGORY


def all_keys_by_category() -> dict[ConfigCategory, list[str]]:
    """Group all known keys by their category, sorted alphabetically."""
    out: dict[ConfigCategory, list[str]] = {c: [] for c in ConfigCategory}
    for key, cat in CATEGORY.items():
        out[cat].append(key)
    for keys in out.values():
        keys.sort()
    return out


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_value(key: str, value: str) -> None:
    """Validate `value` against the field's type / range.

    For keys backed by `Settings` fields, we construct a one-shot
    `Settings(**{field: value})` and let pydantic-settings validate. For
    keys that live only as env vars (e.g. `MNEMO_COPILOT_TIMEOUT`,
    `MNEMO_AUDIT_FILE_BUDGET`), we apply hand-rolled checks.

    Raises `InvalidConfigValue` on failure; returns None on success.
    """
    if key not in CATEGORY:
        raise UnknownConfigKey(f"Unknown key {key!r}.")
    cat = CATEGORY[key]
    if cat == ConfigCategory.HIDDEN:
        raise InvalidConfigValue(
            f"Token keys ({key}) cannot be set via MCP. Export them in "
            "your shell environment instead."
        )
    if cat == ConfigCategory.READONLY:
        raise InvalidConfigValue(
            f"Key {key!r} is read-only via MCP."
        )

    if not isinstance(value, str):
        raise InvalidConfigValue("Value must be a string.")

    field_name = _env_var_to_field_name(key)
    if field_name and field_name in Settings.model_fields:
        try:
            Settings(**{field_name: value})
        except Exception as exc:  # noqa: BLE001 — wrap any pydantic error
            raise InvalidConfigValue(
                f"Invalid value for {key}: {exc}"
            ) from exc
        return

    # Non-Settings env-only keys.
    if key == "MNEMO_COPILOT_TIMEOUT":
        _validate_positive_int(key, value)
    elif key == "MNEMO_AUDIT_FILE_BUDGET":
        _validate_int_at_least(key, value, minimum=500)
    else:
        # If we reach here a key is in CATEGORY but has no Settings field
        # AND no custom validator — that's a configuration bug in this
        # module. Fail loudly so we notice during development.
        raise InvalidConfigValue(
            f"Key {key!r} is classified but has no validator wired up."
        )


def _env_var_to_field_name(key: str) -> str | None:
    """Map `MNEMO_TOP_K` → `top_k`. Returns None if no MNEMO_ prefix."""
    if not key.startswith("MNEMO_"):
        return None
    return key[len("MNEMO_"):].lower()


def _validate_positive_int(key: str, value: str) -> None:
    try:
        n = int(value)
    except ValueError as exc:
        raise InvalidConfigValue(f"{key} must be an integer.") from exc
    if n <= 0:
        raise InvalidConfigValue(f"{key} must be > 0 (got {n}).")


def _validate_int_at_least(key: str, value: str, *, minimum: int) -> None:
    try:
        n = int(value)
    except ValueError as exc:
        raise InvalidConfigValue(f"{key} must be an integer.") from exc
    if n < minimum:
        raise InvalidConfigValue(
            f"{key} must be at least {minimum} (got {n})."
        )
