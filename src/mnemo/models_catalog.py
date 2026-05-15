"""Model name resolution + GitHub Models catalog access.

User-facing model identifiers go through three stages before they hit
the GitHub Models gateway:

1. **Family aliases** (`gpt`, `claude`, `sonnet`, `opus`) — configurable
   via Settings. Lets you pin "current target version" centrally and
   bump everywhere with one `.env` edit. Resolution: `--agentic gpt`
   looks up `settings.family_gpt`, which itself is a short-name or full
   model id, and the chain continues.

2. **Short-name map** — curated convenience mapping. `gpt-5-mini` →
   `openai/gpt-5-mini`. Stable, maintained here.

3. **Full-name pass-through** — any string containing `/` is sent
   verbatim. Fails at the API with HTTP 400 if the model isn't in the
   live catalog.

The special value `copilot` is reserved: it bypasses GitHub Models
entirely and routes to the subprocess `CopilotRunner` (auto-model
mode, no explicit control over which LLM Copilot picks).
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from typing import Any

logger = logging.getLogger(__name__)

# Reserved sentinel: route to the subprocess Copilot CLI instead of GH Models.
COPILOT_SUBPROCESS_MARKER = "copilot"

# Curated short-names → full GitHub Models IDs.
# Update this when GitHub publishes new model versions you want to support.
# For models not in the map, users can always pass the full `provider/model`
# string directly.
MODEL_SHORTNAMES: dict[str, str] = {
    # --- OpenAI GPT-5 family (reasoning models) ---
    "gpt-5":            "openai/gpt-5",
    "gpt-5-mini":       "openai/gpt-5-mini",
    "gpt-5-nano":       "openai/gpt-5-nano",
    # --- OpenAI GPT-4.x (faster, non-reasoning) ---
    "gpt-4.1":          "openai/gpt-4.1",
    "gpt-4.1-mini":     "openai/gpt-4.1-mini",
    # --- Anthropic Claude — pinned to 4-6 by default (4-7 is pricier) ---
    "claude-sonnet-4-6": "anthropic/claude-sonnet-4-6",
    "claude-opus-4-6":   "anthropic/claude-opus-4-6",
}

# Family aliases — each maps to a Settings field name. The resolver reads
# that field at runtime to get the user's current "target" for the family.
FAMILY_TO_SETTING: dict[str, str] = {
    "gpt":    "family_gpt",
    "claude": "family_claude",
    "sonnet": "family_sonnet",
    "opus":   "family_opus",
}
FAMILY_ALIASES: tuple[str, ...] = tuple(FAMILY_TO_SETTING.keys())


class ModelResolutionError(ValueError):
    """Raised when a model name can't be resolved to a full GH Models id."""


def is_subprocess_marker(value: str) -> bool:
    """True iff `value` selects the subprocess Copilot CLI runtime."""
    return value == COPILOT_SUBPROCESS_MARKER


def is_family_alias(value: str) -> bool:
    return value in FAMILY_TO_SETTING


def list_shortnames() -> list[str]:
    """Return the curated short-name keys, sorted."""
    return sorted(MODEL_SHORTNAMES.keys())


def resolve_model(name: str, settings: Any) -> str:
    """Resolve a user-supplied model name to a full GH Models id.

    Algorithm:

    1. If `name` is a family alias (`gpt`/`claude`/`sonnet`/`opus`),
       replace `name` with the corresponding `settings.family_*` value
       and continue.
    2. If the result is in `MODEL_SHORTNAMES`, return its full-name.
    3. If the result contains `/`, treat it as a full-name and
       pass-through (lets users pass arbitrary catalog ids).
    4. Otherwise raise `ModelResolutionError` with the supported
       short-names listed.

    The `copilot` sentinel is NOT handled here — callers must filter it
    out before invoking the resolver (it's not a GH Models id).
    """
    if not isinstance(name, str) or not name.strip():
        raise ModelResolutionError("Empty or non-string model name.")
    name = name.strip()

    if name == COPILOT_SUBPROCESS_MARKER:
        raise ModelResolutionError(
            "The 'copilot' sentinel is for the subprocess runtime, not GH Models. "
            "Callers must dispatch on this value before calling resolve_model()."
        )

    # Stage 1 — family alias → setting lookup
    if name in FAMILY_TO_SETTING:
        attr = FAMILY_TO_SETTING[name]
        try:
            family_target = getattr(settings, attr)
        except AttributeError as exc:
            raise ModelResolutionError(
                f"Settings is missing field {attr!r} required for family alias {name!r}."
            ) from exc
        if not isinstance(family_target, str) or not family_target.strip():
            raise ModelResolutionError(
                f"Family alias {name!r} resolves to empty value via settings.{attr}."
            )
        name = family_target.strip()

    # Stage 2 — short-name map
    if name in MODEL_SHORTNAMES:
        return MODEL_SHORTNAMES[name]

    # Stage 3 — full-name pass-through
    if "/" in name:
        return name

    # Stage 4 — error with helpful hint
    raise ModelResolutionError(
        f"Unknown model {name!r}. Available short-names: "
        f"{', '.join(list_shortnames())}. "
        "Or pass a full id like 'openai/gpt-5' / 'anthropic/claude-opus-4-6'. "
        "Or use 'copilot' for the subprocess Copilot CLI."
    )


# ---------------------------------------------------------------------------
# Live catalog access
# ---------------------------------------------------------------------------


def fetch_catalog_models(
    token: str,
    endpoint: str = "https://models.github.ai/catalog/models",
    *,
    timeout: int = 10,
) -> list[str]:
    """Fetch the list of available model ids from the GH Models catalog.

    The exact JSON shape returned by GitHub varies across API revisions,
    so this function is **defensive**: it tries a few common keys
    (`id`, `name`, `model_id`) and returns whatever string ids it can
    find. Returns an empty list on HTTP/parse error after logging.

    The token must have the `models:read` scope. Use the same token as
    the inference endpoint — typically `MNEMO_GHMODELS_TOKEN`,
    `GH_TOKEN`, or `GITHUB_TOKEN`.
    """
    req = urllib.request.Request(
        endpoint,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
            "User-Agent": "mnemo-admin/1.0",
        },
        method="GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
    except urllib.error.HTTPError as exc:
        logger.warning("Catalog API returned HTTP %s: %s", exc.code, exc.reason)
        return []
    except urllib.error.URLError as exc:
        logger.warning("Catalog API unreachable: %s", exc.reason)
        return []
    except TimeoutError:
        logger.warning("Catalog API timed out after %ss", timeout)
        return []

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.warning("Catalog API returned non-JSON: %s", exc)
        return []

    return _extract_model_ids(data)


def _extract_model_ids(data: Any) -> list[str]:
    """Best-effort extraction of model ids from a catalog response."""
    if isinstance(data, list):
        items = data
    elif isinstance(data, dict):
        items = data.get("models") or data.get("data") or data.get("items") or []
    else:
        return []

    ids: list[str] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        # Try common id fields, in priority order.
        for key in ("id", "name", "model_id", "modelId"):
            value = item.get(key)
            if isinstance(value, str) and "/" in value:
                ids.append(value)
                break
    return ids


def resolve_token_from_env() -> str | None:
    """Find a GH Models token in the standard env var fallback chain."""
    import os
    for var in ("MNEMO_GHMODELS_TOKEN", "GH_TOKEN", "GITHUB_TOKEN"):
        value = os.environ.get(var)
        if value:
            return value
    return None
