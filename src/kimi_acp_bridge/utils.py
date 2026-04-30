"""Shared utility helpers for the bridge."""

from __future__ import annotations

import os
from pathlib import Path

import structlog

logger = structlog.get_logger()

# Environment variables that are safe to forward to Kimi subprocesses.
# Never forward secrets (API keys, tokens, passwords).
_ALLOWED_ENV_KEYS = {
    "PATH",
    "HOME",
    "USER",
    "SHELL",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "TMPDIR",
    "PWD",
    "KIMI_",
}


def build_controlled_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    """Build a controlled environment dict for Kimi subprocesses.

    Only forwards known-safe variables. Logs the key names (never values).
    """
    controlled: dict[str, str] = {}
    for key, value in os.environ.items():
        if any(key.startswith(prefix) or key == prefix for prefix in _ALLOWED_ENV_KEYS):
            controlled[key] = value

    if extra:
        controlled.update(extra)

    logger.debug(
        "controlled_env_prepared",
        keys=list(controlled.keys()),
        count=len(controlled),
    )
    return controlled


def validate_work_dir(path: str) -> str:
    """Validate a work directory path.

    Rules:
    - Must be absolute.
    - Must not contain parent-directory traversal (..).
    - Must exist and be a directory.

    Returns the canonical resolved path.
    """
    p = Path(path)
    if not p.is_absolute():
        raise ValueError(f"Work directory must be absolute: {path}")

    # Check for traversal components
    try:
        resolved = p.resolve(strict=True)
    except FileNotFoundError as exc:
        raise ValueError(f"Work directory does not exist: {path}") from exc

    if not resolved.is_dir():
        raise ValueError(f"Work directory is not a directory: {path}")

    # Ensure no '..' remains after resolve (shouldn't happen, but defensive)
    if ".." in str(p):
        raise ValueError(f"Work directory contains invalid traversal: {path}")

    return str(resolved)
