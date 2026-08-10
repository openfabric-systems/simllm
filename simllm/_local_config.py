"""Helpers for machine-local configuration."""

from __future__ import annotations

import os
from pathlib import Path


def path_from_env(name: str) -> Path | None:
    """Return an absolute, normalized path from an environment variable."""
    value = os.environ.get(name)
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    path = Path(text).expanduser()
    if not path.is_absolute():
        raise ValueError(
            f"{name} must be an absolute path; set it in local configuration "
            "or pass an explicit command-line path"
        )
    return path.resolve(strict=False)
