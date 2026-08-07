"""Cross-platform discovery helpers for native simulator executables."""

from __future__ import annotations

import os
import shutil
from collections.abc import Iterable
from pathlib import Path

_WINDOWS = os.name == "nt"
_CMAKE_CONFIGURATIONS = ("Release", "RelWithDebInfo", "Debug", "MinSizeRel")


def cmake_binary_candidates(
    build_root: Path,
    binary_name: str,
    *,
    subdirectory: str | None = None,
) -> list[Path]:
    """Return single- and multi-config CMake locations for one executable."""

    base = build_root / subdirectory if subdirectory else build_root
    directories = [base, *(base / config for config in _CMAKE_CONFIGURATIONS)]
    names = (f"{binary_name}.exe", binary_name) if _WINDOWS else (binary_name,)
    return [directory / name for directory in directories for name in names]


def is_runnable_file(path: Path) -> bool:
    """Return whether ``path`` can be passed directly to ``subprocess``."""

    return path.is_file() and (_WINDOWS or os.access(path, os.X_OK))


def find_native_binary(
    env_var: str,
    binary_name: str,
    candidates: Iterable[Path],
) -> Path | None:
    """Find a native executable using env, build candidates, then ``PATH``."""

    configured = os.environ.get(env_var)
    if configured:
        path = Path(configured).expanduser()
        if is_runnable_file(path):
            return path

    for path in candidates:
        if is_runnable_file(path):
            return path

    on_path = shutil.which(binary_name)
    return Path(on_path) if on_path else None
