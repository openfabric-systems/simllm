"""Convert GOAL text to the binary format the simulators consume.

The ``txt2bin`` tool builds with the htsim backend; discovery order is the
``SIMLLM_TXT2BIN`` environment variable, the CMake build tree, the legacy
checked-in source-tree executable on Unix, then ``PATH``.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from simllm._native import cmake_binary_candidates, find_native_binary

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_BUILD_ROOT = _REPO_ROOT / "build" / "htsim"
_SUBMODULE_TXT2BIN = _REPO_ROOT / "third_party" / "htsim" / "htsim" / "sim" / "lgs" / "txt2bin"


def find_txt2bin() -> Path | None:
    candidates = [
        *cmake_binary_candidates(_DEFAULT_BUILD_ROOT, "txt2bin"),
        _SUBMODULE_TXT2BIN,
    ]
    return find_native_binary("SIMLLM_TXT2BIN", "txt2bin", candidates)


def to_binary(goal_path: str | Path, bin_path: str | Path | None = None,
              tool: str | Path | None = None) -> Path:
    """Convert ``goal_path`` to binary, returning the ``.bin`` path."""
    goal_path = Path(goal_path)
    bin_path = Path(bin_path) if bin_path else goal_path.with_suffix(".bin")
    tool = Path(tool) if tool else find_txt2bin()
    if tool is None:
        raise FileNotFoundError(
            "txt2bin not found: set SIMLLM_TXT2BIN or init the htsim submodule"
        )
    result = subprocess.run(
        [str(tool), "-i", str(goal_path), "-o", str(bin_path)],
        capture_output=True, text=True, timeout=300, check=False,
    )
    if result.returncode != 0 or not bin_path.is_file() or bin_path.stat().st_size == 0:
        raise RuntimeError(
            f"txt2bin failed (exit {result.returncode}): {result.stderr.strip() or result.stdout.strip()}"
        )
    return bin_path
