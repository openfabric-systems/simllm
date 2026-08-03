"""Convert GOAL text to the binary format the simulators consume.

The ``txt2bin`` tool ships with the backends; discovery order is the
``SIMLLM_TXT2BIN`` environment variable, then the checked-in binary inside
the htsim submodule, then ``PATH``.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SUBMODULE_TXT2BIN = _REPO_ROOT / "third_party" / "htsim" / "htsim" / "sim" / "lgs" / "txt2bin"


def find_txt2bin() -> Path | None:
    env = os.environ.get("SIMLLM_TXT2BIN")
    if env and Path(env).is_file():
        return Path(env)
    if _SUBMODULE_TXT2BIN.is_file() and os.access(_SUBMODULE_TXT2BIN, os.X_OK):
        return _SUBMODULE_TXT2BIN
    on_path = shutil.which("txt2bin")
    return Path(on_path) if on_path else None


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
