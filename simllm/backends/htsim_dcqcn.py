"""Run GOAL schedules on the RoCEv2 DCQCN profile.

``htsim_dcqcn_atlahs`` executes a binary GOAL over a topology-file ns-tm3
Clos with go-back-N RoCEv2 endpoints, CNP-driven DCQCN rate control,
switch-egress RED marking and link-local PFC. Same completion-CSV schema
and quiescence contract as the RNIC profiles.

Binary discovery mirrors :mod:`simllm.backends.htsim_rnic`:
``SIMLLM_HTSIM_DCQCN``, then the README build location, then ``PATH``.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from simllm.backends.htsim_rnic import RnicRunResult, parse_completion_csv

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_BUILD_BINARY = _REPO_ROOT / "build" / "htsim" / "datacenter" / "htsim_dcqcn_atlahs"


def find_htsim_dcqcn() -> Path | None:
    env = os.environ.get("SIMLLM_HTSIM_DCQCN")
    if env and Path(env).is_file():
        return Path(env)
    if _DEFAULT_BUILD_BINARY.is_file() and os.access(_DEFAULT_BUILD_BINARY, os.X_OK):
        return _DEFAULT_BUILD_BINARY
    on_path = shutil.which("htsim_dcqcn_atlahs")
    return Path(on_path) if on_path else None


@dataclass
class HtsimDcqcnConfig:
    """Arguments for one ``htsim_dcqcn_atlahs`` GOAL run."""

    goal_bin: Path
    #: ns-tm3 Clos topology file (required; no generated fallback)
    topology: Path
    #: endpoint link capacity in bits per second
    link_bps: int
    completion_csv: Path | None = None
    #: extra raw flags, e.g. {"-ecn_kmin_bytes": "65536"}
    extra_flags: dict[str, str] = field(default_factory=dict)


def build_htsim_dcqcn_command(binary: Path, cfg: HtsimDcqcnConfig) -> list[str]:
    argv = [
        str(binary),
        "-goal", str(cfg.goal_bin),
        "-topology", str(cfg.topology),
        "-link_bps", str(cfg.link_bps),
    ]
    if cfg.completion_csv is not None:
        argv += ["-completion_csv", str(cfg.completion_csv)]
    for flag, value in cfg.extra_flags.items():
        argv += [flag, value]
    return argv


def run_htsim_dcqcn(cfg: HtsimDcqcnConfig, binary: Path | None = None,
                    timeout_s: int = 600) -> RnicRunResult:
    binary = binary or find_htsim_dcqcn()
    if binary is None:
        raise FileNotFoundError(
            "htsim_dcqcn_atlahs not found: set SIMLLM_HTSIM_DCQCN or build "
            "the htsim submodule (see README)"
        )
    result = subprocess.run(
        build_htsim_dcqcn_command(binary, cfg),
        capture_output=True, text=True, timeout=timeout_s, check=False,
    )
    manifest = [l for l in result.stdout.splitlines() if l.startswith("[DCQCN manifest]")]
    quiescent = any("physical_quiescence=verified" in l for l in manifest)
    if result.returncode != 0:
        raise RuntimeError(
            f"htsim_dcqcn_atlahs exit {result.returncode}: "
            f"{result.stderr.strip() or result.stdout.strip()}"
        )
    if not quiescent:
        raise RuntimeError("htsim_dcqcn_atlahs finished without physical_quiescence=verified")
    flows = (
        parse_completion_csv(cfg.completion_csv)
        if cfg.completion_csv is not None and Path(cfg.completion_csv).is_file()
        else []
    )
    return RnicRunResult(flows=flows, manifest=manifest, quiescent=quiescent)
