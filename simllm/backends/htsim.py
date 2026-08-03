"""htsim invocation helpers.

The htsim submodule (``third_party/htsim``) builds ``htsim_uec``, which
executes a binary GOAL schedule over a Clos topology::

    htsim_uec -goal trace.bin -topo leaf_spine.topo -nodes 8 \
        -linkspeed 400000 -mtu 4096 -q 1000000 -seed 4

This module only builds command lines and parses results; process execution
and the RNIC fidelity profiles (``htsim_rnic``) land with milestone M1.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class HtsimUecConfig:
    """Arguments for a ``htsim_uec`` GOAL run."""

    goal_bin: Path
    topology: Path
    nodes: int
    #: link speed in Mbps (htsim convention)
    linkspeed_mbps: int
    mtu: int = 4096
    queue_size: int = 1_000_000
    seed: int = 1
    paths: int = 128
    load_balancing: str = "ecmp_host"


def build_htsim_uec_command(binary: Path, cfg: HtsimUecConfig) -> list[str]:
    """Return the argv list for a ``htsim_uec`` GOAL-driven run."""
    return [
        str(binary),
        "-goal", str(cfg.goal_bin),
        "-topo", str(cfg.topology),
        "-nodes", str(cfg.nodes),
        "-linkspeed", str(cfg.linkspeed_mbps),
        "-mtu", str(cfg.mtu),
        "-q", str(cfg.queue_size),
        "-seed", str(cfg.seed),
        "-paths", str(cfg.paths),
        "-strat", cfg.load_balancing,
        "-lgs_flow_stats",
    ]
