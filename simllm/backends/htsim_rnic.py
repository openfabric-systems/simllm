"""Run GOAL schedules on the packet-level RNIC profiles and parse results.

``htsim_rnic`` (built from the htsim submodule) executes a binary GOAL over
one of the wired fidelity profiles: ``rnic-nn`` (packetized null-network
manifold), ``rnic-nn-fluid`` (continuous fluid manifold), ``rnic-cn``
(explicit-rate collective network over a two-tier ns-tm3 Clos). A run is
valid only if the simulator reports ``physical_quiescence=verified``.

Binary discovery order: ``SIMLLM_HTSIM_RNIC`` environment variable, then
the single- or multi-configuration ``build/htsim`` CMake layout under the
repo root, then ``PATH``.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path

from simllm._native import cmake_binary_candidates, find_native_binary
from simllm.backends._child_process import (
    prepare_owned_child_runtime,
    run_owned_process,
)

RNIC_PROFILES = ("rnic-nn", "rnic-nn-fluid", "rnic-cn")

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_BUILD_ROOT = _REPO_ROOT / "build" / "htsim"
_GOAL_FINISH_RE = re.compile(r"^Maximum finishing time at host \d+: (\d+)(?:\s|$)")


def find_htsim_rnic() -> Path | None:
    return find_native_binary(
        "SIMLLM_HTSIM_RNIC",
        "htsim_rnic",
        cmake_binary_candidates(
            _DEFAULT_BUILD_ROOT, "htsim_rnic", subdirectory="datacenter"
        ),
    )


@dataclass
class HtsimRnicConfig:
    """Arguments for one ``htsim_rnic`` GOAL run."""

    goal_bin: Path
    profile: str
    #: endpoint link capacity in bits per second (htsim_rnic convention)
    linkspeed_bps: int
    completion_csv: Path | None = None
    #: two-tier Clos topology file (rnic-cn only; omitted = generated Clos)
    topology: Path | None = None
    #: extra raw flags, e.g. {"-rnic_cn_margin_ppm": "900000"}
    extra_flags: dict[str, str] = field(default_factory=dict)
    #: regression-only negative control; production runs must retain False
    unsafe_disable_child_lifetime_binding: bool = False

    def __post_init__(self) -> None:
        if self.profile not in RNIC_PROFILES:
            raise ValueError(f"profile must be one of {RNIC_PROFILES}")
        if type(self.unsafe_disable_child_lifetime_binding) is not bool:
            raise TypeError("unsafe_disable_child_lifetime_binding must be a boolean")


def build_htsim_rnic_command(binary: Path, cfg: HtsimRnicConfig) -> list[str]:
    argv = [
        str(binary),
        "-goal", str(cfg.goal_bin),
        "-linkspeed_bps", str(cfg.linkspeed_bps),
        "-rnic_profile", cfg.profile,
    ]
    if cfg.completion_csv is not None:
        argv += ["-completion_csv", str(cfg.completion_csv)]
    if cfg.topology is not None:
        argv += ["-topo", str(cfg.topology)]
    for flag, value in cfg.extra_flags.items():
        argv += [flag, value]
    return argv


@dataclass(frozen=True)
class FlowCompletion:
    """One WQE-level row of the completion CSV.

    Queue and transport fields are optional so old backend outputs remain
    readable. New HTSIM outputs append them after the stable flow prefix.
    """

    profile: str
    flow_id: int
    source: int
    destination: int
    tag: int
    payload_bytes: int
    start_time_ps: int
    completion_time_ps: int
    fct_ps: int
    wqe_id: int | None = None
    sq_id: int | None = None
    rq_id: int | None = None
    cq_id: int | None = None
    sq_post_sequence: int | None = None
    sq_dispatch_sequence: int | None = None
    cq_post_sequence: int | None = None
    cq_consume_sequence: int | None = None
    transport_kind: str | None = None
    transport_object_id: int | None = None


@dataclass
class RnicRunResult:
    flows: list[FlowCompletion]
    manifest: list[str]
    quiescent: bool
    goal_completion_time_ps: int | None = None

    def job_completion_time_ps(self) -> int:
        """Completion of all represented schedule work released at time zero."""
        candidates = [flow.completion_time_ps for flow in self.flows]
        if self.goal_completion_time_ps is not None:
            candidates.append(self.goal_completion_time_ps)
        if candidates:
            return max(candidates)
        raise ValueError("run produced neither flows nor a GOAL completion time")


def _parse_goal_completion_time_ps(stdout: str) -> int | None:
    """Parse LogGOPSim's whole-nanosecond schedule completion summary."""

    finishing_ns = [
        int(match.group(1))
        for line in stdout.splitlines()
        if (match := _GOAL_FINISH_RE.match(line)) is not None
    ]
    return max(finishing_ns) * 1000 if finishing_ns else None


def parse_completion_csv(path: str | Path) -> list[FlowCompletion]:
    def optional_int(row: dict[str, str], field_name: str) -> int | None:
        value = row.get(field_name)
        return None if value in (None, "") else int(value)

    flows = []
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            flows.append(FlowCompletion(
                profile=row["profile"],
                flow_id=int(row["flow_id"]),
                source=int(row["source"]),
                destination=int(row["destination"]),
                tag=int(row["tag"]),
                payload_bytes=int(row["payload_bytes"]),
                start_time_ps=int(row["start_time_ps"]),
                completion_time_ps=int(row["completion_time_ps"]),
                fct_ps=int(row["fct_ps"]),
                wqe_id=optional_int(row, "wqe_id"),
                sq_id=optional_int(row, "sq_id"),
                rq_id=optional_int(row, "rq_id"),
                cq_id=optional_int(row, "cq_id"),
                sq_post_sequence=optional_int(row, "sq_post_sequence"),
                sq_dispatch_sequence=optional_int(row, "sq_dispatch_sequence"),
                cq_post_sequence=optional_int(row, "cq_post_sequence"),
                cq_consume_sequence=optional_int(row, "cq_consume_sequence"),
                transport_kind=row.get("transport_kind") or None,
                transport_object_id=optional_int(row, "transport_object_id"),
            ))
    return flows


def run_htsim_rnic(cfg: HtsimRnicConfig, binary: Path | None = None,
                   timeout_s: int = 600) -> RnicRunResult:
    binary = binary or find_htsim_rnic()
    if binary is None:
        raise FileNotFoundError(
            "htsim_rnic not found: set SIMLLM_HTSIM_RNIC or build the htsim "
            "submodule (see README)"
        )
    result = run_owned_process(
        build_htsim_rnic_command(binary, cfg),
        timeout_s=timeout_s,
        unsafe_unmanaged=cfg.unsafe_disable_child_lifetime_binding,
    )
    manifest = [l for l in result.stdout.splitlines() if l.startswith("[RNIC manifest]")]
    quiescent = any("physical_quiescence=verified" in l for l in manifest)
    if result.returncode != 0:
        raise RuntimeError(
            f"htsim_rnic exit {result.returncode}: {result.stderr.strip() or result.stdout.strip()}"
        )
    if not quiescent:
        raise RuntimeError("htsim_rnic finished without physical_quiescence=verified")
    flows = (
        parse_completion_csv(cfg.completion_csv)
        if cfg.completion_csv is not None and Path(cfg.completion_csv).is_file()
        else []
    )
    return RnicRunResult(
        flows=flows,
        manifest=manifest,
        quiescent=quiescent,
        goal_completion_time_ps=_parse_goal_completion_time_ps(result.stdout),
    )


def prepare_htsim_child_lifetime() -> None:
    """Install the main-thread cleanup boundary before prepared workers run."""

    prepare_owned_child_runtime()
