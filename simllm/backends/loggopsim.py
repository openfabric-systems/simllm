"""Run GOAL schedules on LogGOPSim, the analytical flow-level simulator.

``LogGOPSim`` consumes the same binary GOAL as the htsim profiles and costs it
with the LogGOPS model instead of a packet fabric, so it is the fast level for
sweeps that only need a schedule completion time. This module is the
invocation seam and its result parser, in the same shape as
:mod:`simllm.backends.htsim_rnic`. It is not a new fidelity level: the fluid
fast level itself is owned by TRAF-20.

Binary discovery order: the ``SIMLLM_LOGGOPSIM`` environment variable, then
the single- or multi-configuration ``build/loggopsim`` CMake layout under the
repo root, then the ATLAHS submodule's own make output, then ``PATH``.

LogGOPS parameters keep the units the tool defines in
``sim/LogGOPSim/simulator.ggo``: ``L``, ``o``, ``g`` and ``O`` in whole
nanoseconds, ``G`` in nanoseconds per byte, and ``S`` in bytes. Parsed
completion times are converted to picoseconds, exactly a factor of 1000, so
callers see the repository's picosecond convention.
"""

from __future__ import annotations

import math
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from simllm._native import cmake_binary_candidates, find_native_binary

#: Network models ``simulator.ggo`` accepts for ``-n``.
LOGGOPSIM_NETWORK_TYPES = ("LogGP", "simple")
LOGGOPSIM_DECLARED_EVIDENCE_SOURCE = "DECLARED"
DEFAULT_LOGGOPSIM_EAGER_THRESHOLD_BYTES = (1 << 63) - 1

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_BUILD_ROOT = _REPO_ROOT / "build" / "loggopsim"
_SUBMODULE_LOGGOPSIM = (
    _REPO_ROOT / "third_party" / "atlahs" / "sim" / "LogGOPSim" / "LogGOPSim"
)

_BANNER_RE = re.compile(
    r"^size: (\d+) \((\d+) CPUs, (\d+) NICs\); "
    r"L=(-?\d+), o=(-?\d+) g=(-?\d+), G=([^,]+), O=(-?\d+), P=(\d+), S=(\d+)"
)
_HOST_TIME_RE = re.compile(r"^Host (\d+): (\d+)\s*$")
_MAX_FINISH_RE = re.compile(r"^Maximum finishing time at host (\d+): (\d+)(?:\s|$)")
_AVERAGE_FCT_RE = re.compile(r"^Average FCT is (\S+)\s*$")
_UNMATCHED_QUEUE_RE = re.compile(
    r"^(?:unexpected|receive) queue on host \d+ contains \d+ elements!"
)


def find_loggopsim() -> Path | None:
    """Find the ``LogGOPSim`` executable, or ``None`` when none is available."""

    return find_native_binary(
        "SIMLLM_LOGGOPSIM",
        "LogGOPSim",
        [
            *cmake_binary_candidates(_DEFAULT_BUILD_ROOT, "LogGOPSim"),
            _SUBMODULE_LOGGOPSIM,
        ],
    )


@dataclass
class LogGopsimConfig:
    """Arguments for one ``LogGOPSim`` GOAL run.

    Every LogGOPS field keeps the tool's own unit, so a configuration reads the
    same way as the option grammar it is rendered into. Defaults are the tool
    defaults, which keeps a bare configuration reproducible.
    """

    goal_bin: Path
    #: LogGOPS L, the latency between two endpoints, in whole nanoseconds
    latency_ns: int = 2500
    #: LogGOPS o, the per-message CPU overhead, in whole nanoseconds
    overhead_ns: int = 1500
    #: LogGOPS g, the gap between two consecutive messages, in nanoseconds
    message_gap_ns: int = 1000
    #: LogGOPS G, the gap per payload byte, in nanoseconds per byte
    byte_gap_ns: float = 6.0
    #: exact decimal spelling for G; None uses the shortest float spelling
    byte_gap_ns_string: str | None = None
    #: LogGOPS O, the CPU overhead per payload byte, in whole nanoseconds
    byte_overhead_ns: int = 0
    #: LogGOPS S, the eager to rendezvous switch point, in bytes
    rendezvous_threshold_bytes: int = 65535
    network_type: str = "LogGP"
    #: annotated dot topology consumed by the ``simple`` network model
    network_file: Path | None = None
    #: print only the maximum finishing time, never the per-host block
    batch_mode: bool = False
    #: optional send/receive dependency dump
    comm_dependency_file: Path | None = None
    #: extra raw flags, e.g. {"--progress": ""} for a valueless option
    extra_flags: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.network_type not in LOGGOPSIM_NETWORK_TYPES:
            raise ValueError(f"network_type must be one of {LOGGOPSIM_NETWORK_TYPES}")
        if type(self.batch_mode) is not bool:
            raise TypeError("batch_mode must be a boolean")
        if self.rendezvous_threshold_bytes < 0:
            raise ValueError("rendezvous_threshold_bytes must not be negative")
        if self.byte_gap_ns_string is not None:
            if not isinstance(self.byte_gap_ns_string, str):
                raise TypeError("byte_gap_ns_string must be a string or None")
            try:
                parsed_gap = float(self.byte_gap_ns_string)
            except ValueError:
                parsed_gap = float("nan")
            if not math.isfinite(parsed_gap) or parsed_gap < 0:
                raise ValueError(
                    "byte_gap_ns_string must be a finite nonnegative decimal"
                )
        if self.network_type == "simple" and self.network_file is None:
            raise ValueError("the simple network model requires network_file")


@dataclass(frozen=True)
class DeclaredLoggpParameter:
    """One LogGP parameter and the evidence class behind its value."""

    value: int | str
    units: str
    source: str
    evidence_source: str = LOGGOPSIM_DECLARED_EVIDENCE_SOURCE

    def to_json(self) -> dict[str, int | str]:
        """Return a JSON-ready declaration record."""

        return {
            "value": self.value,
            "units": self.units,
            "source": self.source,
            "evidence_source": self.evidence_source,
        }


@dataclass(frozen=True)
class DerivedLoggpParams:
    """Declared LogGP parameters derived for one ideal-network level.

    ``L`` comes from the caller's declared latency. ``G`` is the ideal wire
    serialization gap derived from the declared bit rate. The conservative
    defaults set CPU overhead and both message gaps to zero, and put the eager
    threshold at the largest positive signed 64-bit byte count. The sink
    checks every rendered payload against that threshold before invocation.
    """

    latency: DeclaredLoggpParameter
    overhead: DeclaredLoggpParameter
    message_gap: DeclaredLoggpParameter
    byte_gap: DeclaredLoggpParameter
    byte_overhead: DeclaredLoggpParameter
    rendezvous_threshold: DeclaredLoggpParameter

    @property
    def exact_g_string(self) -> str:
        """The exact decimal string passed to LogGOPSim for ``G``."""

        value = self.byte_gap.value
        if not isinstance(value, str):
            raise TypeError("derived byte gap lost its decimal spelling")
        return value

    def to_loggopsim_config(self, goal_bin: Path) -> LogGopsimConfig:
        """Build one native invocation config without changing any spelling."""

        return LogGopsimConfig(
            goal_bin=goal_bin,
            latency_ns=int(self.latency.value),
            overhead_ns=int(self.overhead.value),
            message_gap_ns=int(self.message_gap.value),
            byte_gap_ns=float(self.exact_g_string),
            byte_gap_ns_string=self.exact_g_string,
            byte_overhead_ns=int(self.byte_overhead.value),
            rendezvous_threshold_bytes=int(self.rendezvous_threshold.value),
            network_type="LogGP",
        )

    def to_json(self) -> dict[str, Any]:
        """Return the six named parameters as a JSON-ready run record."""

        return {
            "L": self.latency.to_json(),
            "o": self.overhead.to_json(),
            "g": self.message_gap.to_json(),
            "G": self.byte_gap.to_json(),
            "O": self.byte_overhead.to_json(),
            "S": self.rendezvous_threshold.to_json(),
        }


def _nonnegative_integer(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def derive_loggp_params(
    *,
    rate_bits_per_second: int,
    latency_ns: int,
    overhead_ns: int = 0,
    message_gap_ns: int = 0,
    byte_overhead_ns: int = 0,
    rendezvous_threshold_bytes: int = DEFAULT_LOGGOPSIM_EAGER_THRESHOLD_BYTES,
) -> DerivedLoggpParams:
    """Derive the frozen ideal-network LogGP mapping from declared inputs.

    ``G`` is ``8e9 / rate_bits_per_second`` nanoseconds per byte. Python's
    ``repr`` supplies the shortest decimal that round-trips to the binary64
    value consumed by LogGOPSim, so 400 Gbit/s is exactly ``"0.02"`` and
    200 Gbit/s is exactly ``"0.04"`` in the argument vector and provenance.
    """

    if (
        isinstance(rate_bits_per_second, bool)
        or not isinstance(rate_bits_per_second, int)
        or rate_bits_per_second <= 0
    ):
        raise ValueError("rate_bits_per_second must be a positive integer")
    latency = _nonnegative_integer("latency_ns", latency_ns)
    overhead = _nonnegative_integer("overhead_ns", overhead_ns)
    message_gap = _nonnegative_integer("message_gap_ns", message_gap_ns)
    byte_overhead = _nonnegative_integer("byte_overhead_ns", byte_overhead_ns)
    threshold = _nonnegative_integer(
        "rendezvous_threshold_bytes", rendezvous_threshold_bytes
    )
    exact_g_string = repr(8e9 / rate_bits_per_second)
    return DerivedLoggpParams(
        latency=DeclaredLoggpParameter(
            latency, "ns", "declared latency_ns input"
        ),
        overhead=DeclaredLoggpParameter(
            overhead, "ns", "declared conservative CPU-overhead input"
        ),
        message_gap=DeclaredLoggpParameter(
            message_gap, "ns", "declared conservative message-gap input"
        ),
        byte_gap=DeclaredLoggpParameter(
            exact_g_string,
            "ns/byte",
            "8e9 / declared rate_bits_per_second",
        ),
        byte_overhead=DeclaredLoggpParameter(
            byte_overhead, "ns/byte", "declared conservative byte-overhead input"
        ),
        rendezvous_threshold=DeclaredLoggpParameter(
            threshold, "bytes", "declared eager-protocol threshold input"
        ),
    )


def build_loggopsim_command(binary: Path, cfg: LogGopsimConfig) -> list[str]:
    """Render the exact argument vector for one run."""

    argv = [
        str(binary),
        "-f", str(cfg.goal_bin),
        "-L", str(cfg.latency_ns),
        "-o", str(cfg.overhead_ns),
        "-g", str(cfg.message_gap_ns),
        "-G", (
            cfg.byte_gap_ns_string
            if cfg.byte_gap_ns_string is not None
            else repr(float(cfg.byte_gap_ns))
        ),
        "-O", str(cfg.byte_overhead_ns),
        "-S", str(cfg.rendezvous_threshold_bytes),
        "-n", cfg.network_type,
    ]
    if cfg.network_file is not None:
        argv += ["--network-file", str(cfg.network_file)]
    if cfg.batch_mode:
        argv.append("-b")
    if cfg.comm_dependency_file is not None:
        argv += ["--comm-dep-file", str(cfg.comm_dependency_file)]
    for flag, value in cfg.extra_flags.items():
        argv.append(flag)
        if value != "":
            argv.append(value)
    return argv


@dataclass(frozen=True)
class LogGopsimRunResult:
    """Parsed outcome of one ``LogGOPSim`` run.

    ``host_finish_ps`` is populated only when the tool printed its per-host
    block, which it does at 16 ranks or fewer outside batch mode. The maximum
    finishing time is always available because the tool prints one shape or
    the other.
    """

    rank_count: int
    cpu_count: int
    nic_count: int
    max_finish_host: int
    max_finish_ps: int
    host_finish_ps: dict[int, int]
    average_fct_ns: float | None
    unmatched_queue_diagnostics: tuple[str, ...]

    def job_completion_time_ps(self) -> int:
        """Completion of all schedule work released at time zero."""

        return self.max_finish_ps

    @property
    def flows(self) -> tuple[()]:
        """Per-flow transport rows are outside the ideal level's contract."""

        return ()

    @property
    def quiescent(self) -> bool:
        """A parsed run with no unmatched messages is quiescent."""

        return not self.unmatched_queue_diagnostics


def parse_loggopsim_stdout(stdout: str) -> LogGopsimRunResult:
    """Parse the banner, the finishing times and the queue diagnostics."""

    banner = None
    host_finish_ps: dict[int, int] = {}
    max_finish: tuple[int, int] | None = None
    average_fct_ns: float | None = None
    unmatched: list[str] = []
    for line in stdout.splitlines():
        if banner is None and (match := _BANNER_RE.match(line)) is not None:
            banner = match
            continue
        if (match := _HOST_TIME_RE.match(line)) is not None:
            host_finish_ps[int(match.group(1))] = int(match.group(2)) * 1000
            continue
        if (match := _MAX_FINISH_RE.match(line)) is not None:
            max_finish = (int(match.group(1)), int(match.group(2)) * 1000)
            continue
        if (match := _AVERAGE_FCT_RE.match(line)) is not None:
            # The tool prints -nan when no flow completed, so a nonfinite
            # average stays absent rather than propagating as a number.
            try:
                parsed_average = float(match.group(1))
            except ValueError:
                parsed_average = float("nan")
            average_fct_ns = parsed_average if math.isfinite(parsed_average) else None
            continue
        if _UNMATCHED_QUEUE_RE.match(line) is not None:
            unmatched.append(line)
    if banner is None:
        raise RuntimeError("LogGOPSim printed no schedule banner")
    if max_finish is None:
        if not host_finish_ps:
            raise RuntimeError("LogGOPSim printed no finishing time")
        max_finish = max(host_finish_ps.items(), key=lambda item: (item[1], -item[0]))
    return LogGopsimRunResult(
        rank_count=int(banner.group(1)),
        cpu_count=int(banner.group(2)),
        nic_count=int(banner.group(3)),
        max_finish_host=max_finish[0],
        max_finish_ps=max_finish[1],
        host_finish_ps=host_finish_ps,
        average_fct_ns=average_fct_ns,
        unmatched_queue_diagnostics=tuple(unmatched),
    )


def run_loggopsim(cfg: LogGopsimConfig, binary: Path | None = None,
                  timeout_s: int = 600) -> LogGopsimRunResult:
    """Run one GOAL schedule and return its parsed completion times."""

    binary = binary or find_loggopsim()
    if binary is None:
        raise FileNotFoundError(
            "LogGOPSim not found: set SIMLLM_LOGGOPSIM or build the ATLAHS "
            "submodule's sim/LogGOPSim target"
        )
    result = subprocess.run(
        build_loggopsim_command(binary, cfg),
        capture_output=True, text=True, timeout=timeout_s, check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"LogGOPSim exit {result.returncode}: "
            f"{result.stderr.strip() or result.stdout.strip()}"
        )
    parsed = parse_loggopsim_stdout(result.stdout)
    if parsed.unmatched_queue_diagnostics:
        raise RuntimeError(
            "LogGOPSim finished with unmatched messages: "
            + "; ".join(parsed.unmatched_queue_diagnostics)
        )
    return parsed
