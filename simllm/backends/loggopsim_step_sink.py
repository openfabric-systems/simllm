"""Closed-loop step sink for the selectable LogGOPSim ideal-network level.

The sink delegates lowering, graph projection, GOAL rendering and analytic
intra-node service to :class:`HtsimStepSink`. Only the fabric invocation is
replaced. This keeps one artifact authority and makes the ideal level consume
the same checked GOAL bytes as the packet and fluid levels.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path

from simllm.backends.loggopsim import (
    DEFAULT_LOGGOPSIM_EAGER_THRESHOLD_BYTES,
    DerivedLoggpParams,
    build_loggopsim_command,
    derive_loggp_params,
    find_loggopsim,
    run_loggopsim,
)
from simllm.backends.step_sink import HtsimStepSink, HtsimStepSinkConfig
from simllm.compute import (
    GPU_ENVELOPES,
    ComputeProvider,
    GpuSpec,
    HostInitiationModel,
    ModelDims,
    RooflineProvider,
)
from simllm.core import (
    DependencyLevel,
    NetworkLevel,
    PrecisionConfig,
    StepRecord,
    StepResult,
    check_precision_selection,
    compute_level_for_provider,
    locality_level_for_placement,
)
from simllm.goal import to_binary
from simllm.placement import PlacementManifest
from simllm.traffic import (
    DEFAULT_NVLINK_BANDWIDTH_BYTES_PER_SECOND,
    RoutedMoeSupply,
)

_PAYLOAD_RE = re.compile(r"\b(?:send|recv) (\d+)b\b")
_RANK_RE = re.compile(r"^rank (\d+) \{$")
_RECEIVE_RE = re.compile(
    r"^(?P<label>\S+): recv \d+b from (?P<source>\d+) tag \d+(?: nic \d+)?$"
)
_DEPENDENCY_RE = re.compile(
    r"^(?P<label>\S+) (?P<relation>i?requires) (?P<predecessor>\S+)$"
)

LOGGOPSIM_FAN_IN_STAMP_SCHEMA = "simllm-loggopsim-fan-in-envelope-v1"
LOGGOPSIM_FAN_IN_STUDY = "examples/frontier_ladder_v1/RESULTS.md"


@dataclass(frozen=True)
class LogGopsimFanInDestination:
    """One receiver with incoming flows that may overlap."""

    receiver_rank: int
    source_ranks: tuple[int, ...]

    def to_json(self) -> dict[str, object]:
        """Return a portable record of the detected receiver fan-in."""

        return {
            "receiver_rank": self.receiver_rank,
            "source_ranks": list(self.source_ranks),
        }


@dataclass(frozen=True)
class LogGopsimFanInStamp:
    """Envelope decision made from one rendered GOAL."""

    fan_in_detected: bool
    acknowledged: bool
    destinations: tuple[LogGopsimFanInDestination, ...]

    def to_json(self) -> dict[str, object]:
        """Return the refusal-envelope stamp carried by run records."""

        return {
            "schema": LOGGOPSIM_FAN_IN_STAMP_SCHEMA,
            "fan_in_detected": self.fan_in_detected,
            "acknowledged": self.acknowledged,
            "mechanism": "receiver per-byte gap unmodeled",
            "frozen_cell_error": "about 8x optimistic",
            "study": LOGGOPSIM_FAN_IN_STUDY,
            "destinations": [item.to_json() for item in self.destinations],
        }


class LogGopsimFanInError(ValueError):
    """Raised when the ideal level sees unacknowledged receiver fan-in."""


def _finish_ancestors(
    label: str,
    predecessors: dict[str, set[str]],
    memo: dict[str, set[str]],
) -> set[str]:
    if label in memo:
        return memo[label]
    ancestors: set[str] = set()
    for predecessor in predecessors.get(label, set()):
        ancestors.add(predecessor)
        ancestors.update(_finish_ancestors(predecessor, predecessors, memo))
    memo[label] = ancestors
    return ancestors


def inspect_loggopsim_fan_in(
    goal_text: str,
    *,
    acknowledge_fan_in: bool = False,
) -> LogGopsimFanInStamp:
    """Inspect receiver dependencies and enforce the ideal-level envelope.

    Two receives from different sources may overlap unless one is transitively
    gated by completion of the other through ``requires`` edges. ``irequires``
    preserves issue order but permits overlap, so it cannot make fan-in clean.
    """

    if not isinstance(goal_text, str):
        raise TypeError("goal_text must be a string")
    if type(acknowledge_fan_in) is not bool:
        raise TypeError("acknowledge_fan_in must be a boolean")

    receives: dict[int, list[tuple[str, int]]] = {}
    predecessors: dict[int, dict[str, set[str]]] = {}
    current_rank: int | None = None
    for line in goal_text.splitlines():
        text = line.strip()
        if match := _RANK_RE.fullmatch(text):
            current_rank = int(match.group(1))
            receives.setdefault(current_rank, [])
            predecessors.setdefault(current_rank, {})
            continue
        if text == "}":
            current_rank = None
            continue
        if current_rank is None:
            continue
        if match := _RECEIVE_RE.fullmatch(text):
            receives[current_rank].append(
                (match.group("label"), int(match.group("source")))
            )
            continue
        if (match := _DEPENDENCY_RE.fullmatch(text)) and match.group(
            "relation"
        ) == "requires":
            predecessors[current_rank].setdefault(match.group("label"), set()).add(
                match.group("predecessor")
            )

    fan_in_destinations = []
    for receiver_rank, incoming in sorted(receives.items()):
        memo: dict[str, set[str]] = {}
        overlapping_sources: set[int] = set()
        for (left_label, left_source), (right_label, right_source) in combinations(
            incoming, 2
        ):
            if left_source == right_source:
                continue
            left_ancestors = _finish_ancestors(
                left_label, predecessors[receiver_rank], memo
            )
            right_ancestors = _finish_ancestors(
                right_label, predecessors[receiver_rank], memo
            )
            if right_label not in left_ancestors and left_label not in right_ancestors:
                overlapping_sources.update((left_source, right_source))
        if overlapping_sources:
            fan_in_destinations.append(
                LogGopsimFanInDestination(
                    receiver_rank=receiver_rank,
                    source_ranks=tuple(sorted(overlapping_sources)),
                )
            )

    destinations = tuple(fan_in_destinations)
    detected = bool(destinations)
    stamp = LogGopsimFanInStamp(
        fan_in_detected=detected,
        acknowledged=detected and acknowledge_fan_in,
        destinations=destinations,
    )
    if detected and not acknowledge_fan_in:
        summary = "; ".join(
            f"receiver rank {item.receiver_rank} from sources "
            f"{', '.join(str(rank) for rank in item.source_ranks)}"
            for item in destinations
        )
        raise LogGopsimFanInError(
            "LogGOPSim ideal level refuses receiver fan-in by default "
            f"({summary}): the receiver per-byte gap is unmodeled and the "
            "frozen cell is about 8x optimistic. See "
            f"{LOGGOPSIM_FAN_IN_STUDY}. Pass acknowledge_fan_in=True only "
            "for a deliberate envelope measurement."
        )
    return stamp


@dataclass(frozen=True)
class LogGopsimInvocationProvenance:
    """Exact native invocation and artifact identity for one fabric phase."""

    goal_path: Path
    goal_sha256: str
    goal_binary_sha256: str
    argv: tuple[str, ...]
    exact_g_string: str
    max_finish_ps: int
    fan_in: LogGopsimFanInStamp


@dataclass(frozen=True)
class LogGopsimSinkProvenance:
    """Resolved declared parameters and native identity for one sink."""

    binary_sha256: str
    parameters: DerivedLoggpParams
    invocations: tuple[LogGopsimInvocationProvenance, ...]


@dataclass
class LogGopsimStepSinkConfig:
    """One closed-loop deployment priced by the ideal LogGOP model.

    ``latency_ns`` declares ``L`` and ``linkspeed_bps`` derives ``G``. The
    conservative defaults declare ``o = g = O = 0`` and an eager threshold at
    the largest signed 64-bit payload. Each emitted artifact is checked before
    execution so a payload can never silently cross into rendezvous mode.
    """

    tp_ranks: Sequence[int]
    dims: ModelDims
    workdir: Path
    latency_ns: int
    ep_ranks: Sequence[int] | None = None
    linkspeed_bps: int = 400_000_000_000
    overhead_ns: int = 0
    message_gap_ns: int = 0
    byte_overhead_ns: int = 0
    rendezvous_threshold_bytes: int = DEFAULT_LOGGOPSIM_EAGER_THRESHOLD_BYTES
    binary: Path | None = None
    txt2bin: Path | None = None
    timeout_s: int = 600
    provider: ComputeProvider = field(
        default_factory=lambda: RooflineProvider(efficiency=0.7)
    )
    gpu: GpuSpec = GPU_ENVELOPES["b100"]
    host_model: HostInitiationModel = field(default_factory=HostInitiationModel.ideal)
    base_tag: int = 1000
    num_goal_ranks: int | None = None
    routed_moe_supply: RoutedMoeSupply | None = None
    placement_manifest: PlacementManifest | None = None
    nvlink_bandwidth_bytes_per_second: int = (
        DEFAULT_NVLINK_BANDWIDTH_BYTES_PER_SECOND
    )
    precision: PrecisionConfig | None = None
    acknowledge_fan_in: bool = False
    parameters: DerivedLoggpParams = field(init=False)
    selected_precision_levels: dict[str, object] = field(
        init=False,
        repr=False,
        compare=False,
        default_factory=dict,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.tp_ranks, Sequence):
            raise TypeError("tp_ranks must be a sequence")
        if self.ep_ranks is not None and not isinstance(self.ep_ranks, Sequence):
            raise TypeError("ep_ranks must be a sequence or None")
        self.tp_ranks = tuple(self.tp_ranks)
        self.ep_ranks = None if self.ep_ranks is None else tuple(self.ep_ranks)
        if isinstance(self.timeout_s, bool) or not isinstance(self.timeout_s, int):
            raise TypeError("timeout_s must be an integer")
        if self.timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        if type(self.acknowledge_fan_in) is not bool:
            raise TypeError("acknowledge_fan_in must be a boolean")
        self.parameters = derive_loggp_params(
            rate_bits_per_second=self.linkspeed_bps,
            latency_ns=self.latency_ns,
            overhead_ns=self.overhead_ns,
            message_gap_ns=self.message_gap_ns,
            byte_overhead_ns=self.byte_overhead_ns,
            rendezvous_threshold_bytes=self.rendezvous_threshold_bytes,
        )
        self.selected_precision_levels = check_precision_selection(
            self.precision,
            compute=compute_level_for_provider(self.provider),
            dependency=DependencyLevel.SERIAL,
            locality=locality_level_for_placement(self.placement_manifest),
            network=NetworkLevel.LOGGOPSIM_IDEAL,
            selection_source="LogGopsimStepSinkConfig",
        )


class _LogGopsimExecutionSink(HtsimStepSink):
    """Shared planner with only the native fabric call substituted."""

    def __init__(
        self,
        config: HtsimStepSinkConfig,
        *,
        binary: Path,
        parameters: DerivedLoggpParams,
        txt2bin: Path | None,
        timeout_s: int,
        acknowledge_fan_in: bool,
    ) -> None:
        self._loggopsim_binary = binary
        self._loggp_parameters = parameters
        self._txt2bin = txt2bin
        self._loggopsim_timeout_s = timeout_s
        self._acknowledge_fan_in = acknowledge_fan_in
        self.loggopsim_invocations: list[LogGopsimInvocationProvenance] = []
        super().__init__(config)

    def _run_goal(self, plan, goal_path: Path, completion_csv: Path):
        del plan, completion_csv
        goal_payload = goal_path.read_bytes()
        fan_in = inspect_loggopsim_fan_in(
            goal_payload.decode("utf-8"),
            acknowledge_fan_in=self._acknowledge_fan_in,
        )
        payloads = tuple(
            int(match.group(1))
            for match in _PAYLOAD_RE.finditer(goal_payload.decode("utf-8"))
        )
        threshold = int(self._loggp_parameters.rendezvous_threshold.value)
        if payloads and max(payloads) > threshold:
            raise ValueError(
                "LogGOPSim ideal level requires every rendered payload to be "
                f"eager, but {max(payloads)} bytes exceeds declared S={threshold}"
            )
        goal_bin = to_binary(goal_path, tool=self._txt2bin)
        invocation = self._loggp_parameters.to_loggopsim_config(goal_bin)
        argv = tuple(build_loggopsim_command(self._loggopsim_binary, invocation))
        result = run_loggopsim(
            invocation,
            binary=self._loggopsim_binary,
            timeout_s=self._loggopsim_timeout_s,
        )
        self.loggopsim_invocations.append(
            LogGopsimInvocationProvenance(
                goal_path=goal_path,
                goal_sha256=hashlib.sha256(goal_payload).hexdigest(),
                goal_binary_sha256=hashlib.sha256(goal_bin.read_bytes()).hexdigest(),
                argv=argv,
                exact_g_string=self._loggp_parameters.exact_g_string,
                max_finish_ps=result.max_finish_ps,
                fan_in=fan_in,
            )
        )
        return result


class LogGopsimStepSink:
    """Step sink that prices checked fabric artifacts through LogGOPSim."""

    def __init__(self, config: LogGopsimStepSinkConfig) -> None:
        self.config = config
        binary = config.binary or find_loggopsim()
        if binary is None:
            raise FileNotFoundError(
                "LogGOPSim not found: set SIMLLM_LOGGOPSIM or pass "
                "LogGopsimStepSinkConfig.binary"
            )
        binary = Path(binary)
        binary_sha256 = hashlib.sha256(binary.read_bytes()).hexdigest()
        planner_config = HtsimStepSinkConfig(
            profile="rnic-nn-fluid",
            tp_ranks=config.tp_ranks,
            dims=config.dims,
            workdir=config.workdir,
            ep_ranks=config.ep_ranks,
            linkspeed_bps=config.linkspeed_bps,
            provider=config.provider,
            gpu=config.gpu,
            host_model=config.host_model,
            base_tag=config.base_tag,
            num_goal_ranks=config.num_goal_ranks,
            routed_moe_supply=config.routed_moe_supply,
            placement_manifest=config.placement_manifest,
            nvlink_bandwidth_bytes_per_second=(
                config.nvlink_bandwidth_bytes_per_second
            ),
        )
        self._sink = _LogGopsimExecutionSink(
            planner_config,
            binary=binary,
            parameters=config.parameters,
            txt2bin=config.txt2bin,
            timeout_s=config.timeout_s,
            acknowledge_fan_in=config.acknowledge_fan_in,
        )
        self._binary_sha256 = binary_sha256
        self.outcomes = self._sink.outcomes
        self.locality_outcomes = self._sink.locality_outcomes

    @property
    def provenance(self) -> LogGopsimSinkProvenance:
        """Return the complete provenance accumulated by this sink."""

        return LogGopsimSinkProvenance(
            binary_sha256=self._binary_sha256,
            parameters=self.config.parameters,
            invocations=tuple(self._sink.loggopsim_invocations),
        )

    @property
    def host_model(self) -> HostInitiationModel:
        """The one host model applied by the shared serial lowerer."""

        return self.config.host_model

    def compute_estimate_ps(self, record: StepRecord) -> int:
        """The compute-only whole-step estimate represented by the sink."""

        return self._sink.compute_estimate_ps(record)

    def __call__(self, record: StepRecord) -> StepResult | None:
        return self._sink(record)


__all__ = [
    "LOGGOPSIM_FAN_IN_STAMP_SCHEMA",
    "LOGGOPSIM_FAN_IN_STUDY",
    "LogGopsimFanInDestination",
    "LogGopsimFanInError",
    "LogGopsimFanInStamp",
    "LogGopsimInvocationProvenance",
    "LogGopsimSinkProvenance",
    "LogGopsimStepSink",
    "LogGopsimStepSinkConfig",
    "inspect_loggopsim_fan_in",
]
