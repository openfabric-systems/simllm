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


@dataclass(frozen=True)
class LogGopsimInvocationProvenance:
    """Exact native invocation and artifact identity for one fabric phase."""

    goal_path: Path
    goal_sha256: str
    goal_binary_sha256: str
    argv: tuple[str, ...]
    exact_g_string: str
    max_finish_ps: int


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
        timeout_s: int,
    ) -> None:
        self._loggopsim_binary = binary
        self._loggp_parameters = parameters
        self._loggopsim_timeout_s = timeout_s
        self.loggopsim_invocations: list[LogGopsimInvocationProvenance] = []
        super().__init__(config)

    def _run_goal(self, plan, goal_path: Path, completion_csv: Path):
        del plan, completion_csv
        goal_payload = goal_path.read_bytes()
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
        goal_bin = to_binary(goal_path)
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
            timeout_s=config.timeout_s,
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
    "LogGopsimInvocationProvenance",
    "LogGopsimSinkProvenance",
    "LogGopsimStepSink",
    "LogGopsimStepSinkConfig",
]
