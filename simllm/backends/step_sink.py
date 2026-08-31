"""Closed-loop step sink: checked graph projections executed by ``htsim_rnic``.

:class:`HtsimStepSink` implements the adapters' step-sink contract (a
callable ``StepRecord -> StepResult | None``): for every scheduler step it
lowers the step once to an authoritative ``ExecutionGraph``, projects its
ordering into GOAL artifacts, executes them on a configured ``htsim_rnic``
profile, and returns the simulated makespan as the step latency. Plugged into
``simllm.adapters.vllm.configure(step_sink=...)`` or
``simllm.adapters.sglang.configure(step_sink=...)`` this closes the loop:
the network's completion time advances the virtual clock the frontend
scheduler sees.

Each GOAL artifact currently runs in an isolated backend process.
:class:`HtsimPersistentStepSink` accelerates a finite replay whose records are
known before the scheduler consumes them by preparing complete step plans in a
local worker pool, then serving their results in record order. It does not
preserve simulator state between artifacts or steps. Multi-artifact
``rnic-cn`` plans therefore fail closed until BACK-38 supplies state-preserving
execution; the stateless ``rnic-nn`` profiles remain supported.

A step with no collective work returns ``None``: the TP world has size 1
(or the dims declare no experts, or no EP group is configured) and the
record is a drain record with zero new tokens. ``None`` tells the adapter
that its own compute-only estimate stands, which is exactly right when
there is no network work to simulate. With MoE dims and ``ep_ranks``
configured, the authoritative graph additionally carries the dispatch and
combine all-to-alls of every layer. The legacy direct renderer remains the
byte-locked diagnostic and can be selected as the independent
``atlahs-goal`` dependency cross-check. A cross-check records ordering,
phase-frontier and completion differences without changing the graph-owned
``StepResult``. Active graph-projection artifacts have their own explicitly
accepted manifest.

When ``placement_manifest`` is present, the sink classifies every expanded
directed segment before rendering. Same-node segments become analytic
per-endpoint NVLink service and only cross-node segments reach ``htsim``.
Omitting placement selects all-remote classification. The historical direct
renderer remains available as a diagnostic, but the active sink uses the
checked graph projection. Distributed whole-operation barriers become ordered
artifact boundaries because the pinned GOAL compiler resolves dependency
labels inside one rank block. Dependency cross-check selection stays a
diagnostic switch and never names a fidelity level.

``StepLocalityOutcome`` publishes, for every executed artifact, its analytic
local service, its semantic collective base latency and the medium that owns
the local term. An artifact composes as base plus the maximum of its local and
fabric service, so that triple is the evidence a downstream reducer needs to
charge the artifact to the resource that actually decided its finish time,
rather than refusing every step that carries NVLink work.

An explicit ``collective_latency_profile`` adds one calibrated base latency at
the semantic collective boundary and selects that profile's local endpoint
rate. The default and explicit ``legacy`` selectors retain every historical
artifact and timestamp. The backend remains authoritative for propagation,
wire serialization and contention; the calibrated run record exposes its raw
transport beside the added base and a separately named propagation reference.

``collective_fixed_cost_envelope`` with ``collective_fixed_cost_arm`` is the
same selection expressed as one arm of a named bracket, so a study can run the
``off``, ``lower`` and ``upper`` arms and publish the interval instead of one
silently chosen constant. The two spellings are mutually exclusive, and the
``off`` arm resolves to no profile at all, which is exactly the default path.

The provider object, profile string and placement-manifest presence remain the
operational selectors. ``HtsimStepSinkConfig`` reports the levels they select
in ``selected_precision_levels``, and an explicit ``precision`` surface that
disagrees with any of them is refused during configuration validation, before
the workdir, any GOAL artifact or any backend process exists.

Providers may opt into an exact per-layer duration breakdown. The sink checks
that it sums to the fused estimate and emits the unequal layer costs. Existing
providers inherit the optional hook's ``None`` result and retain the historical
even timing split. An optional exact sample count on the record prices
the LM head correctly; its absence retains the historical scheduled-request
approximation.
"""

from __future__ import annotations

import copy
import hashlib
from collections import deque
from collections.abc import Sequence
from concurrent.futures import Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock

from simllm.backends.dependency_cross_check import (
    DependencyCrossCheckPlan,
    DependencyCrossCheckReport,
    complete_dependency_cross_check,
    plan_dependency_cross_check,
)
from simllm.backends.htsim_rnic import (
    RNIC_PROFILES,
    HtsimRnicConfig,
    prepare_htsim_child_lifetime,
    run_htsim_rnic,
)
from simllm.backends.step_lowerer import (
    SerialStepLowerer,
    SerialStepLowererConfig,
)
from simllm.compute import (
    GPU_ENVELOPES,
    ComputeProvider,
    GpuSpec,
    HostInitiationModel,
    ModelDims,
    RooflineProvider,
)
from simllm.core import (
    CollectiveWork,
    DependencyLevel,
    PrecisionConfig,
    StepRecord,
    StepResult,
    check_precision_selection,
    compute_level_for_provider,
    locality_level_for_placement,
    network_level_for_profile,
)
from simllm.goal import to_binary
from simllm.placement import PlacementManifest, RankMapper
from simllm.traffic import (
    COLLECTIVE_FIXED_COST_ARMS,
    DEFAULT_NVLINK_BANDWIDTH_BYTES_PER_SECOND,
    ClassifiedCommunicationPhase,
    CollectiveFixedCostEnvelope,
    CollectiveFloorCalibration,
    CollectiveFloorEstimate,
    CollectiveLatencyProfile,
    CollectiveRegistrationEvent,
    CollectiveRegistrationLedger,
    CollectiveRegistrationModel,
    RoutedMoeSupply,
    StepLocalityPlan,
    critical_collective_endpoint_bytes,
    distribute_collective_serialization_ps,
    plan_execution_graph_locality,
    project_execution_graph_goal,
    render_fabric_phase_goal,
    render_step_goal,
    resolve_collective_fixed_cost_envelope,
    resolve_collective_latency_profile,
    resolve_collective_registration,
    validate_execution_graph_locality_projection,
)

_STATEFUL_MULTI_ARTIFACT_PROFILES = frozenset({"rnic-cn"})
DEPENDENCY_CROSS_CHECK_MODES = ("atlahs-goal",)


def _require_nonnegative_timing(name: str, value: object) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


@dataclass
class HtsimStepSinkConfig:
    """One closed-loop deployment under simulation.

    ``tp_ranks`` are semantic global ranks of the tensor-parallel group (e.g.
    ``manifest.group_ranks(0, "tp")`` of a declared manifest under the
    gpu-rank mapping). Locality is classified before the later GOAL endpoint
    projection. ``dims`` is the per-rank sharded geometry the same deployment
    declares. ``ep_ranks`` is the optional expert-parallel
    group: when the dims declare experts (``dims.num_experts > 0``) and
    ``ep_ranks`` are given, the graph includes the per-layer MoE dispatch and
    combine all-to-alls over these ranks; leaving it ``None`` (or using dense
    dims) keeps the legacy direct-renderer diagnostic unchanged.
    ``routed_moe_supply`` optionally replaces uniform pair sizes with captured
    request routing at an explicit expert-placement epoch; its absence retains
    uniform routing. ``topology`` is optional: the null-network profiles
    (``rnic-nn``, ``rnic-nn-fluid``) run on the generated manifold,
    ``rnic-cn`` takes a Clos topology file.
    ``dependency_cross_check="atlahs-goal"`` runs the independently rendered
    direct schedule after the authoritative graph projection and publishes a
    diagnostic report. It never selects the scheduler-visible result.
    """

    profile: str
    tp_ranks: Sequence[int]
    dims: ModelDims
    workdir: Path
    ep_ranks: Sequence[int] | None = None
    linkspeed_bps: int = 400_000_000_000
    topology: Path | None = None
    provider: ComputeProvider = field(default_factory=lambda: RooflineProvider(efficiency=0.7))
    gpu: GpuSpec = GPU_ENVELOPES["b100"]
    host_model: HostInitiationModel = field(default_factory=HostInitiationModel.ideal)
    #: first GOAL tag; each allreduce takes a disjoint 2(W-1)-tag block
    base_tag: int = 1000
    #: explicit GOAL rank count for topology padding; None keeps inferred sizing
    num_goal_ranks: int | None = None
    #: optional captured routing and explicit placement-epoch supply
    routed_moe_supply: RoutedMoeSupply | None = None
    #: regression-only negative control; production runs must retain False
    unsafe_disable_child_lifetime_binding: bool = False
    #: optional physical-placement authority; None is exact all-remote compatibility
    placement_manifest: PlacementManifest | None = None
    #: declared one-direction flat per-source NVLink rate; analytic, uncalibrated
    nvlink_bandwidth_bytes_per_second: int = (
        DEFAULT_NVLINK_BANDWIDTH_BYTES_PER_SECOND
    )
    #: optional independent dependency executor; a diagnostic, never a seam level
    dependency_cross_check: str | None = None
    #: absolute completion-time difference reported by the selected cross-check
    dependency_cross_check_tolerance_ps: int = 0
    #: optional explicit fidelity surface checked against the legacy spellings
    precision: PrecisionConfig | None = None
    #: calibrated base-latency model; None and ``legacy`` select the exact off path
    collective_latency_profile: str | CollectiveLatencyProfile | None = None
    #: named per-collective fixed-cost bracket; None keeps the unbracketed surface
    collective_fixed_cost_envelope: str | CollectiveFixedCostEnvelope | None = None
    #: which arm of that bracket to charge; ``off`` is the exact default path
    collective_fixed_cost_arm: str = "off"
    #: one-time channel-and-buffer registration model; None is the exact off path
    collective_registration: str | CollectiveRegistrationModel | None = None
    #: optional aggregate H200 completion authority; None is the exact off path
    collective_floor_calibration: CollectiveFloorCalibration | None = None
    #: source dtype named explicitly because a byte width cannot distinguish half
    collective_floor_dtype: str | None = None
    #: immutable registration model resolved during validation, never set by a caller
    resolved_collective_registration: CollectiveRegistrationModel | None = field(
        init=False,
        repr=False,
        compare=False,
        default=None,
    )
    #: immutable envelope resolved during validation, never set by a caller
    resolved_collective_fixed_cost_envelope: CollectiveFixedCostEnvelope | None = field(
        init=False,
        repr=False,
        compare=False,
        default=None,
    )
    #: immutable profile resolved during validation, never set by a caller
    resolved_collective_latency_profile: CollectiveLatencyProfile | None = field(
        init=False,
        repr=False,
        compare=False,
        default=None,
    )
    #: evidence class published for this selection, downgraded at the point of use
    resolved_collective_evidence_class: str | None = field(
        init=False,
        repr=False,
        compare=False,
        default=None,
    )
    #: the seams this configuration selects; set by validation, never by a caller
    selected_precision_levels: dict[str, object] = field(
        init=False,
        repr=False,
        compare=False,
        default_factory=dict,
    )

    def __post_init__(self) -> None:
        if self.profile not in RNIC_PROFILES:
            raise ValueError(f"profile must be one of {RNIC_PROFILES}")
        if not isinstance(self.host_model, HostInitiationModel):
            raise TypeError("host_model must be a HostInitiationModel")
        self.host_model.validate_device(self.gpu)
        if self.routed_moe_supply is not None and not isinstance(
            self.routed_moe_supply, RoutedMoeSupply
        ):
            raise TypeError("routed_moe_supply must be RoutedMoeSupply or None")
        if type(self.unsafe_disable_child_lifetime_binding) is not bool:
            raise TypeError("unsafe_disable_child_lifetime_binding must be a boolean")
        if self.placement_manifest is not None and not isinstance(
            self.placement_manifest,
            PlacementManifest,
        ):
            raise TypeError("placement_manifest must be PlacementManifest or None")
        if (
            type(self.nvlink_bandwidth_bytes_per_second) is not int
            or self.nvlink_bandwidth_bytes_per_second <= 0
        ):
            raise ValueError(
                "nvlink_bandwidth_bytes_per_second must be a positive integer"
            )
        self.resolved_collective_registration = resolve_collective_registration(
            self.collective_registration
        )
        if self.collective_floor_calibration is not None and not isinstance(
            self.collective_floor_calibration,
            CollectiveFloorCalibration,
        ):
            raise TypeError(
                "collective_floor_calibration must be a CollectiveFloorCalibration "
                "or None"
            )
        if self.collective_floor_calibration is None:
            if self.collective_floor_dtype is not None:
                raise ValueError(
                    "collective_floor_dtype requires collective_floor_calibration"
                )
        elif (
            not isinstance(self.collective_floor_dtype, str)
            or not self.collective_floor_dtype.strip()
        ):
            raise ValueError(
                "collective_floor_dtype must name the fitted source dtype"
            )
        self.resolved_collective_fixed_cost_envelope = (
            resolve_collective_fixed_cost_envelope(self.collective_fixed_cost_envelope)
        )
        if self.collective_fixed_cost_arm not in COLLECTIVE_FIXED_COST_ARMS:
            raise ValueError(
                "collective_fixed_cost_arm must be one of "
                f"{COLLECTIVE_FIXED_COST_ARMS}"
            )
        if (
            self.resolved_collective_fixed_cost_envelope is None
            and self.collective_fixed_cost_arm != "off"
        ):
            raise ValueError(
                "collective_fixed_cost_arm selects an arm of a named envelope; "
                "set collective_fixed_cost_envelope or keep the arm 'off'"
            )
        if (
            self.resolved_collective_fixed_cost_envelope is not None
            and self.collective_latency_profile is not None
        ):
            raise ValueError(
                "collective_fixed_cost_envelope and collective_latency_profile "
                "are two spellings of the same selection; use exactly one"
            )
        if self.resolved_collective_fixed_cost_envelope is not None:
            envelope = self.resolved_collective_fixed_cost_envelope
            self.resolved_collective_latency_profile = envelope.arm_profile(
                self.collective_fixed_cost_arm
            )
            self.resolved_collective_evidence_class = envelope.arm_evidence_class(
                self.collective_fixed_cost_arm
            )
        else:
            self.resolved_collective_latency_profile = (
                resolve_collective_latency_profile(self.collective_latency_profile)
            )
            self.resolved_collective_evidence_class = (
                None
                if self.resolved_collective_latency_profile is None
                else self.resolved_collective_latency_profile.evidence_class
            )
        if (
            self.collective_floor_calibration is not None
            and self.resolved_collective_latency_profile is not None
        ):
            raise ValueError(
                "aggregate collective-floor calibration already contains the "
                "complete measured fixed floor; disable the CollectiveLatencyProfile "
                "surcharge"
            )
        if (
            self.collective_floor_calibration is not None
            and self.resolved_collective_registration is not None
        ):
            raise ValueError(
                "aggregate collective-floor calibration already contains timed "
                "collective setup; disable the registration charge"
            )
        if self.collective_floor_calibration is not None and not self.host_model.is_ideal:
            raise ValueError(
                "aggregate collective-floor calibration already contains launch "
                "service; select the ideal host model"
            )
        if (
            self.collective_floor_calibration is not None
            and self.nvlink_bandwidth_bytes_per_second
            != DEFAULT_NVLINK_BANDWIDTH_BYTES_PER_SECOND
        ):
            raise ValueError(
                "aggregate collective-floor calibration owns local service; do not "
                "also override nvlink_bandwidth_bytes_per_second"
            )
        if (
            self.resolved_collective_latency_profile is not None
            and self.nvlink_bandwidth_bytes_per_second
            != DEFAULT_NVLINK_BANDWIDTH_BYTES_PER_SECOND
        ):
            raise ValueError(
                "a calibrated collective_latency_profile owns its bandwidth; "
                "do not also override nvlink_bandwidth_bytes_per_second"
            )
        if (
            self.resolved_collective_latency_profile is not None
            and self.profile != "rnic-nn-fluid"
        ):
            raise ValueError(
                "the calibrated collective_latency_profile currently requires "
                "profile='rnic-nn-fluid' so its propagation reference is explicit"
            )
        if (
            self.dependency_cross_check is not None
            and self.dependency_cross_check not in DEPENDENCY_CROSS_CHECK_MODES
        ):
            raise ValueError(
                "dependency_cross_check must be None or one of "
                f"{DEPENDENCY_CROSS_CHECK_MODES}"
            )
        if (
            self.resolved_collective_latency_profile is not None
            and self.dependency_cross_check is not None
        ):
            raise ValueError(
                "dependency_cross_check does not model the calibrated collective "
                "latency profile; disable one of the two selections"
            )
        if (
            type(self.dependency_cross_check_tolerance_ps) is not int
            or self.dependency_cross_check_tolerance_ps < 0
        ):
            raise ValueError(
                "dependency_cross_check_tolerance_ps must be a nonnegative integer"
            )
        self.selected_precision_levels = check_precision_selection(
            self.precision,
            compute=compute_level_for_provider(self.provider),
            dependency=DependencyLevel.SERIAL,
            locality=locality_level_for_placement(self.placement_manifest),
            network=network_level_for_profile(self.profile),
            selection_source="HtsimStepSinkConfig",
        )


@dataclass(frozen=True)
class StepNetworkOutcome:
    """Bookkeeping for one simulated step, kept for reporting."""

    step_index: int
    #: the adapter-equivalent compute-only whole-step estimate, ps
    compute_estimate_ps: int
    #: uniform calc cost handed to GOAL, or None for an unequal breakdown
    per_layer_calc_ns: int | None
    #: simulated makespan of the represented step, ps
    makespan_ps: int
    num_flows: int
    #: emitted GOAL calc units in layer order, ns; empty on legacy construction
    layer_calc_ns: tuple[int, ...] = ()
    #: sample count used for the fused estimate
    num_sampled: int = 0
    #: whether num_sampled came from an exact record field
    sample_count_exact: bool = False
    #: backend quiescence, vacuously true when no fabric backend run is needed
    quiescent: bool = False
    #: realized MoE traffic mode: ``none``, ``uniform`` or ``captured``
    routing_mode: str = "uniform"
    #: selected expert placement epoch, present only for captured traffic
    placement_epoch: int | None = None
    #: host profile selected by the serial timing authority
    host_profile: str = "ideal"
    #: measured launch class, or ``none`` on the identity path
    host_launch_class: str = "none"
    #: launches represented by the calibrated fixed step cost
    host_launch_count: int = 0
    #: provider service before host composition
    provider_compute_ps: int = 0
    #: point launch-throughput floor before overlap composition
    host_launch_floor_ps: int = 0
    #: empirical launch-floor endpoints from the accepted capture series
    host_launch_floor_lower_ps: int = 0
    host_launch_floor_upper_ps: int = 0
    #: represented-service endpoints after provider composition
    host_empirical_lower_ps: int = 0
    host_empirical_upper_ps: int = 0
    #: point launch demand exposed above provider service
    exposed_host_ps: int = 0
    #: which term set the represented duration: the provider's own bound name
    #: when it survived, or ``host-initiation`` when the launch floor replaced
    #: it. Empty on legacy construction that predates the field.
    represented_bound: str = ""

    def network_share_for(self, num_layers: int) -> float:
        """One minus represented calc time over makespan."""
        if not self.layer_calc_ns:
            if self.per_layer_calc_ns is None:
                raise ValueError("outcome has neither uniform nor ordered layer calcs")
            calc_ps = num_layers * max(self.per_layer_calc_ns, 1) * 1000
            return 1.0 - calc_ps / self.makespan_ps
        if len(self.layer_calc_ns) != num_layers:
            raise ValueError(
                f"outcome has {len(self.layer_calc_ns)} layer calcs, expected {num_layers}"
            )
        calc_ps = sum(max(calc_ns, 1) for calc_ns in self.layer_calc_ns) * 1000
        return 1.0 - calc_ps / self.makespan_ps


#: resource that owns an executed artifact's analytic local service term.
#: ``gpu-compute`` is kernel service on the artifact's own device;
#: ``nvlink`` is the analytic per-endpoint NVLink serializer of a collective
#: phase. The label is required evidence rather than a convenience: the two
#: services share the ``local_service_ps`` field and cannot be told apart from
#: their value.
GPU_COMPUTE_MEDIUM = "gpu-compute"
NVLINK_MEDIUM = "nvlink"
LOCAL_SERVICE_MEDIA = (GPU_COMPUTE_MEDIUM, NVLINK_MEDIUM)


@dataclass(frozen=True)
class StepLocalityOutcome:
    """Locality and graph-projection accounting for one simulated step."""

    step_index: int
    authority: str
    compatibility_fast_path: bool
    total_directed_bytes: int
    fabric_directed_bytes: int
    nvlink_directed_bytes: int
    fabric_segments: int
    nvlink_segments: int
    phase_count: int
    backend_runs: int
    compute_service_ps: int
    nvlink_service_ps: int
    nvlink_bandwidth_bytes_per_second: int
    #: fabric service by executed artifact; zero for analytic-only artifacts
    fabric_phase_service_ps: tuple[int, ...] = ()
    #: max of local and fabric service for each executed artifact
    composed_phase_service_ps: tuple[int, ...] = ()
    #: analytic local service by executed artifact, owned by the medium below
    local_phase_service_ps: tuple[int, ...] = ()
    #: semantic collective fixed cost by executed artifact, owned by no resource
    base_phase_latency_ps: tuple[int, ...] = ()
    #: fitted aggregate completion floor by artifact, empty on the off path
    collective_floor_phase_ps: tuple[int, ...] = ()
    #: one-time registration charged at each artifact; empty on the off path
    registration_phase_cost_ps: tuple[int, ...] = ()
    #: ``LOCAL_SERVICE_MEDIA`` owner of each artifact's local service term
    local_phase_medium: tuple[str, ...] = ()
    #: semantic ordering owner for every active path
    ordering_authority: str = "execution-graph"
    #: graph identity whose checked projection was executed
    graph_execution_id: str | None = None
    #: complete canonical graph-edge inventory size
    effective_dependency_edge_count: int = 0
    #: number of GOAL or analytic artifacts executed in order
    artifact_count: int = 0
    #: exact graph barrier stages before any collective subphase expansion
    graph_artifact_count: int = 0
    #: graph edges represented by ordered artifact boundaries
    boundary_edge_count: int = 0
    #: other graph edges carried across a boundary and conservatively serialized
    serialized_edge_count: int = 0
    #: semantic operation identities attributed to each executed artifact
    artifact_operation_ids: tuple[tuple[str, ...], ...] = ()


@dataclass(frozen=True)
class CollectiveArtifactTiming:
    """Read-only calibrated decomposition of one executed graph artifact."""

    artifact_id: str
    operation_ids: tuple[str, ...]
    collective_operation_id: str | None
    participant_count: int | None
    critical_endpoint_bytes: int | None
    local_service_ps: int
    fabric_transport_ps: int
    collective_base_latency_ps: int
    composed_service_ps: int
    #: one-time channel-and-buffer registration serialized ahead of this artifact
    registration_cost_ps: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "operation_ids", tuple(self.operation_ids))
        if not isinstance(self.artifact_id, str) or not self.artifact_id.strip():
            raise ValueError("artifact_id must be a nonblank string")
        if any(
            not isinstance(operation_id, str) or not operation_id.strip()
            for operation_id in self.operation_ids
        ):
            raise ValueError("operation_ids must contain nonblank strings")
        for name in (
            "local_service_ps",
            "fabric_transport_ps",
            "collective_base_latency_ps",
            "composed_service_ps",
            "registration_cost_ps",
        ):
            _require_nonnegative_timing(name, getattr(self, name))
        if self.collective_operation_id is None:
            if self.participant_count is not None or self.critical_endpoint_bytes is not None:
                raise ValueError(
                    "non-collective artifacts cannot carry participant or endpoint fields"
                )
            if self.collective_base_latency_ps:
                raise ValueError("non-collective artifacts cannot carry a base latency")
            if self.registration_cost_ps:
                raise ValueError(
                    "non-collective artifacts cannot carry a registration cost"
                )
        else:
            if (
                not isinstance(self.collective_operation_id, str)
                or not self.collective_operation_id.strip()
            ):
                raise ValueError("collective_operation_id must be a nonblank string")
            if self.collective_operation_id not in self.operation_ids:
                raise ValueError(
                    "collective_operation_id must belong to the artifact operations"
                )
            if type(self.participant_count) is not int or self.participant_count < 2:
                raise ValueError("participant_count must be an integer of at least two")
            _require_nonnegative_timing(
                "critical_endpoint_bytes",
                self.critical_endpoint_bytes,
            )
        expected = (
            self.registration_cost_ps
            + self.collective_base_latency_ps
            + max(self.local_service_ps, self.fabric_transport_ps)
        )
        if self.composed_service_ps != expected:
            raise ValueError("composed service disagrees with its calibrated terms")


@dataclass(frozen=True)
class StepCollectiveTimingOutcome:
    """Calibrated collective terms for one scheduler-visible step."""

    step_index: int
    profile_id: str
    bandwidth_bytes_per_second: int
    participant_latency_ps: tuple[tuple[int, int], ...]
    propagation_reference_ps: int
    artifacts: tuple[CollectiveArtifactTiming, ...]
    #: named bracket the profile was selected from, or None for a bare profile
    envelope_id: str | None = None
    #: which arm of that bracket is charged; set together with ``envelope_id``
    arm: str | None = None
    #: declared evidence class of the charged profile
    evidence_class: str = "unattributed"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "participant_latency_ps",
            tuple(tuple(entry) for entry in self.participant_latency_ps),
        )
        object.__setattr__(self, "artifacts", tuple(self.artifacts))
        _require_nonnegative_timing("step_index", self.step_index)
        if not isinstance(self.profile_id, str) or not self.profile_id.strip():
            raise ValueError("profile_id must be a nonblank string")
        if (self.envelope_id is None) != (self.arm is None):
            raise ValueError("envelope_id and arm are reported together or not at all")
        if self.envelope_id is not None:
            if not isinstance(self.envelope_id, str) or not self.envelope_id.strip():
                raise ValueError("envelope_id must be a nonblank string")
            if self.arm not in COLLECTIVE_FIXED_COST_ARMS:
                raise ValueError(f"arm must be one of {COLLECTIVE_FIXED_COST_ARMS}")
        if not isinstance(self.evidence_class, str) or not self.evidence_class.strip():
            raise ValueError("evidence_class must be a nonblank string")
        if (
            type(self.bandwidth_bytes_per_second) is not int
            or self.bandwidth_bytes_per_second <= 0
        ):
            raise ValueError("bandwidth_bytes_per_second must be a positive integer")
        _require_nonnegative_timing(
            "propagation_reference_ps",
            self.propagation_reference_ps,
        )
        latency_by_width: dict[int, int] = {}
        for index, entry in enumerate(self.participant_latency_ps):
            if len(entry) != 2:
                raise ValueError(
                    f"participant_latency_ps[{index}] must be a two-item tuple"
                )
            width, latency_ps = entry
            if type(width) is not int or width < 2:
                raise ValueError("participant latency widths must be at least two")
            _require_nonnegative_timing("participant latency", latency_ps)
            if width in latency_by_width:
                raise ValueError("participant latency widths must be unique")
            latency_by_width[width] = latency_ps
        if tuple(latency_by_width) != tuple(sorted(latency_by_width)):
            raise ValueError("participant latency widths must be increasing")
        if any(
            not isinstance(artifact, CollectiveArtifactTiming)
            for artifact in self.artifacts
        ):
            raise TypeError("artifacts must contain CollectiveArtifactTiming values")
        grouped: dict[str, list[CollectiveArtifactTiming]] = {}
        for artifact in self.artifacts:
            if artifact.collective_operation_id is not None:
                grouped.setdefault(artifact.collective_operation_id, []).append(
                    artifact
                )
        for operation_id, artifacts in grouped.items():
            widths = {artifact.participant_count for artifact in artifacts}
            endpoint_values = {
                artifact.critical_endpoint_bytes for artifact in artifacts
            }
            if len(widths) != 1 or len(endpoint_values) != 1:
                raise ValueError(
                    f"collective {operation_id!r} has inconsistent timing identity"
                )
            width = next(iter(widths))
            endpoint_bytes = next(iter(endpoint_values))
            if width not in latency_by_width:
                raise ValueError(
                    f"collective {operation_id!r} has an unsupported participant count"
                )
            expected_base = latency_by_width[width]
            observed_bases = [
                artifact.collective_base_latency_ps for artifact in artifacts
            ]
            if endpoint_bytes == 0:
                if any(observed_bases):
                    raise ValueError("empty collective received a base latency")
            elif sum(observed_bases) != expected_base or sum(
                base_latency_ps > 0 for base_latency_ps in observed_bases
            ) > 1:
                raise ValueError(
                    f"collective {operation_id!r} did not receive exactly one base latency"
                )


@dataclass(frozen=True)
class CollectiveFloorArtifactTiming:
    """One phase whose local service comes from an aggregate fitted curve."""

    artifact_id: str
    collective_operation_id: str
    semantic_collective: str
    local_service_ps: int
    aggregate_floor_ps: int
    fabric_transport_ps: int
    collective_base_latency_ps: int
    registration_cost_ps: int
    composed_service_ps: int
    estimate: CollectiveFloorEstimate

    def __post_init__(self) -> None:
        for name in (
            "artifact_id",
            "collective_operation_id",
            "semantic_collective",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a nonblank string")
        for name in (
            "local_service_ps",
            "aggregate_floor_ps",
            "fabric_transport_ps",
            "collective_base_latency_ps",
            "registration_cost_ps",
            "composed_service_ps",
        ):
            _require_nonnegative_timing(name, getattr(self, name))
        if not isinstance(self.estimate, CollectiveFloorEstimate):
            raise TypeError("estimate must be a CollectiveFloorEstimate")
        if self.semantic_collective != self.estimate.requested_operation:
            raise ValueError("semantic collective disagrees with its fitted estimate")
        if self.collective_base_latency_ps:
            raise ValueError("aggregate floor cannot carry a semantic base surcharge")
        if self.registration_cost_ps:
            raise ValueError("aggregate floor cannot carry a registration surcharge")
        if self.composed_service_ps != self.aggregate_floor_ps + max(
            self.local_service_ps,
            self.fabric_transport_ps,
        ):
            raise ValueError("composed service disagrees with the aggregate authority")


@dataclass(frozen=True)
class StepCollectiveFloorTimingOutcome:
    """Read-only projection of the aggregate authority used for one step."""

    step_index: int
    calibration_id: str
    dtype: str
    input_surface: tuple[str, ...]
    host_launch_floor_ps: int
    artifacts: tuple[CollectiveFloorArtifactTiming, ...]

    def __post_init__(self) -> None:
        _require_nonnegative_timing("step_index", self.step_index)
        for name in ("calibration_id", "dtype"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a nonblank string")
        object.__setattr__(self, "input_surface", tuple(self.input_surface))
        object.__setattr__(self, "artifacts", tuple(self.artifacts))
        _require_nonnegative_timing(
            "host_launch_floor_ps", self.host_launch_floor_ps
        )
        if self.host_launch_floor_ps:
            raise ValueError("aggregate floor cannot carry a second host launch floor")
        if not self.artifacts:
            raise ValueError("a collective-floor outcome needs fitted artifacts")
        if any(
            not isinstance(artifact, CollectiveFloorArtifactTiming)
            for artifact in self.artifacts
        ):
            raise TypeError(
                "artifacts must contain CollectiveFloorArtifactTiming values"
            )
        if any(
            artifact.estimate.calibration_id != self.calibration_id
            for artifact in self.artifacts
        ):
            raise ValueError("artifact estimates belong to another calibration")
        grouped: dict[
            tuple[str, str], list[CollectiveFloorArtifactTiming]
        ] = {}
        for artifact in self.artifacts:
            grouped.setdefault(
                (
                    artifact.collective_operation_id,
                    artifact.semantic_collective,
                ),
                [],
            ).append(artifact)
        for identity, phases in grouped.items():
            estimates = {phase.estimate for phase in phases}
            if len(estimates) != 1:
                raise ValueError(
                    f"collective half {identity!r} carries inconsistent estimates"
                )
            estimate = next(iter(estimates))
            if sum(phase.local_service_ps for phase in phases) != (
                estimate.serialization_ps
            ):
                raise ValueError(
                    f"collective half {identity!r} does not conserve serialization"
                )
            if sum(phase.aggregate_floor_ps for phase in phases) != (
                estimate.floor_charge_ps
            ):
                raise ValueError(
                    f"collective half {identity!r} does not charge its floor once"
                )


@dataclass(frozen=True)
class StepCollectiveRegistrationOutcome:
    """What one scheduler-visible step paid to register channels and buffers.

    This is a read-only projection of the sink's single registration ledger. It
    is published only when a registration model is selected, so the off path
    leaves the list empty rather than filling it with zeros.
    """

    step_index: int
    model_id: str
    cost_id: str
    evidence_class: str
    registration_cost_ps: int
    channels_per_communicator: int
    charged_ps: int
    events: tuple[CollectiveRegistrationEvent, ...]
    #: per executed artifact, in artifact order
    artifact_registration_ps: tuple[int, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "events", tuple(self.events))
        object.__setattr__(
            self,
            "artifact_registration_ps",
            tuple(self.artifact_registration_ps),
        )
        _require_nonnegative_timing("step_index", self.step_index)
        for name in ("model_id", "cost_id", "evidence_class"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a nonblank string")
        _require_nonnegative_timing("registration_cost_ps", self.registration_cost_ps)
        if (
            type(self.channels_per_communicator) is not int
            or self.channels_per_communicator < 1
        ):
            raise ValueError("channels_per_communicator must be a positive integer")
        _require_nonnegative_timing("charged_ps", self.charged_ps)
        for index, value in enumerate(self.artifact_registration_ps):
            _require_nonnegative_timing(f"artifact_registration_ps[{index}]", value)
        if sum(self.artifact_registration_ps) != self.charged_ps:
            raise ValueError(
                "per-artifact registration does not conserve the ledger charge"
            )
        if sum(event.cost_ps for event in self.events) != self.charged_ps:
            raise ValueError(
                "registration events do not conserve the ledger charge"
            )
        if any(
            event.cost_ps != self.registration_cost_ps for event in self.events
        ):
            raise ValueError("every registration event charges the same declared cost")


@dataclass(frozen=True)
class _PlannedExecutionArtifact:
    """One checked graph-projection artifact executed at an ordered boundary."""

    artifact_id: str
    operation_ids: tuple[str, ...]
    goal_path: Path | None
    completion_csv: Path | None
    local_service_ps: int
    collective_operation_id: str | None = None
    participant_count: int | None = None
    critical_endpoint_bytes: int | None = None
    collective_base_latency_ps: int = 0
    aggregate_collective_floor_ps: int = 0
    collective_floor_semantic: str | None = None
    collective_floor_estimate: CollectiveFloorEstimate | None = None
    registration_cost_ps: int = 0


@dataclass(frozen=True)
class _CollectiveFloorPhaseTerm:
    """One phase-local projection of an aggregate fitted completion."""

    semantic_collective: str
    local_service_ps: int
    aggregate_floor_ps: int
    estimate: CollectiveFloorEstimate


def _ring_collective_floor_terms(
    *,
    work: CollectiveWork,
    phases: tuple[ClassifiedCommunicationPhase, ...],
    calibration: CollectiveFloorCalibration,
    dtype: str,
) -> dict[str, _CollectiveFloorPhaseTerm]:
    """Project two aggregate ring halves over their ordered local phases."""

    if (work.collective, work.algorithm_hint) != ("all-reduce", "ring"):
        raise ValueError(
            "aggregate collective-floor calibration supports ring all-reduce only"
        )
    phase_count = len(work.ranks) - 1
    if len(phases) != 2 * phase_count:
        raise ValueError("ring locality phases disagree with the two aggregate halves")
    if any(phase.fabric_segments or not phase.nvlink_segments for phase in phases):
        raise ValueError(
            "aggregate H200 collective-floor calibration requires a fully "
            "intra-node collective; mixed local and fabric integration remains open"
        )
    terms: dict[str, _CollectiveFloorPhaseTerm] = {}
    for half_index, semantic_collective in enumerate(
        ("reduce_scatter", "all_gather")
    ):
        estimate = calibration.estimate(
            dtype=dtype,
            operation=semantic_collective,
            ranks=len(work.ranks),
            message_bytes=work.payload_bytes,
        )
        distributed = distribute_collective_serialization_ps(
            estimate.serialization_ps,
            phase_count,
        )
        half_phases = phases[
            half_index * phase_count : (half_index + 1) * phase_count
        ]
        for phase_index, (phase, local_service_ps) in enumerate(
            zip(half_phases, distributed, strict=True)
        ):
            terms[phase.phase.phase_id] = _CollectiveFloorPhaseTerm(
                semantic_collective=semantic_collective,
                local_service_ps=local_service_ps,
                aggregate_floor_ps=(
                    estimate.floor_charge_ps if phase_index == 0 else 0
                ),
                estimate=estimate,
            )
    return terms


@dataclass(frozen=True)
class _PlannedDependencyCrossCheck:
    """One independently rendered schedule selected for diagnostic comparison."""

    comparison: DependencyCrossCheckPlan
    goal_path: Path
    completion_csv: Path
    goal_bytes: int
    goal_sha256: str
    tolerance_ps: int


@dataclass(frozen=True)
class _PlannedStep:
    """Immutable handoff from serial record lowering to one backend worker."""

    step_index: int
    virtual_time_ps: int
    artifacts: tuple[_PlannedExecutionArtifact, ...]
    locality: StepLocalityPlan
    compatibility_fast_path: bool
    compute_in_artifacts: bool
    compute_service_ps: int
    compute_estimate_ps: int
    host_profile: str
    host_launch_class: str
    host_launch_count: int
    provider_compute_ps: int
    host_launch_floor_ps: int
    host_launch_floor_lower_ps: int
    host_launch_floor_upper_ps: int
    host_empirical_lower_ps: int
    host_empirical_upper_ps: int
    exposed_host_ps: int
    represented_bound: str
    num_sampled: int
    sample_count_exact: bool
    per_layer_calc_ns: int | None
    layer_calc_ns: tuple[int, ...]
    routing_mode: str
    placement_epoch: int | None
    profile: str
    linkspeed_bps: int
    topology: Path | None
    unsafe_disable_child_lifetime_binding: bool
    effective_dependency_edge_count: int
    boundary_edge_count: int
    serialized_edge_count: int
    graph_artifact_count: int
    dependency_cross_check: _PlannedDependencyCrossCheck | None
    collective_latency_profile: CollectiveLatencyProfile | None
    collective_fixed_cost_envelope_id: str | None
    collective_fixed_cost_arm: str | None
    collective_evidence_class: str | None
    collective_floor_calibration: CollectiveFloorCalibration | None
    collective_registration: CollectiveRegistrationModel | None
    registration_events: tuple[CollectiveRegistrationEvent, ...]


@dataclass(frozen=True)
class _SimulatedStep:
    """One unpublished result, safe to retain until its record is consumed."""

    result: StepResult | None
    outcome: StepNetworkOutcome | None
    locality_outcome: StepLocalityOutcome | None
    collective_timing_outcome: StepCollectiveTimingOutcome | None
    collective_floor_timing_outcome: StepCollectiveFloorTimingOutcome | None
    dependency_cross_check_report: DependencyCrossCheckReport | None
    collective_registration_outcome: StepCollectiveRegistrationOutcome | None = None


@dataclass(frozen=True)
class _PreparedStep:
    """A copied input record joined to its unpublished simulation result."""

    record: StepRecord
    simulation: _SimulatedStep


class HtsimStepSink:
    """Step sink that simulates each step's TP traffic on ``htsim_rnic``."""

    def __init__(self, config: HtsimStepSinkConfig) -> None:
        self.config = config
        placement = copy.deepcopy(config.placement_manifest)
        self._rank_mapper = RankMapper(placement) if placement is not None else None
        if self._rank_mapper is not None:
            groups = (("tp_ranks", config.tp_ranks),)
            if config.ep_ranks is not None:
                groups += (("ep_ranks", config.ep_ranks),)
            for name, ranks_value in groups:
                ranks = tuple(ranks_value)
                if len(ranks) != len(set(ranks)):
                    raise ValueError(f"{name} must contain distinct ranks")
                for rank in ranks:
                    self._rank_mapper.goal_rank(rank)
        self.config.workdir.mkdir(parents=True, exist_ok=True)
        #: the one registration authority for this deployment; disabled by default
        self._registration_ledger = CollectiveRegistrationLedger(
            model=config.resolved_collective_registration
        )
        #: one entry per simulated (non-None) step, in call order
        self.outcomes: list[StepNetworkOutcome] = []
        #: locality projection for the same simulated steps, in call order
        self.locality_outcomes: list[StepLocalityOutcome] = []
        #: calibrated decompositions; deliberately empty on the legacy/off path
        self.collective_timing_outcomes: list[StepCollectiveTimingOutcome] = []
        #: aggregate floor projections; deliberately empty on the exact off path
        self.collective_floor_timing_outcomes: list[
            StepCollectiveFloorTimingOutcome
        ] = []
        #: explicitly selected independent dependency comparisons, in call order
        self.dependency_cross_check_reports: list[DependencyCrossCheckReport] = []
        #: registration ledger projections; deliberately empty on the off path
        self.collective_registration_outcomes: list[
            StepCollectiveRegistrationOutcome
        ] = []

    @property
    def host_model(self) -> HostInitiationModel:
        """The one host model applied by this sink's serial lowerer."""

        return self.config.host_model

    @property
    def collective_registration_ledger(self) -> CollectiveRegistrationLedger:
        """The one registration authority this sink charges against."""

        return self._registration_ledger

    def rebuild_collective_communicators(self) -> None:
        """Model a destroy-and-init cycle of every communicator seen so far.

        This is the live form of the third re-registration event. The next
        collective on each rebuilt communicator re-registers every channel and
        every buffer, and pays for it. A disabled ledger has nothing to rebuild
        and this is a no-op.
        """

        self._registration_ledger.rebuild_all()

    @staticmethod
    def _num_sampled(record: StepRecord) -> int:
        if record.num_sampled is not None:
            return record.num_sampled
        return len(record.scheduled)

    def _compute_estimate(
        self, record: StepRecord
    ) -> tuple[int, tuple[int, ...] | None]:
        """Return the estimate represented by the serial graph lowerer."""

        timing = self._serial_lowerer().timing(record)
        if timing.per_layer_calc_ns is not None:
            return timing.compute_estimate_ps, None
        return (
            timing.compute_estimate_ps,
            tuple(duration_ns * 1000 for duration_ns in timing.layer_calc_ns),
        )

    def compute_estimate_ps(self, record: StepRecord) -> int:
        """The compute-only whole-step estimate represented by the sink."""
        estimate_ps, _ = self._compute_estimate(record)
        return estimate_ps

    def _serial_lowerer(self) -> SerialStepLowerer:
        cfg = self.config
        return SerialStepLowerer(
            SerialStepLowererConfig(
                dims=cfg.dims,
                tp_ranks=cfg.tp_ranks,
                ep_ranks=cfg.ep_ranks,
                provider=cfg.provider,
                gpu=cfg.gpu,
                host_model=cfg.host_model,
                routed_moe_supply=cfg.routed_moe_supply,
            )
        )

    def _plan_step(self, record: StepRecord) -> _PlannedStep | None:
        """Lower and render one record without invoking either native tool."""

        cfg = self.config
        graph, timing = self._serial_lowerer().lower_with_timing(record)
        collectives = tuple(
            operation
            for operation in graph.operations
            if isinstance(operation.work, CollectiveWork)
        )
        if not collectives:
            return None
        collective_profile = cfg.resolved_collective_latency_profile
        collective_floor = cfg.collective_floor_calibration
        effective_nvlink_bandwidth = (
            collective_profile.bandwidth_bytes_per_second
            if collective_profile is not None
            else cfg.nvlink_bandwidth_bytes_per_second
        )
        collective_timing_parameters: dict[str, tuple[int, int, int]] = {}
        if collective_profile is not None:
            for operation in collectives:
                endpoint_bytes = critical_collective_endpoint_bytes(operation.work)
                participant_count = len(operation.work.ranks)
                calibrated_base_latency_ps = collective_profile.base_latency_ps(
                    participant_count
                )
                if endpoint_bytes:
                    collective_profile.validate_endpoint_bytes(
                        participant_count,
                        endpoint_bytes,
                    )
                    base_latency_ps = calibrated_base_latency_ps
                else:
                    base_latency_ps = 0
                collective_timing_parameters[operation.operation_id] = (
                    participant_count,
                    endpoint_bytes,
                    base_latency_ps,
                )
        moe_operations = tuple(
            operation
            for operation in collectives
            if operation.work.collective == "all-to-allv"
        )
        if not moe_operations:
            routing_mode = "none"
        elif cfg.routed_moe_supply is not None:
            routing_mode = "captured"
        else:
            routing_mode = "uniform"
        locality = plan_execution_graph_locality(
            graph,
            rank_mapper=self._rank_mapper,
            nvlink_bandwidth_bytes_per_second=(
                effective_nvlink_bandwidth
            ),
            base_tag=cfg.base_tag,
        )
        validate_execution_graph_locality_projection(
            graph,
            locality,
            rank_mapper=self._rank_mapper,
            nvlink_bandwidth_bytes_per_second=(
                effective_nvlink_bandwidth
            ),
            base_tag=cfg.base_tag,
        )
        projection = project_execution_graph_goal(
            graph,
            num_goal_ranks=cfg.num_goal_ranks,
            base_tag=cfg.base_tag,
        )
        name = f"step-{record.step_index:06d}"
        compatibility_fast_path = self._rank_mapper is None or (
            locality.nvlink_bytes == 0 and self._rank_mapper.mode == "gpu-rank"
        )
        cross_check_trace = None
        cross_check_comparison = None
        if cfg.dependency_cross_check is not None:
            if not compatibility_fast_path:
                raise ValueError(
                    "the atlahs-goal dependency cross-check currently requires "
                    "the all-remote compatibility level; local NVLink segments "
                    "need TRAF-16 participant-frontier comparison"
                )
            cross_check_trace = render_step_goal(
                record,
                cfg.dims,
                cfg.tp_ranks,
                timing.layer_calc_ns,
                ep_ranks=cfg.ep_ranks,
                routed_supply=cfg.routed_moe_supply,
                num_goal_ranks=cfg.num_goal_ranks,
                base_tag=cfg.base_tag,
            )
            cross_check_comparison = plan_dependency_cross_check(
                graph=graph,
                direct_trace=cross_check_trace,
                boundary_edges=tuple(
                    boundary.edge for boundary in projection.boundaries
                ),
                expected_messages=tuple(
                    message
                    for artifact in projection.artifacts
                    for message in artifact.trace.messages
                ),
            )
        #: Charge the one registration ledger in graph order, once per semantic
        #: collective, after every projection has validated and before any
        #: artifact is built. The amount is spent on that collective's first
        #: executed artifact, so a collective split into locality phases pays
        #: exactly once.
        registration_before = len(self._registration_ledger.events)
        registration_by_operation: dict[str, int] = {}
        for operation in collectives:
            registration_by_operation[operation.operation_id] = (
                self._registration_ledger.charge_collective(
                    operation.work,
                    operation.operation_id,
                    step_index=record.step_index,
                )
            )
        registration_events = self._registration_ledger.events[registration_before:]
        operation_by_id = {
            operation.operation_id: operation for operation in graph.operations
        }
        phases_by_operation: dict[str, list] = {}
        for phase in locality.phases:
            operation_id = phase.phase.operation_id
            if operation_id is None:
                raise AssertionError("graph locality phase has no operation identity")
            phases_by_operation.setdefault(operation_id, []).append(phase)
        planned_artifacts = []
        for artifact_index, artifact in enumerate(projection.artifacts):
            operations = tuple(
                operation_by_id[operation_id]
                for operation_id in artifact.operation_ids
            )
            if operations and all(
                not isinstance(operation.work, CollectiveWork)
                for operation in operations
            ):
                durations = tuple(
                    operation.work.nominal_duration_ps
                    for operation in operations
                )
                if any(duration is None for duration in durations):
                    raise ValueError("serial compute artifact has no nominal duration")
                planned_artifacts.append(
                    _PlannedExecutionArtifact(
                        artifact_id=f"graph-artifact-{artifact_index}:compute",
                        operation_ids=artifact.operation_ids,
                        goal_path=None,
                        completion_csv=None,
                        local_service_ps=max(
                            (max(duration, 1_000) for duration in durations),
                            default=0,
                        ),
                    )
                )
                continue
            collective_indexes = tuple(
                index
                for index, operation in enumerate(operations)
                if isinstance(operation.work, CollectiveWork)
            )
            if len(collective_indexes) != 1:
                raise ValueError(
                    "serial graph artifact must contain at most one collective"
                )
            collective_index = collective_indexes[0]
            collective = operations[collective_index]
            operation_id = collective.operation_id
            timing_parameters = collective_timing_parameters.get(operation_id)
            classified_phases = tuple(phases_by_operation.pop(operation_id, ()))
            if not classified_phases:
                work = collective.work
                if (
                    work.collective == "all-to-allv"
                    and work.algorithm_hint == "pairwise"
                    and work.payload_bytes == 0
                    and not work.pair_payload_bytes
                    and operations == (collective,)
                ):
                    planned_artifacts.append(
                        _PlannedExecutionArtifact(
                            artifact_id=(
                                f"graph-artifact-{artifact_index}:empty-collective"
                            ),
                            operation_ids=artifact.operation_ids,
                            goal_path=None,
                            completion_csv=None,
                            local_service_ps=0,
                            collective_operation_id=operation_id,
                            participant_count=(
                                timing_parameters[0]
                                if timing_parameters is not None
                                else None
                            ),
                            critical_endpoint_bytes=(
                                timing_parameters[1]
                                if timing_parameters is not None
                                else None
                            ),
                            registration_cost_ps=registration_by_operation.get(
                                operation_id, 0
                            ),
                        )
                    )
                    continue
                raise ValueError("collective artifact has no locality service phase")
            collective_floor_terms = (
                _ring_collective_floor_terms(
                    work=collective.work,
                    phases=classified_phases,
                    calibration=collective_floor,
                    dtype=cfg.collective_floor_dtype or "",
                )
                if collective_floor is not None
                else {}
            )
            if compatibility_fast_path:
                stem = f"{name}.artifact-{artifact_index:04d}"
                planned_artifacts.append(
                    _PlannedExecutionArtifact(
                        artifact_id=f"graph-artifact-{artifact_index}:collective",
                        operation_ids=artifact.operation_ids,
                        goal_path=artifact.trace.write(
                            cfg.workdir / f"{stem}.goal"
                        ),
                        completion_csv=(
                            cfg.workdir / f"{stem}.{cfg.profile}.csv"
                        ),
                        local_service_ps=0,
                        collective_operation_id=operation_id,
                        participant_count=(
                            timing_parameters[0]
                            if timing_parameters is not None
                            else None
                        ),
                        critical_endpoint_bytes=(
                            timing_parameters[1]
                            if timing_parameters is not None
                            else None
                        ),
                        collective_base_latency_ps=(
                            timing_parameters[2]
                            if timing_parameters is not None
                            else 0
                        ),
                        registration_cost_ps=registration_by_operation.get(
                            operation_id, 0
                        ),
                    )
                )
                continue
            before_compute = operations[:collective_index]
            if before_compute:
                durations = tuple(
                    operation.work.nominal_duration_ps
                    for operation in before_compute
                )
                if any(duration is None for duration in durations):
                    raise ValueError("serial compute artifact has no nominal duration")
                planned_artifacts.append(
                    _PlannedExecutionArtifact(
                        artifact_id=f"graph-artifact-{artifact_index}:compute-before",
                        operation_ids=tuple(
                            operation.operation_id for operation in before_compute
                        ),
                        goal_path=None,
                        completion_csv=None,
                        local_service_ps=max(
                            (max(duration, 1_000) for duration in durations),
                            default=0,
                        ),
                    )
                )
            for phase_index, phase in enumerate(classified_phases):
                collective_floor_term = collective_floor_terms.get(
                    phase.phase.phase_id
                )
                trace = (
                    render_fabric_phase_goal(
                        phase,
                        rank_mapper=self._rank_mapper,
                        num_goal_ranks=cfg.num_goal_ranks,
                    )
                    if phase.fabric_segments
                    else None
                )
                stem = (
                    f"{name}.artifact-{artifact_index:04d}."
                    f"phase-{phase_index:04d}"
                )
                planned_artifacts.append(
                    _PlannedExecutionArtifact(
                        artifact_id=phase.phase.phase_id,
                        operation_ids=artifact.operation_ids,
                        goal_path=(
                            trace.write(cfg.workdir / f"{stem}.goal")
                            if trace is not None
                            else None
                        ),
                        completion_csv=(
                            cfg.workdir / f"{stem}.{cfg.profile}.csv"
                            if trace is not None
                            else None
                        ),
                        local_service_ps=(
                            phase.nvlink_service_ps
                            if collective_floor_term is None
                            else collective_floor_term.local_service_ps
                        ),
                        collective_operation_id=operation_id,
                        participant_count=(
                            timing_parameters[0]
                            if timing_parameters is not None
                            else None
                        ),
                        critical_endpoint_bytes=(
                            timing_parameters[1]
                            if timing_parameters is not None
                            else None
                        ),
                        collective_base_latency_ps=(
                            timing_parameters[2]
                            if timing_parameters is not None and phase_index == 0
                            else 0
                        ),
                        aggregate_collective_floor_ps=(
                            0
                            if collective_floor_term is None
                            else collective_floor_term.aggregate_floor_ps
                        ),
                        collective_floor_semantic=(
                            None
                            if collective_floor_term is None
                            else collective_floor_term.semantic_collective
                        ),
                        collective_floor_estimate=(
                            None
                            if collective_floor_term is None
                            else collective_floor_term.estimate
                        ),
                        registration_cost_ps=(
                            registration_by_operation.get(operation_id, 0)
                            if phase_index == 0
                            else 0
                        ),
                    )
                )
            after_compute = operations[collective_index + 1 :]
            if after_compute:
                durations = tuple(
                    operation.work.nominal_duration_ps
                    for operation in after_compute
                )
                if any(duration is None for duration in durations):
                    raise ValueError("serial compute artifact has no nominal duration")
                planned_artifacts.append(
                    _PlannedExecutionArtifact(
                        artifact_id=f"graph-artifact-{artifact_index}:compute-after",
                        operation_ids=tuple(
                            operation.operation_id for operation in after_compute
                        ),
                        goal_path=None,
                        completion_csv=None,
                        local_service_ps=max(
                            (max(duration, 1_000) for duration in durations),
                            default=0,
                        ),
                    )
                )
        if phases_by_operation:
            raise ValueError(
                "locality projection contains collectives absent from GOAL projection"
            )
        planned_cross_check = None
        if cross_check_trace is not None:
            if cross_check_comparison is None:
                raise AssertionError("cross-check trace has no comparison plan")
            cross_check_dir = cfg.workdir / "cross-check"
            cross_check_dir.mkdir(parents=True, exist_ok=True)
            cross_check_goal_path = cross_check_dir / f"{name}.atlahs-goal.goal"
            cross_check_payload = cross_check_trace.render().encode()
            cross_check_trace.write(cross_check_goal_path)
            planned_cross_check = _PlannedDependencyCrossCheck(
                comparison=cross_check_comparison,
                goal_path=cross_check_goal_path,
                completion_csv=(
                    cross_check_dir / f"{name}.atlahs-goal.{cfg.profile}.csv"
                ),
                goal_bytes=len(cross_check_payload),
                goal_sha256=hashlib.sha256(cross_check_payload).hexdigest(),
                tolerance_ps=cfg.dependency_cross_check_tolerance_ps,
            )
        return _PlannedStep(
            step_index=record.step_index,
            virtual_time_ps=record.virtual_time_ps,
            artifacts=tuple(planned_artifacts),
            locality=locality,
            compatibility_fast_path=compatibility_fast_path,
            compute_in_artifacts=True,
            compute_service_ps=(
                sum(max(calc_ns, 1) for calc_ns in timing.layer_calc_ns) * 1000
            ),
            compute_estimate_ps=timing.compute_estimate_ps,
            host_profile=timing.host_profile,
            host_launch_class=timing.host_launch_class,
            host_launch_count=timing.host_launch_count,
            provider_compute_ps=timing.provider_compute_ps,
            host_launch_floor_ps=timing.host_launch_floor_ps,
            host_launch_floor_lower_ps=timing.host_launch_floor_lower_ps,
            host_launch_floor_upper_ps=timing.host_launch_floor_upper_ps,
            host_empirical_lower_ps=timing.host_empirical_lower_ps,
            host_empirical_upper_ps=timing.host_empirical_upper_ps,
            exposed_host_ps=timing.exposed_host_ps,
            represented_bound=timing.represented_bound,
            num_sampled=timing.num_sampled,
            sample_count_exact=timing.sample_count_exact,
            per_layer_calc_ns=timing.per_layer_calc_ns,
            layer_calc_ns=timing.layer_calc_ns,
            routing_mode=routing_mode,
            placement_epoch=(
                moe_operations[0].placement_epoch
                if routing_mode == "captured"
                else None
            ),
            profile=cfg.profile,
            linkspeed_bps=cfg.linkspeed_bps,
            topology=cfg.topology,
            unsafe_disable_child_lifetime_binding=(
                cfg.unsafe_disable_child_lifetime_binding
            ),
            effective_dependency_edge_count=len(locality.dependency_edges),
            boundary_edge_count=len(projection.boundaries),
            serialized_edge_count=len(projection.serialized_edges),
            graph_artifact_count=len(projection.artifacts),
            dependency_cross_check=planned_cross_check,
            collective_latency_profile=collective_profile,
            collective_fixed_cost_envelope_id=(
                None
                if cfg.resolved_collective_fixed_cost_envelope is None
                else cfg.resolved_collective_fixed_cost_envelope.envelope_id
            ),
            collective_fixed_cost_arm=(
                None
                if cfg.resolved_collective_fixed_cost_envelope is None
                else cfg.collective_fixed_cost_arm
            ),
            collective_evidence_class=cfg.resolved_collective_evidence_class,
            collective_floor_calibration=collective_floor,
            collective_registration=cfg.resolved_collective_registration,
            registration_events=registration_events,
        )

    def _run_goal(
        self,
        plan: _PlannedStep,
        goal_path: Path,
        completion_csv: Path,
    ):
        goal_bin = to_binary(goal_path)
        return run_htsim_rnic(
            HtsimRnicConfig(
                goal_bin=goal_bin,
                profile=plan.profile,
                linkspeed_bps=plan.linkspeed_bps,
                completion_csv=completion_csv,
                topology=plan.topology,
                unsafe_disable_child_lifetime_binding=(
                    plan.unsafe_disable_child_lifetime_binding
                ),
            )
        )

    def _execute_plan(self, plan: _PlannedStep) -> _SimulatedStep:
        """Execute checked artifacts in their authoritative graph order."""

        backend_artifact_count = sum(
            artifact.goal_path is not None for artifact in plan.artifacts
        )
        if (
            plan.profile in _STATEFUL_MULTI_ARTIFACT_PROFILES
            and backend_artifact_count > 1
        ):
            raise RuntimeError(
                f"profile {plan.profile!r} cannot execute an ordered step with "
                f"{backend_artifact_count} GOAL artifacts: each artifact currently "
                "starts a fresh backend process and resets simulator state; use "
                "'rnic-nn' or 'rnic-nn-fluid', or complete BACK-38 "
                "state-preserving artifact execution"
            )

        fabric_services = []
        composed_services = []
        num_flows = 0
        quiescent = True
        backend_runs = 0
        authority_timing_rows: list[tuple[int, int, int]] = []
        authority_artifact_names: list[str] = []
        authority_artifact_sha256: list[str] = []
        authority_artifact_bytes: list[int] = []
        artifact_offset_ps = 0
        for artifact in plan.artifacts:
            if artifact.goal_path is None:
                if artifact.completion_csv is not None:
                    raise AssertionError("analytic artifact has a completion path")
                fabric_service_ps = 0
            else:
                if artifact.completion_csv is None:
                    raise AssertionError("GOAL artifact has no completion path")
                if plan.dependency_cross_check is not None:
                    payload = artifact.goal_path.read_bytes()
                    authority_artifact_names.append(artifact.goal_path.name)
                    authority_artifact_sha256.append(
                        hashlib.sha256(payload).hexdigest()
                    )
                    authority_artifact_bytes.append(len(payload))
                run = self._run_goal(
                    plan,
                    artifact.goal_path,
                    artifact.completion_csv,
                )
                fabric_service_ps = run.job_completion_time_ps()
                if plan.dependency_cross_check is not None:
                    authority_timing_rows.extend(
                        (
                            flow.tag,
                            artifact_offset_ps + flow.start_time_ps,
                            artifact_offset_ps + flow.completion_time_ps,
                        )
                        for flow in run.flows
                    )
                num_flows += len(run.flows)
                quiescent = quiescent and run.quiescent
                backend_runs += 1
            fabric_services.append(fabric_service_ps)
            composed_services.append(
                artifact.registration_cost_ps
                + artifact.collective_base_latency_ps
                + artifact.aggregate_collective_floor_ps
                + max(artifact.local_service_ps, fabric_service_ps)
            )
            artifact_offset_ps += composed_services[-1]
        fabric_phase_service_ps = tuple(fabric_services)
        composed_phase_service_ps = tuple(composed_services)
        represented_compute_ps = 0 if plan.compute_in_artifacts else plan.compute_service_ps
        makespan_ps = represented_compute_ps + sum(composed_services)
        cross_check_report = None
        if plan.dependency_cross_check is not None:
            cross_check = plan.dependency_cross_check
            cross_check_run = self._run_goal(
                plan,
                cross_check.goal_path,
                cross_check.completion_csv,
            )
            cross_check_report = complete_dependency_cross_check(
                cross_check.comparison,
                authority_rows=tuple(authority_timing_rows),
                cross_check_rows=tuple(
                    (
                        flow.tag,
                        flow.start_time_ps,
                        flow.completion_time_ps,
                    )
                    for flow in cross_check_run.flows
                ),
                authority_completion_ps=makespan_ps,
                cross_check_completion_ps=(
                    cross_check_run.job_completion_time_ps()
                ),
                tolerance_ps=cross_check.tolerance_ps,
                authority_artifact_names=tuple(authority_artifact_names),
                authority_artifact_sha256=tuple(authority_artifact_sha256),
                authority_artifact_bytes=tuple(authority_artifact_bytes),
                cross_check_artifact_name=(
                    f"{cross_check.goal_path.parent.name}/"
                    f"{cross_check.goal_path.name}"
                ),
                cross_check_artifact_sha256=cross_check.goal_sha256,
                cross_check_artifact_bytes=cross_check.goal_bytes,
                authority_quiescent=quiescent,
                cross_check_quiescent=cross_check_run.quiescent,
                authority_flow_count=num_flows,
                cross_check_flow_count=len(cross_check_run.flows),
            )
        outcome = StepNetworkOutcome(
            step_index=plan.step_index,
            compute_estimate_ps=plan.compute_estimate_ps,
            num_sampled=plan.num_sampled,
            sample_count_exact=plan.sample_count_exact,
            per_layer_calc_ns=plan.per_layer_calc_ns,
            layer_calc_ns=plan.layer_calc_ns,
            makespan_ps=makespan_ps,
            num_flows=num_flows,
            quiescent=quiescent,
            routing_mode=plan.routing_mode,
            placement_epoch=plan.placement_epoch,
            host_profile=plan.host_profile,
            host_launch_class=plan.host_launch_class,
            host_launch_count=plan.host_launch_count,
            provider_compute_ps=plan.provider_compute_ps,
            host_launch_floor_ps=plan.host_launch_floor_ps,
            host_launch_floor_lower_ps=plan.host_launch_floor_lower_ps,
            host_launch_floor_upper_ps=plan.host_launch_floor_upper_ps,
            host_empirical_lower_ps=plan.host_empirical_lower_ps,
            host_empirical_upper_ps=plan.host_empirical_upper_ps,
            exposed_host_ps=plan.exposed_host_ps,
            represented_bound=plan.represented_bound,
        )
        locality = plan.locality
        locality_outcome = StepLocalityOutcome(
            step_index=plan.step_index,
            authority=locality.authority,
            compatibility_fast_path=plan.compatibility_fast_path,
            total_directed_bytes=locality.total_directed_bytes,
            fabric_directed_bytes=locality.fabric_bytes,
            nvlink_directed_bytes=locality.nvlink_bytes,
            fabric_segments=locality.fabric_segments,
            nvlink_segments=locality.nvlink_segments,
            phase_count=len(locality.phases),
            backend_runs=backend_runs,
            compute_service_ps=plan.compute_service_ps,
            nvlink_service_ps=sum(
                artifact.local_service_ps
                for artifact in plan.artifacts
                if artifact.collective_operation_id is not None
            ),
            nvlink_bandwidth_bytes_per_second=(
                locality.nvlink_bandwidth_bytes_per_second
            ),
            fabric_phase_service_ps=fabric_phase_service_ps,
            composed_phase_service_ps=composed_phase_service_ps,
            local_phase_service_ps=tuple(
                artifact.local_service_ps for artifact in plan.artifacts
            ),
            base_phase_latency_ps=tuple(
                artifact.collective_base_latency_ps for artifact in plan.artifacts
            ),
            collective_floor_phase_ps=(
                tuple(
                    artifact.aggregate_collective_floor_ps
                    for artifact in plan.artifacts
                )
                if plan.collective_floor_calibration is not None
                else ()
            ),
            registration_phase_cost_ps=(
                tuple(artifact.registration_cost_ps for artifact in plan.artifacts)
                if plan.collective_registration is not None
                else ()
            ),
            local_phase_medium=tuple(
                NVLINK_MEDIUM
                if artifact.collective_operation_id is not None
                else GPU_COMPUTE_MEDIUM
                for artifact in plan.artifacts
            ),
            ordering_authority="execution-graph",
            graph_execution_id=locality.graph_execution_id,
            effective_dependency_edge_count=(
                plan.effective_dependency_edge_count
            ),
            artifact_count=len(plan.artifacts),
            graph_artifact_count=plan.graph_artifact_count,
            boundary_edge_count=plan.boundary_edge_count,
            serialized_edge_count=plan.serialized_edge_count,
            artifact_operation_ids=tuple(
                artifact.operation_ids for artifact in plan.artifacts
            ),
        )
        collective_timing_outcome = None
        if plan.collective_latency_profile is not None:
            profile = plan.collective_latency_profile
            collective_timing_outcome = StepCollectiveTimingOutcome(
                step_index=plan.step_index,
                profile_id=profile.profile_id,
                bandwidth_bytes_per_second=profile.bandwidth_bytes_per_second,
                participant_latency_ps=profile.participant_latency_ps,
                propagation_reference_ps=profile.propagation_reference_ps,
                envelope_id=plan.collective_fixed_cost_envelope_id,
                arm=plan.collective_fixed_cost_arm,
                evidence_class=(
                    plan.collective_evidence_class or profile.evidence_class
                ),
                artifacts=tuple(
                    CollectiveArtifactTiming(
                        artifact_id=artifact.artifact_id,
                        operation_ids=artifact.operation_ids,
                        collective_operation_id=artifact.collective_operation_id,
                        participant_count=artifact.participant_count,
                        critical_endpoint_bytes=artifact.critical_endpoint_bytes,
                        local_service_ps=artifact.local_service_ps,
                        fabric_transport_ps=fabric_service_ps,
                        collective_base_latency_ps=(
                            artifact.collective_base_latency_ps
                        ),
                        registration_cost_ps=artifact.registration_cost_ps,
                        composed_service_ps=composed_service_ps,
                    )
                    for artifact, fabric_service_ps, composed_service_ps in zip(
                        plan.artifacts,
                        fabric_phase_service_ps,
                        composed_phase_service_ps,
                        strict=True,
                    )
                ),
            )
        collective_floor_timing_outcome = None
        if plan.collective_floor_calibration is not None:
            calibration = plan.collective_floor_calibration
            fitted_artifacts = tuple(
                CollectiveFloorArtifactTiming(
                    artifact_id=artifact.artifact_id,
                    collective_operation_id=artifact.collective_operation_id or "",
                    semantic_collective=artifact.collective_floor_semantic or "",
                    local_service_ps=artifact.local_service_ps,
                    aggregate_floor_ps=artifact.aggregate_collective_floor_ps,
                    fabric_transport_ps=fabric_service_ps,
                    collective_base_latency_ps=(
                        artifact.collective_base_latency_ps
                    ),
                    registration_cost_ps=artifact.registration_cost_ps,
                    composed_service_ps=composed_service_ps,
                    estimate=artifact.collective_floor_estimate,
                )
                for artifact, fabric_service_ps, composed_service_ps in zip(
                    plan.artifacts,
                    fabric_phase_service_ps,
                    composed_phase_service_ps,
                    strict=True,
                )
                if artifact.collective_floor_estimate is not None
            )
            collective_floor_timing_outcome = StepCollectiveFloorTimingOutcome(
                step_index=plan.step_index,
                calibration_id=calibration.calibration_id,
                dtype=self.config.collective_floor_dtype or "",
                input_surface=calibration.input_surface,
                host_launch_floor_ps=plan.host_launch_floor_ps,
                artifacts=fitted_artifacts,
            )
        registration_outcome = None
        if plan.collective_registration is not None:
            model = plan.collective_registration
            registration_outcome = StepCollectiveRegistrationOutcome(
                step_index=plan.step_index,
                model_id=model.model_id,
                cost_id=model.cost.cost_id,
                evidence_class=model.cost.evidence_class,
                registration_cost_ps=model.cost.registration_cost_ps,
                channels_per_communicator=model.channels_per_communicator,
                charged_ps=sum(
                    event.cost_ps for event in plan.registration_events
                ),
                events=plan.registration_events,
                artifact_registration_ps=tuple(
                    artifact.registration_cost_ps for artifact in plan.artifacts
                ),
            )
        result = StepResult(
            step_index=plan.step_index,
            step_latency_ps=makespan_ps,
            completed_at_ps=plan.virtual_time_ps + makespan_ps,
        )
        return _SimulatedStep(
            result=result,
            outcome=outcome,
            locality_outcome=locality_outcome,
            collective_timing_outcome=collective_timing_outcome,
            collective_floor_timing_outcome=collective_floor_timing_outcome,
            dependency_cross_check_report=cross_check_report,
            collective_registration_outcome=registration_outcome,
        )

    def _simulate_step(self, record: StepRecord) -> _SimulatedStep:
        plan = self._plan_step(record)
        if plan is None:
            return _SimulatedStep(
                result=None,
                outcome=None,
                locality_outcome=None,
                collective_timing_outcome=None,
                collective_floor_timing_outcome=None,
                dependency_cross_check_report=None,
            )
        return self._execute_plan(plan)

    def _publish(self, simulation: _SimulatedStep) -> StepResult | None:
        if simulation.outcome is not None:
            self.outcomes.append(simulation.outcome)
        if simulation.locality_outcome is not None:
            self.locality_outcomes.append(simulation.locality_outcome)
        if simulation.collective_timing_outcome is not None:
            self.collective_timing_outcomes.append(
                simulation.collective_timing_outcome
            )
        if simulation.collective_floor_timing_outcome is not None:
            self.collective_floor_timing_outcomes.append(
                simulation.collective_floor_timing_outcome
            )
        if simulation.dependency_cross_check_report is not None:
            self.dependency_cross_check_reports.append(
                simulation.dependency_cross_check_report
            )
        if simulation.collective_registration_outcome is not None:
            self.collective_registration_outcomes.append(
                simulation.collective_registration_outcome
            )
        return simulation.result

    def __call__(self, record: StepRecord) -> StepResult | None:
        return self._publish(self._simulate_step(record))


class HtsimPersistentStepSink(HtsimStepSink):
    """Opt-in prepared replay using a persistent local worker pool.

    ``prepare`` copies and lowers a finite record sequence before the scheduler
    consumes it. Complete checked artifact plans then run concurrently. Calls
    must replay the prepared values in exact order; no prediction, fallback,
    or record substitution is permitted. This pool does not make backend state
    persistent, so the same stateful multi-artifact guard as the base sink
    applies.

    The executor survives across fully consumed batches until ``close``. Use a
    context manager when possible so worker shutdown belongs to the timed
    end-to-end boundary.
    """

    def __init__(self, config: HtsimStepSinkConfig, *, max_workers: int) -> None:
        super().__init__(config)
        if isinstance(max_workers, bool) or not isinstance(max_workers, int):
            raise TypeError("max_workers must be an integer")
        if max_workers <= 0:
            raise ValueError("max_workers must be positive")
        self.max_workers = max_workers
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="simllm-htsim",
        )
        self._prepared: deque[_PreparedStep] = deque()
        self._state_lock = Lock()
        self._preparing = False
        self._closed = False

    @property
    def prepared_steps_remaining(self) -> int:
        with self._state_lock:
            return len(self._prepared)

    def prepare(self, records: Sequence[StepRecord]) -> None:
        """Prepare one finite replay atomically from the caller's perspective."""

        if not self.config.unsafe_disable_child_lifetime_binding:
            prepare_htsim_child_lifetime()

        copied_records = tuple(copy.deepcopy(record) for record in records)
        if not copied_records:
            raise ValueError("records must not be empty")
        if any(not isinstance(record, StepRecord) for record in copied_records):
            raise TypeError("records must contain StepRecord values")
        step_indices = [record.step_index for record in copied_records]
        if len(step_indices) != len(set(step_indices)):
            raise ValueError("prepared records must have unique step indices")

        with self._state_lock:
            if self._closed:
                raise RuntimeError("persistent step sink is closed")
            if self._preparing:
                raise RuntimeError("a preparation is already in progress")
            if self._prepared:
                raise RuntimeError("consume the prepared replay before preparing another")
            self._preparing = True

        futures: list[Future[_SimulatedStep] | None] = []
        try:
            plans = tuple(self._plan_step(record) for record in copied_records)
            futures = [
                self._executor.submit(self._execute_plan, plan)
                if plan is not None
                else None
                for plan in plans
            ]
            wait(tuple(future for future in futures if future is not None))
            simulations = tuple(
                future.result()
                if future is not None
                else _SimulatedStep(
                    result=None,
                    outcome=None,
                    locality_outcome=None,
                    collective_timing_outcome=None,
                    collective_floor_timing_outcome=None,
                    dependency_cross_check_report=None,
                )
                for future in futures
            )
        except BaseException:
            submitted = tuple(future for future in futures if future is not None)
            for future in submitted:
                future.cancel()
            wait(submitted)
            with self._state_lock:
                self._preparing = False
            raise

        prepared = deque(
            _PreparedStep(record, simulation)
            for record, simulation in zip(copied_records, simulations, strict=True)
        )
        with self._state_lock:
            self._prepared = prepared
            self._preparing = False

    def __call__(self, record: StepRecord) -> StepResult | None:
        with self._state_lock:
            if self._preparing:
                raise RuntimeError("prepared results are not available yet")
            if not self._prepared:
                raise RuntimeError("prepare records before calling the persistent sink")
            prepared = self._prepared[0]
            if record != prepared.record:
                raise ValueError(
                    "record does not match the next prepared step "
                    f"{prepared.record.step_index}"
                )
            self._prepared.popleft()
            return self._publish(prepared.simulation)

    def close(self) -> None:
        with self._state_lock:
            if self._closed:
                return
            self._closed = True
        self._executor.shutdown(wait=True, cancel_futures=False)

    def __enter__(self) -> HtsimPersistentStepSink:  # noqa: PYI034 (typing.Self needs 3.11; CI runs 3.10)
        with self._state_lock:
            if self._closed:
                raise RuntimeError("persistent step sink is closed")
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.close()
