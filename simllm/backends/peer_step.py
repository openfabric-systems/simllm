"""Retained local packet service behind the checked step locality projection."""

from __future__ import annotations

import copy
from collections import Counter
from dataclasses import dataclass, field

from simllm.backends.htsim_nvlink import (
    NvlinkAlignedOptions,
    NvlinkCandidateProfile,
    NvlinkTransfer,
)
from simllm.backends.htsim_rnic import FlowCompletion, RnicRunResult
from simllm.backends.nvlink_runtime import NvlinkPhysicalBinding
from simllm.compute.gpu_packet_port import GpuPacketExtent, GpuPeerPacketSession
from simllm.core.authority import work_completed_bytes
from simllm.core.execution import (
    CollectiveWork,
    CompletionEvent,
    ComputeWork,
    EventPhase,
    ExecutionGraph,
    ExecutionResult,
)
from simllm.core.execution_io import effective_dependency_edges, execution_result_to_json
from simllm.core.step import StepResult
from simllm.goal.emitter import GoalMessage
from simllm.placement import FabricTopologyManifest, PlacementManifest
from simllm.traffic import ClassifiedCommunicationPhase, StepLocalityPlan


@dataclass(frozen=True)
class PeerPacketConfig:
    """Explicit physical inventory plus one declared service profile per domain."""

    fabric: FabricTopologyManifest
    profiles: tuple[tuple[str, NvlinkCandidateProfile], ...]
    options: NvlinkAlignedOptions = field(default_factory=NvlinkAlignedOptions)
    credit_return_processing_ps: int = 0
    acknowledgement_processing_ps: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.fabric, FabricTopologyManifest):
            raise TypeError("peer_packet requires a fabric topology manifest")
        if not isinstance(self.options, NvlinkAlignedOptions):
            raise TypeError("peer_packet requires explicit NvlinkAlignedOptions")
        # Snapshot the caller's mutable manifest before any runtime is created.
        object.__setattr__(self, "fabric", copy.deepcopy(self.fabric))
        self.fabric.validate()
        if not self.fabric.peer_fabrics:
            raise ValueError("peer_packet requires explicit GPU attachment domains")
        if not isinstance(self.profiles, (tuple, list)) or any(
            not isinstance(row, (tuple, list)) or len(row) != 2 for row in self.profiles
        ):
            raise TypeError("peer profiles must be domain and profile pairs")
        object.__setattr__(self, "profiles", tuple(tuple(row) for row in self.profiles))
        if any(not isinstance(name, str) or not isinstance(profile, NvlinkCandidateProfile)
               for name, profile in self.profiles):
            raise TypeError("peer profiles require named NvlinkCandidateProfile values")
        by_id = dict(self.profiles)
        if len(by_id) != len(self.profiles) or set(by_id) != {domain.domain_id for domain in self.fabric.peer_fabrics}:
            raise ValueError("peer profiles must cover every domain exactly once")
        for domain in self.fabric.peer_fabrics:
            binding = NvlinkPhysicalBinding(domain, self.credit_return_processing_ps,
                                            self.acknowledgement_processing_ps)
            # Constructor preflight installs no events or simulator work.
            GpuPeerPacketSession(f"validate:{domain.domain_id}", by_id[domain.domain_id],
                                 binding, options=self.options)

    def validate_placement(self, placement: PlacementManifest | None) -> None:
        if not isinstance(placement, PlacementManifest):
            raise TypeError("peer_packet requires the semantic placement manifest")
        for node in self.fabric.nodes:
            for gpu in node.gpus:
                rank = placement.by_rank(gpu.global_rank)
                if rank.hostname != node.node_id:
                    raise ValueError("peer topology node disagrees with semantic rank placement")


@dataclass(frozen=True)
class PeerPacketPhaseResult:
    phase: ClassifiedCommunicationPhase
    released_at_ps: int
    visible_at_ps: int
    extents: tuple[GpuPacketExtent, ...]

    @property
    def service_ps(self) -> int:
        return self.visible_at_ps - self.released_at_ps


@dataclass(frozen=True)
class PeerArtifactBoundary:
    artifact_id: str
    operation_ids: tuple[str, ...]
    started_at_ps: int
    completed_at_ps: int
    local_service_ps: int
    fabric_service_ps: int
    fixed_service_ps: int
    local_phase: PeerPacketPhaseResult | None
    fabric_messages: tuple[GoalMessage, ...] = ()
    fabric_completions: tuple[FlowCompletion, ...] = ()

    def __post_init__(self) -> None:
        values = (self.started_at_ps, self.completed_at_ps, self.local_service_ps,
                  self.fabric_service_ps, self.fixed_service_ps)
        if any(type(value) is not int or value < 0 for value in values):
            raise ValueError("peer artifact boundaries require nonnegative integer times")
        if self.completed_at_ps != self.started_at_ps + self.fixed_service_ps + max(self.local_service_ps, self.fabric_service_ps):
            raise ValueError("peer artifact boundary does not conserve its realized interval")
        if self.local_phase is not None and (
            self.local_phase.released_at_ps != self.started_at_ps + self.fixed_service_ps
            or self.local_phase.service_ps != self.local_service_ps
        ):
            raise ValueError("peer artifact disagrees with its physical local phase")
        if Counter((m.source_rank, m.destination_rank, m.tag, m.payload_bytes)
                   for m in self.fabric_messages) != Counter(
                       (f.source, f.destination, f.tag, f.payload_bytes) for f in self.fabric_completions):
            raise ValueError("peer artifact lost its remote completion inventory")
        if any(message.operation_id not in self.operation_ids for message in self.fabric_messages):
            raise ValueError("peer artifact remote message lost its original graph owner")


def validate_peer_fabric_result(
    messages: tuple[GoalMessage, ...], run: RnicRunResult, profile: str,
) -> tuple[FlowCompletion, ...]:
    """Join the mixed phase's remote projection before accepting its makespan."""
    if not isinstance(run, RnicRunResult) or not run.quiescent or not messages:
        raise ValueError("peer phase requires a quiescent remote completion result")
    rows = tuple(run.flows)
    if any(not isinstance(row, FlowCompletion) for row in rows):
        raise TypeError("peer phase requires typed remote completion rows")
    expected = Counter((m.source_rank, m.destination_rank, m.tag, m.payload_bytes) for m in messages)
    actual = Counter((f.source, f.destination, f.tag, f.payload_bytes) for f in rows)
    if expected != actual or len({row.flow_id for row in rows}) != len(rows):
        raise ValueError("peer phase remote completion inventory disagrees with rendered messages")
    for row in rows:
        if (row.profile != profile
                or any(type(value) is not int or not 0 <= value < 2**64 for value in (
                    row.flow_id, row.start_time_ps, row.completion_time_ps, row.fct_ps))
                or row.completion_time_ps < row.start_time_ps
                or row.fct_ps != row.completion_time_ps - row.start_time_ps):
            raise ValueError("peer phase received invalid remote completion timestamps or profile")
    if run.goal_completion_time_ps is not None and (
        type(run.goal_completion_time_ps) is not int
        or not max(row.completion_time_ps for row in rows) <= run.goal_completion_time_ps < 2**64
    ):
        raise ValueError("peer phase GOAL completion disagrees with remote completion rows")
    return rows


@dataclass(frozen=True)
class PeerStepEvidence:
    """Original graph completion joined to extent visibility and retained state."""

    graph: ExecutionGraph
    execution_result: ExecutionResult
    artifacts: tuple[PeerArtifactBoundary, ...]
    session_observations: tuple[dict, ...]

    def validate_result(self, result: StepResult) -> None:
        if (result.step_index != self.graph.step_index
                or result.completed_at_ps != self.execution_result.completed_at_ps
                or result.step_latency_ps != result.completed_at_ps - self.graph.released_at_ps):
            raise ValueError("peer packet evidence disagrees with the returned StepResult")


class PeerPacketRuntime:
    """One non-rewindable local runtime retained for the lifetime of a step sink."""

    def __init__(self, config: PeerPacketConfig, placement: PlacementManifest) -> None:
        config = copy.deepcopy(config)
        config.validate_placement(placement)
        self.config = config
        profiles = dict(config.profiles)
        self._sessions = {
            domain.domain_id: GpuPeerPacketSession(
                f"local-peer:{domain.domain_id}", profiles[domain.domain_id],
                NvlinkPhysicalBinding(domain, config.credit_return_processing_ps,
                                      config.acknowledgement_processing_ps), options=config.options,
            ) for domain in config.fabric.peer_fabrics
        }
        self._rank_domain = {port.gpu_rank: domain.domain_id for domain in config.fabric.peer_fabrics
                             for port in domain.ports if port.gpu_rank is not None}
        self._failed = False
        self._closed = False
        self._completed_executions: set[str] = set()

    def _require_live(self) -> None:
        if self._failed or self._closed:
            raise RuntimeError("local peer runtime is closed or invalid after a failed execution")

    def _transfers(self, graph_id: str, phase: ClassifiedCommunicationPhase, at_ps: int):
        groups: dict[str, list[NvlinkTransfer]] = {}
        endpoint_count = max(self._rank_domain, default=0) + 1
        for index, segment in enumerate(phase.nvlink_segments):
            domain = self._rank_domain.get(segment.source_rank)
            if domain is None or self._rank_domain.get(segment.destination_rank) != domain:
                raise ValueError("local segment does not have one declared peer domain")
            extent_id = f"{graph_id}:{phase.phase.phase_id}:extent-{index}"
            groups.setdefault(domain, []).append(NvlinkTransfer(
                extent_id=extent_id, source=segment.source_rank, destination=segment.destination_rank,
                payload_bytes=segment.payload_bytes, released_at_ps=at_ps,
                topology_endpoint_count=endpoint_count,
            ))
        return {name: tuple(values) for name, values in groups.items()}

    def validate_graph(self, graph: ExecutionGraph, locality: StepLocalityPlan) -> None:
        """Validate every local phase before registration or runtime mutation."""
        self._require_live()
        if graph.execution_id in self._completed_executions:
            raise ValueError("local peer runtime cannot execute the same graph twice")
        if any(session.now_ps > graph.released_at_ps for session in self._sessions.values()):
            raise ValueError("next step precedes the retained local calendar")
        for phase in locality.phases:
            groups = self._transfers(graph.execution_id, phase, graph.released_at_ps)
            for name, transfers in groups.items():
                session = self._sessions[name]
                session.validate_phase(graph.execution_id, phase.phase.operation_id, transfers,
                                       ready_at_ps=graph.released_at_ps)

    def advance_to(self, at_ps: int) -> None:
        self._require_live()
        if type(at_ps) is not int or any(at_ps < session.now_ps for session in self._sessions.values()):
            raise ValueError("step execution cannot rewind a physical peer calendar")
        try:
            for session in self._sessions.values():
                session.advance_to(at_ps)
        except Exception:
            self._failed = True
            raise

    def run_phase(self, graph_id: str, phase: ClassifiedCommunicationPhase, at_ps: int) -> PeerPacketPhaseResult:
        self._require_live()
        groups = self._transfers(graph_id, phase, at_ps)
        operation_id = phase.phase.operation_id
        if operation_id is None:
            raise ValueError("local phase has no original graph operation identity")
        # Recheck the whole phase before advancing even existing control tails.
        for name, transfers in groups.items():
            session = self._sessions[name]
            session.validate_phase(graph_id, operation_id, transfers, ready_at_ps=at_ps)
        self.advance_to(at_ps)
        extents, visible = [], at_ps
        try:
            for name, transfers in groups.items():
                extents.extend(self._sessions[name].admit(graph_id, operation_id, transfers))
            for name, transfers in groups.items():
                finish = self._sessions[name].advance_until_visible(tuple(t.extent_id for t in transfers))
                visible = max(visible, finish)
            self.advance_to(visible)
        except Exception:
            self._failed = True
            raise
        return PeerPacketPhaseResult(phase, at_ps, visible, tuple(extents))

    def step_evidence(self, graph: ExecutionGraph, artifacts: tuple[PeerArtifactBoundary, ...]) -> PeerStepEvidence:
        self._require_live()
        try:
            evidence = build_peer_step_evidence(graph, artifacts, tuple(self._sessions.values()))
        except Exception:
            self._failed = True
            raise
        self._completed_executions.add(graph.execution_id)
        return evidence

    def close(self) -> tuple[dict, ...]:
        self._require_live()
        try:
            for session in self._sessions.values():
                if session.extents:
                    session.drain()
                session.port_snapshots(final=True)
        except Exception:
            self._failed = True
            raise
        self._closed = True
        return tuple(session.evidence() for session in self._sessions.values())

    def invalidate(self) -> None:
        """Prevent reuse after an execution failed beyond its admission boundary."""
        self._failed = True


def build_peer_step_evidence(
    graph: ExecutionGraph, artifacts: tuple[PeerArtifactBoundary, ...],
    sessions: tuple[GpuPeerPacketSession, ...],
) -> PeerStepEvidence:
    """Check the realized serial artifact projection against the original graph."""
    if not artifacts or len({row.artifact_id for row in artifacts}) != len(artifacts):
        raise ValueError("peer step lost or duplicated an artifact boundary")
    cursor = graph.released_at_ps
    owned = {operation.operation_id: [] for operation in graph.operations}
    for artifact in artifacts:
        if artifact.started_at_ps != cursor:
            raise ValueError("peer artifacts do not form the selected serial execution")
        cursor = artifact.completed_at_ps
        if not artifact.operation_ids or len(set(artifact.operation_ids)) != len(artifact.operation_ids):
            raise ValueError("peer artifact has no unique semantic owners")
        for name in artifact.operation_ids:
            if name not in owned:
                raise ValueError("peer artifact names an operation outside the graph")
            owned[name].append(artifact)
    starts, finishes = {}, {}
    for operation in graph.operations:
        name = operation.operation_id
        rows = owned[name]
        if not rows:
            raise ValueError("original graph operation has no executed artifact")
        starts[name] = rows[0].started_at_ps
        finishes[name] = rows[-1].completed_at_ps
        if isinstance(operation.work, ComputeWork):
            if len(rows) != 1 or operation.work.nominal_duration_ps is None:
                raise ValueError("compute operation lost its exact provider service")
            finishes[name] = starts[name] + max(operation.work.nominal_duration_ps, 1000)
            if finishes[name] > rows[0].completed_at_ps:
                raise ValueError("compute completes outside its executed artifact")
        elif not isinstance(operation.work, CollectiveWork):
            raise TypeError("peer step projection has an unsupported graph work kind")
        if starts[name] < max(graph.released_at_ps, operation.not_before_ps):
            raise ValueError("operation precedes its graph release or not-before boundary")
    for edge in effective_dependency_edges(graph):
        # This supported executor uses whole-operation frontiers. It may impose
        # a conservative barrier, but cannot violate a participant-local edge.
        if starts[edge.operation_id] < finishes[edge.predecessor_id]:
            raise ValueError("peer packet execution violates an effective graph dependency")
    boundary = graph.completion_operation_ids or tuple(owned)
    completion = max(finishes[name] for name in boundary)
    if completion != cursor:
        raise ValueError("peer step boundary is not the original graph completion")
    events = [event for session in sessions for event in session.events if event.execution_id == graph.execution_id]
    extents = [extent for artifact in artifacts if artifact.local_phase is not None for extent in artifact.local_phase.extents]
    actual_extents = [extent for session in sessions for extent in session.extents if extent.execution_id == graph.execution_id]
    if Counter(extents) != Counter(actual_extents):
        raise ValueError("local phases do not conserve the admitted extent inventory")
    if len({extent.subject_object_id for extent in extents}) != len(extents):
        raise ValueError("peer step repeats a logical extent identity")
    for artifact in artifacts:
        phase = artifact.local_phase
        if phase is None:
            continue
        expected = Counter((row.source_rank, row.destination_rank, row.payload_bytes) for row in phase.phase.nvlink_segments)
        actual = Counter((extent.transfer.source, extent.transfer.destination, extent.transfer.payload_bytes) for extent in phase.extents)
        if expected != actual:
            raise ValueError("peer extents do not conserve the checked locality bytes")
        for extent in phase.extents:
            if (extent.execution_id != graph.execution_id
                    or extent.operation_id != phase.phase.phase.operation_id
                    or extent.operation_id not in artifact.operation_ids):
                raise ValueError("peer extent lost its original graph owner")
            rows = [event for event in events if event.subject_object_id == extent.subject_object_id]
            if any(event.operation_id != extent.operation_id for event in rows):
                raise ValueError("peer extent event disagrees with its original operation")
            if Counter(event.phase for event in rows) != Counter((EventPhase.SUBMITTED, EventPhase.QUEUED, EventPhase.STARTED, EventPhase.COMPLETED)):
                raise ValueError("logical peer extent has missing or duplicate completion events")
            times = {event.phase: event.timestamp_ps for event in rows}
            if not (times[EventPhase.SUBMITTED] == times[EventPhase.QUEUED] == phase.released_at_ps
                    <= times[EventPhase.STARTED] <= times[EventPhase.COMPLETED] <= phase.visible_at_ps):
                raise ValueError("peer extent events disagree with the phase boundary")
            if next(event.completed_bytes for event in rows if event.phase is EventPhase.COMPLETED) != extent.transfer.payload_bytes:
                raise ValueError("peer extent completion event lost payload bytes")
        if phase.extents and max(event.timestamp_ps for event in events if event.phase is EventPhase.COMPLETED
                                 and event.subject_object_id in {x.subject_object_id for x in phase.extents}) != phase.visible_at_ps:
            raise ValueError("local phase boundary disagrees with extent visibility")
    for operation in graph.operations:
        name = operation.operation_id
        for phase, at in ((EventPhase.SUBMITTED, graph.released_at_ps), (EventPhase.QUEUED, starts[name]),
                          (EventPhase.STARTED, starts[name]), (EventPhase.COMPLETED, finishes[name])):
            events.append(CompletionEvent(graph.execution_id, name, phase, at,
                                           completed_bytes=work_completed_bytes(operation) if phase is EventPhase.COMPLETED else None))
    quiesced = None if any(session.has_pending_physical_work for session in sessions) else completion
    result = ExecutionResult(graph.execution_id, completion, tuple(sorted(events, key=lambda row: row.timestamp_ps)), quiesced)
    execution_result_to_json(result)
    return PeerStepEvidence(graph, result, artifacts, tuple(session.evidence() for session in sessions))
