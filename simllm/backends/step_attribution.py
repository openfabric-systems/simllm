"""Per-request TTFT and TPOT attribution for the packet-level step sink.

:class:`~simllm.backends.step_sink.HtsimStepSink` runs the fabric on
``htsim_rnic`` and publishes one whole-step makespan plus, per step, the
ordered service of every artifact it executed. It publishes no per-request
endpoint, so this module does not invent one: every request co-scheduled in a
step is charged that step's whole service, exactly as
``per_request_fidelity_v1`` froze, and the decomposition is of the request's
own elapsed interval rather than a division of the makespan.

The reducer is a read-only projection. Every timestamp it reports comes from
the sink that executed the backend; it introduces no second timing authority
and advances no object. It reuses ``LatencyAttribution`` and ``RequestMetric``
unchanged, and it mirrors :class:`~simllm.core.completion.CompletionReducer`'s
interval rule: a scheduler gap charged to ``queue_ps``, a pending attribution
carried for a co-scheduled request that does not sample this step, and a hard
conservation check on every sampled interval.

Scope: any locality level, including a placement that mixes intra-node NVLink
service with cross-node fabric service. The sink executes artifacts serially
and composes each one as ``base + max(local, fabric)``, so the step's realized
interval is the ordered concatenation of one disjoint subinterval per artifact
and partitioning by artifact is critical-path selection rather than a sum over
concurrent visits. Inside one artifact the two media compose by maximum: the
resource whose own service equals that maximum decided the finish time and
owns the whole realized service, while the other medium's service is masked,
sits off the critical path, and is published as a work sum under its own name.
:class:`MediumAttribution` keeps the NVLink and fabric components separately
named so no additive TTFT or TPOT decomposition can mix two differently owned
waits, and :class:`~simllm.core.LatencyAttribution` keeps its existing coarse
meaning as their roll-up.

An outcome that carries NVLink work but publishes no per-artifact medium
projection is still refused: the ownership evidence is required, not inferred.
An all-remote outcome without that projection keeps its historical partition
exactly, because there every collective artifact is fabric owned by
construction.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from fractions import Fraction

from simllm.backends.step_sink import (
    GPU_COMPUTE_MEDIUM,
    LOCAL_SERVICE_MEDIA,
    NVLINK_MEDIUM,
    StepLocalityOutcome,
)
from simllm.core import (
    AdditiveVisitTotals,
    LatencyAttribution,
    RequestMetric,
    StepRecord,
    StepResult,
    sampled_request_ids,
)

#: The packet-level sink records no ``QueueVisit``, so the additive work sum
#: stays empty rather than being fabricated from wall-clock services. A caller
#: that needs the work sum must use a runtime authority that publishes visits.
_NO_VISIT_TOTALS = AdditiveVisitTotals()


def _require_nonnegative_int(name: str, value: object) -> int:
    if isinstance(value, bool) or type(value) is not int:
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return value


@dataclass(frozen=True)
class MediumAttribution:
    """One realized interval partitioned by the resource that realized it.

    Every component is a disjoint stretch of the same elapsed interval, so the
    partition stays additive along the critical path. ``nvlink_ps`` and
    ``fabric_ps`` are deliberately distinct names: they are differently owned
    waits and are never reported as one interchangeable queue wait.
    """

    #: scheduler gap before the step was released; no resource owns it
    queue_ps: int = 0
    #: GPU compute artifacts
    kernel_ps: int = 0
    #: artifacts whose NVLink endpoint serializer decided the finish time
    nvlink_ps: int = 0
    #: artifacts whose packet-level fabric service decided the finish time
    fabric_ps: int = 0
    #: artifacts where both media realized the same duration, so neither is
    #: uniquely critical; kept out of both single-medium components
    co_critical_ps: int = 0
    #: semantic collective fixed cost; no resource owns it
    collective_base_ps: int = 0
    #: steps the sink did not simulate, settled by the adapter's fallback
    control_ps: int = 0

    def __post_init__(self) -> None:
        for name in (
            "queue_ps",
            "kernel_ps",
            "nvlink_ps",
            "fabric_ps",
            "co_critical_ps",
            "collective_base_ps",
            "control_ps",
        ):
            _require_nonnegative_int(name, getattr(self, name))

    @property
    def total_ps(self) -> int:
        """Return the conserved elapsed time represented by this partition."""

        return (
            self.queue_ps
            + self.kernel_ps
            + self.nvlink_ps
            + self.fabric_ps
            + self.co_critical_ps
            + self.collective_base_ps
            + self.control_ps
        )

    @property
    def collective_ps(self) -> int:
        """Return the coarse collective roll-up of every communication term."""

        return (
            self.nvlink_ps
            + self.fabric_ps
            + self.co_critical_ps
            + self.collective_base_ps
        )

    def __add__(self, other: object) -> MediumAttribution:
        if not isinstance(other, MediumAttribution):
            return NotImplemented
        return MediumAttribution(
            queue_ps=self.queue_ps + other.queue_ps,
            kernel_ps=self.kernel_ps + other.kernel_ps,
            nvlink_ps=self.nvlink_ps + other.nvlink_ps,
            fabric_ps=self.fabric_ps + other.fabric_ps,
            co_critical_ps=self.co_critical_ps + other.co_critical_ps,
            collective_base_ps=self.collective_base_ps + other.collective_base_ps,
            control_ps=self.control_ps + other.control_ps,
        )


@dataclass(frozen=True)
class MaskedMediumService:
    """Service each medium performed off the realized critical path.

    This is an additive work sum, not a latency partition. A masked service ran
    concurrently with the medium that owned its artifact, so it may exceed wall
    time and must never be added to a TTFT, TPOT or JCT decomposition. It has
    no ``total_ps`` for exactly that reason.
    """

    nvlink_ps: int = 0
    fabric_ps: int = 0

    def __post_init__(self) -> None:
        for name in ("nvlink_ps", "fabric_ps"):
            _require_nonnegative_int(name, getattr(self, name))

    def __add__(self, other: object) -> MaskedMediumService:
        if not isinstance(other, MaskedMediumService):
            return NotImplemented
        return MaskedMediumService(
            nvlink_ps=self.nvlink_ps + other.nvlink_ps,
            fabric_ps=self.fabric_ps + other.fabric_ps,
        )


@dataclass(frozen=True)
class StepAttribution:
    """One step's conserved partition in both the coarse and medium views."""

    attribution: LatencyAttribution
    media: MediumAttribution
    masked: MaskedMediumService

    def __post_init__(self) -> None:
        if not isinstance(self.attribution, LatencyAttribution):
            raise TypeError("attribution must be a LatencyAttribution")
        if not isinstance(self.media, MediumAttribution):
            raise TypeError("media must be a MediumAttribution")
        if not isinstance(self.masked, MaskedMediumService):
            raise TypeError("masked must be a MaskedMediumService")
        if self.media.total_ps != self.attribution.total_ps:
            raise ValueError("medium and coarse partitions disagree on the total")
        if self.media.queue_ps != self.attribution.queue_ps:
            raise ValueError("medium and coarse partitions disagree on queue time")
        if self.media.kernel_ps != self.attribution.kernel_ps:
            raise ValueError("medium and coarse partitions disagree on kernel time")
        if self.media.control_ps != self.attribution.control_ps:
            raise ValueError("medium and coarse partitions disagree on control time")
        if self.media.collective_ps != self.attribution.collective_ps:
            raise ValueError(
                "medium components do not roll up to the collective component"
            )


#: component that owns an artifact's realized service. ``unserviced`` is an
#: artifact whose realized service is zero, e.g. an empty collective.
KERNEL_OWNER = "kernel"
NVLINK_OWNER = "nvlink"
FABRIC_OWNER = "fabric"
CO_CRITICAL_OWNER = "co-critical"
UNSERVICED_OWNER = "unserviced"


@dataclass(frozen=True)
class _ArtifactOwnership:
    """One executed artifact resolved into its owner and its masked service."""

    owner: str
    base_ps: int
    owned_ps: int
    #: local service declared for an NVLink-medium artifact, owned or masked
    nvlink_service_ps: int
    masked_nvlink_ps: int
    masked_fabric_ps: int


def _artifact_ownership(
    *,
    index: int,
    composed_ps: int,
    fabric_ps: int,
    local_ps: int,
    base_ps: int,
    medium: str,
) -> _ArtifactOwnership:
    """Resolve one artifact's realized service to the resource that decided it."""

    path = f"artifact[{index}]"
    if medium not in LOCAL_SERVICE_MEDIA:
        raise ValueError(f"{path} names an unsupported local service medium: {medium!r}")
    for name, value in (
        ("composed service", composed_ps),
        ("fabric service", fabric_ps),
        ("local service", local_ps),
        ("base latency", base_ps),
    ):
        _require_nonnegative_int(f"{path} {name}", value)
    if medium == GPU_COMPUTE_MEDIUM:
        if fabric_ps:
            raise ValueError(f"{path} is compute owned and cannot carry fabric service")
        if base_ps:
            raise ValueError(f"{path} is compute owned and cannot carry a base latency")
    realized_ps = max(local_ps, fabric_ps)
    if composed_ps != base_ps + realized_ps:
        raise ValueError(
            f"{path} composed service disagrees with its base and medium terms"
        )
    nvlink_service_ps = local_ps if medium == NVLINK_MEDIUM else 0
    if realized_ps == 0:
        owner = UNSERVICED_OWNER
    elif medium == GPU_COMPUTE_MEDIUM:
        owner = KERNEL_OWNER
    elif local_ps == fabric_ps:
        #: both media drain together, so neither is uniquely critical
        owner = CO_CRITICAL_OWNER
    elif local_ps > fabric_ps:
        owner = NVLINK_OWNER
    else:
        owner = FABRIC_OWNER
    return _ArtifactOwnership(
        owner=owner,
        base_ps=base_ps,
        owned_ps=realized_ps,
        nvlink_service_ps=nvlink_service_ps,
        masked_nvlink_ps=local_ps if owner == FABRIC_OWNER else 0,
        masked_fabric_ps=fabric_ps if owner == NVLINK_OWNER else 0,
    )


def _legacy_medium_projection(
    locality: StepLocalityOutcome,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[str, ...]]:
    """Derive the medium projection of an all-remote outcome that omits it.

    An all-remote step charges no NVLink service, so every artifact's local
    term is either kernel service (zero fabric service) or absent (the fabric
    decided the artifact and the base latency is the composed remainder). That
    makes the projection recoverable exactly, which is what keeps a locality
    outcome recorded before this projection existed byte-identical.
    """

    if locality.nvlink_directed_bytes or locality.nvlink_service_ps:
        raise ValueError(
            "request attribution of NVLink service requires the per-artifact "
            "medium projection: publish local_phase_service_ps, "
            "base_phase_latency_ps and local_phase_medium"
        )
    local = []
    base = []
    medium = []
    for composed_ps, fabric_ps in zip(
        locality.composed_phase_service_ps,
        locality.fabric_phase_service_ps,
        strict=True,
    ):
        if fabric_ps:
            local.append(0)
            base.append(composed_ps - fabric_ps)
            medium.append(NVLINK_MEDIUM)
        else:
            #: The one corner where this fallback would disagree with the
            #: sink's own projection is a collective artifact with zero fabric
            #: service and a nonzero base latency: it lands in kernel here and
            #: in collective there. It is unreachable today, because a base
            #: latency requires an explicitly selected collective profile and
            #: no accepted reducer path carries one, and it stays unreachable
            #: as long as an outcome that publishes base latencies also
            #: publishes the medium projection above.
            local.append(composed_ps)
            base.append(0)
            medium.append(GPU_COMPUTE_MEDIUM)
    return tuple(local), tuple(base), tuple(medium)


def attribute_step_detail(
    result: StepResult,
    locality: StepLocalityOutcome | None,
) -> StepAttribution:
    """Partition one packet-level step's makespan over its executed artifacts.

    Each executed artifact contributes its semantic base latency to
    ``collective_base_ps`` and its realized service to exactly one resource
    component: the medium whose own service equals the composed maximum. The
    partition is complete and disjoint by construction, so the returned totals
    equal ``result.step_latency_ps`` with no unattributed remainder.

    ``locality is None`` describes a step the sink did not simulate, which the
    adapter settles with its own fallback latency. That latency is charged to
    ``control_ps`` because no artifact accounts for it.
    """

    if not isinstance(result, StepResult):
        raise TypeError("result must be a StepResult")
    if locality is None:
        return StepAttribution(
            attribution=LatencyAttribution(control_ps=result.step_latency_ps),
            media=MediumAttribution(control_ps=result.step_latency_ps),
            masked=MaskedMediumService(),
        )
    if not isinstance(locality, StepLocalityOutcome):
        raise TypeError("locality must be a StepLocalityOutcome or None")
    if locality.step_index != result.step_index:
        raise ValueError("locality outcome belongs to another step")
    composed = locality.composed_phase_service_ps
    fabric = locality.fabric_phase_service_ps
    if len(composed) != len(fabric):
        raise ValueError("composed and fabric artifact services disagree in length")
    published = (
        locality.local_phase_service_ps,
        locality.base_phase_latency_ps,
        locality.local_phase_medium,
    )
    if any(published):
        local, base, medium = published
        if any(len(entry) != len(composed) for entry in published):
            raise ValueError(
                "the per-artifact medium projection disagrees in length with the "
                "executed artifact services"
            )
    else:
        local, base, medium = _legacy_medium_projection(locality)

    kernel_ps = 0
    nvlink_ps = 0
    fabric_owned_ps = 0
    co_critical_ps = 0
    base_ps = 0
    masked_nvlink_ps = 0
    masked_fabric_ps = 0
    declared_nvlink_service_ps = 0
    for index, artifact in enumerate(
        zip(composed, fabric, local, base, medium, strict=True)
    ):
        composed_ps, fabric_ps, local_ps, artifact_base_ps, artifact_medium = artifact
        ownership = _artifact_ownership(
            index=index,
            composed_ps=composed_ps,
            fabric_ps=fabric_ps,
            local_ps=local_ps,
            base_ps=artifact_base_ps,
            medium=artifact_medium,
        )
        base_ps += ownership.base_ps
        masked_nvlink_ps += ownership.masked_nvlink_ps
        masked_fabric_ps += ownership.masked_fabric_ps
        declared_nvlink_service_ps += ownership.nvlink_service_ps
        if ownership.owner == KERNEL_OWNER:
            kernel_ps += ownership.owned_ps
        elif ownership.owner == NVLINK_OWNER:
            nvlink_ps += ownership.owned_ps
        elif ownership.owner == FABRIC_OWNER:
            fabric_owned_ps += ownership.owned_ps
        elif ownership.owner == CO_CRITICAL_OWNER:
            co_critical_ps += ownership.owned_ps
    if declared_nvlink_service_ps != locality.nvlink_service_ps:
        raise ValueError(
            "NVLink-owned artifact services do not conserve the step's published "
            "NVLink service"
        )

    attribution = LatencyAttribution(
        kernel_ps=kernel_ps,
        collective_ps=nvlink_ps + fabric_owned_ps + co_critical_ps + base_ps,
    )
    if attribution.total_ps != result.step_latency_ps:
        raise ValueError(
            "executed artifact services do not conserve the step makespan"
        )
    return StepAttribution(
        attribution=attribution,
        media=MediumAttribution(
            kernel_ps=kernel_ps,
            nvlink_ps=nvlink_ps,
            fabric_ps=fabric_owned_ps,
            co_critical_ps=co_critical_ps,
            collective_base_ps=base_ps,
        ),
        masked=MaskedMediumService(
            nvlink_ps=masked_nvlink_ps,
            fabric_ps=masked_fabric_ps,
        ),
    )


def attribute_step(
    result: StepResult,
    locality: StepLocalityOutcome | None,
) -> LatencyAttribution:
    """Return only the coarse partition of :func:`attribute_step_detail`."""

    return attribute_step_detail(result, locality).attribution


@dataclass
class _RequestState:
    arrived_at_ps: int
    accounted_through_ps: int
    first_token_at_ps: int | None = None
    last_token_at_ps: int | None = None
    token_count: int = 0
    inter_token_sum_ps: int = 0
    inter_token_count: int = 0
    pending: LatencyAttribution = field(default_factory=LatencyAttribution)
    pending_media: MediumAttribution = field(default_factory=MediumAttribution)
    ttft_attribution: LatencyAttribution = field(default_factory=LatencyAttribution)
    decode_attribution: LatencyAttribution = field(default_factory=LatencyAttribution)
    ttft_media: MediumAttribution = field(default_factory=MediumAttribution)
    decode_media: MediumAttribution = field(default_factory=MediumAttribution)
    latest_metric: RequestMetric | None = None


@dataclass(frozen=True)
class RequestLatencyTotals:
    """One request's completed TTFT and TPOT with their conserved partitions.

    ``ttft_media`` and ``decode_media`` resolve the same two intervals to the
    resource that realized each stretch, so a reader can separate NVLink from
    fabric time without either component losing its own name.
    """

    request_id: str
    arrived_at_ps: int
    first_token_at_ps: int
    last_token_at_ps: int
    token_count: int
    ttft_ps: int
    tpot_ps: Fraction | None
    ttft_attribution: LatencyAttribution
    decode_attribution: LatencyAttribution
    ttft_media: MediumAttribution
    decode_media: MediumAttribution

    def __post_init__(self) -> None:
        if self.ttft_attribution.total_ps != self.ttft_ps:
            raise ValueError("TTFT attribution does not conserve TTFT")
        expected_decode_ps = self.last_token_at_ps - self.first_token_at_ps
        if self.decode_attribution.total_ps != expected_decode_ps:
            raise ValueError("decode attribution does not conserve the decode span")
        for name, attribution, media in (
            ("TTFT", self.ttft_attribution, self.ttft_media),
            ("decode", self.decode_attribution, self.decode_media),
        ):
            if not isinstance(media, MediumAttribution):
                raise TypeError(f"{name} media must be a MediumAttribution")
            if media.total_ps != attribution.total_ps:
                raise ValueError(
                    f"{name} medium partition does not conserve the {name} span"
                )
            if (
                media.queue_ps != attribution.queue_ps
                or media.kernel_ps != attribution.kernel_ps
                or media.control_ps != attribution.control_ps
                or media.collective_ps != attribution.collective_ps
            ):
                raise ValueError(
                    f"{name} medium components do not roll up to the {name} partition"
                )
        if self.tpot_ps is None:
            if self.token_count != 1:
                raise ValueError("only a single-token request may omit TPOT")
        elif self.tpot_ps * (self.token_count - 1) != expected_decode_ps:
            raise ValueError("TPOT does not reproduce the attributed decode span")


class HtsimRequestMetricReducer:
    """Project packet-level step outcomes into per-request TTFT and TPOT.

    ``arrivals`` are the declared framework-entry timestamps that the admission
    gate released against, so a request's TTFT is queue plus service and never
    starts before the request exists. Feeding the study's own step-completion
    times without those arrivals is what produced negative TTFT in the earlier
    unregistered run.
    """

    def __init__(self, arrivals: Mapping[str, int]) -> None:
        parsed: dict[str, int] = {}
        for request_id, arrived_at_ps in arrivals.items():
            if not isinstance(request_id, str) or not request_id.strip():
                raise ValueError("arrival keys must be nonblank request identities")
            if isinstance(arrived_at_ps, bool) or not isinstance(arrived_at_ps, int):
                raise TypeError(f"arrival {request_id!r} must be an integer")
            if arrived_at_ps < 0:
                raise ValueError(f"arrival {request_id!r} must be nonnegative")
            parsed[request_id] = arrived_at_ps
        self._arrivals = parsed
        self._requests: dict[str, _RequestState] = {}
        self._consumed_step_indices: set[int] = set()

    @property
    def latest_request_metrics(self) -> tuple[RequestMetric, ...]:
        """The most recent completed metric for each observed request."""

        return tuple(
            state.latest_metric
            for state in self._requests.values()
            if state.latest_metric is not None
        )

    def totals(self) -> tuple[RequestLatencyTotals, ...]:
        """Completed per-request TTFT and TPOT with their conserved partitions."""

        rows = []
        for request_id, state in self._requests.items():
            if state.first_token_at_ps is None or state.last_token_at_ps is None:
                continue
            rows.append(
                RequestLatencyTotals(
                    request_id=request_id,
                    arrived_at_ps=state.arrived_at_ps,
                    first_token_at_ps=state.first_token_at_ps,
                    last_token_at_ps=state.last_token_at_ps,
                    token_count=state.token_count,
                    ttft_ps=state.first_token_at_ps - state.arrived_at_ps,
                    tpot_ps=(
                        Fraction(state.inter_token_sum_ps, state.inter_token_count)
                        if state.inter_token_count
                        else None
                    ),
                    ttft_attribution=state.ttft_attribution,
                    decode_attribution=state.decode_attribution,
                    ttft_media=state.ttft_media,
                    decode_media=state.decode_media,
                )
            )
        return tuple(rows)

    def consume(
        self,
        record: StepRecord,
        result: StepResult,
        locality: StepLocalityOutcome | None,
    ) -> tuple[RequestMetric, ...]:
        """Reduce one executed step and commit its request-metric history."""

        if not isinstance(record, StepRecord):
            raise TypeError("record must be a StepRecord")
        if not isinstance(result, StepResult):
            raise TypeError("result must be a StepResult")
        if result.step_index != record.step_index:
            raise ValueError("StepResult belongs to another StepRecord")
        if record.step_index in self._consumed_step_indices:
            raise ValueError(f"step index has already been reduced: {record.step_index}")
        released_at_ps = record.virtual_time_ps
        if result.completed_at_ps != released_at_ps + result.step_latency_ps:
            raise ValueError("StepResult completion disagrees with its own makespan")
        step = attribute_step_detail(result, locality)
        step_attribution = step.attribution
        step_media = step.media

        sampled = sampled_request_ids(record)
        states = {
            request_id: replace(state)
            for request_id, state in self._requests.items()
        }
        metrics: list[RequestMetric] = []
        for scheduled in record.scheduled:
            request_id = scheduled.request_id
            state = states.get(request_id)
            if state is None:
                arrived_at_ps = self._arrivals.get(request_id)
                if arrived_at_ps is None:
                    raise ValueError(
                        f"scheduled request {request_id!r} has no declared arrival"
                    )
                if arrived_at_ps > released_at_ps:
                    raise ValueError(
                        f"scheduled request {request_id!r} predates its arrival at "
                        f"{arrived_at_ps} ps"
                    )
                state = _RequestState(
                    arrived_at_ps=arrived_at_ps,
                    accounted_through_ps=arrived_at_ps,
                )
                states[request_id] = state
            if released_at_ps < state.accounted_through_ps:
                raise ValueError(f"request {request_id!r} would move backward in time")
            queue_ps = released_at_ps - state.accounted_through_ps
            interval = (
                state.pending + LatencyAttribution(queue_ps=queue_ps) + step_attribution
            )
            interval_media = (
                state.pending_media + MediumAttribution(queue_ps=queue_ps) + step_media
            )
            state.accounted_through_ps = result.completed_at_ps

            if request_id not in sampled:
                state.pending = interval
                state.pending_media = interval_media
                continue

            if state.first_token_at_ps is None:
                latency_ps = result.completed_at_ps - state.arrived_at_ps
                state.first_token_at_ps = result.completed_at_ps
                state.ttft_attribution = interval
                state.ttft_media = interval_media
                ttft_ps = latency_ps
            else:
                assert state.last_token_at_ps is not None
                latency_ps = result.completed_at_ps - state.last_token_at_ps
                ttft_ps = state.first_token_at_ps - state.arrived_at_ps
                state.inter_token_sum_ps += latency_ps
                state.inter_token_count += 1
                state.decode_attribution = state.decode_attribution + interval
                state.decode_media = state.decode_media + interval_media
            if interval.total_ps != latency_ps:
                raise ValueError(
                    f"request {request_id!r} interval attribution does not conserve "
                    "elapsed time"
                )
            if interval_media.total_ps != latency_ps:
                raise ValueError(
                    f"request {request_id!r} medium partition does not conserve "
                    "elapsed time"
                )
            state.token_count += 1
            state.last_token_at_ps = result.completed_at_ps
            metric = RequestMetric(
                request_id=request_id,
                phase=scheduled.phase,
                token_index=state.token_count,
                completed_at_ps=result.completed_at_ps,
                latency_ps=latency_ps,
                ttft_ps=ttft_ps,
                tpot_ps=(
                    Fraction(state.inter_token_sum_ps, state.inter_token_count)
                    if state.inter_token_count
                    else None
                ),
                attribution=interval,
                additive_visit_totals=_NO_VISIT_TOTALS,
            )
            state.latest_metric = metric
            state.pending = LatencyAttribution()
            state.pending_media = MediumAttribution()
            metrics.append(metric)

        self._requests = states
        self._consumed_step_indices.add(record.step_index)
        return tuple(metrics)


__all__ = [
    "CO_CRITICAL_OWNER",
    "FABRIC_OWNER",
    "KERNEL_OWNER",
    "NVLINK_OWNER",
    "UNSERVICED_OWNER",
    "HtsimRequestMetricReducer",
    "MaskedMediumService",
    "MediumAttribution",
    "RequestLatencyTotals",
    "StepAttribution",
    "attribute_step",
    "attribute_step_detail",
]
