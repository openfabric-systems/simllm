"""Stamped closed-form capacity estimates for deployment planning."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from fractions import Fraction
from pathlib import PurePosixPath

from simllm.calibration.batch_service_surface import (
    BatchServicePoint,
    interpolate_batch_service_ps,
)
from simllm.compute import GpuSpec, KernelSpec, RooflineProvider
from simllm.core._wire import (
    _array,
    _enum_value,
    _fields,
    _integer,
    _object,
    _require_tuple,
    _string,
)
from simllm.deploy.candidate import (
    PIPELINE_PARALLEL_UNPRICED,
    DeploymentCandidate,
    PoolSpec,
    candidate_key,
)

DEPLOYMENT_ESTIMATE_SCHEMA = "simllm-deployment-estimate-v1"
PICOSECONDS_PER_SECOND = 1_000_000_000_000

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_FROZEN_PARTITIONS = {
    8: (1, 0),
    16: (2, 1),
    72: (9, 8),
}


class EvidenceClass(str, Enum):
    """Origin class for one duration used by the planning estimator."""

    MEASURED = "MEASURED"
    MEASURED_EXTERNAL = "MEASURED-EXTERNAL"
    ROOFLINE = "ROOFLINE"
    DECLARED = "DECLARED"
    SIM_DERIVED = "SIM-DERIVED"


class EstimatorClass(str, Enum):
    """Model class carried by every deployment estimate stamp."""

    ESTIMATE = "ESTIMATE"


def _positive_integer(value: object, path: str) -> int:
    return _integer(value, path, minimum=1)


def _nonnegative_integer(value: object, path: str) -> int:
    return _integer(value, path, nonnegative=True)


def _positive_number(value: object, path: str) -> float:
    if type(value) not in (int, float):
        raise TypeError(f"{path}: expected a finite positive number")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{path}: expected a finite positive number")
    return result


def _sha256(value: object, path: str) -> str:
    digest = _string(value, path)
    if _SHA256_RE.fullmatch(digest) is None:
        raise ValueError(f"{path}: expected 64 lowercase hexadecimal digits")
    return digest


def _relative_record_path(value: object, path: str) -> str:
    text = _string(value, path)
    pure = PurePosixPath(text)
    if pure.is_absolute() or pure == PurePosixPath(".") or ".." in pure.parts:
        raise ValueError(f"{path}: expected a repository-relative tracked-record path")
    if "\\" in text:
        raise ValueError(f"{path}: expected a POSIX repository-relative path")
    return text


@dataclass(frozen=True, slots=True)
class TermEstimate:
    """One duration together with the evidence that identifies it."""

    duration_ps: int
    evidence: EvidenceClass
    source: str

    def __post_init__(self) -> None:
        _nonnegative_integer(self.duration_ps, "term.duration_ps")
        if not isinstance(self.evidence, EvidenceClass):
            raise TypeError("term.evidence: expected EvidenceClass")
        _string(self.source, "term.source")


@dataclass(frozen=True, slots=True)
class NamedTermEstimate:
    """One uniquely named term embedded in an estimate stamp."""

    name: str
    estimate: TermEstimate

    def __post_init__(self) -> None:
        _string(self.name, "stamp.term.name")
        if not isinstance(self.estimate, TermEstimate):
            raise TypeError("stamp.term.estimate: expected TermEstimate")
        self.estimate.__post_init__()


@dataclass(frozen=True, slots=True)
class EstimateStamp:
    """Strict identity and evidence ledger for one analytical estimate."""

    candidate_key: str
    terms: tuple[NamedTermEstimate, ...]
    estimator_class: EstimatorClass = EstimatorClass.ESTIMATE

    def __post_init__(self) -> None:
        _sha256(self.candidate_key, "estimate.candidate_key")
        if not isinstance(self.estimator_class, EstimatorClass):
            raise TypeError("estimate.estimator_class: expected EstimatorClass")
        if self.estimator_class is not EstimatorClass.ESTIMATE:
            raise ValueError("estimate.estimator_class: v1 supports only ESTIMATE")
        terms = _require_tuple(self.terms, "estimate.terms")
        names: list[str] = []
        for index, term in enumerate(terms):
            if not isinstance(term, NamedTermEstimate):
                raise TypeError(
                    f"estimate.terms[{index}]: expected NamedTermEstimate"
                )
            term.__post_init__()
            names.append(term.name)
        if not terms:
            raise ValueError("estimate.terms: must contain at least one evidence term")
        if len(names) != len(set(names)):
            raise ValueError("estimate.terms: contains duplicate term names")

    @property
    def schema(self) -> str:
        return DEPLOYMENT_ESTIMATE_SCHEMA

    @property
    def consumes_sim_derived(self) -> bool:
        return any(
            term.estimate.evidence is EvidenceClass.SIM_DERIVED
            for term in self.terms
        )


@dataclass(frozen=True, slots=True)
class StepEstimate:
    """One composed service step plus its evidence-complete stamp."""

    kernel_floor: TermEstimate
    fabric_floor: TermEstimate
    intra_floor: TermEstimate
    fabric_excess_ps: int
    intra_excess_ps: int
    analytical_step_ps: int
    step_ps: int
    stamp: EstimateStamp
    batch_service: TermEstimate | None = None
    handoff: TermEstimate | None = None

    def __post_init__(self) -> None:
        for name in ("kernel_floor", "fabric_floor", "intra_floor"):
            term = getattr(self, name)
            if not isinstance(term, TermEstimate):
                raise TypeError(f"step.{name}: expected TermEstimate")
            term.__post_init__()
        for name in (
            "fabric_excess_ps",
            "intra_excess_ps",
            "analytical_step_ps",
            "step_ps",
        ):
            _nonnegative_integer(getattr(self, name), f"step.{name}")
        if self.analytical_step_ps != max(
            self.kernel_floor.duration_ps,
            self.fabric_floor.duration_ps,
            self.intra_floor.duration_ps,
        ):
            raise ValueError("step.analytical_step_ps: must equal the maximum floor")
        if self.step_ps < self.analytical_step_ps:
            raise ValueError("step.step_ps: cannot be below the analytical floor")
        if not isinstance(self.stamp, EstimateStamp):
            raise TypeError("step.stamp: expected EstimateStamp")
        self.stamp.__post_init__()
        for name in ("batch_service", "handoff"):
            term = getattr(self, name)
            if term is not None:
                if not isinstance(term, TermEstimate):
                    raise TypeError(f"step.{name}: expected TermEstimate or None")
                term.__post_init__()

    @property
    def request_ps(self) -> int:
        """Return service plus a causally subsequent declared handoff."""

        if self.handoff is None:
            return self.step_ps
        return self.step_ps + self.handoff.duration_ps


@dataclass(frozen=True, slots=True)
class ModelWork:
    """Frozen per-batch work projection from one content-addressed inventory."""

    kernel_name: str
    flops_per_batch_item: int
    static_logical_hbm_bytes: int
    dynamic_hbm_bytes_per_batch_item: int
    logical_collective_bytes_per_gpu_per_batch_item: int
    inventory_sha256: str
    source: str

    def __post_init__(self) -> None:
        _string(self.kernel_name, "model_work.kernel_name")
        _positive_integer(
            self.flops_per_batch_item,
            "model_work.flops_per_batch_item",
        )
        _nonnegative_integer(
            self.static_logical_hbm_bytes,
            "model_work.static_logical_hbm_bytes",
        )
        _nonnegative_integer(
            self.dynamic_hbm_bytes_per_batch_item,
            "model_work.dynamic_hbm_bytes_per_batch_item",
        )
        _nonnegative_integer(
            self.logical_collective_bytes_per_gpu_per_batch_item,
            "model_work.logical_collective_bytes_per_gpu_per_batch_item",
        )
        _sha256(self.inventory_sha256, "model_work.inventory_sha256")
        _string(self.source, "model_work.source")


@dataclass(frozen=True, slots=True)
class EnvelopeSpec:
    """One explicitly sourced GPU roofline envelope."""

    device: str
    peak_flops_per_second: int | float
    hbm_bytes_per_second: int | float
    efficiency: int | float
    source: str

    def __post_init__(self) -> None:
        _string(self.device, "envelope.device")
        _positive_number(
            self.peak_flops_per_second,
            "envelope.peak_flops_per_second",
        )
        _positive_number(
            self.hbm_bytes_per_second,
            "envelope.hbm_bytes_per_second",
        )
        efficiency = _positive_number(self.efficiency, "envelope.efficiency")
        if efficiency > 1.0:
            raise ValueError("envelope.efficiency: must be at most 1.0")
        _string(self.source, "envelope.source")

    def gpu_spec(self) -> GpuSpec:
        return GpuSpec(
            name=self.device,
            peak_flops=float(self.peak_flops_per_second),
            mem_bandwidth=float(self.hbm_bytes_per_second),
        )


@dataclass(frozen=True, slots=True)
class SimDerivedTerms:
    """Raw network excess read from one tracked simulation record."""

    fabric_excess_ps: int
    intra_excess_ps: int
    record_path: str
    record_sha256: str

    def __post_init__(self) -> None:
        _nonnegative_integer(
            self.fabric_excess_ps,
            "sim_derived.fabric_excess_ps",
        )
        _nonnegative_integer(
            self.intra_excess_ps,
            "sim_derived.intra_excess_ps",
        )
        _relative_record_path(self.record_path, "sim_derived.record_path")
        _sha256(self.record_sha256, "sim_derived.record_sha256")

    @property
    def source(self) -> str:
        return f"{self.record_path} sha256:{self.record_sha256}"


@dataclass(slots=True)
class EstimatorInputs:
    """All sourced inputs consumed by one or more estimator calls."""

    model_work: ModelWork
    envelopes: Mapping[str, EnvelopeSpec]
    surfaces: tuple[BatchServicePoint, ...] | None = None
    handoff_ps: int | None = None
    handoff_source: str | None = None
    sim_derived: SimDerivedTerms | None = None
    prefill_service: TermEstimate | None = None
    surface_evidence: EvidenceClass | None = None
    surface_source: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.model_work, ModelWork):
            raise TypeError("inputs.model_work: expected ModelWork")
        self.model_work.__post_init__()
        if not isinstance(self.envelopes, Mapping):
            raise TypeError("inputs.envelopes: expected a mapping")
        for key, envelope in self.envelopes.items():
            _string(key, "inputs.envelopes key")
            if not isinstance(envelope, EnvelopeSpec):
                raise TypeError(
                    f"inputs.envelopes[{key!r}]: expected EnvelopeSpec"
                )
            envelope.__post_init__()
            if key != envelope.device:
                raise ValueError(
                    f"inputs.envelopes[{key!r}]: key must equal envelope.device"
                )
        if self.surfaces is not None:
            points = _require_tuple(self.surfaces, "inputs.surfaces")
            if not points:
                raise ValueError("inputs.surfaces: must not be empty when supplied")
            for index, point in enumerate(points):
                if not isinstance(point, BatchServicePoint):
                    raise TypeError(
                        f"inputs.surfaces[{index}]: expected BatchServicePoint"
                    )
                point.__post_init__()
        if (self.handoff_ps is None) != (self.handoff_source is None):
            raise ValueError(
                "inputs.handoff_ps and inputs.handoff_source must be supplied together"
            )
        if self.handoff_ps is not None:
            _nonnegative_integer(self.handoff_ps, "inputs.handoff_ps")
            _string(self.handoff_source, "inputs.handoff_source")
        if self.sim_derived is not None:
            if not isinstance(self.sim_derived, SimDerivedTerms):
                raise TypeError("inputs.sim_derived: expected SimDerivedTerms")
            self.sim_derived.__post_init__()
        if self.prefill_service is not None:
            if not isinstance(self.prefill_service, TermEstimate):
                raise TypeError("inputs.prefill_service: expected TermEstimate")
            self.prefill_service.__post_init__()
        if self.surfaces is not None and self.surface_evidence is None:
            raise ValueError(
                "inputs.surface_evidence: state the evidence class of the "
                "supplied batch-service points explicitly (MEASURED, "
                "MEASURED-EXTERNAL or DECLARED); nothing defaults"
            )
        if self.surfaces is None and self.surface_evidence is not None:
            raise ValueError(
                "inputs.surface_evidence: supplied without inputs.surfaces"
            )
        if self.surface_evidence is not None:
            if not isinstance(self.surface_evidence, EvidenceClass):
                raise TypeError("inputs.surface_evidence: expected EvidenceClass")
            if self.surface_evidence not in {
                EvidenceClass.MEASURED,
                EvidenceClass.MEASURED_EXTERNAL,
                EvidenceClass.DECLARED,
            }:
                raise ValueError(
                    "inputs.surface_evidence: batch service must be MEASURED, "
                    "MEASURED-EXTERNAL or DECLARED"
                )
        if (
            self.surfaces is not None
            and self.surface_evidence is EvidenceClass.MEASURED_EXTERNAL
            and any(
                point.evidence_class != EvidenceClass.MEASURED_EXTERNAL.value
                for point in self.surfaces
            )
        ):
            raise ValueError(
                "inputs.surfaces: MEASURED-EXTERNAL evidence requires every "
                "batch-service point to carry MEASURED-EXTERNAL"
            )
        if self.surface_source is not None:
            _string(self.surface_source, "inputs.surface_source")
        if (
            self.surfaces is not None
            and self.surface_evidence is EvidenceClass.DECLARED
            and self.surface_source is None
        ):
            raise ValueError(
                "inputs.surface_source: required for a DECLARED batch-service surface"
            )


@dataclass(frozen=True, slots=True)
class PoolRateMatch:
    """Required and configured capacity for one declared role pool."""

    role: str
    configured_engines: int
    required_engines: int
    utilization: Fraction

    def __post_init__(self) -> None:
        _string(self.role, "pool_match.role")
        _positive_integer(
            self.configured_engines,
            "pool_match.configured_engines",
        )
        _positive_integer(self.required_engines, "pool_match.required_engines")
        if not isinstance(self.utilization, Fraction):
            raise TypeError("pool_match.utilization: expected Fraction")
        if self.utilization < 0:
            raise ValueError("pool_match.utilization: must be nonnegative")


@dataclass(frozen=True, slots=True)
class RateMatchReport:
    """Exact steady-state capacity and service-level membership."""

    required_prefill_engines: int
    required_decode_engines: int
    prefill_capacity_requests_per_second: Fraction
    decode_capacity_requests_per_second: Fraction
    prefill_utilization: Fraction
    decode_utilization: Fraction
    pool_matches: tuple[PoolRateMatch, ...]
    sla_pass: bool
    stamp: EstimateStamp

    def __post_init__(self) -> None:
        _positive_integer(
            self.required_prefill_engines,
            "rate_match.required_prefill_engines",
        )
        _positive_integer(
            self.required_decode_engines,
            "rate_match.required_decode_engines",
        )
        for name in (
            "prefill_capacity_requests_per_second",
            "decode_capacity_requests_per_second",
            "prefill_utilization",
            "decode_utilization",
        ):
            value = getattr(self, name)
            if not isinstance(value, Fraction):
                raise TypeError(f"rate_match.{name}: expected Fraction")
            if value < 0:
                raise ValueError(f"rate_match.{name}: must be nonnegative")
        matches = _require_tuple(self.pool_matches, "rate_match.pool_matches")
        if not matches:
            raise ValueError("rate_match.pool_matches: must not be empty")
        for index, match in enumerate(matches):
            if not isinstance(match, PoolRateMatch):
                raise TypeError(
                    f"rate_match.pool_matches[{index}]: expected PoolRateMatch"
                )
            match.__post_init__()
        if type(self.sla_pass) is not bool:
            raise TypeError("rate_match.sla_pass: expected a boolean")
        if not isinstance(self.stamp, EstimateStamp):
            raise TypeError("rate_match.stamp: expected EstimateStamp")
        self.stamp.__post_init__()


def _term_to_json(term: TermEstimate) -> dict[str, object]:
    term.__post_init__()
    return {
        "duration_ps": term.duration_ps,
        "evidence": term.evidence.value,
        "source": term.source,
    }


def estimate_stamp_to_json(stamp: EstimateStamp) -> dict[str, object]:
    """Return the strict schema-tagged wire object for an estimate stamp."""

    if not isinstance(stamp, EstimateStamp):
        raise TypeError("estimate: expected EstimateStamp")
    stamp.__post_init__()
    return {
        "schema": DEPLOYMENT_ESTIMATE_SCHEMA,
        "candidate_key": stamp.candidate_key,
        "estimator_class": stamp.estimator_class.value,
        "terms": [
            {"name": term.name, **_term_to_json(term.estimate)}
            for term in stamp.terms
        ],
    }


def _term_from_json(value: object, path: str) -> TermEstimate:
    payload = _object(value, path)
    _fields(
        payload,
        path,
        required={"name", "duration_ps", "evidence", "source"},
    )
    return TermEstimate(
        duration_ps=_integer(
            payload["duration_ps"],
            f"{path}.duration_ps",
            nonnegative=True,
        ),
        evidence=_enum_value(
            EvidenceClass,
            payload["evidence"],
            f"{path}.evidence",
        ),
        source=_string(payload["source"], f"{path}.source"),
    )


def estimate_stamp_from_json(value: object) -> EstimateStamp:
    """Parse an estimate stamp after checking its schema tag first."""

    payload = _object(value, "estimate")
    schema = _string(payload.get("schema"), "estimate.schema")
    if schema != DEPLOYMENT_ESTIMATE_SCHEMA:
        raise ValueError(
            "estimate.schema: unsupported schema "
            f"{schema!r}; expected {DEPLOYMENT_ESTIMATE_SCHEMA!r}"
        )
    _fields(
        payload,
        "estimate",
        required={"schema", "candidate_key", "estimator_class", "terms"},
    )
    terms: list[NamedTermEstimate] = []
    for index, raw_term in enumerate(_array(payload["terms"], "estimate.terms")):
        path = f"estimate.terms[{index}]"
        term_payload = _object(raw_term, path)
        term = _term_from_json(term_payload, path)
        terms.append(
            NamedTermEstimate(
                name=_string(term_payload["name"], f"{path}.name"),
                estimate=term,
            )
        )
    return EstimateStamp(
        candidate_key=_sha256(payload["candidate_key"], "estimate.candidate_key"),
        estimator_class=_enum_value(
            EstimatorClass,
            payload["estimator_class"],
            "estimate.estimator_class",
        ),
        terms=tuple(terms),
    )


def _ceil_div(numerator: int, denominator: int) -> int:
    if numerator < 0 or denominator <= 0:
        raise ValueError("ceiling division needs a nonnegative numerator and positive denominator")
    return (numerator + denominator - 1) // denominator


def _ceil_fraction(value: Fraction) -> int:
    if value < 0:
        raise ValueError("fraction ceiling needs a nonnegative value")
    return _ceil_div(value.numerator, value.denominator)


def _service_pool(candidate: DeploymentCandidate, role: str) -> PoolSpec:
    explicit = [pool for pool in candidate.pools if pool.role == role]
    combined = [pool for pool in candidate.pools if pool.role == "combined"]
    if explicit and combined:
        raise ValueError(
            f"candidate.pools: {role} service is ambiguous between explicit and combined pools"
        )
    selected = explicit or combined
    if not selected:
        raise ValueError(
            f"candidate.pools: no {role!r} or 'combined' pool is available"
        )
    return selected[0]


def _partition_bytes(pool: PoolSpec, total: int) -> tuple[int, tuple[int, ...]]:
    """Promote deployment_frontier_v1.partition_network_bytes verbatim."""

    if total == 0 and pool.gpus_per_engine in {2, 4, 8}:
        return 0, ()
    shape = _FROZEN_PARTITIONS.get(pool.gpus_per_engine)
    if shape is None:
        raise ValueError(
            "pool.gpus_per_engine: no byte-partition source for width "
            f"{pool.gpus_per_engine}; DEPLOY-4 owns coverage beyond 8, 16 and 72"
        )
    denominator, fan_in = shape
    if fan_in > 0 and total == 0:
        raise ValueError(
            "model_work: zero collective bytes have no partition source at "
            f"width {pool.gpus_per_engine}; supply positive collective bytes "
            "or the width-8 shape (DEPLOY-4 owns wider coverage)"
        )
    local = total // denominator
    remote = total - local
    if fan_in == 0:
        if remote != 0:
            raise AssertionError("a zero-fan-in configuration retained remote bytes")
        flows: tuple[int, ...] = ()
    else:
        quotient, remainder = divmod(remote, fan_in)
        flows = tuple(
            quotient + (index < remainder)
            for index in range(fan_in)
        )
    if local + sum(flows) != total:
        raise AssertionError("logical network byte partition does not conserve")
    if any(payload <= 0 for payload in flows):
        raise AssertionError("the frozen remote split produced an empty flow")
    return local, flows


def _kernel_floor(
    candidate: DeploymentCandidate,
    pool: PoolSpec,
    batch_per_gpu: int,
    inputs: EstimatorInputs,
) -> TermEstimate:
    """Promote deployment_frontier_v1.kernel_service work and rounding."""

    work = inputs.model_work
    if work.inventory_sha256 != candidate.model.inventory_sha256:
        raise ValueError(
            "inputs.model_work: inventory_sha256 does not match candidate.model; "
            "supply the work record selected by the candidate"
        )
    envelope = inputs.envelopes.get(pool.device)
    if envelope is None:
        raise ValueError(
            f"inputs.envelopes: missing sourced envelope for device {pool.device!r}"
        )
    flops = work.flops_per_batch_item * batch_per_gpu
    logical_hbm_bytes = (
        work.static_logical_hbm_bytes
        + work.dynamic_hbm_bytes_per_batch_item * batch_per_gpu
    )
    provider = RooflineProvider(efficiency=float(envelope.efficiency))
    estimate = provider.estimate(
        KernelSpec(
            name=work.kernel_name,
            flops=flops,
            bytes_moved=logical_hbm_bytes,
            config=(("batch_per_gpu", batch_per_gpu),),
        ),
        envelope.gpu_spec(),
    )
    compute_floor_ps = int(
        flops
        / (envelope.peak_flops_per_second * envelope.efficiency)
        * PICOSECONDS_PER_SECOND
    )
    memory_floor_ps = int(
        logical_hbm_bytes
        / (envelope.hbm_bytes_per_second * envelope.efficiency)
        * PICOSECONDS_PER_SECOND
    )
    if estimate.duration_ps != max(compute_floor_ps, memory_floor_ps):
        raise AssertionError(
            "RooflineProvider disagrees with the promoted envelope arithmetic"
        )
    if estimate.bound not in {"compute", "memory"}:
        raise AssertionError("RooflineProvider returned an unexpected bound")
    return TermEstimate(
        duration_ps=estimate.duration_ps,
        evidence=EvidenceClass.ROOFLINE,
        source=(
            f"{work.source} sha256:{work.inventory_sha256}; "
            f"{envelope.source}"
        ),
    )


def _network_floors(
    candidate: DeploymentCandidate,
    pool: PoolSpec,
    batch_per_gpu: int,
    inputs: EstimatorInputs,
) -> tuple[TermEstimate, TermEstimate]:
    total = (
        batch_per_gpu
        * inputs.model_work.logical_collective_bytes_per_gpu_per_batch_item
    )
    local, flows = _partition_bytes(pool, total)

    # Promoted verbatim from deployment_frontier_v1.ideal_network_service.
    max_flow = max(flows, default=0)
    ideal_fabric = (
        max_flow
        * 8
        * PICOSECONDS_PER_SECOND
        // candidate.fabric.inter_node_bits_per_second
    )
    ideal_intra = (
        local
        * PICOSECONDS_PER_SECOND
        // candidate.fabric.intra_node_bytes_per_second
    )
    key = candidate_key(candidate)
    return (
        TermEstimate(
            duration_ps=ideal_fabric,
            evidence=EvidenceClass.DECLARED,
            source=(
                f"candidate:{key}:fabric.inter_node_bits_per_second"
            ),
        ),
        TermEstimate(
            duration_ps=ideal_intra,
            evidence=EvidenceClass.DECLARED,
            source=(
                f"candidate:{key}:fabric.intra_node_bytes_per_second"
            ),
        ),
    )


def _account_step(
    *,
    kernel_floor_ps: int,
    ideal_fabric_wire_ps: int,
    ideal_intra_node_wire_ps: int,
    simulated_fabric_ps: int,
    simulated_intra_node_ps: int,
) -> tuple[int, int]:
    """Promote deployment_frontier_v1.account_step telescoping verbatim."""

    values = (
        kernel_floor_ps,
        ideal_fabric_wire_ps,
        ideal_intra_node_wire_ps,
        simulated_fabric_ps,
        simulated_intra_node_ps,
    )
    if min(values) < 0:
        raise ValueError("step services must be nonnegative")
    analytical = max(values[:3])
    after_inter = max(
        kernel_floor_ps,
        simulated_fabric_ps,
        ideal_intra_node_wire_ps,
    )
    simulated = max(
        kernel_floor_ps,
        simulated_fabric_ps,
        simulated_intra_node_ps,
    )
    inter = after_inter - analytical
    intra = simulated - after_inter
    residual = simulated - analytical - inter - intra
    if residual != 0 or min(inter, intra) < 0:
        raise AssertionError("promoted step accounting did not telescope")
    return analytical, simulated


def _surface_term(inputs: EstimatorInputs, batch_per_gpu: int) -> TermEstimate | None:
    points = inputs.surfaces
    if points is None:
        return None
    duration = interpolate_batch_service_ps(points, batch_per_gpu)
    if inputs.surface_evidence is EvidenceClass.DECLARED:
        assert inputs.surface_source is not None
        source = inputs.surface_source
    else:
        ordered = tuple(sorted(points, key=lambda point: point.batch_size))
        exact = [point for point in ordered if point.batch_size == batch_per_gpu]
        if exact:
            selected = exact
        else:
            lower = max(
                (point for point in ordered if point.batch_size < batch_per_gpu),
                key=lambda point: point.batch_size,
            )
            upper = min(
                (point for point in ordered if point.batch_size > batch_per_gpu),
                key=lambda point: point.batch_size,
            )
            selected = [lower, upper]
        entry_keys = ",".join(point.entry_key_sha256 for point in selected)
        prefix = f"{inputs.surface_source}; " if inputs.surface_source else ""
        source = f"{prefix}batch-service-entry-key-sha256:{entry_keys}"
    return TermEstimate(
        duration_ps=duration,
        evidence=inputs.surface_evidence,
        source=source,
    )


def estimate_decode_step(
    candidate: DeploymentCandidate,
    batch_per_gpu: int,
    inputs: EstimatorInputs,
) -> StepEstimate:
    """Price one decode step through sourced roofline and network forms."""

    if not isinstance(candidate, DeploymentCandidate):
        raise TypeError("candidate: expected DeploymentCandidate")
    candidate.__post_init__()
    batch = _positive_integer(batch_per_gpu, "batch_per_gpu")
    if not isinstance(inputs, EstimatorInputs):
        raise TypeError("inputs: expected EstimatorInputs")
    inputs.__post_init__()
    pool = _service_pool(candidate, "decode")
    if pool.pipeline_parallel > 1:
        raise ValueError(
            "candidate: the decode pool declares pipeline_parallel > 1, which "
            "this rung cannot price (feasibility rejects it as "
            f"{PIPELINE_PARALLEL_UNPRICED!r})"
        )
    batch_service = _surface_term(inputs, batch)
    kernel = (
        batch_service
        if inputs.surface_evidence is EvidenceClass.MEASURED_EXTERNAL
        else _kernel_floor(candidate, pool, batch, inputs)
    )
    fabric, intra = _network_floors(candidate, pool, batch, inputs)
    named_terms = [
        NamedTermEstimate("kernel_floor", kernel),
        NamedTermEstimate("fabric_floor", fabric),
        NamedTermEstimate("intra_floor", intra),
    ]

    fabric_excess = 0
    intra_excess = 0
    if inputs.sim_derived is not None:
        fabric_excess = inputs.sim_derived.fabric_excess_ps
        intra_excess = inputs.sim_derived.intra_excess_ps
        named_terms.extend(
            (
                NamedTermEstimate(
                    "fabric_excess",
                    TermEstimate(
                        fabric_excess,
                        EvidenceClass.SIM_DERIVED,
                        inputs.sim_derived.source,
                    ),
                ),
                NamedTermEstimate(
                    "intra_excess",
                    TermEstimate(
                        intra_excess,
                        EvidenceClass.SIM_DERIVED,
                        inputs.sim_derived.source,
                    ),
                ),
            )
        )

    if (
        batch_service is not None
        and inputs.surface_evidence is not EvidenceClass.MEASURED_EXTERNAL
    ):
        named_terms.append(NamedTermEstimate("batch_service", batch_service))

    analytical, step = _account_step(
        kernel_floor_ps=kernel.duration_ps,
        ideal_fabric_wire_ps=fabric.duration_ps,
        ideal_intra_node_wire_ps=intra.duration_ps,
        simulated_fabric_ps=fabric.duration_ps + fabric_excess,
        simulated_intra_node_ps=intra.duration_ps + intra_excess,
    )
    stamp = EstimateStamp(
        candidate_key=candidate_key(candidate),
        terms=tuple(named_terms),
    )
    return StepEstimate(
        kernel_floor=kernel,
        fabric_floor=fabric,
        intra_floor=intra,
        fabric_excess_ps=fabric_excess,
        intra_excess_ps=intra_excess,
        analytical_step_ps=analytical,
        step_ps=step,
        stamp=stamp,
        batch_service=batch_service,
    )


def estimate_prefill_request(
    candidate: DeploymentCandidate,
    inputs: EstimatorInputs,
) -> StepEstimate:
    """Return a sourced prefill service and its subsequent declared handoff."""

    if not isinstance(candidate, DeploymentCandidate):
        raise TypeError("candidate: expected DeploymentCandidate")
    candidate.__post_init__()
    if not isinstance(inputs, EstimatorInputs):
        raise TypeError("inputs: expected EstimatorInputs")
    inputs.__post_init__()
    _service_pool(candidate, "prefill")
    if inputs.prefill_service is None:
        raise ValueError(
            "inputs.prefill_service: supply a sourced prefill duration; "
            "DEPLOY-5 owns measured prefill service surfaces"
        )
    if inputs.handoff_ps is None or inputs.handoff_source is None:
        raise ValueError(
            "inputs.handoff_ps and inputs.handoff_source: supply the declared "
            "prefill-to-decode handoff"
        )
    key = candidate_key(candidate)
    fabric = TermEstimate(
        0,
        EvidenceClass.DECLARED,
        f"candidate:{key}:prefill service has no concurrent fabric floor",
    )
    intra = TermEstimate(
        0,
        EvidenceClass.DECLARED,
        f"candidate:{key}:prefill service has no concurrent intra-node floor",
    )
    handoff = TermEstimate(
        inputs.handoff_ps,
        EvidenceClass.DECLARED,
        inputs.handoff_source,
    )
    analytical, step = _account_step(
        kernel_floor_ps=inputs.prefill_service.duration_ps,
        ideal_fabric_wire_ps=0,
        ideal_intra_node_wire_ps=0,
        simulated_fabric_ps=0,
        simulated_intra_node_ps=0,
    )
    return StepEstimate(
        kernel_floor=inputs.prefill_service,
        fabric_floor=fabric,
        intra_floor=intra,
        fabric_excess_ps=0,
        intra_excess_ps=0,
        analytical_step_ps=analytical,
        step_ps=step,
        stamp=EstimateStamp(
            candidate_key=key,
            terms=(
                NamedTermEstimate("prefill_service", inputs.prefill_service),
                NamedTermEstimate("fabric_floor", fabric),
                NamedTermEstimate("intra_floor", intra),
                NamedTermEstimate("handoff", handoff),
            ),
        ),
        handoff=handoff,
    )


def decode_capacity_requests_per_second(
    surface_points: tuple[BatchServicePoint, ...],
    *,
    output_tokens: int,
    max_batch: int,
    decode_engines: int,
) -> Fraction:
    """Return exact max-batch decode capacity in requests per second.

    Promote pd_session_load_delay_v1.decode_capacity_requests_per_second
    with parameterized output tokens, max batch and engine count.
    """

    output = _positive_integer(output_tokens, "output_tokens")
    batch = _positive_integer(max_batch, "max_batch")
    engines = _positive_integer(decode_engines, "decode_engines")
    service_ps = interpolate_batch_service_ps(surface_points, batch)
    return Fraction(
        engines * batch * PICOSECONDS_PER_SECOND,
        output * service_ps,
    )


def queue_occupancy(
    surface_points: tuple[BatchServicePoint, ...],
    *,
    offered_load_rps: int,
    output_tokens: int,
    max_batch: int,
    decode_engines: int,
) -> int:
    """Return the exact ceil-clamped deterministic occupancy bucket.

    Promote pd_session_load_delay_v1.predicted_batch_size with
    parameterized output tokens, max batch and engine count.
    """

    load = _positive_integer(offered_load_rps, "offered_load_rps")
    output = _positive_integer(output_tokens, "output_tokens")
    batch = _positive_integer(max_batch, "max_batch")
    engines = _positive_integer(decode_engines, "decode_engines")
    service_ps = interpolate_batch_service_ps(surface_points, batch)
    occupancy = _ceil_div(
        load * output * service_ps,
        engines * PICOSECONDS_PER_SECOND,
    )
    return min(batch, max(1, occupancy))


def queue_delay_ps(
    surface_points: tuple[BatchServicePoint, ...],
    *,
    offered_load_rps: int,
    output_tokens: int,
    max_batch: int,
    decode_engines: int,
    cell_requests: int,
) -> Fraction:
    """Generalize the pd_session_load_delay_v1 exact D/D/c queue form."""

    load = _positive_integer(offered_load_rps, "offered_load_rps")
    output = _positive_integer(output_tokens, "output_tokens")
    batch = _positive_integer(max_batch, "max_batch")
    engines = _positive_integer(decode_engines, "decode_engines")
    requests = _positive_integer(cell_requests, "cell_requests")
    max_service_ps = interpolate_batch_service_ps(surface_points, batch)
    service_interval = Fraction(output * max_service_ps, batch * engines)
    arrival_interval = Fraction(PICOSECONDS_PER_SECOND, load)
    excess = max(Fraction(), service_interval - arrival_interval)
    return Fraction(requests - 1, 2) * excess


def match_pools(
    candidate: DeploymentCandidate,
    *,
    prefill_request_ps: int,
    decode_step_ps: int,
    batch_per_gpu: int,
) -> RateMatchReport:
    """Match exact per-role engine demand to the declared workload point.

    ``sla_pass`` is the conjunction of capacity feasibility (utilization at
    most one) and the declared latency targets: an overloaded pool fails even
    when both latency targets are met, because overload voids any
    steady-state service claim. The caller-supplied durations are stamped
    DECLARED at this boundary regardless of their upstream evidence class;
    carry the originating StepEstimate stamps alongside this report when the
    full provenance chain matters.
    """

    if not isinstance(candidate, DeploymentCandidate):
        raise TypeError("candidate: expected DeploymentCandidate")
    candidate.__post_init__()
    arrival = candidate.workload.arrival_rate_rps
    if arrival is None:
        raise ValueError(
            "candidate.workload.arrival_rate_rps: required for steady-state rate matching"
        )
    prefill_ps = _positive_integer(prefill_request_ps, "prefill_request_ps")
    decode_ps = _positive_integer(decode_step_ps, "decode_step_ps")
    batch = _positive_integer(batch_per_gpu, "batch_per_gpu")
    prefill_pool = _service_pool(candidate, "prefill")
    decode_pool = _service_pool(candidate, "decode")

    prefill_capacity = Fraction(PICOSECONDS_PER_SECOND, prefill_ps)
    decode_capacity = Fraction(
        batch * PICOSECONDS_PER_SECOND,
        candidate.workload.output_tokens * decode_ps,
    )
    required_prefill = _ceil_fraction(Fraction(arrival, 1) / prefill_capacity)
    required_decode = _ceil_fraction(Fraction(arrival, 1) / decode_capacity)
    prefill_utilization = Fraction(arrival, prefill_pool.engines) / prefill_capacity
    decode_utilization = Fraction(arrival, decode_pool.engines) / decode_capacity

    if prefill_pool is decode_pool:
        combined_utilization = prefill_utilization + decode_utilization
        combined_required = _ceil_fraction(
            Fraction(arrival * prefill_ps, PICOSECONDS_PER_SECOND)
            + Fraction(
                arrival * candidate.workload.output_tokens * decode_ps,
                batch * PICOSECONDS_PER_SECOND,
            )
        )
        pool_matches = (
            PoolRateMatch(
                role="combined",
                configured_engines=prefill_pool.engines,
                required_engines=combined_required,
                utilization=combined_utilization,
            ),
        )
    else:
        pool_matches = (
            PoolRateMatch(
                role="prefill",
                configured_engines=prefill_pool.engines,
                required_engines=required_prefill,
                utilization=prefill_utilization,
            ),
            PoolRateMatch(
                role="decode",
                configured_engines=decode_pool.engines,
                required_engines=required_decode,
                utilization=decode_utilization,
            ),
        )

    capacity_pass = all(match.utilization <= 1 for match in pool_matches)
    tpot_pass = (
        candidate.sla.tpot_target_ps is None
        or decode_ps <= candidate.sla.tpot_target_ps
    )
    ttft_pass = (
        candidate.sla.ttft_target_ps is None
        or prefill_ps <= candidate.sla.ttft_target_ps
    )
    prefill_term = TermEstimate(
        prefill_ps,
        EvidenceClass.DECLARED,
        "match_pools argument prefill_request_ps",
    )
    decode_term = TermEstimate(
        decode_ps,
        EvidenceClass.DECLARED,
        "match_pools argument decode_step_ps",
    )
    return RateMatchReport(
        required_prefill_engines=required_prefill,
        required_decode_engines=required_decode,
        prefill_capacity_requests_per_second=prefill_capacity,
        decode_capacity_requests_per_second=decode_capacity,
        prefill_utilization=prefill_utilization,
        decode_utilization=decode_utilization,
        pool_matches=pool_matches,
        sla_pass=capacity_pass and tpot_pass and ttft_pass,
        stamp=EstimateStamp(
            candidate_key=candidate_key(candidate),
            terms=(
                NamedTermEstimate("prefill_request", prefill_term),
                NamedTermEstimate("decode_step", decode_term),
            ),
        ),
    )


__all__ = [
    "DEPLOYMENT_ESTIMATE_SCHEMA",
    "PICOSECONDS_PER_SECOND",
    "EnvelopeSpec",
    "EstimateStamp",
    "EstimatorClass",
    "EstimatorInputs",
    "EvidenceClass",
    "ModelWork",
    "NamedTermEstimate",
    "PoolRateMatch",
    "RateMatchReport",
    "SimDerivedTerms",
    "StepEstimate",
    "TermEstimate",
    "decode_capacity_requests_per_second",
    "estimate_decode_step",
    "estimate_prefill_request",
    "estimate_stamp_from_json",
    "estimate_stamp_to_json",
    "match_pools",
    "queue_delay_ps",
    "queue_occupancy",
]
