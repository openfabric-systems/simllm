"""Bind the pinned external operation database to deployment service terms."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from fractions import Fraction

from simllm.calibration.batch_service_surface import (
    BatchServicePoint,
    interpolate_batch_service_ps,
)
from simllm.calibration.external_composition import (
    COMPOSITION_UNSET,
    EXTERNAL_OPERATION_ADJUSTMENTS,
    ExternalCompositionError,
    ExternalServingComposition,
)
from simllm.calibration.external_db import (
    EXPECTED_MODEL_HASH,
    EXTERNAL_EVIDENCE_CLASS,
    ExternalOperationDatabase,
    ExternalQwen32BPassModel,
)
from simllm.deploy.candidate import DeploymentCandidate, candidate_key
from simllm.deploy.estimator import (
    EstimateStamp,
    EvidenceClass,
    StepEstimate,
    TermEstimate,
    estimate_stamp_to_json,
)

PICOSECONDS_PER_MILLISECOND = 1_000_000_000
AGGREGATE_TPOT_MIXED_STEP_REDUCTION = 3


def _configuration_key(payload: dict[str, object]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class ExternalServiceValue:
    """One externally measured pass composition converted to integer service."""

    configuration_id: str
    phase: str
    tensor_parallel: int
    batch_size: int
    service_ms: float
    total_ms: float
    source: str
    entry_key_sha256: str
    evidence_class: str = EXTERNAL_EVIDENCE_CLASS

    def __post_init__(self) -> None:
        if not self.configuration_id:
            raise ValueError("configuration_id must be non-empty")
        if self.phase not in {"decode", "prefill"}:
            raise ValueError("phase must be decode or prefill")
        if self.tensor_parallel not in {2, 4, 8}:
            raise ValueError("tensor_parallel must be one of 2, 4 or 8")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        for name, value in (("service_ms", self.service_ms), ("total_ms", self.total_ms)):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not self.source:
            raise ValueError("source must be non-empty")
        if len(self.entry_key_sha256) != 64:
            raise ValueError("entry_key_sha256 must be a SHA-256 digest")
        int(self.entry_key_sha256, 16)
        if self.evidence_class != EXTERNAL_EVIDENCE_CLASS:
            raise ValueError(
                f"evidence_class must be {EXTERNAL_EVIDENCE_CLASS}"
            )

    @property
    def service_ms_hex(self) -> str:
        """Return the exact binary64 service spelling."""

        return self.service_ms.hex()

    @property
    def total_ms_hex(self) -> str:
        """Return the exact binary64 pass-total spelling."""

        return self.total_ms.hex()

    @property
    def service_ps(self) -> int:
        """Return the nearest integer picosecond used by deployment records."""

        return round(self.service_ms * PICOSECONDS_PER_MILLISECOND)

    def as_batch_service_point(self) -> BatchServicePoint:
        """Project a decode value into the deployment batch-surface contract."""

        if self.phase != "decode":
            raise ValueError("only decode service can become a batch-service point")
        return BatchServicePoint(
            batch_size=self.batch_size,
            duration_ps=self.service_ps,
            uncertainty_fraction=0.0,
            entry_key_sha256=self.entry_key_sha256,
            evidence_class=EXTERNAL_EVIDENCE_CLASS,
        )

    def as_term(self) -> TermEstimate:
        """Project this value into the deployment evidence ledger."""

        return TermEstimate(
            duration_ps=self.service_ps,
            evidence=EvidenceClass.MEASURED_EXTERNAL,
            source=self.source,
        )


@dataclass(frozen=True, slots=True)
class ExternalAggregatePoint:
    """One co-located in-flight batch composed from measured operation passes."""

    configuration_id: str
    tensor_parallel: int
    batch_size: int
    isl: int
    osl: int
    prefix: int
    context_tokens: int
    mix_steps: int
    tpot_mix_steps: int
    genonly_steps: int
    mix_generation_tokens: int
    genonly_tokens: int
    context_requests: int
    generation_requests: int
    scheduled_tokens: int
    balance_score: float
    mix_step_ms: float
    genonly_step_ms: float
    pure_prefill_step_ms: float
    prefill_passes_per_request: int
    base_prefill_ms: float
    ttft_queue_factor: float
    ttft_queueing_component_ms: float
    ttft_ms: float
    tpot_ms: float
    total_schedule_ms: float
    request_latency_ms: float
    request_rate: float
    tokens_per_second: float
    tokens_per_second_per_gpu: float
    tokens_per_second_per_user: float
    mix_operations_ms: tuple[tuple[str, float], ...]
    genonly_operations_ms: tuple[tuple[str, float], ...]
    source: str
    entry_key_sha256: str
    evidence_class: str = EXTERNAL_EVIDENCE_CLASS

    def __post_init__(self) -> None:
        if not self.configuration_id:
            raise ValueError("configuration_id must be non-empty")
        if self.tensor_parallel not in {2, 4, 8}:
            raise ValueError("tensor_parallel must be one of 2, 4 or 8")
        positive_integers = (
            "batch_size",
            "isl",
            "osl",
            "context_tokens",
            "mix_steps",
            "tpot_mix_steps",
            "genonly_steps",
            "mix_generation_tokens",
            "genonly_tokens",
            "context_requests",
            "generation_requests",
            "scheduled_tokens",
            "prefill_passes_per_request",
        )
        for name in positive_integers:
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.prefix < 0 or self.prefix >= self.isl:
            raise ValueError("prefix must be non-negative and smaller than isl")
        positive_floats = (
            "balance_score",
            "mix_step_ms",
            "genonly_step_ms",
            "pure_prefill_step_ms",
            "base_prefill_ms",
            "ttft_queue_factor",
            "ttft_ms",
            "tpot_ms",
            "total_schedule_ms",
            "request_latency_ms",
            "request_rate",
            "tokens_per_second",
            "tokens_per_second_per_gpu",
            "tokens_per_second_per_user",
        )
        for name in positive_floats:
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if (
            not math.isfinite(self.ttft_queueing_component_ms)
            or self.ttft_queueing_component_ms < 0
        ):
            raise ValueError(
                "ttft_queueing_component_ms must be finite and non-negative"
            )
        for operations in (self.mix_operations_ms, self.genonly_operations_ms):
            if not operations:
                raise ValueError("operation breakdowns must be non-empty")
            for name, value in operations:
                if not name or not math.isfinite(value) or value < 0:
                    raise ValueError("operation breakdown entries must be valid")
        if not self.source:
            raise ValueError("source must be non-empty")
        if len(self.entry_key_sha256) != 64:
            raise ValueError("entry_key_sha256 must be a SHA-256 digest")
        int(self.entry_key_sha256, 16)
        if self.evidence_class != EXTERNAL_EVIDENCE_CLASS:
            raise ValueError(
                f"evidence_class must be {EXTERNAL_EVIDENCE_CLASS}"
            )

    def as_dict(self) -> dict[str, object]:
        """Return the complete deterministic aggregate projection."""

        return {
            field: (
                [[name, value.hex()] for name, value in value]
                if field in {"mix_operations_ms", "genonly_operations_ms"}
                else value.hex()
                if isinstance(value, float)
                else value
            )
            for field, value in (
                (field.name, getattr(self, field.name))
                for field in self.__dataclass_fields__.values()
            )
        }


@dataclass(frozen=True, slots=True, init=False)
class _SelectedService:
    """Immutable configuration and value already admitted by the binding."""

    value: ExternalServiceValue
    configuration_items: tuple[tuple[str, object], ...]

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise TypeError("selected service receipts are issued only by the binding")

    @classmethod
    def _issue(
        cls, value: ExternalServiceValue, configuration: dict[str, object]
    ) -> _SelectedService:
        receipt = object.__new__(cls)
        object.__setattr__(receipt, "value", value)
        object.__setattr__(receipt, "configuration_items", tuple(sorted(configuration.items())))
        return receipt

    @property
    def configuration(self) -> dict[str, object]:
        return dict(self.configuration_items)

    def validate(self, record: ExternalServingComposition) -> None:
        if not isinstance(self.value, ExternalServiceValue):
            raise TypeError("selected value must be ExternalServiceValue")
        self.value.__post_init__()
        if type(self.configuration_items) is not tuple or any(
            type(pair) is not tuple
            or len(pair) != 2
            or type(pair[0]) is not str
            or type(pair[1]) not in {str, int, bool, type(None)}
            for pair in self.configuration_items
        ):
            raise ExternalCompositionError("selected configuration must be immutable scalar pairs")
        config = self.configuration
        if tuple(sorted(config.items())) != self.configuration_items:
            raise ExternalCompositionError(
                "selected configuration has duplicate or unordered members"
            )
        expected_fields = {
            "id",
            "phase",
            "tensor_parallel",
            "batch_size",
            "isl",
            "prefix",
            "latency_correction_scale_hex",
            "gemm_quant_mode",
            "kv_cache_quant_mode",
            "fmha_quant_mode",
            "communication_quant_mode",
        }
        if self.value.phase == "decode":
            expected_fields |= {"osl", "stride"}
        if set(config) != expected_fields:
            raise ExternalCompositionError("selected service has an incomplete configuration")
        for field in ("tensor_parallel", "batch_size", "isl", "prefix", "osl", "stride"):
            if field in config and (
                type(config[field]) is not int or config[field] < (0 if field == "prefix" else 1)
            ):
                raise ExternalCompositionError("selected shape fields must be exact integers")
        if (
            config["id"] != self.value.configuration_id
            or config["phase"] != self.value.phase
            or config["tensor_parallel"] != self.value.tensor_parallel
            or config["batch_size"] != self.value.batch_size
            or config["latency_correction_scale_hex"]
            != record.number(f"{self.value.phase}_latency_correction").hex()
            or (
                config["gemm_quant_mode"],
                config["kv_cache_quant_mode"],
                config["fmha_quant_mode"],
                config["communication_quant_mode"],
            )
            != ("fp8_block", "fp8", "fp8", "half")
        ):
            raise ExternalCompositionError(
                "selected service configuration disagrees with its value"
            )
        expected_key = _configuration_key(
            {
                "external_source": dict(record.record.value["operation_source"]),
                "configuration": config,
                "composition_sha256": record.record_sha256,
            }
        )
        if expected_key != self.value.entry_key_sha256:
            raise ExternalCompositionError(
                "selected service key disagrees with configuration and record"
            )
        applied = (
            ("prefill_latency_correction", *EXTERNAL_OPERATION_ADJUSTMENTS)
            if self.value.phase == "prefill"
            else ("decode_latency_correction", *EXTERNAL_OPERATION_ADJUSTMENTS[:2])
        )
        expected_source = (
            "external-operation-pass:"
            f"slice-sha256:{record.record.value['operation_source']['data_slice_sha256']};"
            f"configuration:{config['id']};service-ms-hex:{self.value.service_ms.hex()};"
            f"entry-key-sha256:{expected_key};composition-sha256:{record.record_sha256};"
            f"composition-applied:{','.join(applied)}"
        )
        if self.value.source != expected_source:
            raise ExternalCompositionError("selected service source disagrees with its value")
        total_service = self.value.total_ms
        if self.value.phase == "decode":
            if config["osl"] <= 1:
                raise ExternalCompositionError("decode selection needs more than one output token")
            total_service /= config["osl"] - 1
        if total_service.hex() != self.value.service_ms_hex:
            raise ExternalCompositionError("selected pass total disagrees with its service")


@dataclass(frozen=True, slots=True)
class ExternalAutoscaledTtft:
    """External first-token heuristic with its unrounded isolated input."""

    selection: _SelectedService
    composition: ExternalServingComposition

    def __post_init__(self) -> None:
        if not isinstance(self.composition, ExternalServingComposition) or not isinstance(
            self.selection, _SelectedService
        ):
            raise TypeError("autoscale requires an immutable composition and selected service")
        self.selection.validate(self.composition)
        if self.selection.value.phase != "prefill":
            raise ExternalCompositionError("autoscale requires isolated prefill service")

    @property
    def isolated_prefill(self) -> ExternalServiceValue:
        return self.selection.value

    @property
    def factor_hex(self) -> str:
        return self.composition.number("autoscale_ttft_correction").hex()

    @property
    def composition_sha256(self) -> str:
        return self.composition.record_sha256

    @property
    def latency_ms(self) -> float:
        return self.isolated_prefill.service_ms * float.fromhex(self.factor_hex)

    def as_term(self) -> TermEstimate:
        return TermEstimate(
            round(self.latency_ms * PICOSECONDS_PER_MILLISECOND),
            EvidenceClass.MEASURED_EXTERNAL,
            f"external-first-token-heuristic;factor-hex:{self.factor_hex};{self.isolated_prefill.source}",
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "composition_sha256": self.composition_sha256,
            "isolated_service_key_sha256": self.isolated_prefill.entry_key_sha256,
            "isolated_service_ms_hex": self.isolated_prefill.service_ms_hex,
            "factor_hex": self.factor_hex,
            "latency_ms_hex": self.latency_ms.hex(),
            "latency_ps": self.as_term().duration_ps,
            "source": self.as_term().source,
            "scope": "external-autoscale-heuristic-excluding-handoff",
        }


@dataclass(frozen=True, slots=True)
class ExternalDisaggregatedCapacity:
    """Read-only pool capacities joined to their original estimator stamps."""

    candidate: DeploymentCandidate
    prefill: StepEstimate
    decode: StepEstimate
    batch_size: int
    prefix: int
    stride: int
    selections: tuple[_SelectedService, ...]
    composition: ExternalServingComposition

    def __post_init__(self) -> None:
        _validate_capacity_projection(self)

    @property
    def prefill_engines(self) -> int:
        return next(pool.engines for pool in self.candidate.pools if pool.role == "prefill")

    @property
    def decode_engines(self) -> int:
        return next(pool.engines for pool in self.candidate.pools if pool.role == "decode")

    @property
    def output_tokens(self) -> int:
        return self.candidate.workload.output_tokens

    @property
    def used_gpus(self) -> int:
        return sum(pool.engines * pool.gpus_per_engine for pool in self.candidate.pools)

    @property
    def prefill_rate_hex(self) -> str:
        return self.composition.number("prefill_rate_matching_degradation").hex()

    @property
    def decode_rate_hex(self) -> str:
        return self.composition.number("decode_rate_matching_degradation").hex()

    @property
    def composition_sha256(self) -> str:
        return self.composition.record_sha256

    @property
    def service_keys(self) -> tuple[str, ...]:
        return tuple(selection.value.entry_key_sha256 for selection in self.selections)

    @property
    def prefill_requests_per_second(self) -> Fraction:
        return Fraction(
            self.prefill_engines * 10**12, self.prefill.request_ps
        ) * Fraction.from_float(float.fromhex(self.prefill_rate_hex))

    @property
    def decode_requests_per_second(self) -> Fraction:
        return Fraction(
            self.decode_engines * self.batch_size * 10**12, self.output_tokens * self.decode.step_ps
        ) * Fraction.from_float(float.fromhex(self.decode_rate_hex))

    @property
    def request_capacity(self) -> Fraction:
        return min(self.prefill_requests_per_second, self.decode_requests_per_second)

    @property
    def capacity_limiter(self) -> str:
        return (
            "prefill"
            if self.prefill_requests_per_second <= self.decode_requests_per_second
            else "decode"
        )

    @property
    def tokens_per_second_per_user(self) -> Fraction:
        return Fraction(10**12, self.decode.step_ps)

    @property
    def tokens_per_second_per_gpu(self) -> Fraction:
        return self.request_capacity * self.output_tokens / self.used_gpus

    def as_dict(self) -> dict[str, object]:
        def fraction(value: Fraction) -> dict[str, int]:
            return {"numerator": value.numerator, "denominator": value.denominator}

        return {
            "composition_sha256": self.composition_sha256,
            "service_keys": list(self.service_keys),
            "prefill_stamp": estimate_stamp_to_json(self.prefill.stamp),
            "decode_stamp": estimate_stamp_to_json(self.decode.stamp),
            "prefill_request_ps": self.prefill.request_ps,
            "decode_step_ps": self.decode.step_ps,
            "prefix": self.prefix,
            "stride": self.stride,
            "prefill_engines": self.prefill_engines,
            "decode_engines": self.decode_engines,
            "batch_size": self.batch_size,
            "output_tokens": self.output_tokens,
            "used_gpus": self.used_gpus,
            "prefill_rate_hex": self.prefill_rate_hex,
            "decode_rate_hex": self.decode_rate_hex,
            "prefill_requests_per_second": fraction(self.prefill_requests_per_second),
            "decode_requests_per_second": fraction(self.decode_requests_per_second),
            "request_capacity": fraction(self.request_capacity),
            "capacity_limiter": self.capacity_limiter,
            "tokens_per_second_per_user": fraction(self.tokens_per_second_per_user),
            "tokens_per_second_per_gpu": fraction(self.tokens_per_second_per_gpu),
        }


def _validate_capacity_projection(projection: ExternalDisaggregatedCapacity) -> None:
    """Join immutable selected prices, record factors and estimator inputs."""

    candidate, prefill, decode = projection.candidate, projection.prefill, projection.decode
    batch_size, prefix, stride = projection.batch_size, projection.prefix, projection.stride
    record = projection.composition
    if not isinstance(record, ExternalServingComposition):
        raise TypeError("capacity requires an immutable composition")
    if type(projection.selections) is not tuple or len(projection.selections) not in {2, 3}:
        raise ExternalCompositionError("capacity requires one prefill and one or two decode prices")
    services = {}
    for index, selection in enumerate(projection.selections):
        if not isinstance(selection, _SelectedService):
            raise TypeError("capacity selections must be immutable selected services")
        selection.validate(record)
        if selection.value.phase != ("prefill" if index == 0 else "decode"):
            raise ExternalCompositionError("capacity selected service phase mismatch")
        key = selection.value.entry_key_sha256
        if key in services:
            raise ExternalCompositionError("capacity has a duplicate selected service")
        services[key] = selection.value, selection.configuration

    def retained(key: str, phase: str) -> tuple[ExternalServiceValue, dict[str, object]]:
        if key not in services or services[key][0].phase != phase:
            raise ExternalCompositionError("capacity source is not a selected service")
        return services[key]

    if not isinstance(candidate, DeploymentCandidate):
        raise TypeError("candidate must be DeploymentCandidate")
    candidate.__post_init__()
    for name, value, minimum in (
        ("batch_size", batch_size, 1),
        ("prefix", prefix, 0),
        ("stride", stride, 1),
    ):
        if type(value) is not int or value < minimum:
            raise ExternalCompositionError(f"{name}: invalid integer")
    if prefix >= candidate.workload.prompt_tokens:
        raise ExternalCompositionError("prefix must be shorter than the prompt")
    pools = {pool.role: pool for pool in candidate.pools}
    if len(candidate.pools) != 2 or set(pools) != {"prefill", "decode"}:
        raise ExternalCompositionError("capacity requires one prefill and one decode pool")
    if (
        candidate.model.model_id != "Qwen/Qwen3-32B-FP8"
        or candidate.model.framework != "external-trtllm"
        or candidate.model.inventory_sha256 != EXPECTED_MODEL_HASH
    ):
        raise ExternalCompositionError("candidate model differs from the imported source")
    for pool in pools.values():
        if (
            pool.device != "h200"
            or pool.tensor_parallel not in {2, 4, 8}
            or pool.gpus_per_engine != pool.tensor_parallel
            or (pool.pipeline_parallel, pool.expert_parallel, pool.data_parallel) != (1, 1, 1)
        ):
            raise ExternalCompositionError("unsupported external serving pool")
    key = candidate_key(candidate)

    def check_step(step: StepEstimate, phase: str) -> None:
        if not isinstance(step, StepEstimate):
            raise TypeError(f"{phase} must be StepEstimate")
        step.__post_init__()
        if step.stamp.candidate_key != key:
            raise ExternalCompositionError("estimate candidate identity mismatch")
        named = {term.name: term.estimate for term in step.stamp.terms}
        expected_names = (
            {"prefill_service", "fabric_floor", "intra_floor", "handoff"}
            if phase == "prefill"
            else {"kernel_floor", "fabric_floor", "intra_floor"}
        )
        if set(named) != expected_names:
            raise ExternalCompositionError("estimate stamp has an unexpected term set")
        kernel_name = "prefill_service" if phase == "prefill" else "kernel_floor"
        if (
            named[kernel_name] != step.kernel_floor
            or named["fabric_floor"] != step.fabric_floor
            or named["intra_floor"] != step.intra_floor
            or (phase == "prefill" and named["handoff"] != step.handoff)
        ):
            raise ExternalCompositionError("estimate fields disagree with its stamp")
        if (
            step.fabric_floor.duration_ps
            or step.intra_floor.duration_ps
            or step.fabric_excess_ps
            or step.intra_excess_ps
            or step.step_ps != step.kernel_floor.duration_ps
            or step.analytical_step_ps != step.step_ps
        ):
            raise ExternalCompositionError("capacity requires the external operation service path")
        if step.kernel_floor.evidence is not EvidenceClass.MEASURED_EXTERNAL:
            raise ExternalCompositionError("operation service is not measured-external")
        if phase == "prefill":
            if step.batch_service is not None or step.handoff is None:
                raise ExternalCompositionError("invalid prefill service projection")
        elif step.handoff is not None or step.batch_service != step.kernel_floor:
            raise ExternalCompositionError("invalid decode surface projection")

    check_step(prefill, "prefill")
    check_step(decode, "decode")
    prefill_matches = [
        (value, config)
        for value, config in services.values()
        if value.phase == "prefill" and value.as_term() == prefill.kernel_floor
    ]
    if len(prefill_matches) != 1:
        raise ExternalCompositionError("prefill term has no unique retained service")
    prefill_value, prefill_config = prefill_matches[0]
    retained(prefill_value.entry_key_sha256, "prefill")
    if (
        prefill_config["tensor_parallel"] != pools["prefill"].tensor_parallel
        or prefill_config["batch_size"] != 1
        or prefill_config["isl"] != candidate.workload.prompt_tokens
        or prefill_config["prefix"] != prefix
    ):
        raise ExternalCompositionError("prefill shape differs from requested capacity")
    marker = "batch-service-entry-key-sha256:"
    source = decode.kernel_floor.source
    if source.count(marker) != 1:
        raise ExternalCompositionError("decode term lacks exact selected surface keys")
    keys = source.split(marker, 1)[1].split(",")
    if not 1 <= len(keys) <= 2 or len(set(keys)) != len(keys):
        raise ExternalCompositionError("invalid selected surface key set")
    if tuple(keys) != tuple(
        selection.value.entry_key_sha256 for selection in projection.selections[1:]
    ):
        raise ExternalCompositionError("capacity selections differ from the estimator keys")
    points = []
    for service_key in keys:
        value, config = retained(service_key, "decode")
        if (
            config["tensor_parallel"] != pools["decode"].tensor_parallel
            or config["isl"] != candidate.workload.prompt_tokens
            or config["osl"] != candidate.workload.output_tokens
            or config["prefix"] != prefix
            or config["stride"] != stride
        ):
            raise ExternalCompositionError("decode shape differs from requested capacity")
        points.append(value.as_batch_service_point())
    if len(points) == 1:
        if points[0].batch_size != batch_size:
            raise ExternalCompositionError("selected decode batch differs from request")
        expected_service_ps = points[0].duration_ps
    else:
        if (
            not min(point.batch_size for point in points)
            < batch_size
            < max(point.batch_size for point in points)
        ):
            raise ExternalCompositionError("selected surface points do not bracket the batch")
        expected_service_ps = interpolate_batch_service_ps(tuple(points), batch_size)
    if decode.step_ps != expected_service_ps:
        raise ExternalCompositionError("decode duration differs from selected surface")


class ExternalQwen32BDeploymentBinding:
    """Build deployment services from the audited external pass model."""

    def __init__(
        self,
        database: ExternalOperationDatabase,
        *,
        composition_record: ExternalServingComposition | None = None,
    ) -> None:
        if not isinstance(database, ExternalOperationDatabase):
            raise TypeError("database must be ExternalOperationDatabase")
        self.database = database
        if composition_record is not None:
            if not isinstance(composition_record, ExternalServingComposition):
                raise TypeError("composition_record must be ExternalServingComposition")
            composition_record.validate_database(database)
        self._composition_record = composition_record
        self._services: dict[str, tuple[ExternalServiceValue, dict[str, object]]] = {}
        self._models: dict[tuple[int, str, str, str], ExternalQwen32BPassModel] = {}

    @property
    def composition_record(self) -> ExternalServingComposition | None:
        """Return the immutable optional parameter authority."""

        return self._composition_record

    def _parameter(
        self,
        name: str,
        explicit: object,
        default: object = COMPOSITION_UNSET,
    ) -> float:
        if self._composition_record is not None:
            return self._composition_record.resolve(name, explicit)
        if explicit is COMPOSITION_UNSET:
            if default is COMPOSITION_UNSET:
                raise TypeError(f"{name}: explicit value required without a composition record")
            return default
        return explicit

    def _identity(self, configuration: dict[str, object]) -> str:
        payload = {
            "external_source": self.database.source.as_dict(),
            "configuration": configuration,
        }
        if self._composition_record is not None:
            payload["composition_sha256"] = self._composition_record.record_sha256
        return _configuration_key(payload)

    def _composition_source(self, applied: tuple[str, ...]) -> str:
        if self._composition_record is None:
            return ""
        return (
            f";composition-sha256:{self._composition_record.record_sha256}"
            f";composition-applied:{','.join(applied)}"
        )

    def _model(
        self,
        tensor_parallel: int,
        *,
        memory_bandwidth_empirical_scale: float | None = None,
        memory_empirical_constant_latency_s: float | None = None,
        context_attention_extra_latency_correction: object = COMPOSITION_UNSET,
    ) -> ExternalQwen32BPassModel:
        if self._composition_record is not None:
            self._composition_record.validate_database(self.database)
        gpu = self.database.system_spec["gpu"]
        bandwidth_scale = self._parameter(
            "memory_bandwidth_empirical_scale",
            COMPOSITION_UNSET
            if memory_bandwidth_empirical_scale is None
            else memory_bandwidth_empirical_scale,
            float(gpu["mem_bw_empirical_scaling_factor"]),
        )
        constant_latency = self._parameter(
            "memory_empirical_constant_latency",
            COMPOSITION_UNSET
            if memory_empirical_constant_latency_s is None
            else memory_empirical_constant_latency_s,
            float(gpu["mem_empirical_constant_latency"]),
        )
        context_attention_extra_latency_correction = self._parameter(
            "context_attention_extra_latency_correction",
            context_attention_extra_latency_correction,
            1.1,
        )
        key = (
            tensor_parallel,
            bandwidth_scale.hex(),
            constant_latency.hex(),
            context_attention_extra_latency_correction.hex(),
        )
        if key not in self._models:
            self._models[key] = ExternalQwen32BPassModel(
                self.database,
                tensor_parallel=tensor_parallel,
                kv_cache_quant_mode="fp8",
                fmha_quant_mode="fp8",
                communication_quant_mode="half",
                memory_bandwidth_empirical_scale=bandwidth_scale,
                memory_empirical_constant_latency_s=constant_latency,
                context_attention_extra_latency_correction=(
                    context_attention_extra_latency_correction
                ),
                composition_record=self._composition_record,
            )
        return self._models[key]

    def _value(
        self,
        *,
        configuration: dict[str, object],
        phase: str,
        tensor_parallel: int,
        batch_size: int,
        service_ms: float,
        total_ms: float,
    ) -> ExternalServiceValue:
        entry_key = self._identity(configuration)
        source = (
            "external-operation-pass:"
            f"slice-sha256:{self.database.source.slice_hash};"
            f"configuration:{configuration['id']};"
            f"service-ms-hex:{service_ms.hex()};"
            f"entry-key-sha256:{entry_key}"
        )
        applied = (
            ("prefill_latency_correction", *EXTERNAL_OPERATION_ADJUSTMENTS)
            if phase == "prefill"
            else ("decode_latency_correction", *EXTERNAL_OPERATION_ADJUSTMENTS[:2])
        )
        source += self._composition_source(applied)
        value = ExternalServiceValue(
            configuration_id=str(configuration["id"]),
            phase=phase,
            tensor_parallel=tensor_parallel,
            batch_size=batch_size,
            service_ms=service_ms,
            total_ms=total_ms,
            source=source,
            entry_key_sha256=entry_key,
        )
        if self._composition_record is not None:
            self._services[entry_key] = (value, dict(configuration))
        return value

    def decode_service(
        self,
        *,
        tensor_parallel: int,
        batch_size: int,
        isl: int,
        osl: int,
        prefix: int,
        stride: int,
        latency_correction_scale: object = COMPOSITION_UNSET,
    ) -> ExternalServiceValue:
        """Return one corrected average decode-step service."""

        latency_correction_scale = self._parameter(
            "decode_latency_correction", latency_correction_scale
        )
        result = self._model(tensor_parallel).run_generation(
            batch_size=batch_size,
            isl=isl,
            osl=osl,
            stride=stride,
            latency_correction_scale=latency_correction_scale,
        )
        service_ms = result.total.latency_ms / (osl - 1)
        configuration = {
            "id": f"decode-tp{tensor_parallel}-b{batch_size}",
            "phase": "decode",
            "tensor_parallel": tensor_parallel,
            "batch_size": batch_size,
            "isl": isl,
            "osl": osl,
            "prefix": prefix,
            "stride": stride,
            "latency_correction_scale_hex": latency_correction_scale.hex(),
            "gemm_quant_mode": "fp8_block",
            "kv_cache_quant_mode": "fp8",
            "fmha_quant_mode": "fp8",
            "communication_quant_mode": "half",
        }
        return self._value(
            configuration=configuration,
            phase="decode",
            tensor_parallel=tensor_parallel,
            batch_size=batch_size,
            service_ms=service_ms,
            total_ms=result.total.latency_ms,
        )

    def decode_surface(
        self,
        *,
        tensor_parallel: int,
        batch_sizes: tuple[int, ...],
        isl: int,
        osl: int,
        prefix: int,
        stride: int,
        latency_correction_scale: object = COMPOSITION_UNSET,
    ) -> tuple[ExternalServiceValue, ...]:
        """Return a stable decode surface in caller-declared batch order."""

        if len(batch_sizes) < 2 or len(set(batch_sizes)) != len(batch_sizes):
            raise ValueError("batch_sizes must contain at least two unique values")
        return tuple(
            self.decode_service(
                tensor_parallel=tensor_parallel,
                batch_size=batch_size,
                isl=isl,
                osl=osl,
                prefix=prefix,
                stride=stride,
                latency_correction_scale=latency_correction_scale,
            )
            for batch_size in batch_sizes
        )

    def prefill_service(
        self,
        *,
        tensor_parallel: int,
        batch_size: int,
        isl: int,
        prefix: int,
        latency_correction_scale: object = COMPOSITION_UNSET,
    ) -> ExternalServiceValue:
        """Return one corrected prefill service pass."""

        latency_correction_scale = self._parameter(
            "prefill_latency_correction", latency_correction_scale
        )
        result = self._model(tensor_parallel).run_context(
            batch_size=batch_size,
            isl=isl,
            prefix=prefix,
            latency_correction_scale=latency_correction_scale,
        )
        configuration = {
            "id": f"prefill-tp{tensor_parallel}-b{batch_size}",
            "phase": "prefill",
            "tensor_parallel": tensor_parallel,
            "batch_size": batch_size,
            "isl": isl,
            "prefix": prefix,
            "latency_correction_scale_hex": latency_correction_scale.hex(),
            "gemm_quant_mode": "fp8_block",
            "kv_cache_quant_mode": "fp8",
            "fmha_quant_mode": "fp8",
            "communication_quant_mode": "half",
        }
        return self._value(
            configuration=configuration,
            phase="prefill",
            tensor_parallel=tensor_parallel,
            batch_size=batch_size,
            service_ms=result.total.latency_ms,
            total_ms=result.total.latency_ms,
        )

    def aggregate_point(
        self,
        *,
        tensor_parallel: int,
        batch_size: int,
        isl: int,
        osl: int,
        prefix: int,
        context_tokens: int,
        memory_bandwidth_empirical_scale: float | None = None,
        memory_empirical_constant_latency_s: float | None = None,
        context_attention_extra_latency_correction: object = COMPOSITION_UNSET,
        tpot_mixed_step_reduction: int = AGGREGATE_TPOT_MIXED_STEP_REDUCTION,
        apply_ttft_queueing: bool = True,
    ) -> ExternalAggregatePoint:
        """Compose one supported co-located aggregate operating point."""

        for name, value in (
            ("batch_size", batch_size),
            ("isl", isl),
            ("osl", osl),
            ("context_tokens", context_tokens),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if batch_size <= 1:
            raise ValueError("aggregate composition requires batch_size greater than one")
        if osl <= 1:
            raise ValueError("aggregate composition requires osl greater than one")
        if prefix < 0 or prefix >= isl:
            raise ValueError("prefix must be non-negative and smaller than isl")
        if (
            isinstance(tpot_mixed_step_reduction, bool)
            or not isinstance(tpot_mixed_step_reduction, int)
            or tpot_mixed_step_reduction < 0
        ):
            raise ValueError("tpot_mixed_step_reduction must be a non-negative integer")
        if not isinstance(apply_ttft_queueing, bool):
            raise TypeError("apply_ttft_queueing must be bool")

        mix_steps = math.ceil(isl * batch_size / context_tokens)
        if mix_steps >= osl:
            raise ValueError("the measured aggregate seam requires mix_steps smaller than osl")
        mix_generation_tokens = max(
            1,
            batch_size - math.ceil(context_tokens / isl),
        )
        genonly_steps = osl - mix_steps
        genonly_tokens = batch_size
        tpot_mix_steps = max(1, mix_steps - tpot_mixed_step_reduction)
        context_requests = math.ceil(context_tokens / isl)
        generation_requests = batch_size - context_requests
        if generation_requests <= 0:
            raise ValueError("aggregate composition requires generation requests")
        scheduled_tokens = context_tokens + generation_requests

        model = self._model(
            tensor_parallel,
            memory_bandwidth_empirical_scale=memory_bandwidth_empirical_scale,
            memory_empirical_constant_latency_s=memory_empirical_constant_latency_s,
            context_attention_extra_latency_correction=(context_attention_extra_latency_correction),
        )
        combined_prefix = prefix * math.floor(context_tokens / isl)
        combined = model._run_context(
            batch_size=1,
            isl=context_tokens + mix_generation_tokens,
            prefix=combined_prefix,
        )
        mix_operations: list[tuple[str, float]] = []
        mix_step_ms = 0.0
        for operation in combined.operations:
            if operation.operation != "context_attention":
                mix_operations.append((operation.operation, operation.latency_ms))
                mix_step_ms += operation.latency_ms

        context = model._run_context(
            batch_size=math.ceil(context_tokens / isl),
            isl=isl,
            prefix=prefix,
        )
        context_attention = next(
            operation.latency_ms
            for operation in context.operations
            if operation.operation == "context_attention"
        ) / math.ceil(isl / context_tokens)
        mix_operations.append(("context_attention (scaled)", context_attention))
        mix_step_ms += context_attention

        mixed_generation = model._run_generation(
            batch_size=mix_generation_tokens,
            isl=isl + osl // 2,
            osl=2,
            stride=32,
        )
        generation_attention = next(
            operation.latency_ms
            for operation in mixed_generation.operations
            if operation.operation == "generation_attention"
        )
        mix_operations.append(("generation_attention", generation_attention))
        mix_step_ms += generation_attention

        genonly = model._run_generation(
            batch_size=genonly_tokens,
            isl=isl + osl // 2,
            osl=2,
            stride=32,
        )
        genonly_operations = tuple(
            (operation.operation, operation.latency_ms) for operation in genonly.operations
        )
        genonly_step_ms = 0.0
        for _, latency_ms in genonly_operations:
            genonly_step_ms += latency_ms

        pure_prefill_step_ms = mix_step_ms
        prefill_passes_per_request = math.ceil(isl / context_tokens)
        base_prefill_ms = pure_prefill_step_ms * prefill_passes_per_request
        ttft_queue_factor = min(2 + (mix_steps - 3) / 20, 4) if apply_ttft_queueing else 1.0
        ttft_ms = base_prefill_ms * ttft_queue_factor
        ttft_queueing_component_ms = ttft_ms - base_prefill_ms
        tpot_ms = (mix_step_ms * tpot_mix_steps + genonly_step_ms * genonly_steps) / (
            tpot_mix_steps + genonly_steps
        )
        total_schedule_ms = mix_step_ms * mix_steps + genonly_step_ms * genonly_steps
        tokens_per_second = 1000 / total_schedule_ms * batch_size * (osl - 1)
        tokens_per_second_per_gpu = tokens_per_second / tensor_parallel
        tokens_per_second_per_user = 1000 / tpot_ms
        request_rate = tokens_per_second / (osl - 1)
        request_latency_ms = ttft_ms + tpot_ms * (osl - 1)
        balance_score = isl * batch_size / context_tokens / osl

        bandwidth_scale = model.memory_bandwidth_empirical_scale
        constant_latency = model.memory_empirical_constant_latency_s
        context_attention_extra_latency_correction = (
            model.context_attention_extra_latency_correction
        )
        configuration = {
            "id": (f"aggregate-tp{tensor_parallel}-b{batch_size}-ctx{context_tokens}"),
            "phase": "aggregate",
            "strategy": "aggregate-co-located-prefill-decode",
            "traffic_definition": "zero-byte-pd-handoff",
            "tensor_parallel": tensor_parallel,
            "batch_size": batch_size,
            "isl": isl,
            "osl": osl,
            "prefix": prefix,
            "context_tokens": context_tokens,
            "memory_bandwidth_empirical_scale_hex": bandwidth_scale.hex(),
            "memory_empirical_constant_latency_s_hex": constant_latency.hex(),
            "context_attention_extra_latency_correction_hex": (
                context_attention_extra_latency_correction.hex()
            ),
            "tpot_mixed_step_reduction": tpot_mixed_step_reduction,
            "apply_ttft_queueing": apply_ttft_queueing,
        }
        entry_key = self._identity(configuration)
        source = (
            "external-operation-aggregate-composition:"
            f"slice-sha256:{self.database.source.slice_hash};"
            f"configuration:{configuration['id']};"
            f"entry-key-sha256:{entry_key}"
        )
        source += self._composition_source(EXTERNAL_OPERATION_ADJUSTMENTS)
        return ExternalAggregatePoint(
            configuration_id=str(configuration["id"]),
            tensor_parallel=tensor_parallel,
            batch_size=batch_size,
            isl=isl,
            osl=osl,
            prefix=prefix,
            context_tokens=context_tokens,
            mix_steps=mix_steps,
            tpot_mix_steps=tpot_mix_steps,
            genonly_steps=genonly_steps,
            mix_generation_tokens=mix_generation_tokens,
            genonly_tokens=genonly_tokens,
            context_requests=context_requests,
            generation_requests=generation_requests,
            scheduled_tokens=scheduled_tokens,
            balance_score=balance_score,
            mix_step_ms=mix_step_ms,
            genonly_step_ms=genonly_step_ms,
            pure_prefill_step_ms=pure_prefill_step_ms,
            prefill_passes_per_request=prefill_passes_per_request,
            base_prefill_ms=base_prefill_ms,
            ttft_queue_factor=ttft_queue_factor,
            ttft_queueing_component_ms=ttft_queueing_component_ms,
            ttft_ms=ttft_ms,
            tpot_ms=tpot_ms,
            total_schedule_ms=total_schedule_ms,
            request_latency_ms=request_latency_ms,
            request_rate=request_rate,
            tokens_per_second=tokens_per_second,
            tokens_per_second_per_gpu=tokens_per_second_per_gpu,
            tokens_per_second_per_user=tokens_per_second_per_user,
            mix_operations_ms=tuple(mix_operations),
            genonly_operations_ms=genonly_operations,
            source=source,
            entry_key_sha256=entry_key,
        )

    def _require_record(self) -> ExternalServingComposition:
        if self._composition_record is None:
            raise ExternalCompositionError(
                "this projection requires an explicitly selected composition record"
            )
        self._composition_record.validate_database(self.database)
        return self._composition_record

    def _retained_service(
        self, key: str, phase: str
    ) -> tuple[ExternalServiceValue, dict[str, object]]:
        try:
            value, configuration = self._services[key]
        except KeyError as error:
            raise ExternalCompositionError(
                "service key was not produced by this binding"
            ) from error
        if value.phase != phase or self._identity(configuration) != value.entry_key_sha256:
            raise ExternalCompositionError("service phase or configuration identity mismatch")
        return value, configuration

    def autoscaled_ttft(self, corrected_prefill: ExternalServiceValue) -> ExternalAutoscaledTtft:
        """Apply only the declared external first-token heuristic to prefill."""

        record = self._require_record()
        if not isinstance(corrected_prefill, ExternalServiceValue):
            raise TypeError("corrected_prefill must be ExternalServiceValue")
        expected, config = self._retained_service(corrected_prefill.entry_key_sha256, "prefill")
        if corrected_prefill != expected:
            raise ExternalCompositionError("prefill value differs from its retained source")
        return ExternalAutoscaledTtft(_SelectedService._issue(expected, config), record)

    def disaggregated_capacity(
        self,
        candidate: DeploymentCandidate,
        prefill: StepEstimate,
        decode: StepEstimate,
        batch_size: int,
        *,
        prefix: int,
        stride: int,
    ) -> ExternalDisaggregatedCapacity:
        """Project exact pool rates from shape- and source-joined estimates."""

        record = self._require_record()
        if not isinstance(prefill, StepEstimate) or not isinstance(decode, StepEstimate):
            raise TypeError("prefill and decode must be StepEstimate")
        selected = [
            _SelectedService._issue(value, config)
            for value, config in self._services.values()
            if value.phase == "prefill" and value.as_term() == prefill.kernel_floor
        ]
        if len(selected) != 1:
            raise ExternalCompositionError("prefill term has no unique retained service")
        marker = "batch-service-entry-key-sha256:"
        source = decode.kernel_floor.source
        if source.count(marker) != 1:
            raise ExternalCompositionError("decode term lacks exact selected surface keys")
        for key in source.split(marker, 1)[1].split(","):
            value, config = self._retained_service(key, "decode")
            selected.append(_SelectedService._issue(value, config))
        return ExternalDisaggregatedCapacity(
            candidate, prefill, decode, batch_size, prefix, stride, tuple(selected), record
        )


def validate_external_scored_stamp(stamp: EstimateStamp) -> None:
    """Reject any positive scored duration not sourced as measured-external."""

    if not isinstance(stamp, EstimateStamp):
        raise TypeError("stamp must be EstimateStamp")
    stamp.__post_init__()
    positive = [term for term in stamp.terms if term.estimate.duration_ps > 0]
    if not positive:
        raise ValueError("external scored stamp has no positive duration")
    disallowed = [
        term.name
        for term in positive
        if term.estimate.evidence is not EvidenceClass.MEASURED_EXTERNAL
    ]
    if disallowed:
        raise ValueError(
            "external scored stamp contains non-MEASURED-EXTERNAL positive terms: "
            + ", ".join(disallowed)
        )


__all__ = [
    "AGGREGATE_TPOT_MIXED_STEP_REDUCTION",
    "PICOSECONDS_PER_MILLISECOND",
    "ExternalAggregatePoint",
    "ExternalAutoscaledTtft",
    "ExternalDisaggregatedCapacity",
    "ExternalQwen32BDeploymentBinding",
    "ExternalServiceValue",
    "validate_external_scored_stamp",
]
