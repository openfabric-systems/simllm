"""Bind the pinned external operation database to deployment service terms."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass

from simllm.calibration.batch_service_surface import BatchServicePoint
from simllm.calibration.external_db import (
    EXTERNAL_EVIDENCE_CLASS,
    ExternalOperationDatabase,
    ExternalQwen32BPassModel,
)
from simllm.deploy.estimator import EstimateStamp, EvidenceClass, TermEstimate

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


class ExternalQwen32BDeploymentBinding:
    """Build deployment services from the audited external pass model."""

    def __init__(self, database: ExternalOperationDatabase) -> None:
        if not isinstance(database, ExternalOperationDatabase):
            raise TypeError("database must be ExternalOperationDatabase")
        self.database = database
        self._models: dict[
            tuple[int, str, str, str], ExternalQwen32BPassModel
        ] = {}

    def _model(
        self,
        tensor_parallel: int,
        *,
        memory_bandwidth_empirical_scale: float | None = None,
        memory_empirical_constant_latency_s: float | None = None,
        context_attention_extra_latency_correction: float = 1.1,
    ) -> ExternalQwen32BPassModel:
        gpu = self.database.system_spec["gpu"]
        bandwidth_scale = (
            float(gpu["mem_bw_empirical_scaling_factor"])
            if memory_bandwidth_empirical_scale is None
            else memory_bandwidth_empirical_scale
        )
        constant_latency = (
            float(gpu["mem_empirical_constant_latency"])
            if memory_empirical_constant_latency_s is None
            else memory_empirical_constant_latency_s
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
        key_payload = {
            "external_source": self.database.source.as_dict(),
            "configuration": configuration,
        }
        entry_key = _configuration_key(key_payload)
        source = (
            "external-operation-pass:"
            f"slice-sha256:{self.database.source.slice_hash};"
            f"configuration:{configuration['id']};"
            f"service-ms-hex:{service_ms.hex()};"
            f"entry-key-sha256:{entry_key}"
        )
        return ExternalServiceValue(
            configuration_id=str(configuration["id"]),
            phase=phase,
            tensor_parallel=tensor_parallel,
            batch_size=batch_size,
            service_ms=service_ms,
            total_ms=total_ms,
            source=source,
            entry_key_sha256=entry_key,
        )

    def decode_service(
        self,
        *,
        tensor_parallel: int,
        batch_size: int,
        isl: int,
        osl: int,
        prefix: int,
        stride: int,
        latency_correction_scale: float,
    ) -> ExternalServiceValue:
        """Return one corrected average decode-step service."""

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
        latency_correction_scale: float,
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
        latency_correction_scale: float,
    ) -> ExternalServiceValue:
        """Return one corrected prefill service pass."""

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
        context_attention_extra_latency_correction: float = 1.1,
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
            raise ValueError(
                "the measured aggregate seam requires mix_steps smaller than osl"
            )
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
            context_attention_extra_latency_correction=(
                context_attention_extra_latency_correction
            ),
        )
        combined_prefix = prefix * math.floor(context_tokens / isl)
        combined = model.run_context(
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

        context = model.run_context(
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

        mixed_generation = model.run_generation(
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

        genonly = model.run_generation(
            batch_size=genonly_tokens,
            isl=isl + osl // 2,
            osl=2,
            stride=32,
        )
        genonly_operations = tuple(
            (operation.operation, operation.latency_ms)
            for operation in genonly.operations
        )
        genonly_step_ms = 0.0
        for _, latency_ms in genonly_operations:
            genonly_step_ms += latency_ms

        pure_prefill_step_ms = mix_step_ms
        prefill_passes_per_request = math.ceil(isl / context_tokens)
        base_prefill_ms = pure_prefill_step_ms * prefill_passes_per_request
        ttft_queue_factor = (
            min(2 + (mix_steps - 3) / 20, 4)
            if apply_ttft_queueing
            else 1.0
        )
        ttft_ms = base_prefill_ms * ttft_queue_factor
        ttft_queueing_component_ms = ttft_ms - base_prefill_ms
        tpot_ms = (
            mix_step_ms * tpot_mix_steps
            + genonly_step_ms * genonly_steps
        ) / (tpot_mix_steps + genonly_steps)
        total_schedule_ms = (
            mix_step_ms * mix_steps + genonly_step_ms * genonly_steps
        )
        tokens_per_second = (
            1000 / total_schedule_ms * batch_size * (osl - 1)
        )
        tokens_per_second_per_gpu = tokens_per_second / tensor_parallel
        tokens_per_second_per_user = 1000 / tpot_ms
        request_rate = tokens_per_second / (osl - 1)
        request_latency_ms = ttft_ms + tpot_ms * (osl - 1)
        balance_score = isl * batch_size / context_tokens / osl

        gpu = self.database.system_spec["gpu"]
        bandwidth_scale = (
            float(gpu["mem_bw_empirical_scaling_factor"])
            if memory_bandwidth_empirical_scale is None
            else memory_bandwidth_empirical_scale
        )
        constant_latency = (
            float(gpu["mem_empirical_constant_latency"])
            if memory_empirical_constant_latency_s is None
            else memory_empirical_constant_latency_s
        )
        configuration = {
            "id": (
                f"aggregate-tp{tensor_parallel}-b{batch_size}-"
                f"ctx{context_tokens}"
            ),
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
        key_payload = {
            "external_source": self.database.source.as_dict(),
            "configuration": configuration,
        }
        entry_key = _configuration_key(key_payload)
        source = (
            "external-operation-aggregate-composition:"
            f"slice-sha256:{self.database.source.slice_hash};"
            f"configuration:{configuration['id']};"
            f"entry-key-sha256:{entry_key}"
        )
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
    "ExternalQwen32BDeploymentBinding",
    "ExternalServiceValue",
    "validate_external_scored_stamp",
]
