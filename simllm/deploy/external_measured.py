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


class ExternalQwen32BDeploymentBinding:
    """Build deployment services from the audited external pass model."""

    def __init__(self, database: ExternalOperationDatabase) -> None:
        if not isinstance(database, ExternalOperationDatabase):
            raise TypeError("database must be ExternalOperationDatabase")
        self.database = database
        self._models: dict[int, ExternalQwen32BPassModel] = {}

    def _model(self, tensor_parallel: int) -> ExternalQwen32BPassModel:
        if tensor_parallel not in self._models:
            self._models[tensor_parallel] = ExternalQwen32BPassModel(
                self.database,
                tensor_parallel=tensor_parallel,
                kv_cache_quant_mode="fp8",
                fmha_quant_mode="fp8",
                communication_quant_mode="half",
            )
        return self._models[tensor_parallel]

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
    "PICOSECONDS_PER_MILLISECOND",
    "ExternalQwen32BDeploymentBinding",
    "ExternalServiceValue",
    "validate_external_scored_stamp",
]
