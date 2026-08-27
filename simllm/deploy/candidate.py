"""Strict deployment candidates and backend-free feasibility checks."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from simllm.calibration.canonical import canonical_sha256
from simllm.compute import GPU_ENVELOPES
from simllm.core._wire import (
    _array,
    _fields,
    _integer,
    _object,
    _optional_integer,
    _require_tuple,
    _string,
)

DEPLOYMENT_CANDIDATE_SCHEMA = "simllm-deployment-candidate-v1"

PIPELINE_PARALLEL_UNPRICED = "pipeline-parallel-unpriced"
HBM_CAPACITY_EXCEEDED = "hbm-capacity-exceeded"
BUDGET_GPUS_EXCEEDED = "budget-gpus-exceeded"
BUDGET_NODES_EXCEEDED = "budget-nodes-exceeded"

_POOL_ROLES = frozenset({"prefill", "decode", "combined"})
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


def _validated_string(value: object, path: str) -> str:
    text = _string(value, path)
    if not text.isascii():
        raise ValueError(f"{path}: v1 candidate strings must be ASCII")
    return text


def _validated_positive_integer(value: object, path: str) -> int:
    return _integer(value, path, minimum=1)


def _validated_nonnegative_integer(value: object, path: str) -> int:
    return _integer(value, path, nonnegative=True)


def _validated_sha256(value: object, path: str) -> str:
    digest = _string(value, path)
    if _SHA256_RE.fullmatch(digest) is None:
        raise ValueError(f"{path}: expected 64 lowercase hexadecimal digits")
    return digest


def _require_instance(value: object, expected_type: type[Any], path: str) -> None:
    if not isinstance(value, expected_type):
        raise TypeError(f"{path}: expected {expected_type.__name__}")


@dataclass(frozen=True, slots=True)
class PoolSpec:
    """One declared serving-engine pool."""

    role: str
    engines: int
    gpus_per_engine: int
    tensor_parallel: int
    pipeline_parallel: int
    expert_parallel: int
    data_parallel: int
    device: str

    def __post_init__(self) -> None:
        role = _validated_string(self.role, "pool.role")
        if role not in _POOL_ROLES:
            raise ValueError(
                f"pool.role: unknown value {role!r}; expected one of {sorted(_POOL_ROLES)}"
            )
        for field_name in (
            "engines",
            "gpus_per_engine",
            "tensor_parallel",
            "pipeline_parallel",
            "expert_parallel",
            "data_parallel",
        ):
            _validated_positive_integer(
                getattr(self, field_name),
                f"pool.{field_name}",
            )
        device = _validated_string(self.device, "pool.device")
        if device not in GPU_ENVELOPES:
            raise ValueError(
                f"pool.device: unknown GPU envelope {device!r}; "
                f"expected one of {sorted(GPU_ENVELOPES)}"
            )


@dataclass(frozen=True, slots=True)
class FabricSpec:
    """Declared inter-node and intra-node payload rates."""

    inter_node_bits_per_second: int
    intra_node_bytes_per_second: int

    def __post_init__(self) -> None:
        _validated_positive_integer(
            self.inter_node_bits_per_second,
            "fabric.inter_node_bits_per_second",
        )
        _validated_positive_integer(
            self.intra_node_bytes_per_second,
            "fabric.intra_node_bytes_per_second",
        )


@dataclass(frozen=True, slots=True)
class WorkloadPoint:
    """One steady-state workload point priced by the deployment estimator."""

    arrival_rate_rps: int | None
    prompt_tokens: int
    output_tokens: int
    kv_context_tokens: int

    def __post_init__(self) -> None:
        _optional_integer(
            self.arrival_rate_rps,
            "workload.arrival_rate_rps",
            minimum=1,
        )
        _validated_positive_integer(self.prompt_tokens, "workload.prompt_tokens")
        _validated_positive_integer(self.output_tokens, "workload.output_tokens")
        _validated_nonnegative_integer(
            self.kv_context_tokens,
            "workload.kv_context_tokens",
        )


@dataclass(frozen=True, slots=True)
class SlaSpec:
    """Optional per-token and first-token latency targets in picoseconds."""

    tpot_target_ps: int | None
    ttft_target_ps: int | None

    def __post_init__(self) -> None:
        _optional_integer(self.tpot_target_ps, "sla.tpot_target_ps", minimum=1)
        _optional_integer(self.ttft_target_ps, "sla.ttft_target_ps", minimum=1)


@dataclass(frozen=True, slots=True)
class BudgetSpec:
    """Optional physical-size limits for one deployment candidate."""

    max_gpus: int | None
    max_nodes: int | None

    def __post_init__(self) -> None:
        _optional_integer(self.max_gpus, "budget.max_gpus", minimum=1)
        _optional_integer(self.max_nodes, "budget.max_nodes", minimum=1)


@dataclass(frozen=True, slots=True)
class ModelRef:
    """Content-addressed model inventory selected by a framework."""

    framework: str
    model_id: str
    inventory_sha256: str

    def __post_init__(self) -> None:
        _validated_string(self.framework, "model.framework")
        _validated_string(self.model_id, "model.model_id")
        _validated_sha256(self.inventory_sha256, "model.inventory_sha256")


@dataclass(frozen=True, slots=True)
class DeploymentCandidate:
    """One caller-stable candidate for the analytical planning rung."""

    candidate_id: str
    model: ModelRef
    pools: tuple[PoolSpec, ...]
    fabric: FabricSpec
    workload: WorkloadPoint
    sla: SlaSpec
    budget: BudgetSpec

    def __post_init__(self) -> None:
        _validated_string(self.candidate_id, "candidate.candidate_id")
        _require_instance(self.model, ModelRef, "candidate.model")
        pools = _require_tuple(self.pools, "candidate.pools")
        if not pools:
            raise ValueError("candidate.pools: must contain at least one pool")
        roles: list[str] = []
        for index, pool in enumerate(pools):
            _require_instance(pool, PoolSpec, f"candidate.pools[{index}]")
            pool.__post_init__()
            roles.append(pool.role)
        if len(roles) != len(set(roles)):
            raise ValueError("candidate.pools: contains duplicate pool roles")
        _require_instance(self.fabric, FabricSpec, "candidate.fabric")
        _require_instance(self.workload, WorkloadPoint, "candidate.workload")
        _require_instance(self.sla, SlaSpec, "candidate.sla")
        _require_instance(self.budget, BudgetSpec, "candidate.budget")
        self.model.__post_init__()
        self.fabric.__post_init__()
        self.workload.__post_init__()
        self.sla.__post_init__()
        self.budget.__post_init__()


@dataclass(frozen=True, slots=True)
class FeasibilityReport:
    """Stable acceptance decision and zero or more rejection reason codes."""

    accepted: bool
    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        if type(self.accepted) is not bool:
            raise TypeError("feasibility.accepted: expected a boolean")
        reasons = _require_tuple(self.reasons, "feasibility.reasons")
        for index, reason in enumerate(reasons):
            _string(reason, f"feasibility.reasons[{index}]")
        if len(reasons) != len(set(reasons)):
            raise ValueError("feasibility.reasons: contains duplicate values")
        if self.accepted != (not reasons):
            raise ValueError(
                "feasibility.accepted: must be true exactly when reasons is empty"
            )


def _pool_to_json(pool: PoolSpec) -> dict[str, object]:
    pool.__post_init__()
    return {
        "role": pool.role,
        "engines": pool.engines,
        "gpus_per_engine": pool.gpus_per_engine,
        "tensor_parallel": pool.tensor_parallel,
        "pipeline_parallel": pool.pipeline_parallel,
        "expert_parallel": pool.expert_parallel,
        "data_parallel": pool.data_parallel,
        "device": pool.device,
    }


def _fabric_to_json(fabric: FabricSpec) -> dict[str, int]:
    fabric.__post_init__()
    return {
        "inter_node_bits_per_second": fabric.inter_node_bits_per_second,
        "intra_node_bytes_per_second": fabric.intra_node_bytes_per_second,
    }


def _workload_to_json(workload: WorkloadPoint) -> dict[str, int | None]:
    workload.__post_init__()
    return {
        "arrival_rate_rps": workload.arrival_rate_rps,
        "prompt_tokens": workload.prompt_tokens,
        "output_tokens": workload.output_tokens,
        "kv_context_tokens": workload.kv_context_tokens,
    }


def _sla_to_json(sla: SlaSpec) -> dict[str, int | None]:
    sla.__post_init__()
    return {
        "tpot_target_ps": sla.tpot_target_ps,
        "ttft_target_ps": sla.ttft_target_ps,
    }


def _budget_to_json(budget: BudgetSpec) -> dict[str, int | None]:
    budget.__post_init__()
    return {"max_gpus": budget.max_gpus, "max_nodes": budget.max_nodes}


def _model_to_json(model: ModelRef) -> dict[str, str]:
    model.__post_init__()
    return {
        "framework": model.framework,
        "model_id": model.model_id,
        "inventory_sha256": model.inventory_sha256,
    }


def to_json(candidate: DeploymentCandidate) -> dict[str, object]:
    """Return the strict schema-tagged JSON object for one candidate."""

    _require_instance(candidate, DeploymentCandidate, "candidate")
    candidate.__post_init__()
    return {
        "schema": DEPLOYMENT_CANDIDATE_SCHEMA,
        "candidate_id": candidate.candidate_id,
        "model": _model_to_json(candidate.model),
        "pools": [_pool_to_json(pool) for pool in candidate.pools],
        "fabric": _fabric_to_json(candidate.fabric),
        "workload": _workload_to_json(candidate.workload),
        "sla": _sla_to_json(candidate.sla),
        "budget": _budget_to_json(candidate.budget),
    }


def _schema_first_candidate_payload(value: object) -> Mapping[str, Any]:
    payload = _object(value, "candidate")
    schema = _string(payload.get("schema"), "candidate.schema")
    if schema != DEPLOYMENT_CANDIDATE_SCHEMA:
        raise ValueError(
            "candidate.schema: unsupported schema "
            f"{schema!r}; expected {DEPLOYMENT_CANDIDATE_SCHEMA!r}"
        )
    _fields(
        payload,
        "candidate",
        required={
            "schema",
            "candidate_id",
            "model",
            "pools",
            "fabric",
            "workload",
            "sla",
            "budget",
        },
    )
    return payload


def _pool_from_json(value: object, path: str) -> PoolSpec:
    payload = _object(value, path)
    _fields(
        payload,
        path,
        required={
            "role",
            "engines",
            "gpus_per_engine",
            "tensor_parallel",
            "pipeline_parallel",
            "expert_parallel",
            "data_parallel",
            "device",
        },
    )
    role = _string(payload["role"], f"{path}.role")
    if role not in _POOL_ROLES:
        raise ValueError(
            f"{path}.role: unknown value {role!r}; "
            f"expected one of {sorted(_POOL_ROLES)}"
        )
    device = _string(payload["device"], f"{path}.device")
    if device not in GPU_ENVELOPES:
        raise ValueError(
            f"{path}.device: unknown GPU envelope {device!r}; "
            f"expected one of {sorted(GPU_ENVELOPES)}"
        )
    return PoolSpec(
        role=role,
        engines=_integer(payload["engines"], f"{path}.engines", minimum=1),
        gpus_per_engine=_integer(
            payload["gpus_per_engine"],
            f"{path}.gpus_per_engine",
            minimum=1,
        ),
        tensor_parallel=_integer(
            payload["tensor_parallel"],
            f"{path}.tensor_parallel",
            minimum=1,
        ),
        pipeline_parallel=_integer(
            payload["pipeline_parallel"],
            f"{path}.pipeline_parallel",
            minimum=1,
        ),
        expert_parallel=_integer(
            payload["expert_parallel"],
            f"{path}.expert_parallel",
            minimum=1,
        ),
        data_parallel=_integer(
            payload["data_parallel"],
            f"{path}.data_parallel",
            minimum=1,
        ),
        device=device,
    )


def _model_from_json(value: object) -> ModelRef:
    path = "candidate.model"
    payload = _object(value, path)
    _fields(
        payload,
        path,
        required={"framework", "model_id", "inventory_sha256"},
    )
    return ModelRef(
        framework=_string(payload["framework"], f"{path}.framework"),
        model_id=_string(payload["model_id"], f"{path}.model_id"),
        inventory_sha256=_validated_sha256(
            payload["inventory_sha256"],
            f"{path}.inventory_sha256",
        ),
    )


def _fabric_from_json(value: object) -> FabricSpec:
    path = "candidate.fabric"
    payload = _object(value, path)
    _fields(
        payload,
        path,
        required={"inter_node_bits_per_second", "intra_node_bytes_per_second"},
    )
    return FabricSpec(
        inter_node_bits_per_second=_integer(
            payload["inter_node_bits_per_second"],
            f"{path}.inter_node_bits_per_second",
            minimum=1,
        ),
        intra_node_bytes_per_second=_integer(
            payload["intra_node_bytes_per_second"],
            f"{path}.intra_node_bytes_per_second",
            minimum=1,
        ),
    )


def _workload_from_json(value: object) -> WorkloadPoint:
    path = "candidate.workload"
    payload = _object(value, path)
    _fields(
        payload,
        path,
        required={
            "arrival_rate_rps",
            "prompt_tokens",
            "output_tokens",
            "kv_context_tokens",
        },
    )
    return WorkloadPoint(
        arrival_rate_rps=_optional_integer(
            payload["arrival_rate_rps"],
            f"{path}.arrival_rate_rps",
            minimum=1,
        ),
        prompt_tokens=_integer(
            payload["prompt_tokens"],
            f"{path}.prompt_tokens",
            minimum=1,
        ),
        output_tokens=_integer(
            payload["output_tokens"],
            f"{path}.output_tokens",
            minimum=1,
        ),
        kv_context_tokens=_integer(
            payload["kv_context_tokens"],
            f"{path}.kv_context_tokens",
            nonnegative=True,
        ),
    )


def _sla_from_json(value: object) -> SlaSpec:
    path = "candidate.sla"
    payload = _object(value, path)
    _fields(payload, path, required={"tpot_target_ps", "ttft_target_ps"})
    return SlaSpec(
        tpot_target_ps=_optional_integer(
            payload["tpot_target_ps"],
            f"{path}.tpot_target_ps",
            minimum=1,
        ),
        ttft_target_ps=_optional_integer(
            payload["ttft_target_ps"],
            f"{path}.ttft_target_ps",
            minimum=1,
        ),
    )


def _budget_from_json(value: object) -> BudgetSpec:
    path = "candidate.budget"
    payload = _object(value, path)
    _fields(payload, path, required={"max_gpus", "max_nodes"})
    return BudgetSpec(
        max_gpus=_optional_integer(
            payload["max_gpus"],
            f"{path}.max_gpus",
            minimum=1,
        ),
        max_nodes=_optional_integer(
            payload["max_nodes"],
            f"{path}.max_nodes",
            minimum=1,
        ),
    )


def from_json(value: object) -> DeploymentCandidate:
    """Parse one strict candidate after checking its schema tag first."""

    payload = _schema_first_candidate_payload(value)
    pools = tuple(
        _pool_from_json(pool, f"candidate.pools[{index}]")
        for index, pool in enumerate(_array(payload["pools"], "candidate.pools"))
    )
    return DeploymentCandidate(
        candidate_id=_string(payload["candidate_id"], "candidate.candidate_id"),
        model=_model_from_json(payload["model"]),
        pools=pools,
        fabric=_fabric_from_json(payload["fabric"]),
        workload=_workload_from_json(payload["workload"]),
        sla=_sla_from_json(payload["sla"]),
        budget=_budget_from_json(payload["budget"]),
    )


def candidate_to_json(candidate: DeploymentCandidate) -> dict[str, object]:
    """Named alias for callers that keep several wire schemas in scope."""

    return to_json(candidate)


def candidate_from_json(value: object) -> DeploymentCandidate:
    """Named alias for callers that keep several wire schemas in scope."""

    return from_json(value)


def candidate_key(candidate: DeploymentCandidate) -> str:
    """Return the SHA-256 identity of the candidate's canonical JSON object.

    Total over schema-valid candidates: v1 restricts every candidate string
    field to ASCII, so the canonical family accepts every candidate that
    validation accepts, independent of the interpreter's Unicode handling.
    """

    return canonical_sha256(to_json(candidate))


def _validated_integer_mapping(
    value: Mapping[str, int],
    path: str,
    *,
    minimum: int,
) -> Mapping[str, int]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{path}: expected a mapping")
    for key, item in value.items():
        _string(key, f"{path} key")
        _integer(item, f"{path}[{key!r}]", minimum=minimum)
    return value


def check_feasibility(
    candidate: DeploymentCandidate,
    *,
    static_rank_bytes_per_pool: Mapping[str, int],
    device_hbm_capacity_bytes: Mapping[str, int],
) -> FeasibilityReport:
    """Apply v1 capacity and declared-size refusals without running a backend.

    ``static_rank_bytes_per_pool`` is keyed by pool role and
    ``device_hbm_capacity_bytes`` is keyed by the candidate's GPU envelope
    name. One declared engine is one node slot for the v1 node-budget check;
    an engine spanning more than one physical node is still counted as one
    slot, so the budget-nodes-exceeded refusal is a lower bound until
    DEPLOY-2 returns rendered host packing.
    """

    _require_instance(candidate, DeploymentCandidate, "candidate")
    candidate.__post_init__()
    static_bytes = _validated_integer_mapping(
        static_rank_bytes_per_pool,
        "static_rank_bytes_per_pool",
        minimum=0,
    )
    capacities = _validated_integer_mapping(
        device_hbm_capacity_bytes,
        "device_hbm_capacity_bytes",
        minimum=1,
    )

    for pool in candidate.pools:
        if pool.role not in static_bytes:
            raise ValueError(
                "static_rank_bytes_per_pool: missing candidate pool role "
                f"{pool.role!r}"
            )
        if pool.device not in capacities:
            raise ValueError(
                "device_hbm_capacity_bytes: missing candidate device "
                f"{pool.device!r}"
            )

    reasons: list[str] = []
    if any(pool.pipeline_parallel > 1 for pool in candidate.pools):
        reasons.append(PIPELINE_PARALLEL_UNPRICED)
    if any(
        static_bytes[pool.role] >= capacities[pool.device]
        for pool in candidate.pools
    ):
        reasons.append(HBM_CAPACITY_EXCEEDED)

    required_gpus = sum(
        pool.engines * pool.gpus_per_engine for pool in candidate.pools
    )
    if candidate.budget.max_gpus is not None and required_gpus > candidate.budget.max_gpus:
        reasons.append(BUDGET_GPUS_EXCEEDED)

    required_nodes = sum(pool.engines for pool in candidate.pools)
    if candidate.budget.max_nodes is not None and required_nodes > candidate.budget.max_nodes:
        reasons.append(BUDGET_NODES_EXCEEDED)

    return FeasibilityReport(accepted=not reasons, reasons=tuple(reasons))


__all__ = [
    "BUDGET_GPUS_EXCEEDED",
    "BUDGET_NODES_EXCEEDED",
    "DEPLOYMENT_CANDIDATE_SCHEMA",
    "HBM_CAPACITY_EXCEEDED",
    "PIPELINE_PARALLEL_UNPRICED",
    "BudgetSpec",
    "DeploymentCandidate",
    "FabricSpec",
    "FeasibilityReport",
    "ModelRef",
    "PoolSpec",
    "SlaSpec",
    "WorkloadPoint",
    "candidate_from_json",
    "candidate_key",
    "candidate_to_json",
    "check_feasibility",
    "from_json",
    "to_json",
]
