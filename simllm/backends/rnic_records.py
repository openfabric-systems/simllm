"""Strict readers and bypass checks for native RNIC session records."""

from __future__ import annotations

import csv
import enum
import hashlib
import io
import itertools
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from simllm.core._wire import (
    _array,
    _boolean,
    _fields,
    _integer,
    _object,
    _optional_integer,
    _optional_string,
    _string,
)

RNIC_EFFECTIVE_HARDWARE_SCHEMA = "simllm-rnic-effective-hardware-v1"
RNIC_SESSION_CONFIG_SCHEMA = "simllm-rnic-session-config-v1"
RNIC_SESSION_RESULT_SCHEMA = "simllm-rnic-session-result-v1"
RNIC_BOOKKEEPING_SCHEMA = "simllm-rnic-bookkeeping-v1"

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_BYPASS_PARAMETER_RE = re.compile(r"[a-z][a-z0-9_]*\Z")
_BYPASS_PARAMETER_EXCLUSIONS = frozenset(
    {
        "argv",
        "binary",
        "command",
        "command_line",
        "completion_csv",
        "completion_output",
        "executable",
        "goal",
        "goal_bin",
        "goal_binary",
        "output",
        "profile",
        "random_seed",
        "rnic_profile",
        "seed",
        "topo",
        "topology",
    }
)
_WQE_TIMELINE_FIELDS = {
    "admitted_at_ps",
    "cqe_visible_at_ps",
    "doorbell_seen_at_ps",
    "doorbelled_at_ps",
    "first_packet_at_ps",
    "last_packet_at_ps",
    "network_accepted_at_ps",
    "network_outcome_at_ps",
    "polled_at_ps",
    "posted_at_ps",
    "qpc_ready_at_ps",
    "sq_reclaimed_at_ps",
    "transport_retired_at_ps",
    "wqe_fetch_begin_at_ps",
    "wqe_fetch_end_at_ps",
}
_COMPLETION_CSV_HEADER = (
    "profile",
    "flow_id",
    "source",
    "destination",
    "tag",
    "payload_bytes",
    "start_time_ps",
    "completion_time_ps",
    "fct_ps",
    "wqe_id",
    "sq_id",
    "rq_id",
    "cq_id",
    "sq_post_sequence",
    "sq_dispatch_sequence",
    "cq_post_sequence",
    "cq_consume_sequence",
    "transport_kind",
    "transport_object_id",
)
_EXPECTED_INACTIVE_HASH_GUARDS = frozenset(
    {
        "correlation_identity",
        "disabled_dma_payload",
        "disabled_path_payload",
        "path_declaration_order",
        "policy_permutation",
        "session_id",
        "unused_nonposted_data_credits",
    }
)
_EXPECTED_PROJECTION_CHECKS = frozenset(
    {"native_timestamps", "no_structural_rq", "one_to_one", "stable_keys"}
)
_EXPECTED_AUTHORITY_NEGATIVE_CONTROLS = frozenset(
    {
        "both_rejected",
        "failed_counters_unchanged",
        "neither_rejected",
        "wrong_legacy_rejected",
        "wrong_native_rejected",
    }
)
_EXPECTED_SENSITIVITY_BY_GROUP = {
    "scalar": frozenset(
        {
            "dma.enabled.activation",
            "network.enabled",
            "qpc.enabled",
            "work_queue.cq_depth",
            "work_queue.cqe_write_service_ps",
            "work_queue.doorbell_service_ps",
            "work_queue.qpc_lookup_service_ps",
            "work_queue.scheduler_service_ps",
            "work_queue.sq_depth",
            "work_queue.wqe_fetch_service_ps",
        }
    ),
    "dma_binding": frozenset(
        {
            "fabric_scope",
            "pcie_completion_ordering_domain",
            "pcie_cq_first_byte_offset",
            "pcie_cq_memory_path_id",
            "pcie_cqe_bytes",
            "pcie_doorbell_record_bytes",
            "pcie_doorbell_record_first_byte_offset",
            "pcie_doorbell_record_path_id",
            "pcie_sq_first_byte_offset",
            "pcie_sq_memory_path_id",
            "pcie_submission_ordering_domain",
            "pcie_uar_doorbell_bytes",
            "pcie_uar_first_byte_offset",
            "pcie_uar_path_id",
            "pcie_wqe_bytes",
            "shared_ordering_domain_namespace",
        }
    ),
    "pcie_fabric": frozenset(
        {
            "analytical_seed",
            "completion_buffer_bytes",
            "completion_buffer_release_latency_ps",
            "completion_overhead_bytes",
            "credit_return_latency_ps",
            "data_credit_unit_bytes",
            "device_to_host_credits.completion_data_credits",
            "device_to_host_credits.completion_header_credits",
            "device_to_host_credits.nonposted_header_credits",
            "device_to_host_credits.posted_data_credits",
            "device_to_host_credits.posted_header_credits",
            "generation",
            "host_store_latency_ps",
            "host_to_device_credits.completion_data_credits",
            "host_to_device_credits.completion_header_credits",
            "host_to_device_credits.nonposted_header_credits",
            "host_to_device_credits.posted_data_credits",
            "host_to_device_credits.posted_header_credits",
            "lane_count",
            "max_outstanding_read_requests",
            "max_payload_size_bytes",
            "max_read_request_size_bytes",
            "max_tlps_per_transaction",
            "posted_write_overhead_bytes",
            "posted_write_visibility_latency_ps",
            "read_completion_boundary_bytes",
            "read_completion_latency_ps",
            "read_request_overhead_bytes",
        }
    ),
    "pcie_path": frozenset({"base_latency_ps", "enabled", "endpoint", "path_id"}),
    "analytical": frozenset(
        {
            "component.acs.activation",
            "component.ddio_miss.activation",
            "component.gpu_direct.activation",
            "component.iommu.activation",
            "component.numa.activation",
            "component.switch_path.activation",
            "profile.gaussian.activation",
            "profile.gaussian_tail_mixture.activation",
            "profile.incidence_probability_ppm",
            "profile.mean_ps",
            "profile.standard_deviation_ps",
            "profile.tail_mean_ps",
            "profile.tail_probability_ppm",
            "profile.tail_standard_deviation_ps",
        }
    ),
}


class RnicHardwareMode(enum.Enum):
    STRUCTURAL = "structural"
    BYPASS = "bypass"


class RnicWqeAuthority(enum.Enum):
    NATIVE = "SimllmNativeRnicSession"
    ATLAHS_LEDGER = "AtlahsWqeLedger"


@dataclass(frozen=True)
class RnicSessionConfigRecord:
    session_id: str
    hardware_mode: RnicHardwareMode
    authority: RnicWqeAuthority
    transport_policy: str
    hardware_config_sha256: str | None
    effective_hardware: Mapping[str, Any] | None


@dataclass(frozen=True)
class RnicAuthorityCounters:
    native_session_constructed: int
    legacy_ledger_constructed: int
    native_posts: int
    legacy_mutations: int


@dataclass(frozen=True)
class RnicWqeProjectionKey:
    session_id: str
    endpoint: int
    wq_kind: str
    wq_id: int
    post_sequence: int


@dataclass(frozen=True)
class RnicWqeTimeline:
    posted_at_ps: int
    doorbelled_at_ps: int | None
    doorbell_seen_at_ps: int | None
    wqe_fetch_begin_at_ps: int | None
    wqe_fetch_end_at_ps: int | None
    qpc_ready_at_ps: int | None
    admitted_at_ps: int | None
    network_accepted_at_ps: int | None
    network_outcome_at_ps: int | None
    first_packet_at_ps: int | None
    last_packet_at_ps: int | None
    transport_retired_at_ps: int | None
    cqe_visible_at_ps: int | None
    polled_at_ps: int | None
    sq_reclaimed_at_ps: int | None


@dataclass(frozen=True)
class RnicWqeProjectionRecord:
    key: RnicWqeProjectionKey
    wqe_id: int
    wr_id: int | None
    flow_id: int
    flow_tag: int
    source: int
    destination: int
    payload_bytes: int
    qpn: int | None
    signaled: bool
    opcode: str
    timeline: RnicWqeTimeline
    state: str
    completion_status: str | None
    cq_id: int
    cqe_sequence: int | None
    cq_producer_index: int | None
    cq_consume_sequence: int | None
    transport_kind: str
    transport_object_id: int


@dataclass(frozen=True)
class RnicCompletionProjection:
    profile: str
    flow_id: int
    source: int
    destination: int
    tag: int
    payload_bytes: int
    start_time_ps: int
    completion_time_ps: int
    fct_ps: int
    wqe_id: int | None
    sq_id: int | None
    rq_id: int | None
    cq_id: int | None
    sq_post_sequence: int | None
    sq_dispatch_sequence: int | None
    cq_post_sequence: int | None
    cq_consume_sequence: int | None
    transport_kind: str | None
    transport_object_id: int | None


@dataclass(frozen=True)
class RnicSessionResultRecord:
    session_id: str
    hardware_mode: RnicHardwareMode
    authority: RnicWqeAuthority
    transport_policy: str
    hardware_config_sha256: str | None
    authority_counters: RnicAuthorityCounters
    quiescent: bool
    wqes: tuple[RnicWqeProjectionRecord, ...]
    completion_rows: tuple[RnicCompletionProjection, ...]


@dataclass(frozen=True)
class RnicBookkeepingProjection:
    session_id: str
    hardware_mode: RnicHardwareMode
    authority: RnicWqeAuthority
    hardware_config_sha256: str | None
    wqes: tuple[RnicWqeProjectionRecord, ...]


@dataclass(frozen=True)
class BypassArtifacts:
    """Bytes covered by the frozen bypass-identity contract.

    Run records and diagnostic command metadata are deliberately absent.
    """

    goal_text: bytes
    goal_binary: bytes
    topology: bytes
    profile: str
    seed: int
    baseline_parameters: tuple[tuple[str, str], ...]
    completion_csv: bytes
    canonical_completion: bytes
    step_results: bytes
    replay_summary: bytes


@dataclass(frozen=True)
class BypassArtifactPaths:
    goal_text: Path
    goal_binary: Path
    topology: Path
    profile: str
    seed: int
    baseline_parameters: tuple[tuple[str, str], ...]
    completion_csv: Path
    canonical_completion: Path
    step_results: Path
    replay_summary: Path


@dataclass(frozen=True)
class BypassArtifactComparison:
    input_matches: tuple[tuple[str, bool], ...]
    behavioral_matches: tuple[tuple[str, bool], ...]

    @property
    def equivalent(self) -> bool:
        return all(match for _, match in self.input_matches) and all(
            match for _, match in self.behavioral_matches
        )

    @property
    def changed_inputs(self) -> tuple[str, ...]:
        return tuple(name for name, match in self.input_matches if not match)

    @property
    def changed_artifacts(self) -> tuple[str, ...]:
        return tuple(name for name, match in self.behavioral_matches if not match)


def _enum_value(cls: type[enum.Enum], value: Any, path: str) -> enum.Enum:
    raw = _string(value, path)
    try:
        return cls(raw)
    except ValueError as error:
        choices = [member.value for member in cls]
        raise ValueError(f"{path}: unknown value {raw!r}; expected one of {choices}") from error


def _choice(value: Any, choices: set[str], path: str) -> str:
    raw = _string(value, path)
    if raw not in choices:
        raise ValueError(f"{path}: unknown value {raw!r}; expected one of {sorted(choices)}")
    return raw


def _sha256(value: Any, path: str) -> str:
    digest = _string(value, path)
    if _SHA256_RE.fullmatch(digest) is None:
        raise ValueError(f"{path}: expected 64 lowercase hexadecimal digits")
    return digest


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _freeze_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {name: _freeze_json(item) for name, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


def _credits(value: Any, path: str) -> dict[str, int]:
    obj = _object(value, path)
    names = {
        "completion_data_credits",
        "completion_header_credits",
        "nonposted_header_credits",
        "posted_data_credits",
        "posted_header_credits",
    }
    _fields(obj, path, required=names)
    return {
        name: _integer(obj[name], f"{path}.{name}", minimum=1)
        for name in sorted(names)
    }


def _analytical_profile(value: Any, path: str) -> dict[str, Any]:
    obj = _object(value, path)
    kind = _choice(
        obj.get("kind"),
        {"disabled", "fixed", "gaussian", "gaussian_tail_mixture"},
        f"{path}.kind",
    )
    fields_by_kind = {
        "disabled": {"kind"},
        "fixed": {"incidence_probability_ppm", "kind", "mean_ps"},
        "gaussian": {
            "incidence_probability_ppm",
            "kind",
            "mean_ps",
            "standard_deviation_ps",
        },
        "gaussian_tail_mixture": {
            "incidence_probability_ppm",
            "kind",
            "mean_ps",
            "standard_deviation_ps",
            "tail_mean_ps",
            "tail_probability_ppm",
            "tail_standard_deviation_ps",
        },
    }
    names = fields_by_kind[kind]
    _fields(obj, path, required=names)
    result: dict[str, Any] = {"kind": kind}
    for name in sorted(names - {"kind"}):
        result[name] = _integer(obj[name], f"{path}.{name}", nonnegative=True)
    for name in names - {"kind", "incidence_probability_ppm", "tail_probability_ppm"}:
        if result[name] > (1 << 63) - 1:
            raise ValueError(f"{path}.{name}: exceeds the signed timestamp domain")
    incidence = result.get("incidence_probability_ppm", 0)
    if incidence > 1_000_000:
        raise ValueError(f"{path}.incidence_probability_ppm: exceeds one million ppm")
    if kind != "disabled" and incidence == 0:
        raise ValueError(f"{path}: active profile requires nonzero incidence")
    if kind in {"gaussian", "gaussian_tail_mixture"} and not result[
        "standard_deviation_ps"
    ]:
        raise ValueError(f"{path}: Gaussian profile requires nonzero deviation")
    if kind == "gaussian_tail_mixture":
        tail_probability = result["tail_probability_ppm"]
        if not 0 < tail_probability < 1_000_000:
            raise ValueError(f"{path}.tail_probability_ppm: must be between zero and one")
        if result["tail_mean_ps"] <= result["mean_ps"]:
            raise ValueError(f"{path}.tail_mean_ps: must exceed the main mean")
        if result["tail_standard_deviation_ps"] == 0:
            raise ValueError(f"{path}: tail mixture requires nonzero tail deviation")
    return result


def _penalties(value: Any, path: str) -> dict[str, Any]:
    obj = _object(value, path)
    names = {"acs", "ddio_miss", "gpu_direct", "iommu", "numa", "switch_path"}
    _fields(obj, path, required=names)
    return {
        name: _analytical_profile(obj[name], f"{path}.{name}")
        for name in sorted(names)
    }


def _path(value: Any, path: str) -> dict[str, Any]:
    obj = _object(value, path)
    enabled = _boolean(obj.get("enabled"), f"{path}.enabled")
    if not enabled:
        _fields(obj, path, required={"enabled", "path_id"})
        return {
            "enabled": False,
            "path_id": _integer(obj["path_id"], f"{path}.path_id", minimum=1),
        }
    _fields(
        obj,
        path,
        required={
            "analytical_penalties",
            "base_latency_ps",
            "enabled",
            "endpoint",
            "path_id",
        },
    )
    base_latency = _integer(
        obj["base_latency_ps"], f"{path}.base_latency_ps", nonnegative=True
    )
    if base_latency > (1 << 63) - 1:
        raise ValueError(f"{path}.base_latency_ps: exceeds the signed timestamp domain")
    return {
        "analytical_penalties": _penalties(
            obj["analytical_penalties"], f"{path}.analytical_penalties"
        ),
        "base_latency_ps": base_latency,
        "enabled": True,
        "endpoint": _choice(
            obj["endpoint"],
            {"mmio_bar", "host_pinned_memory", "gpu_memory", "device_memory"},
            f"{path}.endpoint",
        ),
        "path_id": _integer(obj["path_id"], f"{path}.path_id", minimum=1),
    }


def _latency_profile(value: Any, path: str) -> list[int]:
    values = _array(value, path)
    if len(values) != 1:
        raise ValueError(f"{path}: expected exactly one fixed v1 sample")
    sample = _integer(values[0], f"{path}[0]", nonnegative=True)
    if sample > (1 << 63) - 1:
        raise ValueError(f"{path}[0]: exceeds the signed timestamp domain")
    return [sample]


def _fabric(value: Any, path: str) -> dict[str, Any]:
    obj = _object(value, path)
    names = {
        "analytical_seed",
        "completion_buffer_bytes",
        "completion_buffer_release_latency_ps",
        "completion_overhead_bytes",
        "credit_return_latency_ps",
        "data_credit_unit_bytes",
        "device_to_host_credits",
        "generation",
        "host_store_latency_ps",
        "host_to_device_credits",
        "lane_count",
        "max_outstanding_read_requests",
        "max_payload_size_bytes",
        "max_read_request_size_bytes",
        "max_tlps_per_transaction",
        "paths",
        "posted_write_overhead_bytes",
        "posted_write_visibility_latency_ps",
        "read_completion_boundary_bytes",
        "read_completion_latency_ps",
        "read_request_overhead_bytes",
    }
    _fields(obj, path, required=names)
    paths = [
        _path(item, f"{path}.paths[{index}]")
        for index, item in enumerate(_array(obj["paths"], f"{path}.paths"))
    ]
    if not paths:
        raise ValueError(f"{path}.paths: expected at least one path")
    path_ids = [item["path_id"] for item in paths]
    if path_ids != sorted(path_ids) or len(path_ids) != len(set(path_ids)):
        raise ValueError(f"{path}.paths: expected unique ascending path IDs")
    result = {
        name: _integer(obj[name], f"{path}.{name}", nonnegative=True)
        for name in names
        - {
            "device_to_host_credits",
            "generation",
            "host_store_latency_ps",
            "host_to_device_credits",
            "paths",
            "posted_write_visibility_latency_ps",
            "read_completion_latency_ps",
        }
    }
    result.update(
        {
            "device_to_host_credits": _credits(
                obj["device_to_host_credits"], f"{path}.device_to_host_credits"
            ),
            "generation": _integer(obj["generation"], f"{path}.generation", minimum=1),
            "host_store_latency_ps": _latency_profile(
                obj["host_store_latency_ps"], f"{path}.host_store_latency_ps"
            ),
            "host_to_device_credits": _credits(
                obj["host_to_device_credits"], f"{path}.host_to_device_credits"
            ),
            "paths": paths,
            "posted_write_visibility_latency_ps": _latency_profile(
                obj["posted_write_visibility_latency_ps"],
                f"{path}.posted_write_visibility_latency_ps",
            ),
            "read_completion_latency_ps": _latency_profile(
                obj["read_completion_latency_ps"],
                f"{path}.read_completion_latency_ps",
            ),
        }
    )
    if result["generation"] > 5:
        raise ValueError(f"{path}.generation: expected a value in [1, 5]")
    if result["lane_count"] == 0 or result["lane_count"] > 32 or result["lane_count"] & (
        result["lane_count"] - 1
    ):
        raise ValueError(f"{path}.lane_count: expected a power of two up to 32")
    for name in ("max_payload_size_bytes", "max_read_request_size_bytes"):
        size = result[name]
        if size < 128 or size > 4096 or size & (size - 1):
            raise ValueError(f"{path}.{name}: expected a power of two in [128, 4096]")
    if result["read_completion_boundary_bytes"] not in {64, 128}:
        raise ValueError(f"{path}.read_completion_boundary_bytes: expected 64 or 128")
    for name in (
        "posted_write_overhead_bytes",
        "read_request_overhead_bytes",
        "completion_overhead_bytes",
        "max_outstanding_read_requests",
        "completion_buffer_bytes",
        "max_tlps_per_transaction",
    ):
        if result[name] == 0:
            raise ValueError(f"{path}.{name}: must be positive")
    data_credit_unit = result["data_credit_unit_bytes"]
    if data_credit_unit == 0 or data_credit_unit & (data_credit_unit - 1):
        raise ValueError(f"{path}.data_credit_unit_bytes: expected a power of two")
    for direction in ("host_to_device_credits", "device_to_host_credits"):
        credits = result[direction]
        for credit_name in ("posted_data_credits", "completion_data_credits"):
            if credits[credit_name] * data_credit_unit < result["max_payload_size_bytes"]:
                raise ValueError(
                    f"{path}.{direction}.{credit_name}: cannot hold one MPS payload"
                )
    for name in (
        "credit_return_latency_ps",
        "completion_buffer_release_latency_ps",
    ):
        if result[name] > (1 << 63) - 1:
            raise ValueError(f"{path}.{name}: exceeds the signed timestamp domain")
    return result


def _binding(value: Any, path: str) -> dict[str, int]:
    obj = _object(value, path)
    names = {
        "pcie_completion_ordering_domain",
        "pcie_cq_first_byte_offset",
        "pcie_cq_memory_path_id",
        "pcie_cqe_bytes",
        "pcie_doorbell_record_bytes",
        "pcie_doorbell_record_first_byte_offset",
        "pcie_doorbell_record_path_id",
        "pcie_sq_first_byte_offset",
        "pcie_sq_memory_path_id",
        "pcie_submission_ordering_domain",
        "pcie_uar_doorbell_bytes",
        "pcie_uar_first_byte_offset",
        "pcie_uar_path_id",
        "pcie_wqe_bytes",
    }
    _fields(obj, path, required=names)
    result = {
        name: _integer(obj[name], f"{path}.{name}", nonnegative=True)
        for name in sorted(names)
    }
    for name in names:
        if name.endswith(("_path_id", "_bytes", "_ordering_domain")) and result[
            name
        ] == 0:
            raise ValueError(f"{path}.{name}: must be positive")
        if name.endswith("_first_byte_offset") and result[name] >= 4096:
            raise ValueError(f"{path}.{name}: must be below 4096")
    return result


def _effective_hardware(value: Any, path: str) -> dict[str, Any]:
    obj = _object(value, path)
    _fields(obj, path, required={"dma", "network", "qpc", "schema", "work_queue"})
    schema = _string(obj["schema"], f"{path}.schema")
    if schema != RNIC_EFFECTIVE_HARDWARE_SCHEMA:
        raise ValueError(
            f"{path}.schema: expected {RNIC_EFFECTIVE_HARDWARE_SCHEMA!r}, got {schema!r}"
        )

    dma = _object(obj["dma"], f"{path}.dma")
    dma_enabled = _boolean(dma.get("enabled"), f"{path}.dma.enabled")
    if dma_enabled:
        _fields(dma, f"{path}.dma", required={"enabled", "fabric", "fabric_scope", "work_queue"})
        fabric = _fabric(dma["fabric"], f"{path}.dma.fabric")
        binding = _binding(dma["work_queue"], f"{path}.dma.work_queue")
        path_by_id = {item["path_id"]: item for item in fabric["paths"]}
        expected_endpoints = {
            "pcie_uar_path_id": "mmio_bar",
            "pcie_doorbell_record_path_id": "host_pinned_memory",
            "pcie_sq_memory_path_id": "host_pinned_memory",
            "pcie_cq_memory_path_id": "host_pinned_memory",
        }
        for field, endpoint in expected_endpoints.items():
            selected = path_by_id.get(binding[field])
            if selected is None or not selected["enabled"] or selected["endpoint"] != endpoint:
                raise ValueError(
                    f"{path}.dma.work_queue.{field}: references an incompatible path"
                )
        dma_value: dict[str, Any] = {
            "enabled": True,
            "fabric": fabric,
            "fabric_scope": _choice(
                dma["fabric_scope"], {"owned", "shared"}, f"{path}.dma.fabric_scope"
            ),
            "work_queue": binding,
        }
    else:
        _fields(dma, f"{path}.dma", required={"enabled"})
        dma_value = {"enabled": False}

    module_values: dict[str, dict[str, bool]] = {}
    for name in ("network", "qpc"):
        module = _object(obj[name], f"{path}.{name}")
        _fields(module, f"{path}.{name}", required={"enabled"})
        module_values[name] = {
            "enabled": _boolean(module["enabled"], f"{path}.{name}.enabled")
        }

    work_queue = _object(obj["work_queue"], f"{path}.work_queue")
    queue_fields = {"cq_depth", "scheduler_service_ps", "sq_depth"}
    if module_values["qpc"]["enabled"]:
        queue_fields.add("qpc_lookup_service_ps")
    if not dma_enabled:
        queue_fields.update(
            {"cqe_write_service_ps", "doorbell_service_ps", "wqe_fetch_service_ps"}
        )
    _fields(work_queue, f"{path}.work_queue", required=queue_fields)
    queue_value = {
        name: _integer(
            work_queue[name],
            f"{path}.work_queue.{name}",
            nonnegative=name not in {"sq_depth", "cq_depth"},
            minimum=1 if name in {"sq_depth", "cq_depth"} else None,
        )
        for name in sorted(queue_fields)
    }
    for name, item in queue_value.items():
        if name.endswith("_ps") and item > (1 << 63) - 1:
            raise ValueError(
                f"{path}.work_queue.{name}: exceeds the signed timestamp domain"
            )
    return {
        "dma": dma_value,
        "network": module_values["network"],
        "qpc": module_values["qpc"],
        "schema": schema,
        "work_queue": queue_value,
    }


def _config_identity(
    obj: Mapping[str, Any], path: str
) -> tuple[str, RnicHardwareMode, RnicWqeAuthority, str, str | None]:
    session_id = _string(obj["session_id"], f"{path}.session_id")
    mode = _enum_value(RnicHardwareMode, obj["hardware_mode"], f"{path}.hardware_mode")
    authority = _enum_value(
        RnicWqeAuthority, obj["authority"], f"{path}.authority"
    )
    transport_policy = _string(obj["transport_policy"], f"{path}.transport_policy")
    digest = None
    if obj["hardware_config_sha256"] is not None:
        digest = _sha256(obj["hardware_config_sha256"], f"{path}.hardware_config_sha256")
    if mode is RnicHardwareMode.STRUCTURAL:
        if authority is not RnicWqeAuthority.NATIVE or digest is None:
            raise ValueError(f"{path}: structural mode requires native authority and hash")
    elif authority is not RnicWqeAuthority.ATLAHS_LEDGER or digest is not None:
        raise ValueError(f"{path}: bypass mode requires AtlahsWqeLedger and no hash")
    return session_id, mode, authority, transport_policy, digest


def rnic_session_config_from_json(value: Any) -> RnicSessionConfigRecord:
    """Parse and validate one strict RNIC session configuration record."""

    path = "rnic_session_config"
    obj = _object(value, path)
    _fields(
        obj,
        path,
        required={
            "authority",
            "effective_hardware",
            "hardware_config_sha256",
            "hardware_mode",
            "schema",
            "session_id",
            "transport_policy",
        },
    )
    schema = _string(obj["schema"], f"{path}.schema")
    if schema != RNIC_SESSION_CONFIG_SCHEMA:
        raise ValueError(
            f"{path}.schema: expected {RNIC_SESSION_CONFIG_SCHEMA!r}, got {schema!r}"
        )
    session_id, mode, authority, policy, digest = _config_identity(obj, path)
    effective = None
    if obj["effective_hardware"] is not None:
        effective = _effective_hardware(
            obj["effective_hardware"], f"{path}.effective_hardware"
        )
    if mode is RnicHardwareMode.STRUCTURAL:
        if effective is None:
            raise ValueError(f"{path}.effective_hardware: required in structural mode")
        actual = hashlib.sha256(_canonical_json(effective)).hexdigest()
        if digest != actual:
            raise ValueError(f"{path}.hardware_config_sha256: digest mismatch")
    elif effective is not None:
        raise ValueError(f"{path}.effective_hardware: must be null in bypass mode")
    return RnicSessionConfigRecord(
        session_id,
        mode,
        authority,
        policy,
        digest,
        _freeze_json(effective) if effective is not None else None,
    )


def _authority_counters(value: Any, path: str) -> RnicAuthorityCounters:
    obj = _object(value, path)
    names = {
        "native_session_constructed",
        "legacy_ledger_constructed",
        "native_posts",
        "legacy_mutations",
    }
    _fields(obj, path, required=names)
    values = {
        name: _integer(obj[name], f"{path}.{name}", nonnegative=True)
        for name in names
    }
    return RnicAuthorityCounters(**values)


def _wqe_key(value: Any, path: str) -> RnicWqeProjectionKey:
    obj = _object(value, path)
    _fields(
        obj,
        path,
        required={"endpoint", "post_sequence", "session_id", "wq_id", "wq_kind"},
    )
    return RnicWqeProjectionKey(
        session_id=_string(obj["session_id"], f"{path}.session_id"),
        endpoint=_integer(obj["endpoint"], f"{path}.endpoint", nonnegative=True),
        wq_kind=_choice(
            obj["wq_kind"], {"send", "receive", "shared_receive"}, f"{path}.wq_kind"
        ),
        wq_id=_integer(obj["wq_id"], f"{path}.wq_id", minimum=1),
        post_sequence=_integer(
            obj["post_sequence"], f"{path}.post_sequence", minimum=1
        ),
    )


def _timeline(value: Any, path: str) -> RnicWqeTimeline:
    obj = _object(value, path)
    _fields(obj, path, required=_WQE_TIMELINE_FIELDS)
    values = {
        name: _optional_integer(obj[name], f"{path}.{name}", nonnegative=True)
        for name in _WQE_TIMELINE_FIELDS - {"posted_at_ps"}
    }
    values["posted_at_ps"] = _integer(
        obj["posted_at_ps"], f"{path}.posted_at_ps", nonnegative=True
    )
    timeline = RnicWqeTimeline(**values)
    ordered = (
        timeline.doorbelled_at_ps,
        timeline.doorbell_seen_at_ps,
        timeline.wqe_fetch_begin_at_ps,
        timeline.wqe_fetch_end_at_ps,
        timeline.qpc_ready_at_ps,
        timeline.admitted_at_ps,
        timeline.network_accepted_at_ps,
        timeline.first_packet_at_ps,
        timeline.last_packet_at_ps,
        timeline.network_outcome_at_ps,
        timeline.transport_retired_at_ps,
        timeline.cqe_visible_at_ps,
        timeline.polled_at_ps,
        timeline.sq_reclaimed_at_ps,
    )
    previous = timeline.posted_at_ps
    for timestamp in ordered:
        if timestamp is None:
            continue
        if timestamp < previous:
            raise ValueError(f"{path}: timestamps must be monotonic")
        previous = timestamp
    if (timeline.first_packet_at_ps is None) != (timeline.last_packet_at_ps is None):
        raise ValueError(f"{path}: first and last packet timestamps must appear together")
    if timeline.first_packet_at_ps is not None and timeline.network_accepted_at_ps is None:
        raise ValueError(f"{path}: packet timestamps require network acceptance")
    return timeline


def _wqe(value: Any, path: str) -> RnicWqeProjectionRecord:
    obj = _object(value, path)
    names = {
        "completion_status",
        "cq_consume_sequence",
        "cq_id",
        "cq_producer_index",
        "cqe_sequence",
        "destination",
        "flow_id",
        "flow_tag",
        "key",
        "opcode",
        "payload_bytes",
        "qpn",
        "signaled",
        "source",
        "state",
        "timeline",
        "transport_kind",
        "transport_object_id",
        "wqe_id",
        "wr_id",
    }
    _fields(obj, path, required=names)
    completion_status = _optional_string(
        obj["completion_status"], f"{path}.completion_status"
    )
    if completion_status is not None:
        completion_status = _choice(
            completion_status,
            {"success", "transport_error", "network_rejected"},
            f"{path}.completion_status",
        )
    record = RnicWqeProjectionRecord(
        key=_wqe_key(obj["key"], f"{path}.key"),
        wqe_id=_integer(obj["wqe_id"], f"{path}.wqe_id", minimum=1),
        wr_id=_optional_integer(obj["wr_id"], f"{path}.wr_id", nonnegative=True),
        flow_id=_integer(obj["flow_id"], f"{path}.flow_id", nonnegative=True),
        flow_tag=_integer(obj["flow_tag"], f"{path}.flow_tag", nonnegative=True),
        source=_integer(obj["source"], f"{path}.source", nonnegative=True),
        destination=_integer(
            obj["destination"], f"{path}.destination", nonnegative=True
        ),
        payload_bytes=_integer(
            obj["payload_bytes"], f"{path}.payload_bytes", nonnegative=True
        ),
        qpn=_optional_integer(obj["qpn"], f"{path}.qpn", minimum=1),
        signaled=_boolean(obj["signaled"], f"{path}.signaled"),
        opcode=_choice(obj["opcode"], {"send"}, f"{path}.opcode"),
        timeline=_timeline(obj["timeline"], f"{path}.timeline"),
        state=_choice(
            obj["state"],
            {
                "posted",
                "doorbelled",
                "in_flight",
                "awaiting_ordered_retirement",
                "retired_unsignaled",
                "completion_pending",
                "cqe_visible",
                "reclaimed",
                "completed",
                "error",
            },
            f"{path}.state",
        ),
        completion_status=completion_status,
        cq_id=_integer(obj["cq_id"], f"{path}.cq_id", minimum=1),
        cqe_sequence=_optional_integer(
            obj["cqe_sequence"], f"{path}.cqe_sequence", minimum=1
        ),
        cq_producer_index=_optional_integer(
            obj["cq_producer_index"], f"{path}.cq_producer_index", nonnegative=True
        ),
        cq_consume_sequence=_optional_integer(
            obj["cq_consume_sequence"], f"{path}.cq_consume_sequence", minimum=1
        ),
        transport_kind=_string(obj["transport_kind"], f"{path}.transport_kind"),
        transport_object_id=_integer(
            obj["transport_object_id"], f"{path}.transport_object_id", nonnegative=True
        ),
    )
    if record.key.session_id.strip() == "" or record.key.endpoint != record.source:
        raise ValueError(f"{path}.key: stable key disagrees with WQE fields")
    if (record.transport_kind == "none") != (record.transport_object_id == 0):
        raise ValueError(f"{path}: transport kind none must match object ID zero")
    cqe_presence = {
        record.cqe_sequence is not None,
        record.cq_producer_index is not None,
        record.cq_consume_sequence is not None,
    }
    if len(cqe_presence) != 1:
        raise ValueError(f"{path}: CQ sequence fields must be present together")
    has_cqe = record.cqe_sequence is not None
    requires_cqe = record.signaled or record.completion_status != "success"
    if has_cqe != requires_cqe:
        raise ValueError(f"{path}: signaling and status must match CQ sequence presence")
    if (record.timeline.cqe_visible_at_ps is not None) != requires_cqe or (
        record.timeline.polled_at_ps is not None
    ) != requires_cqe:
        raise ValueError(f"{path}: signaling and status must match the CQ timeline")
    if has_cqe and record.cqe_sequence != record.cq_consume_sequence:
        raise ValueError(f"{path}: CQ post and consume sequences must match")
    if record.completion_status is None:
        raise ValueError(f"{path}: terminal WQE requires a completion status")
    if (
        record.timeline.network_outcome_at_ps is None
        or record.timeline.transport_retired_at_ps is None
    ):
        raise ValueError(f"{path}: terminal WQE requires outcome and retirement")
    return record


def _completion(value: Any, path: str) -> RnicCompletionProjection:
    obj = _object(value, path)
    names = {
        "completion_time_ps",
        "cq_consume_sequence",
        "cq_id",
        "cq_post_sequence",
        "destination",
        "fct_ps",
        "flow_id",
        "payload_bytes",
        "profile",
        "rq_id",
        "source",
        "sq_dispatch_sequence",
        "sq_id",
        "sq_post_sequence",
        "start_time_ps",
        "tag",
        "transport_kind",
        "transport_object_id",
        "wqe_id",
    }
    _fields(obj, path, required=names)
    record = RnicCompletionProjection(
        profile=_string(obj["profile"], f"{path}.profile"),
        flow_id=_integer(obj["flow_id"], f"{path}.flow_id", nonnegative=True),
        source=_integer(obj["source"], f"{path}.source", nonnegative=True),
        destination=_integer(
            obj["destination"], f"{path}.destination", nonnegative=True
        ),
        tag=_integer(obj["tag"], f"{path}.tag", nonnegative=True),
        payload_bytes=_integer(
            obj["payload_bytes"], f"{path}.payload_bytes", nonnegative=True
        ),
        start_time_ps=_integer(
            obj["start_time_ps"], f"{path}.start_time_ps", nonnegative=True
        ),
        completion_time_ps=_integer(
            obj["completion_time_ps"],
            f"{path}.completion_time_ps",
            nonnegative=True,
        ),
        fct_ps=_integer(obj["fct_ps"], f"{path}.fct_ps", nonnegative=True),
        wqe_id=_optional_integer(obj["wqe_id"], f"{path}.wqe_id", minimum=1),
        sq_id=_optional_integer(obj["sq_id"], f"{path}.sq_id", minimum=1),
        rq_id=_optional_integer(obj["rq_id"], f"{path}.rq_id", minimum=1),
        cq_id=_optional_integer(obj["cq_id"], f"{path}.cq_id", minimum=1),
        sq_post_sequence=_optional_integer(
            obj["sq_post_sequence"], f"{path}.sq_post_sequence", minimum=1
        ),
        sq_dispatch_sequence=_optional_integer(
            obj["sq_dispatch_sequence"], f"{path}.sq_dispatch_sequence", minimum=1
        ),
        cq_post_sequence=_optional_integer(
            obj["cq_post_sequence"], f"{path}.cq_post_sequence", minimum=1
        ),
        cq_consume_sequence=_optional_integer(
            obj["cq_consume_sequence"], f"{path}.cq_consume_sequence", minimum=1
        ),
        transport_kind=_optional_string(
            obj["transport_kind"], f"{path}.transport_kind"
        ),
        transport_object_id=_optional_integer(
            obj["transport_object_id"],
            f"{path}.transport_object_id",
            nonnegative=True,
        ),
    )
    if (
        record.completion_time_ps < record.start_time_ps
        or record.fct_ps != record.completion_time_ps - record.start_time_ps
    ):
        raise ValueError(f"{path}: FCT does not match completion minus start")
    return record


def _validate_projection_set(
    *,
    path: str,
    session_id: str,
    mode: RnicHardwareMode,
    policy: str,
    wqes: tuple[RnicWqeProjectionRecord, ...],
    completions: tuple[RnicCompletionProjection, ...] | None,
) -> None:
    keys = [wqe.key for wqe in wqes]
    ids = [wqe.wqe_id for wqe in wqes]
    if len(keys) != len(set(keys)) or len(ids) != len(set(ids)):
        raise ValueError(f"{path}.wqes: duplicate stable key or WQE ID")
    if any(wqe.key.session_id != session_id or wqe.key.wq_kind != "send" for wqe in wqes):
        raise ValueError(f"{path}.wqes: key does not belong to the session send queue")
    if mode is RnicHardwareMode.STRUCTURAL and any(wqe.qpn is None for wqe in wqes):
        raise ValueError(f"{path}.wqes: structural WQE requires a QP identity")
    if any(
        (
            (wqe.signaled or wqe.completion_status != "success")
            and wqe.state != "completed"
        )
        or (
            not wqe.signaled
            and wqe.completion_status == "success"
            and wqe.state != "reclaimed"
        )
        for wqe in wqes
    ):
        raise ValueError(f"{path}.wqes: result contains a nonterminal WQE state")
    cqe_sequences = [
        (wqe.cq_id, wqe.cqe_sequence)
        for wqe in wqes
        if wqe.cqe_sequence is not None
    ]
    if len(cqe_sequences) != len(set(cqe_sequences)):
        raise ValueError(f"{path}.wqes: duplicate CQE sequence")
    if completions is None:
        return
    ordered = sorted(completions, key=lambda row: (row.flow_id, row.wqe_id or 0))
    if list(completions) != ordered:
        raise ValueError(f"{path}.completion_rows: rows are not sorted by flow ID")
    if len(completions) != len(wqes):
        raise ValueError(f"{path}: completion and WQE projections are not one to one")
    by_id = {wqe.wqe_id: wqe for wqe in wqes}
    seen: set[int] = set()
    for index, row in enumerate(completions):
        row_path = f"{path}.completion_rows[{index}]"
        if row.wqe_id is None or row.wqe_id in seen or row.wqe_id not in by_id:
            raise ValueError(f"{row_path}.wqe_id: unknown or duplicate WQE")
        seen.add(row.wqe_id)
        wqe = by_id[row.wqe_id]
        expected = (
            policy,
            wqe.flow_id,
            wqe.source,
            wqe.destination,
            wqe.flow_tag,
            wqe.payload_bytes,
            wqe.timeline.posted_at_ps,
            wqe.timeline.network_outcome_at_ps,
            wqe.key.wq_id,
            wqe.cq_id,
            wqe.key.post_sequence,
            wqe.cqe_sequence,
            wqe.cq_consume_sequence,
            wqe.transport_kind,
            wqe.transport_object_id,
        )
        actual = (
            row.profile,
            row.flow_id,
            row.source,
            row.destination,
            row.tag,
            row.payload_bytes,
            row.start_time_ps,
            row.completion_time_ps,
            row.sq_id,
            row.cq_id,
            row.sq_post_sequence,
            row.cq_post_sequence,
            row.cq_consume_sequence,
            row.transport_kind,
            row.transport_object_id,
        )
        if actual != expected or row.sq_dispatch_sequence != row.sq_post_sequence:
            raise ValueError(f"{row_path}: completion row disagrees with WQE projection")
        if mode is RnicHardwareMode.STRUCTURAL and row.rq_id is not None:
            raise ValueError(f"{row_path}.rq_id: structural send cannot own a receive WQ")


def rnic_session_result_from_json(value: Any) -> RnicSessionResultRecord:
    """Parse and reconcile one strict RNIC session result record."""

    path = "rnic_session_result"
    obj = _object(value, path)
    _fields(
        obj,
        path,
        required={
            "authority",
            "authority_counters",
            "completion_rows",
            "hardware_config_sha256",
            "hardware_mode",
            "quiescent",
            "schema",
            "session_id",
            "transport_policy",
            "wqes",
        },
    )
    schema = _string(obj["schema"], f"{path}.schema")
    if schema != RNIC_SESSION_RESULT_SCHEMA:
        raise ValueError(
            f"{path}.schema: expected {RNIC_SESSION_RESULT_SCHEMA!r}, got {schema!r}"
        )
    session_id, mode, authority, policy, digest = _config_identity(obj, path)
    counters = _authority_counters(obj["authority_counters"], f"{path}.authority_counters")
    wqes = tuple(
        _wqe(item, f"{path}.wqes[{index}]")
        for index, item in enumerate(_array(obj["wqes"], f"{path}.wqes"))
    )
    completions = tuple(
        _completion(item, f"{path}.completion_rows[{index}]")
        for index, item in enumerate(
            _array(obj["completion_rows"], f"{path}.completion_rows")
        )
    )
    if mode is RnicHardwareMode.STRUCTURAL:
        expected = (1, 0, len(wqes), 0)
    else:
        expected = (0, 1, 0, len(wqes) * 2)
    actual = (
        counters.native_session_constructed,
        counters.legacy_ledger_constructed,
        counters.native_posts,
        counters.legacy_mutations,
    )
    if actual != expected:
        raise ValueError(f"{path}.authority_counters: mode-exclusivity mismatch")
    _validate_projection_set(
        path=path,
        session_id=session_id,
        mode=mode,
        policy=policy,
        wqes=wqes,
        completions=completions,
    )
    return RnicSessionResultRecord(
        session_id,
        mode,
        authority,
        policy,
        digest,
        counters,
        _boolean(obj["quiescent"], f"{path}.quiescent"),
        wqes,
        completions,
    )


def rnic_bookkeeping_projection_from_json(value: Any) -> RnicBookkeepingProjection:
    """Parse the read-only WQE bookkeeping projection of a session result."""

    path = "rnic_bookkeeping"
    obj = _object(value, path)
    _fields(
        obj,
        path,
        required={
            "authority",
            "hardware_config_sha256",
            "hardware_mode",
            "schema",
            "session_id",
            "wqes",
        },
    )
    schema = _string(obj["schema"], f"{path}.schema")
    if schema != RNIC_BOOKKEEPING_SCHEMA:
        raise ValueError(
            f"{path}.schema: expected {RNIC_BOOKKEEPING_SCHEMA!r}, got {schema!r}"
        )
    shim = dict(obj)
    shim["transport_policy"] = "bookkeeping-projection"
    session_id, mode, authority, _, digest = _config_identity(shim, path)
    wqes = tuple(
        _wqe(item, f"{path}.wqes[{index}]")
        for index, item in enumerate(_array(obj["wqes"], f"{path}.wqes"))
    )
    _validate_projection_set(
        path=path,
        session_id=session_id,
        mode=mode,
        policy="bookkeeping-projection",
        wqes=wqes,
        completions=None,
    )
    return RnicBookkeepingProjection(session_id, mode, authority, digest, wqes)


def canonical_bypass_parameters(
    parameters: Mapping[str, str | int | bool],
) -> tuple[tuple[str, str], ...]:
    """Canonicalize path-free semantic knobs from a structured run config.

    Executable, GOAL, topology and output paths are deliberately not accepted.
    Profile and seed have their own explicit identity fields.
    """

    if not isinstance(parameters, Mapping) or not parameters:
        raise TypeError("parameters: expected a nonempty mapping")
    result: list[tuple[str, str]] = []
    for name, value in parameters.items():
        if type(name) is not str or _BYPASS_PARAMETER_RE.fullmatch(name) is None:
            raise ValueError(
                "parameters: names must be lowercase semantic identifiers"
            )
        if name in _BYPASS_PARAMETER_EXCLUSIONS or name.endswith(
            (
                "_argv",
                "_binary",
                "_command",
                "_dir",
                "_executable",
                "_file",
                "_output",
                "_path",
            )
        ):
            raise ValueError(f"parameters.{name}: diagnostic identity is excluded")
        if type(value) is bool:
            rendered = "true" if value else "false"
        elif type(value) is int and value >= 0:
            rendered = str(value)
        elif (
            type(value) is str
            and value
            and "\x00" not in value
            and "/" not in value
            and "\\" not in value
            and not value.startswith(("-", ".", "~"))
            and not value.endswith(
                (".bin", ".csv", ".goal", ".json", ".txt")
            )
        ):
            rendered = value
        else:
            raise TypeError(
                f"parameters.{name}: expected a nonnegative int, bool or nonempty string"
            )
        result.append((name, rendered))
    result.sort()
    return tuple(result)


def read_bypass_artifacts(paths: BypassArtifactPaths) -> BypassArtifacts:
    """Load one immutable bypass comparison bundle from explicit paths."""

    if not isinstance(paths, BypassArtifactPaths):
        raise TypeError("paths: expected BypassArtifactPaths")
    artifacts = BypassArtifacts(
        goal_text=paths.goal_text.read_bytes(),
        goal_binary=paths.goal_binary.read_bytes(),
        topology=paths.topology.read_bytes(),
        profile=paths.profile,
        seed=paths.seed,
        baseline_parameters=paths.baseline_parameters,
        completion_csv=paths.completion_csv.read_bytes(),
        canonical_completion=paths.canonical_completion.read_bytes(),
        step_results=paths.step_results.read_bytes(),
        replay_summary=paths.replay_summary.read_bytes(),
    )
    _validate_artifacts(artifacts, "paths")
    return artifacts


def _validate_artifacts(value: BypassArtifacts, path: str) -> None:
    if not isinstance(value, BypassArtifacts):
        raise TypeError(f"{path}: expected BypassArtifacts")
    for name in (
        "goal_text",
        "goal_binary",
        "topology",
        "completion_csv",
        "canonical_completion",
        "step_results",
        "replay_summary",
    ):
        payload = getattr(value, name)
        if type(payload) is not bytes:
            raise TypeError(f"{path}.{name}: expected bytes")
    if not isinstance(value.profile, str) or not value.profile.strip():
        raise TypeError(f"{path}.profile: expected a nonblank string")
    if type(value.seed) is not int or value.seed < 0:
        raise TypeError(f"{path}.seed: expected a nonnegative integer")
    parameters = value.baseline_parameters
    if (
        not isinstance(parameters, tuple)
        or not parameters
        or any(
            not isinstance(item, tuple)
            or len(item) != 2
            or type(item[0]) is not str
            or type(item[1]) is not str
            for item in parameters
        )
    ):
        raise TypeError(
            f"{path}.baseline_parameters: expected canonical string pairs"
        )
    names = [name for name, _ in parameters]
    if len(names) != len(set(names)):
        raise ValueError(f"{path}.baseline_parameters: duplicate semantic key")
    canonical = canonical_bypass_parameters(dict(parameters))
    if parameters != canonical:
        raise ValueError(
            f"{path}.baseline_parameters: pairs are not canonical and sorted"
        )


def compare_bypass_artifacts(
    reference: BypassArtifacts, candidate: BypassArtifacts
) -> BypassArtifactComparison:
    """Compare exactly the frozen bypass inputs and behavioral artifacts."""

    _validate_artifacts(reference, "reference")
    _validate_artifacts(candidate, "candidate")
    return BypassArtifactComparison(
        input_matches=tuple(
            (name, getattr(reference, name) == getattr(candidate, name))
            for name in (
                "goal_text",
                "goal_binary",
                "topology",
                "profile",
                "seed",
                "baseline_parameters",
            )
        ),
        behavioral_matches=tuple(
            (name, getattr(reference, name) == getattr(candidate, name))
            for name in (
                "completion_csv",
                "canonical_completion",
                "step_results",
                "replay_summary",
            )
        ),
    )


def assert_bypass_artifact_identity(
    reference: BypassArtifacts, candidate: BypassArtifacts
) -> BypassArtifactComparison:
    """Return an exact comparison or raise with the changed byte classes."""

    comparison = compare_bypass_artifacts(reference, candidate)
    if not comparison.equivalent:
        raise ValueError(
            "bypass artifact identity failed: "
            f"changed_inputs={list(comparison.changed_inputs)}, "
            f"changed_artifacts={list(comparison.changed_artifacts)}"
        )
    return comparison


def _completion_csv_rows(payload: str) -> tuple[tuple[str, ...], ...]:
    reader = csv.reader(io.StringIO(payload, newline=""))
    return tuple(tuple(row) for row in reader)


def _render_completion_csv(rows: Sequence[RnicCompletionProjection]) -> str:
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(_COMPLETION_CSV_HEADER)
    for row in rows:
        writer.writerow(
            (
                row.profile,
                row.flow_id,
                row.source,
                row.destination,
                row.tag,
                row.payload_bytes,
                row.start_time_ps,
                row.completion_time_ps,
                row.fct_ps,
                row.wqe_id,
                row.sq_id,
                row.rq_id,
                row.cq_id,
                row.sq_post_sequence,
                row.sq_dispatch_sequence,
                row.cq_post_sequence,
                row.cq_consume_sequence,
                row.transport_kind,
                row.transport_object_id,
            )
        )
    return output.getvalue()


def _true_guard_map(
    value: Any, path: str, expected_names: frozenset[str]
) -> dict[str, bool]:
    obj = _object(value, path)
    _fields(obj, path, required=set(expected_names))
    result = {name: _boolean(obj[name], f"{path}.{name}") for name in sorted(obj)}
    if not all(result.values()):
        failed = [name for name, passed in result.items() if not passed]
        raise AssertionError(f"{path}: failed guards {failed}")
    return result


def validate_session_record_study(
    native: Mapping[str, Any],
    *,
    policies: Sequence[str],
    sq_depths: Sequence[int],
    doorbell_services_ps: Sequence[int],
    artifact_names: Sequence[str],
) -> dict[str, Any]:
    """Validate the frozen component-study report emitted by the native test."""

    report = _object(native, "study")
    _fields(
        report,
        "study",
        required={
            "authority_negative_controls",
            "bypass_bookkeeping",
            "bypass_config",
            "bypass_result",
            "dma_config",
            "hash_rows",
            "inactive_hash_guards",
            "projection_checks",
            "schema",
            "sensitivity_rows",
            "structural_bookkeeping",
            "structural_completion_csv",
            "structural_config",
            "structural_result",
        },
    )
    if report["schema"] != "simllm-rnic-session-record-study-v1":
        raise ValueError("study.schema: unexpected native study schema")

    expected_cells = {
        (sq_depth, doorbell, policy)
        for sq_depth in sq_depths
        for doorbell in doorbell_services_ps
        for policy in policies
    }
    hashes: dict[tuple[int, int, str], str] = {}
    for index, item in enumerate(_array(report["hash_rows"], "study.hash_rows")):
        path = f"study.hash_rows[{index}]"
        row = _object(item, path)
        _fields(
            row,
            path,
            required={"config", "doorbell_service_ps", "policy", "sq_depth"},
        )
        config = rnic_session_config_from_json(row["config"])
        key = (
            _integer(row["sq_depth"], f"{path}.sq_depth", minimum=1),
            _integer(
                row["doorbell_service_ps"],
                f"{path}.doorbell_service_ps",
                nonnegative=True,
            ),
            _string(row["policy"], f"{path}.policy"),
        )
        if config.transport_policy != key[2] or config.hardware_config_sha256 is None:
            raise ValueError(f"{path}.config: row identity mismatch")
        if key in hashes:
            raise ValueError(f"{path}: duplicate hash cell")
        hashes[key] = config.hardware_config_sha256
    if set(hashes) != expected_cells:
        raise ValueError("study.hash_rows: frozen 2 by 2 by 3 grid mismatch")

    policy_pair_checks = 0
    cell_hashes: dict[tuple[int, int], str] = {}
    for sq_depth in sq_depths:
        for doorbell in doorbell_services_ps:
            values = [hashes[(sq_depth, doorbell, policy)] for policy in policies]
            policy_pair_checks += len(values) * (len(values) - 1) // 2
            if len(set(values)) != 1:
                raise AssertionError("transport policy changed the hardware hash")
            cell_hashes[(sq_depth, doorbell)] = values[0]
    if len(set(cell_hashes.values())) != len(cell_hashes):
        raise AssertionError("distinct hardware cells collided")
    adjacent_changes = []
    for sq_depth in sq_depths:
        for first, second in itertools.pairwise(doorbell_services_ps):
            adjacent_changes.append(
                cell_hashes[(sq_depth, first)] != cell_hashes[(sq_depth, second)]
            )
    for doorbell in doorbell_services_ps:
        for first, second in itertools.pairwise(sq_depths):
            adjacent_changes.append(
                cell_hashes[(first, doorbell)] != cell_hashes[(second, doorbell)]
            )
    if not all(adjacent_changes):
        raise AssertionError("an adjacent hardware-axis mutation did not change the hash")

    sensitivity_rows: list[dict[str, str]] = []
    observed_sensitivity: dict[str, set[str]] = {
        group: set() for group in _EXPECTED_SENSITIVITY_BY_GROUP
    }
    for index, item in enumerate(
        _array(report["sensitivity_rows"], "study.sensitivity_rows")
    ):
        path = f"study.sensitivity_rows[{index}]"
        row = _object(item, path)
        _fields(row, path, required={"after_hash", "before_hash", "field", "group"})
        field = _string(row["field"], f"{path}.field")
        group = _string(row["group"], f"{path}.group")
        before = _sha256(row["before_hash"], f"{path}.before_hash")
        after = _sha256(row["after_hash"], f"{path}.after_hash")
        if before == after:
            raise AssertionError(f"effective hardware mutation did not change hash: {field}")
        if group not in observed_sensitivity:
            raise AssertionError(f"unexpected effective-field sensitivity group: {group}")
        if field in observed_sensitivity[group]:
            raise AssertionError(f"duplicate effective-field sensitivity label: {field}")
        observed_sensitivity[group].add(field)
        sensitivity_rows.append(
            {
                "after_hash": after,
                "before_hash": before,
                "field": field,
                "group": group,
            }
        )
    expected_sensitivity = {
        group: set(fields) for group, fields in _EXPECTED_SENSITIVITY_BY_GROUP.items()
    }
    if observed_sensitivity != expected_sensitivity:
        missing = {
            group: sorted(expected_sensitivity[group] - observed_sensitivity[group])
            for group in expected_sensitivity
            if expected_sensitivity[group] - observed_sensitivity[group]
        }
        extra = {
            group: sorted(observed_sensitivity[group] - expected_sensitivity[group])
            for group in expected_sensitivity
            if observed_sensitivity[group] - expected_sensitivity[group]
        }
        raise AssertionError(
            f"effective-field sensitivity census mismatch: missing={missing}, extra={extra}"
        )

    guards = _true_guard_map(
        report["inactive_hash_guards"],
        "study.inactive_hash_guards",
        _EXPECTED_INACTIVE_HASH_GUARDS,
    )

    dma_config = rnic_session_config_from_json(report["dma_config"])
    if (
        dma_config.effective_hardware is None
        or not dma_config.effective_hardware["dma"]["enabled"]
    ):
        raise AssertionError("DMA reader fixture did not retain enabled hardware")
    structural_config = rnic_session_config_from_json(report["structural_config"])
    bypass_config = rnic_session_config_from_json(report["bypass_config"])
    structural_result = rnic_session_result_from_json(report["structural_result"])
    bypass_result = rnic_session_result_from_json(report["bypass_result"])
    structural_bookkeeping = rnic_bookkeeping_projection_from_json(
        report["structural_bookkeeping"]
    )
    bypass_bookkeeping = rnic_bookkeeping_projection_from_json(
        report["bypass_bookkeeping"]
    )
    structural_identity = (
        structural_config.session_id,
        structural_config.hardware_mode,
        structural_config.authority,
        structural_config.transport_policy,
        structural_config.hardware_config_sha256,
    )
    bypass_identity = (
        bypass_config.session_id,
        bypass_config.hardware_mode,
        bypass_config.authority,
        bypass_config.transport_policy,
        bypass_config.hardware_config_sha256,
    )
    if structural_identity != (
        structural_result.session_id,
        structural_result.hardware_mode,
        structural_result.authority,
        structural_result.transport_policy,
        structural_result.hardware_config_sha256,
    ):
        raise AssertionError("structural config/result identity mismatch")
    if bypass_identity != (
        bypass_result.session_id,
        bypass_result.hardware_mode,
        bypass_result.authority,
        bypass_result.transport_policy,
        bypass_result.hardware_config_sha256,
    ):
        raise AssertionError("bypass config/result identity mismatch")
    if (
        structural_bookkeeping.session_id,
        structural_bookkeeping.hardware_mode,
        structural_bookkeeping.authority,
        structural_bookkeeping.hardware_config_sha256,
        structural_bookkeeping.wqes,
    ) != (
        structural_result.session_id,
        structural_result.hardware_mode,
        structural_result.authority,
        structural_result.hardware_config_sha256,
        structural_result.wqes,
    ):
        raise AssertionError("structural bookkeeping projection drifted")
    if (
        bypass_bookkeeping.session_id,
        bypass_bookkeeping.hardware_mode,
        bypass_bookkeeping.authority,
        bypass_bookkeeping.hardware_config_sha256,
        bypass_bookkeeping.wqes,
    ) != (
        bypass_result.session_id,
        bypass_result.hardware_mode,
        bypass_result.authority,
        bypass_result.hardware_config_sha256,
        bypass_result.wqes,
    ):
        raise AssertionError("bypass bookkeeping projection drifted")
    if bypass_config.hardware_mode is not RnicHardwareMode.BYPASS:
        raise AssertionError("bypass config is not explicitly labeled")
    if not structural_result.quiescent or not bypass_result.quiescent:
        raise AssertionError("study result records must be quiescent")
    structural_counters = structural_result.authority_counters
    bypass_counters = bypass_result.authority_counters
    if len(structural_result.wqes) != 2 or (
        structural_counters.native_session_constructed,
        structural_counters.legacy_ledger_constructed,
        structural_counters.native_posts,
        structural_counters.legacy_mutations,
    ) != (1, 0, 2, 0):
        raise AssertionError(
            "structural study fixture must retain two WQEs and counters (1, 0, 2, 0)"
        )
    if len(bypass_result.wqes) != 2 or (
        bypass_counters.native_session_constructed,
        bypass_counters.legacy_ledger_constructed,
        bypass_counters.native_posts,
        bypass_counters.legacy_mutations,
    ) != (0, 1, 0, 4):
        raise AssertionError(
            "bypass study fixture must retain two WQEs and counters (0, 1, 0, 4)"
        )

    projection_checks = _true_guard_map(
        report["projection_checks"],
        "study.projection_checks",
        _EXPECTED_PROJECTION_CHECKS,
    )
    negative = _true_guard_map(
        report["authority_negative_controls"],
        "study.authority_negative_controls",
        _EXPECTED_AUTHORITY_NEGATIVE_CONTROLS,
    )

    completion_csv = _string(
        report["structural_completion_csv"], "study.structural_completion_csv"
    )
    if "\r" in completion_csv or completion_csv != _render_completion_csv(
        structural_result.completion_rows
    ):
        raise AssertionError("completion CSV bytes disagree with the result projection")
    csv_rows = _completion_csv_rows(completion_csv)
    if len(csv_rows) != len(structural_result.completion_rows) + 1:
        raise AssertionError("completion CSV row count disagrees with result")

    reference = BypassArtifacts(
        goal_text=b"num_ranks 2\nrank 0 { send 4096b to 1 tag 7 }\n",
        goal_binary=b"\x00GOAL\x01fixture",
        topology=b"nodes 2\nlinks 0 1 400G\n",
        profile="rnic-nn-fluid",
        seed=7,
        baseline_parameters=canonical_bypass_parameters(
            {"linkspeed_bps": 400_000_000_000}
        ),
        completion_csv=b"profile,flow_id\nrnic-nn-fluid,1\n",
        canonical_completion=b'[{"flow_id":1,"jct_ps":80}]\n',
        step_results=b'[[0,0,80,"rnic-nn-fluid"]]\n',
        replay_summary=b'{"tpot_ps":80,"ttft_ps":80}\n',
    )
    identical = compare_bypass_artifacts(reference, reference)
    if not identical.equivalent:
        raise AssertionError("equal bypass bundle did not compare equal")
    expected_artifacts = {
        "completion_csv",
        "canonical_completion",
        "step_results",
        "replay_summary",
    }
    if set(artifact_names) != expected_artifacts or len(artifact_names) != len(
        expected_artifacts
    ):
        raise AssertionError("bypass artifact inventory drifted")
    mutation_rejections = 0
    for name in artifact_names:
        values = {
            field: getattr(reference, field)
            for field in reference.__dataclass_fields__
        }
        values[name] += b"!"
        comparison = compare_bypass_artifacts(reference, BypassArtifacts(**values))
        if comparison.changed_artifacts != (name,) or comparison.changed_inputs:
            raise AssertionError(f"bypass checker misclassified mutation {name}")
        mutation_rejections += 1
    input_mutation_rejections = 0
    input_names = (
        "goal_text",
        "goal_binary",
        "topology",
        "profile",
        "seed",
        "baseline_parameters",
    )
    for name in input_names:
        values = {
            field: getattr(reference, field)
            for field in reference.__dataclass_fields__
        }
        original = values[name]
        if type(original) is bytes:
            values[name] = original + b"!"
        elif type(original) is str:
            values[name] = original + "-changed"
        elif type(original) is int:
            values[name] = original + 1
        elif name == "baseline_parameters":
            changed_parameters = dict(original)
            first_parameter = min(changed_parameters)
            changed_parameters[first_parameter] += "-changed"
            values[name] = canonical_bypass_parameters(changed_parameters)
        else:
            raise AssertionError(f"unsupported bypass input guard: {name}")
        comparison = compare_bypass_artifacts(reference, BypassArtifacts(**values))
        if comparison.changed_inputs != (name,) or comparison.changed_artifacts:
            raise AssertionError(f"bypass checker misclassified input mutation {name}")
        input_mutation_rejections += 1

    return {
        "schema": "simllm-rnic-session-record-study-results-v1",
        "policy_invariance": {
            "hardware_cells": len(cell_hashes),
            "pair_checks_passed": policy_pair_checks,
            "pair_checks_total": policy_pair_checks,
            "unique_cell_hashes": len(set(cell_hashes.values())),
        },
        "hardware_hashes": [
            {
                "doorbell_service_ps": doorbell,
                "hardware_config_sha256": cell_hashes[(sq_depth, doorbell)],
                "sq_depth": sq_depth,
            }
            for sq_depth in sq_depths
            for doorbell in doorbell_services_ps
        ],
        "policy_hash_rows": [
            {
                "doorbell_service_ps": doorbell,
                "hardware_config_sha256": hashes[(sq_depth, doorbell, policy)],
                "policy": policy,
                "sq_depth": sq_depth,
            }
            for sq_depth in sq_depths
            for doorbell in doorbell_services_ps
            for policy in policies
        ],
        "active_sensitivity": {
            "adjacent_axis_changes_passed": sum(adjacent_changes),
            "adjacent_axis_changes_total": len(adjacent_changes),
            "census_passed": len(sensitivity_rows),
            "census_total": len(sensitivity_rows),
            "groups": sorted(observed_sensitivity),
        },
        "sensitivity_rows": sensitivity_rows,
        "bypass_identity": {
            "equal_artifacts_passed": len(artifact_names),
            "equal_artifacts_total": len(artifact_names),
            "mutation_rejections_passed": mutation_rejections,
            "mutation_rejections_total": len(artifact_names),
            "input_mutation_rejections_passed": input_mutation_rejections,
            "input_mutation_rejections_total": len(input_names),
        },
        "authority_counters": {
            "structural": [
                structural_result.authority_counters.native_session_constructed,
                structural_result.authority_counters.legacy_ledger_constructed,
                structural_result.authority_counters.native_posts,
                structural_result.authority_counters.legacy_mutations,
            ],
            "bypass": [
                bypass_result.authority_counters.native_session_constructed,
                bypass_result.authority_counters.legacy_ledger_constructed,
                bypass_result.authority_counters.native_posts,
                bypass_result.authority_counters.legacy_mutations,
            ],
        },
        "fatal_guards": {
            "authority_negative_controls": negative,
            "inactive_hash_guards": guards,
            "projection_checks": projection_checks,
        },
        "reader_projection": {
            "bypass_mode": bypass_result.hardware_mode.value,
            "bypass_wqes": len(bypass_result.wqes),
            "completion_csv_sha256": hashlib.sha256(
                completion_csv.encode("utf-8")
            ).hexdigest(),
            "dma_config_sha256": dma_config.hardware_config_sha256,
            "structural_mode": structural_result.hardware_mode.value,
            "structural_wqes": len(structural_result.wqes),
        },
        "projection_wqes": len(structural_result.wqes),
    }
