"""Offline kernel-cycle lookup records, fixture analysis and projections.

The lookup is evidence. It is content-addressed with the existing calibration
canonicalizer, then compiled into the established scalar profile table and
device-model service-entry types. It never becomes a serving-side authority.
"""

from __future__ import annotations

import copy
import csv
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from fractions import Fraction
from pathlib import Path
from typing import Any, NoReturn

from simllm.calibration.canonical import (
    canonical_bytes,
    canonical_sha256,
    sha256_bytes,
    validate_sha256,
)
from simllm.calibration.record_types import RecordObject
from simllm.compute.device_model import (
    DeviceResourceAxis,
    DeviceResourceRegistry,
    DeviceResourceVector,
    DeviceServiceEntry,
    DeviceServiceEntryRecord,
    ExactRate,
    ResourceAxisClass,
    ResourceServiceScope,
    ServiceEpochDefinition,
    ShapeAxis,
    ShapeSchema,
    ShapeVector,
    validate_device_service_entries,
)
from simllm.compute.provider import (
    ComputeProvider,
    GpuSpec,
    KernelSpec,
    ProfileLookupBinding,
    ProfileTableProvenance,
    ProfileTableProvider,
    RooflineProvider,
)

KERNEL_CYCLE_LUT_SCHEMA = "simllm-kernel-cycle-lut-v1"
KERNEL_CYCLE_INPUT_SCHEMA = "simllm-kernel-cycle-input-v1"
KERNEL_CYCLE_PRICING_PROVENANCE_SCHEMA = (
    "simllm-kernel-cycle-pricing-provenance-v1"
)
PS_PER_SECOND = 1_000_000_000_000

_IMPLEMENTATION_CLASSES = {
    "triton-jit",
    "wheel-precompiled-cuda",
    "closed-library",
    "unknown",
}
_POOLS = {"decode", "prefill"}
_LAUNCH_MODES = {"cuda-graph", "eager"}
_ACCEPTANCE_STATUSES = {"candidate", "validated"}
_DISTRIBUTION_VERDICTS = {
    "tight-single-peak",
    "conditioned-multipeak",
    "unstable",
    "insufficient-replays",
}
_ROUTING_AVAILABILITY = {"captured", "not-captured"}
_SERVICE_EVIDENCE_CLASSES = {"MEASURED", "DECLARED"}
_COMPONENT_EVIDENCE_CLASSES = {"DISCLOSED"}
_CELL_SPLITS = {"calibration", "held-out"}
_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/+-]*\Z")


def _fail(path: str, message: str) -> NoReturn:
    raise ValueError(f"{path}: {message}")


def _object(value: object, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(path, "expected an object")
    if any(not isinstance(key, str) for key in value):
        _fail(path, "object keys must be strings")
    return value


def _array(value: object, path: str) -> Sequence[Any]:
    if not isinstance(value, list | tuple):
        _fail(path, "expected an array")
    return value


def _fields(value: Mapping[str, Any], path: str, expected: set[str]) -> None:
    missing = sorted(expected - value.keys())
    if missing:
        _fail(path, f"missing fields {missing}")
    unknown = sorted(value.keys() - expected)
    if unknown:
        _fail(path, f"unknown fields {unknown}")


def _string(value: object, path: str) -> str:
    if not isinstance(value, str):
        _fail(path, "expected a string")
    if not value or value.strip() != value:
        _fail(path, "expected a nonblank string without edge whitespace")
    return value


def _optional_string(value: object, path: str) -> str | None:
    return None if value is None else _string(value, path)


def _identifier(value: object, path: str) -> str:
    result = _string(value, path)
    if _SAFE_ID.fullmatch(result) is None:
        _fail(path, "contains unsupported identifier characters")
    return result


def _enum(value: object, path: str, choices: set[str]) -> str:
    result = _string(value, path)
    if result not in choices:
        _fail(path, f"expected one of {sorted(choices)}")
    return result


def _integer(
    value: object,
    path: str,
    *,
    minimum: int | None = None,
    positive: bool = False,
) -> int:
    if type(value) is not int:
        _fail(path, "expected an integer")
    if positive and value <= 0:
        _fail(path, "must be positive")
    if minimum is not None and value < minimum:
        _fail(path, f"must be at least {minimum}")
    return value


def _optional_integer(
    value: object,
    path: str,
    *,
    minimum: int = 0,
) -> int | None:
    return None if value is None else _integer(value, path, minimum=minimum)


def _sha256(value: object, path: str) -> str:
    try:
        return validate_sha256(value, path)
    except Exception as error:
        raise ValueError(str(error)) from error


def _optional_sha256(value: object, path: str) -> str | None:
    return None if value is None else _sha256(value, path)


def _identity(value: object, path: str, *, framework: bool) -> None:
    payload = _object(value, path)
    fields = (
        {"name", "version", "revision", "package_sha256"}
        if framework
        else {"family", "name", "revision", "config_sha256", "weights_sha256"}
    )
    _fields(payload, path, fields)
    _string(payload["name"], f"{path}.name")
    if framework:
        _string(payload["version"], f"{path}.version")
        _optional_string(payload["revision"], f"{path}.revision")
        _sha256(payload["package_sha256"], f"{path}.package_sha256")
        return
    _enum(payload["family"], f"{path}.family", {"dense", "routed"})
    _optional_string(payload["revision"], f"{path}.revision")
    _optional_sha256(payload["config_sha256"], f"{path}.config_sha256")
    _optional_sha256(payload["weights_sha256"], f"{path}.weights_sha256")


def _parallelism(value: object, path: str) -> None:
    payload = _object(value, path)
    _fields(
        payload,
        path,
        {"tensor_parallel", "pipeline_parallel", "data_parallel", "expert_parallel"},
    )
    for name in payload:
        _integer(payload[name], f"{path}.{name}", positive=True)


def _routing(value: object, path: str, *, acceptance_status: str) -> None:
    payload = _object(value, path)
    _fields(payload, path, {"availability", "expert_loads", "evidence_sha256"})
    availability = _enum(
        payload["availability"],
        f"{path}.availability",
        _ROUTING_AVAILABILITY,
    )
    _sha256(payload["evidence_sha256"], f"{path}.evidence_sha256")
    loads = payload["expert_loads"]
    if loads is None:
        if availability != "not-captured":
            _fail(f"{path}.expert_loads", "null requires not-captured availability")
        if acceptance_status == "validated":
            _fail(f"{path}.expert_loads", "validated routed entries require captured loads")
        return
    if availability != "captured":
        _fail(f"{path}.availability", "expert loads require captured availability")
    rows = _array(loads, f"{path}.expert_loads")
    if not rows:
        _fail(f"{path}.expert_loads", "captured loads must not be empty")
    expert_ids: list[int] = []
    for index, row in enumerate(rows):
        row_path = f"{path}.expert_loads[{index}]"
        item = _object(row, row_path)
        _fields(item, row_path, {"expert_id", "tokens"})
        expert_ids.append(_integer(item["expert_id"], f"{row_path}.expert_id", minimum=0))
        _integer(item["tokens"], f"{row_path}.tokens", minimum=0)
    if expert_ids != sorted(expert_ids) or len(expert_ids) != len(set(expert_ids)):
        _fail(f"{path}.expert_loads", "expert IDs must be sorted and unique")


def _lookup_key(value: object, path: str, *, acceptance_status: str) -> None:
    payload = _object(value, path)
    base = {
        "framework_identity",
        "model_identity",
        "pool",
        "launch_mode",
        "parallelism",
        "shape",
        "input_dependency",
    }
    model = _object(payload.get("model_identity"), f"{path}.model_identity")
    family = model.get("family")
    expected = base | ({"routing"} if family == "routed" else set())
    _fields(payload, path, expected)
    _identity(payload["framework_identity"], f"{path}.framework_identity", framework=True)
    _identity(payload["model_identity"], f"{path}.model_identity", framework=False)
    pool = _enum(payload["pool"], f"{path}.pool", _POOLS)
    _enum(payload["launch_mode"], f"{path}.launch_mode", _LAUNCH_MODES)
    _parallelism(payload["parallelism"], f"{path}.parallelism")
    shape = _object(payload["shape"], f"{path}.shape")
    if pool == "decode":
        _fields(shape, f"{path}.shape", {"batch_size", "per_request_kv_lengths"})
        batch = _integer(shape["batch_size"], f"{path}.shape.batch_size", positive=True)
        lengths = _array(
            shape["per_request_kv_lengths"],
            f"{path}.shape.per_request_kv_lengths",
        )
        if len(lengths) != batch:
            _fail(
                f"{path}.shape.per_request_kv_lengths",
                "length must equal batch_size",
            )
        for index, length in enumerate(lengths):
            _integer(
                length,
                f"{path}.shape.per_request_kv_lengths[{index}]",
                minimum=0,
            )
    else:
        _fields(
            shape,
            f"{path}.shape",
            {"computed_new_tokens", "existing_context_tokens"},
        )
        _integer(
            shape["computed_new_tokens"],
            f"{path}.shape.computed_new_tokens",
            positive=True,
        )
        _integer(
            shape["existing_context_tokens"],
            f"{path}.shape.existing_context_tokens",
            minimum=0,
        )
    dependency = _enum(
        payload["input_dependency"],
        f"{path}.input_dependency",
        {"dense-content-independent", "moe-routing-dependent"},
    )
    if family == "dense":
        if "routing" in payload:
            _fail(f"{path}.routing", "dense families forbid routing")
        if dependency != "dense-content-independent":
            _fail(f"{path}.input_dependency", "dense families are content-independent")
    else:
        if dependency != "moe-routing-dependent":
            _fail(f"{path}.input_dependency", "routed families depend on routing")
        _routing(payload["routing"], f"{path}.routing", acceptance_status=acceptance_status)


def _clock_range(value: object, path: str) -> None:
    payload = _object(value, path)
    _fields(payload, path, {"min", "median", "max"})
    low = _integer(payload["min"], f"{path}.min", positive=True)
    median = _integer(payload["median"], f"{path}.median", positive=True)
    high = _integer(payload["max"], f"{path}.max", positive=True)
    if not low <= median <= high:
        _fail(path, "expected min <= median <= max")


def _distribution(value: object, path: str) -> None:
    payload = _object(value, path)
    _fields(
        payload,
        path,
        {
            "replay_count",
            "peak_count",
            "peak_centers_ps",
            "trimmed_coefficient_of_variation_ppm",
            "clock_correlation_ppm",
            "verdict",
        },
    )
    _integer(payload["replay_count"], f"{path}.replay_count", positive=True)
    peaks = _integer(payload["peak_count"], f"{path}.peak_count", positive=True)
    centers = _array(payload["peak_centers_ps"], f"{path}.peak_centers_ps")
    if len(centers) != peaks:
        _fail(f"{path}.peak_centers_ps", "length must equal peak_count")
    for index, center in enumerate(centers):
        _integer(center, f"{path}.peak_centers_ps[{index}]", positive=True)
    _integer(
        payload["trimmed_coefficient_of_variation_ppm"],
        f"{path}.trimmed_coefficient_of_variation_ppm",
        minimum=0,
    )
    correlation = payload["clock_correlation_ppm"]
    if correlation is not None:
        _integer(correlation, f"{path}.clock_correlation_ppm", minimum=-1_000_000)
        if correlation > 1_000_000:
            _fail(f"{path}.clock_correlation_ppm", "must be at most 1000000")
    _enum(payload["verdict"], f"{path}.verdict", _DISTRIBUTION_VERDICTS)


def _entry_evidence(value: object, path: str) -> None:
    payload = _object(value, path)
    _fields(
        payload,
        path,
        {
            "service_class",
            "component_class",
            "split",
            "source_sha256s",
            "derivation",
        },
    )
    service_class = _enum(
        payload["service_class"],
        f"{path}.service_class",
        _SERVICE_EVIDENCE_CLASSES,
    )
    _enum(
        payload["component_class"],
        f"{path}.component_class",
        _COMPONENT_EVIDENCE_CLASSES,
    )
    _enum(payload["split"], f"{path}.split", _CELL_SPLITS)
    sources = _array(payload["source_sha256s"], f"{path}.source_sha256s")
    if not sources:
        _fail(f"{path}.source_sha256s", "must not be empty")
    digests = [
        _sha256(source, f"{path}.source_sha256s[{index}]")
        for index, source in enumerate(sources)
    ]
    if digests != sorted(digests) or len(digests) != len(set(digests)):
        _fail(f"{path}.source_sha256s", "digests must be sorted and unique")
    derivation = _optional_string(payload["derivation"], f"{path}.derivation")
    if service_class == "MEASURED" and derivation is not None:
        _fail(f"{path}.derivation", "measured service must not have a derivation")
    if service_class == "DECLARED" and derivation is None:
        _fail(f"{path}.derivation", "declared service requires a derivation")


def _memory_components(value: object, path: str) -> None:
    payload = _object(value, path)
    _fields(
        payload,
        path,
        {
            "weight_bytes",
            "kv_bytes",
            "other_bytes",
            "achieved_bandwidth_bytes_per_second",
            "observed_memory_clock_hz",
            "service_ps",
        },
    )
    parts = [
        _optional_integer(payload[name], f"{path}.{name}")
        for name in ("weight_bytes", "kv_bytes", "other_bytes")
    ]
    bandwidth = _optional_integer(
        payload["achieved_bandwidth_bytes_per_second"],
        f"{path}.achieved_bandwidth_bytes_per_second",
    )
    _integer(
        payload["observed_memory_clock_hz"],
        f"{path}.observed_memory_clock_hz",
        positive=True,
    )
    service_ps = _integer(payload["service_ps"], f"{path}.service_ps", minimum=0)
    if all(part is None for part in parts):
        if bandwidth is not None or service_ps != 0:
            _fail(path, "unknown bytes require null bandwidth and zero service")
        return
    if any(part is None for part in parts):
        _fail(path, "known memory requires all three byte fields")
    if bandwidth is None or bandwidth <= 0:
        _fail(path, "known memory requires positive achieved bandwidth")
    assert all(part is not None for part in parts)
    expected = _ceil_div(sum(parts) * PS_PER_SECOND, bandwidth)
    if service_ps != expected:
        _fail(f"{path}.service_ps", f"expected bytes-over-bandwidth value {expected}")


def _kernel(value: object, path: str) -> int:
    payload = _object(value, path)
    _fields(
        payload,
        path,
        {
            "kernel_id",
            "name",
            "implementation_class",
            "launch_count",
            "measured_elapsed_ps",
            "measured_elapsed_sm_cycles",
            "nsys_median_elapsed_ps",
            "cross_instrument_ratio_ppm",
            "components",
            "code_object",
        },
    )
    _identifier(payload["kernel_id"], f"{path}.kernel_id")
    _string(payload["name"], f"{path}.name")
    _enum(
        payload["implementation_class"],
        f"{path}.implementation_class",
        _IMPLEMENTATION_CLASSES,
    )
    launch_count = _integer(payload["launch_count"], f"{path}.launch_count", positive=True)
    measured_ps = _integer(
        payload["measured_elapsed_ps"],
        f"{path}.measured_elapsed_ps",
        positive=True,
    )
    _integer(
        payload["measured_elapsed_sm_cycles"],
        f"{path}.measured_elapsed_sm_cycles",
        positive=True,
    )
    nsys_ps = _integer(
        payload["nsys_median_elapsed_ps"],
        f"{path}.nsys_median_elapsed_ps",
        positive=True,
    )
    ratio_ppm = _integer(
        payload["cross_instrument_ratio_ppm"],
        f"{path}.cross_instrument_ratio_ppm",
        positive=True,
    )
    expected_ratio = _round_half_up(Decimal(measured_ps) * 1_000_000 / Decimal(nsys_ps))
    if ratio_ppm != expected_ratio:
        _fail(f"{path}.cross_instrument_ratio_ppm", f"expected {expected_ratio}")
    components = _object(payload["components"], f"{path}.components")
    _fields(
        components,
        f"{path}.components",
        {"compute_sm_cycles", "memory", "fixed_overhead_ps", "method"},
    )
    _integer(
        components["compute_sm_cycles"],
        f"{path}.components.compute_sm_cycles",
        minimum=0,
    )
    memory = _object(components["memory"], f"{path}.components.memory")
    _memory_components(memory, f"{path}.components.memory")
    _integer(
        components["fixed_overhead_ps"],
        f"{path}.components.fixed_overhead_ps",
        minimum=0,
    )
    _string(components["method"], f"{path}.components.method")
    code = _object(payload["code_object"], f"{path}.code_object")
    _fields(
        code,
        f"{path}.code_object",
        {"ptx_sha256", "sass_sha256", "compile_configuration_sha256"},
    )
    _optional_sha256(code["ptx_sha256"], f"{path}.code_object.ptx_sha256")
    _optional_sha256(code["sass_sha256"], f"{path}.code_object.sass_sha256")
    _optional_sha256(
        code["compile_configuration_sha256"],
        f"{path}.code_object.compile_configuration_sha256",
    )
    return launch_count * measured_ps


def validate_kernel_cycle_lut(
    value: Mapping[str, Any] | str | bytes | bytearray | memoryview,
) -> RecordObject:
    """Validate one strict lookup record and return its content-addressed view."""

    record = (
        RecordObject.from_value(value)
        if isinstance(value, Mapping)
        else RecordObject.from_bytes(value, expected_schema=KERNEL_CYCLE_LUT_SCHEMA)
    )
    payload = _object(record.value, "record")
    _fields(
        payload,
        "record",
        {
            "schema",
            "acceptance_status",
            "campaign_id",
            "created",
            "device",
            "capture_protocol",
            "sources",
            "entries",
        },
    )
    if payload["schema"] != KERNEL_CYCLE_LUT_SCHEMA:
        _fail("record.schema", f"expected {KERNEL_CYCLE_LUT_SCHEMA!r}")
    status = _enum(
        payload["acceptance_status"],
        "record.acceptance_status",
        _ACCEPTANCE_STATUSES,
    )
    _identifier(payload["campaign_id"], "record.campaign_id")
    _string(payload["created"], "record.created")
    device = _object(payload["device"], "record.device")
    _fields(
        device,
        "record.device",
        {
            "device_kind_id",
            "gpu_name",
            "gpu_uuid",
            "architecture",
            "memory_capacity_bytes_per_second",
        },
    )
    _identifier(device["device_kind_id"], "record.device.device_kind_id")
    _string(device["gpu_name"], "record.device.gpu_name")
    _string(device["gpu_uuid"], "record.device.gpu_uuid")
    _identifier(device["architecture"], "record.device.architecture")
    _integer(
        device["memory_capacity_bytes_per_second"],
        "record.device.memory_capacity_bytes_per_second",
        positive=True,
    )
    protocol = _object(payload["capture_protocol"], "record.capture_protocol")
    _fields(
        protocol,
        "record.capture_protocol",
        {
            "graph_minimum_replays",
            "eager_minimum_replays",
            "program_counter_sampling",
            "compile_graph_inference",
            "code_object_double_harvest",
        },
    )
    _integer(
        protocol["graph_minimum_replays"],
        "record.capture_protocol.graph_minimum_replays",
        positive=True,
    )
    _integer(
        protocol["eager_minimum_replays"],
        "record.capture_protocol.eager_minimum_replays",
        positive=True,
    )
    for name in (
        "program_counter_sampling",
        "compile_graph_inference",
        "code_object_double_harvest",
    ):
        _enum(
            protocol[name],
            f"record.capture_protocol.{name}",
            {"complete", "granted", "denied", "unavailable", "not-captured"},
        )
    sources = _array(payload["sources"], "record.sources")
    if not sources:
        _fail("record.sources", "must not be empty")
    source_names: list[str] = []
    for index, source in enumerate(sources):
        path = f"record.sources[{index}]"
        item = _object(source, path)
        _fields(
            item,
            path,
            {
                "name",
                "fixture_sha256",
                "fixture_bytes",
                "retained_source_name",
                "retained_source_sha256",
            },
        )
        source_names.append(_string(item["name"], f"{path}.name"))
        _sha256(item["fixture_sha256"], f"{path}.fixture_sha256")
        _integer(item["fixture_bytes"], f"{path}.fixture_bytes", positive=True)
        _string(item["retained_source_name"], f"{path}.retained_source_name")
        _sha256(item["retained_source_sha256"], f"{path}.retained_source_sha256")
    if source_names != sorted(source_names) or len(source_names) != len(set(source_names)):
        _fail("record.sources", "names must be sorted and unique")
    entries = _array(payload["entries"], "record.entries")
    if not entries:
        _fail("record.entries", "must not be empty")
    implementation_ids: list[str] = []
    for index, entry in enumerate(entries):
        path = f"record.entries[{index}]"
        item = _object(entry, path)
        entry_fields = {
            "key",
            "implementation_id",
            "coverage",
            "measured_service_ps",
            "observed_clocks",
            "distribution",
            "kernels",
        }
        if "evidence" in item:
            entry_fields.add("evidence")
        _fields(
            item,
            path,
            entry_fields,
        )
        if "evidence" in item:
            _entry_evidence(item["evidence"], f"{path}.evidence")
        _lookup_key(item["key"], f"{path}.key", acceptance_status=status)
        implementation_ids.append(
            _identifier(item["implementation_id"], f"{path}.implementation_id")
        )
        _enum(
            item["coverage"],
            f"{path}.coverage",
            {"complete-kernel-stream", "partial-kernel-subset"},
        )
        measured_service_ps = _integer(
            item["measured_service_ps"],
            f"{path}.measured_service_ps",
            positive=True,
        )
        clocks = _object(item["observed_clocks"], f"{path}.observed_clocks")
        _fields(clocks, f"{path}.observed_clocks", {"sm_hz", "memory_hz"})
        _clock_range(clocks["sm_hz"], f"{path}.observed_clocks.sm_hz")
        _clock_range(clocks["memory_hz"], f"{path}.observed_clocks.memory_hz")
        sm_hz = _object(clocks["sm_hz"], f"{path}.observed_clocks.sm_hz")["median"]
        _distribution(item["distribution"], f"{path}.distribution")
        kernels = _array(item["kernels"], f"{path}.kernels")
        if not kernels:
            _fail(f"{path}.kernels", "must not be empty")
        kernel_ids: list[str] = []
        reconstructed = 0
        for kernel_index, kernel_value in enumerate(kernels):
            kernel_path = f"{path}.kernels[{kernel_index}]"
            launch_total = _kernel(kernel_value, kernel_path)
            kernel = _object(kernel_value, kernel_path)
            kernel_ids.append(_string(kernel["kernel_id"], f"{kernel_path}.kernel_id"))
            components = _object(kernel["components"], f"{kernel_path}.components")
            memory = _object(components["memory"], f"{kernel_path}.components.memory")
            compute_ps = _ceil_div(
                int(components["compute_sm_cycles"]) * PS_PER_SECOND,
                int(sm_hz),
            )
            service_ps = max(compute_ps, int(memory["service_ps"])) + int(
                components["fixed_overhead_ps"]
            )
            measured_ps = int(kernel["measured_elapsed_ps"])
            if abs(service_ps - measured_ps) > 1:
                _fail(
                    f"{kernel_path}.components",
                    f"reconstructs {service_ps} ps, measured {measured_ps} ps",
                )
            reconstructed += service_ps * int(kernel["launch_count"])
            if launch_total != measured_ps * int(kernel["launch_count"]):
                raise AssertionError("kernel launch total invariant failed")
        if kernel_ids != sorted(kernel_ids) or len(kernel_ids) != len(set(kernel_ids)):
            _fail(f"{path}.kernels", "kernel IDs must be sorted and unique")
        if abs(reconstructed - measured_service_ps) > len(kernels):
            _fail(
                f"{path}.measured_service_ps",
                f"expected reconstructed kernel total {reconstructed}",
            )
    if implementation_ids != sorted(implementation_ids):
        _fail("record.entries", "implementation IDs must be sorted")
    if len(implementation_ids) != len(set(implementation_ids)):
        _fail("record.entries", "implementation IDs must be unique")
    return record


def _ceil_div(numerator: int, denominator: int) -> int:
    if denominator <= 0:
        raise ValueError("denominator must be positive")
    return -(-numerator // denominator)


def _round_half_up(value: Decimal) -> int:
    return int(value.quantize(Decimal(1), rounding=ROUND_HALF_UP))


def _median_int(values: Sequence[int]) -> int:
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return _round_half_up(Decimal(ordered[middle - 1] + ordered[middle]) / 2)


def _trimmed_cv_ppm(values: Sequence[int]) -> int:
    ordered = sorted(values)
    trim = len(ordered) // 10
    retained = ordered[trim : len(ordered) - trim] if trim else ordered
    if not retained:
        raise ValueError("distribution must retain at least one sample")
    mean = Decimal(sum(retained)) / Decimal(len(retained))
    if mean == 0:
        return 0
    variance = sum((Decimal(value) - mean) ** 2 for value in retained) / Decimal(len(retained))
    return _round_half_up(variance.sqrt() * Decimal(1_000_000) / mean)


def _distribution_from_samples(samples_ps: Sequence[int], launch_mode: str) -> dict[str, Any]:
    if not samples_ps:
        raise ValueError("ordered-kernel excerpt has no distribution samples")
    minimum = 256 if launch_mode == "cuda-graph" else 64
    return {
        "replay_count": len(samples_ps),
        "peak_count": 1,
        "peak_centers_ps": [_median_int(samples_ps)],
        "trimmed_coefficient_of_variation_ppm": _trimmed_cv_ppm(samples_ps),
        "clock_correlation_ppm": None,
        "verdict": ("insufficient-replays" if len(samples_ps) < minimum else "tight-single-peak"),
    }


def _load_input_manifest(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    payload = dict(_object(value, "input_manifest"))
    if payload.get("schema") != KERNEL_CYCLE_INPUT_SCHEMA:
        _fail("input_manifest.schema", f"expected {KERNEL_CYCLE_INPUT_SCHEMA!r}")
    return payload


def _verified_sources(root: Path, manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for index, raw in enumerate(_array(manifest.get("sources"), "input_manifest.sources")):
        path = f"input_manifest.sources[{index}]"
        source = _object(raw, path)
        _fields(
            source,
            path,
            {
                "name",
                "path",
                "sha256",
                "bytes",
                "retained_source_name",
                "retained_source_sha256",
            },
        )
        relative = Path(_string(source["path"], f"{path}.path"))
        if relative.is_absolute() or ".." in relative.parts:
            _fail(f"{path}.path", "must stay below the input directory")
        source_path = root / relative
        raw_bytes = source_path.read_bytes()
        expected_digest = _sha256(source["sha256"], f"{path}.sha256")
        actual_digest = sha256_bytes(raw_bytes)
        if actual_digest != expected_digest:
            _fail(f"{path}.sha256", f"expected {expected_digest}, found {actual_digest}")
        expected_bytes = _integer(source["bytes"], f"{path}.bytes", positive=True)
        if len(raw_bytes) != expected_bytes:
            _fail(f"{path}.bytes", f"expected {expected_bytes}, found {len(raw_bytes)}")
        result.append(
            {
                "name": _string(source["name"], f"{path}.name"),
                "fixture_sha256": actual_digest,
                "fixture_bytes": len(raw_bytes),
                "retained_source_name": _string(
                    source["retained_source_name"],
                    f"{path}.retained_source_name",
                ),
                "retained_source_sha256": _sha256(
                    source["retained_source_sha256"],
                    f"{path}.retained_source_sha256",
                ),
                "path": source_path,
            }
        )
    if not result:
        _fail("input_manifest.sources", "must not be empty")
    if [source["name"] for source in result] != sorted(source["name"] for source in result):
        _fail("input_manifest.sources", "names must be sorted")
    return result


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream, skipinitialspace=True))
    if not rows:
        raise ValueError(f"CSV has no data rows: {path.name}")
    return rows


def _source_path(sources: Sequence[Mapping[str, Any]], name: str) -> Path:
    for source in sources:
        if source["name"] == name:
            return Path(source["path"])
    raise ValueError(f"input manifest has no source named {name!r}")


def _parse_decimal_int(value: str, path: str, *, scale: int = 1) -> int:
    try:
        decimal = Decimal(value)
    except Exception as error:
        raise ValueError(f"{path}: invalid decimal {value!r}") from error
    result = decimal * scale
    integral = result.to_integral_value()
    if result != integral:
        raise ValueError(f"{path}: {value!r} is not integral after scaling by {scale}")
    return int(integral)


def _clock_hz(row: Mapping[str, str], names: Sequence[str], path: str) -> int:
    for name in names:
        if name not in row:
            continue
        value = row[name].strip()
        if value.endswith("MHz"):
            value = value[:-3].strip()
        return _parse_decimal_int(value, path, scale=1_000_000)
    raise ValueError(f"{path}: no supported column in {list(names)!r}")


def _utilization_percent(row: Mapping[str, str]) -> Decimal | None:
    for name in ("utilization_gpu_percent", "utilization.gpu [%]"):
        if name not in row:
            continue
        value = row[name].strip()
        if value.endswith("%"):
            value = value[:-1].strip()
        try:
            return Decimal(value)
        except Exception as error:
            raise ValueError(f"clocks.{name}: invalid percentage {row[name]!r}") from error
    return None


def _median_integer(values: Sequence[int], path: str) -> int:
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return _round_half_up(Decimal(ordered[middle - 1] + ordered[middle]) / 2)


def _observed_clock_ranges(rows: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    active = [row for row in rows if (_utilization_percent(row) or Decimal(0)) > 0]
    selected = active or list(rows)
    sm_values = [
        _clock_hz(
            row,
            ("sm_clock_mhz", "clocks.current.sm [MHz]"),
            "clocks.sm_clock_mhz",
        )
        for row in selected
    ]
    memory_values = [
        _clock_hz(
            row,
            ("memory_clock_mhz", "clocks.current.memory [MHz]"),
            "clocks.memory_clock_mhz",
        )
        for row in selected
    ]
    return {
        "sm_hz": {
            "min": min(sm_values),
            "median": _median_integer(sm_values, "clocks.sm_hz"),
            "max": max(sm_values),
        },
        "memory_hz": {
            "min": min(memory_values),
            "median": _median_integer(memory_values, "clocks.memory_hz"),
            "max": max(memory_values),
        },
    }


def analyze_kernel_cycle_capture(input_dir: str | Path) -> RecordObject:
    """Analyze one verified raw-export directory into a canonical LUT record."""

    root = Path(input_dir)
    manifest = _load_input_manifest(root / "manifest.json")
    sources = _verified_sources(root, manifest)
    ordered = _csv_rows(_source_path(sources, "nsys-ordered-kernels"))
    summaries = _csv_rows(_source_path(sources, "nsys-kernel-summary"))
    ncu_rows = _csv_rows(_source_path(sources, "ncu-kernel-metrics"))
    clocks = _csv_rows(_source_path(sources, "observed-clocks"))

    selection = _object(manifest["selection"], "input_manifest.selection")
    mappings = _array(selection["kernel_mappings"], "input_manifest.selection.kernel_mappings")
    summary_by_id: dict[str, dict[str, str]] = {}
    ncu_by_id = {row["kernel_id"]: row for row in ncu_rows}
    for mapping_value in mappings:
        mapping = _object(mapping_value, "input_manifest.selection.kernel_mappings[]")
        kernel_id = _identifier(mapping["kernel_id"], "kernel_mapping.kernel_id")
        needle = _string(mapping["nsys_name_contains"], "kernel_mapping.nsys_name_contains")
        matches = [row for row in summaries if needle in row["name"]]
        if len(matches) != 1:
            raise ValueError(f"kernel mapping {kernel_id!r} matched {len(matches)} summary rows")
        summary_by_id[kernel_id] = matches[0]
    expected_ids = sorted(summary_by_id)
    if sorted(ncu_by_id) != expected_ids:
        raise ValueError("Nsight Systems and Nsight Compute kernel ID sets differ")

    observed_clocks = _observed_clock_ranges(clocks)
    sm_hz = observed_clocks["sm_hz"]["median"]
    memory_hz = observed_clocks["memory_hz"]["median"]
    repeat_kernel = _identifier(
        selection["repeat_kernel_id"],
        "input_manifest.selection.repeat_kernel_id",
    )
    repeat_needle = next(
        _string(mapping["nsys_name_contains"], "kernel_mapping.nsys_name_contains")
        for mapping in mappings
        if mapping["kernel_id"] == repeat_kernel
    )
    samples_ps = [
        _parse_decimal_int(row["duration_ns"], "ordered.duration_ns", scale=1_000)
        for row in ordered
        if repeat_needle in row["name"]
    ]
    key = dict(_object(manifest["key"], "input_manifest.key"))
    launch_mode = _string(key["launch_mode"], "input_manifest.key.launch_mode")
    kernels: list[dict[str, Any]] = []
    mapping_by_id = {str(mapping["kernel_id"]): mapping for mapping in mappings}
    for kernel_id in expected_ids:
        summary = summary_by_id[kernel_id]
        ncu = ncu_by_id[kernel_id]
        mapping = _object(mapping_by_id[kernel_id], f"kernel_mapping.{kernel_id}")
        measured_ps = _parse_decimal_int(
            ncu["elapsed_us"],
            f"ncu.{kernel_id}.elapsed_us",
            scale=1_000_000,
        )
        cycles = _parse_decimal_int(
            ncu["elapsed_sm_cycles"],
            f"ncu.{kernel_id}.elapsed_sm_cycles",
        )
        compute_ps = _ceil_div(cycles * PS_PER_SECOND, sm_hz)
        if compute_ps > measured_ps:
            raise ValueError(
                f"ncu.{kernel_id}: elapsed cycles at observed clock exceed elapsed time"
            )
        fixed_ps = measured_ps - compute_ps
        nsys_median_ps = _parse_decimal_int(
            summary["median_duration_ns"],
            f"nsys.{kernel_id}.median_duration_ns",
            scale=1_000,
        )
        launch_count = _parse_decimal_int(
            summary["count_per_step"],
            f"nsys.{kernel_id}.count_per_step",
        )
        code = _object(mapping["code_object"], f"kernel_mapping.{kernel_id}.code_object")
        kernels.append(
            {
                "kernel_id": kernel_id,
                "name": ncu["kernel_name"],
                "implementation_class": mapping["implementation_class"],
                "launch_count": launch_count,
                "measured_elapsed_ps": measured_ps,
                "measured_elapsed_sm_cycles": cycles,
                "nsys_median_elapsed_ps": nsys_median_ps,
                "cross_instrument_ratio_ppm": _round_half_up(
                    Decimal(measured_ps) * 1_000_000 / Decimal(nsys_median_ps)
                ),
                "components": {
                    "compute_sm_cycles": cycles,
                    "memory": {
                        "weight_bytes": None,
                        "kv_bytes": None,
                        "other_bytes": None,
                        "achieved_bandwidth_bytes_per_second": None,
                        "observed_memory_clock_hz": memory_hz,
                        "service_ps": 0,
                    },
                    "fixed_overhead_ps": fixed_ps,
                    "method": "elapsed-cycles-plus-residual-with-memory-bytes-unavailable",
                },
                "code_object": {
                    "ptx_sha256": code["ptx_sha256"],
                    "sass_sha256": code["sass_sha256"],
                    "compile_configuration_sha256": code["compile_configuration_sha256"],
                },
            }
        )
    kernels.sort(key=lambda kernel: kernel["kernel_id"])
    measured_service_ps = sum(
        kernel["launch_count"] * kernel["measured_elapsed_ps"] for kernel in kernels
    )
    record_value = {
        "schema": KERNEL_CYCLE_LUT_SCHEMA,
        "acceptance_status": manifest["acceptance_status"],
        "campaign_id": manifest["campaign_id"],
        "created": manifest["created"],
        "device": manifest["device"],
        "capture_protocol": manifest["capture_protocol"],
        "sources": [
            {key: value for key, value in source.items() if key != "path"} for source in sources
        ],
        "entries": [
            {
                "key": key,
                "implementation_id": manifest["implementation_id"],
                "coverage": manifest["coverage"],
                "measured_service_ps": measured_service_ps,
                "observed_clocks": observed_clocks,
                "distribution": _distribution_from_samples(samples_ps, launch_mode),
                "kernels": kernels,
            }
        ],
    }
    return validate_kernel_cycle_lut(record_value)


def _profile_config(key: Mapping[str, Any]) -> tuple[tuple[str, int], ...]:
    parallelism = _object(key["parallelism"], "entry.key.parallelism")
    config = [
        ("tensor_parallel", int(parallelism["tensor_parallel"])),
        ("pipeline_parallel", int(parallelism["pipeline_parallel"])),
        ("data_parallel", int(parallelism["data_parallel"])),
        ("expert_parallel", int(parallelism["expert_parallel"])),
    ]
    shape = _object(key["shape"], "entry.key.shape")
    if key["pool"] == "decode":
        config.append(("batch_size", int(shape["batch_size"])))
        config.extend(
            (f"kv_length_{index:04d}", int(length))
            for index, length in enumerate(shape["per_request_kv_lengths"])
        )
    else:
        config.extend(
            (
                ("computed_new_tokens", int(shape["computed_new_tokens"])),
                ("existing_context_tokens", int(shape["existing_context_tokens"])),
            )
        )
    routing = key.get("routing")
    if isinstance(routing, Mapping) and routing["expert_loads"] is not None:
        config.extend(
            (f"expert_{int(row['expert_id']):04d}", int(row["tokens"]))
            for row in routing["expert_loads"]
        )
    return tuple(config)


def compile_profile_table(
    value: Mapping[str, Any] | str | bytes | bytearray | memoryview,
    *,
    lookup_binding: ProfileLookupBinding | None = None,
    comparator: ComputeProvider | None = None,
) -> ProfileTableProvider:
    """Compile a lookup record into the established scalar profile provider."""

    record = validate_kernel_cycle_lut(value)
    payload = record.value
    device = _object(payload["device"], "record.device")
    table = {}
    for entry_value in payload["entries"]:
        entry = _object(entry_value, "record.entries[]")
        key = _object(entry["key"], "record.entries[].key")
        distribution = _object(entry["distribution"], "record.entries[].distribution")
        table[
            (
                str(entry["implementation_id"]),
                _profile_config(key),
                str(device["gpu_name"]),
            )
        ] = (
            int(entry["measured_service_ps"]),
            int(distribution["trimmed_coefficient_of_variation_ppm"]) / 1_000_000,
        )
    return ProfileTableProvider(
        table,
        provenance=ProfileTableProvenance(
            source="capture-candidate",
            version=KERNEL_CYCLE_LUT_SCHEMA,
            gpu=str(device["gpu_name"]),
            created=str(payload["created"]),
            references=(f"record-sha256:{record.record_id}",),
        ),
        lookup_binding=lookup_binding,
        comparator=comparator,
    )


def _selection_key_template(
    record: RecordObject,
    *,
    pool: str,
    selection_entry_index: int,
) -> dict[str, Any]:
    if pool not in _POOLS:
        raise ValueError(f"pool must be one of {sorted(_POOLS)}")
    entries = record.value["entries"]
    if (
        isinstance(selection_entry_index, bool)
        or type(selection_entry_index) is not int
    ):
        raise TypeError("selection_entry_index must be an integer")
    if not 0 <= selection_entry_index < len(entries):
        raise IndexError("selection_entry_index is outside the lookup record")
    key = json.loads(record.canonical)["entries"][selection_entry_index]["key"]
    key["pool"] = pool
    key.pop("shape")
    return key


class KernelCycleLookupBinding:
    """Exact kernel-cycle key selection inside ``ProfileTableProvider``.

    The selected record owns framework, model, launch, parallelism and routing
    identity. Each live ``KernelSpec`` supplies only the pool-specific dynamic
    shape. A complete canonical key selects exactly one compiled profile-table
    row. The owning profile provider delegates a miss to its explicit
    comparator without changing that estimate.

    This provider accepts candidate records because candidate status is part
    of its required provenance. Accepting the record never promotes it to a
    calibration claim.
    """

    def __init__(
        self,
        record: RecordObject,
        *,
        pool: str,
        selection_entry_index: int = 0,
    ) -> None:
        self.record = record
        self.pool = pool
        self._key_template = _selection_key_template(
            record,
            pool=pool,
            selection_entry_index=selection_entry_index,
        )
        self._record_gpu = GpuSpec(
            name=str(record.value["device"]["gpu_name"]),
            peak_flops=1.0,
            mem_bandwidth=1.0,
        )
        self._entries_by_key: dict[bytes, list[Mapping[str, Any]]] = {}
        for entry_value in record.value["entries"]:
            entry = _object(entry_value, "record.entries[]")
            key_bytes = canonical_bytes(entry["key"])
            self._entries_by_key.setdefault(key_bytes, []).append(entry)
        self._lookup_hits = 0
        self._lookup_misses = 0
        self._selected_key_sha256s: list[str] = []

    def _query_key(self, kernel: KernelSpec) -> dict[str, Any] | None:
        requests = kernel.request_shapes
        if not requests:
            return None
        query = copy.deepcopy(self._key_template)
        if self.pool == "decode":
            query["shape"] = {
                "batch_size": len(requests),
                "per_request_kv_lengths": [
                    request.prior_context_tokens for request in requests
                ],
            }
        else:
            query["shape"] = {
                "computed_new_tokens": sum(
                    request.num_new_tokens for request in requests
                ),
                "existing_context_tokens": sum(
                    request.prior_context_tokens for request in requests
                ),
            }
        return query

    def _select(
        self,
        kernel: KernelSpec,
    ) -> tuple[dict[str, Any] | None, Mapping[str, Any] | None]:
        query = self._query_key(kernel)
        if query is None:
            return None, None
        matches = self._entries_by_key.get(canonical_bytes(query), [])
        if len(matches) > 1:
            raise ValueError(
                "kernel-cycle lookup found duplicate rows for complete key "
                f"{canonical_sha256(query)}"
            )
        return query, matches[0] if matches else None

    def selection_for(self, kernel: KernelSpec) -> dict[str, Any]:
        """Describe one pure exact-key decision without pricing or counting."""

        query, entry = self._select(kernel)
        return {
            "query_key_sha256": (
                None if query is None else canonical_sha256(query)
            ),
            "selected": entry is not None,
            "selected_entry_key_sha256": (
                None if entry is None else canonical_sha256(entry["key"])
            ),
            "implementation_id": (
                None if entry is None else str(entry["implementation_id"])
            ),
        }

    def profile_query(
        self,
        kernel: KernelSpec,
    ) -> tuple[KernelSpec, GpuSpec] | None:
        """Return one compiled exact query, or ``None`` for comparator use."""

        _, entry = self._select(kernel)
        if entry is None:
            return None
        key_sha256 = canonical_sha256(entry["key"])
        if key_sha256 not in self._selected_key_sha256s:
            self._selected_key_sha256s.append(key_sha256)
        return (
            KernelSpec(
                name=str(entry["implementation_id"]),
                flops=0,
                bytes_moved=0,
                config=_profile_config(entry["key"]),
            ),
            self._record_gpu,
        )

    def record_lookup(self, selected: bool) -> None:
        """Record one estimate decision without conflating layer queries."""

        if selected:
            self._lookup_hits += 1
        else:
            self._lookup_misses += 1

    def pricing_provenance(self) -> dict[str, Any]:
        """Return the candidate-safe record identity and selection ledger."""

        payload = self.record.value
        entries = payload["entries"]
        coverages = sorted({str(entry["coverage"]) for entry in entries})
        selected = tuple(self._selected_key_sha256s)
        return {
            "schema": KERNEL_CYCLE_PRICING_PROVENANCE_SCHEMA,
            "record_sha256": self.record.record_id,
            "acceptance_status": str(payload["acceptance_status"]),
            "campaign_id": str(payload["campaign_id"]),
            "coverage": coverages[0] if len(coverages) == 1 else "mixed",
            "record_device_kind_id": str(payload["device"]["device_kind_id"]),
            "pool": self.pool,
            "selected_entry_key_sha256": selected[-1] if selected else None,
            "selected_entry_key_sha256s": list(selected),
            "lookup_hits": self._lookup_hits,
            "lookup_misses": self._lookup_misses,
            "calibration_claim": False,
        }


def compile_session_profile_provider(
    value: Mapping[str, Any] | str | bytes | bytearray | memoryview,
    *,
    expected_sha256: str,
    pool: str,
    comparator: ComputeProvider | None = None,
    selection_entry_index: int = 0,
) -> ProfileTableProvider:
    """Bind a content-addressed lookup record into its profile-table form."""

    record = validate_kernel_cycle_lut(value)
    expected = validate_sha256(expected_sha256, "expected_sha256")
    if record.record_id != expected:
        raise ValueError(
            "kernel-cycle record content address disagrees: "
            f"expected {expected}, found {record.record_id}"
        )
    fallback = (
        RooflineProvider(efficiency=0.7)
        if comparator is None
        else comparator
    )
    if not isinstance(fallback, ComputeProvider):
        raise TypeError("comparator must implement ComputeProvider")
    binding = KernelCycleLookupBinding(
        record,
        pool=pool,
        selection_entry_index=selection_entry_index,
    )
    return compile_profile_table(
        record.canonical,
        lookup_binding=binding,
        comparator=fallback,
    )


@dataclass(frozen=True, slots=True)
class DeviceServiceCompilation:
    """Candidate projection into existing device-model component types."""

    acceptance_status: str
    lookup_record_sha256: str
    resource_registry: DeviceResourceRegistry
    resource_registry_sha256: str
    shape_schemas: tuple[ShapeSchema, ...]
    service_entries: tuple[DeviceServiceEntryRecord, ...]


def _rate_per_ps(bytes_per_second: int) -> ExactRate:
    rate = Fraction(bytes_per_second, PS_PER_SECOND)
    return ExactRate(rate.numerator, rate.denominator)


def _shape_signature(key: Mapping[str, Any]) -> tuple[str, tuple[tuple[str, int], ...]]:
    shape = _object(key["shape"], "entry.key.shape")
    if key["pool"] == "decode":
        axes = [("batch_size", int(shape["batch_size"]))]
        axes.extend(
            (f"kv_length_{index:04d}", int(length))
            for index, length in enumerate(shape["per_request_kv_lengths"])
        )
        return f"kernel-cycle-decode-b{len(shape['per_request_kv_lengths'])}-v1", tuple(axes)
    return "kernel-cycle-prefill-v1", (
        ("computed_new_tokens", int(shape["computed_new_tokens"])),
        ("existing_context_tokens", int(shape["existing_context_tokens"])),
    )


def compile_device_service_entries(
    value: Mapping[str, Any] | str | bytes | bytearray | memoryview,
) -> DeviceServiceCompilation:
    """Compile max-plus-fixed components into ordered existing service epochs."""

    record = validate_kernel_cycle_lut(value)
    payload = record.value
    device = _object(payload["device"], "record.device")
    entries = [_object(entry, "record.entries[]") for entry in payload["entries"]]
    memory_known = any(
        _object(
            _object(kernel, "kernel")["components"],
            "kernel.components",
        )["memory"]["weight_bytes"]
        is not None
        for entry in entries
        for kernel in entry["kernels"]
    )
    if memory_known and any(
        _object(
            _object(kernel, "kernel")["components"],
            "kernel.components",
        )["memory"]["weight_bytes"]
        is None
        for entry in entries
        for kernel in entry["kernels"]
    ):
        raise ValueError("device service compilation rejects mixed known and unknown memory")
    axes: list[DeviceResourceAxis] = []
    if memory_known:
        rate = _rate_per_ps(int(device["memory_capacity_bytes_per_second"]))
        for axis_id in ("hbm-kv-bytes", "hbm-other-bytes", "hbm-weight-bytes"):
            axes.append(
                DeviceResourceAxis(
                    axis_id=axis_id,
                    axis_class=ResourceAxisClass.THROUGHPUT,
                    service_scope=ResourceServiceScope.DEVICE_INTERNAL,
                    base_unit="bytes",
                    clock_domain_id=None,
                    capacity_source_id="kernel-cycle-lut-device-memory-capacity",
                    rate=rate,
                    residency_capacity=None,
                    exclusive_capacity=None,
                )
            )
    axes.append(
        DeviceResourceAxis(
            axis_id="sm-cycles",
            axis_class=ResourceAxisClass.THROUGHPUT,
            service_scope=ResourceServiceScope.DEVICE_INTERNAL,
            base_unit="cycles",
            clock_domain_id="sm-clock",
            capacity_source_id="kernel-cycle-lut-observed-sm-clock",
            rate=ExactRate(1, 1),
            residency_capacity=None,
            exclusive_capacity=None,
        )
    )
    axes_tuple = tuple(sorted(axes, key=lambda axis: axis.axis_id))
    registry = DeviceResourceRegistry(
        device_kind_id=str(device["device_kind_id"]),
        active_axis_ids=tuple(axis.axis_id for axis in axes_tuple),
        axes=axes_tuple,
    )
    registry_sha256 = canonical_sha256(registry.to_obj())

    signatures = [_shape_signature(_object(entry["key"], "entry.key")) for entry in entries]
    bounds: dict[str, dict[str, list[int]]] = {}
    for schema_id, coordinates in signatures:
        schema_bounds = bounds.setdefault(schema_id, {})
        for axis_id, coordinate in coordinates:
            schema_bounds.setdefault(axis_id, []).append(coordinate)
    shape_schemas = tuple(
        ShapeSchema(
            shape_schema_id=schema_id,
            axes=tuple(
                ShapeAxis(
                    axis_id=axis_id,
                    unit="tokens",
                    minimum=min(values),
                    maximum=max(values),
                )
                for axis_id, values in axis_bounds.items()
            ),
        )
        for schema_id, axis_bounds in sorted(bounds.items())
    )

    records: list[DeviceServiceEntryRecord] = []
    axis_ids = registry.axis_ids
    for entry, (shape_schema_id, coordinates) in zip(entries, signatures, strict=True):
        epochs: list[ServiceEpochDefinition] = []
        observed_clocks = _object(entry["observed_clocks"], "entry.observed_clocks")
        observed_sm = _object(observed_clocks["sm_hz"], "entry.observed_clocks.sm_hz")
        sm_hz = int(observed_sm["median"])
        for kernel_value in entry["kernels"]:
            kernel = _object(kernel_value, "entry.kernels[]")
            components = _object(kernel["components"], "kernel.components")
            memory = _object(components["memory"], "kernel.components.memory")
            launch_count = int(kernel["launch_count"])
            demand_by_axis = {
                "sm-cycles": int(components["compute_sm_cycles"]) * launch_count,
            }
            if memory_known:
                demand_by_axis.update(
                    {
                        "hbm-weight-bytes": int(memory["weight_bytes"]) * launch_count,
                        "hbm-kv-bytes": int(memory["kv_bytes"]) * launch_count,
                        "hbm-other-bytes": int(memory["other_bytes"]) * launch_count,
                    }
                )
            compute_service_ps = _ceil_div(
                demand_by_axis["sm-cycles"] * PS_PER_SECOND,
                sm_hz,
            )
            memory_bytes = sum(
                demand_by_axis.get(axis_id, 0)
                for axis_id in (
                    "hbm-kv-bytes",
                    "hbm-other-bytes",
                    "hbm-weight-bytes",
                )
            )
            memory_service_ps = _ceil_div(
                memory_bytes * PS_PER_SECOND,
                int(device["memory_capacity_bytes_per_second"]),
            )
            compiled_work_ps = max(compute_service_ps, memory_service_ps)
            measured_total_ps = int(kernel["measured_elapsed_ps"]) * launch_count
            compiled_fixed_ps = measured_total_ps - compiled_work_ps
            if compiled_fixed_ps < 0:
                raise ValueError(
                    f"kernel {kernel['kernel_id']!r} compiled work exceeds measured time"
                )
            epochs.append(
                ServiceEpochDefinition(
                    resource_vector=DeviceResourceVector(
                        registry_sha256=registry_sha256,
                        device_kind_id=str(device["device_kind_id"]),
                        values=tuple(demand_by_axis[axis_id] for axis_id in axis_ids),
                        known=tuple(True for _ in axis_ids),
                    ),
                    fixed_floor_ps=None,
                )
            )
            epochs.append(
                ServiceEpochDefinition(
                    resource_vector=DeviceResourceVector(
                        registry_sha256=registry_sha256,
                        device_kind_id=str(device["device_kind_id"]),
                        values=tuple(0 for _ in axis_ids),
                        known=tuple(True for _ in axis_ids),
                    ),
                    fixed_floor_ps=compiled_fixed_ps,
                )
            )
        entry_hash = canonical_sha256(entry)
        records.append(
            DeviceServiceEntryRecord(
                service_entry_id=f"kernel-cycle-{entry_hash}",
                entry=DeviceServiceEntry(
                    implementation_id=str(entry["implementation_id"]),
                    shape_vector=ShapeVector(
                        shape_schema_id=shape_schema_id,
                        values=tuple(coordinate for _, coordinate in coordinates),
                    ),
                    epochs=tuple(epochs),
                ),
            )
        )
    records_tuple = tuple(sorted(records, key=lambda item: item.service_entry_id))
    validate_device_service_entries(
        registry=registry,
        registry_sha256=registry_sha256,
        shape_schemas=shape_schemas,
        entries=tuple(record.entry for record in records_tuple),
    )
    return DeviceServiceCompilation(
        acceptance_status=str(payload["acceptance_status"]),
        lookup_record_sha256=record.record_id,
        resource_registry=registry,
        resource_registry_sha256=registry_sha256,
        shape_schemas=shape_schemas,
        service_entries=records_tuple,
    )


__all__ = [
    "KERNEL_CYCLE_INPUT_SCHEMA",
    "KERNEL_CYCLE_LUT_SCHEMA",
    "DeviceServiceCompilation",
    "analyze_kernel_cycle_capture",
    "compile_device_service_entries",
    "compile_profile_table",
    "validate_kernel_cycle_lut",
]
