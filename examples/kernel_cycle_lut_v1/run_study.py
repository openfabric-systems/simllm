#!/usr/bin/env python3
"""Run the frozen retained-fixture kernel-cycle lookup study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from simllm.calibration.canonical import canonical_bytes, canonical_loads, sha256_bytes
from simllm.calibration.kernel_cycle_lut import (
    analyze_kernel_cycle_capture,
    compile_device_service_entries,
    compile_profile_table,
    validate_kernel_cycle_lut,
)
from simllm.compute.device_model_io import (
    resource_registry_from_obj,
    service_entry_record_from_obj,
)
from simllm.compute.provider import GpuSpec, KernelSpec, ProfileTableProvider

HERE = Path(__file__).resolve().parent
FREEZE_PATH = HERE / "expectations.json"
EXPECTATION_COMMIT = "10f4ad2b450d5d559cd67d50ccb87e2557e7123d"
PS_PER_SECOND = 1_000_000_000_000


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return value


def _profile_config(key: dict[str, Any]) -> tuple[tuple[str, int], ...]:
    parallelism = key["parallelism"]
    shape = key["shape"]
    result = [
        ("tensor_parallel", parallelism["tensor_parallel"]),
        ("pipeline_parallel", parallelism["pipeline_parallel"]),
        ("data_parallel", parallelism["data_parallel"]),
        ("expert_parallel", parallelism["expert_parallel"]),
        ("batch_size", shape["batch_size"]),
    ]
    result.extend(
        (f"kv_length_{index:04d}", length)
        for index, length in enumerate(shape["per_request_kv_lengths"])
    )
    return tuple(result)


def _key_rejection_guard(record) -> dict[str, Any]:
    routed = json.loads(record.canonical)
    routed["entries"][0]["key"].pop("routing")
    dense = json.loads(record.canonical)
    dense_key = dense["entries"][0]["key"]
    dense_key["model_identity"]["family"] = "dense"
    dense_key["input_dependency"] = "dense-content-independent"
    dense_key.pop("routing")
    dense_key["shape"]["per_request_kv_lengths"] = []
    rejected: list[str] = []
    for name, payload in (("routed-missing-routing", routed), ("dense-missing-kv", dense)):
        try:
            validate_kernel_cycle_lut(payload)
        except ValueError:
            rejected.append(name)
    return {
        "passed": rejected == ["routed-missing-routing", "dense-missing-kv"],
        "rejected": rejected,
    }


def _component_guard(record) -> dict[str, Any]:
    entry = record.value["entries"][0]
    sm_hz = entry["observed_clocks"]["sm_hz"]["median"]
    errors = []
    reconstructed_total = 0
    for kernel in entry["kernels"]:
        components = kernel["components"]
        compute_ps = -(-(components["compute_sm_cycles"] * PS_PER_SECOND) // sm_hz)
        memory_ps = components["memory"]["service_ps"]
        reconstructed = max(compute_ps, memory_ps) + components["fixed_overhead_ps"]
        errors.append(abs(reconstructed - kernel["measured_elapsed_ps"]))
        reconstructed_total += reconstructed * kernel["launch_count"]
    max_error = max(errors)
    return {
        "passed": max_error <= 1 and reconstructed_total == entry["measured_service_ps"],
        "max_abs_reconstruction_error_ps": max_error,
        "reconstructed_service_ps": reconstructed_total,
        "measured_service_ps": entry["measured_service_ps"],
    }


def _compilation_guard(record, output_dir: Path) -> dict[str, Any]:
    entry = record.value["entries"][0]
    provider = compile_profile_table(record.canonical)
    profile_path = output_dir / "profile-table.json"
    provider.save(profile_path)
    loaded = ProfileTableProvider.load(profile_path)
    estimate = loaded.estimate(
        KernelSpec(
            name=entry["implementation_id"],
            flops=0,
            bytes_moved=0,
            config=_profile_config(entry["key"]),
        ),
        GpuSpec(
            name=record.value["device"]["gpu_name"],
            peak_flops=1,
            mem_bandwidth=1,
        ),
    )
    compiled = compile_device_service_entries(record.canonical)
    registry_round_trip = resource_registry_from_obj(compiled.resource_registry.to_obj())
    entry_round_trips = [
        service_entry_record_from_obj(service_entry.to_obj())
        for service_entry in compiled.service_entries
    ]
    sm_hz = entry["observed_clocks"]["sm_hz"]["median"]
    device_duration_ps = 0
    for epoch in entry_round_trips[0].entry.epochs:
        cycles = epoch.resource_vector.values[
            compiled.resource_registry.axis_ids.index("sm-cycles")
        ]
        device_duration_ps += max(
            -(-(cycles * PS_PER_SECOND) // sm_hz),
            epoch.fixed_floor_ps or 0,
        )
    target = entry["measured_service_ps"]
    return {
        "passed": estimate.duration_ps == target
        and device_duration_ps == target
        and registry_round_trip == compiled.resource_registry
        and tuple(entry_round_trips) == compiled.service_entries,
        "profile_table_duration_ps": estimate.duration_ps,
        "device_service_duration_ps": device_duration_ps,
        "target_duration_ps": target,
        "profile_table_schema": "simllm-profile-table-v1",
        "device_service_entry_count": len(compiled.service_entries),
        "acceptance_status": compiled.acceptance_status,
    }


def evaluate(fixture: Path, output_dir: Path) -> dict[str, Any]:
    freeze = _load_json(FREEZE_PATH)
    output_dir.mkdir(parents=True, exist_ok=True)
    record = analyze_kernel_cycle_capture(fixture)
    second = analyze_kernel_cycle_capture(fixture)
    entry = record.value["entries"][0]
    ceiling = freeze["retained_fixture"]["enclosing_wall_step_ps_ceiling"]

    guards = {
        "G1": {
            "passed": len(record.value["sources"]) == 4,
            "verified_fixture_sources": len(record.value["sources"]),
        },
        "G2": {
            "passed": canonical_bytes(canonical_loads(record.canonical)) == record.canonical
            and sha256_bytes(record.canonical) == record.record_id,
            "record_sha256": record.record_id,
        },
        "G3": _component_guard(record),
        "G4": _key_rejection_guard(record),
        "G5": {
            "passed": second.canonical == record.canonical and second.record_id == record.record_id,
            "first_sha256": record.record_id,
            "second_sha256": second.record_id,
        },
        "G6": _compilation_guard(record, output_dir),
        "G7": {
            "passed": all(
                0 < kernel["measured_elapsed_ps"] <= ceiling for kernel in entry["kernels"]
            )
            and entry["observed_clocks"]["sm_hz"]["median"] > 0
            and entry["observed_clocks"]["memory_hz"]["median"] > 0,
            "per_kernel_floor_ps": 1,
            "per_kernel_ceiling_ps": ceiling,
            "maximum_observed_kernel_ps": max(
                kernel["measured_elapsed_ps"] for kernel in entry["kernels"]
            ),
            "observed_sm_clock_hz": entry["observed_clocks"]["sm_hz"]["median"],
            "observed_memory_clock_hz": entry["observed_clocks"]["memory_hz"]["median"],
        },
    }
    violated = [guard_id for guard_id, guard in guards.items() if not guard["passed"]]
    relation_freeze = freeze["scored_relations"][0]
    instances = [
        {
            "kernel_id": kernel["kernel_id"],
            "ratio_ppm": kernel["cross_instrument_ratio_ppm"],
            "passed": relation_freeze["ratio_lower_ppm"]
            <= kernel["cross_instrument_ratio_ppm"]
            <= relation_freeze["ratio_upper_ppm"],
        }
        for kernel in entry["kernels"]
    ]
    relation_passed = not violated and all(instance["passed"] for instance in instances)
    return {
        "schema": "simllm-kernel-cycle-lut-study-result-v1",
        "study": freeze["study"],
        "expectation_commit": EXPECTATION_COMMIT,
        "freeze_sha256": sha256_bytes(FREEZE_PATH.read_bytes()),
        "run_state": "void" if violated else "nonvoid",
        "voiding_guards": violated,
        "evidence_classes": freeze["evidence_classes"],
        "fatal_guards": guards,
        "scored_denominator_relation_families": 1,
        "scored_parameterized_instances": 5,
        "scored_passed_relation_families": int(relation_passed),
        "relations": {
            relation_freeze["id"]: {
                "scored": True,
                "passed": relation_passed,
                "ratio_lower_ppm": relation_freeze["ratio_lower_ppm"],
                "ratio_upper_ppm": relation_freeze["ratio_upper_ppm"],
                "maximum_ratio_ppm": max(instance["ratio_ppm"] for instance in instances),
                "instances": instances,
            }
        },
        "lookup_record": {
            "schema": record.schema,
            "sha256": record.record_id,
            "acceptance_status": record.value["acceptance_status"],
            "coverage": entry["coverage"],
            "measured_service_ps": entry["measured_service_ps"],
            "distribution_verdict": entry["distribution"]["verdict"],
            "replay_count": entry["distribution"]["replay_count"],
        },
        "closure": freeze["closure"],
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = evaluate(args.fixture, args.out.parent)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    if result["run_state"] == "void":
        raise SystemExit(2)
    if (
        result["scored_passed_relation_families"]
        != (result["scored_denominator_relation_families"])
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
