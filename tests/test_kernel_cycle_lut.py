from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from simllm.calibration.kernel_cycle_lut import (
    KERNEL_CYCLE_LUT_SCHEMA,
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

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "kernel_cycle_lut_v1"
EXPECTED_RECORD_SHA256 = "e495f3ca5d0858cf371b19205ae6b7747d633695020d10f58645c5f245086070"
EXPECTED_SERVICE_PS = 2_047_488_000


@pytest.fixture(scope="module")
def record():
    return analyze_kernel_cycle_capture(FIXTURE)


def _mutable(record) -> dict:
    return json.loads(record.canonical)


def test_retained_fixture_emits_one_canonical_candidate_record(record) -> None:
    assert record.schema == KERNEL_CYCLE_LUT_SCHEMA
    assert record.record_id == EXPECTED_RECORD_SHA256
    assert record.value["acceptance_status"] == "candidate"
    assert record.value["entries"][0]["coverage"] == "partial-kernel-subset"
    assert record.value["entries"][0]["measured_service_ps"] == EXPECTED_SERVICE_PS
    assert validate_kernel_cycle_lut(record.canonical).canonical == record.canonical


def test_identical_fixture_inputs_emit_identical_bytes(record) -> None:
    second = analyze_kernel_cycle_capture(FIXTURE)

    assert second.canonical == record.canonical
    assert second.record_id == record.record_id


def test_fixture_digest_mismatch_rejects_before_analysis(tmp_path: Path) -> None:
    copied = tmp_path / "fixture"
    shutil.copytree(FIXTURE, copied)
    path = copied / "observed-clocks.csv"
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"input_manifest.sources\[3\].sha256"):
        analyze_kernel_cycle_capture(copied)


def test_raw_nvidia_smi_clock_columns_select_active_samples(tmp_path: Path) -> None:
    copied = tmp_path / "fixture"
    shutil.copytree(FIXTURE, copied)
    path = copied / "observed-clocks.csv"
    path.write_text(
        "clocks.current.sm [MHz],clocks.current.memory [MHz],utilization.gpu [%]\n"
        "210 MHz,1215 MHz,0 %\n"
        "1410 MHz,1593 MHz,96 %\n"
        "1395 MHz,1593 MHz,92 %\n",
        encoding="utf-8",
    )
    manifest_path = copied / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source = next(row for row in manifest["sources"] if row["name"] == "observed-clocks")
    data = path.read_bytes()
    source["bytes"] = len(data)
    source["sha256"] = hashlib.sha256(data).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    record = analyze_kernel_cycle_capture(copied)

    clocks = record.value["entries"][0]["observed_clocks"]
    assert clocks["sm_hz"] == {
        "min": 1_395_000_000,
        "median": 1_402_500_000,
        "max": 1_410_000_000,
    }
    assert clocks["memory_hz"] == {
        "min": 1_593_000_000,
        "median": 1_593_000_000,
        "max": 1_593_000_000,
    }


def test_dense_decode_key_requires_every_request_kv_length(record) -> None:
    payload = _mutable(record)
    key = payload["entries"][0]["key"]
    key["model_identity"]["family"] = "dense"
    key["input_dependency"] = "dense-content-independent"
    key.pop("routing")
    key["shape"]["per_request_kv_lengths"] = []

    with pytest.raises(ValueError, match="length must equal batch_size"):
        validate_kernel_cycle_lut(payload)


def test_routed_key_requires_routing_member(record) -> None:
    payload = _mutable(record)
    payload["entries"][0]["key"].pop("routing")

    with pytest.raises(ValueError, match=r"missing fields \['routing'\]"):
        validate_kernel_cycle_lut(payload)


def test_validated_routed_key_requires_captured_loads(record) -> None:
    payload = _mutable(record)
    payload["acceptance_status"] = "validated"

    with pytest.raises(ValueError, match="validated routed entries require captured loads"):
        validate_kernel_cycle_lut(payload)


def test_dense_key_forbids_routing_even_when_shape_is_complete(record) -> None:
    payload = _mutable(record)
    key = payload["entries"][0]["key"]
    key["model_identity"]["family"] = "dense"
    key["input_dependency"] = "dense-content-independent"

    with pytest.raises(ValueError, match=r"unknown fields \['routing'\]"):
        validate_kernel_cycle_lut(payload)


def test_component_rows_reconstruct_exactly_at_the_observed_clock(record) -> None:
    entry = record.value["entries"][0]
    sm_hz = entry["observed_clocks"]["sm_hz"]["median"]
    total = 0
    for kernel in entry["kernels"]:
        components = kernel["components"]
        compute_ps = -(-(components["compute_sm_cycles"] * 1_000_000_000_000) // sm_hz)
        memory_ps = components["memory"]["service_ps"]
        reconstructed = max(compute_ps, memory_ps) + components["fixed_overhead_ps"]
        assert reconstructed == kernel["measured_elapsed_ps"]
        total += reconstructed * kernel["launch_count"]
    assert total == entry["measured_service_ps"] == EXPECTED_SERVICE_PS


def test_distribution_is_explicitly_insufficient_for_a_graph_claim(record) -> None:
    distribution = record.value["entries"][0]["distribution"]

    assert distribution["replay_count"] == 16
    assert distribution["peak_count"] == 1
    assert distribution["verdict"] == "insufficient-replays"
    assert distribution["trimmed_coefficient_of_variation_ppm"] == 7681


def test_all_five_cross_instrument_ratios_meet_the_frozen_candidate_band(record) -> None:
    ratios = [
        kernel["cross_instrument_ratio_ppm"] for kernel in record.value["entries"][0]["kernels"]
    ]

    assert ratios == [1_739_130, 1_262_626, 1_441_026, 1_375_744, 1_355_311]
    assert all(500_000 <= ratio <= 2_000_000 for ratio in ratios)


def test_profile_table_projection_round_trips_without_changing_duration(
    record,
    tmp_path: Path,
) -> None:
    provider = compile_profile_table(record.canonical)
    path = tmp_path / "profile-table.json"
    provider.save(path)
    loaded = ProfileTableProvider.load(path)
    entry = record.value["entries"][0]
    key = entry["key"]
    shape = key["shape"]
    config = (
        ("tensor_parallel", 1),
        ("pipeline_parallel", 1),
        ("data_parallel", 1),
        ("expert_parallel", 1),
        ("batch_size", 1),
        ("kv_length_0000", shape["per_request_kv_lengths"][0]),
    )
    estimate = loaded.estimate(
        KernelSpec(name=entry["implementation_id"], flops=0, bytes_moved=0, config=config),
        GpuSpec(
            name=record.value["device"]["gpu_name"],
            peak_flops=1,
            mem_bandwidth=1,
        ),
    )

    assert estimate.duration_ps == EXPECTED_SERVICE_PS
    assert loaded.provenance is not None
    assert loaded.provenance.source == "capture-candidate"
    assert loaded.provenance.references == (f"record-sha256:{record.record_id}",)


def test_device_service_projection_uses_existing_round_trip_forms(record) -> None:
    compiled = compile_device_service_entries(record.canonical)

    assert compiled.acceptance_status == "candidate"
    assert compiled.lookup_record_sha256 == record.record_id
    assert resource_registry_from_obj(compiled.resource_registry.to_obj()) == (
        compiled.resource_registry
    )
    assert len(compiled.service_entries) == 1
    service_record = compiled.service_entries[0]
    assert service_entry_record_from_obj(service_record.to_obj()) == service_record
    assert [axis.axis_id for axis in compiled.resource_registry.axes] == ["sm-cycles"]
    assert len(service_record.entry.epochs) == 10

    sm_hz = record.value["entries"][0]["observed_clocks"]["sm_hz"]["median"]
    duration_ps = 0
    for epoch in service_record.entry.epochs:
        cycles = epoch.resource_vector.values[0]
        duration_ps += max(
            -(-(cycles * 1_000_000_000_000) // sm_hz),
            epoch.fixed_floor_ps or 0,
        )
    assert duration_ps == EXPECTED_SERVICE_PS


def test_unknown_code_objects_remain_nullable(record) -> None:
    for kernel in record.value["entries"][0]["kernels"]:
        assert kernel["code_object"] == {
            "ptx_sha256": None,
            "sass_sha256": None,
            "compile_configuration_sha256": None,
        }


def test_candidate_entry_accepts_optional_measured_evidence(record) -> None:
    payload = _mutable(record)
    source_sha256 = payload["sources"][0]["fixture_sha256"]
    payload["entries"][0]["evidence"] = {
        "service_class": "MEASURED",
        "component_class": "DISCLOSED",
        "split": "calibration",
        "source_sha256s": [source_sha256],
        "derivation": None,
    }

    validated = validate_kernel_cycle_lut(payload)

    assert validated.value["entries"][0]["evidence"]["service_class"] == "MEASURED"


def test_declared_entry_evidence_requires_derivation(record) -> None:
    payload = _mutable(record)
    source_sha256 = payload["sources"][0]["fixture_sha256"]
    payload["entries"][0]["evidence"] = {
        "service_class": "DECLARED",
        "component_class": "DISCLOSED",
        "split": "held-out",
        "source_sha256s": [source_sha256],
        "derivation": None,
    }

    with pytest.raises(ValueError, match="declared service requires a derivation"):
        validate_kernel_cycle_lut(payload)
