from __future__ import annotations

import hashlib
import json
import runpy
from pathlib import Path

from simllm.calibration.kernel_cycle_lut import (
    compile_device_service_entries,
    compile_profile_table,
    validate_kernel_cycle_lut,
)
from simllm.compute.device_model_io import (
    resource_registry_from_obj,
    service_entry_record_from_obj,
)
from simllm.compute.provider import GpuSpec, KernelSpec

STUDY_DIR = Path(__file__).resolve().parents[1] / "examples" / "hopper_kernel_cycle_candidate_v1"
PUBLISHED_RECORD_SHA256 = "ff46f6d8a79ddae899da89d4db6eb34373f8042acd06cab50b6336c8fb9a8f52"


def _candidate_value() -> dict:
    namespace = runpy.run_path(str(STUDY_DIR / "run_study.py"))
    inputs = json.loads((STUDY_DIR / "retained_evidence.json").read_text(encoding="utf-8"))
    sources = {str(source["name"]): source for source in inputs["sources"]}
    return namespace["_build_record"](inputs, sources)


def test_candidate_contract_accepts_all_and_only_retained_service_rows() -> None:
    record = validate_kernel_cycle_lut(_candidate_value())
    entries = record.value["entries"]

    assert record.value["acceptance_status"] == "candidate"
    assert len(entries) == 20
    assert [entry["implementation_id"] for entry in entries] == sorted(
        entry["implementation_id"] for entry in entries
    )
    assert all("evidence" in entry for entry in entries)
    assert all("mtp" not in entry["implementation_id"] for entry in entries)


def test_candidate_evidence_ledger_preserves_non_additive_classes() -> None:
    entries = validate_kernel_cycle_lut(_candidate_value()).value["entries"]
    granite = [entry for entry in entries if entry["implementation_id"].startswith("granite")]
    deepseek = [entry for entry in entries if entry["implementation_id"].startswith("deepseek")]

    assert len(granite) == 12
    assert {entry["evidence"]["service_class"] for entry in granite} == {"MEASURED"}
    assert sum(entry["evidence"]["service_class"] == "MEASURED" for entry in deepseek) == 4
    assert sum(entry["evidence"]["service_class"] == "DECLARED" for entry in deepseek) == 4
    assert {entry["evidence"]["component_class"] for entry in entries} == {"DISCLOSED"}


def test_profile_and_device_contracts_compile_without_a_binding_adapter() -> None:
    record = validate_kernel_cycle_lut(_candidate_value())
    profile = compile_profile_table(record.canonical)
    device = compile_device_service_entries(record.canonical)
    measured = next(
        entry
        for entry in record.value["entries"]
        if entry["implementation_id"]
        == "granite-3.0-1b-a400m-instruct-vllm-graph-decode-b1-kv16"
    )
    estimate = profile.estimate(
        KernelSpec(
            name=measured["implementation_id"],
            flops=0,
            bytes_moved=0,
            config=(
                ("tensor_parallel", 1),
                ("pipeline_parallel", 1),
                ("data_parallel", 1),
                ("expert_parallel", 1),
                ("batch_size", 1),
                ("kv_length_0000", 16),
            ),
        ),
        GpuSpec(name=record.value["device"]["gpu_name"], peak_flops=1, mem_bandwidth=1),
    )

    assert estimate.duration_ps == measured["measured_service_ps"]
    assert profile.provenance is not None
    assert profile.provenance.references == (f"record-sha256:{record.record_id}",)
    assert device.lookup_record_sha256 == record.record_id
    assert device.acceptance_status == "candidate"
    assert len(device.service_entries) == len(record.value["entries"])


def test_granite_identity_is_the_exact_retained_routed_model() -> None:
    entries = validate_kernel_cycle_lut(_candidate_value()).value["entries"]
    granite = [entry for entry in entries if entry["implementation_id"].startswith("granite")]

    assert {entry["key"]["model_identity"]["name"] for entry in granite} == {
        "ibm-granite/granite-3.0-1b-a400m-instruct"
    }
    assert {entry["key"]["model_identity"]["family"] for entry in granite} == {"routed"}
    assert {entry["key"]["routing"]["availability"] for entry in granite} == {
        "not-captured"
    }


def test_declared_deepseek_rows_are_exact_61_over_4_transforms() -> None:
    entries = validate_kernel_cycle_lut(_candidate_value()).value["entries"]
    by_id = {entry["implementation_id"]: entry for entry in entries}
    for implementation_id, declared in by_id.items():
        if "deepseek-v3-full61" not in implementation_id:
            continue
        measured_id = implementation_id.replace("full61", "reduced4")
        measured = by_id[measured_id]
        assert declared["measured_service_ps"] == measured["measured_service_ps"] * 61 // 4
        assert declared["evidence"]["derivation"] == (
            "retained reduced-depth service multiplied by 61 / 4"
        )
        assert declared["key"]["model_identity"]["name"] == "deepseek-ai/DeepSeek-V3"
        assert declared["key"]["model_identity"]["weights_sha256"] == (
            "ec8b878368c5fdb9f3288bd3a36a723a1637ec76464135a3f5b2e9aeff4072b4"
        )


def test_frozen_score_accepts_an_omitted_zero_class() -> None:
    namespace = runpy.run_path(str(STUDY_DIR / "run_study.py"))
    freeze = json.loads((STUDY_DIR / "expectations.json").read_text(encoding="utf-8"))

    score = namespace["_score"](_candidate_value(), freeze)

    assert score["verdict"] == "CANDIDATE_COMPILED"
    assert score["evidence_ledger"]["granite"] == {"MEASURED": 12}


def test_published_artifacts_bind_the_candidate_and_evidence_classes() -> None:
    record_bytes = (STUDY_DIR / "candidate-record.json").read_bytes()
    record = validate_kernel_cycle_lut(record_bytes)
    profile_payload = json.loads((STUDY_DIR / "profile-table.json").read_text(encoding="utf-8"))
    device_payload = json.loads(
        (STUDY_DIR / "device-service-entries.json").read_text(encoding="utf-8")
    )
    result = json.loads((STUDY_DIR / "result.json").read_text(encoding="utf-8"))

    assert record.record_id == PUBLISHED_RECORD_SHA256
    assert record.canonical == record_bytes
    assert profile_payload["lookup_record_sha256"] == PUBLISHED_RECORD_SHA256
    assert device_payload["lookup_record_sha256"] == PUBLISHED_RECORD_SHA256
    assert result["lookup_record_sha256"] == PUBLISHED_RECORD_SHA256
    assert all("evidence" in entry for entry in profile_payload["entries"])
    assert all("evidence" in entry for entry in device_payload["service_entries"])
    resource_registry_from_obj(device_payload["resource_registry"])
    for entry in device_payload["service_entries"]:
        service_entry_record_from_obj(entry["service_entry"])


def test_published_manifest_matches_every_payload_byte() -> None:
    manifest = json.loads((STUDY_DIR / "artifact-manifest.json").read_text(encoding="utf-8"))

    for artifact in manifest["artifacts"]:
        data = (STUDY_DIR / artifact["name"]).read_bytes()
        assert len(data) == artifact["bytes"]
        assert hashlib.sha256(data).hexdigest() == artifact["sha256"]
