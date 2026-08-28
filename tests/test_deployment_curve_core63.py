from __future__ import annotations

import csv
import hashlib
import importlib.util
import io
import json
from fractions import Fraction
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
STUDY_DIR = REPOSITORY_ROOT / "examples/deployment_curve_v1"


def _module(filename: str, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, STUDY_DIR / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _reader():
    return _module("core63_field_reader.py", "deployment_curve_core63_reader")


def _sign():
    return _module("core63_independent_sign.py", "deployment_curve_core63_sign")


def _expectations() -> dict:
    return json.loads(
        (STUDY_DIR / "core63_expectations.json").read_text(encoding="utf-8")
    )


def _selected_entry() -> dict:
    return {
        "coverage": "complete-kernel-stream",
        "evidence": {"component_class": "DISCLOSED", "service_class": "MEASURED"},
        "implementation_id": "deepseek-v3-reduced4-vllm-ep72-decode-b32-c2000",
        "kernels": [
            {
                "components": {
                    "compute_sm_cycles": 3_751_359,
                    "fixed_overhead_ps": 500,
                    "memory": {"service_ps": 0, "forbidden": "not decoded"},
                    "method": "retained additive service",
                },
                "kernel_id": "aggregate_noncollective_step_service",
                "launch_count": 1,
                "measured_elapsed_ps": 1_875_680_000,
                "forbidden": "not decoded",
            }
        ],
        "key": {
            "launch_mode": "cuda-graph",
            "parallelism": {
                "tensor_parallel": 1,
                "pipeline_parallel": 1,
                "data_parallel": 1,
                "expert_parallel": 1,
            },
            "pool": "decode",
            "shape": {"batch_size": 32, "per_request_kv_lengths": [2000] * 32},
        },
        "measured_service_ps": 1_875_680_000,
        "observed_clocks": {"sm_hz": {"median": 2_000_000_000}},
    }


def _synthetic_candidate() -> bytes:
    prefix = [
        {"implementation_id": f"forbidden-{index}", "secret": "not decoded"}
        for index in range(7)
    ]
    return json.dumps(
        {
            "entries": prefix
            + [_selected_entry()]
            + [{"forbidden_tail": "must remain unread"}],
            "forbidden_top_level_tail": "must remain unread",
        },
        sort_keys=True,
    ).encode("utf-8")


def _csv_header() -> bytes:
    return (",".join(_reader().EXPECTED_HEADER) + "\n").encode()


def _legacy_csv_header() -> bytes:
    return (",".join(_reader().LEGACY_HEADER) + "\r\n").encode()


def _csv_row(
    *,
    pool: str,
    shape: int,
    name: str,
    collective: bool = False,
    total_ns: str = "100.5",
) -> bytes:
    values = [
        pool,
        str(shape),
        "0",
        "7",
        name,
        str(collective),
        "nccl" if collective else "",
        "2",
        "50.25",
        "50.25",
        total_ns,
        "0.5",
        "32",
        "32",
    ]
    output = io.StringIO(newline="")
    csv.writer(output, lineterminator="\n").writerow(values)
    return output.getvalue().encode()


def _legacy_csv_row(*, pool: str, shape: int, name: str) -> bytes:
    current = dict(
        zip(
            _reader().EXPECTED_HEADER,
            next(csv.reader([_csv_row(pool=pool, shape=shape, name=name).decode()])),
            strict=True,
        )
    )
    output = io.StringIO(newline="")
    csv.writer(output, lineterminator="\r\n").writerow(
        [current[field] for field in _reader().LEGACY_HEADER]
    )
    return output.getvalue().encode()


def test_expectations_freeze_exact_residency_scale_and_direction() -> None:
    frozen = _expectations()
    arithmetic = frozen["architecture_arithmetic"]

    expected = Fraction(
        arithmetic["disclosed_batch_per_node"]
        * arithmetic["top_k"]
        * arithmetic["resident_physical_slots_per_rank"],
        arithmetic["physical_expert_slots"],
    )
    captured = Fraction(arithmetic["captured_routed_assignments_per_rank"]["value"])

    assert frozen["status"] == "EXPECTATIONS_ONLY"
    assert expected == Fraction(256, 9)
    assert expected / captured == Fraction(1, 9)
    assert arithmetic["assignment_scale"] == {"numerator": 1, "denominator": 9}
    assert frozen["component_rule"]["zero_free_or_fitted_constants"] is True
    assert frozen["expected_signed_direction"] == {
        "corrected_step": "decrease",
        "prediction": "increase",
        "signed_residual": "less_negative_before_any_possible_crossing",
    }


def test_independent_sign_is_anchor_free_and_strict() -> None:
    observed = _sign().residency_sign(
        retained_service_ps=3,
        routed_service_ps=9,
        fixed_service_ps=1,
        routed_scale=Fraction(1, 9),
    )

    assert observed["corrected_step"] == "decrease"
    assert observed["predicted_throughput"] == "increase"


def test_candidate_projection_stops_before_record_tail() -> None:
    payload = _synthetic_candidate()

    value, consumed = _reader().extract_component_basis(io.BytesIO(payload))

    assert value["implementation_id"] == _reader().EXPECTED_IMPLEMENTATION_ID
    assert "forbidden" not in value["kernels"][0]
    assert "forbidden" not in value["kernels"][0]["components"]["memory"]
    assert consumed < len(payload)
    assert b"must remain unread" in payload[consumed:]


def test_csv_reader_decodes_only_selected_payloads_and_filters_collectives() -> None:
    payload = b"".join(
        [
            _csv_header(),
            b"decode,8,0," + b"\xff" * 20 + b"\n",
            _csv_row(pool="decode", shape=32, name="fused_moe_kernel"),
            _csv_row(
                pool="decode",
                shape=32,
                name="ncclKernel",
                collective=True,
            ),
            _csv_row(pool="prefill", shape=32, name="forbidden"),
        ]
    )

    rows, consumed = _reader().extract_standard_decode_kernels(io.BytesIO(payload))

    assert consumed == len(payload)
    assert [row["name"] for row in rows] == ["fused_moe_kernel"]
    assert rows[0]["total_duration_per_step_ns"] == "100.5"


def test_csv_reader_accepts_unselected_schema_columns() -> None:
    header = _csv_header().decode().rstrip("\n") + ",diagnostic\r\n"
    selected = _csv_row(
        pool="decode",
        shape=32,
        name="fused_moe_kernel",
    ).decode().rstrip("\n")
    payload = (header + selected + ",ignored\r\n").encode()

    rows, _ = _reader().extract_standard_decode_kernels(io.BytesIO(payload))

    assert [row["name"] for row in rows] == ["fused_moe_kernel"]


def test_csv_reader_accepts_exact_logged_legacy_schema() -> None:
    payload = _legacy_csv_header() + _legacy_csv_row(
        pool="decode",
        shape=32,
        name="fused_moe_kernel",
    )

    rows, consumed = _reader().extract_standard_decode_kernels(io.BytesIO(payload))

    assert consumed == len(payload)
    assert [row["name"] for row in rows] == ["fused_moe_kernel"]


def test_csv_reader_header_rejection_carries_observed_fields() -> None:
    reader = _reader()
    payload = b"pool,shape,name\r\ndecode,32,fused_moe_kernel\r\n"

    with pytest.raises(reader.KernelSummaryHeaderError) as caught:
        reader.extract_standard_decode_kernels(io.BytesIO(payload))

    assert caught.value.observed_header == ("pool", "shape", "name")


def test_public_reader_refuses_other_candidate_and_logs_attempt(tmp_path: Path) -> None:
    access_log = tmp_path / "access.jsonl"

    with pytest.raises(ValueError, match="refuses"):
        _reader().read_component_basis(tmp_path / "candidate-record.json", access_log)

    entries = [
        json.loads(line)
        for line in access_log.read_text(encoding="utf-8").splitlines()
    ]
    assert entries == [
        {
            "classification": "retained_measured_component_decomposition",
            "error": "ValueError",
            "held_out_numeric_value_accessed": False,
            "record": "examples/hopper_kernel_cycle_candidate_v1/candidate-record.json",
            "schema": "simllm-deployment-curve-core63-access-v1",
            "selector": "/entries[7]",
            "status": "REJECTED",
            "unselected_values_decoded": False,
            "whole_record_loaded": False,
        }
    ]


def test_preservation_manifest_covers_and_matches_all_93_artifacts() -> None:
    manifest = STUDY_DIR / "core63_preservation_lock.sha256"
    rows = [line.split("  ", 1) for line in manifest.read_text().splitlines()]

    assert len(rows) == _expectations()["preservation_lock"]["artifact_count"] == 93
    assert hashlib.sha256(manifest.read_bytes()).hexdigest() == (
        _expectations()["preservation_lock"]["manifest_sha256"]
    )
    for expected, relative in rows:
        path = REPOSITORY_ROOT / relative
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected


def test_core63_registered_and_core64_remains_free() -> None:
    core = (REPOSITORY_ROOT / "docs/modules/core.md").read_text(encoding="utf-8")
    ledger = (REPOSITORY_ROOT / "docs/task-ledger.json").read_text(encoding="utf-8")

    assert core.count("- CORE-63 (") == 1
    assert "CORE-63" not in ledger
    assert "- CORE-64 (" not in core
    assert "CORE-64" not in ledger
