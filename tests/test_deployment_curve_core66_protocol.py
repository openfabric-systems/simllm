from __future__ import annotations

import importlib.util
import io
import json
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]
READER_PATH = ROOT / "examples/deployment_curve_v1/core66_field_reader.py"
PROTOCOL_PATH = ROOT / "examples/deployment_curve_v1/core66_reader_protocol.json"
EP8_EXPECTATIONS_PATH = (
    ROOT / "examples/deployment_curve_v1/core66_ep8_expectations.json"
)
EP4_EXPECTATIONS_PATH = (
    ROOT / "examples/deployment_curve_v1/core66_ep4_expectations.json"
)
EP4_RESULT_PATH = (
    ROOT / "examples/deployment_curve_v1/core66_ep4_capture_result.json"
)
EP4_ENV_RETRY_EXPECTATIONS_PATH = (
    ROOT / "examples/deployment_curve_v1/core66_ep4_env_retry_expectations.json"
)
EP4_ENV_RETRY_RESULT_PATH = (
    ROOT / "examples/deployment_curve_v1/core66_ep4_env_retry_result.json"
)
EP4_FALLBACK_EXPECTATIONS_PATH = (
    ROOT / "examples/deployment_curve_v1/core66_ep4_fallback_expectations.json"
)


def _load_reader() -> ModuleType:
    spec = importlib.util.spec_from_file_location("core66_field_reader", READER_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


reader = _load_reader()


def _source(payload: bytes):
    return reader._PartialSource(io.BytesIO(payload), len(payload))


def test_markdown_section_is_atomic_and_reports_next_heading() -> None:
    payload = (
        b"# CORE-66\n"
        b"Preamble.\n"
        b"## Capture\n"
        b"Selected text.\n"
        b"## Forbidden MTP value\n"
        b"numeric payload must remain unread\n"
    )
    source = _source(payload)

    value = reader.extract_markdown_section(source, "## Capture")

    assert value == {
        "heading": "## Capture",
        "next_heading": "## Forbidden MTP value",
        "text": "## Capture\nSelected text.\n",
    }
    assert source.bytes_accessed <= payload.index(b"numeric payload")


def test_final_markdown_section_leaves_guard_byte_unread() -> None:
    payload = b"# Intro\ntext\n## Final\nlast line\n"
    source = _source(payload)

    value = reader.extract_markdown_section(source, "## Final")

    assert value["text"] == "## Final\nlast line"
    assert value["next_heading"] is None
    assert source.bytes_accessed == len(payload) - 1


def test_missing_markdown_section_is_rejected_as_whole_file_attempt() -> None:
    payload = b"# Intro\ntext\n"
    source = _source(payload)

    with pytest.raises(reader.WholeFileAccessRejected):
        reader.extract_markdown_section(source, "## Missing")

    assert source.bytes_accessed == len(payload) - 1


def test_json_pointer_stops_before_held_out_tail() -> None:
    payload = (
        b'{"capture":{"ep":12,"experts":48},"held_out_mtp":{"numeric":"must remain unread"}}\n'
    )
    source = _source(payload)

    value = reader.extract_json_pointer(source, "/capture")

    assert value == {"ep": 12, "experts": 48}
    assert source.bytes_accessed <= payload.index(b"held_out_mtp")


def test_public_reader_logs_begin_before_open_and_keeps_forbidden_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record = tmp_path / "record.md"
    record.write_text("# Selected\nvalue\n# Tail\nsecret\n", encoding="utf-8", newline="\n")
    access = tmp_path / "access.jsonl"
    forbidden = tmp_path / "forbidden.jsonl"
    recorder = reader.AccessRecorder(access, forbidden)
    monkeypatch.setitem(reader.ALLOWED_RECORDS, "synthetic", record)
    monkeypatch.setattr(reader, "REPOSITORY_ROOT", Path("/"))

    def extractor(source):
        events = [json.loads(line) for line in access.read_text(encoding="utf-8").splitlines()]
        assert [event["event"] for event in events] == ["BEGIN"]
        return reader.extract_markdown_section(source, "# Selected")

    value = reader.read_allowlisted(
        label="synthetic",
        selector="markdown-section:# Selected",
        recorder=recorder,
        extractor=extractor,
    )

    assert value["text"] == "# Selected\nvalue\n"
    events = [json.loads(line) for line in access.read_text(encoding="utf-8").splitlines()]
    assert [event["event"] for event in events] == ["BEGIN", "END"]
    assert events[-1]["bytes_accessed"] < events[-1]["record_size_bytes"]
    assert forbidden.read_bytes() == b""


def test_held_out_selector_is_denied_before_source_open(tmp_path: Path) -> None:
    access = tmp_path / "access.jsonl"
    forbidden = tmp_path / "forbidden.jsonl"
    recorder = reader.AccessRecorder(access, forbidden)

    with pytest.raises(reader.SelectorRejected, match="MTP"):
        reader.read_allowlisted(
            label="hardware-remainder",
            selector="/held_out_mtp/numeric",
            recorder=recorder,
            extractor=lambda source: None,
        )

    assert access.read_bytes() == b""
    denied = [json.loads(line) for line in forbidden.read_text(encoding="utf-8").splitlines()]
    assert denied[0]["source_opened"] is False


def test_protocol_freezes_feasible_cell_and_publication_gate() -> None:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    capture = protocol["capture_freeze"]

    assert protocol["task"] == "CORE-66"
    assert capture["hardware"] == {
        "gpu_model": "NVIDIA GH200",
        "gpus_per_node": 4,
        "node_count": 3,
        "rank_count": 12,
    }
    assert capture["model"]["expert_parallel_width"] == 12
    assert capture["model"]["logical_experts_per_rank"] == 4
    assert capture["model"]["physical_expert_slots_per_rank"] == 4
    assert capture["model"]["routed_expert_total"] == 48
    assert capture["decode"]["requests_per_rank"] == 32
    assert capture["decode"]["kv_length_per_rank"] == 2000
    assert capture["decode"]["mtp_enabled"] is False
    assert capture["decode"]["measured_iterations"] == 1
    assert protocol["comparison_gate"]["never_publish_downward_correction_alone"] is True
    assert protocol["reader"]["forbidden_access_ledger_expected"] == []
    assert protocol["registered_capture_disposition"]["registered_unique_routed_experts"] == 256
    assert protocol["registered_capture_disposition"]["registered_expert_slots"] == 288
    assert protocol["expected_direction_freeze"]["layer_type_composition"] == {
        "dense_only_direction": "decrease",
        "moe_only_direction": "increase",
        "net_direction": "indeterminate until both physical services are bound",
    }
    assert protocol["scale_checks"] == {
        "assignment_compute_candidate": "1/9",
        "count_and_resident_weight_candidate": "1/64",
        "routing_evidence_required": (
            "per layer, rank, routed expert ID, assignment count and local physical slot ID"
        ),
    }
    assert protocol["reader"]["allowed_records"]["core65-result"].endswith(
        "core65_physical_binding_result.json"
    )
    assert protocol["service_multiplier_freeze"] == {
        "common": "61/4",
        "dense": 1,
        "moe": 58,
        "output": 1,
        "step": 1,
    }


def test_ep8_expectations_freeze_cell_scheduler_and_survivable_exposure() -> None:
    expectations = json.loads(EP8_EXPECTATIONS_PATH.read_text(encoding="utf-8"))
    capture = expectations["capture_freeze"]

    assert expectations["task"] == "CORE-66"
    assert capture["hardware"] == {
        "gpu_model": "NVIDIA GH200",
        "gpus_per_node": 4,
        "node_count": 2,
        "rank_count": 8,
    }
    assert capture["model"] == {
        "expert_parallel_width": 8,
        "logical_experts_per_rank": 4,
        "physical_expert_slots_per_rank": 4,
        "routed_expert_total": 32,
        "weights": "dummy-only",
    }
    assert capture["scheduler"] == {
        "account": "merlin",
        "cluster": "gmerlin7",
        "partition": "gh-hourly",
        "qos": "gpu_hourly",
        "submission_limit": 1,
    }
    assert capture["execution"] == {
        "cuda_graph_enabled": False,
        "reason": (
            "eager execution preserves per-layer routing identities and semantic launch "
            "ranges; kernel service remains deterministic across launch modes"
        ),
    }
    assert expectations["comparison_gate"]["never_publish_downward_correction_alone"]
    assert expectations["disclosure_guard"]["fatal_not_survivable"]["disposition"] == (
        "the run is void and CORE-66 remains open"
    )
    exposure = expectations["disclosure_guard"]["incidental_exposure"]
    assert exposure["disposition"].startswith("survivable")
    assert "zero free or fitted parameters" in exposure["reason"]
    assert expectations["service_multiplier_freeze"] == {
        "common": "61/4",
        "dense": 1,
        "moe": 58,
        "output": 1,
        "step": 1,
    }


def test_ep4_expectations_are_a_new_single_gpu_general_cell() -> None:
    expectations = json.loads(EP4_EXPECTATIONS_PATH.read_text(encoding="utf-8"))
    capture = expectations["capture_freeze"]

    assert capture["hardware"] == {
        "gpu_model": "NVIDIA GH200",
        "gpus_per_node": 4,
        "node_count": 1,
        "rank_count": 4,
    }
    assert capture["model"] == {
        "expert_parallel_width": 4,
        "logical_experts_per_rank": 4,
        "physical_expert_slots_per_rank": 4,
        "routed_expert_total": 16,
        "weights": "dummy-only",
    }
    assert capture["scheduler"] == {
        "account": "merlin",
        "cluster": "gmerlin7",
        "partition": "gh-hourly",
        "qos": "gpu_general",
        "submission_limit": 1,
        "time_limit": "00:55:00",
    }
    assert expectations["chronology"]["prior_refusal_records_may_be_amended"] is False
    assert expectations["comparison_gate"]["never_publish_downward_correction_alone"]
    assert expectations["service_multiplier_freeze"] == {
        "common": "61/4",
        "dense": 1,
        "moe": 58,
        "output": 1,
        "step": 1,
    }


def test_ep4_fallback_freeze_is_distinct_and_forces_null_movement() -> None:
    expectations = json.loads(
        EP4_FALLBACK_EXPECTATIONS_PATH.read_text(encoding="utf-8")
    )
    capture = expectations["capture_freeze"]

    assert capture["hardware"] == {
        "gpu_model": "NVIDIA GH200",
        "gpus_per_node": 4,
        "node_count": 1,
        "rank_count": 4,
    }
    assert capture["model"] == {
        "expert_parallel_width": 4,
        "logical_experts_per_rank": 4,
        "physical_expert_slots_per_rank": 4,
        "routed_expert_total": 16,
        "weights": "dummy-only",
    }
    backend = capture["backend"]
    assert backend["deep_ep_enabled"] is False
    assert backend["moe_a2a_backend_argument"] == "none"
    assert backend["moe_dispatcher"].endswith(".StandardDispatcher")
    assert backend["source_builds_forbidden"] == ["DeepEP", "NVSHMEM"]
    assert expectations["chronology"]["this_is_a_new_cell"] is True
    assert expectations["chronology"]["prior_records_may_be_amended"] is False
    gate = expectations["comparison_gate"]
    assert gate["deep_ep_status_for_this_cell"] == "UNPRICED_BY_CONSTRUCTION"
    assert gate["signed_movement_tokens_per_second_per_node"] is None
    assert gate["never_publish_downward_correction_alone"] is True
    differences = {
        row["difference"] for row in expectations["declared_extrapolation_ledger"]
    }
    assert "MoE communication backend" in differences
    assert expectations["service_multiplier_freeze"] == {
        "common": "61/4",
        "dense": 1,
        "moe": 58,
        "output": 1,
        "step": 1,
    }


def test_ep4_result_preserves_failed_capture_and_null_movement() -> None:
    result = json.loads(EP4_RESULT_PATH.read_text(encoding="utf-8"))

    assert result["status"] == "VOID_LAUNCH_ENVIRONMENT_NO_PHYSICAL_CAPTURE"
    assert result["achieved_capture_configuration"] is None
    assert result["hardware"] == {
        "allocated_gpu_count": 4,
        "allocated_job_id": 200879,
        "allocated_node": "gpu002",
        "elapsed_seconds": 14,
        "exit_code": "127:0",
        "feasible_cell_status": "ALLOCATED_LAUNCH_FAILED_BEFORE_SGLANG",
        "gpu_hours_consumed": 56 / 3600,
        "gpu_seconds_consumed": 56,
        "partition": "gh-hourly",
        "qos": "gpu_general",
        "registered_ep72_status": "BLOCKED_IMPOSSIBLE_ON_PROJECT_CLUSTER",
        "scheduler_state": "FAILED",
        "started_at": "2026-08-28T16:17:49",
        "finished_at": "2026-08-28T16:18:03",
        "submission_count": 1,
    }
    assert result["launch_failure"] == {
        "counter_pass_status": "not-run",
        "failure_phase": "before the SGLang process and before CUDA profiling",
        "frozen_module_request": "cuda/13.2.1",
        "module_error": "module load: module does not exist -- cuda/13.2.1",
        "profiler_error": (
            "core66d_node_capture.sh: line 56: nsys: command not found"
        ),
        "timing_pass_exit_code": 127,
    }
    identities = result["identity_and_physics"]
    assert identities["physical_identity_binding_count"] == 0
    assert identities["physical_identity_binding_target"] == 37
    assert identities["deep_ep_dispatch_launch_count"] == 0
    assert identities["deep_ep_combine_launch_count"] == 0
    assert identities["hbm_counter_permission"] == "NOT_REACHED"
    movement = result["calibration_only"]
    assert movement["signed_movement_tokens_per_second_per_node"] is None
    assert movement["service_multipliers_applied"] is False
    assert movement["downward_correction_published_alone"] is False
    assert result["protocol"]["fatal_held_out_use_occurred"] is False


def test_ep4_environment_retry_reuses_core61_and_keeps_cell_unchanged() -> None:
    expectations = json.loads(
        EP4_ENV_RETRY_EXPECTATIONS_PATH.read_text(encoding="utf-8")
    )

    assert expectations["status"] == "EXPECTATIONS_ONLY"
    reference = expectations["capture_reference"]
    assert reference["cell_changes_permitted"] is False
    assert reference["deviation_ledger_changes_permitted"] is False
    assert reference["ep4_expectations_sha256"] == (
        "8202f70f578408fd3aaa985bc7fa3ab3a88726019d8cbef896ae0d049f781d35"
    )
    provenance = expectations["environment_provenance"]
    assert provenance["core61_decode_job"] == 200138
    assert provenance["core61_decode_job_state"] == "COMPLETED"
    assert provenance["core61_merged_script_sha256"] == (
        provenance["core61_remote_script_sha256"]
    )
    environment = expectations["environment_selection"]
    assert environment["module_commands"] == [
        "module purge",
        "module load gcc/12.3.0",
        "module load cuda/12.9.1",
    ]
    assert environment["path_prefix"] == "${SIMLLM_CORE66E_GH200_VENV}/bin"
    assert environment["interpreter"] == (
        "${SIMLLM_CORE66E_GH200_VENV}/bin/python"
    )
    assert environment["python_version"] == "3.11.11"
    assert environment["interpreter_machine"] == "aarch64"
    assert environment["torch_version"] == "2.13.0+cu129"
    assert environment["torch_cuda_version"] == "12.9"
    preflight = expectations["fail_fast_preflight"]
    assert preflight["status"] == "FROZEN_FAIL_CLOSED"
    assert len(preflight["required_before_profiler"]) == 10
    assert expectations["retry_submission"]["submission_limit"] == 1
    assert expectations["comparison_gate"][
        "never_publish_downward_correction_alone"
    ]


def test_ep4_environment_retry_fails_closed_on_deepep_cuda_major() -> None:
    result = json.loads(EP4_ENV_RETRY_RESULT_PATH.read_text(encoding="utf-8"))

    assert result["status"] == "VOID_PREFLIGHT_DEEP_EP_CUDA_MAJOR_MISMATCH"
    assert result["achieved_capture_configuration"] is None
    assert result["hardware"]["allocated_job_id"] == 200891
    assert result["hardware"]["allocated_gpu_count"] == 4
    assert result["hardware"]["elapsed_seconds"] == 75
    assert result["hardware"]["exit_code"] == "4:0"
    assert result["hardware"]["gpu_hours_consumed"] == 300 / 3600
    preflight = result["environment_preflight"]
    assert preflight["status"] == "FAILED_CLOSED_BEFORE_PROFILER"
    assert preflight["profiler_call_count"] == 0
    assert preflight["passed_by_direct_record"]["loaded_modules"] == [
        "gcc/12.3.0",
        "cuda/12.9.1",
    ]
    assert preflight["passed_by_direct_record"]["nvcc_version"] == "12.9.86"
    assert preflight["passed_by_direct_record"]["nsys_version"].startswith(
        "2025.1.3.140"
    )
    deep_ep = preflight["deep_ep"]
    assert deep_ep["build_expected_cuda_major"] == 13
    assert deep_ep["build_cuda_tag"] == "cu130"
    assert deep_ep["selected_interpreter_machine"] == "aarch64"
    assert deep_ep["static_wheel_tags_compatible"] is False
    identities = result["identity_and_physics"]
    assert identities["physical_identity_binding_count"] == 0
    assert identities["deep_ep_dispatch_launch_count"] == 0
    assert identities["deep_ep_combine_launch_count"] == 0
    assert identities["hbm_counter_permission"] == "NOT_REACHED"
    movement = result["calibration_only"]
    assert movement["signed_movement_tokens_per_second_per_node"] is None
    assert movement["service_multipliers_applied"] is False
    assert movement["downward_correction_published_alone"] is False
    assert result["protocol"]["fatal_held_out_use_occurred"] is False
