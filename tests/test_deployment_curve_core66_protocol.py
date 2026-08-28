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
