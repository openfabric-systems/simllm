from __future__ import annotations

import hashlib
import json
from pathlib import Path

from simllm.calibration.kernel_cycle_lut import validate_kernel_cycle_lut

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "examples/hopper_kernel_cycle_candidate_v1"
CORE61_RESULT = ROOT / "examples/deployment_curve_v1/core61_depth_comp78_result.json"
PREDECESSOR_SHA256 = "ff46f6d8a79ddae899da89d4db6eb34373f8042acd06cab50b6336c8fb9a8f52"
PARTIAL_SHA256 = "d868a4f35d633032daa238168d00f42c2ab47fc569db649b19b907008072e107"
COMP78_SHA256 = "58d169865109a5eaca3e69978a48080c25a6bb48ee6607d32e82ed8487d17fdd"
PARTIAL = STUDY / "successors" / PARTIAL_SHA256
COMP78 = STUDY / "successors" / COMP78_SHA256


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_comp78_successor_preserves_both_immutable_records_and_frozen_entries() -> None:
    predecessor_bytes = (STUDY / "candidate-record.json").read_bytes()
    partial_bytes = (PARTIAL / "candidate-record.json").read_bytes()
    comp78_bytes = (COMP78 / "candidate-record.json").read_bytes()
    predecessor = validate_kernel_cycle_lut(predecessor_bytes)
    partial = validate_kernel_cycle_lut(partial_bytes)
    comp78 = validate_kernel_cycle_lut(comp78_bytes)

    assert predecessor.record_id == PREDECESSOR_SHA256
    assert partial.record_id == PARTIAL_SHA256
    assert comp78.record_id == COMP78_SHA256
    assert comp78.canonical == comp78_bytes
    assert comp78.value["acceptance_status"] == "candidate"
    assert comp78.value["entries"] == partial.value["entries"]


def test_comp78_score_records_no_invented_depth_or_granite_measurement() -> None:
    result = _read_json(COMP78 / "result.json")
    score = result["score"]
    core61 = score["core61"]
    granite = score["granite"]

    assert result["lookup_record_sha256"] == COMP78_SHA256
    assert result["predecessor_lookup_record_sha256"] == PREDECESSOR_SHA256
    assert core61["preregistered_prediction_ps"] == 3_751_359_511
    assert core61["measured_service_ps"] is None
    assert core61["signed_residual_ps"] is None
    assert core61["linearity_verdict"] == "UNAVAILABLE_WITHOUT_MEASURED_DECODE_SERVICE"
    assert core61["registered_commands_changed"] is False
    assert [attempt["job_id"] for attempt in core61["attempts"]] == [
        200120,
        200123,
        200128,
    ]
    assert granite["completed_cell_count"] == 0
    assert granite["registered_cell_count"] == 1212
    assert granite["completed_prefix_sha256"] == hashlib.sha256(b"").hexdigest()
    assert granite["first_incomplete_cell"] == (
        "sglang-decode-cuda-graph-te1-pi1-da1-ex1-b1-kv1311-deliberately-fragmented"
    )
    assert score["requested_physical_cell_ledger"]["granite_registered_campaign"] == {
        "ABSENT": 1212,
        "MEASURED": 0,
    }
    assert score["task_movement"]["comp78"] == "OPEN_EXACT_REGISTERED_REMAINDER"


def test_comp78_core61_ledger_keeps_residual_unavailable() -> None:
    result = _read_json(CORE61_RESULT)
    held_out = result["held_out_depth"]

    assert result["successor_lookup_record_sha256"] == COMP78_SHA256
    assert result["acceptance"] == {
        "held_out_tolerance_percent": "5",
        "prediction_within_tolerance": None,
        "registered_base_digest_complete": False,
        "registered_decode_digest_complete": False,
    }
    assert held_out["measured_service_ps"] is None
    assert held_out["signed_residual_ps"] is None
    assert held_out["signed_residual_percent"] is None
    assert held_out["signed_residual_rule"].startswith("measured_service_ps - ")
    residual = result["signed_residual_ledger"][-1]
    assert residual["signed_ps"] is None
    assert residual["state"] == "BLOCKED_EXACT_DECODE_STARTUP_OOM"


def test_comp78_manifest_and_retained_source_ledger_are_exact() -> None:
    manifest = _read_json(COMP78 / "artifact-manifest.json")
    for artifact in manifest["artifacts"]:
        data = (COMP78 / artifact["name"]).read_bytes()
        assert len(data) == artifact["bytes"]
        assert hashlib.sha256(data).hexdigest() == artifact["sha256"]

    evidence = _read_json(STUDY / "campaign_evidence_comp78.json")
    sources = evidence["sources"]
    names = [source["name"] for source in sources]
    assert names == sorted(names)
    assert {
        "comp78-core61-base-digest",
        "comp78-core61-decode-200123-digest",
        "comp78-core61-decode-200128-digest",
        "comp78-core61-jobs",
        "comp78-granite-prefix-audit",
        "comp78-granite-target-audit",
        "comp78-granite-target-cell",
        "comp78-granite-target-python",
        "comp78-sglang-target",
        "comp78-vllm-target",
    }.issubset(names)
    assert all(not Path(source["path"]).is_absolute() for source in sources)
