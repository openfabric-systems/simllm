from __future__ import annotations

import hashlib
import json
from pathlib import Path

from simllm.calibration.kernel_cycle_lut import validate_kernel_cycle_lut

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "examples/hopper_kernel_cycle_candidate_v1"
PREDECESSOR_SHA256 = "ff46f6d8a79ddae899da89d4db6eb34373f8042acd06cab50b6336c8fb9a8f52"
SUCCESSOR_SHA256 = "d868a4f35d633032daa238168d00f42c2ab47fc569db649b19b907008072e107"
SUCCESSOR = STUDY / "successors" / SUCCESSOR_SHA256


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_successor_preserves_every_frozen_price_and_distribution() -> None:
    predecessor_bytes = (STUDY / "candidate-record.json").read_bytes()
    successor_bytes = (SUCCESSOR / "candidate-record.json").read_bytes()
    predecessor = validate_kernel_cycle_lut(predecessor_bytes)
    successor = validate_kernel_cycle_lut(successor_bytes)

    assert predecessor.record_id == PREDECESSOR_SHA256
    assert successor.record_id == SUCCESSOR_SHA256
    assert successor.canonical == successor_bytes
    old_entries = {entry["implementation_id"]: entry for entry in predecessor.value["entries"]}
    new_entries = {entry["implementation_id"]: entry for entry in successor.value["entries"]}
    assert new_entries.keys() == old_entries.keys()
    for implementation_id, entry in new_entries.items():
        old = old_entries[implementation_id]
        assert entry["measured_service_ps"] == old["measured_service_ps"]
        assert entry["distribution"] == old["distribution"]
    assert all("mtp" not in implementation_id for implementation_id in new_entries)


def test_successor_ledger_records_measured_mtp_and_exact_remainder() -> None:
    result = _read_json(SUCCESSOR / "result.json")
    score = result["score"]

    assert result["lookup_record_sha256"] == SUCCESSOR_SHA256
    assert result["predecessor_lookup_record_sha256"] == PREDECESSOR_SHA256
    assert score["mtp"] == {
        "evidence_class": "MEASURED",
        "lookup_pricing": "FORBIDDEN_BY_FREEZE",
        "measured_service_ps": 2_033_951_000,
    }
    assert score["requested_physical_cell_ledger"] == {
        "deepseek_v3": {"ABSENT": 0, "MEASURED": 5},
        "granite_registered_campaign": {"ABSENT": 1212, "MEASURED": 0},
    }
    assert len(score["priced_repeat_observations"]) == 4
    assert all(
        row["retained_independent_observations"] >= 2
        for row in score["priced_repeat_observations"]
    )
    assert score["core61"]["status"] == "NOT_SUBMITTED_TIME_GATE"
    assert score["core61"]["preregistered_prediction_ps"] == 3_751_359_511
    assert score["task_movement"]["remainder_owner"] == "COMP-78"


def test_successor_manifest_matches_every_payload_byte() -> None:
    manifest = _read_json(SUCCESSOR / "artifact-manifest.json")

    for artifact in manifest["artifacts"]:
        data = (SUCCESSOR / artifact["name"]).read_bytes()
        assert len(data) == artifact["bytes"]
        assert hashlib.sha256(data).hexdigest() == artifact["sha256"]


def test_campaign_source_paths_are_portable_and_sorted() -> None:
    evidence = _read_json(STUDY / "campaign_evidence.json")
    sources = evidence["sources"]
    names = [source["name"] for source in sources]

    assert names == sorted(names)
    assert all(not Path(source["path"]).is_absolute() for source in sources)
    assert evidence["granite"]["completed_prefix_sha256"] == hashlib.sha256(b"").hexdigest()
