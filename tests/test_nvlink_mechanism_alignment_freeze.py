import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "examples" / "nvlink_mechanism_alignment_v1"
EXPECTATIONS = STUDY / "expectations.json"
EXPECTATIONS_SHA256 = "fafe9bbe730d9c424f7d4f72fd2df3d5fa2cdd8ad9f370f63ff60194374c58cc"


def _frozen() -> dict[str, object]:
    return json.loads(EXPECTATIONS.read_text(encoding="utf-8"))


def test_freeze_is_expectations_only_and_precedes_behavior() -> None:
    frozen = _frozen()

    assert hashlib.sha256(EXPECTATIONS.read_bytes()).hexdigest() == EXPECTATIONS_SHA256
    assert frozen["schema"] == "simllm-nvlink-mechanism-alignment-expectations-v1"
    assert frozen["study"]["task"] == "TRAF-80"
    assert frozen["study"]["status"] == "EXPECTATIONS_ONLY"
    assert frozen["study"]["cluster_time"] is False
    assert "before the aligned mechanism implementation" in frozen["study"][
        "chronology"
    ]
    assert not (STUDY / "results.json").exists()
    assert not (STUDY / "RESULTS.md").exists()
    assert not (STUDY / "run_study.py").exists()


def test_minimum_physical_oracles_are_exactly_frozen() -> None:
    frozen = _frozen()
    physical = frozen["physical_oracles"]
    rows = {row["packet_flits"]: row for row in physical["occupancy_rows"]}

    assert physical["fixed_job_payload_bytes"] == 1_048_576
    assert physical["links"] == 4
    assert physical["a100_per_link_bytes_per_second"] == 25_000_000_000
    assert rows[17]["payload_ceiling_gbps_reported"] == 94.117647
    assert rows[18]["payload_ceiling_gbps_reported"] == 88.888889
    assert physical["optional_flit_serialization_shift_percent_reported"] == 5.882
    assert physical["optional_flit_serialization_shift_percent_exact"] > 0
    assert physical["error_free_replay"] == {
        "added_wire_bytes": 0,
        "added_time_ps": 0,
    }
    assert physical["injected_error_replay"]["minimum_added_wire_bytes"] == 0
    assert physical["injected_error_replay"]["minimum_added_time_ps"] == 0
    assert "greater than or equal" in physical["credit_ownership"]


def test_sanity_sweep_varies_occupancy_and_link_rate_for_one_job() -> None:
    frozen = _frozen()
    sweep = frozen["sanity_sweep"]

    assert sweep["packet_occupancy_flits"] == [17, 18]
    assert sweep["per_link_bytes_per_second"] == [12_500_000_000, 25_000_000_000]
    assert sweep["measured_outcome"] == "job_completion_time_ps"
    assert "1048576-byte" in sweep["fixed_job"]
    assert len(sweep["relations"]) == 5


def test_unidentified_values_remain_declared_candidates() -> None:
    candidates = _frozen()["declared_candidate_defaults"]

    assert len(candidates) == 7
    assert all(row["status"] == "DECLARED_CANDIDATE" for row in candidates)
    assert all("TRAF-79" in row["provenance"] for row in candidates)
    assert {row["parameter"] for row in candidates} == {
        "A100 credit quantum",
        "A100 credit-pool scope and depth",
        "A100 virtual-channel count and class map",
        "A100 receive-buffer depth",
        "A100 credit-return encoding",
        "A100 bonded-link striping granularity",
        "NVSwitch product arbiter",
    }


def test_every_inherited_envelope_is_pinned_to_a_signed_zero_shift() -> None:
    frozen = _frozen()
    envelopes = frozen["inherited_envelopes"]

    assert len(envelopes) == 6
    assert {row["id"] for row in envelopes} == {
        "nvlink-flow-dynamics-v1",
        "nvlink-rnic-comparison-v1",
        "nvlink-rnic-comparison-v2",
        "nvlink-incast-validation-v1",
        "deployment-frontier-v1",
        "deployment-curve-v1-run3",
    }
    assert all(row["required_signed_shift"].startswith("+0") for row in envelopes)
    assert all(
        row["authority"] == "compatibility"
        for row in envelopes
        if row["id"] != "deployment-curve-v1-run3"
    )


def test_all_root_consumer_pins_are_byte_identical() -> None:
    frozen = _frozen()
    pins = frozen["consumer_pins"]

    assert len(pins) == frozen["preservation_lock_rule"]["root_pin_count"] == 22
    assert len({row["path"] for row in pins}) == 22
    for pin in pins:
        payload = (ROOT / pin["path"]).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == pin["sha256"]


def test_freeze_and_report_language_are_portable() -> None:
    for path in (EXPECTATIONS, STUDY / "expectations.md"):
        payload = path.read_bytes()
        text = payload.decode("utf-8")
        assert b"\r" not in payload
        assert "\N{EM DASH}" not in text
        assert "/data3/" not in text
        assert "/home/" not in text
        assert "+/-" not in text
