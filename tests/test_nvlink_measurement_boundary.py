"""Reject confounded timing inferences without rewriting captured evidence."""

import importlib.util
import json
from fractions import Fraction
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "examples/nvlink_measurement_boundary_v1"


@pytest.fixture
def audit():
    spec = importlib.util.spec_from_file_location(
        "measurement_boundary_test", HERE / "run_study.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fixed_launch_work_triggers_packet_label_without_any_packet_mechanism(audit):
    pair = audit.packet_free_pair(4 * 1024**2, 100_000_000_000, 5_000_000)
    assert pair[0]["duration_ps"] == Fraction(46_943_040)
    assert pair[1]["duration_ps"] == Fraction(88_886_080)
    assert pair[0]["error"] - pair[1]["error"] == Fraction(390625, 6553600)
    labels = audit.load_scorer().attribute_misses(audit.classifier_rows(pair, [1, 2, 3]))
    assert set(labels.values()) == {"packetization"}


def test_zero_overhead_is_an_identity_control(audit):
    pair = audit.packet_free_pair(4 * 1024**2, 100_000_000_000, 0)
    assert all(row["duration_ps"] == row["floor_ps"] and row["error"] == 0 for row in pair)
    labels = audit.load_scorer().attribute_misses(audit.classifier_rows(pair, [1, 2, 3]))
    assert "packetization" not in labels.values()


def test_equal_local_durations_do_not_identify_common_phase(audit):
    aligned = audit.phase_case(3, 100_000_000, 0)
    staggered = audit.phase_case(3, 100_000_000, 20_000_000)
    assert aligned["maximum_local_duration_ps"] == staggered["maximum_local_duration_ps"]
    assert aligned["common_phase_ps"] == 100_000_000
    assert staggered["common_phase_ps"] == 140_000_000
    assert staggered["goodput_overstatement"]["exact"] == "7/5"


@pytest.mark.parametrize("size,rate,overhead", [(0, 1, 0), (1, 0, 0), (1, 1, -1)])
def test_invalid_counterexample_coordinates_are_rejected(audit, size, rate, overhead):
    with pytest.raises(ValueError):
        audit.packet_free_pair(size, rate, overhead)


def test_history_has_a_budget_fraction_and_no_observed_start_interval(audit):
    frozen = json.loads((HERE / "expectations.json").read_text())
    result, findings = audit.historical_audit(frozen)
    assert findings == []
    assert result["unchanged_capture_row_count"] == 42
    assert result["hardware_behavioral_score"] is None
    assert result["qualification_verdict"] == "VOID_UNDECIDABLE_ALIGNMENT_PRECONDITION"
    assert all(row["source_start_observation"] is None for row in result["budget_reconstructions"])


def test_four_card_request_keeps_reservation_and_measurements_absent():
    request = json.loads((HERE / "measurement-request.json").read_text())
    assert request["reservation_status"] == "awaiting-reservation"
    assert request["attempts"] == []
    assert request["measurements"] == []
    assert request["hardware"]["gpu_count"] == 4
    assert request["capture_ready"] is False
    assert request["required_pre_capture_freeze"]
    for stage in request["stages"]:
        assert stage["owner"] in {"TRAF-73", "TRAF-86"}
        assert stage["discriminates"]
        assert stage["required_observations"]
        assert stage["rejection_conditions"]
    assert len({stage["id"] for stage in request["stages"]}) == len(request["stages"])
    assert "common_clock_mapping_and_uncertainty" in request["required_observations"]


def test_preservation_failure_voids_audit_before_historical_code_load(audit, monkeypatch, tmp_path):
    original_git = audit.git

    def source_for_local_test(*arguments):
        if arguments[0] == "show" and arguments[1].startswith("HEAD:"):
            return (HERE / "run_study.py").read_bytes()
        return original_git(*arguments)

    monkeypatch.setattr(audit, "git", source_for_local_test)
    monkeypatch.setattr(audit, "digest", lambda _: "0" * 64)
    monkeypatch.setattr(audit, "load_scorer", lambda: pytest.fail("must not load changed source"))
    output = tmp_path / "changed-history"
    result = audit.run_study(output)
    assert result["verdict"] == "VOID"
    assert result["behavioral_score"] is None
    assert result["packet_free_configuration_count"] == 0
    assert (output / "summary.json").is_file()
