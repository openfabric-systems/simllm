import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "examples" / "nvlink_credit_arbitration_v1"


def _expectations() -> dict[str, object]:
    return json.loads((STUDY / "expectations.json").read_text(encoding="utf-8"))


def test_freeze_registers_physical_credit_ownership_without_measurement_claim():
    frozen = _expectations()
    background = frozen["architecture_background"]

    assert frozen["task_id"] == "TRAF-73"
    assert background["credit_ownership"] == (
        "hard_allocated_per_physical_link_per_link_layer_virtual_channel"
    )
    assert background["evidence_class"] == (
        "public_architecture_background_not_our_measurement"
    )
    assert background["modeled_virtual_channels"] == 1
    assert {source["kind"] for source in background["sources"]} == {
        "vendor_overview",
        "vendor_technical_overview",
        "encyclopedic_description",
    }


def test_freeze_keeps_candidate_numbers_declared_and_physical_arithmetic_exact():
    frozen = _expectations()
    candidate = frozen["candidate"]
    sanity = frozen["physical_sanity"]

    assert candidate["credits_per_link_per_modeled_virtual_channel"] == 256
    assert candidate["credit_unit_bytes"] == 272
    assert candidate["links_per_pair"] == 4
    assert candidate["credit_return_latency_ps"] == 200_000
    assert candidate["arbitration_default"] == {
        "evidence_class": "declared_default_candidate_not_hardware_measurement",
        "value": "release_aware_round_robin",
    }
    assert 4 * 256 * 272 == 278_528
    assert 4 * 256 * candidate["payload_bytes_per_packet"] == 262_144
    assert sanity["per_link_packet_serialization_ps"] == 10_880
    assert sanity["bond_packet_cadence_ps"] == 2_720
    assert sanity["candidate_per_link_window_serialization_ps"] == 2_785_280
    assert sanity["candidate_per_link_window_serialization_ps"] > sanity[
        "candidate_credit_return_ps"
    ]


def test_freeze_registers_exact_hardware_and_simulation_cells():
    frozen = _expectations()
    hardware = frozen["hardware_cells"]
    simulation = frozen["simulation"]

    assert hardware["credit_window"]["directed_pairs"] == "all_12_nv4_directed_pairs"
    assert 262_144 in hardware["credit_window"]["payload_sizes_bytes"]
    assert hardware["pool_scope"]["source_sets"] == [[0], [0, 1], [0, 1, 2]]
    assert hardware["sustained_arbitration"]["offered_rates_bytes_per_second"] == [
        100_000_000_000,
        60_000_000_000,
        60_000_000_000,
    ]
    assert simulation["degrees"] == [2, 3, 4, 8, 16]
    assert simulation["extrapolated_degrees"] == [4, 8, 16]
    assert simulation["policies"] == [
        "release_aware_round_robin",
        "static_interleave",
        "greedy_capture",
    ]


def test_freeze_locks_every_merged_nvlink_study_tree():
    lock = _expectations()["preservation_lock"]

    assert lock["tracked_file_count"] == 89
    assert lock["tracked_bytes"] == 6_429_838
    assert lock["path_content_digest_sha256"] == (
        "61af15faf7c7080f40a33f8f9d5503b3b0278f15be15997e90c6895cddf85c72"
    )
    assert lock["candidate_profile_sha256"] == (
        "d33ef5b2c6fa87cc97e1e7b45a43a841a5da45f5462311e3981fbc903c56deb2"
    )


def test_freeze_text_uses_plain_punctuation_and_marks_extrapolation():
    text = (STUDY / "expectations.md").read_text(encoding="utf-8")

    assert "SIMULATED MESH EXTRAPOLATION" in text
    assert "not our measurement" in text
    assert "No knee through 8 MiB" in text
    assert "\N{EM DASH}" not in text
