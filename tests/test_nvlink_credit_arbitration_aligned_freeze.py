import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "examples" / "nvlink_credit_arbitration_v1"


@pytest.fixture(scope="module")
def frozen() -> dict[str, object]:
    return json.loads(
        (STUDY / "aligned_expectations.json").read_text(encoding="utf-8")
    )


def test_freeze_precedes_aligned_simulation_and_hardware(
    frozen: dict[str, object],
) -> None:
    study = frozen["study"]
    assert isinstance(study, dict)
    assert study["task_id"] == "TRAF-73"
    assert study["status"] == "EXPECTATIONS_ONLY"
    assert study["hardware_executed"] is False
    assert study["base_commit"] == "4073b4e762d2a209e8f9f642e360054290d41465"
    assert "before the producer extension" in study["chronology"]
    assert "unobservable H2 23-size subset" in study["supersedes"]
    assert "pre-run correction" in study["supersedes"]


def test_aligned_candidate_arithmetic_is_internal_and_exact(
    frozen: dict[str, object],
) -> None:
    authority = frozen["aligned_authority"]
    candidate = frozen["candidate_set"]["effective_window"]
    physical = frozen["physical_sanity"]

    assert authority["implementation"] == "simllm-htsim-nvlink-domain-v2"
    assert authority["packet"] == {
        "credit_units_per_maximum_packet": 1,
        "flit_bytes": 16,
        "header_flits": 1,
        "maximum_payload_bytes": 256,
        "maximum_payload_flits": 16,
        "maximum_wire_bytes": 272,
        "maximum_wire_flits": 17,
        "virtual_channel": "vc0",
    }
    assert candidate == {
        "payload_bytes": 262_144,
        "return_latency_ps": 200_000,
        "status": "DECLARED_CANDIDATE",
        "wire_bytes": 278_528,
    }
    assert physical["per_link_window_serialization_ps"] == 2_785_280
    assert physical["declared_return_latency_ps"] < physical[
        "per_link_window_serialization_ps"
    ]
    assert physical["return_to_window_serialization_ratio"] == pytest.approx(
        200_000 / 2_785_280
    )


def test_h1_freezes_all_pairs_sizes_and_honest_no_break_case(
    frozen: dict[str, object],
) -> None:
    h1 = frozen["h1_credit_window_and_return"]
    assert len(h1["directed_pairs"]) == 12
    assert len({tuple(pair) for pair in h1["directed_pairs"]}) == 12
    assert len(h1["payload_sizes_bytes"]) == 31
    assert h1["payload_sizes_bytes"][0] == 4096
    assert h1["payload_sizes_bytes"][-1] == 8 * 1024 * 1024
    assert h1["configuration_count"] == 372
    assert h1["timed_sample_count"] == 74_400
    prediction = h1["aligned_candidate_prediction"]
    assert prediction["outcome"] == "NO_BREAK_THROUGH_8_MIB"
    assert prediction["interpretation"].startswith("INCONCLUSIVE")
    assert "does not confirm" in prediction["interpretation"]


def test_h2_freezes_aggregate_outstanding_discriminator(
    frozen: dict[str, object],
) -> None:
    h1 = frozen["h1_credit_window_and_return"]
    h2 = frozen["h2_pool_scope"]
    assert h2["payload_sizes_bytes"] == h1["payload_sizes_bytes"]
    assert h2["source_sets"] == [[0], [0, 1], [0, 1, 2]]
    rows = h2["aggregate_outstanding_discriminator"]
    assert [row["sender_count"] for row in rows] == [1, 2, 3]
    assert [
        row["per_link_pool"]["ideal_aggregate_outstanding_payload_bytes"]
        for row in rows
    ] == [262_144, 524_288, 786_432]
    assert {
        row["shared_destination_pool"][
            "ideal_aggregate_outstanding_payload_bytes"
        ]
        for row in rows
    } == {262_144}
    assert rows[1]["shared_destination_pool"][
        "ideal_per_sender_knee_payload_bytes"
    ] == 131_072
    assert rows[2]["shared_destination_pool"][
        "ideal_per_sender_knee_payload_bytes"
    ] == pytest.approx(262_144 / 3)
    assert rows[2]["shared_destination_pool"][
        "registered_sweep_bracket_bytes"
    ] == [65_536, 131_072]


def _matrix_row(
    frozen: dict[str, object], degree: int, policy: str
) -> dict[str, object]:
    rows = frozen["h3_arbitration"]["aligned_policy_predictions"]
    return next(
        row
        for row in rows
        if row["degree"] == degree and row["policy"] == policy
    )


def test_h3_freezes_each_aligned_policy_direction(
    frozen: dict[str, object],
) -> None:
    h3 = frozen["h3_arbitration"]
    assert h3["offered_raw_bytes_per_second_by_role"] == {
        "greedy": 100_000_000_000,
        "small": 60_000_000_000,
    }
    assert h3["warmup_ms"] == 50
    assert h3["measurement_ms"] == 500
    assert h3["drain_ms"] == 50

    fair = _matrix_row(frozen, 3, "release_aware_round_robin")
    static = _matrix_row(frozen, 3, "static_interleave")
    greedy = _matrix_row(frozen, 3, "greedy_capture")
    assert fair["expected_raw_gbps_per_source"] == pytest.approx(
        [87.101921876, 60.0, 60.0]
    )
    assert static["expected_raw_gbps_per_source"] == pytest.approx(
        [60.0, 60.0, 60.0]
    )
    assert greedy["expected_raw_gbps_per_source"] == pytest.approx(
        [100.0, 53.550960938, 53.550960938]
    )
    assert fair["expected_aggregate_raw_gbps"] == pytest.approx(207.101921876)
    assert static["expected_aggregate_raw_gbps"] == pytest.approx(180.0)
    assert greedy["expected_aggregate_raw_gbps"] == pytest.approx(207.101921876)


def test_extrapolated_policy_rows_are_labeled_without_hardware_claim(
    frozen: dict[str, object],
) -> None:
    rows = frozen["h3_arbitration"]["aligned_policy_predictions"]
    assert len(rows) == 15
    assert {
        row["degree"]
        for row in rows
        if row["topology_class"] == "SIMULATED_MESH_EXTRAPOLATION"
    } == {4, 8, 16}
    assert {
        row["degree"]
        for row in rows
        if row["topology_class"] == "PHYSICAL_NV4"
    } == {2, 3}


def test_freeze_retains_every_candidate_and_separate_promotion(
    frozen: dict[str, object],
) -> None:
    candidate = frozen["candidate_set"]
    assert candidate["pool_scope"] == [
        "per_link_destination_virtual_channel",
        "shared_destination_virtual_channel",
    ]
    assert candidate["arbitration"] == [
        "release_aware_round_robin",
        "static_interleave",
        "greedy_capture",
    ]
    assert "unseparated" in candidate["retention_rule"]
    promotion = frozen["promotion_rule"]
    assert promotion["task_id"] == "TRAF-85"
    assert promotion["free_on_base_commit"] is True
    assert "Do not edit the module or profile" in promotion["scope"]


def test_freeze_keeps_live_tree_hashes_out_of_tests_and_ip_stats_out_of_evidence(
    frozen: dict[str, object],
) -> None:
    preservation = frozen["preservation"]
    assert "do not pin current live-tree hashes" in preservation["test_rule"]
    assert len(preservation["recorded_artifacts"]) == 6
    assert all(len(row["sha256"]) == 64 for row in preservation["recorded_artifacts"])
    producer = frozen["producer_contract"]
    assert producer["lineage"] == "corrected_TRAF_70_nvlink_packet_lane"
    assert "Do not create another CUDA capture harness" in producer["reuse_rule"]
    assert producer["forbidden_evidence"] == (
        "ip_link_stats64_is_not_an_nvlink_wire_authority"
    )


def test_freeze_has_no_survivable_fatal_guard_or_plain_text_style_violation(
    frozen: dict[str, object],
) -> None:
    assert len(frozen["fatal_guards"]) == 17
    assert frozen["void_rule"].startswith("Any fatal-guard violation")
    markdown = (STUDY / "aligned_expectations.md").read_text(encoding="utf-8")
    assert "+/-" not in markdown
    assert "\N{EM DASH}" not in markdown
    assert "No cluster time has been requested" in markdown
