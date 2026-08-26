import csv
import hashlib
import json
import struct
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "examples" / "deployment_frontier_v1"


def _result():
    return json.loads((STUDY / "result.json").read_text(encoding="utf-8"))


def test_result_publishes_zero_residual_gate_and_honest_refutation():
    result = _result()
    points = result["points"]

    assert result["schema"] == "simllm-deployment-frontier-result-v1"
    assert result["status"] == "REFUTED"
    assert len(points) == 18
    assert {point["accounting"]["residual_ps"] for point in points} == {0}
    for point in points:
        accounting = point["accounting"]
        assert accounting["simulated_step_ps"] == (
            accounting["analytical_step_ps"]
            + accounting["inter_node_attributed_ps"]
            + accounting["intra_node_attributed_ps"]
        )
        assert accounting["inter_node_attributed_ps"] >= 0
        assert accounting["intra_node_attributed_ps"] >= 0
        for axis in (
            "x_tokens_per_second_per_request",
            "y_tokens_per_second_per_gpu",
        ):
            assert (
                point["simulated_operating_point"][axis]["decimal"]
                <= point["analytical_operating_point"][axis]["decimal"]
            )


def test_bottleneck_map_retains_the_frozen_miss_and_raw_incast():
    result = _result()
    points = result["points"]

    assert Counter(
        point["bottleneck"]["classification"] for point in points
    ) == Counter({"neither": 17, "intra-node": 1})
    misses = [
        row["name"] for row in result["expected_direction_checks"] if not row["passed"]
    ]
    assert misses == ["nine-node incast has positive elapsed inter-node attribution"]
    nine_node_b32 = next(
        point
        for point in points
        if point["configuration_id"] == "h100-nine-node-incast"
        and point["batch_per_gpu"] == 32
    )
    assert nine_node_b32["fabric_attribution"]["dominant_mechanism"] == "incast"
    assert nine_node_b32["fabric_attribution"]["raw_excess_ps"] == 6_743_004_420
    assert nine_node_b32["fabric_observation"]["concurrent_service_ps"] == 7_689_053_000
    assert nine_node_b32["kernel"]["kernel_floor_ps"] == 9_535_537_623
    assert nine_node_b32["accounting"]["inter_node_attributed_ps"] == 0


def test_candidate_disclosure_and_module_evidence_are_complete():
    result = _result()

    assert "A100 NVLink3 candidate" in result["intra_node_candidate_disclosure"]
    assert "not promoted" in result["intra_node_candidate_disclosure"]
    assert all(
        point["intra_node_evidence"]["status"] == "candidate"
        and point["intra_node_attribution"]["dominant_module"]
        == "TX credits and packetization"
        and point["intra_node_attribution"]["switch_contention_ps"] == 0
        for point in result["points"]
    )
    assert result["preservation_lock"]["artifacts_checked"] == 43
    assert result["preservation_lock"]["all_byte_identical"] is True


def test_publication_figure_hashes_dimensions_and_csv_rows_match():
    result = _result()
    figures = result["publication"]["figures"]
    expected_dimensions = {
        "figures/deployment-frontier.png": (1260, 779),
        "figures/two-network-bottleneck.png": (1260, 1080),
    }

    assert len(figures) == 4
    for artifact in figures:
        path = STUDY / artifact["path"]
        payload = path.read_bytes()
        assert len(payload) == artifact["bytes"]
        assert hashlib.sha256(payload).hexdigest() == artifact["sha256"]
        if path.suffix == ".png":
            assert payload[:8] == b"\x89PNG\r\n\x1a\n"
            assert struct.unpack(">II", payload[16:24]) == expected_dimensions[
                artifact["path"]
            ]
        else:
            assert payload.startswith(b"%PDF-")

    with (STUDY / "results.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 18
    assert {int(row["residual_ps"]) for row in rows} == {0}
