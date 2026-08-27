from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

import pytest

from examples.frontier_ladder_v1 import run_study as study
from simllm.backends import build_loggopsim_command, parse_loggopsim_stdout
from simllm.deploy import (
    FrontierRung,
    frontier_ladder_record_from_json,
    frontier_ladder_record_to_json,
)
from simllm.goal import find_txt2bin

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
STUDY_DIR = REPOSITORY_ROOT / "examples" / "frontier_ladder_v1"
FROZEN_EXPECTATIONS_SHA256 = (
    "e3e83264df6e72e83736a06dddcba11a501c75a25c8c1fb0a9c7b1e9c0caeea3"
)
PINNED_BINARY_SHA256 = (
    "7e0f13ee3c87a20e9d2e94dbbd74c46075fd03df2f1b04d1ed9739c43ee0a2bf"
)
ARTIFACT_SHA256 = {
    "RESULTS.md": "27dfb96ec392aea3f3211d440fb953622296b75fa69f738fcc55495496608c48",
    "figures/frontier-ladder.pdf": "d681868fb34b3282ddfc70f800c99dd8d81cf901cac81109659dd8f1374a6eea",
    "figures/frontier-ladder.png": "16bb26584e17c4c6cc146f42bca210b2df0654492f55deacd2951d2d5179e910",
    "result.json": "6b399398917b751344a307ab5ed7c59cfcbdb383621720eb0f0ae293df979d6c",
    "results.csv": "6e682f19ad7c828e0fa167c323def1cf8a62f5b98bfb7ea565fb3ca37d237f6d",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("relative, expected", sorted(ARTIFACT_SHA256.items()))
def test_frontier_ladder_publication_bytes(relative: str, expected: str) -> None:
    assert _sha256(STUDY_DIR / relative) == expected


def test_frontier_ladder_expectations_bytes_are_unchanged() -> None:
    assert _sha256(STUDY_DIR / "expectations.md") == FROZEN_EXPECTATIONS_SHA256


def test_frontier_ladder_record_keeps_scores_guards_and_authorities_separate() -> None:
    result_path = STUDY_DIR / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))

    assert result["verdict"] == "PASS"
    assert result["findings"] == []
    assert result["chronology"]["implementation_commit"] == (
        "b66e621a395af0e614825d2b1268e7efc13d6956"
    )
    exact = result["score_classes"]["exact_oracles"]["families"]
    assert (exact["L-A"]["passed"], exact["L-A"]["denominator"]) == (6, 6)
    assert (exact["L-B"]["passed"], exact["L-B"]["denominator"]) == (6, 6)
    relations = result["score_classes"]["behavioral_relations"]["families"]
    for family, denominator in (("M-1", 6), ("M-2", 6), ("M-3", 6), ("S", 18)):
        assert (relations[family]["passed"], relations[family]["denominator"]) == (
            denominator,
            denominator,
        )
    plot = result["score_classes"]["plot_contract"]
    wall = result["score_classes"]["wall_time"]
    assert (plot["passed"], plot["denominator"]) == (4, 4)
    assert (wall["passed"], wall["denominator"]) == (1, 1)
    assert wall["packet_executions"] == 0
    assert len(wall["samples_seconds"]) == 7
    corrections = result["post_specified_corrections"]
    assert corrections["withdrawn_closure"] == "TRAF-20"
    assert corrections["preserved_closure"] == "TRAF-68"
    assert corrections["original_published_wall_median_seconds"] == 0.034612214
    assert corrections["modeled_quantity_reproduction"]["matched"] is True
    assert corrections["enforcement_acceptance_study"] == (
        "not authored or run; integrator-owned"
    )

    guards = result["fatal_guards"]
    assert [guard["id"] for guard in guards] == ["FG-1", "FG-2", "FG-3", "FG-4"]
    assert all(guard["held"] for guard in guards)
    assert all(
        guard["mutation_control"]["kind"] == "predicate-exercised"
        and guard["mutation_control"]["rejected"]
        for guard in guards
    )

    record = frontier_ladder_record_from_json(result["ladder_record"])
    assert len(record.points) == 18
    corrected_record = frontier_ladder_record_to_json(record)
    assert corrected_record == result["ladder_record"]
    assert plot["point_class_tallies"] == {
        "ESTIMATE": 24,
        "MEASURED": 0,
        "SIMULATED": 30,
    }
    for point in record.points:
        assert [rung.rung for rung in point.rungs] == list(FrontierRung)
        ideal = point.rung(FrontierRung.LOGGOPSIM_IDEAL)
        packet = point.rung(FrontierRung.PACKET)
        if ideal.fabric_leg_ps:
            assert ideal.point_class.value == "SIMULATED"
            assert ideal.provenance.authority == "loggopsim-ideal"
        else:
            assert ideal.point_class.value == "ESTIMATE"
            assert ideal.provenance.authority == "closed-form"
            assert ideal.provenance.binary_sha256 is None
        assert packet.provenance.authority == "rnic-nn"
        assert ideal.provenance != packet.provenance
        if ideal.fabric_leg_ps:
            argv = list(ideal.provenance.argv)
            assert argv[0] == "LogGOPSim"
            assert argv[argv.index("-G") + 1] == "0.02"
            assert not Path(argv[argv.index("-f") + 1]).is_absolute()

    exact = result["score_classes"]["exact_oracles"]["families"]
    assert all(
        not row["fan_in"]["fan_in_detected"]
        and not row["fan_in"]["acknowledged"]
        for row in exact["L-A"]["rows"]
    )
    assert all(
        row["fan_in"]["fan_in_detected"] and row["fan_in"]["acknowledged"]
        for row in exact["L-B"]["rows"]
    )
    schedule = exact["L-B"]["rows"][-1]["schedule_comparison"]
    assert schedule["pinned_packet_flow_payload_bytes"] == [
        47_302_429,
        47_302_429,
        47_302_429,
        47_302_429,
        47_302_428,
        47_302_428,
        47_302_428,
        47_302_428,
    ]
    assert schedule["ideal_flow_payload_bytes"] == [47_302_429] * 8
    assert schedule["flow_sets_equal"] is False

    curves = {
        curve["configuration_id"]: curve
        for curve in plot["plot_data"]["curves"]
    }
    two_node = curves["h100-two-node-serialized"]
    nine_node = curves["h100-nine-node-incast"]
    assert two_node["line_style"] == "--"
    assert two_node["line_width"] < nine_node["line_width"]
    assert [point["rungs"]["packet"]["x"] for point in two_node["points"]] == [
        point["rungs"]["packet"]["x"] for point in nine_node["points"]
    ]
    assert [point["rungs"]["packet"]["y"] for point in two_node["points"]] == [
        point["rungs"]["packet"]["y"] for point in nine_node["points"]
    ]

    serialized = json.dumps(result)
    assert "/data3/" not in serialized
    assert "/home/" not in serialized
    assert b"\r" not in result_path.read_bytes()


def test_fg1_mutant_changes_only_the_observed_digest() -> None:
    rows = [
        {
            "expected_sha256": "1" * 64,
            "observed_sha256": "1" * 64,
            "matched": True,
        }
    ]
    mutant = [dict(rows[0])]
    mutant[0]["observed_sha256"] = "0" * 64

    assert study._fg1_holds(rows)
    assert not study._fg1_holds(mutant)
    assert mutant[0]["matched"] is True


def test_native_l_a_batch_1_cell_when_pinned_binary_is_available(tmp_path: Path) -> None:
    configured_binary = os.environ.get("SIMLLM_LOGGOPSIM")
    if not configured_binary:
        pytest.skip("SIMLLM_LOGGOPSIM is unset; CI does not install pinned LogGOPSim")
    binary = Path(configured_binary)
    if not binary.is_file():
        pytest.skip(f"SIMLLM_LOGGOPSIM is not an available file: {binary}")
    assert _sha256(binary) == PINNED_BINARY_SHA256

    txt2bin = find_txt2bin()
    if txt2bin is None:
        pytest.skip("SIMLLM_TXT2BIN is unavailable; the native cell needs txt2bin")
    cell = study._cells()[0]
    assert cell.cell_id == "L-A-b1"
    goal, binary_goal, fan_in = study._render_goal(cell, tmp_path, txt2bin)
    argv = build_loggopsim_command(binary, study._config(cell, binary_goal))

    completed = subprocess.run(
        argv,
        capture_output=True,
        check=False,
        timeout=60,
    )

    assert completed.returncode == 0, completed.stderr.decode(errors="replace")
    parsed = parse_loggopsim_stdout(completed.stdout.decode("utf-8"))
    assert parsed.quiescent
    assert parsed.max_finish_ps == 135_038_000
    assert argv[argv.index("-G") + 1] == "0.02"
    assert fan_in.fan_in_detected is False
    assert fan_in.acknowledged is False
    rendered = goal.read_text(encoding="utf-8")
    assert rendered.count("send 6651904b to 0") == 1
    assert rendered.count("recv 6651904b from 1") == 1
