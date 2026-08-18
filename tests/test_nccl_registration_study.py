"""Lock the nccl_registration_v1 results against their freeze.

These checks read the committed artifacts only. They never run the study, need
no backend binary, and assert exactly the relations the freeze named, so a
later change that silently moves a published number fails here.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

STUDY = Path(__file__).resolve().parents[1] / "examples" / "nccl_registration_v1"
RESULTS_JSON = STUDY / "results.json"
EXPECTATIONS_JSON = STUDY / "expectations.json"
RESULTS_MD = STUDY / "RESULTS.md"

COST_PS = 20_000_000


@pytest.fixture(scope="module")
def results() -> dict:
    return json.loads(RESULTS_JSON.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def freeze() -> dict:
    return json.loads(EXPECTATIONS_JSON.read_text(encoding="utf-8"))


def test_the_run_is_interpretable_and_every_fatal_guard_held(results: dict) -> None:
    guards = results["fatal_guards"]
    assert [guard["id"] for guard in guards] == [f"G{index}" for index in range(1, 10)]
    assert all(guard["held"] for guard in guards)
    assert results["verdict"] == "interpretable"


def test_the_identity_block_equals_the_freeze(results: dict, freeze: dict) -> None:
    assert results["identity"] == results["frozen_identity"]
    assert results["identity"]["registration_cost_ps"] == (
        freeze["declared"]["registration_cost_ps"]
    )
    assert results["identity"]["evidence_class"] == "declared"


def test_the_exact_oracle_rows_all_passed_at_their_frozen_values(
    results: dict,
    freeze: dict,
) -> None:
    scored = results["scored"]["exact_oracle_rows"]
    frozen = {row["id"]: row for row in freeze["scored"]["exact_oracle_rows"]["rows"]}
    assert scored["denominator"] == freeze["scored"]["exact_oracle_rows"]["denominator"]
    assert scored["passed"] == scored["denominator"]
    for row in scored["rows"]:
        assert row["passed"], row["id"]
        if "value_ps" in frozen[row["id"]] and row["id"] != "O2":
            assert row["expected_ps"] == frozen[row["id"]]["value_ps"], row["id"]


def test_the_behavioral_families_all_passed_over_their_frozen_instances(
    results: dict,
    freeze: dict,
) -> None:
    scored = results["scored"]["behavioral_families"]
    frozen = freeze["scored"]["behavioral_families"]
    assert scored["denominator_families"] == frozen["denominator_families"]
    assert scored["denominator_instances"] == frozen["denominator_instances"]
    assert scored["passed_families"] == scored["denominator_families"]


def test_the_two_evidence_classes_are_never_summed(results: dict) -> None:
    scored = results["scored"]
    assert set(scored) == {"exact_oracle_rows", "behavioral_families"}
    assert "total" not in scored
    text = RESULTS_MD.read_text(encoding="utf-8")
    assert "6 of 6" in text
    assert "3 of 3 over 7 instances" in text


def test_the_registration_charge_is_the_declared_cost_times_the_identities(
    results: dict,
) -> None:
    local = results["cells"]["local-tp2"]
    mixed = results["cells"]["mixed-tp4"]
    assert local["identities_one_channel"] == 64
    assert mixed["identities_one_channel"] == 4
    assert local["arms"]["on"]["ledger_charged_ps"] == 64 * COST_PS
    assert local["arms"]["on-2ch"]["ledger_charged_ps"] == 128 * COST_PS
    assert mixed["arms"]["on"]["ledger_charged_ps"] == 4 * COST_PS


def test_ttft_moved_by_exactly_the_registered_amount(results: dict) -> None:
    local = results["cells"]["local-tp2"]
    mixed = results["cells"]["mixed-tp4"]
    assert (
        local["arms"]["on"]["ttft_ps"] - local["arms"]["off"]["ttft_ps"]
        == 64 * COST_PS
    )
    assert (
        local["arms"]["on-2ch"]["ttft_ps"] - local["arms"]["off"]["ttft_ps"]
        == 128 * COST_PS
    )
    assert (
        mixed["arms"]["on"]["ttft_ps"] - mixed["arms"]["off"]["ttft_ps"]
        == 4 * COST_PS
    )
    assert local["arms"]["on-rebuild"]["ttft_ps"] == local["arms"]["on"]["ttft_ps"]


def test_the_off_arm_changed_no_emitted_byte_and_no_later_step(results: dict) -> None:
    for cell in results["cells"].values():
        off = cell["arms"]["off"]
        off_makespans = [step["makespan_ps"] for step in off["steps"]]
        for arm, payload in cell["arms"].items():
            assert payload["goal_sha256"] == off["goal_sha256"], arm
            if arm == "on-rebuild":
                continue
            tail = [step["makespan_ps"] for step in payload["steps"]][1:]
            assert tail == off_makespans[1:], arm
        assert off["ledger_charged_ps"] == 0
        assert cell["default_construction"]["goal_sha256"] == off["goal_sha256"]
        assert cell["default_construction"]["registration_outcomes"] == 0
        assert cell["default_construction"]["registration_phase_projection_empty"]


def test_a_phase_split_collective_paid_exactly_once(results: dict) -> None:
    step = results["cells"]["mixed-tp4"]["arms"]["on"]["steps"][0]
    charges = [value for value in step["registration_phase_cost_ps"] if value]
    assert step["artifact_count"] == 26
    assert step["backend_runs"] == 24
    assert charges == [COST_PS] * 4
    assert step["charged_registration_ps"] == 4 * COST_PS


def test_every_step_conserved_its_partition(results: dict) -> None:
    for cell in results["cells"].values():
        for payload in cell["arms"].values():
            for step in payload["steps"]:
                assert step["media"]["total_ps"] == step["makespan_ps"]
                assert (
                    step["media"]["collective_registration_ps"]
                    == step["charged_registration_ps"]
                )


def test_the_mirrored_seam_off_path_and_gate_are_recorded(results: dict) -> None:
    seam = results["seam"]
    assert seam["ungated_stream_identical"]
    assert seam["ungated_event_count"] == 394
    assert seam["gated_registration_events"] == 2
    assert seam["gated_declared_total_ps"] == COST_PS
    assert seam["clock_unmoved"]
    assert seam["unregistered_buffer_refused"]


def test_the_calibration_hook_failed_closed(results: dict) -> None:
    fail_closed = results["fail_closed"]
    assert fail_closed["calibrated_request_raises"]
    assert fail_closed["unknown_selector_raises"]
    assert "not calibrated" in fail_closed["calibrated_message"]
    assert "TRAF-56" in fail_closed["calibrated_message"]


def test_the_report_states_what_the_run_does_not_establish() -> None:
    text = RESULTS_MD.read_text(encoding="utf-8")
    assert "What this run does not establish" in text
    assert "The cost is not calibrated" in text
    assert "No packetized handshake" in text
    assert "No live new-peer re-registration" in text
    assert "—" not in text
