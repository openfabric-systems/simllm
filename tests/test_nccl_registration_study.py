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
EM_DASH = "\u2014"


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


def test_every_published_guard_carries_its_frozen_claim_verbatim(
    results: dict,
    freeze: dict,
) -> None:
    """The one assertion that stops a weaker check being published as a guard.

    A results guard identity may only appear beside the exact claim string the
    freeze wrote for it. What the runner actually checked belongs in the
    separate ``evaluated`` field, which must be present and nonblank.
    """

    frozen = {guard["id"]: guard["claim"] for guard in freeze["fatal_guards"]}
    for guard in results["fatal_guards"]:
        assert guard["claim"] == frozen[guard["id"]], guard["id"]
        assert guard["evaluated"].strip(), guard["id"]
        if guard["partially_evaluated"]:
            assert guard["remainder_covered_by"].strip(), guard["id"]


def test_no_frozen_guard_is_partially_evaluated(results: dict) -> None:
    assert not any(
        guard["partially_evaluated"] for guard in results["fatal_guards"]
    )


def test_the_off_arm_matches_the_feature_absent_arm_field_by_field(
    results: dict,
) -> None:
    """G3 as frozen: the exact per-artifact field set, not a weaker summary."""

    fields = (
        "makespan_ps",
        "completed_at_ps",
        "fabric_phase_service_ps",
        "local_phase_service_ps",
        "base_phase_latency_ps",
        "composed_phase_service_ps",
        "local_phase_medium",
        "media",
    )
    for name, cell in results["cells"].items():
        off_steps = cell["arms"]["off"]["steps"]
        absent_steps = cell["default_construction"]["steps"]
        assert len(off_steps) == len(absent_steps), name
        for off_step, absent_step in zip(off_steps, absent_steps, strict=True):
            for field in fields:
                assert off_step[field] == absent_step[field], (name, field)


def test_the_identity_geometry_comes_from_the_executed_sinks(results: dict) -> None:
    """G8 is only meaningful if the two sides are derived independently."""

    for name, cell in results["cells"].items():
        executed = cell["arms"]["off"]["executed_geometry"]
        published = results["identity"]["cells"][name]
        assert published["dims"] == executed["dims"]
        assert published["tp_ranks"] == executed["tp_ranks"]
        assert published["hosts"] == executed["hosts"]
        assert published["steps"] == executed["steps"]


def test_the_accepted_nccl_stack_sequences_were_reproduced(results: dict) -> None:
    seam = results["seam"]
    assert seam["accepted_sequences_reproduced"]
    assert seam["accepted_sequence_count"] == 8
    for row in seam["accepted_sequence_rows"]:
        assert row["matched"], row["check"]
        assert row["observed"] == row["expected"]


def test_the_request_partitions_conserve_their_spans(results: dict) -> None:
    for cell in results["cells"].values():
        for arm, payload in cell["arms"].items():
            assert payload["ttft_partition_conserves"], arm
            assert payload["decode_partition_conserves"], arm


def test_the_counterfactual_arithmetic_is_published_per_cell(results: dict) -> None:
    rows = {row["cell"]: row for row in results["counterfactual"]["rows"]}
    assert rows["mixed-tp4"]["observed_delta_ps"] == 80_000_000
    assert rows["mixed-tp4"]["folded_into_max_delta_ps"] == 69_378_560
    assert rows["mixed-tp4"]["per_artifact_realized_service_ps"] == [2_655_360]
    assert not rows["mixed-tp4"]["fully_hidden_under_fold"]
    assert rows["local-tp2"]["folded_into_max_delta_ps"] == 981_696_000
    slow = rows["slow-fabric-tp4"]
    assert slow["provenance"] == "post-specified"
    assert slow["observed_delta_ps"] == 80_000_000
    assert slow["folded_into_max_delta_ps"] == 0
    assert slow["fully_hidden_under_fold"]


def test_the_post_specified_cell_is_labeled_and_discriminating(results: dict) -> None:
    cell = results["post_specified_cells"]["slow-fabric-tp4"]
    assert cell["linkspeed_bps"] == 10_000_000_000
    delta = cell["arms"]["on"]["ttft_ps"] - cell["arms"]["off"]["ttft_ps"]
    assert delta == cell["identities_one_channel"] * COST_PS
    text = RESULTS_MD.read_text(encoding="utf-8")
    assert "post-specified" in text
    assert "slow-fabric-tp4" in text


def test_the_disclosed_vacuities_are_recorded(results: dict) -> None:
    local = results["cells"]["local-tp2"]
    assert local["arms"]["off"]["goal_sha256"] == []
    families = {
        family["id"]: family
        for family in results["scored"]["behavioral_families"]["families"]
    }
    b3 = {instance["cell"]: instance for instance in families["B3"]["instances"]}
    assert b3["local-tp2"]["goal_conjunct_vacuous"]
    assert not b3["mixed-tp4"]["goal_conjunct_vacuous"]
    text = RESULTS_MD.read_text(encoding="utf-8")
    assert "Vacuous in `local-tp2`" in text
    assert "vacuous in `local-tp2`" in text


def test_the_scored_overlap_is_disclosed(results: dict) -> None:
    note = results["scored"]["overlap_note"]
    assert "four views of one event" in note
    assert "entailed" in note
    text = RESULTS_MD.read_text(encoding="utf-8")
    assert "not seven independent" in text


def test_the_corrections_are_recorded() -> None:
    text = RESULTS_MD.read_text(encoding="utf-8")
    assert "## Corrections" in text
    for phrase in (
        "G3 was substituted",
        "G1 was vacuous",
        "The counterfactual arithmetic was wrong",
        "The evidence attribution was too broad",
        "A step count was wrong",
        "The layering claim overstated",
    ):
        assert phrase in text, phrase


def test_the_results_digest_recorded_in_the_report_is_current() -> None:
    import hashlib

    digest = hashlib.sha256(RESULTS_JSON.read_bytes()).hexdigest()
    assert digest in RESULTS_MD.read_text(encoding="utf-8")


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
    assert set(scored) == {
        "exact_oracle_rows",
        "behavioral_families",
        "overlap_note",
    }
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
    assert fail_closed["calibrated_without_measurement_raises"]
    assert "not calibrated" in fail_closed["calibrated_message"]
    assert "TRAF-56" in fail_closed["calibrated_message"]


def test_the_report_states_what_the_run_does_not_establish() -> None:
    text = RESULTS_MD.read_text(encoding="utf-8")
    assert "What this run does not establish" in text
    assert "Almost none of the model is measured" in text
    assert "No packetized handshake" in text
    assert "No live new-peer re-registration" in text
    assert "Two registration states, not one" in text
    assert "Two compositions untested" in text
    assert EM_DASH not in text
