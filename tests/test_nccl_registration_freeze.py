"""Lock the nccl_registration_v1 freeze document.

The freeze is the contract the study is scored against, so it is locked the
moment it lands and before the mechanism it describes exists. These checks
read the frozen document only; they never execute the study and never assert a
measured value.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

STUDY = Path(__file__).resolve().parents[1] / "examples" / "nccl_registration_v1"
EXPECTATIONS_JSON = STUDY / "expectations.json"
EXPECTATIONS_MD = STUDY / "expectations.md"

DECLARED_REGISTRATION_COST_PS = 20_000_000
EM_DASH = "\u2014"


@pytest.fixture(scope="module")
def freeze() -> dict:
    return json.loads(EXPECTATIONS_JSON.read_text(encoding="utf-8"))


def test_the_freeze_documents_are_present() -> None:
    assert EXPECTATIONS_MD.is_file()
    assert EXPECTATIONS_JSON.is_file()


def test_the_declared_cost_is_the_frozen_constant(freeze: dict) -> None:
    declared = freeze["declared"]
    assert declared["registration_cost_ps"] == DECLARED_REGISTRATION_COST_PS
    assert declared["evidence_class"] == "declared"
    assert declared["default_state"] == "disabled"
    assert declared["identity_fields"] == [
        "communicator",
        "generation",
        "channel",
        "buffer",
    ]
    assert declared["reregistration_events"] == [
        "new buffer",
        "new peer set",
        "communicator rebuild",
    ]


def test_the_cell_geometries_imply_the_frozen_identity_counts(freeze: dict) -> None:
    for name, cell in freeze["cells"].items():
        sites_per_layer = 2
        expected = cell["dims"]["num_layers"] * sites_per_layer
        assert cell["collectives_per_step"] == expected, name
        assert cell["identities_one_channel"] == expected, name


def test_the_exact_oracle_rows_follow_from_the_declared_cost(freeze: dict) -> None:
    rows = {row["id"]: row for row in freeze["scored"]["exact_oracle_rows"]["rows"]}
    local = freeze["cells"]["local-tp2"]["identities_one_channel"]
    mixed = freeze["cells"]["mixed-tp4"]["identities_one_channel"]
    assert rows["O1"]["value_ps"] == local * DECLARED_REGISTRATION_COST_PS
    assert rows["O2"]["value_ps"] == 0
    assert rows["O3"]["value_ps"] == 2 * local * DECLARED_REGISTRATION_COST_PS
    assert rows["O4"]["value_ps"] == local * DECLARED_REGISTRATION_COST_PS
    assert rows["O5"]["value_ps"] == mixed * DECLARED_REGISTRATION_COST_PS


def test_the_scored_denominators_match_the_declared_rows(freeze: dict) -> None:
    scored = freeze["scored"]
    assert scored["exact_oracle_rows"]["denominator"] == len(
        scored["exact_oracle_rows"]["rows"]
    )
    families = scored["behavioral_families"]
    assert families["denominator_families"] == len(families["families"])
    assert families["denominator_instances"] == sum(
        family["instances"] for family in families["families"]
    )


def test_the_physical_ceilings_match_the_cell_geometries(freeze: dict) -> None:
    bounds = freeze["physical_bounds"]
    local = freeze["cells"]["local-tp2"]["identities_one_channel"]
    mixed = freeze["cells"]["mixed-tp4"]["identities_one_channel"]
    assert bounds["charge_ceiling_local_tp2_one_channel_ps"] == (
        local * DECLARED_REGISTRATION_COST_PS
    )
    assert bounds["charge_ceiling_local_tp2_two_channel_ps"] == (
        2 * local * DECLARED_REGISTRATION_COST_PS
    )
    assert bounds["charge_ceiling_mixed_tp4_one_channel_ps"] == (
        mixed * DECLARED_REGISTRATION_COST_PS
    )


def test_every_fatal_guard_is_declared_with_a_claim(freeze: dict) -> None:
    guards = freeze["fatal_guards"]
    assert [guard["id"] for guard in guards] == [f"G{index}" for index in range(1, 10)]
    assert all(guard["claim"].strip() for guard in guards)


def test_the_freeze_prose_has_no_em_dash() -> None:
    text = EXPECTATIONS_MD.read_text(encoding="utf-8")
    assert EM_DASH not in text
