from __future__ import annotations

import json
from pathlib import Path, PureWindowsPath

from simllm.calibration.canonical import canonical_sha256

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "examples" / "hopper_kernel_cycle_candidate_v1"
FREEZE = json.loads((STUDY / "expectations.json").read_text(encoding="utf-8"))


def test_freeze_is_expectations_only_and_selects_the_local_arm() -> None:
    assert FREEZE["status"] == "EXPECTATIONS_ONLY"
    assert FREEZE["reachability"] == {
        "command": "timeout 12 ssh -o BatchMode=yes -o ConnectTimeout=8 merlin hostname",
        "verbatim_output": [
            "Connection timed out during banner exchange",
            "Connection to UNKNOWN port 65535 timed out",
        ],
        "exit_status": 255,
        "verdict": "LOCAL_ARM",
        "retry_count": 0,
    }
    assert not (STUDY / "results.json").exists()


def test_retained_nsys_inventory_is_cell_exact() -> None:
    inventory = FREEZE["retained_nsys_inventory"]
    granite = {
        cell
        for source in inventory["granite"]["present"]
        for cell in source["cells"]
    }
    deepseek = {
        cell
        for source in inventory["deepseek_v3"]["present"]
        for cell in source["cells"]
    }

    assert len(granite) == 12
    assert len(deepseek) == 4
    assert inventory["granite"]["rejected_or_absent"][-1]["cells"] == 1212
    assert inventory["deepseek_v3"]["rejected_or_absent"][-1]["cells"] == [
        "ep72-mtp-decode-b16-c4000"
    ]


def test_evidence_ledgers_keep_non_additive_axes_separate() -> None:
    ledgers = FREEZE["ledger_expectations"]

    assert ledgers["granite"] == {
        "service_entries": {"MEASURED": 12, "DECLARED": 0},
        "component_overlays": {"DISCLOSED": 12},
        "registered_campaign_cells": {"ABSENT": 1212},
        "non_additive_axes": True,
    }
    assert ledgers["deepseek_v3"] == {
        "service_entries": {"MEASURED": 4, "DECLARED": 4},
        "component_overlays": {"DISCLOSED": 8},
        "requested_physical_cells": {"MEASURED": 4, "ABSENT": 1},
        "non_additive_axes": True,
    }


def test_split_holds_out_the_largest_shapes_and_mtp() -> None:
    split = FREEZE["cell_split"]

    assert len(split["calibration"]) == 11
    assert len(split["held_out"]) == 6
    assert "deepseek-ep32-prefill-r4-l4096-t16384" in split["held_out"]
    assert "deepseek-ep72-mtp-decode-b16-c4000" in split["held_out"]
    assert not set(split["calibration"]) & set(split["held_out"])


def test_bands_are_numeric_and_absence_is_explicit() -> None:
    ratios = {row["family"]: row for row in FREEZE["a100_over_gh200_ratio_envelopes"]}
    for family in (
        "granite-graph-decode",
        "granite-graph-prefill",
        "granite-eager-decode",
        "granite-eager-prefill",
    ):
        row = ratios[family]
        assert row["lower_ppm"] <= row["retained_observed_lower_ppm"]
        assert row["retained_observed_upper_ppm"] <= row["upper_ppm"]
    assert ratios["deepseek-v3"]["state"] == "ABSENT"

    slopes = {row["family"]: row for row in FREEZE["kv_slope_bounds"]}
    assert slopes["granite-tp1-flashattention"]["tolerance_ppm"] == 100_000
    assert slopes["deepseek-mla"]["state"] == "ABSENT"


def test_depth_projection_is_the_only_declared_service_rule() -> None:
    projection = FREEZE["deepseek_depth_projection"]

    assert projection["reduced_layers"] == 4
    assert projection["full_layers"] == 61
    assert projection["evidence_class"] == "DECLARED"
    assert projection["rule"] == (
        "full_service_ps = round_half_up(reduced_service_ps * 61 / 4)"
    )


def test_deferred_commands_are_posix_and_pin_the_freeze() -> None:
    deferred = FREEZE["deferred_merlin"]

    assert deferred["task_id"] == "COMP-72"
    assert deferred["freeze"] == (
        "examples/hopper_kernel_cycle_candidate_v1/expectations.json"
    )
    for name, command in deferred.items():
        if not name.endswith("_command"):
            continue
        assert "\\" not in command
        assert PureWindowsPath(command).as_posix() == command


def test_projection_source_is_its_own_content_address() -> None:
    source = next(
        row
        for row in FREEZE["source_authorities"]
        if row["id"] == "deepseek-deployment-projection"
    )
    projection = json.loads((ROOT / source["configured_path"]).read_text(encoding="utf-8"))

    assert canonical_sha256(projection) == source["sha256"]
