from __future__ import annotations

import hashlib
import json
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "examples/deployment_curve_v1"
FREEZE_PATH = STUDY / "scored_run4_expectations.json"
PS_PER_SECOND = 1_000_000_000_000


def _freeze() -> dict[str, object]:
    return json.loads(FREEZE_PATH.read_text(encoding="utf-8"))


def _fraction(value: dict[str, int]) -> Fraction:
    return Fraction(value["numerator"], value["denominator"])


def test_run4_freeze_precedes_held_out_access_and_every_scored_output() -> None:
    frozen = _freeze()

    assert frozen["schema"] == "simllm-deployment-curve-scored-run4-expectations-v1"
    assert frozen["status"] == "EXPECTATIONS_ONLY"
    chronology = frozen["chronology"]
    assert chronology["evidence_projection_read_after_reader_commit"] is True
    assert chronology["held_out_mtp_anchor_numeric_value_accessed"] is False
    assert chronology["run4_fit_performed"] is False
    assert chronology["run4_runner_existed_before_this_freeze"] is False
    assert chronology["run4_score_existed_before_this_freeze"] is False
    assert chronology["run4_figure_existed_before_this_freeze"] is False


def test_run4_freeze_contains_no_mtp_anchor_numeric_value() -> None:
    raw = FREEZE_PATH.read_text(encoding="utf-8")

    assert "17373" not in raw


def test_run4_mtp_step_depth_and_throughput_arithmetic_are_exact() -> None:
    frozen = _freeze()
    evidence = frozen["evidence"]
    model = frozen["mtp_model"]
    row = frozen["pre_fit_prediction_layers"][0]

    assert evidence["evidence_class"] == "MEASURED"
    assert evidence["component_overlay"] == "DISCLOSED"
    assert evidence["independent_observations"] == 2
    assert model["batch_mapping"] == {
        "batch_per_gpu": 16,
        "batch_per_node": 128,
        "gpus_per_node": 8,
    }
    assert model["emission"]["emitted_tokens_per_node_step"] == 256
    service = Fraction(evidence["measured_four_layer_step_service_ps"] * 61, 4)
    expected = Fraction(256 * PS_PER_SECOND, service)
    assert service == model["depth"]["target_step_service_ps"]
    for layer in (
        "physics_only",
        "physics_plus_boundary",
        "physics_plus_boundary_plus_attenuation",
    ):
        assert _fraction(row[layer]["lower"]) == expected
        assert _fraction(row[layer]["point"]) == expected
        assert _fraction(row[layer]["upper"]) == expected


def test_run4_has_no_admissible_decode_attenuation_or_refit() -> None:
    frozen = _freeze()
    attenuation = frozen["attenuation_layer"]
    fit = frozen["fit_rule"]

    assert attenuation["status"] == "NO_ADMISSIBLE_DECODE_FACTOR"
    assert attenuation["admitted_factor_count"] == 0
    assert attenuation["anchor_numeric_input_count"] == 0
    assert attenuation["run3_ep32_factor_reused"] is False
    assert attenuation["fresh_ep72_derivation"]["status"] == (
        "DERIVABLE_BUT_NOT_APPLICABLE"
    )
    assert fit["decision"] == "INHERIT_RUN3_FROZEN_FIT_WITHOUT_REFIT"
    assert fit["refit_allowed"] is False
    assert fit["mtp_parameter_count"] == 0
    assert fit["held_out_anchor_input_count"] == 0


def test_run4_source_citations_are_pinned_and_integration_caveat_is_explicit() -> None:
    model = _freeze()["mtp_model"]
    citations = model["source_citations"]

    assert citations[0]["commit"] == "bfeae4e79a8dc4600e006f1a5fbc85321a01c1a3"
    assert citations[0]["lines"] == "584-650"
    assert citations[1]["lines"] == "899-920"
    assert "incomplete" in model["integration_caveat"]
    assert model["depth"]["core61_question_open"] is True


def test_run4_preservation_lock_matches_all_57_named_artifacts() -> None:
    artifacts = _freeze()["preservation_lock"]["artifacts"]

    assert len(artifacts) == 57
    for artifact in artifacts:
        path = ROOT / artifact["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == artifact["sha256"]
