"""Published-result locks for the third CORE-54 scored run."""

from __future__ import annotations

import hashlib
import json
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "examples" / "deployment_curve_v1"
RESULT = STUDY / "flagship_run3_result.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load() -> dict[str, object]:
    with RESULT.open(encoding="utf-8", newline="") as stream:
        return json.load(stream)


def _fraction(value: dict[str, int]) -> Fraction:
    return Fraction(value["numerator"], value["denominator"])


def test_run3_publication_has_the_literal_scoped_verdict():
    result = _load()
    score = result["held_out_score"]

    assert result["schema"] == (
        "simllm-deployment-curve-flagship-run3-publication-v1"
    )
    assert result["status"] == "PASS"
    assert result["verdict"] == "SCORABLE_HELD_OUT_PASS_MTP_BLOCKED"
    assert result["scope"] == (
        "priced held-out prefill anchors under the declared benchmark-bias model"
    )
    assert result["core54_closure"] is False
    assert score["status"] == "PASS"
    assert score["unattenuated_status"] == "REFUTED"
    assert score["accessed_anchor_ids"] == [
        "sglang_prefill_2k",
        "sglang_prefill_4k",
    ]
    assert score["forbidden_anchor_ids_accessed"] == []
    assert score["blocked_rows"] == [
        {
            "anchor_id": "sglang_decode_simulated_mtp",
            "dependency": "COMP-72 resumable Merlin execution",
            "numeric_anchor_read": False,
            "prediction": None,
            "published": None,
            "reason": "candidate record has no EP72 MTP batch-16 KV-4000 cell",
            "status": "BLOCKED",
        }
    ]


def test_run3_scored_layers_and_frozen_fit_are_exact():
    result = _load()
    constants = {row["id"]: row for row in result["constant_fit"]["constants"]}
    rows = {row["anchor_id"]: row for row in result["held_out_score"]["rows"]}

    assert result["constant_fit_sha256"] == (
        "78a798178234932325381aa7328ebd0dc816400e5a9caa3d6e5577edd0724883"
    )
    assert constants["intra_node_collective_surcharge_ps"]["fitted"] == 0
    assert constants["intra_node_collective_surcharge_ps"][
        "application_count_per_step"
    ] == 0
    assert _fraction(constants["overlap_exposed_fraction"]["fitted"]) == 0
    assert constants["overlap_exposed_fraction"]["clamped_to_envelope"] is True

    point = Fraction(3_276_800_000_000_000, 57_154_494_009)
    attenuated = Fraction(
        3_079_182_591_456_051_200_000_000,
        59_126_568_730_699_352_529,
    )
    for anchor_id, published in (
        ("sglang_prefill_2k", 54_543),
        ("sglang_prefill_4k", 50_302),
    ):
        row = rows[anchor_id]
        assert _fraction(row["published"]) == published
        assert _fraction(row["layers"]["physics_only"]["prediction"]["point"]) == point
        assert _fraction(
            row["layers"]["physics_plus_boundary"]["prediction"]["point"]
        ) == point
        scored = row["layers"]["physics_plus_boundary_plus_attenuation"]
        assert _fraction(scored["prediction"]["point"]) == attenuated
        assert scored["point_passes_5_percent"] is True
        assert row["layers"]["physics_only"]["point_passes_5_percent"] is False
        assert row["layers"]["physics_plus_boundary"][
            "point_passes_5_percent"
        ] is False


def test_run3_attenuation_is_frozen_independent_and_fewer_than_anchors():
    result = _load()
    layer = result["attenuation_layer"]
    factor = layer["factors"][0]

    assert layer["admitted_factor_count"] == 1
    assert factor["status"] == "ADMITTED"
    assert factor["anchor_numeric_input_count"] == 0
    assert factor["factor_count_is_less_than_touched_anchor_count"] is True
    assert factor["applies_to_anchor_ids"] == [
        "sglang_prefill_1k",
        "sglang_prefill_2k",
        "sglang_prefill_4k",
    ]
    assert _fraction(factor["magnitude"]["factor"]) == Fraction(
        939_691_952_959,
        1_034_504_281_000,
    )
    assert result["constant_fit"]["attenuation_factor_fitted"] is False
    assert [row["status"] for row in layer["rejected_candidates"]] == [
        "NOT_ADMITTED",
        "FORBIDDEN_BY_POLICY_RULE_FIVE",
    ]


def test_run3_access_chronology_keeps_mtp_unread():
    result = _load()
    access = result["anchor_access"]

    assert access["status"] == "PASS"
    assert access["whole_anchor_record_loaded"] is False
    assert access["mtp_numeric_access_count"] == 0
    assert [row["anchor_id"] for row in access["sequence"]] == [
        "sglang_prefill_1k",
        "sglang_decode_standard",
        "sglang_prefill_2k",
        "sglang_prefill_4k",
    ]
    assert [row["classification"] for row in access["sequence"]] == [
        "calibration",
        "calibration",
        "held_out",
        "held_out",
    ]
    assert all(row["whole_record_loaded"] is False for row in access["sequence"])


def test_run3_decode_remains_unattenuated_and_all_candidate_keys_bind():
    result = _load()
    decode = result["decode_calibration_miss"]
    selections = {row["anchor_id"]: row for row in result["candidate_selections"]}
    expected_key = "05d1c33cdef9c12e25eb9159adc9dc80f1cd57b6333778f9efb5fb24cd6a74aa"

    assert set(selections) == {
        "sglang_prefill_1k",
        "sglang_prefill_2k",
        "sglang_prefill_4k",
        "sglang_decode_standard",
    }
    assert all(row["lookup_hits"] == 1 for row in selections.values())
    assert all(row["lookup_misses"] == 0 for row in selections.values())
    assert decode["candidate_key_sha256"] == expected_key
    assert decode["declared_step_ps"] == 28_604_120_000
    assert decode["attenuation_applied"] is False
    assert decode["in_run_adjustment_performed"] is False
    assert decode["signed_direction"] == "prediction low"


def test_run3_artifacts_and_preservation_lock_match_bytes():
    result = _load()
    identities = result["artifact_identities"]

    assert _sha256(RESULT) == (
        "255a73b120e2ad6e3a7b202475419d30174298590d6c9d3c22f9cfb6063489fe"
    )
    for key, suffix in (("publication_pdf", ".pdf"), ("publication_png", ".png")):
        artifact = identities[key]
        path = STUDY / "figures" / artifact["filename"]
        assert path.suffix == suffix
        assert _sha256(path) == artifact["sha256"]
    assert (STUDY / "figures/deepseek-deployment-curve-run3.pdf").read_bytes().startswith(
        b"%PDF"
    )
    assert (STUDY / "figures/deepseek-deployment-curve-run3.png").read_bytes().startswith(
        b"\x89PNG"
    )
    assert result["preservation_lock"]["status"] == "PASS"
    assert len(result["preservation_lock"]["artifacts"]) == 33
    for item in result["preservation_lock"]["artifacts"]:
        assert _sha256(ROOT / item["path"]) == item["sha256"]


def test_run3_runtime_provenance_keeps_weights_and_web_off():
    result = _load()
    provenance = result["provenance"]

    assert provenance["python_version"] == "3.10.18"
    assert provenance["sglang_commit"] == (
        "bfeae4e79a8dc4600e006f1a5fbc85321a01c1a3"
    )
    assert provenance["model_weights_loaded"] is False
    assert provenance["web_pages_fetched"] is False
