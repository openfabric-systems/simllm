"""Published-result locks for the second CORE-54 scored run."""

from __future__ import annotations

import hashlib
import json
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "examples" / "deployment_curve_v1"
RESULT = STUDY / "flagship_run2_result.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load() -> dict[str, object]:
    with RESULT.open(encoding="utf-8", newline="") as stream:
        return json.load(stream)


def _fraction(value: dict[str, int]) -> Fraction:
    return Fraction(value["numerator"], value["denominator"])


def test_run2_publication_has_the_literal_scoped_verdict():
    result = _load()

    assert result["schema"] == (
        "simllm-deployment-curve-flagship-run2-publication-v1"
    )
    assert result["status"] == "REFUTED"
    assert result["verdict"] == "SCORABLE_HELD_OUT_REFUTED_MTP_BLOCKED"
    assert result["scope"] == "priced held-out prefill anchors only"
    assert result["core54_closure"] is False
    assert result["held_out_score"]["accessed_anchor_ids"] == [
        "sglang_prefill_2k",
        "sglang_prefill_4k",
    ]
    assert result["held_out_score"]["forbidden_anchor_ids_accessed"] == []
    assert result["held_out_score"]["blocked_rows"][0]["status"] == "BLOCKED"


def test_run2_scored_points_and_frozen_fit_are_exact():
    result = _load()
    rows = {
        row["anchor_id"]: row for row in result["held_out_score"]["rows"]
    }

    assert result["constant_fit"]["fitted_ps"] == 0
    assert result["constant_fit"]["application_count_per_step"] == 0
    assert result["constant_fit_sha256"] == (
        "be96c1de5b6a9eff3b8529ee1947482453faeed30d08c4d3132624dfbef72fde"
    )
    for anchor_id, published in (
        ("sglang_prefill_2k", 54_543),
        ("sglang_prefill_4k", 50_302),
    ):
        row = rows[anchor_id]
        assert _fraction(row["predicted"]["point"]) == Fraction(
            3_276_800_000_000_000,
            57_154_494_009,
        )
        assert _fraction(row["published"]) == published
        assert row["point_passes_5_percent"] is False


def test_run2_decode_key_binds_and_miss_stays_disclosed():
    result = _load()
    decode = result["decode_calibration_miss"]
    selections = {
        row["anchor_id"]: row for row in result["candidate_selections"]
    }
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
    assert selections["sglang_decode_standard"][
        "selected_entry_key_sha256s"
    ] == [expected_key]
    assert decode["declared_step_ps"] == 28_604_120_000
    assert decode["signed_direction"] == "prediction low"
    assert decode["in_run_adjustment_performed"] is False


def test_run2_artifacts_and_preservation_lock_match_bytes():
    result = _load()
    identities = result["artifact_identities"]

    assert _sha256(RESULT) == (
        "0e3db1a8d8ecc79d54618bbef7d2d2801862d1ec3188e3cc2a209f225a3919dd"
    )
    for key, suffix in (("publication_pdf", ".pdf"), ("publication_png", ".png")):
        artifact = identities[key]
        path = STUDY / "figures" / artifact["filename"]
        assert path.suffix == suffix
        assert _sha256(path) == artifact["sha256"]
    assert (STUDY / "figures/deepseek-deployment-curve-run2.pdf").read_bytes().startswith(
        b"%PDF"
    )
    assert (STUDY / "figures/deepseek-deployment-curve-run2.png").read_bytes().startswith(
        b"\x89PNG"
    )
    assert result["preservation_lock"]["status"] == "PASS"
    assert len(result["preservation_lock"]["artifacts"]) == 24
    for item in result["preservation_lock"]["artifacts"]:
        assert _sha256(ROOT / item["path"]) == item["sha256"]
