"""Published-result locks for the fourth CORE-54 scored run."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import sys
from fractions import Fraction
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "examples/deployment_curve_v1"
RESULT = STUDY / "flagship_run4_result.json"
DIGEST = STUDY / "flagship_run4_result.sha256"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load() -> dict[str, object]:
    with RESULT.open(encoding="utf-8", newline="") as stream:
        return json.load(stream)


def _fraction(value: dict[str, int]) -> Fraction:
    return Fraction(value["numerator"], value["denominator"])


def _plot_module():
    sys.path.insert(0, str(STUDY))
    try:
        path = STUDY / "plot_flagship_run4.py"
        spec = importlib.util.spec_from_file_location("flagship_run4_plot_test", path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)


def test_run4_publication_has_the_literal_combined_refutation() -> None:
    result = _load()

    assert result["schema"] == (
        "simllm-deployment-curve-flagship-run4-publication-v1"
    )
    assert result["status"] == "REFUTED"
    assert result["verdict"] == "ALL_SCORABLE_HELD_OUT_REFUTED"
    assert result["core54_closure"] is False
    assert [row["anchor_id"] for row in result["combined_held_out_rows"]] == [
        "sglang_prefill_2k",
        "sglang_prefill_4k",
        "sglang_decode_simulated_mtp",
    ]
    assert result["run3_carry_forward"]["status"] == "BYTE_IDENTICAL_NOT_RESCORED"
    assert result["run3_carry_forward"]["authority_sha256"] == (
        "255a73b120e2ad6e3a7b202475419d30174298590d6c9d3c22f9cfb6063489fe"
    )


def test_run4_mtp_arithmetic_and_all_three_layer_misses_are_exact() -> None:
    result = _load()
    arithmetic = result["mtp_per_layer_arithmetic"]
    score = result["mtp_score"]
    prediction = Fraction(1_024_000_000_000, 124_071_011)
    signed = Fraction(-1_131_485_674_103, 2_155_485_674_103)

    assert arithmetic["measured_four_layer_step_service_ps"] == 2_033_951_000
    assert _fraction(arithmetic["target_step_service_ps"]) == 31_017_752_750
    assert _fraction(
        arithmetic["predicted_throughput_tokens_per_second_per_node"]
    ) == prediction
    assert score["score_attempt_count"] == 1
    assert score["attenuation_applied"] is False
    assert _fraction(score["published"]) == 17_373
    assert len(score["layers"]) == 3
    for comparison in score["layers"].values():
        assert comparison["status"] == "REFUTED"
        assert comparison["point_passes_5_percent"] is False
        assert _fraction(comparison["prediction"]["lower"]) == prediction
        assert _fraction(comparison["prediction"]["point"]) == prediction
        assert _fraction(comparison["prediction"]["upper"]) == prediction
        assert _fraction(comparison["signed_relative_error"]) == signed


def test_run4_carries_both_run3_pass_rows_without_mutation() -> None:
    result = _load()
    rows = result["run3_carry_forward"]["held_out_score"]["rows"]
    expected_error = {
        "sglang_prefill_2k": Fraction(
            -145_757_846_822_483_584_989_247,
            3_224_940_438_278_534_784_989_247,
        ),
        "sglang_prefill_4k": Fraction(
            52_498_965_582_206_184_543_121,
            1_487_092_330_145_819_415_456_879,
        ),
    }
    for row in rows:
        comparison = row["layers"]["physics_plus_boundary_plus_attenuation"]
        assert comparison["point_passes_5_percent"] is True
        assert _fraction(comparison["signed_relative_error"]) == expected_error[
            row["anchor_id"]
        ]


def test_run4_access_and_preservation_prove_one_score() -> None:
    result = _load()
    scored = result["access"]["scored_run"]
    publication = result["access"]["publication_reader"]

    assert scored["mtp_anchor_access_count"] == 1
    assert [row["classification"] for row in scored["rows"]] == [
        "measured_mtp_evidence",
        "inherited_run3_publication",
        "held_out",
    ]
    assert publication["successful_projection_count"] == 5
    assert len(publication["rows"]) == 8
    assert all(row["whole_record_loaded"] is False for row in publication["rows"])
    artifacts = result["preservation_lock"]["artifacts"]
    assert result["preservation_lock"]["status"] == "PASS"
    assert len(artifacts) == 57
    for artifact in artifacts:
        assert _sha256(ROOT / artifact["path"]) == artifact["sha256"]


def test_run4_content_address_and_repository_artifacts_match() -> None:
    result = _load()
    expected = "e2fd0811638af02ea4389f456e0e796d9a2b24e550da3217dddd2ecc6872a6cd"
    external_identities = {
        "full_scored_result": (
            "89443a2e9b98828b2f9350b8411eb79ca78566d7e88650281105cd1e974fc26d"
        ),
        "frozen_prediction": (
            "56b37ac4b36eff16d5f2be527b7b1a234147d2dfd1c031dec67dc84a81b7d652"
        ),
        "held_out_score": (
            "da4458dfc097a7990805528a3ce824101c927398670c726f8428442d64a1f3bf"
        ),
        "run3_carry_forward": (
            "0badb89adc0c95f0f98104cab174042341f06cd80e61360a5920ef24be5dae97"
        ),
        "access_ledger": (
            "550b7c82ab3de8a95b4263734b79cda760acd0eb57433bc77a4b9c9856645719"
        ),
    }

    assert _sha256(RESULT) == expected
    assert DIGEST.read_text(encoding="utf-8") == f"{expected}  {RESULT.name}\n"
    assert {
        name: result["artifact_identities"][name]["sha256"]
        for name in external_identities
    } == external_identities
    for name, relative in (
        ("figure_data", "flagship_run4_score_table.csv"),
        ("publication_pdf", "figures/deepseek-deployment-curve-run4.pdf"),
        ("publication_png", "figures/deepseek-deployment-curve-run4.png"),
    ):
        identity = result["artifact_identities"][name]
        path = STUDY / relative
        assert identity["filename"] == path.name
        assert _sha256(path) == identity["sha256"]
    assert (STUDY / "figures/deepseek-deployment-curve-run4.pdf").read_bytes().startswith(
        b"%PDF"
    )
    assert (STUDY / "figures/deepseek-deployment-curve-run4.png").read_bytes().startswith(
        b"\x89PNG"
    )


def test_run4_score_table_has_three_layers_for_each_anchor() -> None:
    path = STUDY / "flagship_run4_score_table.csv"
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))

    assert len(rows) == 9
    assert {row["anchor_id"] for row in rows} == {
        "sglang_prefill_2k",
        "sglang_prefill_4k",
        "sglang_decode_simulated_mtp",
    }
    mtp = [row for row in rows if row["source_run"] == "run4"]
    assert len(mtp) == 3
    assert {row["status"] for row in mtp} == {"REFUTED"}
    assert {row["signed_error_percent"] for row in mtp} == {"-52.493305230"}


def test_run4_frontier_and_remaining_scope_stay_honest() -> None:
    result = _load()

    assert result["deployment_frontier"]["status"] == "UNCHANGED_FROZEN_CONTRACT"
    assert result["remaining_work"] == [
        "decode calibration reproduction",
        "COMP-74 distribution propagation from retained repeats",
        "Granite campaign arm",
        "depth linearity",
    ]
    assert result["attenuation_layer"]["admitted_factor_count"] == 0
    assert result["provenance"]["model_weights_loaded"] is False
    assert result["provenance"]["web_pages_fetched"] is False


def test_run4_renderer_writes_pdf_and_png(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    plot = _plot_module()
    prepared = plot.prepare_flagship_plot(_load())

    pdf, png = plot.render_flagship_figure(prepared, tmp_path / "flagship-run4")

    assert prepared["mtp"]["status"] == "REFUTED"
    assert prepared["mtp"]["attempt_count"] == 1
    assert prepared["frontier_status"] == "UNCHANGED_FROZEN_CONTRACT"
    assert pdf.read_bytes().startswith(b"%PDF")
    assert png.read_bytes().startswith(b"\x89PNG")
