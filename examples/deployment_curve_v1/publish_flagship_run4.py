#!/usr/bin/env python3
"""Publish the compact content-addressed fourth CORE-54 scored record."""

from __future__ import annotations

import argparse
import csv
import json
from fractions import Fraction
from pathlib import Path
from typing import Any

from flagship_run4_field_reader import read_run4_result
from flagship_run4_tools import as_fraction, fraction_json, sha256, write_json
from plot_flagship_run4 import prepare_flagship_plot, render_flagship_figure

PUBLICATION_SCHEMA = "simllm-deployment-curve-flagship-run4-publication-v1"
RESULT_SCHEMA = "simllm-deployment-curve-flagship-run4-result-v1"
PREDICTED_THROUGHPUT = Fraction(1_024_000_000_000, 124_071_011)
MEASURED_FOUR_LAYER_STEP_PS = 2_033_951_000


def _load_access(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8", newline="") as stream:
        for line in stream:
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError("run-4 access row must be an object")
            rows.append(value)
    return rows


def _arithmetic() -> dict[str, Any]:
    target_step = Fraction(MEASURED_FOUR_LAYER_STEP_PS * 61, 4)
    throughput = Fraction(256 * 1_000_000_000_000, target_step)
    if throughput != PREDICTED_THROUGHPUT:
        raise ValueError("run-4 publication arithmetic differs from the freeze")
    return {
        "measured_four_layer_step_service_ps": MEASURED_FOUR_LAYER_STEP_PS,
        "measured_layers": 4,
        "target_layers": 61,
        "depth_treatment": "LINEAR_61_OVER_4_CORE61_OPEN",
        "batch_per_node": 128,
        "gpus_per_node": 8,
        "batch_per_gpu": 16,
        "emitted_tokens_per_request": 2,
        "emitted_tokens_per_node_step": 256,
        "target_step_service_ps": fraction_json(target_step),
        "predicted_throughput_tokens_per_second_per_node": fraction_json(throughput),
        "predicted_throughput_decimal": float(throughput),
    }


def build_publication_result(
    result: dict[str, Any],
    artifact_identities: dict[str, dict[str, str]],
    publication_access: list[dict[str, Any]],
) -> dict[str, Any]:
    """Drop the bulk trace while retaining every fourth-run conclusion."""

    if result.get("schema") != RESULT_SCHEMA:
        raise ValueError("fourth scored result schema disagrees")
    if "requests" in result["shape_observation"]:
        raise ValueError("bulk request trace crossed the field projection")
    score = result["mtp_score"]
    if score["score_attempt_count"] != 1 or score["attenuation_applied"]:
        raise ValueError("MTP one-shot or attenuation invariant differs")
    for comparison in score["layers"].values():
        if as_fraction(comparison["prediction"]["point"], "mtp.point") != (
            PREDICTED_THROUGHPUT
        ):
            raise ValueError("MTP point differs from the frozen prediction")
    run3 = result["run3_carry_forward"]
    return {
        "schema": PUBLICATION_SCHEMA,
        "status": result["status"],
        "verdict": result["verdict"],
        "classification": result["classification"],
        "scope": result["scope"],
        "core54_closure": result["core54_closure"],
        "closure_reason": result["closure_reason"],
        "mtp_per_layer_arithmetic": _arithmetic(),
        "allocation": result["allocation"],
        "shape_observation": result["shape_observation"],
        "fit": result["fit"],
        "attenuation_layer": result["attenuation_layer"],
        "run3_carry_forward": run3,
        "mtp_score": score,
        "combined_held_out_rows": [*run3["held_out_score"]["rows"], score],
        "access": {
            "scored_run": result["access"],
            "publication_reader": {
                "rows": publication_access,
                "whole_record_loaded": False,
                "successful_projection_count": sum(
                    row.get("classification") == "run4_publication"
                    and row.get("status") == "PASS"
                    for row in publication_access
                ),
            },
        },
        "preservation_lock": result["preservation_lock"],
        "dominant_contributor": result["dominant_contributor"],
        "remaining_work": result["remaining_work"],
        "deployment_frontier": result["deployment_frontier"],
        "provenance": result["provenance"],
        "artifact_identities": artifact_identities,
    }


def _fraction_text(value: object, name: str) -> str:
    fraction = as_fraction(value, name)
    return f"{fraction.numerator}/{fraction.denominator}"


def write_score_table(path: Path, result: dict[str, Any]) -> None:
    """Write all three layer rows for the three held-out anchors."""

    columns = (
        "anchor_id",
        "source_run",
        "layer",
        "published_tokens_per_second_per_node",
        "prediction_fraction",
        "prediction_tokens_per_second_per_node",
        "lower_fraction",
        "upper_fraction",
        "signed_relative_error_fraction",
        "signed_error_percent",
        "status",
    )
    rows = []
    layer_names = (
        "physics_only",
        "physics_plus_boundary",
        "physics_plus_boundary_plus_attenuation",
    )
    scored_rows = [
        *(result["run3_carry_forward"]["held_out_score"]["rows"]),
        result["mtp_score"],
    ]
    for scored in scored_rows:
        source_run = "run4" if scored["anchor_id"].endswith("mtp") else "run3-carried"
        for layer_name in layer_names:
            comparison = scored["layers"][layer_name]
            prediction = comparison["prediction"]
            signed = as_fraction(comparison["signed_relative_error"], "signed error")
            rows.append(
                {
                    "anchor_id": scored["anchor_id"],
                    "source_run": source_run,
                    "layer": layer_name,
                    "published_tokens_per_second_per_node": (
                        f"{float(as_fraction(scored['published'], 'published')):.9f}"
                    ),
                    "prediction_fraction": _fraction_text(
                        prediction["point"], "prediction point"
                    ),
                    "prediction_tokens_per_second_per_node": (
                        f"{float(as_fraction(prediction['point'], 'prediction point')):.9f}"
                    ),
                    "lower_fraction": _fraction_text(prediction["lower"], "lower"),
                    "upper_fraction": _fraction_text(prediction["upper"], "upper"),
                    "signed_relative_error_fraction": _fraction_text(
                        comparison["signed_relative_error"], "signed error"
                    ),
                    "signed_error_percent": f"{100 * float(signed):.9f}",
                    "status": comparison.get(
                        "status",
                        (
                            "PASS"
                            if comparison["point_passes_5_percent"]
                            else "REFUTED"
                        ),
                    ),
                }
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--digest-output", type=Path, required=True)
    parser.add_argument("--figure-data", type=Path, required=True)
    parser.add_argument("--figure-stem", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result_path = args.run_root / "attempt-1" / "result.json"
    access_path = args.run_root / "access.jsonl"
    projected = read_run4_result(result_path, access_path, args.run_root)
    publication_access = _load_access(access_path)
    publication = build_publication_result(projected, {}, publication_access)
    write_score_table(args.figure_data, publication)
    pdf, png = render_flagship_figure(
        prepare_flagship_plot(publication),
        args.figure_stem,
    )
    identities = {
        name: {"filename": path.name, "sha256": sha256(path)}
        for name, path in {
            "full_scored_result": result_path,
            "frozen_prediction": args.run_root / "attempt-1/frozen-prediction.json",
            "held_out_score": args.run_root / "attempt-1/held-out-score.json",
            "run3_carry_forward": args.run_root / "attempt-1/run3-carry-forward.json",
            "access_ledger": access_path,
            "figure_data": args.figure_data,
            "publication_pdf": pdf,
            "publication_png": png,
        }.items()
    }
    publication = build_publication_result(projected, identities, publication_access)
    write_json(args.output, publication)
    digest = sha256(args.output)
    args.digest_output.write_text(
        f"{digest}  {args.output.name}\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        f"wrote {args.output.as_posix()} at sha256:{digest}; "
        f"figure {pdf.as_posix()} and {png.as_posix()}"
    )


if __name__ == "__main__":
    main()
