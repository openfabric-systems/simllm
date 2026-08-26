#!/usr/bin/env python3
"""Publish the compact, repository-sized second CORE-54 scored record."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from flagship_run2_tools import load_json, sha256, write_json

PUBLICATION_SCHEMA = "simllm-deployment-curve-flagship-run2-publication-v1"
RESULT_SCHEMA = "simllm-deployment-curve-flagship-run2-result-v1"


def build_publication_result(
    result: dict[str, Any],
    artifact_identities: dict[str, dict[str, str]],
) -> dict[str, Any]:
    """Remove bulk request traces while preserving every scored conclusion."""

    if result.get("schema") != RESULT_SCHEMA:
        raise ValueError("second scored result schema disagrees")
    session_summary = []
    for observation in result["session_observations"]:
        session_summary.append(
            {
                name: observation[name]
                for name in (
                    "anchor_id",
                    "pool",
                    "candidate_entry_index",
                    "admissions",
                    "terminals",
                    "prompt_tokens_per_request",
                    "total_prompt_tokens",
                    "remote_kv_projection_enabled",
                    "stable_projection_sha256",
                    "batches",
                    "prefill_ranks",
                    "decode_ranks",
                )
            }
        )
    return {
        "schema": PUBLICATION_SCHEMA,
        "status": result["status"],
        "verdict": result["verdict"],
        "scope": result["scope"],
        "core54_closure": result["core54_closure"],
        "closure_reason": result["closure_reason"],
        "provenance": result["provenance"],
        "allocation": result["allocation"],
        "scale_mapping": result["scale_mapping"],
        "topology": result["topology"],
        "pricing_configuration": result["pricing_configuration"],
        "constant_fit": result["constant_fit"],
        "constant_fit_sha256": result["constant_fit_sha256"],
        "held_out_score": result["held_out_score"],
        "held_out_score_sha256": result["held_out_score_sha256"],
        "anchor_predictions": result["anchor_predictions"],
        "curves": result["curves"],
        "offered_load_sweep_requests_per_second": result[
            "offered_load_sweep_requests_per_second"
        ],
        "stable_identity_guard": result["stable_identity_guard"],
        "packet_observation": result["packet_observation"],
        "scored_session_summary": session_summary,
        "candidate_selections": result["candidate_selections"],
        "decode_calibration_miss": result["decode_calibration_miss"],
        "dominant_held_out_contributor": result[
            "dominant_held_out_contributor"
        ],
        "preservation_lock": result["preservation_lock"],
        "residuals_required": result["residuals_required"],
        "artifact_identities": artifact_identities,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--fit", type=Path, required=True)
    parser.add_argument("--score", type=Path, required=True)
    parser.add_argument("--figure-pdf", type=Path, required=True)
    parser.add_argument("--figure-png", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inputs = {
        "scored_result": args.result,
        "frozen_fit": args.fit,
        "held_out_score": args.score,
        "publication_pdf": args.figure_pdf,
        "publication_png": args.figure_png,
    }
    identities = {
        name: {"filename": path.name, "sha256": sha256(path)}
        for name, path in inputs.items()
    }
    publication = build_publication_result(load_json(args.result), identities)
    write_json(args.output, publication)
    print(f"wrote {args.output.as_posix()}")


if __name__ == "__main__":
    main()
