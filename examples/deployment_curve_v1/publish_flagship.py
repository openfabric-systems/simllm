#!/usr/bin/env python3
"""Publish a compact, repository-sized projection of the CORE-54 run."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from flagship_tools import load_json, sha256, write_json

PUBLICATION_SCHEMA = "simllm-deployment-curve-flagship-publication-v1"


def build_publication_result(
    result: dict[str, Any],
    binding: dict[str, Any],
    artifact_identities: dict[str, dict[str, str]],
) -> dict[str, Any]:
    """Remove bulk request traces while preserving every scored conclusion."""

    if result.get("schema") != "simllm-deployment-curve-flagship-result-v1":
        raise ValueError("scored result schema disagrees")
    if binding.get("schema") != "simllm-deployment-curve-binding-qualification-v1":
        raise ValueError("binding qualification schema disagrees")
    if any(
        binding[name]
        for name in (
            "fit_performed",
            "anchor_numeric_values_accessed",
            "held_out_score_performed",
        )
    ):
        raise ValueError("post-score binding qualification crossed the score boundary")
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
        "constant_fit": result["constant_fit"],
        "constant_fit_sha256": result["constant_fit_sha256"],
        "held_out_score": result["held_out_score"],
        "held_out_score_sha256": result["held_out_score_sha256"],
        "anchor_predictions": result["anchor_predictions"],
        "curves": result["curves"],
        "stable_identity_guard": result["stable_identity_guard"],
        "packet_observation": result["packet_observation"],
        "scored_session_summary": session_summary,
        "candidate_binding": {
            "scored_attempt_selections": result["candidate_selections"],
            "post_score_qualification_status": binding["status"],
            "post_score_qualification_run_head": binding["run_head"],
            "fit_performed": binding["fit_performed"],
            "anchor_numeric_values_accessed": binding[
                "anchor_numeric_values_accessed"
            ],
            "held_out_score_performed": binding["held_out_score_performed"],
            "post_score_selections": binding["candidate_selections"],
        },
        "runtime_finding": result["runtime_finding"],
        "dominant_held_out_contributor": result[
            "dominant_held_out_contributor"
        ],
        "residuals_required": result["residuals_required"],
        "publication_residuals_registered": ["CORE-59", "COMP-74", "SGL-38"],
        "existing_residual_owners": ["COMP-72", "SGL-36", "TRAF-64"],
        "artifact_identities": artifact_identities,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--binding", type=Path, required=True)
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
        "post_score_binding_qualification": args.binding,
        "frozen_fit": args.fit,
        "held_out_score": args.score,
        "publication_pdf": args.figure_pdf,
        "publication_png": args.figure_png,
    }
    identities = {
        name: {"filename": path.name, "sha256": sha256(path)}
        for name, path in inputs.items()
    }
    publication = build_publication_result(
        load_json(args.result),
        load_json(args.binding),
        identities,
    )
    write_json(args.output, publication)
    print(f"wrote {args.output.as_posix()}")


if __name__ == "__main__":
    main()
