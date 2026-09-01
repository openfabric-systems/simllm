"""Build the expectations-only VLLM-42 service-predictor freeze."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from service_model import (
    HANDOFF_PS,
    HELD_OUT_LOADS,
    HELD_OUT_POOL_RATIOS,
    MAX_BATCH_SIZE,
    OFFERED_LOADS,
    OUTPUT_TOKENS,
    POOL_RATIOS,
    PREFILL_SERVICE_PS,
    PROMPT_LENGTHS,
    PS_PER_SECOND,
    REQUESTS_PER_CELL,
    THREE_SIGMA_MULTIPLIER,
    TIMING_SCENARIOS,
    all_predictions,
    fraction_from_json,
    fraction_json,
    physical_service_bounds_ps,
    surface_cv_envelope_ppm,
)

from simllm.calibration.batch_service_surface import BatchServicePoint

STUDY_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = STUDY_DIR.parents[1]
REFERENCE_STUDY_DIR = REPOSITORY_ROOT / "examples" / "pd_session_load_delay_v1"
SURFACE_PATH = REFERENCE_STUDY_DIR / "surface.json"
VLLM39_QUEUE_MODEL_PATH = REFERENCE_STUDY_DIR / "queue_model.py"
FIELD_READER_PATH = STUDY_DIR / "field_reader.py"
ACCESS_PROTOCOL_PATH = STUDY_DIR / "access_protocol.json"
FORBIDDEN_ACCESS_LEDGER_PATH = STUDY_DIR / "forbidden_access_ledger.json"
AUTHORED_AGAINST = "7c8ddf1"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_lines(*args: str) -> tuple[str, ...]:
    completed = subprocess.run(
        ["git", *args],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return tuple(line for line in completed.stdout.splitlines() if line)


def preservation_manifest() -> dict[str, Any]:
    """Lock every earlier tracked pd_session artifact by Git blob identity."""

    rows = []
    for line in _git_lines("ls-files", "-s", "examples/pd_session*"):
        metadata, path = line.split("\t", 1)
        _, blob_sha1, stage = metadata.split()
        if path.startswith("examples/pd_session_batching_service_v1/"):
            continue
        rows.append({"path": path, "blob_sha1": blob_sha1, "stage": int(stage)})
    canonical = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    queue_onset = [
        row
        for row in rows
        if row["path"].startswith("examples/pd_session_queue_onset_v1/")
    ]
    return {
        "selection_rule": "every tracked examples/pd_session* path except this successor study",
        "artifact_count": len(rows),
        "queue_onset_artifact_count": len(queue_onset),
        "manifest_sha256": hashlib.sha256(canonical).hexdigest(),
        "rows": rows,
    }


def _points(surface: dict[str, Any]) -> tuple[BatchServicePoint, ...]:
    return tuple(
        BatchServicePoint(
            batch_size=row["batch_size"],
            duration_ps=row["measured_service_ps"],
            uncertainty_fraction=(
                row["trimmed_coefficient_of_variation_ppm"] / 1_000_000
            ),
            entry_key_sha256=row["entry_key_sha256"],
            evidence_class=row["evidence_class"],
            split=row["split"],
        )
        for row in surface["points"]
    )


def build_freeze() -> dict[str, Any]:
    """Build the service-only predictions, split, and decision rules."""

    surface = _load_json(SURFACE_PATH)
    points = _points(surface)
    predictions = all_predictions(points)
    floor, ceiling = physical_service_bounds_ps(points)
    non_held_out = [row for row in predictions if row["split"] == "non-held-out"]
    held_out = [row for row in predictions if row["split"] == "held-out"]
    signed_rows = [
        row
        for row in predictions
        if row["phase_completion_signed_delta_ps"]["numerator"] != 0
    ]
    return {
        "schema": "simllm-pd-session-batching-service-expectations-v1",
        "status": "EXPECTATIONS_ONLY",
        "task": "VLLM-42",
        "date": "2026-09-01",
        "authored_against": AUTHORED_AGAINST,
        "chronology": {
            "field_reader_commit": "ffc9bbc",
            "sizing_commit": "7c8ddf1",
            "successor_run_existed_before_freeze": False,
            "vllm41_observed_batching_service_accessed_before_freeze": False,
            "vllm41_held_out_value_accessed_before_freeze": False,
            "observed_curve_fit_permitted": False,
        },
        "source_access": {
            "status": "CLEAN",
            "field_reader_path": FIELD_READER_PATH.relative_to(
                REPOSITORY_ROOT
            ).as_posix(),
            "field_reader_sha256": _sha256(FIELD_READER_PATH),
            "access_protocol_path": ACCESS_PROTOCOL_PATH.relative_to(
                REPOSITORY_ROOT
            ).as_posix(),
            "access_protocol_sha256": _sha256(ACCESS_PROTOCOL_PATH),
            "guarded_access_count_before_freeze": 0,
            "forbidden_access_ledger_path": FORBIDDEN_ACCESS_LEDGER_PATH.relative_to(
                REPOSITORY_ROOT
            ).as_posix(),
            "forbidden_access_ledger": _load_json(FORBIDDEN_ACCESS_LEDGER_PATH),
            "whole_record_loaded": False,
        },
        "independent_inputs": {
            "decode_batch_service_surface": {
                "path": SURFACE_PATH.relative_to(REPOSITORY_ROOT).as_posix(),
                "sha256": _sha256(SURFACE_PATH),
                "record_sha256": surface["record_sha256"],
                "acceptance_status": surface["acceptance_status"],
                "calibration_claim": False,
                "interpolation": surface["interpolation"],
                "points": [
                    {
                        "batch_size": row["batch_size"],
                        "measured_service_ps": row["measured_service_ps"],
                        "trimmed_coefficient_of_variation_ppm": row[
                            "trimmed_coefficient_of_variation_ppm"
                        ],
                        "entry_key_sha256": row["entry_key_sha256"],
                    }
                    for row in surface["points"]
                ],
            },
            "arrival_schedule": {
                "rule": "request i arrives at cell start plus i times floor(10^12 divided by offered requests per second)",
                "clock_ps_per_second": PS_PER_SECOND,
            },
            "prefill_service_ps": {str(key): value for key, value in PREFILL_SERVICE_PS.items()},
            "handoff_ps": HANDOFF_PS,
            "prefill_and_handoff_source": {
                "path": VLLM39_QUEUE_MODEL_PATH.relative_to(
                    REPOSITORY_ROOT
                ).as_posix(),
                "sha256": _sha256(VLLM39_QUEUE_MODEL_PATH),
                "relationship": "frozen before VLLM-41 and independent of its observed curves",
            },
        },
        "sweep": {
            "offered_load_requests_per_second": list(OFFERED_LOADS),
            "prompt_tokens": list(PROMPT_LENGTHS),
            "pool_ratios": [list(ratio) for ratio in POOL_RATIOS],
            "requests_per_cell": REQUESTS_PER_CELL,
            "decode_output_tokens_per_request": OUTPUT_TOKENS,
            "maximum_scheduler_batch_size": MAX_BATCH_SIZE,
            "cell_count": len(predictions),
        },
        "holdout": {
            "loads": list(HELD_OUT_LOADS),
            "pool_ratios": [list(ratio) for ratio in HELD_OUT_POOL_RATIOS],
            "union_rule": "a cell is held out when its load or pool ratio is held out",
            "non_held_out_cell_count": len(non_held_out),
            "held_out_cell_count": len(held_out),
            "disclosure_order": "publish and commit all non-held-out verdicts before running or scoring any held-out cell",
        },
        "predictor": {
            "name": "phase-complete-service-only-bulk-scheduler-v1",
            "derivation": {
                "arrival": "a_i equals i times floor(10^12 divided by lambda)",
                "prefill_completion": "a ready prefill batch advances the shared clock by the independently frozen prompt service",
                "handoff_completion": "decode eligibility is prefill completion plus the independently frozen handoff service",
                "decode_grant": "the next legal driver in prefill-then-decode round-robin order takes at most eight ready requests",
                "decode_release": "a decode grant of size b advances the shared clock by measured interpolated S(b); nonterminal requests retain queue order for their next token visit",
                "service_projection": "sum central S(b_j) over predicted decode grants j and divide by 64 times four request-token visits",
            },
            "observed_curve_inputs": [],
            "fit_parameters": [],
            "timing_scenarios": list(TIMING_SCENARIOS),
            "surface_uncertainty": {
                "rule": "scale only the decode service clock by plus or minus three times the maximum independent trimmed coefficient of variation; price every scenario batch with central S(b)",
                "multiplier": THREE_SIGMA_MULTIPLIER,
                "envelope_ppm": surface_cv_envelope_ppm(points),
            },
            "old_model_refutation_mechanism": {
                "omitted_term": "the old batching-service component predictor made prefill and handoff zero-time boundaries",
                "causal_effect": "removing those positive delays grants decode before some external arrivals are ready, so transition cells contain fewer requests per decode batch",
                "signed_bias": "smaller batches raise service per request-token, so the old predictor is biased high where the omitted phase changes batch membership",
                "locality": "away from a discrete batch transition, the added phase does not change batch membership and the service projection is unchanged",
            },
            "phase_changed_cell_count": len(signed_rows),
            "phase_changed_cells": [
                {
                    "configuration": row["configuration"],
                    "offered_load_requests_per_second": row[
                        "offered_load_requests_per_second"
                    ],
                    "signed_delta_ps": row["phase_completion_signed_delta_ps"],
                }
                for row in signed_rows
            ],
        },
        "physical_bounds": {
            "floor_service_per_token_ps": fraction_json(floor),
            "ceiling_service_per_token_ps": fraction_json(ceiling),
            "floor_derivation": "minimum measured interpolated S(b) divided by b for b from one through eight",
            "ceiling_derivation": "maximum measured interpolated S(b) divided by b for b from one through eight",
            "all_prediction_bands_inside_bounds": all(
                floor
                <= fraction_from_json(
                    row["batch_service_per_token_band_ps"]["lower"]
                )
                <= fraction_from_json(
                    row["batch_service_per_token_band_ps"]["upper"]
                )
                <= ceiling
                for row in predictions
            ),
        },
        "prediction_bands": predictions,
        "preservation": preservation_manifest(),
        "decision_rule": {
            "fatal_guards": "any conservation, identity, pricing, frozen-input, preservation, or chronology violation makes the run VOID and leaves VLLM-42 open",
            "per_cell": "the observed amortized batching-service field must lie inside its inclusive frozen band",
            "close_vllm42": "close only if every one of 78 cells holds, all fatal guards hold, and held-out cells are scored after the committed non-held-out publication",
            "refutation": "publish every miss without widening; keep VLLM-42 open and register the predictor residual on VLLM-50",
            "unscored_fields": "arrival-to-prefill wait and handoff-to-decode admission wait are published diagnostics and conservation inputs, not service-band scores",
            "settled_claims": "the 210 to 220 requests per second onset and the 250 to 8,000 requests per second monotonic direction remain preserved and unscored",
        },
    }


def _microseconds(value: dict[str, int]) -> str:
    return f"{float(fraction_from_json(value) / 1_000_000):.6f}"


def render_markdown(freeze: dict[str, Any]) -> str:
    """Render the human expectations record from the exact JSON authority."""

    bounds = freeze["physical_bounds"]
    rows = [
        "# VLLM-42 batching-service expectations",
        "",
        "Status: `EXPECTATIONS_ONLY`. No successor cell has run, and no observed",
        "VLLM-41 batching-service or held-out value entered this freeze.",
        "",
        "## Mechanism and derivation",
        "",
        "The old component predictor made prefill and handoff zero-time boundaries.",
        "That omission grants decode before some external arrivals are ready. Near a",
        "discrete batch transition, the predicted decode batch is therefore too small.",
        "Smaller batches cost more service per request-token, so the old prediction is",
        "biased high exactly when the omitted positive phase changes batch membership.",
        "Away from a transition the batch membership is unchanged, so the correction",
        "has zero signed effect.",
        "",
        "The replacement is service-only. Request `i` arrives at `i` times the floor",
        "of 10^12 divided by `lambda` picoseconds. A ready prefill batch advances the shared clock by the",
        "independently frozen prompt service, and decode becomes eligible after the",
        "independently frozen handoff service. The next legal pool driver takes at",
        "most eight ready requests. Decode batch `j` of size `b_j` advances the clock",
        "by the independently measured and interpolated `S(b_j)`. The predicted field",
        "is `sum_j S(b_j) / (64 * 4)`.",
        "",
        "The lower and upper timing scenarios scale only the decode service clock by",
        "plus or minus three times the largest independent trimmed coefficient of",
        "variation. Every scenario is still priced with central `S(b_j)`, so the band",
        "measures batch-transition uncertainty rather than changing the scored field.",
        "There are no observed-curve inputs and no fitted parameters.",
        "",
        "## Holdout and disclosure order",
        "",
        "Load 240 requests per second is the held-out batch-transition load. Pool ratio",
        "2:1 is the held-out ratio. Their union contains 30 cells. The other 48 cells",
        "must be run, scored, published, and committed before any held-out cell runs or",
        "is scored.",
        "",
        "## Physical sanity and acceptance",
        "",
        f"The service floor is {_microseconds(bounds['floor_service_per_token_ps'])} microseconds per request-token: the minimum `S(b)` divided by `b` over batch sizes one through eight.",
        f"The ceiling is {_microseconds(bounds['ceiling_service_per_token_ps'])} microseconds per request-token: the maximum `S(b)` divided by `b` over the same measured surface.",
        "Every frozen band lies within those bounds. At the maximum studied load,",
        "arrivals are four milliseconds apart; prompt service plus handoff is at most",
        "0.214936 milliseconds, while a measured decode batch takes 1.110576 to",
        "1.8928315 milliseconds. Those independent scales admit a transition from",
        "single-request batches without permitting service outside the measured bounds.",
        "",
        "A cell holds only when its observed amortized batching-service field lies in",
        "the inclusive exact rational band in `expectations.json`. Any conservation,",
        "identity, pricing, input-lock, preservation, or chronology failure makes the",
        "run void. VLLM-42 closes only if all 78 cells hold and disclosure order holds.",
        "Otherwise every miss publishes unchanged and the residual registers on",
        "VLLM-50. Arrival-to-prefill and handoff-to-decode waits publish separately but",
        "are unscored. The settled 210 to 220 requests per second onset and the 250 to",
        "8,000 requests per second monotonic direction remain unscored.",
        "",
        "## Frozen per-cell service bands",
        "",
        "Values below are microseconds per request-token. `expectations.json` retains",
        "the exact rational acceptance values.",
        "",
        "| Prefill | Decode | Prompt | Load | Split | Predicted | Lower | Upper |",
        "|---:|---:|---:|---:|:---|---:|---:|---:|",
    ]
    for prediction in freeze["prediction_bands"]:
        prefill, decode, prompt = prediction["configuration"]
        band = prediction["batch_service_per_token_band_ps"]
        rows.append(
            "| "
            f"{prefill} | {decode} | {prompt} | "
            f"{prediction['offered_load_requests_per_second']} | "
            f"{prediction['split']} | "
            f"{_microseconds(prediction['predicted_batch_service_per_token_ps'])} | "
            f"{_microseconds(band['lower'])} | {_microseconds(band['upper'])} |"
        )
    return "\n".join(rows) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    freeze = build_freeze()
    _write_json(args.output_dir / "expectations.json", freeze)
    (args.output_dir / "EXPECTATIONS.md").write_text(
        render_markdown(freeze),
        encoding="utf-8",
        newline="\n",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
