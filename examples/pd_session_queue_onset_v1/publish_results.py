"""Publish the compact VLLM-41 result and its human-readable report."""

from __future__ import annotations

import argparse
import hashlib
import json
from decimal import Decimal, localcontext
from fractions import Fraction
from pathlib import Path
from typing import Any

STUDY_DIR = Path(__file__).resolve().parent
EXPECTATIONS_PATH = STUDY_DIR / "expectations.json"
DEFAULT_RESULT = STUDY_DIR / "results.json"
DEFAULT_REPORT = STUDY_DIR / "RESULTS.md"
RAW_RESULT_PATH = "$SIMLLM_VLLM41_RUN_ROOT/qualified-sharded-v1/result.json"
RAW_RESULT_SHA256 = "0cdd0f2bf6244d7c3daf75cfbaee5e56fa3fcc95bfb4718bbb11f7e5beca0248"
RAW_RESULT_BYTES = 18_833_582
RAW_RESULT_SCHEMA = "simllm-pd-session-queue-onset-result-v1"
COMPACT_RESULT_SCHEMA = "simllm-pd-session-queue-onset-compact-result-v1"
COMPACT_RESULT_SHA256 = "27ec9540979302625a85d2f1f1866e885bb980df69947dc50ee39e52dad26488"


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _load_json(path: Path) -> tuple[dict[str, Any], bytes]:
    payload = path.read_bytes()
    return json.loads(payload), payload


def _fraction(value: dict[str, int]) -> Fraction:
    return Fraction(value["numerator"], value["denominator"])


def _milliseconds(value: dict[str, int], places: int = 9) -> str:
    fraction = _fraction(value)
    with localcontext() as context:
        context.prec = 40
        milliseconds = Decimal(fraction.numerator) / Decimal(fraction.denominator)
        milliseconds /= Decimal(1_000_000_000)
        return f"{milliseconds:.{places}f}"


def _signed_milliseconds(value: dict[str, int]) -> str:
    rendered = _milliseconds(value)
    if _fraction(value) > 0:
        return f"+{rendered}"
    return rendered


def _configuration(value: list[int]) -> str:
    return "(" + ",".join(map(str, value)) + ")"


def compact_result(raw: dict[str, Any], raw_payload: bytes) -> dict[str, Any]:
    """Reduce raw request timelines to the complete scored public record."""

    if raw["schema"] != RAW_RESULT_SCHEMA:
        raise ValueError("raw result schema disagrees")
    if len(raw_payload) != RAW_RESULT_BYTES:
        raise ValueError("raw result byte count disagrees")
    raw_sha256 = _sha256_bytes(raw_payload)
    if raw_sha256 != RAW_RESULT_SHA256:
        raise ValueError(f"raw result hash disagrees: {raw_sha256}")

    expectations, expectations_payload = _load_json(EXPECTATIONS_PATH)
    provenance = raw["provenance"]
    analysis = raw["analysis"]
    observation = raw["observation"]
    if _sha256_bytes(expectations_payload) != provenance["expectations_sha256"]:
        raise ValueError("expectations hash disagrees")
    if len(raw["cell_run_manifest"]) != 78 or len(observation["cells"]) != 78:
        raise ValueError("raw cell registry is incomplete")
    if observation["total_delay_curves"] is not None:
        raise ValueError("raw result reopened total-delay curves")
    if analysis["status"] != "IDENTIFIED":
        raise ValueError("VLLM-41 onset was not identified")
    if analysis["fatal_guards"] != {"findings": [], "status": "HELD"}:
        raise ValueError("fatal guard disagrees")
    if analysis["held_out_band_summary"] != {
        "batch_service_held": 14,
        "evaluated": 30,
        "joint_held": 14,
        "queue_wait_held": 30,
    }:
        raise ValueError("held-out component verdict disagrees")
    if analysis["onset_summary"] != {
        "configurations_inside_prediction_band": 0,
        "configurations_resolved": 6,
        "distinct_observed_segments": [[210, 220]],
        "predicted_central_segment": [225, 230],
        "predicted_inclusive_segments": [[220, 225], [225, 230]],
    }:
        raise ValueError("onset verdict disagrees")
    if analysis["closure"] != {
        "VLLM-41": "CLOSED",
        "VLLM-42": "REGISTER_RESIDUAL",
        "VLLM-43": "UNUSED_RESERVED",
    }:
        raise ValueError("task closure disagrees")

    return {
        "schema": COMPACT_RESULT_SCHEMA,
        "status": analysis["status"],
        "raw_run": {
            "path": RAW_RESULT_PATH,
            "sha256": raw_sha256,
            "bytes": len(raw_payload),
            "run_head": provenance["run_head"],
            "runner_exit_status": 0,
            "cell_run_manifest": raw["cell_run_manifest"],
        },
        "runtime": observation["runtime"],
        "freeze": {
            "commit": provenance["freeze_commit"],
            "expectations_sha256": provenance["expectations_sha256"],
            "access_ledger_sha256": provenance["access_ledger_sha256"],
            "surface_sha256": provenance["surface_sha256"],
            "queue_model_sha256": provenance["queue_model_sha256"],
            "chronology": expectations["chronology"],
            "source_access": expectations["source_access"],
            "surface": expectations["surface"],
            "sweep": expectations["sweep"],
            "queue_model": expectations["queue_model"],
            "decomposition": expectations["decomposition"],
            "decision_rule": expectations["decision_rule"],
            "held_out": {
                "loads": expectations["held_out"]["loads"],
                "pool_ratios": expectations["held_out"]["pool_ratios"],
                "point_count": expectations["held_out"]["point_count"],
            },
        },
        "preservation_locks": provenance["preservation_locks"],
        "fatal_guards": analysis["fatal_guards"],
        "conservation": analysis["conservation"],
        "onset": {
            "summary": analysis["onset_summary"],
            "configurations": analysis["observed_onsets"],
        },
        "held_out_bands": {
            "summary": analysis["held_out_band_summary"],
            "verdicts": analysis["held_out_band_verdicts"],
        },
        "decomposition_rows": analysis["decomposition_rows"],
        "segment_decompositions": analysis["segment_decompositions"],
        "total_delay_direction_scored": analysis["total_delay_direction_scored"],
        "prior_250_to_8000_monotonic_direction": analysis["prior_250_to_8000_monotonic_direction"],
        "closure": analysis["closure"],
    }


def render_report(result: dict[str, Any]) -> str:
    """Render the complete compact result as Markdown."""

    onset = result["onset"]
    bands = result["held_out_bands"]
    lines = [
        "# VLLM-41 scheduler queue-wait onset",
        "",
        "Status: onset identified below 250 requests/s; VLLM-41 closed.",
        "",
        "## Predicted versus observed onset",
        "",
        (
            "The surface-and-arrival-only model predicted a central first "
            "queue-dominated segment of 225 to 230 requests/s. Its frozen "
            "uncertainty admitted 220 to 225 or 225 to 230 requests/s. Every "
            "observed configuration instead begins at 210 to 220 requests/s."
        ),
        "",
        (
            "| Configuration | Predicted central | Frozen admitted segments | "
            "Observed first segment | Prior segments not queue-dominated | Band |"
        ),
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in onset["configurations"]:
        lines.append(
            f"| `{_configuration(row['configuration'])}` | 225 to 230 | "
            "220 to 225; 225 to 230 | "
            f"{row['observed_first_queue_dominated_segment'][0]} to "
            f"{row['observed_first_queue_dominated_segment'][1]} | "
            f"{row['preceding_non_queue_dominated_segments']} | "
            f"{'HELD' if row['inside_predicted_segment_band'] else 'MISSED'} |"
        )
    lines += [
        "",
        (
            "The observed onset is common to all six configurations, is strictly "
            "below 250 requests/s, and has five preceding non-queue-dominated "
            "segments per configuration. This satisfies the literal VLLM-41 "
            "closure rule. The earlier frozen knees of 1,056.6 and 2,113.2 "
            "requests/s remain refuted and are replaced by the observed 210 to "
            "220 requests/s bracket."
        ),
        "",
        "## Held-out band verdicts",
        "",
        (
            f"All {bands['summary']['queue_wait_held']} of "
            f"{bands['summary']['evaluated']} scheduler-wait bands held. Only "
            f"{bands['summary']['batch_service_held']} of "
            f"{bands['summary']['evaluated']} batching-service bands held, so "
            "14 of 30 joint component comparisons held and VLLM-42 is registered. "
            "No band was widened after observation."
        ),
        "",
        (
            "Displayed values are milliseconds. Frozen comparisons use the exact "
            "inclusive rational picosecond bounds retained in `results.json`."
        ),
        "",
        (
            "| Configuration | Load | Queue predicted | Queue band | Queue observed | "
            "Queue | Service predicted | Service band | Service observed | Service | "
            "Joint |"
        ),
        "|---|---:|---:|---:|---:|---|---:|---:|---:|---|---|",
    ]
    for row in bands["verdicts"]:
        wait_band = row["scheduler_queue_wait_band_ps"]
        service_band = row["batch_service_per_token_band_ps"]
        lines.append(
            f"| `{_configuration(row['configuration'])}` | "
            f"{row['offered_load_requests_per_second']} | "
            f"{_milliseconds(row['predicted_scheduler_queue_wait_ps'])} | "
            f"[{_milliseconds(wait_band['lower'])}, "
            f"{_milliseconds(wait_band['upper'])}] | "
            f"{_milliseconds(row['observed_scheduler_queue_wait_ps'])} | "
            f"{'HELD' if row['queue_wait_held'] else 'MISSED'} | "
            f"{_milliseconds(row['predicted_batch_service_per_token_ps'])} | "
            f"[{_milliseconds(service_band['lower'])}, "
            f"{_milliseconds(service_band['upper'])}] | "
            f"{_milliseconds(row['observed_batch_service_per_token_ps'])} | "
            f"{'HELD' if row['batch_service_held'] else 'MISSED'} | "
            f"{'HELD' if row['joint_held'] else 'MISSED'} |"
        )

    lines += [
        "",
        "## Per-cell decomposition",
        "",
        (
            "Batching service per token is reported separately from arrival-to-"
            "prefill wait and handoff-to-decode admission wait. Scheduler wait is "
            "only the sum of those two wait fields; provider service is excluded."
        ),
        "",
        "All displayed component values are milliseconds.",
        "",
        (
            "| Configuration | Load | Arrival to prefill | Handoff to decode | "
            "Scheduler wait | Batch service/token | Max prefill batch | Max decode "
            "batch |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["decomposition_rows"]:
        lines.append(
            f"| `{_configuration(row['cell'][:3])}` | {row['cell'][3]} | "
            f"{_milliseconds(row['mean_prefill_queue_ps'])} | "
            f"{_milliseconds(row['mean_decode_admission_wait_ps'])} | "
            f"{_milliseconds(row['mean_scheduler_queue_wait_ps'])} | "
            f"{_milliseconds(row['amortized_decode_batch_service_per_token_ps'])} "
            f"| {row['maximum_prefill_batch_size']} | "
            f"{row['maximum_decode_batch_size']} |"
        )

    lines += [
        "",
        "## Per-segment decomposition",
        "",
        (
            "The queue-dominated rule requires a positive scheduler-wait delta and "
            "a positive sum of scheduler-wait delta per four output tokens plus "
            "batch-service delta per token. Values below are milliseconds per token."
        ),
        "",
        (
            "| Configuration | Segment | Wait delta/token | Service delta/token | "
            "Component sum | Predicted | Observed | Prediction |"
        ),
        "|---|---:|---:|---:|---:|---|---|---|",
    ]
    for row in result["segment_decompositions"]:
        lines.append(
            f"| `{_configuration(row['configuration'])}` | "
            f"{row['from_load']} to {row['to_load']} | "
            f"{_signed_milliseconds(row['observed_scheduler_wait_delta_per_token_ps'])} "
            f"| {_signed_milliseconds(row['observed_batch_service_per_token_delta_ps'])} "
            f"| {_signed_milliseconds(row['observed_component_total_delta_per_token_ps'])} "
            f"| {'queue' if row['predicted_queue_dominated'] else 'not queue'} | "
            f"{'queue' if row['queue_dominated'] else 'not queue'} | "
            f"{'HELD' if row['prediction_held'] else 'MISSED'} |"
        )

    freeze = result["freeze"]
    model = freeze["queue_model"]
    isolated = model["isolated_onset_rate_band_requests_per_second"]
    lines += [
        "",
        "## Frozen derivation and refusal",
        "",
        (
            "- Lower offered-load ladder: 50, 100, 150, 175, 200, 210, 220, "
            "225, 230, 235, 240, 245 and 250 requests/s."
        ),
        (
            "- Six configurations: pool ratios 1:1, 1:2 and 2:1 crossed with "
            "8-token and 16-token prompts; 64 requests and four decode tokens per "
            "cell."
        ),
        (
            "- Held out: load 240 across the non-held-out ratios, plus the entire "
            "2:1 pool ratio. Their union contains 30 component comparisons."
        ),
        (
            "- The numerical queue model consumes only the imported batch-1 and "
            "batch-8 measured service/CV rows and deterministic interarrival "
            "times. Its shared virtual-clock simulation has no observed curve "
            "inputs and no fitted parameters."
        ),
        (
            "- The isolated central onset is "
            f"{float(_fraction(isolated['central'])):.6f} requests/s; the frozen "
            f"surface envelope spans {float(_fraction(isolated['lower'])):.6f} to "
            f"{float(_fraction(isolated['upper'])):.6f} requests/s. The envelope "
            f"is three times the maximum measured CV, "
            f"{model['surface_uncertainty']['envelope_ppm']:,} ppm."
        ),
        (
            "- The model and bands were committed before any VLLM-41 lower-ladder "
            "observation. The observed curve was never fitted, and the prior "
            "250-to-8,000 requests/s direction was preserved without rescoring."
        ),
        "",
        "## Imported surface and logged access",
        "",
        (
            f"The imported surface remains candidate evidence with calibration "
            f"claim `{str(freeze['surface']['calibration_claim']).lower()}`. It was "
            "read through the committed field-addressed reader with five passing "
            "ledger events; no whole record, DeepSeek row or held-out batch-32 row "
            "was decoded or captured."
        ),
        "",
        "| Batch | Service (ps) | CV (ppm) | Replays | Evidence | Key SHA-256 |",
        "|---:|---:|---:|---:|---|---|",
    ]
    for point in freeze["surface"]["selected_keys"]:
        lines.append(
            f"| {point['batch_size']} | {point['measured_service_ps']:,} | "
            f"{point['trimmed_coefficient_of_variation_ppm']:,} | "
            f"{point['replay_count']} | {point['evidence_class']} "
            f"{point['split']} | `{point['entry_key_sha256']}` |"
        )

    conservation = result["conservation"]
    lines += [
        "",
        "## Conservation and preservation",
        "",
        (
            f"- Cells: {conservation['cells']}; admissions, handoffs and terminals: "
            f"{conservation['admissions']:,} / {conservation['handoffs']:,} / "
            f"{conservation['terminals']:,}."
        ),
        (
            f"- Terminal decode tokens: {conservation['terminal_decode_tokens']:,}; "
            f"maximum TTFT decomposition residual: "
            f"{conservation['maximum_ttft_residual_ps']} ps."
        ),
        (
            "- Imported-surface candidate/no-calibration pricing held in every "
            "request record, and all pool-local identities remained unique."
        ),
        "",
        "| Preservation class | Files | Bytes | Manifest SHA-256 |",
        "|---|---:|---:|---|",
    ]
    for key, label in (
        ("prior_load_delay_lineage", "Prior VLLM-39/VLLM-40 lineage"),
        ("core51_one_request_control", "CORE-51 one-request control"),
        ("deterministic_concurrent_comparator", "Concurrent comparator"),
        ("scored_flagship_artifacts", "Scored flagship artifacts"),
    ):
        lock = result["preservation_locks"][key]
        lines.append(
            f"| {label} | {lock['artifact_count']} | {lock['total_bytes']:,} | "
            f"`{lock['manifest_sha256']}` |"
        )

    raw = result["raw_run"]
    runtime = result["runtime"]
    lines += [
        "",
        "## Run evidence",
        "",
        (f"- Scored HEAD: `{raw['run_head']}`; freeze commit: `{freeze['commit']}`."),
        (f"- Raw result: `{raw['path']}`, {raw['bytes']:,} bytes, SHA-256 `{raw['sha256']}`."),
        f"- Tracked compact result SHA-256: `{COMPACT_RESULT_SHA256}`.",
        (
            f"- Shard runner exit status: {raw['runner_exit_status']}; Python "
            f"{runtime['python']}; vLLM {runtime['vllm']}; offline mode "
            f"`{str(runtime['offline']).lower()}`."
        ),
        (
            f"- Expectations SHA-256: `{freeze['expectations_sha256']}`; access "
            f"ledger SHA-256: `{freeze['access_ledger_sha256']}`; surface SHA-256: "
            f"`{freeze['surface_sha256']}`."
        ),
        ("- The complete run was offline; no model weights or web content were downloaded."),
        (
            "- An initial infrastructure-only directory stopped before constructing "
            "a session because the historical helper import resolved to the new "
            "queue-model namespace. It remains retained and unscored. A separate "
            "sequential duplicate was stopped after the complete sharded result; "
            "its partial external directory and log remain retained and unscored. "
            "The published evidence is only the complete 78-cell sharded registry "
            "above."
        ),
        "",
        "## Registry movement",
        "",
        (
            "- **VLLM-41 closed**: all six configurations identify the common 210 "
            "to 220 requests/s first queue-dominated segment, strictly below 250 "
            "with five preceding non-queue-dominated segments."
        ),
        (
            "- **VLLM-42 registered**: 16 batching-service component bands missed "
            "without widening, despite all 30 scheduler-wait bands holding."
        ),
        ("- **VLLM-43 unused**: all six configurations share one resolved onset segment."),
        (
            "- The validated monotonic direction over 250 to 8,000 requests/s was "
            "preserved and not reopened."
        ),
    ]
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-result", required=True, type=Path)
    parser.add_argument("--result", default=DEFAULT_RESULT, type=Path)
    parser.add_argument("--report", default=DEFAULT_REPORT, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    raw, payload = _load_json(args.raw_result)
    result = compact_result(raw, payload)
    args.result.parent.mkdir(parents=True, exist_ok=True)
    with args.result.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    args.report.write_text(
        render_report(result),
        encoding="utf-8",
        newline="\n",
    )
    print(args.result.as_posix())
    print(args.report.as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
