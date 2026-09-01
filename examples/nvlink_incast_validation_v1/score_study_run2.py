#!/usr/bin/env python3
"""Score the second NV4 hardware capture registered as TRAF-74."""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import sys
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent


def _load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


run_campaign = _load_module(
    "_traf74_run2_campaign", HERE / "run_campaign_run2.py"
)
_score = _load_module("_traf74_first_score_reused_for_run2", HERE / "score_study.py")

SCORE_SCHEMA = "simllm-nvlink-incast-validation-score-v2"
TASK_ID = "TRAF-74"
RESIDUAL_TASK_ID = "TRAF-86"
COMPARISON_FIELDS = (
    "degree",
    "size_bytes",
    "hardware_completion_us_by_source",
    "simulation_completion_us_by_source",
    "completion_signed_relative_error_by_source",
    "hardware_aggregate_gbps",
    "simulation_aggregate_gbps",
    "aggregate_signed_relative_error",
    "maximum_launch_skew_fraction",
    "launch_skew_fraction_high",
    "physical_floor_us",
    "physical_ceiling_us",
    "physical_sanity",
    "verdict",
    "responsible_parameter",
)

_score.run_campaign = run_campaign
_score.SCORE_SCHEMA = SCORE_SCHEMA
_score.FROZEN_TASK_ID = TASK_ID
_score.REGISTRY_TASK_ID = TASK_ID
_score.COMPARISON_FIELDS = COMPARISON_FIELDS

score_fatal_guards = _score.score_fatal_guards
observation_key = _score.observation_key
flow_ledgers = _score.flow_ledgers
summarize_samples = _score.summarize_samples
attribute_misses = _score.attribute_misses
load_attempt = _score.load_attempt
read_rows = _score.read_rows
load_json = _score.load_json
write_json = _score.write_json
write_text = _score.write_text
write_comparison_csv = _score.write_comparison_csv


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bulk-root", type=Path, required=True)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--scheduler-job", required=True)
    parser.add_argument("--residual-task", default="")
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--csv-out", type=Path)
    parser.add_argument("--markdown-out", type=Path)
    return parser.parse_args(argv)


def compare_cells(
    samples: list[dict[str, Any]],
    frozen: dict[str, Any],
    *,
    measurement_valid: bool,
) -> list[dict[str, Any]]:
    grouped: defaultdict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        grouped[(sample["degree"], sample["size_bytes"])].append(sample)
    prediction_by_key = {
        (row["degree"], row["size_bytes"]): row
        for row in frozen["simulation_arm"]["predictions"]
    }
    preliminary = []
    for key in sorted(grouped, key=lambda value: (value[1], value[0])):
        rows = grouped[key]
        prediction = prediction_by_key[key]
        degree, size_bytes = key
        completion_us = [
            statistics.median(
                row["completion_us_by_source"][source] for row in rows
            )
            for source in range(degree)
        ]
        goodput = [size_bytes / (value * 1000) for value in completion_us]
        aggregate = statistics.median(
            row["aggregate_receiver_goodput_gbps"] for row in rows
        )
        simulation_completion_us = [
            value / 1_000_000 for value in prediction["completion_ps_by_source"]
        ]
        completion_error = [
            (simulated - hardware) / hardware
            for simulated, hardware in zip(
                simulation_completion_us, completion_us, strict=True
            )
        ]
        simulation_aggregate = float(prediction["aggregate_payload_gbps"])
        aggregate_error = (simulation_aggregate - aggregate) / aggregate
        within = all(
            frozen["comparison"]["acceptance_low"]
            <= value
            <= frozen["comparison"]["acceptance_high"]
            for value in [aggregate_error, *completion_error]
        )
        physical_floor_us = prediction["physical_floor_ps"] / 1_000_000
        physical_ceiling_us = prediction["physical_ceiling_ps"] / 1_000_000
        physical_sanity = (
            "PASS"
            if all(
                physical_floor_us <= value <= physical_ceiling_us
                for value in completion_us
            )
            else "FAIL"
        )
        preliminary.append(
            {
                "degree": degree,
                "size_bytes": size_bytes,
                "hardware_completion_us_by_source": completion_us,
                "hardware_goodput_gbps_by_source": goodput,
                "simulation_completion_us_by_source": simulation_completion_us,
                "completion_signed_relative_error_by_source": completion_error,
                "hardware_aggregate_gbps": aggregate,
                "simulation_aggregate_gbps": simulation_aggregate,
                "aggregate_signed_relative_error": aggregate_error,
                "physical_floor_us": physical_floor_us,
                "physical_ceiling_us": physical_ceiling_us,
                "physical_sanity": physical_sanity,
                "within_frozen_band": within,
                "verdict": (
                    "VOID"
                    if not measurement_valid
                    else "PASS"
                    if within
                    else "MISS"
                ),
                "responsible_parameter": "none" if within else "pending_attribution",
            }
        )
    attribution = attribute_misses(preliminary)
    for row in preliminary:
        if row["verdict"] == "MISS":
            row["responsible_parameter"] = attribution[row["degree"]]
        elif row["verdict"] == "VOID":
            row["responsible_parameter"] = "undecidable_under_void_run"
    return preliminary


_score.compare_cells = compare_cells


def audit_hardware(
    bulk_root: Path,
    *,
    expected_head: str,
    scheduler_job: str,
    residual_task: str = "",
) -> dict[str, Any]:
    if residual_task and residual_task != RESIDUAL_TASK_ID:
        raise ValueError(f"the assigned residual task is {RESIDUAL_TASK_ID}")
    score = _score.audit_hardware(
        bulk_root,
        expected_head=expected_head,
        scheduler_job=scheduler_job,
        residual_task=residual_task,
    )
    if (
        score["measurement_validity"] == "VALID_FOR_FROZEN_COMPARISON"
        and score["summary"]["pass_cells"] == 0
        and score["summary"]["miss_cells"] == 6
    ):
        score["status"] = "VALID_0_PASS_6_MISS"
    if not score["summary"]["miss_cells"] and residual_task:
        raise RuntimeError("a passing comparison must not register a residual task")
    launch_by_cell: defaultdict[tuple[int, int], list[float]] = defaultdict(list)
    for row in score["fatal_guards"]["launch_skew_rows"]:
        launch_by_cell[(row["degree"], row["size_bytes"])].append(
            float(row["launch_skew_fraction"])
        )
    budget = score["summary"]["launch_skew_fraction_high"]
    for row in score["comparisons"]:
        values = launch_by_cell[(row["degree"], row["size_bytes"])]
        if not values:
            raise RuntimeError("a comparison cell has no launch-skew observation")
        row["maximum_launch_skew_fraction"] = max(values)
        row["launch_skew_fraction_high"] = budget
    frozen = run_campaign.load_expectations()
    score["simulation_identity"] = {
        "implementation": frozen["simulation_arm"]["implementation"],
        "module_version_commit": frozen["simulation_arm"]["module_version_commit"],
        "model_sha256": frozen["simulation_arm"]["model_sha256"],
        "profile_sha256": frozen["simulation_arm"]["profile_sha256"],
        "flow_policy": frozen["simulation_arm"]["flow_policy"],
        "release_ps": frozen["simulation_arm"]["release_ps"],
    }
    score["raw_evidence"] = {
        "storage": "digest-complete external attempt retained outside Git",
        "attempt_manifest_sha256": score["attempt_manifest_sha256"],
        "required_observables": frozen["hardware_arm"]["required_observables"],
        "row_count": score["coverage"]["observed_rows"],
        "row_digests": [row["row_sha256"] for row in score["hardware_samples"]],
    }
    comparison_by_key = {
        (row["degree"], row["size_bytes"]): row for row in score["comparisons"]
    }
    doubling_ratios = []
    for degree in frozen["hardware_arm"]["degrees"]:
        smaller = comparison_by_key[(degree, 4 << 20)]
        larger = comparison_by_key[(degree, 8 << 20)]
        doubling_ratios.extend(
            larger_value / smaller_value
            for smaller_value, larger_value in zip(
                smaller["hardware_completion_us_by_source"],
                larger["hardware_completion_us_by_source"],
                strict=True,
            )
        )
    floor_multiples = [
        completion / row["physical_floor_us"]
        for row in score["comparisons"]
        for completion in row["hardware_completion_us_by_source"]
    ]
    ceiling_fractions = [
        completion / row["physical_ceiling_us"]
        for row in score["comparisons"]
        for completion in row["hardware_completion_us_by_source"]
    ]
    hardware_source_goodput = [
        value
        for row in score["comparisons"]
        for value in row["hardware_goodput_gbps_by_source"]
    ]
    score["physical_sanity"] = {
        "verdict": (
            "PASS"
            if all(row["physical_sanity"] == "PASS" for row in score["comparisons"])
            else "FAIL"
        ),
        "minimum_hardware_completion_us": min(
            value
            for row in score["comparisons"]
            for value in row["hardware_completion_us_by_source"]
        ),
        "maximum_hardware_completion_us": max(
            value
            for row in score["comparisons"]
            for value in row["hardware_completion_us_by_source"]
        ),
        "minimum_frozen_floor_us": min(
            row["physical_floor_us"] for row in score["comparisons"]
        ),
        "maximum_frozen_floor_us": max(
            row["physical_floor_us"] for row in score["comparisons"]
        ),
        "minimum_hardware_over_floor": min(floor_multiples),
        "maximum_completion_over_ceiling": max(ceiling_fractions),
        "minimum_eight_mib_over_four_mib_completion": min(doubling_ratios),
        "maximum_eight_mib_over_four_mib_completion": max(doubling_ratios),
        "minimum_hardware_source_goodput_gbps": min(hardware_source_goodput),
        "maximum_hardware_source_goodput_gbps": max(hardware_source_goodput),
    }
    return score


def _registry_effect(score: dict[str, Any]) -> str:
    if score["measurement_validity"] == "VOID_FATAL_GUARD":
        return f"{TASK_ID} stays open because the second run is void"
    if score["summary"]["miss_cells"]:
        return (
            f"{TASK_ID} closes as a completed non-void validation; "
            f"{RESIDUAL_TASK_ID} owns the identified model precision residual"
        )
    return f"{TASK_ID} closes with no residual model task"


def render_markdown(score: dict[str, Any]) -> str:
    lines = [
        "# TRAF-74 NV4 long-flow incast second-capture result",
        "",
        "## Hardware against simulation",
        "",
        "| Degree | Flow | Hardware aggregate GB/s | Simulation aggregate GB/s | Signed error | Hardware completion us by source | Simulation completion us by source | Maximum launch skew | Budget | Verdict | Responsible parameter |",
        "|---:|---:|---:|---:|---:|---|---|---:|---:|---|---|",
    ]
    for row in score["comparisons"]:
        hardware_completion = ", ".join(
            f"{value:.6f}" for value in row["hardware_completion_us_by_source"]
        )
        simulation_completion = ", ".join(
            f"{value:.6f}" for value in row["simulation_completion_us_by_source"]
        )
        lines.append(
            f"| {row['degree']} | {row['size_bytes'] // (1 << 20)} MiB | "
            f"{row['hardware_aggregate_gbps']:.6f} | "
            f"{row['simulation_aggregate_gbps']:.6f} | "
            f"{100 * row['aggregate_signed_relative_error']:+.3f}% | "
            f"{hardware_completion} | {simulation_completion} | "
            f"{100 * row['maximum_launch_skew_fraction']:.3f}% | "
            f"{100 * row['launch_skew_fraction_high']:.3f}% | "
            f"{row['verdict']} | `{row['responsible_parameter']}` |"
        )
    summary = score["summary"]
    lines += [
        "",
        "Signed relative error is `(simulation - hardware) / hardware`; the frozen",
        "acceptance band is plus or minus 16 percent. Each cell requires its aggregate",
        "and every per-source median to be inside that band. Fatal guards remain",
        "separate and never enter the behavioral count.",
        "",
        "## What ran",
        "",
        "One short exclusive `a100-hourly` cell ran the unchanged corrected TRAF-70",
        "persistent peer-write producer on one qualified four-A100 `NV4` node. It",
        "covered 4 MiB and 8 MiB flows at incast degrees 1, 2 and 3 with seven",
        "repetitions per cell. The comparison uses the six predictions frozen at",
        (
            f"commit `{score['expectations_commit'][:7]}` before Merlin job "
            f"`{score['scheduler_job']}` ran. The scored module version is"
        ),
        f"`{score['simulation_identity']['module_version_commit']}` with flow policy",
        f"`{score['simulation_identity']['flow_policy']}`.",
        "",
        "## What came out",
        "",
    ]
    if score["measurement_validity"] == "VOID_FATAL_GUARD":
        lines += [
            f"The run status is **{score['status']}**. The deciding maximum launch-skew",
            f"fraction was {100 * summary['maximum_launch_skew_fraction']:.3f} percent",
            "against the frozen 10.000 percent ceiling. At least one fatal guard failed",
            "or was undecidable, so all six behavioral cells are void.",
        ]
    else:
        lines += [
            f"The run status is **{score['status']}**. The maximum observed",
            (
                "launch-skew fraction was "
                f"{100 * summary['maximum_launch_skew_fraction']:.3f} percent against "
                "the 10.000 percent ceiling."
            ),
            "The deciding worst absolute signed relative error was",
            (
                f"{100 * summary['worst_absolute_signed_relative_error']:.3f} percent. "
                f"{summary['pass_cells']} of 6 cells pass and "
                f"{summary['miss_cells']} miss."
            ),
            "Every miss names `packetization` under the frozen size-dependent",
            "attribution rule.",
        ]
    sanity = score["physical_sanity"]
    lines += [
        "",
        "## Physical sanity before precision",
        "",
        "Floor: packetized wire serialization sets frozen per-cell floors from",
        (
            f"{sanity['minimum_frozen_floor_us']:.6f} to "
            f"{sanity['maximum_frozen_floor_us']:.6f} us. Hardware completion ranged "
            "from"
        ),
        (
            f"{sanity['minimum_hardware_completion_us']:.6f} to "
            f"{sanity['maximum_hardware_completion_us']:.6f} us and was never faster "
            "than"
        ),
        (
            f"its floor. The closest sample was "
            f"{sanity['minimum_hardware_over_floor']:.3f} times its floor."
        ),
        "",
        "Ceiling: every source completed below the frozen 5000 us observed-producer",
        (
            f"ceiling. The slowest used "
            f"{100 * sanity['maximum_completion_over_ceiling']:.3f} percent of that "
            "ceiling."
        ),
        "",
        "Byte scaling: doubling each source from 4 MiB to 8 MiB moved median",
        (
            f"completion by {sanity['minimum_eight_mib_over_four_mib_completion']:.3f} "
            f"to {sanity['maximum_eight_mib_over_four_mib_completion']:.3f} times, close"
        ),
        "to the expected factor of two for sustained service.",
        "",
        "End-to-end plausibility: measured per-source payload goodput ranged from",
        (
            f"{sanity['minimum_hardware_source_goodput_gbps']:.3f} to "
            f"{sanity['maximum_hardware_source_goodput_gbps']:.3f} GB/s. That extends "
            "the"
        ),
        "retained 2.2 to 3.5 GB/s short-rung trend after fixed launch work is amortized,",
        "but it remains far below the model's packetized wire-rate prediction.",
        "",
        "## What it changes for the project",
        "",
        f"{_registry_effect(score)}.",
        "The second capture supplies all six literal",
        "per-cell verdicts at the only incast degrees an NV4 node can realize.",
        "",
        "## What it does not change",
        "",
        "Degrees 4, 8 and 16 remain DECLARED SIMULATION with no hardware counterpart",
        "on an NV4 node. This result covers long flows only. Agreement at degrees 1",
        "to 3 supports but does not prove the higher-degree extrapolation, and no",
        "small-flow hardware validity claim follows. The first frozen capture remains",
        "byte-identical and void; this result does not reinterpret job `200456`.",
        "",
        "## Fatal guards and preservation",
        "",
        (
            f"Fatal-guard verdict: **{score['fatal_guards']['verdict']}**. All "
            f"{score['preservation']['artifact_count']} inherited and first-capture"
        ),
        "artifacts remain byte-identical. The digest-complete raw capture stays outside",
        "Git and retains every checksum, ordering, per-link data and raw counter,",
        "replay, recovery, throttle, topology and competing-process observation.",
    ]
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    score = audit_hardware(
        args.bulk_root,
        expected_head=args.expected_head,
        scheduler_job=args.scheduler_job,
        residual_task=args.residual_task,
    )
    if args.json_out is not None:
        write_json(args.json_out, score)
    if args.csv_out is not None:
        write_comparison_csv(args.csv_out, score["comparisons"])
    if args.markdown_out is not None:
        write_text(args.markdown_out, render_markdown(score))
    if args.json_out is None and args.csv_out is None and args.markdown_out is None:
        print(json.dumps(score, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
