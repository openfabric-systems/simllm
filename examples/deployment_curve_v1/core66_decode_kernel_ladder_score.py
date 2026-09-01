#!/usr/bin/env python3
"""Score the frozen CORE-66 decode graph from an Nsight Systems export."""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import re
import sqlite3
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

LABEL = re.compile(
    r"^core66-kladder\|repeat=(?P<repeat>\d+)\|trial=(?P<trial>\d+)"
    r"\|call=(?P<call>\d+)$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sqlite", type=Path, required=True)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--expectations", type=Path, required=True)
    parser.add_argument("--weights-before", type=Path, required=True)
    parser.add_argument("--weights-after", type=Path, required=True)
    parser.add_argument("--process-status", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def median(values: list[int]) -> int:
    if not values:
        raise RuntimeError("cannot take a median of an empty graph sample")
    return round(statistics.median(values))


def interval_union_ps(rows: list[dict[str, Any]]) -> int:
    intervals = sorted((row["start_ns"], row["end_ns"]) for row in rows)
    if not intervals:
        return 0
    total_ns = 0
    start, end = intervals[0]
    for next_start, next_end in intervals[1:]:
        if next_start <= end:
            end = max(end, next_end)
        else:
            total_ns += end - start
            start, end = next_start, next_end
    return (total_ns + end - start) * 1000


def native_calls(sqlite_path: Path) -> dict[tuple[int, int, int], list[dict[str, Any]]]:
    connection = sqlite3.connect(sqlite_path)
    ranges = connection.execute(
        """
        SELECT n.start, n.end, n.globalTid, COALESCE(n.text, labels.value)
        FROM NVTX_EVENTS AS n
        LEFT JOIN StringIds AS labels ON n.textId = labels.id
        WHERE COALESCE(n.text, labels.value) LIKE 'core66-kladder|%|call=%'
        ORDER BY n.globalTid, n.start
        """
    ).fetchall()
    launches_by_thread: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
    for start_ns, end_ns, global_tid, correlation_id in connection.execute(
        """
        SELECT runtime.start, runtime.end, runtime.globalTid, runtime.correlationId
        FROM CUPTI_ACTIVITY_KIND_RUNTIME AS runtime
        JOIN StringIds AS names ON names.id = runtime.nameId
        WHERE names.value LIKE 'cudaGraphLaunch%'
          AND runtime.correlationId IS NOT NULL
        ORDER BY runtime.globalTid, runtime.start
        """
    ):
        launches_by_thread[global_tid].append((start_ns, end_ns, correlation_id))

    correlation_labels: dict[int, str] = {}
    for range_start, range_end, global_tid, label in ranges:
        launches = launches_by_thread[global_tid]
        starts = [row[0] for row in launches]
        index = bisect.bisect_left(starts, range_start)
        matches = []
        while index < len(launches) and launches[index][0] <= range_end:
            launch = launches[index]
            if launch[0] >= range_start and launch[1] <= range_end:
                matches.append(launch)
            index += 1
        if len(matches) != 1:
            raise RuntimeError(
                f"NVTX graph-call range maps to {len(matches)} launches: {label}"
            )
        correlation_labels[matches[0][2]] = label

    calls: dict[tuple[int, int, int], list[dict[str, Any]]] = defaultdict(list)
    for correlation_id, name, start_ns, end_ns, stream_id in connection.execute(
        """
        SELECT
            kernel.correlationId,
            names.value,
            kernel.start,
            kernel.end,
            kernel.streamId
        FROM CUPTI_ACTIVITY_KIND_KERNEL AS kernel
        JOIN StringIds AS names ON names.id = kernel.demangledName
        WHERE kernel.correlationId IS NOT NULL
        ORDER BY kernel.start, kernel.gridId
        """
    ):
        label = correlation_labels.get(correlation_id)
        if label is None:
            continue
        match = LABEL.match(label)
        if match is None:
            continue
        key = (int(match["repeat"]), int(match["trial"]), int(match["call"]))
        calls[key].append(
            {
                "name": name,
                "start_ns": start_ns,
                "end_ns": end_ns,
                "duration_ps": (end_ns - start_ns) * 1000,
                "stream_id": stream_id,
            }
        )
    connection.close()
    return calls


def summarize_calls(
    capture: dict[str, Any],
    expectations: dict[str, Any],
    calls: dict[tuple[int, int, int], list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    trials = expectations["confirmatory_capture"]["measurement"]["trials"]
    layer_types = capture["installed"]["layer_mlp_types"]
    segment_bounds = []
    offset = 2
    for index, layer_type in enumerate(layer_types):
        kernel_count = 34 if "MoE" in layer_type else 21
        segment_bounds.append((f"layer_{index}", offset, offset + kernel_count))
        offset += kernel_count
    expected_kernel_count = offset + 1
    repeat_summaries = []
    for timing in capture["measurement"]["repeat_rows"]:
        repeat = timing["repeat"]
        cold_services = []
        resident_services = []
        cold_unions = []
        resident_unions = []
        kernel_counts: set[int] = set()
        streams: set[int] = set()
        names: set[str] = set()
        cold_segments: dict[str, list[int]] = defaultdict(list)
        resident_segments: dict[str, list[int]] = defaultdict(list)
        for trial in range(trials):
            for call_index in range(repeat):
                kernels = calls.get((repeat, trial, call_index), [])
                if not kernels:
                    raise RuntimeError(
                        "native graph call is absent: "
                        f"repeat={repeat}, trial={trial}, call={call_index}"
                    )
                service_ps = sum(row["duration_ps"] for row in kernels)
                union_ps = interval_union_ps(kernels)
                kernel_counts.add(len(kernels))
                streams.update(row["stream_id"] for row in kernels)
                names.update(row["name"] for row in kernels)
                target_services = cold_services if call_index == 0 else resident_services
                target_unions = cold_unions if call_index == 0 else resident_unions
                target_services.append(service_ps)
                target_unions.append(union_ps)
                if len(kernels) == expected_kernel_count:
                    segments = {"root_prefix": kernels[0:2]}
                    segments.update(
                        {
                            segment: kernels[start:end]
                            for segment, start, end in segment_bounds
                        }
                    )
                    segments["root_suffix"] = kernels[offset : offset + 1]
                    target_segments = (
                        cold_segments if call_index == 0 else resident_segments
                    )
                    for segment, rows in segments.items():
                        target_segments[segment].append(
                            sum(row["duration_ps"] for row in rows)
                        )
        repeat_summaries.append(
            {
                "repeat": repeat,
                "raw_service_ps": round(timing["raw_median_ps"] / repeat),
                "subtracted_service_ps": timing["subtracted_service_ps"],
                "cold_native_service_ps": median(cold_services),
                "resident_native_service_ps": median(resident_services),
                "cold_native_union_ps": median(cold_unions),
                "resident_native_union_ps": median(resident_unions),
                "kernel_count_values": sorted(kernel_counts),
                "expected_kernel_count": expected_kernel_count,
                "native_stream_ids": sorted(streams),
                "native_stream_count": len(streams),
                "cold_segment_service_ps": {
                    segment: median(values)
                    for segment, values in sorted(cold_segments.items())
                },
                "resident_segment_service_ps": {
                    segment: median(values)
                    for segment, values in sorted(resident_segments.items())
                },
                "has_dense_gate_up": any(
                    "36864, (unsigned int)7168" in name for name in names
                ),
                "has_dense_down": any(
                    "7168, (unsigned int)18432" in name for name in names
                ),
                "has_moe": any("fused_moe_kernel" in name for name in names),
                "has_mla": any("FlashAttnFwdSm90" in name for name in names),
                "has_embedding": any(
                    "vectorized_gather_kernel" in name for name in names
                ),
            }
        )
    return repeat_summaries


def preservation_status(repo_root: Path, expectations: dict[str, Any]) -> list[dict[str, Any]]:
    manifest_path = repo_root / "examples/deployment_curve_v1" / expectations[
        "preservation"
    ]["manifest"]
    rows = []
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", maxsplit=1)
        actual = hashlib.sha256((repo_root / relative).read_bytes()).hexdigest()
        rows.append(
            {
                "path": relative,
                "expected_sha256": expected,
                "actual_sha256": actual,
                "pass": actual == expected,
            }
        )
    return rows


def exact_marker(marker: dict[str, Any]) -> bool:
    active = marker.get("active", {})
    scheduler = marker.get("scheduler", {})
    return (
        active
        == {"enabled": True, "batch_size": 32, "remote_kv_tokens": 2000}
        and scheduler.get("is_exact_decode") is True
        and scheduler.get("num_requests") == 32
        and len(scheduler.get("request_ids", [])) == 32
        and scheduler.get("num_computed_tokens") == [2000] * 32
        and len(scheduler.get("num_output_tokens", [])) == 32
        and all(value > 0 for value in scheduler.get("num_output_tokens", []))
    )


def score(
    capture: dict[str, Any],
    expectations: dict[str, Any],
    repeat_summaries: list[dict[str, Any]],
    *,
    process_exit_zero: bool,
    weights_empty_and_identical: bool,
    preservation: list[dict[str, Any]],
    commit: str,
) -> dict[str, Any]:
    authority = max(repeat_summaries, key=lambda row: row["repeat"])
    cold_values = [row["cold_native_service_ps"] for row in repeat_summaries]
    capture_freeze = expectations["confirmatory_capture"]
    relation_freeze = capture_freeze["scored_relations"]
    predicted_ps = capture_freeze["composition"]["predicted_graph_service_ps"]
    residual_ps = authority["cold_native_service_ps"] - predicted_ps
    repeat_fraction = (max(cold_values) - min(cold_values)) / min(cold_values)
    layer_types = capture["installed"]["layer_mlp_types"]
    expected_kernel_count = authority["expected_kernel_count"]

    fatal_guards = {
        "process_exit_zero": process_exit_zero,
        "empty_and_identical_weight_snapshots": weights_empty_and_identical,
        "exact_batch_32_kv_2000_marker": exact_marker(capture["marker"]),
        "three_dense_then_one_moe": (
            len(layer_types) == 4
            and all("MoE" not in value for value in layer_types[:3])
            and "MoE" in layer_types[3]
        ),
        "output_shape_32_by_7168_bf16": capture["measurement"][
            "capture_output_shape"
        ]
        == {"shape": [32, 7168], "dtype": "torch.bfloat16"},
        "one_native_cuda_stream": all(
            row["native_stream_count"] == 1 for row in repeat_summaries
        ),
        "stable_100_node_graph": (
            expected_kernel_count == 100
            and all(
                row["kernel_count_values"] == [expected_kernel_count]
                for row in repeat_summaries
            )
        ),
        "dense_moe_mla_and_embedding_identities": all(
            row["has_dense_gate_up"]
            and row["has_dense_down"]
            and row["has_moe"]
            and row["has_mla"]
            and row["has_embedding"]
            for row in repeat_summaries
        ),
    }
    behavioral_relations = {
        "composition_residual": abs(residual_ps)
        <= relation_freeze["composition_residual_absolute_ps_at_most"],
        "repeat_stability": repeat_fraction
        <= float(relation_freeze["native_repeat_difference_fraction_at_most"]),
        "physical_bounds": relation_freeze["physical_service_floor_ps"]
        <= authority["cold_native_service_ps"]
        <= relation_freeze["physical_service_ceiling_ps"],
    }
    preservation_pass = (
        len(preservation) == expectations["preservation"]["required_count"]
        and all(row["pass"] for row in preservation)
    )
    fatal_pass = all(fatal_guards.values()) and preservation_pass
    if not fatal_pass:
        status = "VOID"
    elif all(behavioral_relations.values()):
        status = "PASS"
    else:
        status = "FAIL"

    return {
        "schema": "simllm-deployment-curve-core66-decode-analysis-v1",
        "task": "CORE-66",
        "status": status,
        "chronology": {
            "expectations_commit": "7919f7b",
            "confirmatory_commit": commit,
            "scratch_not_published": True,
        },
        "run_configuration": {
            "framework": capture["framework"],
            "model": capture["model"],
            "revision": capture["revision"],
            "shape": capture["shape"],
            "evidence": capture["evidence"],
        },
        "prediction": {
            "graph_service_ps": predicted_ps,
            "tolerance_ps": relation_freeze[
                "composition_residual_absolute_ps_at_most"
            ],
        },
        "measurement": {
            "authority_repeat": authority["repeat"],
            "measured_graph_service_ps": authority["cold_native_service_ps"],
            "composition_residual_ps": residual_ps,
            "composition_residual_fraction": residual_ps / predicted_ps,
            "repeat_stability_fraction": repeat_fraction,
            "raw_service_ps": authority["raw_service_ps"],
            "subtracted_service_ps": authority["subtracted_service_ps"],
            "subtraction_method": capture["measurement"]["subtraction_method"],
            "repeat_summaries": repeat_summaries,
        },
        "evidence_classes": {
            "behavioral_relations": behavioral_relations,
            "fatal_structural_guards": fatal_guards,
            "preservation_locks": {
                "pass": preservation_pass,
                "rows": preservation,
            },
            "run_configuration_is_unscored": True,
        },
        "runtime_substitution": expectations["runtime_substitution"],
    }


def main() -> int:
    args = parse_args()
    capture = json.loads(args.capture.read_text(encoding="utf-8"))
    expectations = json.loads(args.expectations.read_text(encoding="utf-8"))
    summaries = summarize_calls(capture, expectations, native_calls(args.sqlite))
    before = args.weights_before.read_bytes()
    after = args.weights_after.read_bytes()
    result = score(
        capture,
        expectations,
        summaries,
        process_exit_zero=args.process_status.read_text(encoding="utf-8").strip()
        == "0",
        weights_empty_and_identical=before == after == b"",
        preservation=preservation_status(args.repo_root, expectations),
        commit=args.commit,
    )
    write_json(args.output, result)
    print(result["status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
