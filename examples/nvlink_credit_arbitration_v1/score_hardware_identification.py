#!/usr/bin/env python3
"""Score the three frozen TRAF-73 NV4 hardware families literally."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import re
import statistics
import subprocess
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
TRAF70_SCORE_PATH = HERE.parent / "a100_nvlink_packet_v2/score_hardware.py"
RUNNER_PATH = HERE / "run_hardware_campaign.py"
SCORE_SCHEMA = "simllm-nvlink-credit-identification-score-v1"
RESULT_JSON = HERE / "hardware_identification.json"
RESULT_MARKDOWN = HERE / "RESULTS_HARDWARE.md"
OUTSTANDING_CSV = HERE / "aggregate_outstanding_bytes.csv"


def _load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


runner = _load_module("_traf73_hardware_campaign", RUNNER_PATH)
_previous_run_study = sys.modules.get("run_study")
sys.modules["run_study"] = runner.traf70_run
try:
    traf70_score = _load_module("_traf73_traf70_score", TRAF70_SCORE_PATH)
finally:
    if _previous_run_study is None:
        del sys.modules["run_study"]
    else:
        sys.modules["run_study"] = _previous_run_study


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bulk-root", type=Path, required=True)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument(
        "--scheduler-job",
        action="append",
        default=[],
        metavar="FAMILY=JOB",
        help="repeat for h1, h2 and h3",
    )
    parser.add_argument("--json-out", type=Path, default=RESULT_JSON)
    parser.add_argument("--markdown-out", type=Path, default=RESULT_MARKDOWN)
    parser.add_argument("--outstanding-csv-out", type=Path, default=OUTSTANDING_CSV)
    return parser.parse_args(argv)


def parse_scheduler_jobs(values: Sequence[str]) -> dict[str, str]:
    jobs = {}
    for value in values:
        match = re.fullmatch(r"(h1|h2|h3)=([0-9]+)", value)
        if match is None:
            raise ValueError("scheduler jobs use FAMILY=JOB with h1, h2 or h3")
        family, job = match.groups()
        if family in jobs:
            raise ValueError(f"duplicate scheduler job for {family}")
        jobs[family] = job
    if set(jobs) != {"h1", "h2", "h3"}:
        raise ValueError("scheduler jobs are required for h1, h2 and h3")
    return jobs


def load_json(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8", newline="") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object in {path.name}")
    return value


def read_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with open(path, encoding="utf-8", newline="") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise RuntimeError(f"invalid result JSON at line {line_number}") from error
            if row.get("schema") != runner.OBSERVATION_SCHEMA:
                raise RuntimeError("unexpected TRAF-73 observation schema")
            if row.get("mode") != "hardware" or row.get("measurement_claim") != "unscored":
                raise RuntimeError("a TRAF-73 row is not unscored hardware evidence")
            rows.append(row)
    return rows


def find_attempt(
    bulk_root: Path,
    *,
    family: str,
    expected_head: str,
    scheduler_job: str,
) -> Path:
    matches = []
    root = runner.cell_root(bulk_root, family)
    for attempt in sorted(root.glob("attempt-*")):
        if not runner.verify_attempt(attempt):
            continue
        plan = load_json(attempt / "plan.json")
        environment = load_json(attempt / "environment.json")
        summary = load_json(attempt / "summary.json")
        if plan.get("family") != family or plan.get("mode") != "hardware":
            continue
        if plan.get("expected_head") != expected_head:
            continue
        if plan.get("implementation_sha256") != runner.implementation_sha256():
            continue
        if plan.get("producer_derived_source_sha256") != runner.DERIVED_PRODUCER_SHA256:
            continue
        if environment.get("slurm_job_id") != scheduler_job:
            continue
        if environment.get("slurm_partition") != "a100-hourly":
            continue
        if summary.get("status") != "hardware_unscored":
            continue
        matches.append(attempt)
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one digest-complete {family} attempt, found {len(matches)}"
        )
    return matches[0]


def _fit_line(xs: Sequence[float], ys: Sequence[float]) -> tuple[float, float]:
    if len(xs) != len(ys) or len(xs) < 2:
        raise ValueError("a line fit requires two aligned samples")
    x_mean = statistics.fmean(xs)
    y_mean = statistics.fmean(ys)
    denominator = sum((value - x_mean) ** 2 for value in xs)
    if denominator == 0:
        raise ValueError("a line fit requires distinct x values")
    slope = sum(
        (x - x_mean) * (y - y_mean) for x, y in zip(xs, ys, strict=True)
    ) / denominator
    return y_mean - slope * x_mean, slope


def _median_absolute_deviation(values: Sequence[float]) -> float:
    median = statistics.median(values)
    return statistics.median(abs(value - median) for value in values)


def detect_knee(
    sizes: Sequence[int],
    completion_us: Sequence[float],
) -> dict[str, object]:
    """Apply the frozen positive five-MAD, three-point persistent break rule."""

    if len(sizes) != len(completion_us) or len(sizes) < 9:
        raise ValueError("the knee fit requires the full aligned sweep")
    candidates = []
    x_values = [float(value) for value in sizes]
    for index in range(5, len(sizes) - 3):
        below_x = x_values[:index]
        below_y = [float(value) for value in completion_us[:index]]
        above_x = x_values[index:]
        above_y = [float(value) for value in completion_us[index:]]
        below_intercept, below_slope = _fit_line(below_x, below_y)
        above_intercept, above_slope = _fit_line(above_x, above_y)
        residuals = [
            observed - (below_intercept + below_slope * size)
            for size, observed in zip(below_x, below_y, strict=True)
        ]
        mad = _median_absolute_deviation(residuals)
        threshold = 5.0 * mad
        numeric_floor = math.ulp(max(abs(value) for value in below_y)) * 32.0
        persistent = [
            float(completion_us[position])
            - (below_intercept + below_slope * float(sizes[position]))
            for position in range(index, index + 3)
        ]
        jump_us = (
            above_intercept + above_slope * float(sizes[index])
            - below_intercept
            - below_slope * float(sizes[index])
        )
        if jump_us > numeric_floor and all(
            value > max(threshold, numeric_floor) for value in persistent
        ):
            candidates.append(
                {
                    "index": index,
                    "payload_bytes": sizes[index],
                    "jump_us": jump_us,
                    "return_delay_ps": jump_us * 1_000_000.0,
                    "below_mad_us": mad,
                    "threshold_us": threshold,
                    "numeric_floor_us": numeric_floor,
                    "persistent_positive_residual_us": persistent,
                }
            )
    if not candidates:
        return {
            "status": "NO_REPEATED_PERSISTENT_BREAK",
            "payload_bytes": None,
            "return_delay_ps": None,
            "candidate_count": 0,
        }
    selected = min(candidates, key=lambda row: int(row["index"]))
    return {
        "status": "REPEATED_PERSISTENT_BREAK",
        "payload_bytes": selected["payload_bytes"],
        "return_delay_ps": selected["return_delay_ps"],
        "candidate_count": len(candidates),
        "selected": selected,
    }


def detect_repeated_knee(
    sizes: Sequence[int],
    completion_us_by_size: Sequence[Sequence[float]],
) -> dict[str, object]:
    """Require the same first persistent break on every device-timed pass."""

    if len(sizes) != len(completion_us_by_size):
        raise ValueError("the repeated knee input differs from the size sweep")
    repetition_counts = {len(values) for values in completion_us_by_size}
    if len(repetition_counts) != 1 or not repetition_counts or 0 in repetition_counts:
        raise ValueError("the repeated knee input has incomplete repetitions")
    repetition_count = repetition_counts.pop()
    medians = [statistics.median(values) for values in completion_us_by_size]
    median_fit = detect_knee(sizes, medians)
    pass_fits = [
        detect_knee(
            sizes,
            [values[repetition] for values in completion_us_by_size],
        )
        for repetition in range(repetition_count)
    ]
    repeated_payloads = [fit["payload_bytes"] for fit in pass_fits]
    selected_payload = median_fit["payload_bytes"]
    pass_count = sum(value == selected_payload for value in repeated_payloads)
    if selected_payload is None or pass_count != repetition_count:
        return {
            "status": "NO_REPEATED_PERSISTENT_BREAK",
            "payload_bytes": None,
            "return_delay_ps": None,
            "candidate_count": median_fit["candidate_count"],
            "median_fit": median_fit,
            "repetition_count": repetition_count,
            "matching_repetition_count": pass_count,
        }
    return_delays = [
        float(fit["return_delay_ps"])
        for fit in pass_fits
        if fit["return_delay_ps"] is not None
    ]
    return {
        **median_fit,
        "return_delay_ps": statistics.median(return_delays),
        "median_return_delay_ps": median_fit["return_delay_ps"],
        "repetition_count": repetition_count,
        "matching_repetition_count": pass_count,
    }


def _size_order(frozen: Mapping[str, Any], family: str) -> list[int]:
    key = "h1_credit_window_and_return" if family == "h1" else "h2_pool_scope"
    return [int(value) for value in frozen[key]["payload_sizes_bytes"]]


def _latency_samples(row: Mapping[str, Any]) -> list[list[float]]:
    values = row.get("traf73", {}).get("repetition_completion_us_by_flow")
    if not isinstance(values, list) or any(not isinstance(value, list) for value in values):
        raise RuntimeError("a latency row has no per-flow repetition ledger")
    return [[float(item) for item in value] for value in values]


def classify_h1(
    rows: list[dict[str, Any]], frozen: Mapping[str, Any], *, valid: bool
) -> dict[str, Any]:
    sizes = _size_order(frozen, "h1")
    by_pair: dict[tuple[int, int], dict[int, list[float]]] = defaultdict(dict)
    for row in rows:
        controls = row["applied_controls"]
        pair = (int(controls["source"]), int(controls["destination"]))
        samples = _latency_samples(row)
        if len(samples) != 1 or len(samples[0]) != 200:
            raise RuntimeError("an H1 row does not contain one 200-sample flow")
        by_pair[pair][int(row["payload_bytes"])] = samples[0]
    pair_fits = []
    for pair in sorted(by_pair):
        if set(by_pair[pair]) != set(sizes):
            raise RuntimeError(f"H1 pair {pair} does not contain the complete sweep")
        fit = detect_repeated_knee(
            sizes, [by_pair[pair][size] for size in sizes]
        )
        pair_fits.append({"source": pair[0], "destination": pair[1], **fit})
    breaks = [row for row in pair_fits if row["payload_bytes"] is not None]
    if not valid:
        verdict = "VOID"
        window = None
        return_ps = None
    elif not breaks:
        verdict = "INCONCLUSIVE_NO_BREAK_THROUGH_8_MIB"
        window = None
        return_ps = None
    elif len(breaks) != len(pair_fits):
        verdict = "INCONCLUSIVE_INCONSISTENT_PAIRS"
        window = None
        return_ps = None
    else:
        values = [int(row["payload_bytes"]) for row in breaks]
        indices = [sizes.index(value) for value in values]
        if max(indices) - min(indices) > 1:
            verdict = "INCONCLUSIVE_INCONSISTENT_PAIRS"
            window = None
            return_ps = None
        else:
            window = int(statistics.median(values))
            return_ps = statistics.median(
                float(row["return_delay_ps"]) for row in breaks
            )
            candidate = int(frozen["candidate_set"]["effective_window"]["payload_bytes"])
            verdict = (
                "IDENTIFIED_SUPPORTS_DECLARED_EFFECTIVE_WINDOW"
                if abs(sizes.index(min(sizes, key=lambda size: abs(size - window))) -
                       sizes.index(candidate)) <= 1
                else "IDENTIFIED_REFUTES_DECLARED_EFFECTIVE_WINDOW"
            )
    return {
        "verdict": verdict,
        "effective_window_payload_bytes": window,
        "effective_return_delay_ps": return_ps,
        "pair_fits": pair_fits,
        "promotion_required": verdict == "IDENTIFIED_REFUTES_DECLARED_EFFECTIVE_WINDOW",
    }


def _within_one_interval(value: float, target: float, sizes: Sequence[int]) -> bool:
    nearest = min(range(len(sizes)), key=lambda index: abs(sizes[index] - target))
    lower = sizes[max(0, nearest - 1)]
    upper = sizes[min(len(sizes) - 1, nearest + 1)]
    return lower <= value <= upper


def classify_h2(
    rows: list[dict[str, Any]], frozen: Mapping[str, Any], *, valid: bool
) -> dict[str, Any]:
    sizes = _size_order(frozen, "h2")
    by_count_source: dict[tuple[int, int], dict[int, list[float]]] = defaultdict(dict)
    for row in rows:
        sources = [
            int(value)
            for value in str(row["applied_controls"]["sources"]).split(",")
            if value
        ]
        samples = _latency_samples(row)
        if len(samples) != len(sources) or any(len(value) != 200 for value in samples):
            raise RuntimeError("an H2 row has an incomplete per-source timing ledger")
        for index, source in enumerate(sources):
            steady = samples[index][20:-20]
            by_count_source[(len(sources), source)][int(row["payload_bytes"])] = steady
    knee_rows = []
    for key in sorted(by_count_source):
        values = by_count_source[key]
        if set(values) != set(sizes):
            raise RuntimeError(f"H2 source ledger {key} does not contain the sweep")
        knee_rows.append(
            {
                "sender_count": key[0],
                "source": key[1],
                **detect_repeated_knee(sizes, [values[size] for size in sizes]),
            }
        )
    by_count: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in knee_rows:
        by_count[int(row["sender_count"])].append(row)
    observed = []
    for count in (1, 2, 3):
        knees = [
            int(row["payload_bytes"])
            for row in by_count[count]
            if row["payload_bytes"] is not None
        ]
        observed.append(
            {
                "sender_count": count,
                "per_sender_knee_payload_bytes": knees,
                "aggregate_outstanding_payload_bytes": sum(knees) if len(knees) == count else None,
            }
        )
    single = observed[0]["aggregate_outstanding_payload_bytes"]
    if not valid:
        verdict = "VOID"
    elif single is None or any(
        row["aggregate_outstanding_payload_bytes"] is None for row in observed
    ):
        verdict = "INCONCLUSIVE_MISSING_OR_INCONSISTENT_KNEES"
    else:
        per_link = all(
            all(_within_one_interval(value, float(single), sizes)
                for value in row["per_sender_knee_payload_bytes"])
            and _within_one_interval(
                float(row["aggregate_outstanding_payload_bytes"]),
                float(single) * int(row["sender_count"]),
                [value * int(row["sender_count"]) for value in sizes],
            )
            for row in observed
        )
        shared = all(
            all(_within_one_interval(value, float(single) / int(row["sender_count"]), sizes)
                for value in row["per_sender_knee_payload_bytes"])
            and _within_one_interval(
                float(row["aggregate_outstanding_payload_bytes"]),
                float(single),
                sizes,
            )
            for row in observed
        )
        verdict = (
            "IDENTIFIED_PER_LINK_POOLS"
            if per_link and not shared
            else "IDENTIFIED_SHARED_DESTINATION_POOL"
            if shared and not per_link
            else "INCONCLUSIVE_MISSING_OR_INCONSISTENT_KNEES"
        )
    return {
        "verdict": verdict,
        "aggregate_outstanding_discriminator": observed,
        "source_knee_fits": knee_rows,
        "promotion_required": verdict == "IDENTIFIED_SHARED_DESTINATION_POOL",
    }


def _window_raw_rates(row: Mapping[str, Any]) -> list[float]:
    controls = row["applied_controls"]
    sources = [int(value) for value in str(controls["sources"]).split(",") if value]
    destination = int(controls["destination"])
    links = row["traf73"]["window_counter_deltas"][
        "per_gpu_per_link_per_direction"
    ]
    duration_s = float(controls["traf73_window_measurement_ms"]) / 1000.0
    rates = []
    for source in sources:
        tx_bytes = sum(
            int(link["raw_tx_kib_delta"]) * 1024
            for link in links
            if int(link["gpu"]) == source and int(link["remote_gpu"]) == destination
        )
        rx_bytes = sum(
            int(link["raw_rx_kib_delta"]) * 1024
            for link in links
            if int(link["gpu"]) == destination and int(link["remote_gpu"]) == source
        )
        rates.append(max(tx_bytes, rx_bytes) / duration_s / 1e9)
    return rates


def classify_h3(
    rows: list[dict[str, Any]], frozen: Mapping[str, Any], *, valid: bool
) -> dict[str, Any]:
    rotations = []
    for row in sorted(rows, key=lambda value: int(value["applied_controls"]["source"])):
        rates = _window_raw_rates(row)
        aggregate = sum(rates)
        greedy = rates[0]
        small = rates[1:]
        if all(57.0 <= value <= 63.0 for value in small) and aggregate > 189.0:
            policy = "release_aware_round_robin"
        elif greedy >= 95.0 and min(small) < 57.0:
            policy = "greedy_capture"
        elif all(57.0 <= value <= 63.0 for value in rates) and aggregate <= 189.0:
            policy = "static_interleave"
        else:
            policy = "mixed_or_inconclusive"
        rotations.append(
            {
                "greedy_source": int(row["applied_controls"]["source"]),
                "source_order": [
                    int(value)
                    for value in str(row["applied_controls"]["sources"]).split(",")
                ],
                "achieved_raw_gbps_by_source_order": rates,
                "aggregate_achieved_raw_gbps": aggregate,
                "selected_policy": policy,
                "completed_bytes_by_source_order": row["traf73"][
                    "window_completed_bytes_by_flow"
                ],
            }
        )
    selected = {row["selected_policy"] for row in rotations}
    if not valid:
        verdict = "VOID"
        policy = None
    elif len(selected) == 1 and "mixed_or_inconclusive" not in selected:
        policy = selected.pop()
        verdict = f"IDENTIFIED_{policy.upper()}"
    else:
        policy = None
        verdict = "INCONCLUSIVE_MIXED_SHAPE"
    return {
        "verdict": verdict,
        "identified_policy": policy,
        "rotations": rotations,
        "promotion_required": policy not in (None, "release_aware_round_robin"),
    }


def _contains_stats64(value: object) -> bool:
    if isinstance(value, Mapping):
        return any(
            "stats64" in str(key).lower() or _contains_stats64(child)
            for key, child in value.items()
        )
    if isinstance(value, list):
        return any(_contains_stats64(child) for child in value)
    return "stats64" in str(value).lower()


def _wire_bytes(payload_bytes: int) -> int:
    return ((payload_bytes + 255) // 256) * 272


def _counter_agreement(links: Sequence[Mapping[str, Any]]) -> bool:
    for gpu in range(4):
        for remote in range(4):
            if gpu == remote:
                continue
            outgoing = [
                link
                for link in links
                if int(link["gpu"]) == gpu and int(link["remote_gpu"]) == remote
            ]
            incoming = [
                link
                for link in links
                if int(link["gpu"]) == remote and int(link["remote_gpu"]) == gpu
            ]
            allowance_kib = max(len(outgoing), len(incoming), 1)
            for counter in ("data", "raw"):
                tx_kib = sum(int(link[f"{counter}_tx_kib_delta"]) for link in outgoing)
                rx_kib = sum(int(link[f"{counter}_rx_kib_delta"]) for link in incoming)
                if abs(tx_kib - rx_kib) > allowance_kib:
                    return False
    return True


def _logical_extent_accounting(row: Mapping[str, Any]) -> bool:
    ledger = row["ordering_ledger"]
    flows = row["flow_rate_ledger"]
    logical_bytes = int(row["logical_bytes"])
    if logical_bytes != sum(int(flow["logical_bytes"]) for flow in flows):
        return False
    if int(ledger["expected_extents"]) != int(ledger["terminal_extents"]):
        return False
    if any(int(ledger[key]) != 0 for key in ("missing", "duplicate", "out_of_order")):
        return False
    traf73 = row["traf73"]
    completed = traf73["window_completed_bytes_by_flow"]
    return not completed or (
        len(completed) == len(flows)
        and all(
            0 < int(value) <= int(flow["logical_bytes"])
            for value, flow in zip(completed, flows, strict=True)
        )
    )


def _control_and_order_match(
    rows_by_family: Mapping[str, list[dict[str, Any]]],
    frozen: Mapping[str, Any],
) -> bool:
    for family in ("h1", "h2", "h3"):
        expected = runner.campaign_points(dict(frozen), family)
        rows = rows_by_family[family]
        if [row["point_id"] for row in rows] != [point["point_id"] for point in expected]:
            return False
        for row, point in zip(rows, expected, strict=True):
            controls = row["applied_controls"]
            if any(controls[field] != point[field] for field in runner.TSV_FIELDS[3:]):
                return False
            canonical = "".join(
                f"{value}\n"
                for value in (
                    row["case_name"],
                    row["point_id"],
                    row["producer"],
                    *(controls[field] for field in runner.TSV_FIELDS[3:]),
                )
            )
            if hashlib.sha256(canonical.encode("utf-8")).hexdigest() != row[
                "applied_control_sha256"
            ]:
                return False
    return True


def _latency_physical_sanity(
    rows_by_family: Mapping[str, list[dict[str, Any]]],
) -> dict[str, object]:
    minimum_floor_ratio = math.inf
    maximum_ceiling_ratio = 0.0
    passed = True
    for family in ("h1", "h2"):
        for row in rows_by_family[family]:
            payload_bytes = int(row["payload_bytes"])
            wire_bytes = _wire_bytes(payload_bytes)
            packet_count = (payload_bytes + 255) // 256
            floor_us = wire_bytes / 100.0e9 * 1.0e6
            ceiling_us = wire_bytes / 25.0e9 * 1.0e6 + packet_count * 0.2
            for samples in _latency_samples(row):
                for value in samples:
                    minimum_floor_ratio = min(minimum_floor_ratio, value / floor_us)
                    maximum_ceiling_ratio = max(
                        maximum_ceiling_ratio, value / ceiling_us
                    )
                    passed = passed and floor_us <= value <= ceiling_us
    return {
        "status": "PASS" if passed else "FAIL",
        "minimum_completion_over_wire_floor": minimum_floor_ratio,
        "maximum_completion_over_loose_ceiling": maximum_ceiling_ratio,
        "floor": "packetized wire bytes divided by 100 GB/s",
        "ceiling": "one-link serialization plus 200000 ps return per packet",
    }


def _guard_environment(attempt: Path) -> tuple[bool, dict[str, object]]:
    before = (attempt / "guards_before.txt").read_text(encoding="utf-8")
    after = (attempt / "guards_after.txt").read_text(encoding="utf-8")
    environment = load_json(attempt / "environment.json")
    topology = all(
        traf70_score._gpu_list_count(text) == 4
        and traf70_score._nv4_row_count(text) == 4
        for text in (before, after)
    )
    processes = all(
        not traf70_score._process_section(text).strip() for text in (before, after)
    )
    allocation = (
        environment.get("slurm_partition") == "a100-hourly"
        and bool(environment.get("slurm_job_id"))
    )
    return topology and processes and allocation, {
        "topology": topology,
        "competing_processes_clear": processes,
        "allocation": allocation,
    }


def score_fatal_guards(
    attempts: Mapping[str, Path],
    rows_by_family: Mapping[str, list[dict[str, Any]]],
    frozen: Mapping[str, Any],
    *,
    expected_head: str,
) -> dict[str, Any]:
    all_rows = [row for family in ("h1", "h2", "h3") for row in rows_by_family[family]]
    row_guards = [traf70_score.score_row_guards(row)["guards"] for row in all_rows]
    environment_rows = {
        family: _guard_environment(attempt) for family, attempt in attempts.items()
    }
    expected_counts = {
        "h1": int(frozen["h1_credit_window_and_return"]["configuration_count"]),
        "h2": int(frozen["h2_pool_scope"]["configuration_count"]),
        "h3": 3,
    }
    coverage = all(
        len(rows_by_family[family]) == expected_counts[family]
        and len({row["point_id"] for row in rows_by_family[family]}) == expected_counts[family]
        for family in expected_counts
    )
    integrity = all(values["FG01"]["pass"] for values in row_guards)
    counters = all(values["FG03"]["pass"] for values in row_guards)
    raw_data = all(values["FG04"]["pass"] for values in row_guards)
    errors = all(values["FG05"]["pass"] for values in row_guards)
    physical = all(values["FG06"]["pass"] for values in row_guards)
    throttle = all(values["FG07"]["pass"] for values in row_guards)
    logical_extents = all(_logical_extent_accounting(row) for row in all_rows)
    controls_and_order = _control_and_order_match(rows_by_family, frozen)
    paired_counters = all(
        _counter_agreement(
            row["observed_counter_deltas"]["per_gpu_per_link_per_direction"]
        )
        and (
            row["traf73"]["mode"] != "steady_arbitration"
            or _counter_agreement(
                row["traf73"]["window_counter_deltas"][
                    "per_gpu_per_link_per_direction"
                ]
            )
        )
        for row in all_rows
    )
    latency_physical = _latency_physical_sanity(rows_by_family)
    latency_shape = all(
        len(_latency_samples(row)) == len(row["flow_rate_ledger"])
        and all(len(values) == 200 and all(value > 0 for value in values)
                for values in _latency_samples(row))
        for family in ("h1", "h2")
        for row in rows_by_family[family]
    )
    h3_windows = all(
        len(row["traf73"]["window_device_us_by_flow"]) == 3
        and all(math.isclose(float(value), 500_000.0, rel_tol=0.0, abs_tol=0.5)
                for value in row["traf73"]["window_device_us_by_flow"])
        for row in rows_by_family["h3"]
    )
    h3_offer = True
    h3_physical = True
    for row in rows_by_family["h3"]:
        controls = row["applied_controls"]
        duration_s = float(controls["traf73_window_measurement_ms"]) / 1000.0
        ring_bytes = int(controls["traf73_ring_bytes"])
        rates = [
            int(value)
            for value in str(controls["traf73_flow_offered_rate_percents"]).split(",")
        ]
        completed = row["traf73"]["window_completed_bytes_by_flow"]
        allowance_gbps = _wire_bytes(ring_bytes) / duration_s / 1e9
        h3_offer = h3_offer and all(
            float(value) / ring_bytes * _wire_bytes(ring_bytes) /
            duration_s / 1e9 <= rate + allowance_gbps
            for value, rate in zip(completed, rates, strict=True)
        )
        raw_rates = _window_raw_rates(row)
        h3_offer = h3_offer and all(
            value <= rate + allowance_gbps
            for value, rate in zip(raw_rates, rates, strict=True)
        )
        h3_physical = h3_physical and all(
            value <= 100.0 + allowance_gbps for value in raw_rates
        ) and sum(raw_rates) <= 207.101921876 + allowance_gbps
    preservation = all(
        runner.sha256(ROOT / artifact["path"]) == artifact["sha256"]
        for artifact in frozen["preservation"]["recorded_artifacts"]
    )
    changed = subprocess.run(
        ("git", "diff", "--name-only", f"{frozen['study']['base_commit']}..{expected_head}"),
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    forbidden_changes = not any(
        path == "README.md"
        or path.endswith("/README.md")
        or path == frozen["aligned_authority"]["module_path"]
        or path == frozen["aligned_authority"]["candidate_profile"]["path"]
        for path in changed
    )
    conditions = [
        ("FG01", frozen["study"]["status"] == "EXPECTATIONS_ONLY"),
        ("FG02", preservation),
        ("FG03", all(row.get("protocol_scope", "").startswith("traf73_") for row in all_rows)),
        ("FG04", all(value[0] for value in environment_rows.values())),
        ("FG05", coverage),
        ("FG06", logical_extents),
        ("FG07", integrity),
        ("FG08", counters),
        ("FG09", errors),
        ("FG10", raw_data and paired_counters),
        ("FG11", throttle and all(value[0] for value in environment_rows.values())),
        (
            "FG12",
            latency_shape
            and controls_and_order
            and latency_physical["status"] == "PASS",
        ),
        ("FG13", h3_windows),
        ("FG14", h3_offer),
        ("FG15", h3_physical and physical),
        ("FG16", not _contains_stats64(all_rows)),
        ("FG17", preservation and forbidden_changes),
    ]
    guards = [
        {
            "id": guard_id,
            "description": frozen["fatal_guards"][index],
            "status": "PASS" if passed else "FAIL",
            "decidable": True,
        }
        for index, (guard_id, passed) in enumerate(conditions)
    ]
    return {
        "verdict": "PASS" if all(row["status"] == "PASS" for row in guards) else "VOID",
        "guards": guards,
        "environment": {
            family: evidence for family, (_, evidence) in environment_rows.items()
        },
        "coverage": expected_counts,
        "physical_sanity": latency_physical,
    }


def audit_hardware(
    bulk_root: Path,
    *,
    expected_head: str,
    scheduler_jobs: Mapping[str, str],
) -> dict[str, Any]:
    if re.fullmatch(r"[0-9a-f]{40}", expected_head) is None:
        raise ValueError("expected head must be a full lowercase Git SHA")
    frozen = runner.load_expectations()
    runner.verify_frozen_authority(frozen)
    attempts = {
        family: find_attempt(
            bulk_root,
            family=family,
            expected_head=expected_head,
            scheduler_job=scheduler_jobs[family],
        )
        for family in ("h1", "h2", "h3")
    }
    rows_by_family = {
        family: read_rows(attempt / "results.jsonl")
        for family, attempt in attempts.items()
    }
    fatal = score_fatal_guards(
        attempts,
        rows_by_family,
        frozen,
        expected_head=expected_head,
    )
    valid = fatal["verdict"] == "PASS"
    h1 = classify_h1(rows_by_family["h1"], frozen, valid=valid)
    h2 = classify_h2(rows_by_family["h2"], frozen, valid=valid)
    h3 = classify_h3(rows_by_family["h3"], frozen, valid=valid)
    promotion_cells = []
    if h1["promotion_required"]:
        promotion_cells.extend(
            f"H1:{row['source']}->{row['destination']}:bytes={row['payload_bytes']}"
            for row in h1["pair_fits"]
        )
    if h2["promotion_required"]:
        promotion_cells.append("H2:receiver=3:shared-destination-pool")
    if h3["promotion_required"]:
        promotion_cells.extend(
            f"H3:greedy={row['greedy_source']}:receiver=3"
            for row in h3["rotations"]
        )
    return {
        "schema": SCORE_SCHEMA,
        "task_id": "TRAF-73",
        "status": "VOID" if not valid else "COMPLETE_NONVOID_IDENTIFICATION",
        "measurement_validity": "VOID_FATAL_GUARD" if not valid else "VALID_FOR_FROZEN_RULES",
        "expectations_commit": runner.EXPECTATIONS_COMMIT,
        "expectations_sha256": runner.EXPECTATIONS_SHA256,
        "execution_head": expected_head,
        "scheduler_jobs": dict(scheduler_jobs),
        "fatal_guards": fatal,
        "h1_credit_window_and_return": h1,
        "h2_pool_scope": h2,
        "h3_arbitration": h3,
        "aggregate_outstanding_bytes": h2["aggregate_outstanding_discriminator"],
        "traf85_residual": {
            "required": bool(promotion_cells) and valid,
            "task_id": "TRAF-85" if promotion_cells and valid else None,
            "exact_promotion_cells": promotion_cells if valid else [],
        },
        "raw_evidence": {
            family: {
                "attempt_manifest_sha256": runner.sha256(attempt / "manifest.json"),
                "row_count": len(rows_by_family[family]),
                "row_sha256": hashlib.sha256(
                    (attempt / "results.jsonl").read_bytes()
                ).hexdigest(),
            }
            for family, attempt in attempts.items()
        },
    }


def render_markdown(score: Mapping[str, Any]) -> str:
    h1 = score["h1_credit_window_and_return"]
    h2 = score["h2_pool_scope"]
    h3 = score["h3_arbitration"]
    residual = score["traf85_residual"]
    lines = [
        "# TRAF-73 NV4 credit and arbitration identification result",
        "",
        "## Identification verdicts",
        "",
        f"- H1, credit window and return: **{h1['verdict']}**. The frozen aligned candidate predicted no break through 8 MiB because return overlaps serialization.",
        f"- H2, pool scope: **{h2['verdict']}**. The frozen discriminator is aggregate outstanding bytes at the per-sender knees.",
        f"- H3, arbitration: **{h3['verdict']}**. The frozen release-aware prediction gives both small senders 60 GB/s and the greedy sender the receiver remainder.",
        "",
        "## Aggregate outstanding bytes discriminator",
        "",
        "| Senders | Per-sender knees, B | Aggregate outstanding, B |",
        "|---:|---|---:|",
    ]
    for row in score["aggregate_outstanding_bytes"]:
        knees = ", ".join(str(value) for value in row["per_sender_knee_payload_bytes"])
        aggregate = row["aggregate_outstanding_payload_bytes"]
        lines.append(
            f"| {row['sender_count']} | {knees or 'none'} | "
            f"{'none' if aggregate is None else aggregate} |"
        )
    lines.extend(
        [
            "",
            "## What ran",
            "",
            "The three frozen H1, H2 and H3 families ran serially on one qualified",
            "four-A100 NV4 node through the corrected TRAF-70 producer lineage.",
            "",
            "## What came out",
            "",
            f"The complete hardware verdict is **{score['status']}**. The deciding H1, H2 and H3 outcomes are listed above.",
            "",
            "## What it changes for the project",
            "",
        ]
    )
    if score["measurement_validity"] == "VOID_FATAL_GUARD":
        lines.append("TRAF-73 stays open because a fatal guard voided the run.")
    elif residual["required"]:
        cells = ", ".join(f"`{value}`" for value in residual["exact_promotion_cells"])
        lines.append(
            f"TRAF-73 closes as a completed non-void identification. TRAF-85 owns model-value promotion from {cells}."
        )
    else:
        lines.append(
            "TRAF-73 closes as a completed non-void identification and no TRAF-85 promotion residual is required."
        )
    lines.extend(
        [
            "",
            "## What it does not change",
            "",
            "This result does not edit the aligned module, candidate profile or any",
            "README. Degrees 4, 8 and 16 remain simulated mesh extrapolations. H1 and",
            "H2 inconclusive verdicts promote no declared value or architecture prior.",
            "",
            "## Fatal guards",
            "",
            f"Fatal-guard verdict: **{score['fatal_guards']['verdict']}**.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_outstanding_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "sender_count",
                "per_sender_knee_payload_bytes",
                "aggregate_outstanding_payload_bytes",
            ),
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "sender_count": row["sender_count"],
                    "per_sender_knee_payload_bytes": ";".join(
                        str(value) for value in row["per_sender_knee_payload_bytes"]
                    ),
                    "aggregate_outstanding_payload_bytes": (
                        ""
                        if row["aggregate_outstanding_payload_bytes"] is None
                        else row["aggregate_outstanding_payload_bytes"]
                    ),
                }
            )


def write_json(path: Path, payload: object) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_text(path: Path, value: str) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(value)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    scheduler_jobs = parse_scheduler_jobs(args.scheduler_job)
    score = audit_hardware(
        args.bulk_root.resolve(),
        expected_head=args.expected_head,
        scheduler_jobs=scheduler_jobs,
    )
    write_json(args.json_out, score)
    write_text(args.markdown_out, render_markdown(score))
    write_outstanding_csv(
        args.outstanding_csv_out, score["aggregate_outstanding_bytes"]
    )
    print(json.dumps({"status": score["status"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
