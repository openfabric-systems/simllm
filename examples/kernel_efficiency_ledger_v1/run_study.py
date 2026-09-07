#!/usr/bin/env python3
"""Analyze the frozen retained kernel population without hardware or frameworks."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
from collections import Counter
from itertools import combinations
from pathlib import Path

from simllm.calibration.kernel_cycle_lut import analyze_kernel_cycle_capture
from simllm.compute.transformer import GPU_ENVELOPES

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXPECTATION_COMMIT = "0f2fe42056a90465fe7f1b0c2cfa12cf4545e09a"
CONSTANTS = "examples/a100_kernel_constants_v1/measurements/results.json"
LUT_RESULT = "examples/kernel_cycle_lut_v1/results.json"
LUT_FIXTURE = "tests/fixtures/kernel_cycle_lut_v1"
THRESHOLDS = (0.25, 0.5, 0.75)
SOURCES = ("measured", "datasheet")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition: bool, reason: str) -> None:
    if not condition:
        raise ValueError(f"VOID analysis: {reason}")


def positive(value: float, label: str) -> float:
    require(not isinstance(value, bool) and math.isfinite(value) and value > 0, label)
    return value


def close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=5e-9, abs_tol=1e-12)


def source_digest(path: Path) -> str:
    """SHA-256 of a tracked text source with line endings normalized to LF.

    Windows checkouts may materialize CRLF line endings for text files; the
    frozen digests were taken over LF content, so the digest is computed on
    the normalized bytes on every platform.
    """
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def verify_sources(root: Path, freeze: dict) -> None:
    for name, digest in freeze["source_sha256"].items():
        require(source_digest(root / name) == digest, f"source digest mismatch: {name}")
    require(freeze["thresholds"] == list(THRESHOLDS), "threshold freeze mismatch")
    require(freeze["envelope_sources"] == list(SOURCES), "envelope freeze mismatch")
    require(freeze["compute_peak_factors"] == [0.5, 1, 2], "peak sweep mismatch")


def declared_work(cell: dict) -> tuple[int, int]:
    family = cell["family"]
    m, n, k = (cell[key] for key in ("m", "n", "k"))
    if family.startswith(("G", "expert_")) or family == "gemm":
        return 2 * (m * k + k * n + m * n), 2 * m * n * k
    if family.startswith("attn_prefill"):
        return 4 * n * (2 * m * k + m * m), 4 * n * m * m * k
    if family == "attn_decode":
        require(n % 2 == 0, "decode key/value head geometry")
        return 4 * m * (n // 2) * cell["length"] * k, 4 * m * n * cell["length"] * k
    multipliers = {"hbm_read": 1, "hbm_write": 1, "hbm_copy": 2, "hbm_triad": 3,
                   "elem_scale": 2, "elem_add": 3, "elem_rmsnorm": 3}
    require(family in multipliers, f"unknown work family: {family}")
    return multipliers[family] * cell["size_bytes"], 0


def clock_peak(device: str, sm_mhz: float) -> float:
    positive(sm_mhz, "invalid SM clock")
    return (108 * 2048 if device == "a100" else 132 * 4096) * sm_mhz * 1e6


def load_envelopes(root: Path) -> dict:
    envelopes = {}
    for device in ("a100", "gh200"):
        prefix = f"examples/{device}_hardware_envelope_v1/measurements"
        lane = read_json(root / prefix / "lane_a_result.json")
        link = read_json(root / prefix / "lane_b_result.json")
        choices = [(row["bytes"] * (2 if op == "copy" else 1) / (row[f"{op}_ms"] / 1000),
                    f"hbm_{op}_{row['size_mib']}mib")
                   for row in lane["hbm"] if row["size_mib"] >= 256
                   for op in ("read", "write", "copy")]
        rate, anchor = max(choices)
        model = GPU_ENVELOPES["a100" if device == "a100" else "h100"]
        envelopes[device] = {
            "measured_hbm_bytes_s": positive(rate, "invalid measured envelope"),
            "measured_anchor": anchor,
            "datasheet_hbm_bytes_s": 2 * lane["mem_clock_khz"] * 1000
            * lane["mem_bus_bits"] / 8,
            "sm_mhz_reference": lane["sm_clock_khz"] / 1000,
            "clock_peak_flops_s": clock_peak(device, lane["sm_clock_khz"] / 1000),
            "memory_mhz": lane["mem_clock_khz"] / 1000,
            "launch_scale_s": lane["launch"]["pipelined_period_us"] * 1e-6,
            "l2_bytes": lane["l2_bytes"],
            "model_comparison_name": model.name,
            "model_comparison_exact_device": device == "a100",
            "model_peak_flops_s": model.peak_flops,
            "model_hbm_bytes_s": model.mem_bandwidth,
            "nvlink_pair_measured_min_bytes_s": min(r["gbps"] for r in link["p2p_unidirectional"]) * 1e9,
            "nvlink_pair_measured_max_bytes_s": max(r["gbps"] for r in link["p2p_unidirectional"]) * 1e9,
            "nvlink_egress_measured_bytes_s": link["p2p_fanout_0_to_all"]["aggregate_gbps"] * 1e9,
            "nvlink_pair_payload_bytes_s": (100 if device == "a100" else 150) * 1e9,
            "nvlink_egress_payload_bytes_s": (300 if device == "a100" else 450) * 1e9,
            "evidence_class": "retained-hardware-envelope",
            "source_state": "nonvoid",
        }
    return envelopes


def make_cell(identity: str, raw: dict, *, device: str, source: str, arm: str,
              state: str, evidence: str, sm_low: float, sm_high: float,
              memory_mhz: float, reasons: list[str] | None = None) -> dict:
    work_bytes, flops = declared_work(raw)
    require(close(work_bytes, raw["total_bytes"]) and close(flops, raw["flops"]),
            f"work declaration mismatch: {identity}")
    time = positive(raw["constant_s"], f"invalid elapsed time: {identity}")
    positive(memory_mhz, f"invalid memory clock: {identity}")
    positive(sm_low, f"invalid SM clock: {identity}")
    require(sm_low <= sm_high, f"reversed clock interval: {identity}")
    return {
        "cell_id": identity, "source_file": source, "evidence_class": evidence,
        "source_state": state, "source_void": state == "void", "arm": arm,
        "device": device, "family": raw["family"],
        "shape": {key: raw.get(key, 0) for key in ("m", "n", "k", "length", "size_bytes", "rotate")},
        "time_s": time, "sm_mhz_low": sm_low, "sm_mhz_high": sm_high,
        "memory_mhz": memory_mhz, "bytes_declared": work_bytes,
        "flops_declared": flops, "distinct_bytes": raw["distinct_bytes"],
        "flop_accounting": "declared-contractions-only" if flops else "source-zero-scalar-work-uninventoried",
        "source_void_for_scoring": raw.get("void_for_scoring", False),
        "source_in_scored_scope": raw.get("in_scored_scope"),
        "host_issue_bound": raw.get("host_issue_bound", False),
        "source_guard_reasons": reasons or [], "work_state": "declared",
        "launch_count_in_cell": 2 if raw["family"].startswith("attn_prefill") else 1,
    }


def load_cells(root: Path, envelopes: dict) -> list[dict]:
    result = []
    constants = read_json(root / CONSTANTS)
    for arm, rows in constants["cells"].items():
        for raw in rows:
            row = make_cell(f"constants/{arm}/{raw['id']}", raw, device="a100",
                            source=CONSTANTS, arm=arm, state=constants["verdict"],
                            evidence="retained-standalone-microbenchmark",
                            sm_low=raw["scored_state"], sm_high=raw["scored_state"],
                            memory_mhz=raw["memory_clocks"][0],
                            reasons=constants["voiding_guards"])
            row["source_batch_cv"] = raw["batch_cv"]
            row["host_loop_mean_s"] = raw["host_loop_mean_s"]
            row["device_batch_mean_s"] = raw["device_batch_mean_s"]
            row["host_ratio_max"] = raw["host_ratio_max"]
            row["source_group"] = raw["group"]
            result.append(row)
    for device in envelopes:
        source = f"examples/{device}_hardware_envelope_v1/measurements/lane_a_result.json"
        lane = read_json(root / source)
        for source_row in lane["hbm"]:
            for op in ("read", "write", "copy"):
                size = source_row["bytes"]
                raw = {"family": f"hbm_{op}", "m": 0, "n": 0, "k": 0,
                       "size_bytes": size, "total_bytes": size * (2 if op == "copy" else 1),
                       "flops": 0, "distinct_bytes": size * (2 if op == "copy" else 1),
                       "constant_s": source_row[f"{op}_ms"] / 1000}
                name = f"hbm_{op}_{source_row['size_mib']}mib"
                result.append(envelope_cell(device, source, name, raw, source_row))
        for source_row in lane["gemm"]:
            raw = {**source_row, "family": "gemm", "total_bytes": source_row["bytes"],
                   "distinct_bytes": source_row["bytes"], "constant_s": source_row["time_ms"] / 1000}
            name = f"gemm_{source_row['sweep']}_m{raw['m']}_n{raw['n']}_k{raw['k']}"
            result.append(envelope_cell(device, source, name, raw, source_row))
    retained = read_json(root / LUT_RESULT)
    record = analyze_kernel_cycle_capture(root / LUT_FIXTURE)
    require(record.record_id == retained["lookup_record"]["sha256"], "lookup record identity")
    entry = json.loads(record.canonical)["entries"][0]
    for kernel in entry["kernels"]:
        result.append({
            "cell_id": f"lut/{kernel['kernel_id']}", "source_file": LUT_FIXTURE,
            "evidence_class": "retained-framework-candidate-partial-subset",
            "source_state": retained["run_state"], "source_void": False,
            "device": "a100", "arm": "cuda-graph", "family": kernel["kernel_id"],
            "time_s": positive(kernel["measured_elapsed_ps"] / 1e12, "invalid lookup elapsed"),
            "nsys_time_s": kernel["nsys_median_elapsed_ps"] / 1e12,
            "sm_mhz_low": entry["observed_clocks"]["sm_hz"]["min"] / 1e6,
            "sm_mhz_high": entry["observed_clocks"]["sm_hz"]["max"] / 1e6,
            "memory_mhz": entry["observed_clocks"]["memory_hz"]["median"] / 1e6,
            "shape": {"step_shape": entry["key"]["shape"]},
            "bytes_declared": None, "flops_declared": None, "work_state": "missing",
            "flop_accounting": "unavailable", "source_guard_reasons": [],
            "source_void_for_scoring": False, "host_issue_bound": False,
            "record_sha256": record.record_id, "launch_count_per_step": kernel["launch_count"],
            "distribution_verdict": entry["distribution"]["verdict"],
            "work_missing_reason": "per-kernel operand shapes, byte attribution and operation counts absent",
        })
    return result


def envelope_cell(device: str, source: str, name: str, raw: dict, telemetry: dict) -> dict:
    clocks = [telemetry[key]["sm_mhz"] for key in ("clocks_before", "clocks_after")]
    return make_cell(f"envelope/{device}/{name}", raw, device=device, source=source,
                     arm="lane-a", state="nonvoid", evidence="retained-hardware-envelope",
                     sm_low=min(clocks), sm_high=max(clocks),
                     memory_mhz=telemetry["clocks_before"]["mem_mhz"])


def roofline(work_bytes: int, flops: int, time: float, bandwidth: float, peak: float) -> dict:
    positive(time, "invalid roofline time")
    positive(bandwidth, "invalid roofline bandwidth")
    positive(peak, "invalid roofline compute peak")
    require(work_bytes > 0 and flops >= 0, "invalid roofline work")
    memory = work_bytes / bandwidth
    compute = flops / peak
    return {"bytes_s": work_bytes / time, "flops_s": flops / time,
            "intensity_flops_byte": flops / work_bytes, "ridge_flops_byte": peak / bandwidth,
            "memory_floor_s": memory, "compute_floor_s": compute,
            "binding_floor_s": max(memory, compute), "fraction": max(memory, compute) / time,
            "memory_fraction": memory / time, "compute_fraction": compute / time,
            "ridge_class": "compute-bound" if compute >= memory else "HBM-bound"}


def analyze_cell(cell: dict, env: dict) -> dict:
    row = dict(cell)
    if row["work_state"] == "missing":
        row.update({"class": None, "cell_void": False, "verdict": "unavailable-work",
                    "cell_guard_findings": [], "views": {source: None for source in SOURCES}})
        return row
    peak = clock_peak(row["device"], row["sm_mhz_high"])
    row["compute_peak_flops_s"] = peak
    row["views"] = {source: roofline(row["bytes_declared"], row["flops_declared"],
                                    row["time_s"], env[f"{source}_hbm_bytes_s"], peak)
                    for source in SOURCES}
    measured = row["views"]["measured"]
    launch = row["launch_count_in_cell"] * env["launch_scale_s"]
    row["launch_scale_s"] = launch
    row["launch_bound"] = measured["binding_floor_s"] <= launch or row["host_issue_bound"]
    row["class"] = "launch-bound" if row["launch_bound"] else measured["ridge_class"]
    floor = max(row["flops_declared"] / peak,
                max(row["distinct_bytes"] - env["l2_bytes"], 0) / env["datasheet_hbm_bytes_s"])
    row["physical_floor_s"] = floor
    findings = []
    if measured["memory_fraction"] > 1:
        findings.append("nominal-byte-rate-above-measured-envelope")
    if measured["compute_fraction"] > 1:
        findings.append("declared-flop-rate-above-clock-ceiling")
    if row["time_s"] < floor:
        findings.append("elapsed-below-cache-credited-physical-floor")
    row["cell_guard_findings"] = findings
    row["cell_void"] = bool(findings)
    row["verdict_void"] = (row["source_void"] or row["cell_void"]
                           or row["source_void_for_scoring"] or row["host_issue_bound"])
    for view in row["views"].values():
        view["class"] = "launch-bound" if row["launch_bound"] else view["ridge_class"]
        view["diagnostic_flags"] = {str(t): view["fraction"] < t and not row["launch_bound"]
                                    for t in THRESHOLDS}
    row["verdict"] = ("void-with-findings" if row["verdict_void"] else
                      "launch-screen-excluded" if row["launch_bound"] else
                      "inefficient-under-declared-threshold" if measured["fraction"] < 0.5 else
                      "not-below-declared-threshold")
    return row


def ranked(rows: list[dict], source: str) -> list[dict]:
    return sorted((row for row in rows if row["views"][source] is not None),
                  key=lambda row: (row["views"][source]["fraction"], row["cell_id"]))


def rank_comparison(rows: list[dict]) -> dict:
    orders = {s: [row["cell_id"] for row in ranked(rows, s)] for s in SOURCES}
    positions = {s: {key: rank for rank, key in enumerate(order, 1)} for s, order in orders.items()}
    delta = {key: positions["datasheet"][key] - positions["measured"][key] for key in orders["measured"]}
    n = len(delta)
    inversions = sum(positions["datasheet"][a] > positions["datasheet"][b]
                     for a, b in combinations(orders["measured"], 2))
    return {"spearman": 1 - 6 * sum(d * d for d in delta.values()) / (n * (n * n - 1)),
            "max_rank_displacement": max(abs(d) for d in delta.values()),
            "pairwise_inversions": inversions, "comparison_pairs": n * (n - 1) // 2,
            "changed_ranks": {key: d for key, d in delta.items() if d},
            "orders": orders}


def structural_checks(rows: list[dict], envelopes: dict, comparison: dict) -> dict:
    peak_ids = []
    for row in rows:
        if row["work_state"] == "missing":
            require(all(value is None for value in row["views"].values()), "missing work gained a rate")
            continue
        env = envelopes[row["device"]]
        views = row["views"]
        for view in views.values():
            flags = [view["diagnostic_flags"][str(t)] for t in THRESHOLDS]
            require(flags == sorted(flags), "threshold candidate nesting")
        probes = [roofline(row["bytes_declared"], row["flops_declared"], row["time_s"],
                           env["measured_hbm_bytes_s"], row["compute_peak_flops_s"] * factor)
                  for factor in (0.5, 1, 2)]
        if all(p["ridge_class"] == "HBM-bound" for p in probes):
            require(probes[0]["fraction"] == probes[1]["fraction"] == probes[2]["fraction"],
                    "memory-bound efficiency depends on compute peak")
            peak_ids.append(row["cell_id"])
        if all(v["ridge_class"] == "HBM-bound" for v in views.values()):
            require(close(views["datasheet"]["fraction"] / views["measured"]["fraction"],
                          env["measured_hbm_bytes_s"] / env["datasheet_hbm_bytes_s"]),
                    "memory denominator scaling")
    for device in envelopes:
        eligible = {r["cell_id"] for r in rows if r["work_state"] == "declared"
                    and r["device"] == device and not r["launch_bound"]
                    and all(v["ridge_class"] == "HBM-bound" for v in r["views"].values())}
        left, right = ([key for key in comparison["orders"][s] if key in eligible] for s in SOURCES)
        require(left == right, "within-device memory ordering")
    return {"state": "held", "scored": False,
            "checks": ["source-digests", "inventory-and-identity", "positive-time-and-clocks",
                       "work-declaration-agreement", "lookup-record-digest", "missing-work-nullability",
                       "threshold-nesting-and-order-identity", "memory-denominator-scaling",
                       "within-device-memory-order", "compute-peak-independence"],
            "compute_peak_independence_cell_ids": peak_ids}


def relation(label: str, selected: list[dict], lo: float, hi: float, wanted_class: str) -> dict:
    values = [row["views"]["measured"]["fraction"] for row in selected]
    require(bool(values), f"empty relation: {label}")
    return {"cell_ids": [row["cell_id"] for row in selected], "lower": lo, "upper": hi,
            "observed_min": min(values), "observed_max": max(values),
            "descriptive_holds": all(lo <= value <= hi for value in values)
            and all(row["class"] == wanted_class for row in selected),
            "accepted_score": None,
            "void_cells": [row["cell_id"] for row in selected if row["verdict_void"]]}


def evaluate(root: Path = ROOT) -> tuple[list[dict], dict]:
    freeze = read_json(HERE / "expectations.json")
    verify_sources(root, freeze)
    envelopes = load_envelopes(root)
    cells = load_cells(root, envelopes)
    ids = [cell["cell_id"] for cell in cells]
    require(len(ids) == len(set(ids)), "duplicate source cell")
    require(set(ids) == set(freeze["cell_ids"]) and len(ids) == 385, "frozen inventory mismatch")
    rows = [analyze_cell(cell, envelopes[cell["device"]]) for cell in cells]
    require(sum(r["work_state"] == "declared" for r in rows) == 380, "declared work coverage")
    comparison = rank_comparison(rows)
    guards = structural_checks(rows, envelopes, comparison)
    for source in SOURCES:
        for rank, row in enumerate(ranked(rows, source), 1):
            row["views"][source]["rank"] = rank
    decode = [r for r in rows if r["family"] == "attn_decode" and r["arm"] == "boosted"
              and r["bytes_declared"] >= 160 * 2**20]
    large = [r for r in rows if r["cell_id"] == "constants/boosted/gemm_G4_m8192"
             or (r["cell_id"].startswith("envelope/") and "/gemm_square_" in r["cell_id"]
                 and r["shape"]["m"] >= 8192)]
    copy = [r for r in rows if r["family"] == "hbm_copy" and r["shape"]["size_bytes"] >= 256 * 2**20]
    relations = {"R1-decode": relation("decode", decode, 0.04, 0.16, "HBM-bound"),
                 "R2-large-gemm": relation("large-gemm", large, 0.8, 1, "compute-bound"),
                 "R3-copy": relation("copy", copy, 0.8, 1, "HBM-bound"),
                 "R4-rank-stability": {"spearman_lower": 0.98,
                                       "observed_spearman": comparison["spearman"],
                                       "descriptive_holds": comparison["spearman"] >= 0.98,
                                       "accepted_score": None}}
    configurations = []
    for source in SOURCES:
        order = ranked(rows, source)
        for threshold in THRESHOLDS:
            flags = [r["cell_id"] for r in order if r["views"][source]["diagnostic_flags"][str(threshold)]]
            accepted = [r["cell_id"] for r in order if r["cell_id"] in flags and not r["verdict_void"]]
            configurations.append({"envelope": source, "threshold": threshold,
                                   "diagnostic_candidates": flags, "accepted_source_candidates": accepted,
                                   "rank_order_sha256": hashlib.sha256(
                                       json.dumps([r["cell_id"] for r in order]).encode()).hexdigest()})
    for source in SOURCES:
        require(len({c["rank_order_sha256"] for c in configurations if c["envelope"] == source}) == 1,
                "threshold changed rank order")
    summary = {
        "schema": "simllm-kernel-efficiency-study-v1", "expectation_commit": EXPECTATION_COMMIT,
        "freeze_sha256": {name: hashlib.sha256((HERE / name).read_bytes()).hexdigest()
                          for name in ("expectations.md", "expectations.json")},
        "source_sha256": freeze["source_sha256"], "analysis_state": "complete",
        "run_state": "void-for-calibration-with-findings", "behavioral_score": None,
        "evidence_classes": {"source_cells": len(rows), "work_declared_cells": 380,
                             "work_missing_cells": 5, "run_configurations": 6,
                             "behavioral_relation_families": 4, "exact_oracle_rows": 0,
                             "native_executables": 0},
        "source_population": dict(Counter(r["evidence_class"] for r in rows)),
        "envelopes": envelopes, "structural_guards": guards, "relations": relations,
        "rank_comparison": comparison, "configurations": configurations,
        "primary_class_counts": dict(Counter(r["class"] or "unavailable" for r in rows)),
        "source_void_cells": sum(r["source_void"] for r in rows),
        "cell_guard_findings": {r["cell_id"]: r["cell_guard_findings"] for r in rows if r["cell_void"]},
        "missing_work_cells": [r["cell_id"] for r in rows if r["work_state"] == "missing"],
        "closure": {"closes": [], "keeps_open": ["COMP-45", "COMP-46"],
                    "residuals": ["nonvoid-constant-measurements", "production-decode-comparator",
                                  "Granite-per-kernel-work-attribution"]},
    }
    return rows, summary


def ledger_csv(rows: list[dict]) -> str:
    flat = []
    order = ranked(rows, "measured") + sorted((r for r in rows if r["work_state"] == "missing"),
                                             key=lambda row: row["cell_id"])
    for row in order:
        record = {key: value for key, value in row.items() if key != "views"}
        for source, view in row["views"].items():
            for field in ("rank", "class", "ridge_class", "fraction", "memory_fraction", "compute_fraction",
                          "bytes_s", "flops_s", "intensity_flops_byte", "ridge_flops_byte", "binding_floor_s"):
                record[f"{source}_{field}"] = view[field] if view else None
            for threshold in THRESHOLDS:
                record[f"{source}_diagnostic_below_{threshold}"] = (
                    view["diagnostic_flags"][str(threshold)] if view else None)
        flat.append({key: json.dumps(value, sort_keys=True, separators=(",", ":"))
                     if isinstance(value, (list, dict)) else value for key, value in record.items()})
    output = io.StringIO(newline="")
    keys = ["cell_id", "measured_rank", "datasheet_rank", "class", "verdict"]
    keys += sorted(set().union(*(r.keys() for r in flat)) - set(keys))
    writer = csv.DictWriter(output, fieldnames=keys, lineterminator="\n")
    writer.writeheader()
    writer.writerows(flat)
    return output.getvalue()


FAMILY_GROUPS = (
    ("decode attention", ("attn_decode", "flash_combine", "flash_split_kv")),
    ("prefill attention", ("attn_prefill_granite", "attn_prefill_synthetic")),
    ("GEMM", ("gemm", "G1:", "G2:", "G3:", "G4:", "G5:")),
    ("MoE and routing", ("expert_down", "expert_gate_up", "fused_moe", "topk_gating", "gemvx")),
    ("elementwise and norm", ("elem_add", "elem_scale", "elem_rmsnorm")),
    ("HBM streaming", ("hbm_copy", "hbm_read", "hbm_write", "hbm_triad")),
)
GROUP_COLORS = ("#d62728", "#ff7f0e", "#1f77b4", "#9467bd", "#2ca02c", "#7f7f7f")


def family_group(family: str) -> str:
    for label, prefixes in FAMILY_GROUPS:
        if any(family.startswith(prefix) for prefix in prefixes):
            return label
    return "other"


def render(rows: list[dict], output: Path) -> None:
    """Three views of one ledger: the compute roofline, the memory side, the ranked fractions.

    Panel A puts every A100 cell with declared arithmetic on the classic roofline
    (achieved FLOP/s against arithmetic intensity) under the measured HBM
    envelope and the clock-derived arithmetic ceiling. Panel B shows the memory
    side for both devices: achieved bytes per second against declared bytes,
    with each device's measured envelope as a horizontal line, so a kernel that
    moves its bytes far below the roof is visible at a glance. Panel C keeps the
    ranked fraction curve with the per-cell breaches. Hollow markers are cells
    whose source study or own guard is void; filled markers are nonvoid.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    ranked_rows = [r for r in rows if r["views"]["measured"] is not None]
    colors = dict(zip((label for label, _ in FAMILY_GROUPS), GROUP_COLORS))
    colors["other"] = "#17becf"
    device_markers = {"a100": "o", "gh200": "^"}

    def device_roof(device: str) -> tuple[float, float]:
        selected = [r for r in ranked_rows if r["device"] == device]
        peak = max(r["compute_peak_flops_s"] for r in selected)
        bandwidth = max(r["compute_peak_flops_s"] / r["views"]["measured"]["ridge_flops_byte"]
                        for r in selected)
        return peak, bandwidth

    def style(row: dict) -> dict:
        void = row["source_void"] or row["cell_void"]
        color = colors[family_group(row["family"])]
        return {"marker": device_markers[row["device"]], "s": 13,
                "facecolors": "none" if void else color, "edgecolors": color, "linewidths": 0.7}

    fig = plt.figure(figsize=(7, 7.4))
    grid = fig.add_gridspec(2, 2, height_ratios=(1, 0.8), left=0.10, right=0.98,
                            bottom=0.265, top=0.89, hspace=0.52, wspace=0.35)
    axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1]),
            fig.add_subplot(grid[1, :])]

    # Panel A: A100 roofline for cells with declared arithmetic.
    ax = axes[0]
    peak, bandwidth = device_roof("a100")
    ridge = peak / bandwidth
    xs = [0.5, ridge, 1e4]
    ax.plot(xs, [min(peak, bandwidth * x) / 1e12 for x in xs], color="black", linewidth=1.0)
    ax.vlines(ridge, 0.01, peak / 1e12, color="gray", linewidth=0.6, linestyle="--", zorder=0)
    for row in ranked_rows:
        if row["device"] != "a100" or row["flops_declared"] <= 0:
            continue
        view = row["views"]["measured"]
        ax.scatter([view["intensity_flops_byte"]], [view["flops_s"] / 1e12], **style(row))
    ax.set(xscale="log", yscale="log", xlim=(0.5, 1e4), ylim=(0.01, 1200),
           xlabel="Arithmetic intensity (FLOP/byte)", ylabel="Achieved rate (TFLOP/s)")
    ax.set_title("A: A100 arithmetic roofline", fontsize=9, loc="left")
    ax.text(0.03, 0.95, f"Clock ceiling: {peak / 1e12:.3f} TFLOP/s",
            transform=ax.transAxes, fontsize=7.5, va="top")
    ax.text(ridge * 1.18, 0.075, f"Ridge\n{ridge:.3f}\nFLOP/byte",
            fontsize=7.5, color="0.35")

    # Panel B: memory side on both devices.
    ax = axes[1]
    for device, line_style in (("a100", "-"), ("gh200", "--")):
        _, dev_bandwidth = device_roof(device)
        ax.axhline(dev_bandwidth / 1e9, color="black", linewidth=1.0, linestyle=line_style,
                   zorder=0)
        ax.text(0.02, dev_bandwidth / 1e9 * 1.10,
                f"{device.upper()}: {dev_bandwidth / 1e9:.3f}",
                transform=ax.get_yaxis_transform(), fontsize=7.5, va="bottom")
    for row in ranked_rows:
        view = row["views"]["measured"]
        ax.scatter([row["bytes_declared"]], [view["bytes_s"] / 1e9], **style(row))
    ax.set(xscale="log", yscale="log", xlim=(2e5, 1.5e10), ylim=(7, 8000),
           xlabel="Declared traffic per kernel (bytes)", ylabel="Achieved rate (GB/s)")
    ax.set_title("B: Memory rates, both devices", fontsize=9, loc="left")

    # Panel C: ranked fractions with breaches.
    ax = axes[2]
    order = ranked(rows, "measured")
    x = list(range(1, len(order) + 1))
    for source, line_style in (("measured", "-"), ("datasheet", ":")):
        ax.plot(x, [r["views"][source]["fraction"] for r in order], line_style, color="black",
                linewidth=1.0, label=f"{source} envelope")
    bad = [i for i, r in enumerate(order) if r["cell_void"]]
    ax.scatter([x[i] for i in bad], [order[i]["views"]["measured"]["fraction"] for i in bad],
               marker="x", s=20, linewidths=1.0, color="#d62728",
               label="cell envelope breach", zorder=3)
    for threshold in THRESHOLDS:
        ax.axhline(threshold, color="gray", linewidth=0.6, linestyle="--")
    ax.axhline(1, color="black", linewidth=0.7)
    ax.set(xlabel="Rank by measured-envelope fraction (1 = lowest)",
           ylabel="Binding roofline fraction (unitless)", xlim=(0, len(order) + 12),
           yticks=[0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75])
    ax.set_title(f"C: All {len(order)} declared-work cells, including void cells", fontsize=9, loc="left")
    ax.legend(fontsize=7.5, loc="upper left", frameon=False)
    for ax in axes:
        ax.tick_params(axis="both", labelsize=8)
        ax.xaxis.label.set_size(8)
        ax.yaxis.label.set_size(8)
        ax.spines[["top", "right"]].set_visible(False)

    handles = [Line2D([], [], color=colors[label], linewidth=3, label=label)
               for label, _ in FAMILY_GROUPS]
    fig.legend(handles=handles, fontsize=8, title="Kernel family (panels A and B)", title_fontsize=8,
               loc="lower center", bbox_to_anchor=(0.54, 0.105), ncol=3, frameon=False,
               handlelength=1.1, columnspacing=1.5)
    handles = [Line2D([], [], color="black", marker=marker, linestyle="", markersize=4,
                      markerfacecolor="none", label=f"Shape: {device.upper()}")
               for device, marker in device_markers.items()]
    handles += [Line2D([], [], color="black", marker="o", linestyle="", markersize=4,
                       label="Filled: nonvoid source and cell"),
                Line2D([], [], color="black", marker="o", linestyle="", markersize=4,
                       markerfacecolor="none", label="Hollow: void source or cell")]
    fig.legend(handles=handles, fontsize=8, loc="lower center", bbox_to_anchor=(0.54, 0.03),
               ncol=2, frameon=False, handlelength=1.1, columnspacing=2)
    fig.suptitle("Kernel efficiency ledger", fontsize=11, y=0.985)
    fig.text(0.5, 0.945, "264 source-void cells retained; 5 missing-work kernels unranked",
             ha="center", fontsize=8)
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / "ledger.png", dpi=300)
    fig.savefig(output / "ledger.pdf", metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)


def write_artifacts(rows: list[dict], summary: dict, output: Path, plot: bool = True) -> None:
    csv_text = ledger_csv(rows)
    output.mkdir(parents=True, exist_ok=True)
    (output / "results.csv").write_text(csv_text, encoding="utf-8")
    summary = {**summary, "ledger_sha256": hashlib.sha256(csv_text.encode()).hexdigest()}
    (output / "results.json").write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
                                         encoding="utf-8")
    if plot:
        render(rows, output / "figures")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=os.environ.get("SIMLLM_STUDY_OUTPUT"))
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()
    if args.output_dir is None:
        parser.error("provide --output-dir or configure SIMLLM_STUDY_OUTPUT")
    rows, summary = evaluate()
    write_artifacts(rows, summary, args.output_dir, plot=not args.no_plot)
    print(f"Analysis: {summary['analysis_state']}; evidence: {summary['run_state']}")
    print(f"Inventory: {len(rows)} source cells; 380 declared work; 5 unavailable work")
    print(f"Source-void cells: {summary['source_void_cells']}")
    print(f"Cell envelope/physical findings: {len(summary['cell_guard_findings'])}")
    print(f"Rank Spearman: {summary['rank_comparison']['spearman']:.9f}; "
          f"inversions: {summary['rank_comparison']['pairwise_inversions']}")
    print("Structural guards: held; behavioral score: unavailable; closes: none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
