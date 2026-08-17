#!/usr/bin/env python3
"""Score the TRAF-51 calibration comparison against the frozen expectations.

Reads the byte-locked Merlin capture dataset (re-verifying every consumed
file against its manifest, and the manifest against its frozen SHA-256) and
the simulation artifacts produced by run_cells.py, re-derives every measured
anchor and target from tracked bytes, evaluates the frozen fatal guards, the
simulation relations (EX, BE, ST), and the conditional composed rows (CO),
and writes a deterministic calibration_summary.json plus a verdict table on
stdout. Every band evaluated here is stated in expectations.md; nothing is
tuned here.

    python analyze_calibration.py --dataset-root DIR --run-root DIR --out DIR
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import re
import statistics
from fractions import Fraction
from pathlib import Path

DATASET_MANIFEST_SHA256 = (
    "a6b7e61e294d87d76ce69ee7042e15c2eade99bbc8789e296377615d2bd4af88")

CHUNK_BYTES = 8_388_608
# Frozen fabric constants (expectations.md napkin table). The latency term
# is the hand-derived first-chunk delivery; EX-1 verifies the simulation
# confirms it before any composed row is trusted.
T_FAB_LAT_PS = 340_417_280
CROSS_BIN_END_PS = {"cal-solo-a": 341_000_000,
                    "cal-solo-100g": 680_000_000,
                    "cal-solo-4160": 342_000_000}

# Per-port payload ceilings as exact fractions of bytes per second.
CP_9038 = Fraction(25_000_000_000) * 8948 / 9038
CP_4160 = Fraction(25_000_000_000) * 4096 / 4160
CP_100G = CP_9038 / 2

FRAMING = {
    "cal-solo-a": (9038, 8948, 200_000_000_000, CP_9038),
    "cal-solo-b": (9038, 8948, 200_000_000_000, CP_9038),
    "cal-solo-4160": (4160, 4096, 200_000_000_000, CP_4160),
    "cal-solo-100g": (9038, 8948, 100_000_000_000, CP_100G),
    "cal-echo-2": (9038, 8948, 200_000_000_000, CP_9038),
    "cal-echo-4": (9038, 8948, 200_000_000_000, CP_9038),
}

ECHO_DEGREE = {"cal-echo-2": 2, "cal-echo-4": 4}

# One 100-us bin holds payload / 1e-4 s of goodput: bytes times 1e4 per second.
BIN_100US_TO_BYTES_PER_S = 10_000

MANIFEST_RE = re.compile(
    r"\[ss-dragonfly manifest\] injected=(\d+) delivered=(\d+) "
    r"delivered_payload_bytes=(\d+) dropped=(\d+)")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class Dataset:
    """Hash-verified access to the capture dataset."""

    def __init__(self, root: Path) -> None:
        self.root = root
        manifest_path = root / "MANIFEST.json"
        self.manifest_sha256 = sha256_file(manifest_path)
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.verified: dict[str, bool] = {}

    def open_verified(self, relative: str) -> bytes:
        path = self.root / relative
        entry = self.manifest.get(relative)
        if entry is None:
            raise KeyError(f"dataset file not in manifest: {relative}")
        data = path.read_bytes()
        ok = (len(data) == entry["bytes"]
              and hashlib.sha256(data).hexdigest() == entry["sha256"])
        self.verified[relative] = ok
        if not ok:
            raise ValueError(f"dataset file failed verification: {relative}")
        return data

    def series(self, relative: str) -> list[int]:
        """Destination completion timestamps (ns since T0), series order."""
        raw = gzip.decompress(self.open_verified(relative))
        rows = csv.DictReader(raw.decode("utf-8").splitlines())
        return [int(row["t_ns_since_t0"]) for row in rows]

    def stats(self, relative: str) -> dict:
        return json.loads(self.open_verified(relative).decode("utf-8"))


def window_rate_gbps(times_ns: list[int], lo_s: float, hi_s: float) -> float:
    lo, hi = lo_s * 1e9, hi_s * 1e9
    count = sum(1 for t in times_ns if lo <= t < hi)
    return count * CHUNK_BYTES / (hi_s - lo_s) / 1e9


def window_p50_us(times_ns: list[int], lo_s: float, hi_s: float) -> float:
    lo, hi = lo_s * 1e9, hi_s * 1e9
    deltas = [(times_ns[i] - times_ns[i - 1]) / 1000.0
              for i in range(1, len(times_ns)) if lo <= times_ns[i] < hi]
    return statistics.median(deltas)


def load_sim_bins(csv_path: Path) -> dict[int, dict[int, int]]:
    """bins[bin_start_ps][flow_id] = delivered payload bytes."""
    bins: dict[int, dict[int, int]] = {}
    with csv_path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            start = int(row["bin_start_ps"])
            flow = int(row["flow_id"])
            bins.setdefault(start, {})[flow] = int(
                row["delivered_payload_bytes"])
    return bins


def parse_log_counts(log_path: Path) -> dict[str, int]:
    match = None
    for line in log_path.read_text(encoding="utf-8").splitlines():
        found = MANIFEST_RE.search(line)
        if found:
            match = found
    if match is None:
        raise ValueError(f"no manifest counts in {log_path}")
    injected, delivered, payload, dropped = (int(g) for g in match.groups())
    return {"injected": injected, "delivered": delivered,
            "delivered_payload_bytes": payload, "dropped": dropped}


def chunk_crossings(bins: dict[int, dict[int, int]], bin_ps: int,
                    flow: int) -> list[int]:
    """Bin-end times (ps) at which cumulative payload crosses k * CHUNK."""
    crossings: list[int] = []
    cumulative = 0
    next_boundary = CHUNK_BYTES
    for start in sorted(bins):
        cumulative += bins[start].get(flow, 0)
        while cumulative >= next_boundary:
            crossings.append(start + bin_ps)
            next_boundary += CHUNK_BYTES
    return crossings


def aggregate_100us(bins: dict[int, dict[int, int]]) -> dict[int, int]:
    grouped: dict[int, int] = {}
    for start, flows in bins.items():
        grouped[start // 100_000_000] = (
            grouped.get(start // 100_000_000, 0) + sum(flows.values()))
    return grouped


def per_flow_100us(bins: dict[int, dict[int, int]], flow: int) -> dict[int, int]:
    grouped: dict[int, int] = {}
    for start, flows in bins.items():
        grouped[start // 100_000_000] = (
            grouped.get(start // 100_000_000, 0) + flows.get(flow, 0))
    return grouped


def packet_bin_bound_bytes(wire: int, payload: int, rate_bps: int) -> int:
    serialization_ps = wire * 8 * 10**12 // rate_bps
    return ((100_000_000 + serialization_ps - 1) // serialization_ps + 1) * payload


def rnd(value: float, digits: int = 6) -> float:
    return float(f"{value:.{digits}f}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--out", required=True)
    options = parser.parse_args()

    dataset = Dataset(Path(options.dataset_root).resolve())
    run_root = Path(options.run_root).resolve()
    out_dir = Path(options.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    guards: dict[str, dict] = {}
    fatal: list[str] = []

    # FG-6 dataset integrity.
    guards["FG-6_dataset_manifest"] = {
        "sha256": dataset.manifest_sha256,
        "expected": DATASET_MANIFEST_SHA256,
        "ok": dataset.manifest_sha256 == DATASET_MANIFEST_SHA256,
    }
    if not guards["FG-6_dataset_manifest"]["ok"]:
        fatal.append("FG-6: dataset manifest hash mismatch")

    # FG-1/FG-2/FG-3/FG-5 from the run manifest.
    run_manifest = json.loads(
        (run_root / "run_manifest.json").read_text(encoding="utf-8"))
    guards["FG-1_binary"] = {
        "sha256": run_manifest["binary_sha256"],
        "expected": run_manifest["binary_sha256_expected"],
        "ok": run_manifest["binary_sha256"]
        == run_manifest["binary_sha256_expected"],
    }
    if "submodule_head" in run_manifest:
        guards["FG-1_binary"]["submodule_head"] = run_manifest["submodule_head"]
        guards["FG-1_binary"]["submodule_ok"] = (
            run_manifest["submodule_head"]
            == run_manifest["submodule_head_expected"])
        if not guards["FG-1_binary"]["submodule_ok"]:
            fatal.append("FG-1: submodule pin mismatch")
    if not guards["FG-1_binary"]["ok"]:
        fatal.append("FG-1: binary hash mismatch")
    sanity_ok = all(arm["repeat_identical"] and arm["exit_a"] == 0
                    and arm["exit_b"] == 0
                    for arm in run_manifest["sanity_arms"].values())
    guards["FG-2_sanity_determinism"] = {
        "arms": {name: arm["repeat_identical"]
                 for name, arm in run_manifest["sanity_arms"].items()},
        "ok": sanity_ok,
    }
    if not sanity_ok:
        fatal.append("FG-2: sanity determinism failed")
    cells_ok = all(cell["repeat_identical"] and cell["exit_a"] == 0
                   and cell["exit_b"] == 0
                   for cell in run_manifest["cells"].values())
    guards["FG-3_FG-5_cells"] = {
        "cells": {name: cell["repeat_identical"]
                  for name, cell in run_manifest["cells"].items()},
        "ok": cells_ok,
    }
    if not cells_ok:
        fatal.append("FG-3/FG-5: cell determinism or exit failure")

    # Simulation cell processing (the .a repeat; FG-3 asserts a == b).
    sim: dict[str, dict] = {}
    conservation_ok = True
    for name, framing in FRAMING.items():
        wire, payload, rate_bps, cp = framing
        bins = load_sim_bins(run_root / f"{name}.a.csv")
        counts = parse_log_counts(run_root / f"{name}.a.log")
        record: dict[str, object] = dict(counts)
        bound = packet_bin_bound_bytes(wire, payload, rate_bps)
        grouped = aggregate_100us(bins)
        worst = max(grouped.values())
        record["max_100us_bin_bytes"] = worst
        record["bin_bound_bytes"] = bound
        if worst > bound:
            conservation_ok = False
            fatal.append(f"FG-4a: {name} bin {worst} above bound {bound}")
        if counts["delivered_payload_bytes"] != counts["delivered"] * payload:
            conservation_ok = False
            fatal.append(f"FG-4c: {name} payload arithmetic mismatch")
        if name.startswith("cal-solo"):
            if counts["dropped"] != 0 or counts["delivered"] != counts["injected"]:
                conservation_ok = False
                fatal.append(f"FG-4b: {name} dropped or lost packets")
            bin_ps = 1_000_000
            steady_bytes = sum(
                sum(flows.values()) for start, flows in bins.items()
                if 50_000_000_000 <= start < 150_000_000_000)
            record["steady_rate_bytes_per_s"] = steady_bytes * 10
            record["steady_over_cp"] = rnd(
                float(Fraction(steady_bytes * 10) / cp), 6)
            crossings = chunk_crossings(bins, bin_ps, flow=0)
            record["first_crossing_bin_end_ps"] = crossings[0]
            deltas = [crossings[i] - crossings[i - 1]
                      for i in range(1, len(crossings))]
            record["chunk_delta_p50_us"] = rnd(
                statistics.median(deltas) / 1e6, 3)
            record["chunk_count"] = len(crossings)
            settle = [
                index for index in range(1, 2000)
                if abs(grouped.get(index, 0) * BIN_100US_TO_BYTES_PER_S
                       - float(cp)) / float(cp) > 0.01]
            record["settle_violating_bins"] = settle[:10]
            record["settle_ok"] = not settle
        else:
            final_bins = range(950, 1000)
            aggregate_mean = statistics.mean(
                grouped.get(index, 0) for index in final_bins)
            record["final50_aggregate_over_cp"] = rnd(
                aggregate_mean * BIN_100US_TO_BYTES_PER_S / float(cp), 6)
            starved, means = [], {}
            for flow in range(ECHO_DEGREE[name]):
                per_flow = per_flow_100us(bins, flow)
                flow_mean = statistics.mean(
                    per_flow.get(index, 0) for index in final_bins)
                means[str(flow)] = rnd(
                    flow_mean * BIN_100US_TO_BYTES_PER_S / float(cp), 6)
                if all(per_flow.get(index, 0) * BIN_100US_TO_BYTES_PER_S
                       < 0.02 * float(cp) for index in final_bins):
                    starved.append(flow)
            record["final50_flow_over_cp"] = means
            record["starved_flows"] = starved
        sim[name] = record

    guards["FG-4_conservation"] = {"ok": conservation_ok}

    # Simulation relations.
    rows: list[dict[str, object]] = []

    def add_row(class_name: str, name: str, quantity: str, expected: str,
                observed: object, ok: bool) -> None:
        rows.append({"class": class_name, "row": name, "quantity": quantity,
                     "expected": expected, "observed": observed,
                     "verdict": "pass" if ok else "fail"})

    for cell, expect in CROSS_BIN_END_PS.items():
        observed = sim[cell]["first_crossing_bin_end_ps"]
        add_row("exact", {"cal-solo-a": "EX-1", "cal-solo-100g": "EX-2",
                          "cal-solo-4160": "EX-3"}[cell],
                f"{cell} first-chunk crossing bin end",
                f"== {expect} ps", observed, observed == expect)

    ratio = sim["cal-solo-a"]["steady_over_cp"]
    add_row("behavioral", "BE-1", "cal-solo-a steady rate over C_p",
            "[0.99, 1.0005]", ratio, 0.99 <= ratio <= 1.0005)
    p50 = sim["cal-solo-a"]["chunk_delta_p50_us"]
    add_row("behavioral", "BE-2", "cal-solo-a chunk delta p50 us",
            "[335, 345]", p50, 335.0 <= p50 <= 345.0)
    settle_ok = all(sim[cell]["settle_ok"] for cell in
                    ("cal-solo-a", "cal-solo-b", "cal-solo-4160",
                     "cal-solo-100g"))
    add_row("behavioral", "BE-3", "solo cells settle within one 100-us bin",
            "bins 1..1999 within 1 percent of C_p", settle_ok, settle_ok)

    echo2 = sim["cal-echo-2"]
    be4 = (0.90 <= echo2["final50_aggregate_over_cp"] <= 1.005
           and len(echo2["starved_flows"]) == 1 and echo2["dropped"] > 0)
    add_row("behavioral", "BE-4", "cal-echo-2 admission-capture signature",
            "aggregate in [0.90, 1.005] C_p, exactly one starved flow, drops > 0",
            {"aggregate_over_cp": echo2["final50_aggregate_over_cp"],
             "starved_flows": echo2["starved_flows"],
             "dropped": echo2["dropped"]}, be4)
    echo4 = sim["cal-echo-4"]
    top_share = max(echo4["final50_flow_over_cp"].values())
    be5 = (0.90 <= echo4["final50_aggregate_over_cp"] <= 1.005
           and len(echo4["starved_flows"]) >= 1 and top_share >= 0.5
           and echo4["dropped"] > 0)
    add_row("behavioral", "BE-5", "cal-echo-4 admission-capture signature",
            "aggregate in band, >= 1 starved flow, top flow >= 0.5 C_p, drops > 0",
            {"aggregate_over_cp": echo4["final50_aggregate_over_cp"],
             "starved_flows": echo4["starved_flows"],
             "top_flow_over_cp": top_share,
             "dropped": echo4["dropped"]}, be5)

    framing_ratio = rnd(sim["cal-solo-4160"]["steady_rate_bytes_per_s"]
                        / sim["cal-solo-a"]["steady_rate_bytes_per_s"], 6)
    add_row("behavioral", "BE-6", "framing-shift steady-rate ratio",
            "[0.9935, 0.9955]", framing_ratio,
            0.9935 <= framing_ratio <= 0.9955)
    ratio_100g = sim["cal-solo-100g"]["steady_over_cp"]
    add_row("behavioral", "BE-7", "cal-solo-100g steady rate over C_p(100G)",
            "[0.99, 1.0005]", ratio_100g, 0.99 <= ratio_100g <= 1.0005)

    bins_a = load_sim_bins(run_root / "cal-solo-a.a.csv")
    bins_b = load_sim_bins(run_root / "cal-solo-b.a.csv")
    series_a = [(start, sum(flows.values())) for start, flows
                in sorted(bins_a.items())]
    series_b = [(start, sum(flows.values())) for start, flows
                in sorted(bins_b.items())]
    add_row("structural", "ST-1", "port symmetry of the delivered series",
            "identical (bin, payload) sequences", series_a == series_b,
            series_a == series_b)

    # Measured side, re-derived from tracked bytes.
    s1_times = dataset.series("s1-stream/s1-stream_flow0_dest.csv.gz")
    i2_f0 = dataset.series("i2-incast/i2-incast_flow0_dest.csv.gz")
    i2_f1 = dataset.series("i2-incast/i2-incast_flow1_dest.csv.gz")
    j_f0 = dataset.series("j2x-join/j2x-join_flow0_dest.csv.gz")
    j_f1 = dataset.series("j2x-join/j2x-join_flow1_dest.csv.gz")
    s1_stats = dataset.stats("stats/s1-stream_stats.json")
    i2_stats = dataset.stats("stats/i2-incast_stats.json")
    j_stats = dataset.stats("stats/j2x-join_stats.json")
    mx_a2gh = dataset.stats("stats/mx-a2gh_stats.json")
    mx_gh2a = dataset.stats("stats/mx-gh2a_stats.json")

    measured = {
        "anchor_s1_rate_gbps": window_rate_gbps(s1_times, 280, 300),
        "anchor_s1_p50_us": window_p50_us(s1_times, 280, 300),
        "anchor_j2x_st0_rate_gbps": window_rate_gbps(j_f0, 40, 60),
        "anchor_j2x_st0_p50_us": window_p50_us(j_f0, 40, 60),
        "i2_f0_rate_gbps": window_rate_gbps(i2_f0, 160, 180),
        "i2_f0_p50_us": window_p50_us(i2_f0, 160, 180),
        "i2_f1_rate_gbps": window_rate_gbps(i2_f1, 160, 180),
        "i2_f1_p50_us": window_p50_us(i2_f1, 160, 180),
        "j2x_st1_f0_rate_gbps": window_rate_gbps(j_f0, 160, 180),
        "j2x_st1_f0_p50_us": window_p50_us(j_f0, 160, 180),
        "j2x_st1_f1_rate_gbps": window_rate_gbps(j_f1, 160, 180),
        "j2x_st1_f1_p50_us": window_p50_us(j_f1, 160, 180),
        "i2_convergence_s": i2_stats["stages"][0]["convergence_s"],
        "j2x_st1_convergence_s": j_stats["stages"][1]["convergence_s"],
    }
    measured = {key: rnd(value, 6) for key, value in measured.items()}

    # Internal consistency against the published stats (exact re-derivation).
    published = {
        "anchor_s1_rate_gbps":
            s1_stats["stages"][0]["steady_per_flow_gbps"]["0"],
        "anchor_j2x_st0_rate_gbps":
            j_stats["stages"][0]["steady_per_flow_gbps"]["0"],
        "i2_f0_rate_gbps": i2_stats["stages"][0]["steady_per_flow_gbps"]["0"],
        "i2_f1_rate_gbps": i2_stats["stages"][0]["steady_per_flow_gbps"]["1"],
        "j2x_st1_f0_rate_gbps":
            j_stats["stages"][1]["steady_per_flow_gbps"]["0"],
        "j2x_st1_f1_rate_gbps":
            j_stats["stages"][1]["steady_per_flow_gbps"]["1"],
    }
    stats_agreement = {
        key: abs(measured[key] - value) < 1e-6
        for key, value in published.items()}
    if not all(stats_agreement.values()):
        fatal.append("FG-6: re-derived steady rates disagree with stats")

    # Composed rows. The fabric terms come from the simulation: R_fab is the
    # simulated solo steady rate (distinct-port mapping, expectations.md) and
    # T_fab_lat is the frozen hand value that EX-1 just verified.
    ex1_ok = any(row["row"] == "EX-1" and row["verdict"] == "pass"
                 for row in rows)
    r_fab_gbps = sim["cal-solo-a"]["steady_rate_bytes_per_s"] / 1e9
    h_rate = {
        "gpu105-gpu102": measured["anchor_s1_rate_gbps"],
        "gpu103-gpu102": measured["anchor_j2x_st0_rate_gbps"],
    }
    h_time = {
        "gpu105-gpu102": rnd(
            measured["anchor_s1_p50_us"] - T_FAB_LAT_PS / 1e6, 3),
        "gpu103-gpu102": rnd(
            measured["anchor_j2x_st0_p50_us"] - T_FAB_LAT_PS / 1e6, 3),
    }
    composed_rate = {
        pair: rnd(min(value, r_fab_gbps), 6) for pair, value in h_rate.items()}
    composed_p50 = {
        pair: rnd(h_time[pair] + T_FAB_LAT_PS / 1e6, 3) for pair in h_time}

    co_rows: list[dict[str, object]] = []

    def add_co(name: str, quantity: str, composed: float, observed: float,
               band: tuple[float, float], unit: str) -> None:
        ratio_value = rnd(composed / observed, 6)
        co_rows.append({
            "row": name, "quantity": quantity, "composed": composed,
            "measured": observed, "composed_over_measured": ratio_value,
            "band": list(band), "unit": unit,
            "verdict": "pass" if band[0] <= ratio_value <= band[1] else "fail",
        })

    rate_band = (0.85, 1.15)
    p50_band = (1 / 1.2, 1.2)
    add_co("CO-1", "i2 flow 0 steady rate",
           composed_rate["gpu103-gpu102"], measured["i2_f0_rate_gbps"],
           rate_band, "GB/s")
    add_co("CO-2", "i2 flow 1 steady rate",
           composed_rate["gpu105-gpu102"], measured["i2_f1_rate_gbps"],
           rate_band, "GB/s")
    add_co("CO-3", "j2x stage-1 flow 0 steady rate",
           composed_rate["gpu103-gpu102"], measured["j2x_st1_f0_rate_gbps"],
           rate_band, "GB/s")
    add_co("CO-4", "j2x stage-1 flow 1 steady rate",
           composed_rate["gpu105-gpu102"], measured["j2x_st1_f1_rate_gbps"],
           rate_band, "GB/s")
    join_ratio = rnd(measured["j2x_st1_f0_rate_gbps"]
                     / measured["anchor_j2x_st0_rate_gbps"], 6)
    co_rows.append({
        "row": "CO-5", "quantity": "join-unharmed flow 0 stage1/stage0",
        "composed": 1.0, "measured": join_ratio,
        "composed_over_measured": rnd(1.0 / join_ratio, 6),
        "band": [0.90, 1.10], "unit": "ratio",
        "verdict": "pass" if 0.90 <= join_ratio <= 1.10 else "fail",
    })
    add_co("CO-6", "i2 flow 0 steady-window p50",
           composed_p50["gpu103-gpu102"], measured["i2_f0_p50_us"],
           p50_band, "us")
    add_co("CO-7", "i2 flow 1 steady-window p50",
           composed_p50["gpu105-gpu102"], measured["i2_f1_p50_us"],
           p50_band, "us")
    add_co("CO-8", "j2x stage-1 flow 0 steady-window p50",
           composed_p50["gpu103-gpu102"], measured["j2x_st1_f0_p50_us"],
           p50_band, "us")
    add_co("CO-9", "j2x stage-1 flow 1 steady-window p50",
           composed_p50["gpu105-gpu102"], measured["j2x_st1_f1_p50_us"],
           p50_band, "us")
    co_rows.append({
        "row": "CO-10", "quantity": "staggered-join composed settling",
        "composed": "fabric settling, one 100-us bin" if settle_ok else
        "fabric transient present",
        "measured": measured["j2x_st1_convergence_s"],
        "band": ["<= 1 s"], "unit": "s",
        "verdict": "pass"
        if settle_ok and measured["j2x_st1_convergence_s"] <= 1.0 else "fail",
    })

    # Composition validity condition.
    validity = {
        "t_fab_rate_us": rnd(CHUNK_BYTES / float(CP_9038) * 1e6, 3),
        "t_fab_lat_us": rnd(T_FAB_LAT_PS / 1e6, 3),
        "binding_stage_is_stack": all(
            CHUNK_BYTES / float(CP_9038) * 1e6 < CHUNK_BYTES / (rate * 1e9)
            * 1e6 for rate in h_rate.values()),
        "ex1_confirmed_t_fab_lat": ex1_ok,
    }

    # Derived, excluded and refuted-by-construction quantities (unscored).
    derived = {
        "i2_aggregate_composed_gbps": rnd(
            composed_rate["gpu103-gpu102"] + composed_rate["gpu105-gpu102"], 6),
        "i2_aggregate_measured_gbps": rnd(
            measured["i2_f0_rate_gbps"] + measured["i2_f1_rate_gbps"], 6),
        "j2x_st1_aggregate_composed_gbps": rnd(
            composed_rate["gpu103-gpu102"] + composed_rate["gpu105-gpu102"], 6),
        "j2x_st1_aggregate_measured_gbps": rnd(
            measured["j2x_st1_f0_rate_gbps"]
            + measured["j2x_st1_f1_rate_gbps"], 6),
        "i2_over_solo_composed": rnd(
            (composed_rate["gpu103-gpu102"] + composed_rate["gpu105-gpu102"])
            / composed_rate["gpu105-gpu102"], 6),
        "i2_over_solo_measured_stage_def": rnd(
            (measured["i2_f0_rate_gbps"] + measured["i2_f1_rate_gbps"])
            / measured["anchor_s1_rate_gbps"], 6),
        "i2_convergence_measured_s": measured["i2_convergence_s"],
        "i2_convergence_composed": "not producible by the frozen rule",
        "h_time_us": h_time,
        "h_rate_gbps": h_rate,
        "endpoint_share": {
            pair: rnd(h_time[pair] / (h_time[pair] + T_FAB_LAT_PS / 1e6), 4)
            for pair in h_time},
        "mx_excluded": {
            "reason": "endpoint-stack-owned asymmetry, no independent "
                      "sustained solo anchor, GH nodes outside the declared "
                      "instance (expectations.md)",
            "a2gh_p50_us": mx_a2gh["flows"]["0"]["delta_us"]["p50"],
            "gh2a_p50_us": mx_gh2a["flows"]["0"]["delta_us"]["p50"],
            "a2gh_endpoint_floor_us": rnd(
                mx_a2gh["flows"]["0"]["delta_us"]["p50"]
                - T_FAB_LAT_PS / 1e6, 3),
            "gh2a_endpoint_floor_us": rnd(
                mx_gh2a["flows"]["0"]["delta_us"]["p50"]
                - T_FAB_LAT_PS / 1e6, 3),
        },
    }

    def jain(values: list[float]) -> float:
        return rnd(sum(values) ** 2 / (len(values) * sum(v * v for v in values)), 6)

    derived["i2_jain_composed"] = jain(list(composed_rate.values()))
    derived["i2_jain_measured"] = jain(
        [measured["i2_f0_rate_gbps"], measured["i2_f1_rate_gbps"]])
    derived["j2x_st1_jain_composed"] = derived["i2_jain_composed"]
    derived["j2x_st1_jain_measured"] = jain(
        [measured["j2x_st1_f0_rate_gbps"], measured["j2x_st1_f1_rate_gbps"]])

    summary = {
        "study": "merlin_ss_fabric_calibration_v1",
        "dataset_manifest_sha256": dataset.manifest_sha256,
        "binary_sha256": run_manifest["binary_sha256"],
        "fatal_guards": guards,
        "fatal_failures": fatal,
        "void": bool(fatal),
        "simulation_rows": rows,
        "conditional_rows": co_rows,
        "composition_validity": validity,
        "measured": measured,
        "stats_agreement": stats_agreement,
        "derived_unscored": derived,
        "simulation_cells": sim,
        "dataset_files_verified": dataset.verified,
    }
    out_path = out_dir / "calibration_summary.json"
    out_path.write_text(
        json.dumps(summary, indent=1, sort_keys=True) + "\n", encoding="utf-8")

    print(f"void: {summary['void']}")
    for row in rows:
        print(f"  {row['row']:5} [{row['class']:10}] {row['verdict']:4} "
              f"{row['quantity']}: {row['observed']} (expected {row['expected']})")
    for row in co_rows:
        print(f"  {row['row']:5} [conditional] {row['verdict']:4} "
              f"{row['quantity']}: composed {row['composed']} measured "
              f"{row['measured']}")
    sim_pass = sum(1 for row in rows if row["verdict"] == "pass")
    co_pass = sum(1 for row in co_rows if row["verdict"] == "pass")
    print(f"simulation rows: {sim_pass}/{len(rows)} pass; "
          f"conditional rows: {co_pass}/{len(co_rows)} confirm")
    print(f"wrote {out_path}")
    return 1 if fatal else 0


if __name__ == "__main__":
    raise SystemExit(main())
