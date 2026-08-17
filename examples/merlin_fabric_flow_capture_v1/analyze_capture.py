#!/usr/bin/env python3
"""Analyze the Merlin fabric flow capture against the freeze-1 expectations.

Implements exactly the frozen definitions of expectations.md:

- goodput series: 1-second bins over destination per-chunk completions; a
  chunk's bytes land in the bin its completion timestamp falls in;
- steady value of a stage: mean goodput over the final 20 seconds of the
  stage;
- per-chunk service time: the inter-completion delta between consecutive
  chunks of one flow at the destination;
- convergence after a join: the smallest tau such that from T_join + tau to
  the end of the stage every active flow's 1-second-bin goodput lies within
  25 percent of that flow's stage steady value;
- E-T-1: every rank's in-job tracer floor p50 under 1.0 percent, and floor
  p95 minus p5 under 0.3 percent, of the cell's median per-chunk service
  time (all destination inter-completion deltas pooled).

The script is deterministic, reads only files under --capture-root, and
writes per-cell stats JSON plus a verdict table. Nothing here mutates the
freeze. Cells with no completed run are reported unevaluated.
"""

from __future__ import annotations

import argparse
import csv
import glob
import gzip
import itertools
import json
import math
import os
import re
import statistics
import sys

GB = 1e9

# Frozen cell inventory: cell name -> (flows, window_s, offsets_s, ceiling
# for the destination aggregate in GB/s).
CELLS = {
    "s1-stream": (1, 300.0, [0.0], 26.78),
    "i2-incast": (2, 180.0, [0.0, 0.0], 26.78),
    "i3-incast": (3, 180.0, [0.0, 0.0, 0.0], 26.78),
    "j3-join": (3, 240.0, [0.0, 60.0, 120.0], 26.78),
    # Post-specified cell, added after freeze 1: descriptive statistics
    # only, no scored relation reads it.
    "j2x-join": (2, 180.0, [0.0, 60.0], 26.78),
    "i4-incast": (4, 180.0, [0.0, 0.0, 0.0, 0.0], 26.78),
    "x4-incast": (4, 180.0, [0.0, 0.0, 0.0, 0.0], 26.78),
    "mx-gh2a": (1, 60.0, [0.0], 26.78),
    "mx-a2gh": (1, 60.0, [0.0], 100.0),
}

PER_FLOW_CEILING_GBPS = 25.0
STEADY_TAIL_S = 20.0
CONV_TOL = 0.25


def pctl(sorted_vals, q):
    if not sorted_vals:
        return float("nan")
    pos = q * (len(sorted_vals) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = pos - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


def open_maybe_gz(path):
    if path.endswith(".gz"):
        return gzip.open(path, "rt")
    return open(path)


def load_series(path):
    """Return (chunk_bytes, [t_ns_since_t0 ...]) for one series CSV."""
    times = []
    chunk_bytes = None
    with open_maybe_gz(path) as f:
        for row in csv.DictReader(f):
            chunk_bytes = int(row["chunk_bytes"])
            times.append(int(row["t_ns_since_t0"]))
    return chunk_bytes, times


def bin_goodput(times_ns, chunk_bytes, n_bins):
    """1-second bins of bytes per second."""
    bins = [0.0] * n_bins
    for t in times_ns:
        b = int(t // 1_000_000_000)
        if 0 <= b < n_bins:
            bins[b] += chunk_bytes
    return bins


def stage_bounds(cell):
    """Stage boundaries in seconds for a cell, from its frozen offsets."""
    _, window, offsets, _ = CELLS[cell]
    starts = sorted(set(offsets))
    bounds = []
    for i, s in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else window
        bounds.append((s, end))
    return bounds


def steady_mean(bins, start_s, end_s):
    """Mean of whole-second bins over the final STEADY_TAIL_S of a stage."""
    lo = max(int(end_s - STEADY_TAIL_S), int(start_s))
    hi = int(end_s)
    vals = bins[lo:hi]
    return sum(vals) / len(vals) if vals else float("nan")


def jain(values):
    if not values:
        return float("nan")
    num = sum(values) ** 2
    den = len(values) * sum(v * v for v in values)
    return num / den if den > 0 else float("nan")


def flow_deltas_ns(times_ns):
    return [b - a for a, b in itertools.pairwise(times_ns)]


def analyze_cell(cell, job_dir):
    """Compute the frozen descriptive statistics for one cell run."""
    n_flows, window, offsets, agg_ceiling = CELLS[cell]
    n_bins = int(window)
    prefix = os.path.join(job_dir, cell)
    flows = {}
    for f in range(n_flows):
        for suffix in (".csv", ".csv.gz"):
            p = f"{prefix}_flow{f}_dest{suffix}"
            if os.path.exists(p):
                chunk_bytes, times = load_series(p)
                flows[f] = (chunk_bytes, times)
                break
    if len(flows) != n_flows:
        return None

    summary = {}
    sp = f"{prefix}_summary.json"
    if os.path.exists(sp):
        with open(sp) as f:
            summary = json.load(f)
    rank_meta = []
    for p in sorted(glob.glob(f"{prefix}_rank*.json")):
        with open(p) as f:
            rank_meta.append(json.load(f))

    out = {
        "cell": cell,
        # Two path components only, so the stats carry no machine-local
        # absolute path (portability rule).
        "job_dir": os.path.join(
            os.path.basename(os.path.dirname(job_dir)),
            os.path.basename(job_dir)),
        "hosts": sorted({m.get("host") for m in rank_meta if m.get("host")}),
        "window_s": window,
        "offsets_s": offsets,
        "flows": {},
        "guards": {},
    }

    per_flow_bins = {}
    all_deltas = []
    for f, (chunk_bytes, times) in flows.items():
        bins = bin_goodput(times, chunk_bytes, n_bins)
        per_flow_bins[f] = bins
        deltas = flow_deltas_ns(times)
        all_deltas.extend(deltas)
        d_us = sorted(d / 1000.0 for d in deltas)
        d_us_excl10 = sorted(
            (b - a) / 1000.0 for a, b in itertools.pairwise(times)
            if a > 10_000_000_000)
        out["flows"][f] = {
            "chunks": len(times),
            "bytes": len(times) * chunk_bytes,
            "first_completion_s": times[0] / 1e9 if times else None,
            "last_completion_s": times[-1] / 1e9 if times else None,
            "delta_us_excl10_p95_over_p50": (
                pctl(d_us_excl10, 0.95) / pctl(d_us_excl10, 0.50)
                if d_us_excl10 and pctl(d_us_excl10, 0.50) > 0 else None),
            "delta_us": {
                "n": len(d_us),
                "mean": statistics.fmean(d_us) if d_us else None,
                "sd": statistics.stdev(d_us) if len(d_us) > 1 else None,
                "p5": pctl(d_us, 0.05),
                "p25": pctl(d_us, 0.25),
                "p50": pctl(d_us, 0.50),
                "p75": pctl(d_us, 0.75),
                "p95": pctl(d_us, 0.95),
                "p99": pctl(d_us, 0.99),
                "max": d_us[-1] if d_us else None,
            },
        }

    agg_bins = [sum(per_flow_bins[f][i] for f in per_flow_bins)
                for i in range(n_bins)]
    out["aggregate_bins_gbps"] = [round(b / GB, 6) for b in agg_bins]
    out["per_flow_bins_gbps"] = {
        f: [round(b / GB, 6) for b in bins] for f, bins in per_flow_bins.items()
    }

    # Stages, steady values, convergence.
    bounds = stage_bounds(cell)
    stages = []
    for si, (s0, s1) in enumerate(bounds):
        active = [f for f in range(n_flows) if offsets[f] <= s0]
        st = {
            "stage": si,
            "start_s": s0,
            "end_s": s1,
            "active_flows": active,
            "steady_aggregate_gbps":
                steady_mean(agg_bins, s0, s1) / GB,
            "steady_per_flow_gbps": {
                f: steady_mean(per_flow_bins[f], s0, s1) / GB for f in active
            },
        }
        st["jain_steady"] = jain(list(st["steady_per_flow_gbps"].values()))
        # Convergence: only meaningful for stages opened by a join (si>0) or
        # reported for stage 0 as settling from cold start.
        conv = None
        last_bad = None
        for t in range(int(s0), int(s1)):
            for f in active:
                sf = st["steady_per_flow_gbps"][f] * GB
                if math.isnan(sf) or sf == 0:
                    continue
                if abs(per_flow_bins[f][t] - sf) > CONV_TOL * sf:
                    last_bad = t
        if last_bad is None:
            conv = 0.0
        elif last_bad + 1 < s1:
            conv = (last_bad + 1) - s0
        st["convergence_s"] = conv  # None means not converged within stage
        stages.append(st)
    out["stages"] = stages

    # Guards computable here.
    g2_ok = True
    for f, (_, times) in flows.items():
        if any(t <= 0 for t in times[:1]) or any(
                b <= a for a, b in itertools.pairwise(times)):
            g2_ok = False
    g3_ok = True
    for fl in summary.get("flows", []):
        if (fl["dest_chunks"] != fl["source_chunks"]
                or fl["mismatches"] != 0
                or fl["final_probe_mismatches"] != 0
                or fl["sentinel_seen"] != 1):
            g3_ok = False
    # G5: skip the first and last bin of every stage boundary.
    skip = set()
    for s0, s1 in bounds:
        skip.add(int(s0))
        skip.add(int(s1) - 1)
    g5_ok = True
    for f, bins in per_flow_bins.items():
        for i, b in enumerate(bins):
            if i in skip:
                continue
            if b / GB > PER_FLOW_CEILING_GBPS:
                g5_ok = False
    for i, b in enumerate(agg_bins):
        if i in skip:
            continue
        if b / GB > agg_ceiling:
            g5_ok = False
    # Homogeneous jobs write transport_summary.txt at the job root; the
    # heterogeneous mx job records its transport section inside each side's
    # side.txt (and additionally enforces the transport half of G1 in-job
    # before exiting zero). The packager copies both kinds of file into the
    # dataset, so the tracked dataset alone re-derives this verdict.
    transport = ""
    for tp in [os.path.join(job_dir, "transport_summary.txt"),
               *sorted(glob.glob(os.path.join(job_dir, "*", "side.txt")))]:
        if os.path.exists(tp):
            with open(tp) as f:
                transport += f.read()
    g1_ok = ("Using network Socket" in transport) and ("GDR 1" not in transport)

    # Port-inventory, exclusive-use and placement evidence. Sources: per-rank
    # guards_before.txt (run tree rank_*/ layout or packaged guards/rank_*/
    # layout) for homogeneous cells, side.txt per side for the mx cells.
    guard_files = sorted(
        glob.glob(os.path.join(job_dir, "rank_*", "guards_before.txt"))
        + glob.glob(os.path.join(job_dir, "guards", "rank_*",
                                 "guards_before.txt")))
    side_files = sorted(glob.glob(os.path.join(job_dir, "*", "side.txt")))
    port_counts = []
    ib_counts = []
    foreign_rows = 0
    gpu_uuids = set()
    evidence_units = 0
    for gf in guard_files:
        with open(gf) as f:
            text = f.read()
        evidence_units += 1
        port_counts += [int(x) for x in
                        re.findall(r"^cassini_port_count=(\d+)", text,
                                   re.MULTILINE)]
        ib_counts += [int(x) for x in
                      re.findall(r"^ib_device_count=(\d+)", text,
                                 re.MULTILINE)]
        g4_section = text.split("=== G4 foreign processes ===")
        if len(g4_section) > 1:
            body = g4_section[1].split("=== G1 host NIC inventory ===")[0]
            foreign_rows += len(re.findall(r"^\d+,", body, re.MULTILINE))
        gpu_uuids |= set(re.findall(r"^gpu_uuid=(\S+)", text, re.MULTILINE))
    for sf in side_files:
        with open(sf) as f:
            text = f.read()
        evidence_units += 1
        before = text.split("=== guard evidence before ===")
        # The guard-before block of a side.txt ends where the build starts.
        scope = before[1].split("builds ok")[0] if len(before) > 1 else text
        port_counts += [int(x) for x in
                        re.findall(r"cassini_port_count=(\d+)", scope)]
        ib_counts += [int(x) for x in
                      re.findall(r"ib_device_count=(\d+)", scope)]
        # Foreign compute processes would appear as CSV data rows under the
        # "pid, process_name" header inside this block.
        apps = scope.split("pid, process_name")
        if len(apps) > 1:
            foreign_rows += len(re.findall(
                r"^\d+,", apps[1].split("cassini_port_count")[0],
                re.MULTILINE))
        gpu_uuids |= set(re.findall(r"GPU-[0-9a-f-]+", scope))
    ranks_expected = len([m for m in rank_meta
                          if m.get("role") in ("dest", "source")])
    g1_ports_ok = (evidence_units > 0
                   and all(c == 4 for c in port_counts)
                   and len(port_counts) >= evidence_units
                   and all(c == 0 for c in ib_counts))
    g4_ok = evidence_units > 0 and foreign_rows == 0
    g6_ok = (evidence_units > 0
             and len(gpu_uuids) >= max(ranks_expected, 1)
             and len({m.get("host") for m in rank_meta
                      if m.get("role") in ("dest", "source")})
             >= min(ranks_expected, len(rank_meta)))
    out["guards"] = {
        "G1_socket_transport": g1_ok,
        "G1_port_inventory": g1_ports_ok,
        "G2_timer_sanity": g2_ok,
        "G3_conservation": g3_ok,
        "G4_exclusive": g4_ok,
        "G5_ceilings": g5_ok,
        "G6_placement": g6_ok,
    }

    # E-T-1 inputs.
    cell_median_us = (
        pctl(sorted(d / 1000.0 for d in all_deltas), 0.5) if all_deltas
        else float("nan"))
    out["cell_median_chunk_service_us"] = cell_median_us
    floors = []
    for m in rank_meta:
        fl = m.get("tracer_floor_us", {})
        floors.append({
            "rank": m.get("rank"),
            "host": m.get("host"),
            "role": m.get("role"),
            "p50": fl.get("p50"),
            "spread_p95_minus_p5": (fl.get("p95", 0) - fl.get("p5", 0)),
        })
    out["tracer_floors"] = floors
    ok = True
    for fl in floors:
        if fl["role"] == "idle":
            continue
        if not (fl["p50"] < 0.01 * cell_median_us
                and fl["spread_p95_minus_p5"] < 0.003 * cell_median_us):
            ok = False
    out["tracer_criterion_ok"] = ok
    return out


def relation(rid, verdict, measured, band):
    return {"relation": rid, "verdict": verdict, "measured": measured,
            "band": band}


def evaluate_relations(cells):
    """Evaluate the 18 frozen relations. cells maps cell name to stats."""
    rel = []

    def steady_excl10(cell):
        st = cells[cell]
        bins = [b * GB for b in st["aggregate_bins_gbps"]]
        vals = bins[10:int(st["window_s"])]
        return sum(vals) / len(vals) / GB if vals else float("nan")

    # E-T-1 over every captured FROZEN cell; the post-specified j2x cell is
    # reported in stats but enters no scored relation.
    captured = [c for name, c in cells.items()
                if c is not None and name != "j2x-join"]
    if captured:
        ok = all(c["tracer_criterion_ok"] for c in captured)
        rel.append(relation(
            "E-T-1", "pass" if ok else "FAIL",
            {c["cell"]: c["tracer_criterion_ok"] for c in captured},
            "floor p50 under 1.0 percent and spread under 0.3 percent"))
    else:
        rel.append(relation("E-T-1", "unevaluated", None, ""))

    s1 = cells.get("s1-stream")
    s1_steady = steady_excl10("s1-stream") if s1 else None
    if s1:
        rel.append(relation(
            "E-S-1", "pass" if 2.0 <= s1_steady <= 5.5 else "FAIL",
            round(s1_steady, 4), "[2.0, 5.5] GB/s"))
        f0 = s1["flows"][0]
        ratio = f0["delta_us_excl10_p95_over_p50"]
        rel.append(relation(
            "E-S-2", "pass" if ratio is not None and ratio <= 1.5 else "FAIL",
            round(ratio, 4) if ratio is not None else None,
            "p95 over p50 at most 1.5, region past 10 s"))
    else:
        rel.append(relation("E-S-1", "unevaluated", None, "[2.0, 5.5] GB/s"))
        rel.append(relation("E-S-2", "unevaluated", None, "<= 1.5"))

    def agg_steady(cell):
        st = cells[cell]
        return st["stages"][-1]["steady_aggregate_gbps"]

    i2 = cells.get("i2-incast")
    if i2:
        v = agg_steady("i2-incast")
        rel.append(relation("E-I-1", "pass" if 1.8 <= v <= 9.0 else "FAIL",
                            round(v, 4), "[1.8, 9.0] GB/s"))
        if s1:
            r = v / s1_steady
            rel.append(relation("E-I-2", "pass" if 0.9 <= r <= 2.2 else "FAIL",
                                round(r, 4), "[0.9, 2.2] x s1"))
        else:
            rel.append(relation("E-I-2", "unevaluated", None, "[0.9, 2.2]"))
    else:
        rel.append(relation("E-I-1", "unevaluated", None, "[1.8, 9.0] GB/s"))
        rel.append(relation("E-I-2", "unevaluated", None, "[0.9, 2.2]"))

    i3 = cells.get("i3-incast")
    if i3:
        v = agg_steady("i3-incast")
        rel.append(relation("E-I-3", "pass" if 1.8 <= v <= 12.0 else "FAIL",
                            round(v, 4), "[1.8, 12.0] GB/s"))
        if s1:
            r = v / s1_steady
            rel.append(relation("E-I-4", "pass" if 0.9 <= r <= 3.2 else "FAIL",
                                round(r, 4), "[0.9, 3.2] x s1"))
        else:
            rel.append(relation("E-I-4", "unevaluated", None, "[0.9, 3.2]"))
    else:
        rel.append(relation("E-I-3", "unevaluated", None, "[1.8, 12.0] GB/s"))
        rel.append(relation("E-I-4", "unevaluated", None, "[0.9, 3.2]"))

    fair_cells = [c for c in ("i2-incast", "i3-incast", "i4-incast",
                              "x4-incast") if cells.get(c)]
    if fair_cells:
        jains = {c: cells[c]["stages"][-1]["jain_steady"] for c in fair_cells}
        ok = all(j >= 0.6 for j in jains.values())
        rel.append(relation("E-I-5", "pass" if ok else "FAIL",
                            {c: round(j, 4) for c, j in jains.items()},
                            "Jain >= 0.6 in every incast cell that runs"))
    else:
        rel.append(relation("E-I-5", "unevaluated", None, "Jain >= 0.6"))

    for rid, cname, band in (("E-I-6", "i4-incast", (1.8, 14.0)),
                             ("E-I-7", "x4-incast", (1.8, 12.0))):
        if cells.get(cname):
            v = agg_steady(cname)
            rel.append(relation(
                rid, "pass" if band[0] <= v <= band[1] else "FAIL",
                round(v, 4), f"[{band[0]}, {band[1]}] GB/s"))
        else:
            rel.append(relation(rid, "unevaluated", None,
                                f"[{band[0]}, {band[1]}] GB/s"))

    j3 = cells.get("j3-join")
    if j3:
        st = j3["stages"]
        f0 = [s["steady_per_flow_gbps"].get(0,
              s["steady_per_flow_gbps"].get("0")) for s in st]
        if s1:
            r = f0[0] / s1_steady
            rel.append(relation("E-J-1",
                                "pass" if 0.7 <= r <= 1.3 else "FAIL",
                                round(r, 4), "[0.7, 1.3] x s1"))
        else:
            rel.append(relation("E-J-1", "unevaluated", None, "[0.7, 1.3]"))
        r1 = f0[1] / f0[0]
        rel.append(relation("E-J-2", "pass" if r1 <= 1.05 else "FAIL",
                            round(r1, 4), "<= 1.05"))
        r2 = f0[2] / f0[1]
        rel.append(relation("E-J-3", "pass" if r2 <= 1.05 else "FAIL",
                            round(r2, 4), "<= 1.05"))
        aggs = [s["steady_aggregate_gbps"] for s in st]
        r3 = min(aggs[1], aggs[2]) / aggs[0]
        rel.append(relation("E-J-4", "pass" if r3 >= 0.7 else "FAIL",
                            round(r3, 4), ">= 0.7 x stage 0"))
    else:
        for rid, band in (("E-J-1", "[0.7, 1.3]"), ("E-J-2", "<= 1.05"),
                          ("E-J-3", "<= 1.05"), ("E-J-4", ">= 0.7")):
            rel.append(relation(rid, "unevaluated", None, band))

    gh2a = cells.get("mx-gh2a")
    a2gh = cells.get("mx-a2gh")
    v_gh2a = steady_excl10("mx-gh2a") if gh2a else None
    v_a2gh = steady_excl10("mx-a2gh") if a2gh else None
    if gh2a:
        rel.append(relation("E-M-1",
                            "pass" if 0.7 <= v_gh2a <= 3.5 else "FAIL",
                            round(v_gh2a, 4), "[0.7, 3.5] GB/s"))
    else:
        rel.append(relation("E-M-1", "unevaluated", None, "[0.7, 3.5] GB/s"))
    if a2gh:
        rel.append(relation("E-M-2",
                            "pass" if 2.0 <= v_a2gh <= 8.0 else "FAIL",
                            round(v_a2gh, 4), "[2.0, 8.0] GB/s"))
    else:
        rel.append(relation("E-M-2", "unevaluated", None, "[2.0, 8.0] GB/s"))
    # E-M-3 comes from the mx matrix probe, evaluated by the caller.
    if gh2a and a2gh:
        r = v_a2gh / v_gh2a
        rel.append(relation("E-M-4", "pass" if r >= 1.3 else "FAIL",
                            round(r, 4), ">= 1.3 (signed)"))
    else:
        rel.append(relation("E-M-4", "unevaluated", None, ">= 1.3"))
    return rel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture-root", default=None,
                    help="directory holding capture/<cell>/<jobid>/ trees")
    ap.add_argument("--dataset-root", default=None,
                    help="analyze the packaged dataset directory instead of"
                    " the raw capture tree; proves the tracked dataset alone"
                    " re-derives every published verdict")
    ap.add_argument("--out", required=True, help="stats output directory")
    ap.add_argument("--mx-matrix", default=None,
                    help="mx_matrix.json for E-M-3, if the mixed cell ran")
    args = ap.parse_args()

    if (args.capture_root is None) == (args.dataset_root is None):
        ap.error("give exactly one of --capture-root or --dataset-root")
    os.makedirs(args.out, exist_ok=True)
    cells = {}
    for cell in CELLS:
        stats = None
        if args.dataset_root is not None:
            job_dir = os.path.join(args.dataset_root, cell)
            if os.path.isdir(job_dir):
                stats = analyze_cell(cell, job_dir)
        else:
            base = os.path.join(args.capture_root, "capture",
                                "mx-pair" if cell.startswith("mx-")
                                else cell)
            for job_dir in sorted(glob.glob(os.path.join(base, "*"))):
                got = analyze_cell(cell, job_dir)
                if got is not None:
                    stats = got  # latest complete run wins
        cells[cell] = stats
        if stats is not None:
            with open(os.path.join(args.out, f"{cell}_stats.json"),
                      "w") as f:
                json.dump(stats, f, indent=1, sort_keys=True)

    rel = evaluate_relations(cells)
    if args.mx_matrix and os.path.exists(args.mx_matrix):
        with open(args.mx_matrix) as f:
            m = json.load(f)
        v = None
        for p in m.get("pairs", []):
            if p["bytes"] == 8:
                v = p["oneway_recv_isolated_median_us"]
        if v is not None:
            rel.append(relation("E-M-3", "pass" if 8 <= v <= 200 else "FAIL",
                                v, "[8, 200] us"))
        else:
            rel.append(relation("E-M-3", "unevaluated", None, "[8, 200] us"))
    else:
        rel.append(relation("E-M-3", "unevaluated", None, "[8, 200] us"))

    with open(os.path.join(args.out, "relations.json"), "w") as f:
        json.dump(rel, f, indent=1, sort_keys=True)

    guards_ok = True
    for c, st in cells.items():
        if st is None:
            continue
        for g, ok in st["guards"].items():
            if not ok:
                guards_ok = False
                print(f"GUARD VIOLATED {g} in {c}")
    n_pass = sum(1 for r in rel if r["verdict"] == "pass")
    n_fail = sum(1 for r in rel if r["verdict"] == "FAIL")
    n_un = sum(1 for r in rel if r["verdict"] == "unevaluated")
    print(f"cells captured: "
          f"{sorted(c for c, s in cells.items() if s is not None)}")
    for r in rel:
        print(f"  {r['relation']:>7} {r['verdict']:>12} "
              f"measured={r['measured']} band={r['band']}")
    print(f"relations: {n_pass} pass, {n_fail} fail, {n_un} unevaluated "
          f"of {len(rel)}; guards {'held' if guards_ok else 'VIOLATED'}")
    if not guards_ok:
        sys.exit(2)


if __name__ == "__main__":
    main()
