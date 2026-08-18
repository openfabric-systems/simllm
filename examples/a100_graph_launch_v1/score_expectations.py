"""Score the a100_graph_launch_v1 measurement against its frozen freeze.

Reads the raw harness output and `expectations.json`, evaluates the 15 scored
expectations and the 8 fatal guards exactly as written, and emits
`results.json`. It never edits the freeze and never invents a bound.

Usage:

    python examples/a100_graph_launch_v1/score_expectations.py \
        --raw <stage2_result.json> --identity <gpu_identity_before.csv> \
        --out <results.json>
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

STUDY = Path(__file__).resolve().parent
FREEZE = json.loads((STUDY / "expectations.json").read_text(encoding="utf-8"))

SUBSTRATE = FREEZE["substrate"]
PEAK = {1410: SUBSTRATE["peak_flops_1410"], 1275: SUBSTRATE["peak_flops_1275"]}
HBM_ROOF = SUBSTRATE["hbm_roof_bytes_per_second"]
MEMORY_CLOCK_MHZ = SUBSTRATE["memory_clock_mhz"]
MIN_STATIONARY = SUBSTRATE["min_stationary_blocks"]
K_REF = SUBSTRATE["reference_chain_length"]
TURING_EAGER_PS = FREEZE["comparison_points"]["turing_eager_host_ps"]

#: the tag whose eager launches are a pure host launch path, so its slope is
#: the quantity the Turing eager-host point measured with an empty kernel.
LAUNCH_TAG = "nop"
CV_CEILING = 0.04


def stationary(cell: dict) -> list[tuple[int, int]]:
    """Indices and clock states of the cell's clock-stationary blocks."""

    out = []
    for index, clocks in enumerate(cell["block_clocks"]):
        if clocks["sm_before"] != clocks["sm_after"]:
            continue
        if clocks["mem_before"] != MEMORY_CLOCK_MHZ or clocks["mem_after"] != MEMORY_CLOCK_MHZ:
            continue
        if clocks["th_before"] != 0 or clocks["th_after"] != 0:
            continue
        out.append((index, clocks["sm_before"]))
    return out


def reduce_cell(cell: dict) -> dict:
    blocks = stationary(cell)
    by_state: dict[int, list[int]] = {}
    for index, state in blocks:
        by_state.setdefault(state, []).append(index)
    scored_state = None
    for state, indices in sorted(by_state.items(), key=lambda kv: -len(kv[1])):
        if len(indices) >= MIN_STATIONARY:
            scored_state = state
            break

    reduced = {
        "tag": cell["tag"],
        "mode": cell["mode"],
        "length": cell["length"],
        "graph_nodes": cell["graph_nodes"],
        "graph_replays": cell["graph_replays"],
        "graph_instantiate_ms": cell["graph_instantiate_ms"],
        "flops": cell["flops"],
        "bytes": cell["bytes"],
        "host_before_sync": all(value == 1 for value in cell["host_before_sync"]),
        "stationary_by_state": {str(s): len(v) for s, v in by_state.items()},
        "scored_state": scored_state,
        "void_for_scoring": scored_state is None,
        "inner_kernel_ms": cell["inner_kernel_ms"],
    }
    if scored_state is None:
        return reduced

    indices = by_state[scored_state]
    makespans = [cell["makespan_ms"][i] * 1e-3 for i in indices]
    hosts = [cell["host_ms"][i] * 1e-3 for i in indices]
    makespan = statistics.fmean(makespans)
    reduced["makespan_s"] = makespan
    reduced["makespan_cv"] = statistics.pstdev(makespans) / makespan if makespan else 0.0
    reduced["period_s"] = makespan / cell["length"]
    reduced["host_loop_s"] = statistics.fmean(hosts)
    reduced["host_loop_cv"] = (
        statistics.pstdev(hosts) / reduced["host_loop_s"] if reduced["host_loop_s"] else 0.0
    )
    if cell["mode"].startswith("graph"):
        reduced["host_per_replay_s"] = reduced["host_loop_s"] / cell["graph_replays"]
        reduced["host_per_kernel_s"] = reduced["host_per_replay_s"] / cell["length"]
    else:
        reduced["host_per_replay_s"] = reduced["host_loop_s"]
        reduced["host_per_kernel_s"] = reduced["host_loop_s"] / cell["length"]
    peak = PEAK[scored_state]
    reduced["peak_flops"] = peak
    t_flop = cell["flops"] / peak
    t_mem = cell["bytes"] / HBM_ROOF
    reduced["roofline_floor_s"] = max(t_flop, t_mem)
    return reduced


def ols(xs: list[float], ys: list[float]) -> tuple[float, float, float]:
    """Return slope, intercept and R-squared of an ordinary least squares fit."""

    n = len(xs)
    mean_x = statistics.fmean(xs)
    mean_y = statistics.fmean(ys)
    sxx = sum((x - mean_x) ** 2 for x in xs)
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True))
    slope = sxy / sxx if sxx else 0.0
    intercept = mean_y - slope * mean_x
    ss_tot = sum((y - mean_y) ** 2 for y in ys)
    ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(xs, ys, strict=True))
    r2 = 1.0 - ss_res / ss_tot if ss_tot else 1.0
    del n
    return slope, intercept, r2


class Recorder:
    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    def score(
        self,
        expectation_id: str,
        passed: bool | None,
        evaluated: str,
        detail: Any = None,
    ) -> None:
        row = next(r for r in FREEZE["scored_expectations"] if r["id"] == expectation_id)
        status = "unevaluated" if passed is None else ("pass" if passed else "fail")
        self.rows.append(
            {
                "id": expectation_id,
                "claim": row["claim"],
                "risk": row["risk"],
                "group": row["group"],
                "status": status,
                "passed": bool(passed) if passed is not None else None,
                "evaluated": evaluated,
                "detail": detail,
            }
        )


class Guards:
    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    def check(self, guard_id: str, held: bool, evaluated: str, detail: Any = None) -> None:
        claim = next(r["claim"] for r in FREEZE["fatal_guards"] if r["id"] == guard_id)
        self.rows.append(
            {
                "id": guard_id,
                "claim": claim,
                "held": bool(held),
                "evaluated": evaluated,
                "detail": detail,
            }
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", required=True)
    parser.add_argument("--identity", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    raw = json.loads(Path(args.raw).read_text(encoding="utf-8"))
    identity_text = Path(args.identity).read_text(encoding="utf-8")
    cells = [reduce_cell(cell) for cell in raw["cells"]]

    index: dict[tuple[str, str, int], dict] = {
        (cell["mode"], cell["tag"], cell["length"]): cell for cell in cells
    }
    lengths = SUBSTRATE["chain_lengths"]
    tags = [kernel["tag"] for kernel in FREEZE["kernels"]]
    members = next(k for k in FREEZE["kernels"] if k["tag"] == "mix")["cycle"]

    def period(mode: str, tag: str, length: int) -> float | None:
        cell = index.get((mode, tag, length))
        if cell is None or cell["void_for_scoring"]:
            return None
        return cell["period_s"]

    scored = Recorder()
    guards = Guards()
    k_max = max(lengths)

    # ---- Quantity 1, the falsifier --------------------------------------
    nop_graph = period("graph", "nop", k_max)
    nop_eager = period("eager", "nop", k_max)
    f1 = {}
    for tag in ("g1", "g2", "g4"):
        graph = period("graph", tag, k_max)
        eager = period("eager", tag, k_max)
        if None in (graph, eager, nop_graph, nop_eager):
            continue
        s_graph = graph - nop_graph
        s_eager = eager - nop_eager
        if s_eager <= 0:
            continue
        f1[tag] = {
            "s_graph_s": s_graph,
            "s_eager_s": s_eager,
            "ratio": s_graph / s_eager,
            "p_graph_s": graph,
            "p_eager_s": eager,
        }
    scored.score(
        "F1",
        (None if not f1 else all(0.95 <= row["ratio"] <= 1.05 for row in f1.values())),
        "differenced service-time ratio S_graph over S_eager at K = "
        f"{k_max}: " + ", ".join(f"{k} {v['ratio']:.4f}" for k, v in sorted(f1.items())),
        f1,
    )

    g4_graph = period("graph", "g4", k_max)
    g4_eager = period("eager", "g4", k_max)
    f2_ratio = g4_graph / g4_eager if g4_graph and g4_eager else None
    scored.score(
        "F2",
        (None if f2_ratio is None else 0.97 <= f2_ratio <= 1.03),
        f"raw period ratio for g4 at K = {k_max}: "
        + (f"{f2_ratio:.4f}" if f2_ratio else "unmeasured"),
        {"ratio": f2_ratio, "p_graph_s": g4_graph, "p_eager_s": g4_eager},
    )

    f3_ratio = nop_eager / nop_graph if nop_eager and nop_graph else None
    scored.score(
        "F3",
        (None if f3_ratio is None else f3_ratio >= 1.5),
        f"null-kernel period ratio eager over graph at K = {k_max}: "
        + (f"{f3_ratio:.4f}" if f3_ratio else "unmeasured"),
        {"ratio": f3_ratio, "p_eager_s": nop_eager, "p_graph_s": nop_graph},
    )

    # ---- Quantity 2, host submission ------------------------------------
    fit_lengths = [k for k in lengths if k >= 8]
    eager_fits = {}
    for tag in tags:
        xs, ys = [], []
        for k in fit_lengths:
            cell = index.get(("eager", tag, k))
            if cell and not cell["void_for_scoring"]:
                xs.append(float(k))
                ys.append(cell["host_loop_s"])
        if len(xs) >= 3:
            slope, intercept, r2 = ols(xs, ys)
            eager_fits[tag] = {"slope_s": slope, "intercept_s": intercept, "r2": r2}
    scored.score(
        "H1",
        (None if not eager_fits else all(row["r2"] >= 0.99 for row in eager_fits.values())),
        "eager host loop time against K, R-squared per tag: "
        + ", ".join(f"{k} {v['r2']:.5f}" for k, v in sorted(eager_fits.items())),
        eager_fits,
    )

    eager_slope = eager_fits.get(LAUNCH_TAG, {}).get("slope_s")
    scored.score(
        "H2",
        (None if eager_slope is None else 0.8e-6 <= eager_slope <= 4.0e-6),
        f"fitted eager per-launch host slope on the {LAUNCH_TAG} chain: "
        + (f"{eager_slope * 1e6:.4f} microseconds" if eager_slope else "unmeasured"),
        {"slope_s": eager_slope, "tag": LAUNCH_TAG},
    )

    def host_per_replay(tag: str, length: int) -> float | None:
        cell = index.get(("graph", tag, length))
        if cell is None or cell["void_for_scoring"]:
            return None
        return cell["host_per_replay_s"]

    h3 = {}
    for tag in tags:
        big = host_per_replay(tag, k_max)
        small = host_per_replay(tag, 1)
        if big and small:
            h3[tag] = big / small
    scored.score(
        "H3",
        (None if not h3 else all(0.5 <= v <= 2.0 for v in h3.values())),
        f"graph host cost per replay, K = {k_max} over K = 1: "
        + ", ".join(f"{k} {v:.4f}" for k, v in sorted(h3.items())),
        h3,
    )

    graph_fits = {}
    for tag in tags:
        xs, ys = [], []
        for k in fit_lengths:
            value = host_per_replay(tag, k)
            if value is not None:
                xs.append(float(k))
                ys.append(value)
        if len(xs) >= 3:
            slope, intercept, r2 = ols(xs, ys)
            graph_fits[tag] = {"slope_s": slope, "intercept_s": intercept, "r2": r2}
    graph_slope = graph_fits.get(LAUNCH_TAG, {}).get("slope_s")
    h4_ratio = (
        abs(graph_slope) / eager_slope if graph_slope is not None and eager_slope else None
    )
    scored.score(
        "H4",
        (None if h4_ratio is None else h4_ratio <= 0.10),
        "graph per-node host slope over eager per-launch slope: "
        + (f"{h4_ratio:.5f}" if h4_ratio is not None else "unmeasured"),
        {"graph_slope_s": graph_slope, "eager_slope_s": eager_slope, "ratio": h4_ratio},
    )

    graph_per_kernel = None
    cell = index.get(("graph", LAUNCH_TAG, k_max))
    if cell and not cell["void_for_scoring"]:
        graph_per_kernel = cell["host_per_kernel_s"]
    eager_per_launch = None
    cell = index.get(("eager", LAUNCH_TAG, k_max))
    if cell and not cell["void_for_scoring"]:
        eager_per_launch = cell["host_per_kernel_s"]
    h5_ratio = (
        eager_per_launch / graph_per_kernel
        if graph_per_kernel and eager_per_launch
        else None
    )
    scored.score(
        "H5",
        (None if h5_ratio is None else h5_ratio >= 20.0),
        f"eager per-launch over graph per-enqueued-kernel host cost at K = {k_max}: "
        + (f"{h5_ratio:.2f}" if h5_ratio else "unmeasured"),
        {
            "eager_per_launch_s": eager_per_launch,
            "graph_per_kernel_s": graph_per_kernel,
            "ratio": h5_ratio,
        },
    )

    eager_slope_ps = round(eager_slope * 1e12) if eager_slope else None
    h6_delta = (
        abs(eager_slope_ps - TURING_EAGER_PS) / TURING_EAGER_PS if eager_slope_ps else None
    )
    scored.score(
        "H6",
        (
            None
            if eager_slope_ps is None
            else (h6_delta > 0.10 and eager_slope_ps < TURING_EAGER_PS)
        ),
        f"measured eager per-launch {eager_slope_ps} ps against the Turing "
        f"{TURING_EAGER_PS} ps, relative difference "
        + (f"{h6_delta * 100:.2f} percent" if h6_delta is not None else "unmeasured"),
        {"measured_ps": eager_slope_ps, "turing_ps": TURING_EAGER_PS, "delta": h6_delta},
    )

    # ---- Quantity 3, reserved device gap --------------------------------
    scored.score(
        "D1",
        (None if nop_graph is None else 0.3e-6 <= nop_graph <= 1.5e-6),
        "in-graph null-kernel period: "
        + (f"{nop_graph * 1e6:.4f} microseconds" if nop_graph else "unmeasured"),
        {"p_graph_nop_s": nop_graph},
    )
    nop_graph_16 = period("graph", "nop", 16)
    d2_ratio = nop_graph / nop_graph_16 if nop_graph and nop_graph_16 else None
    scored.score(
        "D2",
        (None if d2_ratio is None else 0.8 <= d2_ratio <= 1.25),
        f"in-graph null-kernel period at K = {k_max} over K = 16: "
        + (f"{d2_ratio:.4f}" if d2_ratio else "unmeasured"),
        {"ratio": d2_ratio, "k256_s": nop_graph, "k16_s": nop_graph_16},
    )
    scored.score(
        "D3",
        (None if nop_eager is None else 1.5e-6 <= nop_eager <= 3.0e-6),
        "eager null-kernel period: "
        + (f"{nop_eager * 1e6:.4f} microseconds" if nop_eager else "unmeasured"),
        {"p_eager_nop_s": nop_eager},
    )
    d4_ratio = nop_graph / eager_slope if nop_graph and eager_slope else None
    scored.score(
        "D4",
        (None if d4_ratio is None else d4_ratio <= 0.6),
        "in-graph null-kernel period over eager per-launch host slope: "
        + (f"{d4_ratio:.4f}" if d4_ratio else "unmeasured"),
        {"ratio": d4_ratio},
    )

    # ---- Mixed chains ----------------------------------------------------
    def mix_additivity(mode: str) -> dict:
        rows = {}
        for k in lengths:
            if k % 4:
                continue
            mixed = period(mode, "mix", k)
            if mixed is None:
                continue
            member_periods = [period(mode, tag, k) for tag in members]
            if any(value is None for value in member_periods):
                continue
            expected = sum(member_periods) / 4.0
            rows[k] = {
                "measured_period_s": mixed,
                "expected_period_s": expected,
                "ratio": mixed / expected,
            }
        return rows

    for expectation_id, mode, tolerance in (("M1", "graph", 0.05), ("M2", "eager", 0.08)):
        rows = mix_additivity(mode)
        at_max = rows.get(k_max)
        scored.score(
            expectation_id,
            (None if at_max is None else abs(at_max["ratio"] - 1.0) <= tolerance),
            f"{mode} mix cycle period over the mean of its four members at K = {k_max}: "
            + (f"{at_max['ratio']:.4f}" if at_max else "unmeasured"),
            rows,
        )

    # ---- Fatal guards ----------------------------------------------------
    identity_rows = [line for line in identity_text.splitlines() if line.strip()][1:]
    guards.check(
        "GG1",
        len(identity_rows) == 1
        and "A100-SXM4-80GB" in identity_rows[0]
        and "Disabled" in identity_rows[0]
        and "Enabled" in identity_rows[0]
        and raw["visible_device_count"] == 1
        and bool(raw["gpu_uuid"])
        and raw["gpu_uuid"] in identity_text,
        f"{len(identity_rows)} identity rows, UUID {raw['gpu_uuid']}",
        {"identity": identity_rows},
    )

    void_cells = [
        f"{cell['mode']}:{cell['tag']}:{cell['length']}"
        for cell in cells
        if cell["void_for_scoring"]
    ]
    guards.check(
        "GG2",
        not void_cells,
        f"{len(cells)} cells measured, {len(void_cells)} without {MIN_STATIONARY} "
        "stationary blocks in any single clock state",
        void_cells,
    )

    # GG3 as written: the node count a chain length implies is the per-kernel
    # node count observed at K = 1 for that tag, times K. cuBLAS may emit more
    # than one node per GEMM, so the per-kernel factor is read back, not assumed.
    node_breaches = []
    unit_nodes = {}
    for tag in tags:
        base = index.get(("graph", tag, 1))
        if base is None:
            continue
        unit_nodes[tag] = base["graph_nodes"]
    for cell in cells:
        if cell["mode"] != "graph":
            continue
        unit = unit_nodes.get(cell["tag"])
        if unit is None:
            continue
        if cell["tag"] == "mix":
            continue  # its unit varies with the cycle position
        if cell["graph_nodes"] != unit * cell["length"]:
            node_breaches.append(
                {
                    "tag": cell["tag"],
                    "length": cell["length"],
                    "nodes": cell["graph_nodes"],
                    "expected": unit * cell["length"],
                }
            )
    guards.check(
        "GG3",
        not node_breaches and bool(unit_nodes),
        f"per-kernel graph node counts read back at K = 1: {unit_nodes}; "
        f"{len(node_breaches)} chains did not scale linearly",
        node_breaches,
    )

    host_order = [
        f"{cell['mode']}:{cell['tag']}:{cell['length']}"
        for cell in cells
        if not cell["host_before_sync"]
    ]
    guards.check(
        "GG4",
        not host_order,
        f"{len(host_order)} cells closed their host interval after a synchronization",
        host_order,
    )

    floor_breaches = [
        {
            "cell": f"{cell['mode']}:{cell['tag']}:{cell['length']}",
            "makespan_s": cell["makespan_s"],
            "floor_s": cell["roofline_floor_s"],
        }
        for cell in cells
        if not cell["void_for_scoring"]
        and (cell["makespan_s"] < 0 or cell["makespan_s"] < cell["roofline_floor_s"])
    ]
    guards.check(
        "GG5",
        not floor_breaches,
        f"{len(floor_breaches)} chains ran below their own roofline floor or negative",
        floor_breaches,
    )

    residuals = [row for row in raw["correctness"] if row["residual"] > 1e-3]
    guards.check(
        "GG6",
        not residuals and bool(raw["correctness"]),
        f"{len(raw['correctness'])} GEMMs sampled, {len(residuals)} outside 1e-3 relative",
        raw["correctness"],
    )

    cv_breaches = [
        {
            "cell": f"{cell['mode']}:{cell['tag']}:{cell['length']}",
            "makespan_cv": cell["makespan_cv"],
        }
        for cell in cells
        if not cell["void_for_scoring"] and cell["makespan_cv"] > CV_CEILING
    ]
    guards.check(
        "GG7",
        not cv_breaches,
        f"{len(cv_breaches)} cells exceeded the {CV_CEILING * 100:.0f} percent "
        "block-mean CV ceiling",
        cv_breaches,
    )

    mode_state_breaches = []
    for tag in tags:
        for k in lengths:
            eager_cell = index.get(("eager", tag, k))
            graph_cell = index.get(("graph", tag, k))
            if eager_cell is None or graph_cell is None:
                continue
            if eager_cell["void_for_scoring"] or graph_cell["void_for_scoring"]:
                continue
            if eager_cell["scored_state"] != graph_cell["scored_state"]:
                mode_state_breaches.append(
                    {
                        "tag": tag,
                        "length": k,
                        "eager_state": eager_cell["scored_state"],
                        "graph_state": graph_cell["scored_state"],
                    }
                )
    guards.check(
        "GG8",
        not mode_state_breaches,
        f"{len(mode_state_breaches)} tag and chain-length pairs compared two clock states",
        mode_state_breaches,
    )

    # ---- Output profiles -------------------------------------------------
    eager_costs = [
        index[("eager", LAUNCH_TAG, k)]["host_per_kernel_s"]
        for k in fit_lengths
        if ("eager", LAUNCH_TAG, k) in index
        and not index[("eager", LAUNCH_TAG, k)]["void_for_scoring"]
    ]
    graph_costs = [
        index[("graph", LAUNCH_TAG, k)]["host_per_kernel_s"]
        for k in fit_lengths
        if ("graph", LAUNCH_TAG, k) in index
        and not index[("graph", LAUNCH_TAG, k)]["void_for_scoring"]
    ]
    graph_ref = index.get(("graph", LAUNCH_TAG, K_REF))
    profiles = {
        "a100-epyc-eager-host": {
            "launch_class": "eager-host-bound",
            "point_ps": eager_slope_ps,
            "empirical_min_ps": round(min(eager_costs) * 1e12) if eager_costs else None,
            "empirical_max_ps": round(max(eager_costs) * 1e12) if eager_costs else None,
            "definition": "fitted eager per-launch host slope over K in [8, 256]",
        },
        "a100-epyc-cuda-graph": {
            "launch_class": "cuda-graph-node",
            "point_ps": (
                round(graph_ref["host_per_kernel_s"] * 1e12)
                if graph_ref and not graph_ref["void_for_scoring"]
                else None
            ),
            "empirical_min_ps": round(min(graph_costs) * 1e12) if graph_costs else None,
            "empirical_max_ps": round(max(graph_costs) * 1e12) if graph_costs else None,
            "definition": (
                f"graph host submission cost per enqueued kernel at K_ref = {K_REF}"
            ),
            "fixed_per_replay_ps": (
                round(graph_ref["host_per_replay_s"] * 1e12)
                if graph_ref and not graph_ref["void_for_scoring"]
                else None
            ),
        },
    }

    reserved_gap = {
        "in_graph_null_kernel_period_ps": round(nop_graph * 1e12) if nop_graph else None,
        "eager_null_kernel_period_ps": round(nop_eager * 1e12) if nop_eager else None,
        "provenance": "examples/a100_graph_launch_v1, job recorded in RESULTS.md",
        "wired_to": "nothing",
    }

    every_guard_held = all(row["held"] for row in guards.rows)
    passed = sum(1 for row in scored.rows if row["passed"] is True)
    failed = sum(1 for row in scored.rows if row["passed"] is False)
    unevaluated = sum(1 for row in scored.rows if row["passed"] is None)
    results = {
        "schema": "simllm-study-results-v1",
        "study": "a100_graph_launch_v1",
        "stage": 2,
        "verdict": "interpretable" if every_guard_held else "void",
        "voiding_guards": [row["id"] for row in guards.rows if not row["held"]],
        "scored_passed": passed,
        "scored_failed": failed,
        "scored_unevaluated": unevaluated,
        "scored_total": len(scored.rows),
        "declared_denominator": FREEZE["scored_denominator"],
        "launch_tag": LAUNCH_TAG,
        "device": {
            "name": raw["device_name"],
            "uuid": raw["gpu_uuid"],
            "sm_count": raw["sm_count"],
            "driver_api_version": raw["driver_api_version"],
            "runtime_api_version": raw["runtime_api_version"],
        },
        "eager_host_fits": eager_fits,
        "graph_host_fits": graph_fits,
        "output_profiles": profiles,
        "reserved_device_gap": reserved_gap,
        "scored": scored.rows,
        "fatal_guards": guards.rows,
        "cells": sorted(cells, key=lambda c: (c["mode"], c["tag"], c["length"])),
    }
    Path(args.out).write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    print(
        f"verdict={results['verdict']} scored={passed} pass / {failed} fail / "
        f"{unevaluated} unevaluated of {len(scored.rows)}"
    )
    for row in scored.rows:
        if row["passed"] is not True:
            print(f"  {row['status'].upper()} {row['id']}: {row['evaluated']}")
    for row in guards.rows:
        if not row["held"]:
            print(f"  GUARD VIOLATED {row['id']}: {row['evaluated']}")


if __name__ == "__main__":
    main()
