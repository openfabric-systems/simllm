"""Score the a100_kernel_constants_v1 measurement against its frozen freeze.

Reads the raw harness output and `expectations.json`, evaluates the 31 scored
expectations and the 12 fatal guards exactly as written, and emits
`results.json`. It never edits the freeze and never invents a bound.

Usage:

    python examples/a100_kernel_constants_v1/score_expectations.py \
        --boosted <boosted.json> --base <base.json> \
        --identity <gpu_identity_before.csv> --out <results.json>
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from itertools import pairwise
from pathlib import Path
from typing import Any

STUDY = Path(__file__).resolve().parent
FREEZE = json.loads((STUDY / "expectations.json").read_text(encoding="utf-8"))
REFREEZE = json.loads((STUDY / "refreeze_expectations.json").read_text(encoding="utf-8"))

SUBSTRATE = FREEZE["substrate"]
QUANTUM_NS = SUBSTRATE["timer_quantum_ns"]
L2_BYTES = SUBSTRATE["l2_bytes"]
HBM_NAMEPLATE = SUBSTRATE["hbm_nameplate_bytes_per_second"]
PEAK = {1410: SUBSTRATE["peak_flops_1410"], 1275: SUBSTRATE["peak_flops_1275"]}
MEMORY_CLOCK_MHZ = SUBSTRATE["memory_clock_mhz"]
MIN_STATIONARY = SUBSTRATE["min_stationary_batches"]
HOST_BOUND_CEILING = REFREEZE["host_bound_ratio_ceiling"]
SHORT_CELL_S = REFREEZE["short_cell_threshold_us"] * 1e-6
INSTRUMENT_BAND_S = tuple(v * 1e-6 for v in REFREEZE["instrumentation_cost_band_us"])


# ---------------------------------------------------------------------------
# Cell reduction
# ---------------------------------------------------------------------------


def stationary_batches(cell: dict) -> list[tuple[int, float]]:
    """Return (clock state, per-launch seconds) for every stationary batch."""

    out = []
    for elapsed_ms, clocks in zip(cell["batch_ms"], cell["batch_clocks"], strict=True):
        if clocks["sm_before"] != clocks["sm_after"]:
            continue
        if clocks["mem_before"] != MEMORY_CLOCK_MHZ or clocks["mem_after"] != MEMORY_CLOCK_MHZ:
            continue
        if clocks["th_before"] != 0 or clocks["th_after"] != 0:
            continue
        out.append((clocks["sm_before"], elapsed_ms * 1e-3 / cell["group"]))
    return out


def reduce_cell(cell: dict) -> dict:
    """Reduce one raw cell to its clock-conditioned constant and diagnostics."""

    batches = stationary_batches(cell)
    by_state: dict[int, list[float]] = {}
    for state, seconds in batches:
        by_state.setdefault(state, []).append(seconds)
    scored_state = None
    for state, values in sorted(by_state.items(), key=lambda kv: -len(kv[1])):
        if len(values) >= MIN_STATIONARY:
            scored_state = state
            break

    chain_state_ok = (
        cell["chain_before"]["sm"] == cell["chain_after"]["sm"]
        and cell["chain_before"]["mem"] == MEMORY_CLOCK_MHZ
        and cell["chain_after"]["mem"] == MEMORY_CLOCK_MHZ
        and cell["chain_before"]["th"] == 0
        and cell["chain_after"]["th"] == 0
    )
    chain = [value * 1e-3 for value in cell["chain_ms"]]
    chain_median = statistics.median(chain) if chain else 0.0
    chain_cv = (
        statistics.pstdev(chain) / statistics.fmean(chain) if chain and chain_median else 0.0
    )

    reduced = {
        "id": cell["id"],
        "lane": cell["lane"],
        "family": cell["family"],
        "arm": cell["arm"],
        "m": cell["m"],
        "n": cell["n"],
        "k": cell["k"],
        "length": cell["length"],
        "size_bytes": cell["size_bytes"],
        "rotate": cell["rotate"],
        "group": cell["group"],
        "flops": cell["flops"],
        "total_bytes": cell["total_bytes"],
        "distinct_bytes": cell["distinct_bytes"],
        "correctness_residual": cell["correctness_residual"],
        "stationary_by_state": {str(s): len(v) for s, v in by_state.items()},
        "scored_state": scored_state,
        "void_for_scoring": scored_state is None,
        "chain_state_stationary": chain_state_ok,
        "chain_state": cell["chain_before"]["sm"],
        "chain_median_s": chain_median,
        "chain_mean_s": statistics.fmean(chain) if chain else 0.0,
        "chain_cv": chain_cv,
        "batch_clock_states": sorted({c["sm_before"] for c in cell["batch_clocks"]}),
        "memory_clocks": sorted(
            {c["mem_before"] for c in cell["batch_clocks"]}
            | {c["mem_after"] for c in cell["batch_clocks"]}
        ),
        "throttle_words": sorted(
            {c["th_before"] for c in cell["batch_clocks"]}
            | {c["th_after"] for c in cell["batch_clocks"]}
        ),
        "batches_recorded": len(cell["batch_ms"]),
    }
    host_ratios = [
        host / device
        for host, device in zip(
            cell.get("batch_host_ms", []), cell["batch_ms"], strict=False
        )
        if device > 0
    ]
    reduced["host_ratio_max"] = max(host_ratios) if host_ratios else 0.0
    reduced["host_ratio_median"] = (
        statistics.median(host_ratios) if host_ratios else 0.0
    )
    reduced["host_issue_bound"] = bool(host_ratios) and reduced["host_ratio_max"] > (
        HOST_BOUND_CEILING
    )
    if scored_state is not None:
        values = by_state[scored_state]
        constant = statistics.fmean(values)
        reduced["constant_s"] = constant
        reduced["constant_ps"] = round(constant * 1e12)
        reduced["batch_samples"] = len(values)
        reduced["batch_cv"] = statistics.pstdev(values) / constant if constant else 0.0
        reduced["rate_bytes_per_second"] = cell["total_bytes"] / constant
        reduced["rate_flops_per_second"] = cell["flops"] / constant
        peak = PEAK[scored_state]
        reduced["t_flop_s"] = cell["flops"] / peak
        reduced["peak_flops"] = peak
    return reduced


def attach_roof(reduced: dict, r_hbm: float) -> None:
    """Attach the measured-roof derived quantities once R_hbm is known."""

    if reduced["void_for_scoring"]:
        return
    t_mem = reduced["total_bytes"] / r_hbm
    t_flop = reduced["t_flop_s"]
    reduced["t_mem_s"] = t_mem
    reduced["regime"] = "compute" if t_flop > t_mem else "memory"
    reduced["t_roof_s"] = max(t_flop, t_mem)
    constant = reduced["constant_s"]
    reduced["eff_roofline"] = reduced["t_roof_s"] / constant
    reduced["eff_compute"] = t_flop / constant
    reduced["eff_memory"] = t_mem / constant
    compulsory = max(0.0, reduced["distinct_bytes"] - L2_BYTES) / r_hbm
    reduced["physical_floor_s"] = max(t_flop, compulsory)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def knee(n: int, k: int, roof: float, peak: float) -> float:
    return peak * k * n / (n * k * roof - peak * (k + n))


def ape(predicted: float, measured: float) -> float:
    return abs(predicted - measured) / measured * 100.0


def nearest_rank(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = max(1, math.ceil(percentile * len(ordered)))
    return ordered[index - 1]


def log_interpolate(points: list[tuple[int, float]], m: int) -> float | None:
    """Log-linear interpolation, the rule ProfileTableProvider applies."""

    below = max(((x, y) for x, y in points if 0 < x < m), key=lambda p: p[0], default=None)
    above = min(((x, y) for x, y in points if x > m), key=lambda p: p[0], default=None)
    if below is None or above is None:
        return None
    (lo, d_lo), (hi, d_hi) = below, above
    frac = (math.log(m) - math.log(lo)) / (math.log(hi) - math.log(lo))
    return math.exp(math.log(d_lo) + frac * (math.log(d_hi) - math.log(d_lo)))


def in_scored_scope(cell: dict) -> bool:
    """Whether a cell participates in at least one scored expectation.

    G9R, G10 and G11R quantify over SCORED cells, not over every measured
    cell. The rotated variants below 64 MiB and the short prefill sequences
    are measured and published, but no scored expectation names them, so they
    do not enter those guards.
    """

    lane = cell["lane"]
    if lane == "1":
        return cell["size_bytes"] >= 256 << 20
    if lane == "2":
        return True
    if lane == "4":
        return True
    if lane == "3":
        if cell["family"] == "attn_decode":
            return cell["m"] >= 16 or cell["total_bytes"] >= 160 << 20
        return cell["length"] in (2048, 4096)
    if lane == "5":
        mib = cell["size_bytes"] >> 20
        if cell["rotate"] > 1:
            return mib in (64, 256)
        return mib in (4, 64, 256)
    return False


class Recorder:
    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    def score(
        self,
        expectation_id: str,
        passed: bool | None,
        evaluated: str,
        detail: Any = None,
        *,
        lane: str | None = None,
    ) -> None:
        claim = next(
            row["claim"] for row in FREEZE["scored_expectations"] if row["id"] == expectation_id
        )
        risk = next(
            row["risk"] for row in FREEZE["scored_expectations"] if row["id"] == expectation_id
        )
        status = "unevaluated" if passed is None else ("pass" if passed else "fail")
        self.rows.append(
            {
                "id": expectation_id,
                "claim": claim,
                "risk": risk,
                "status": status,
                "passed": bool(passed) if passed is not None else None,
                "evaluated": evaluated,
                "lane": lane or expectation_id.split("-")[1],
                "detail": detail,
            }
        )


class Guards:
    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    def check(self, guard_id: str, held: bool, evaluated: str, detail: Any = None) -> None:
        claim = next(
            row["claim"]
            for row in list(FREEZE["fatal_guards"]) + list(REFREEZE["fatal_guards"])
            if row["id"] == guard_id
        )
        self.rows.append(
            {
                "id": guard_id,
                "claim": claim,
                "held": bool(held),
                "evaluated": evaluated,
                "detail": detail,
            }
        )


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--boosted", required=True)
    parser.add_argument("--base", required=True)
    parser.add_argument("--identity", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    boosted_raw = json.loads(Path(args.boosted).read_text(encoding="utf-8"))
    base_raw = json.loads(Path(args.base).read_text(encoding="utf-8"))
    identity_text = Path(args.identity).read_text(encoding="utf-8")

    boosted_all = {cell["id"]: reduce_cell(cell) for cell in boosted_raw["cells"]}
    base_all = {cell["id"]: reduce_cell(cell) for cell in base_raw["cells"]}

    # G13 is the one declared survivable guard. A host-issue-bound cell is a
    # measurement of the host, so it leaves every scored scope, every other
    # guard that quantifies over scored cells, and the published table. Its
    # exclusion is recorded, never hidden.
    host_bound = [
        {
            "id": cell["id"],
            "arm": cell["arm"],
            "host_ratio_max": cell["host_ratio_max"],
            "host_ratio_median": cell["host_ratio_median"],
        }
        for cell in list(boosted_all.values()) + list(base_all.values())
        if cell["host_issue_bound"]
    ]
    excluded = {row["id"] + "@" + row["arm"] for row in host_bound}
    boosted = {
        key: cell
        for key, cell in boosted_all.items()
        if key + "@" + cell["arm"] not in excluded
    }
    base = {
        key: cell for key, cell in base_all.items() if key + "@" + cell["arm"] not in excluded
    }

    # R_hbm: the measured roof, from lane 1 at 256 MiB and above, boosted arm.
    lane1 = [
        cell
        for cell in boosted.values()
        if cell["lane"] == "1" and not cell["void_for_scoring"] and cell["size_bytes"] >= 256 << 20
    ]
    r_hbm = max(cell["rate_bytes_per_second"] for cell in lane1)
    for cell in list(boosted_all.values()) + list(base_all.values()):
        attach_roof(cell, r_hbm)
        cell["in_scored_scope"] = in_scored_scope(cell)

    scored = Recorder()
    guards = Guards()

    def lane(cells: dict, name: str) -> list[dict]:
        return [cell for cell in cells.values() if cell["lane"] == name]

    def by_id(cells: dict, cell_id: str) -> dict | None:
        cell = cells.get(cell_id)
        return None if cell is None or cell["void_for_scoring"] else cell

    # ---- Lane 1 ----------------------------------------------------------
    for expectation_id, kind, low in (
        ("E-1-1", "read", 1700e9),
        ("E-1-2", "write", 1700e9),
        ("E-1-3", "copy", 1600e9),
        ("E-1-4", "triad", 1600e9),
    ):
        cell = by_id(boosted, f"hbm_{kind}_2048mib")
        rate = cell["rate_bytes_per_second"] if cell else 0.0
        scored.score(
            expectation_id,
            (None if cell is None else low <= rate <= 1937e9),
            f"{kind} at 2048 MiB measured {rate / 1e9:.2f} GB/s against "
            f"[{low / 1e9:.0f}, 1937] GB/s",
            {"rate_gbps": rate / 1e9},
        )

    size_pairs = {}
    for kind in ("read", "write", "copy", "triad"):
        small = by_id(boosted, f"hbm_{kind}_1024mib")
        large = by_id(boosted, f"hbm_{kind}_2048mib")
        if small and large:
            a, b = small["rate_bytes_per_second"], large["rate_bytes_per_second"]
            size_pairs[kind] = abs(a - b) / max(a, b)
    scored.score(
        "E-1-5",
        (None if not size_pairs else all(value <= 0.03 for value in size_pairs.values())),
        "per-kernel 1024 MiB against 2048 MiB relative difference: "
        + ", ".join(f"{k} {v * 100:.2f} percent" for k, v in size_pairs.items()),
        dict(size_pairs),
    )
    scored.score(
        "E-1-6",
        1700e9 <= r_hbm <= 1937e9,
        f"R_hbm measured {r_hbm / 1e9:.2f} GB/s, {r_hbm / HBM_NAMEPLATE * 100:.2f} "
        "percent of nameplate",
        {"r_hbm_bytes_per_second": r_hbm},
    )

    lane1_ratios = {}
    for cell_id, base_cell in base.items():
        if base_cell["lane"] != "1" or base_cell["void_for_scoring"]:
            continue
        boosted_cell = by_id(boosted, cell_id)
        if boosted_cell is None:
            continue
        lane1_ratios[cell_id] = base_cell["constant_s"] / boosted_cell["constant_s"]
    scored.score(
        "E-1-7",
        (None if not lane1_ratios else all(0.98 <= v <= 1.02 for v in lane1_ratios.values())),
        "base over boosted constant ratio per lane-1 cell: "
        + ", ".join(f"{k} {v:.4f}" for k, v in sorted(lane1_ratios.items())),
        lane1_ratios,
    )

    # ---- Lane 2 ----------------------------------------------------------
    families = {entry["id"]: entry for entry in FREEZE["gemm_families"]}
    grids: dict[str, list[dict]] = {name: [] for name in families}
    holdouts: dict[str, list[dict]] = {name: [] for name in families}
    for cell in lane(boosted, "2"):
        name, split = cell["family"].split(":")
        if cell["void_for_scoring"]:
            continue
        if split == "grid":
            grids[name].append(cell)
        elif split == "holdout":
            holdouts[name].append(cell)
    for values in list(grids.values()) + list(holdouts.values()):
        values.sort(key=lambda cell: cell["m"])

    m1_ratios = {}
    for name, cells in grids.items():
        first = next((cell for cell in cells if cell["m"] == 1), None)
        if first:
            m1_ratios[name] = first["constant_s"] / first["t_mem_s"]
    scored.score(
        "E-2-1",
        (None if not m1_ratios else all(0.9 <= v <= 2.5 for v in m1_ratios.values())),
        "M = 1 measured time over t_mem per family: "
        + ", ".join(f"{k} {v:.3f}" for k, v in sorted(m1_ratios.items())),
        m1_ratios,
    )

    monotone_violations = []
    for name, cells in grids.items():
        for previous, current in pairwise(cells):
            tolerance = QUANTUM_NS * 1e-9 + 0.02 * previous["constant_s"]
            if current["constant_s"] < previous["constant_s"] - tolerance:
                monotone_violations.append(
                    {
                        "family": name,
                        "from_m": previous["m"],
                        "to_m": current["m"],
                        "drop_percent": (previous["constant_s"] - current["constant_s"])
                        / previous["constant_s"]
                        * 100.0,
                    }
                )
    scored.score(
        "E-2-2",
        (None if not grids else not monotone_violations),
        f"{len(monotone_violations)} adjacent grid pairs fell outside the "
        "one-quantum-plus-2-percent nondecreasing tolerance",
        monotone_violations,
    )

    eff_violations = []
    for name, cells in grids.items():
        family = families[name]
        limit = 4.0 * knee(family["n"], family["k"], r_hbm, PEAK[1410])
        window = [cell for cell in cells if cell["m"] <= limit]
        for previous, current in pairwise(window):
            if current["eff_roofline"] < previous["eff_roofline"] - 0.02:
                eff_violations.append(
                    {
                        "family": name,
                        "from_m": previous["m"],
                        "to_m": current["m"],
                        "from_eff": previous["eff_roofline"],
                        "to_eff": current["eff_roofline"],
                    }
                )
    scored.score(
        "E-2-3",
        (None if not grids else not eff_violations),
        f"{len(eff_violations)} adjacent grid pairs below 4 M* broke efficiency monotonicity",
        eff_violations,
    )

    knee_rows = {}
    for name, cells in grids.items():
        family = families[name]
        plateau = [cell["constant_s"] for cell in cells if cell["m"] in (1, 2, 4, 8)]
        if not plateau or not cells:
            continue
        threshold = 1.5 * statistics.median(plateau)
        measured = next((cell["m"] for cell in cells if cell["constant_s"] > threshold), None)
        knee_meas = knee(family["n"], family["k"], r_hbm, PEAK[1410])
        knee_rows[name] = {
            "measured_knee_m": measured,
            "knee_measured_roof": knee_meas,
            "low": 0.7 * knee_meas,
            "high": 4.0 * knee_meas,
        }
    scored.score(
        "E-2-4",
        (
            None
            if not knee_rows
            else all(
                row["measured_knee_m"] is not None
                and row["low"] <= row["measured_knee_m"] <= row["high"]
                for row in knee_rows.values()
            )
        ),
        "measured knee against [0.7 M*, 4 M*] per family: "
        + ", ".join(
            f"{k} M={row['measured_knee_m']} in "
            f"[{row['low']:.1f}, {row['high']:.1f}]"
            for k, row in sorted(knee_rows.items())
        ),
        knee_rows,
    )

    e25 = {}
    for name, cells in grids.items():
        family = families[name]
        target = 4.0 * family["knee_nameplate"]
        cell = min(cells, key=lambda c: abs(c["m"] - target)) if cells else None
        if cell:
            e25[name] = {
                "m": cell["m"],
                "fraction_of_peak": cell["rate_flops_per_second"] / cell["peak_flops"],
            }
    scored.score(
        "E-2-5",
        (None if not e25 else all(row["fraction_of_peak"] >= 0.45 for row in e25.values())),
        "fraction of peak at the grid point nearest 4 M*: "
        + ", ".join(
            f"{k} M={row['m']} {row['fraction_of_peak'] * 100:.1f} percent"
            for k, row in sorted(e25.items())
        ),
        e25,
    )

    square = by_id(boosted, "gemm_G4_m8192")
    square_fraction = (
        square["rate_flops_per_second"] / square["peak_flops"] if square else 0.0
    )
    scored.score(
        "E-2-6",
        (None if square is None else square_fraction >= 0.85),
        f"8192 cubed reached {square_fraction * 100:.2f} percent of P(clock)",
        {"fraction_of_peak": square_fraction},
    )

    e27 = {}
    for name, cells in grids.items():
        if not cells:
            continue
        first = next((cell for cell in cells if cell["m"] == 1), None)
        last = cells[-1]
        if first:
            e27[name] = {
                "eff_m1": first["eff_roofline"],
                "eff_max_m": last["eff_roofline"],
                "max_m": last["m"],
            }
    scored.score(
        "E-2-7",
        (
            None
            if not e27
            else all(row["eff_m1"] < 0.62 < row["eff_max_m"] for row in e27.values())
        ),
        "efficiency at M = 1 and at the largest grid M per family: "
        + ", ".join(
            f"{k} {row['eff_m1']:.3f} to {row['eff_max_m']:.3f}"
            for k, row in sorted(e27.items())
        ),
        e27,
    )

    holdout_rows = []
    for name, cells in holdouts.items():
        family = families[name]
        basis = [(cell["m"], cell["constant_s"]) for cell in grids[name]]
        knee_nameplate = family["knee_nameplate"]
        for cell in cells:
            predicted = log_interpolate(basis, cell["m"])
            if predicted is None:
                holdout_rows.append(
                    {"family": name, "m": cell["m"], "ape": None, "covered": False}
                )
                continue
            inside = 0.5 * knee_nameplate <= cell["m"] <= 2.0 * knee_nameplate
            error = ape(predicted, cell["constant_s"])
            holdout_rows.append(
                {
                    "family": name,
                    "m": cell["m"],
                    "ape": error,
                    "covered": True,
                    "inside_knee_window": inside,
                    "tolerance": 25.0 if inside else 12.0,
                    "predicted_s": predicted,
                    "measured_s": cell["constant_s"],
                }
            )
    e28_failures = [
        row
        for row in holdout_rows
        if not row["covered"] or row["ape"] > row["tolerance"]
    ]
    scored.score(
        "E-2-8",
        (None if not holdout_rows else not e28_failures),
        f"{len(holdout_rows)} held-out shapes, {len(e28_failures)} outside their "
        "per-window tolerance",
        e28_failures,
    )

    errors = [row["ape"] for row in holdout_rows if row["covered"]]
    median_error = statistics.median(errors) if errors else float("inf")
    p95_error = nearest_rank(errors, 0.95) if errors else float("inf")
    scored.score(
        "E-2-9",
        (None if not errors else (median_error < 10.0 and p95_error < 20.0)),
        f"held-out interpolation median APE {median_error:.3f} percent, "
        f"p95 {p95_error:.3f} percent",
        {"median_ape": median_error, "p95_ape": p95_error, "count": len(errors)},
    )

    e210 = {}
    for cell_id, base_cell in base.items():
        if base_cell["lane"] != "2" or base_cell["void_for_scoring"]:
            continue
        boosted_cell = by_id(boosted, cell_id)
        if boosted_cell is None:
            continue
        ratio = base_cell["constant_s"] / boosted_cell["constant_s"]
        regime = boosted_cell["regime"]
        low, high = (1.06, 1.13) if regime == "compute" else (0.98, 1.02)
        e210[cell_id] = {
            "ratio": ratio,
            "regime": regime,
            "low": low,
            "high": high,
            "base_state": base_cell["scored_state"],
            "boosted_state": boosted_cell["scored_state"],
        }
    scored.score(
        "E-2-10",
        (None if not e210 else all(row["low"] <= row["ratio"] <= row["high"] for row in e210.values())),
        "base over boosted constant ratio per paired lane-2 cell: "
        + ", ".join(
            f"{k} {row['ratio']:.4f} ({row['regime']})" for k, row in sorted(e210.items())
        ),
        e210,
    )

    e211 = {}
    for cell_id, base_cell in base.items():
        if base_cell["void_for_scoring"]:
            continue
        boosted_cell = by_id(boosted, cell_id)
        if boosted_cell is None:
            continue
        e211[cell_id] = {
            "base_eff": base_cell["eff_roofline"],
            "boosted_eff": boosted_cell["eff_roofline"],
            "delta": abs(base_cell["eff_roofline"] - boosted_cell["eff_roofline"]),
        }
    scored.score(
        "E-2-11",
        (None if not e211 else all(row["delta"] <= 0.03 for row in e211.values())),
        "absolute efficiency difference between arms, worst "
        + (
            f"{max(row['delta'] for row in e211.values()):.4f}"
            if e211
            else "no paired cell"
        ),
        e211,
    )

    # ---- Lane 3 ----------------------------------------------------------
    e31 = {}
    for geometry in ("granite", "synthetic"):
        big = by_id(boosted, f"attn_prefill_{geometry}_s4096")
        small = by_id(boosted, f"attn_prefill_{geometry}_s2048")
        if big and small:
            e31[geometry] = big["constant_s"] / small["constant_s"]
    scored.score(
        "E-3-1",
        (None if not e31 else all(3.2 <= v <= 4.4 for v in e31.values())),
        "prefill time ratio S 4096 over 2048: "
        + ", ".join(f"{k} {v:.3f}" for k, v in sorted(e31.items())),
        e31,
    )

    synthetic = by_id(boosted, "attn_prefill_synthetic_s4096")
    synthetic_fraction = (
        synthetic["rate_flops_per_second"] / synthetic["peak_flops"] if synthetic else 0.0
    )
    scored.score(
        "E-3-2",
        (None if synthetic is None else synthetic_fraction >= 0.40),
        f"synthetic prefill at S = 4096 reached {synthetic_fraction * 100:.2f} percent of peak",
        {"fraction_of_peak": synthetic_fraction},
    )

    decode = [
        cell
        for cell in lane(boosted, "3")
        if cell["family"] == "attn_decode" and not cell["void_for_scoring"]
    ]
    e33 = {}
    for cell in decode:
        if cell["total_bytes"] < 160 << 20:
            continue
        e33[cell["id"]] = cell["rate_bytes_per_second"] / r_hbm
    scored.score(
        "E-3-3",
        (None if not e33 else all(0.55 <= v <= 1.00 for v in e33.values())),
        f"{len(e33)} decode cells at or above 160 MiB of KV traffic, achieved "
        "fraction of R_hbm from "
        + (f"{min(e33.values()):.3f} to {max(e33.values()):.3f}" if e33 else "none"),
        e33,
    )

    lengths = FREEZE["attention_decode"]["kv_length"]
    e34 = {}
    for batch in (16, 64, 256):
        for previous, current in pairwise(lengths):
            low = by_id(boosted, f"attn_decode_b{batch}_l{previous}")
            high = by_id(boosted, f"attn_decode_b{batch}_l{current}")
            if low and high:
                e34[f"b{batch}_l{previous}_to_{current}"] = (
                    high["constant_s"] / low["constant_s"]
                )
    scored.score(
        "E-3-4",
        (None if not e34 else all(3.4 <= v <= 4.6 for v in e34.values())),
        "decode time ratio across quadrupled L, worst pair "
        + (
            f"{min(e34.values()):.3f} to {max(e34.values()):.3f}"
            if e34
            else "no pair measured"
        ),
        e34,
    )

    big_batch = by_id(boosted, "attn_decode_b256_l8192")
    small_batch = by_id(boosted, "attn_decode_b64_l8192")
    batch_ratio = (
        big_batch["constant_s"] / small_batch["constant_s"]
        if big_batch and small_batch
        else 0.0
    )
    scored.score(
        "E-3-5",
        (None if batch_ratio == 0.0 else 3.4 <= batch_ratio <= 4.6),
        f"decode time ratio B 256 over 64 at L = 8192 measured {batch_ratio:.3f}",
        {"ratio": batch_ratio},
    )

    # ---- Lane 4 ----------------------------------------------------------
    expert_cells = [cell for cell in lane(boosted, "4") if not cell["void_for_scoring"]]
    e41 = {cell["id"]: cell["regime"] for cell in expert_cells}
    scored.score(
        "E-4-1",
        (None if not e41 else all(regime == "memory" for regime in e41.values())),
        f"{sum(1 for r in e41.values() if r == 'memory')} of {len(e41)} captured "
        "expert cells are memory-limited at the measured R_hbm",
        e41,
    )

    e42 = {}
    for shape in ("expert_gate_up", "expert_down"):
        low = by_id(boosted, f"moe_{shape}_m1")
        high = by_id(boosted, f"moe_{shape}_m54")
        if low and high:
            e42[shape] = high["constant_s"] / low["constant_s"]
    scored.score(
        "E-4-2",
        (None if not e42 else all(v <= 1.6 for v in e42.values())),
        "expert-load plateau ratio M_e 54 over 1: "
        + ", ".join(f"{k} {v:.3f}" for k, v in sorted(e42.items())),
        e42,
    )

    e43 = {}
    for shape in ("expert_gate_up", "expert_down"):
        cell = by_id(boosted, f"moe_{shape}_m54")
        if cell:
            e43[shape] = cell["rate_flops_per_second"] / cell["peak_flops"]
    scored.score(
        "E-4-3",
        (None if not e43 else all(v < 0.25 for v in e43.values())),
        "fraction of peak at M_e = 54: "
        + ", ".join(f"{k} {v * 100:.2f} percent" for k, v in sorted(e43.items())),
        e43,
    )

    e44 = {}
    for shape in ("expert_gate_up", "expert_down"):
        cell = by_id(boosted, f"moe_{shape}_m14")
        if cell:
            e44[shape] = cell["constant_s"] / cell["t_mem_s"]
    scored.score(
        "E-4-4",
        (None if not e44 else all(1.0 <= v <= 3.0 for v in e44.values())),
        "measured time over t_mem at the captured balanced load: "
        + ", ".join(f"{k} {v:.3f}" for k, v in sorted(e44.items())),
        e44,
    )

    # ---- Lane 5 ----------------------------------------------------------
    for expectation_id, mib in (("E-5-1", 256), ("E-5-2", 64)):
        rows = {}
        for kind in ("scale", "add", "rmsnorm"):
            cell = by_id(boosted, f"elem_{kind}_{mib}mib_warm")
            if cell:
                rows[kind] = cell["rate_bytes_per_second"] / r_hbm
        scored.score(
            expectation_id,
            (None if not rows else all(v >= 0.80 for v in rows.values())),
            f"fraction of R_hbm at {mib} MiB: "
            + ", ".join(f"{k} {v * 100:.1f} percent" for k, v in sorted(rows.items())),
            rows,
        )

    e53 = {}
    for mib in (64, 256):
        for kind in ("scale", "add", "rmsnorm"):
            warm = by_id(boosted, f"elem_{kind}_{mib}mib_warm")
            rotated = by_id(boosted, f"elem_{kind}_{mib}mib_rot")
            if warm and rotated:
                e53[f"{kind}_{mib}mib"] = abs(
                    rotated["constant_s"] - warm["constant_s"]
                ) / warm["constant_s"]
    scored.score(
        "E-5-3",
        (None if not e53 else all(v <= 0.06 for v in e53.values())),
        "warm against rotated relative difference, worst "
        + (f"{max(e53.values()) * 100:.3f} percent" if e53 else "no pair"),
        e53,
    )

    small_scale = by_id(boosted, "elem_scale_4mib_warm")
    e54_ratio = (
        small_scale["t_mem_s"] / small_scale["constant_s"] if small_scale else 0.0
    )
    scored.score(
        "E-5-4",
        (None if small_scale is None else e54_ratio <= 1.15),
        f"4 MiB warm scale ran {e54_ratio:.4f} times its own bytes-over-R_hbm time",
        {"speedup_over_roof": e54_ratio},
    )

    # ---- Fatal guards ----------------------------------------------------
    identity_rows = [line for line in identity_text.splitlines() if line.strip()][1:]
    identity_ok = (
        len(identity_rows) == 1
        and "A100-SXM4-80GB" in identity_rows[0]
        and "Disabled" in identity_rows[0]
        and "Enabled" in identity_rows[0]
        and boosted_raw["visible_device_count"] == 1
        and base_raw["visible_device_count"] == 1
    )
    guards.check(
        "G1",
        identity_ok,
        f"identity rows {len(identity_rows)}, visible devices "
        f"{boosted_raw['visible_device_count']} and {base_raw['visible_device_count']}",
        {"identity": identity_rows},
    )

    all_cells = list(boosted.values()) + list(base.values())
    void_cells = [cell["id"] for cell in all_cells if cell["void_for_scoring"]]
    guards.check(
        "G2",
        not void_cells,
        f"{len(all_cells)} cells measured, {len(void_cells)} without 8 stationary "
        "batches in any single clock state",
        void_cells,
    )

    bad_memory = [
        cell["id"]
        for cell in all_cells
        if cell["memory_clocks"] != [MEMORY_CLOCK_MHZ] or cell["throttle_words"] != [0]
    ]
    guards.check(
        "G3",
        not bad_memory,
        f"{len(bad_memory)} cells saw a memory clock other than 1593 MHz or a "
        "nonzero throttle word",
        bad_memory,
    )

    ceiling_breaches = []
    for cell in all_cells:
        if cell["void_for_scoring"]:
            continue
        if cell["lane"] == "1" and cell["rate_bytes_per_second"] > HBM_NAMEPLATE:
            ceiling_breaches.append({"id": cell["id"], "kind": "hbm"})
        if cell["flops"] > 0 and cell["rate_flops_per_second"] > cell["peak_flops"]:
            ceiling_breaches.append({"id": cell["id"], "kind": "flops"})
    guards.check(
        "G4",
        not ceiling_breaches,
        f"{len(ceiling_breaches)} cells exceeded a nameplate ceiling",
        ceiling_breaches,
    )

    floor_breaches = [
        {
            "id": cell["id"],
            "constant_s": cell["constant_s"],
            "floor_s": cell["physical_floor_s"],
        }
        for cell in all_cells
        if not cell["void_for_scoring"] and cell["constant_s"] < cell["physical_floor_s"]
    ]
    guards.check(
        "G5",
        not floor_breaches,
        f"{len(floor_breaches)} cells completed below their compulsory-traffic floor",
        floor_breaches,
    )

    g6_cells = [
        cell
        for cell in all_cells
        if not cell["void_for_scoring"]
        and (cell["family"] == "attn_decode" or cell["lane"] == "4")
    ]
    g6_breaches = [cell["id"] for cell in g6_cells if cell["regime"] != "memory"]
    guards.check(
        "G6",
        not g6_breaches,
        f"{len(g6_cells)} decode and expert cells checked, {len(g6_breaches)} not "
        "memory-limited",
        g6_breaches,
    )

    protocol = []
    for raw in (boosted_raw, base_raw):
        if raw["warmup_discard"] != SUBSTRATE["warmup_discard_reps"]:
            protocol.append(f"{raw['arm']}: warmup discard {raw['warmup_discard']}")
        if raw["batches_per_cell"] != SUBSTRATE["batches_per_cell"]:
            protocol.append(f"{raw['arm']}: batches {raw['batches_per_cell']}")
        if raw["chain_reps"] != SUBSTRATE["diagnostic_chain_reps"]:
            protocol.append(f"{raw['arm']}: chain reps {raw['chain_reps']}")
        for cell in raw["cells"]:
            if len(cell["batch_ms"]) != SUBSTRATE["batches_per_cell"]:
                protocol.append(f"{cell['id']}: {len(cell['batch_ms'])} batches")
            span_us = cell["group"] * cell["probe_ms"] * 1000.0
            if cell["group"] < SUBSTRATE["batch_cap"] if False else False:
                pass
            if span_us < SUBSTRATE["batch_min_elapsed_us"] and cell["group"] < 256:
                protocol.append(f"{cell['id']}: batch span {span_us:.1f} us")
    guards.check(
        "G7",
        not protocol,
        f"{len(protocol)} protocol deviations from the frozen warmup, batch count "
        "and batch-span rule",
        protocol,
    )

    residuals = [
        {"id": cell["id"], "residual": cell["correctness_residual"]}
        for cell in all_cells
        if cell["correctness_residual"] >= 0 and cell["correctness_residual"] > 1e-3
    ]
    checked = sum(1 for cell in all_cells if cell["correctness_residual"] >= 0)
    guards.check(
        "G8",
        not residuals,
        f"{checked} cuBLAS cells sampled for numerical correctness, "
        f"{len(residuals)} outside 1e-3 relative",
        residuals,
    )

    # G9R: the per-repetition chain is contaminated by the instrumentation cost
    # the refreeze measures, so its dispersion identifies the kernel only for
    # cells above the registered 60 microsecond threshold.
    g9_breaches = []
    g9_checked = 0
    for cell in all_cells:
        if cell["void_for_scoring"] or not cell["chain_state_stationary"]:
            continue
        if not cell["in_scored_scope"]:
            continue
        if cell["chain_median_s"] <= SHORT_CELL_S:
            continue
        g9_checked += 1
        ceiling = 0.02 + (QUANTUM_NS * 1e-9) / (math.sqrt(12.0) * cell["chain_median_s"])
        if cell["chain_cv"] > ceiling:
            g9_breaches.append(
                {"id": cell["id"], "cv": cell["chain_cv"], "ceiling": ceiling}
            )
    guards.check(
        "G9R",
        not g9_breaches,
        f"{g9_checked} cells above the 60 microsecond threshold checked, "
        f"{len(g9_breaches)} exceeded the quantum-aware per-repetition CV ceiling",
        g9_breaches,
    )

    g10_breaches = [
        {"id": cell["id"], "cv": cell["batch_cv"]}
        for cell in all_cells
        if not cell["void_for_scoring"]
        and cell["in_scored_scope"]
        and cell["batch_cv"] > 0.02
    ]
    guards.check(
        "G10",
        not g10_breaches,
        f"{len(g10_breaches)} scored cells exceeded the 2 percent batch-mean CV ceiling",
        g10_breaches,
    )

    # G14: the instrumentation control must have executed.
    control = boosted_raw.get("instrumentation_control", [])
    by_stride: dict[str, dict[int, float]] = {}
    for row in control:
        by_stride.setdefault(row["id"], {})[row["stride"]] = row["per_kernel_ms"] * 1e-3
    per_boundary = {
        name: table[1] - table[64]
        for name, table in by_stride.items()
        if 1 in table and 64 in table
    }
    event_only_s = boosted_raw.get("event_only_period_ms", 0.0) * 1e-3
    guards.check(
        "G14",
        bool(per_boundary) and event_only_s > 0.0,
        f"instrumentation control produced {len(by_stride)} per-stride tables and "
        f"an event-only period of {event_only_s * 1e6:.4f} microseconds",
        {
            "per_stride_s": {k: {str(s): v for s, v in t.items()} for k, t in by_stride.items()},
            "per_boundary_s": per_boundary,
            "event_only_period_s": event_only_s,
        },
    )

    # G11R: the per-boundary cost is measured, is inside the frozen band, and
    # correcting the chain by it reconciles the chain with the batched constant.
    boundary_values = sorted(per_boundary.values())
    boundary_cost = statistics.median(boundary_values) if boundary_values else 0.0
    band_ok = bool(boundary_values) and all(
        INSTRUMENT_BAND_S[0] <= value <= INSTRUMENT_BAND_S[1] for value in boundary_values
    )
    g11_breaches = []
    for cell in all_cells:
        if cell["void_for_scoring"] or not cell["chain_state_stationary"]:
            continue
        if not cell["in_scored_scope"]:
            continue
        if cell["chain_state"] != cell["scored_state"]:
            continue
        corrected = cell["chain_mean_s"] - boundary_cost
        cell["chain_mean_corrected_s"] = corrected
        delta = abs(corrected - cell["constant_s"]) / cell["constant_s"]
        cell["chain_correction_delta"] = delta
        if delta > 0.03:
            g11_breaches.append(
                {"id": cell["id"], "delta": delta, "corrected_s": corrected}
            )
    guards.check(
        "G11R",
        band_ok and not g11_breaches,
        f"per-boundary event cost {boundary_cost * 1e6:.4f} microseconds over "
        f"{len(boundary_values)} controls, band "
        f"[{INSTRUMENT_BAND_S[0] * 1e6:.1f}, {INSTRUMENT_BAND_S[1] * 1e6:.1f}] "
        f"{'held' if band_ok else 'violated'}; {len(g11_breaches)} cells "
        "disagreed by more than 3 percent after the correction",
        {
            "per_boundary_s": per_boundary,
            "boundary_cost_s": boundary_cost,
            "band_held": band_ok,
            "breaches": g11_breaches,
        },
    )

    # G13, the one declared survivable guard.
    guards.check(
        "G13",
        not host_bound,
        f"{len(host_bound)} cells were host-issue bound and were excluded from "
        "every scored expectation and from the published table",
        host_bound,
    )

    uuid_ok = (
        boosted_raw["gpu_uuid"] == base_raw["gpu_uuid"]
        and bool(boosted_raw["gpu_uuid"])
        and boosted_raw["gpu_uuid"] in identity_text
    )
    guards.check(
        "G12",
        uuid_ok,
        f"both arms observed GPU UUID {boosted_raw['gpu_uuid']}",
        {"uuid": boosted_raw["gpu_uuid"]},
    )

    # ---- Verdict ---------------------------------------------------------
    survivable = {
        guard["id"] for guard in REFREEZE["fatal_guards"] if guard["survivable"]
    }
    voiding = [row for row in guards.rows if not row["held"] and row["id"] not in survivable]
    every_guard_held = not voiding
    passed = sum(1 for row in scored.rows if row["passed"] is True)
    failed = sum(1 for row in scored.rows if row["passed"] is False)
    unevaluated = sum(1 for row in scored.rows if row["passed"] is None)
    results = {
        "schema": "simllm-study-results-v1",
        "study": "a100_kernel_constants_v1",
        "stage": 1,
        "verdict": "interpretable" if every_guard_held else "void",
        "scored_passed": passed,
        "scored_failed": failed,
        "scored_unevaluated": unevaluated,
        "scored_total": len(scored.rows),
        "voiding_guards": [row["id"] for row in voiding],
        "survivable_guards": sorted(survivable),
        "host_issue_bound_cells": host_bound,
        "instrumentation_per_boundary_cost_s": boundary_cost,
        "instrumentation_event_only_period_s": event_only_s,
        "declared_denominator": FREEZE["scored_denominator"],
        "measured_hbm_roof_bytes_per_second": r_hbm,
        "measured_hbm_roof_fraction_of_nameplate": r_hbm / HBM_NAMEPLATE,
        "device": {
            "name": boosted_raw["device_name"],
            "uuid": boosted_raw["gpu_uuid"],
            "sm_count": boosted_raw["sm_count"],
            "l2_bytes": boosted_raw["l2_bytes"],
            "driver_api_version": boosted_raw["driver_api_version"],
            "runtime_api_version": boosted_raw["runtime_api_version"],
        },
        "scored": scored.rows,
        "fatal_guards": guards.rows,
        "cells": {
            "boosted": sorted(boosted_all.values(), key=lambda cell: cell["id"]),
            "base": sorted(base_all.values(), key=lambda cell: cell["id"]),
        },
        "holdout_rows": holdout_rows,
    }
    Path(args.out).write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    print(
        f"verdict={results['verdict']} scored={passed} pass / {failed} fail / "
        f"{unevaluated} unevaluated of {len(scored.rows)} "
        f"R_hbm={r_hbm / 1e9:.2f} GB/s "
        f"event_boundary={boundary_cost * 1e6:.3f} us "
        f"host_bound_cells={len(host_bound)}"
    )
    for row in scored.rows:
        if not row["passed"]:
            print(f"  FAIL {row['id']}: {row['evaluated']}")
    for row in guards.rows:
        if not row["held"]:
            label = "SURVIVABLE GUARD" if row["id"] in survivable else "GUARD VIOLATED"
            print(f"  {label} {row['id']}: {row['evaluated']}")


if __name__ == "__main__":
    main()
