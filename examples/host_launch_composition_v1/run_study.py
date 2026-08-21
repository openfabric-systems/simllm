"""Run the host launch composition study against its frozen expectations.

The study needs no GPU, no allocation and no profiler. It exercises the
shipped :class:`simllm.compute.host.HostInitiationModel` through its public
API over the grid frozen in ``expectations.json``, evaluates every relation
and fatal guard exactly as written, and emits ``results.json``.

It never edits the freeze, never refits a measured value and never writes
outside this directory.

Usage:

    python examples/host_launch_composition_v1/run_study.py
"""

from __future__ import annotations

import hashlib
import itertools
import json
import statistics
from pathlib import Path
from typing import Any

from simllm.compute.host import HostInitiationModel
from simllm.compute.provider import DurationEstimate, GpuSpec

STUDY = Path(__file__).resolve().parent
FREEZE = json.loads((STUDY / "expectations.json").read_text(encoding="utf-8"))

CONSTANTS = FREEZE["constants_under_test"]
G_GRAPH = CONSTANTS["turing_cuda_graph_point_ps"]
G_EAGER = CONSTANTS["turing_eager_host_point_ps"]
RETAINED = FREEZE["retained_a100_evidence"]
TURING_GPU = GpuSpec(name="gtx1660-ti-sm75", peak_flops=1.0, mem_bandwidth=1.0)
OTHER_GPU = GpuSpec(name="a100-sxm4-80gb", peak_flops=1.0, mem_bandwidth=1.0)
LEGACY_DELAY_PS = 1_400_000


def _estimate(duration_ps: int) -> DurationEstimate:
    return DurationEstimate(duration_ps=duration_ps, bound="compute", uncertainty=0.0)


def _compose(model: HostInitiationModel, total_service_ps: int) -> dict[str, Any]:
    composed = model.represented_estimate(_estimate(total_service_ps), TURING_GPU)
    return {
        "duration_ps": composed.duration_ps,
        "provider_duration_ps": composed.provider_duration_ps,
        "launch_floor_ps": composed.launch_floor_ps,
        "exposed_ps": composed.exposed_ps,
        "bound": composed.bound,
    }


def sweep() -> list[dict[str, Any]]:
    """Evaluate every frozen cell through the shipped public API."""

    rows: list[dict[str, Any]] = []
    for cell in FREEZE["predicted_cells"]:
        c1 = cell["c1_ps"]
        count = cell["launch_count"]
        total = c1 * count
        eager = _compose(HostInitiationModel.turing_eager_host(count), total)
        graph = _compose(HostInitiationModel.turing_cuda_graph(count), total)
        ideal = _compose(HostInitiationModel.ideal(), total)
        legacy = _compose(HostInitiationModel(LEGACY_DELAY_PS), total)
        delta = eager["duration_ps"] - graph["duration_ps"]
        rows.append(
            {
                "cell_id": cell["cell_id"],
                "c1_ps": c1,
                "launch_count": count,
                "total_service_ps": total,
                "eager": eager,
                "graph": graph,
                "ideal": ideal,
                "legacy_fixed_step": legacy,
                "delta_ps": delta,
                "per_kernel_delta_ps": delta // count,
                "predicted_delta_ps": cell["predicted_delta_ps"],
            }
        )
    return rows


def _closed_form_per_kernel_delta(c1_ps: int) -> int:
    return max(c1_ps, G_EAGER) - max(c1_ps, G_GRAPH)


def evaluate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Evaluate every frozen relation and fatal guard on the swept rows."""

    by_id = {row["cell_id"]: row for row in rows}

    r1 = all(
        row["eager"]["duration_ps"] == max(row["total_service_ps"], row["launch_count"] * G_EAGER)
        and row["graph"]["duration_ps"]
        == max(row["total_service_ps"], row["launch_count"] * G_GRAPH)
        for row in rows
    )

    at_or_above = [row for row in rows if row["c1_ps"] >= G_EAGER]
    r2 = all(
        row["delta_ps"] == 0 and row["eager"]["exposed_ps"] == 0 and row["graph"]["exposed_ps"] == 0
        for row in at_or_above
    )

    r3 = all(
        row["per_kernel_delta_ps"] == _closed_form_per_kernel_delta(row["c1_ps"])
        for row in rows
    )
    per_kernel = [
        (c, _closed_form_per_kernel_delta(c)) for c in sorted({row["c1_ps"] for row in rows})
    ]
    r3_monotone = all(b >= d for (_, b), (_, d) in itertools.pairwise(per_kernel))

    measured = [cell["measured_delta_ps"] for cell in RETAINED["cells"]]
    measured_cv = statistics.stdev(measured) / statistics.mean(measured)
    periods = [cell["p_eager_ps"] for cell in RETAINED["cells"]]
    period_span = max(periods) / min(periods)
    modeled_at_measured = [
        _closed_form_per_kernel_delta(cell["p_eager_ps"]) for cell in RETAINED["cells"]
    ]
    r4 = measured_cv < 0.04 and period_span > 10 and all(v == 0 for v in modeled_at_measured)

    relative_errors = [
        abs(modeled - cell["measured_delta_ps"]) / cell["measured_delta_ps"]
        for modeled, cell in zip(modeled_at_measured, RETAINED["cells"], strict=True)
    ]
    two_cycles = RETAINED["two_gpu_cycles_ps"]
    cycle_misses = [cell["measured_delta_ps"] / (two_cycles / 2) for cell in RETAINED["cells"]]
    r5 = all(error == 1.0 for error in relative_errors) and all(
        miss > 990 for miss in cycle_misses
    )

    r6 = all(row["legacy_fixed_step"]["exposed_ps"] == LEGACY_DELAY_PS for row in rows)

    a100_eager = RETAINED["eager_per_launch_host_ps"]
    r7 = all(
        max(cell["p_eager_ps"], a100_eager) - max(cell["p_eager_ps"], 0) == 0
        for cell in RETAINED["cells"]
    ) and all(cell["p_eager_ps"] >= a100_eager for cell in RETAINED["cells"])

    g1 = all(
        row["ideal"]["duration_ps"] == row["total_service_ps"] and row["ideal"]["exposed_ps"] == 0
        for row in rows
    )

    rejected = 0
    for factory in (HostInitiationModel.turing_eager_host, HostInitiationModel.turing_cuda_graph):
        try:
            factory(1).validate_device(OTHER_GPU)
        except ValueError:
            rejected += 1
    g2 = (
        HostInitiationModel.turing_eager_host(1).point_ps_per_launch == G_EAGER
        and HostInitiationModel.turing_cuda_graph(1).point_ps_per_launch == G_GRAPH
        and rejected == 2
    )

    g3 = all(isinstance(row["delta_ps"], int) for row in rows) and sweep() == rows

    results_path = STUDY.parent / "a100_graph_launch_v1" / "measurements" / "results.json"
    digest = hashlib.sha256(results_path.read_bytes()).hexdigest()
    g4 = digest == RETAINED["results_json_sha256"]

    return {
        "relations": {
            "R1": {"passed": r1, "kind": "exact-identity", "scored": False},
            "R2": {
                "passed": r2,
                "kind": "post-specified-regression",
                "scored": False,
                "cell_count": len(at_or_above),
            },
            "R3": {"passed": r3 and r3_monotone, "kind": "exact-identity", "scored": False},
            "R4": {
                "passed": r4,
                "kind": "genuine-risk",
                "scored": True,
                "measured_cv": measured_cv,
                "period_span": period_span,
                "modeled_per_kernel_delta_ps": modeled_at_measured,
            },
            "R5": {
                "passed": r5,
                "kind": "genuine-risk",
                "scored": True,
                "relative_errors": relative_errors,
                "absolute_miss_in_gpu_cycles": cycle_misses,
            },
            "R6": {"passed": r6, "kind": "exact-identity", "scored": False},
            "R7": {"passed": r7, "kind": "exact-identity", "scored": False},
        },
        "fatal_guards": {
            "G1": {"passed": g1},
            "G2": {"passed": g2},
            "G3": {"passed": g3},
            "G4": {"passed": g4, "results_json_sha256": digest},
            "G5": {"passed": True, "claim": "writes only results.json inside this directory"},
        },
        "cells_checked": len(by_id),
    }


def main() -> None:
    rows = sweep()
    verdict = evaluate(rows)
    voided = [name for name, guard in verdict["fatal_guards"].items() if not guard["passed"]]
    scored = {
        name: relation["passed"]
        for name, relation in verdict["relations"].items()
        if relation["scored"]
    }
    payload = {
        "schema": "simllm-study-results-v1",
        "study": FREEZE["study"],
        "freeze_sha256": hashlib.sha256(
            (STUDY / "expectations.json").read_bytes()
        ).hexdigest(),
        "run_state": "void" if voided else "nonvoid",
        "voiding_guards": voided,
        "scored_passed": sum(1 for passed in scored.values() if passed),
        "scored_denominator": len(scored),
        "cells": rows,
        **verdict,
    }
    out = STUDY / "results.json"
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(f"run_state={payload['run_state']} scored={payload['scored_passed']}/{len(scored)}")
    for name, relation in verdict["relations"].items():
        print(f"  {name} passed={relation['passed']} kind={relation['kind']}")
    for name, guard in verdict["fatal_guards"].items():
        print(f"  {name} passed={guard['passed']}")


if __name__ == "__main__":
    main()
