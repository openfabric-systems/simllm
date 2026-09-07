"""Price and audit the frozen decode HBM crossover study."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
from collections import Counter
from contextlib import ExitStack
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from unittest.mock import patch

from simllm.compute import (
    GPU_ENVELOPES,
    KernelSpec,
    ModelDims,
    RooflineProvider,
    step_kernel,
)
from simllm.core import RequestPhase, ScheduledRequest, StepRecord
from simllm.deploy import (
    BudgetSpec,
    DeploymentCandidate,
    EnvelopeSpec,
    EstimatorInputs,
    FabricSpec,
    ModelRef,
    ModelWork,
    PoolSpec,
    SlaSpec,
    WorkloadPoint,
    estimate_decode_step,
)

ROOT = Path(__file__).resolve().parents[2]
STUDY = Path(__file__).resolve().parent
PS = 10**12
EXPECTATIONS_COMMIT = "5d353fb89695c7fb72e4af39e269495a9cd2fa43"
FROZEN_DIGESTS = {
    "expectations.md": "1522b28ba525d9895b2a9479848a53fa8af37b8bf7ce395d640a98af222a2ec6",
    "expectations.json": "867571b71f585df51620d79914e9af6482858fac811a81e873768574db20a1df",
    "expected_cells.csv": "663ba8f4dcb16405fb6d7253c7b9c68deef9349eb255eec4a0b541c4a3deae84",
}
IDENTITY = ("model", "context", "device", "hbm_scale", "batch")
INTEGERS = {
    "context", "batch", "weight_bytes", "kv_bytes", "flops", "memory_ps",
    "compute_ps", "tpot_ps", "rational_floor_ps", "serial_ceiling_ps",
    "weight_floor_ps", "tp", "ep", "hbm_bytes_per_second", "peak_flops",
}


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = []
        for raw in csv.DictReader(handle):
            row = {k: int(v) if k in INTEGERS else v for k, v in raw.items()}
            for key in ("tpot_ms", "throughput_decimal"):
                if key in row:
                    row[key] = float(row[key])
            if "deployment_guards_ok" in row:
                row["deployment_guards_ok"] = row["deployment_guards_ok"] == "True"
                if row["deployment_step_ps"] != "unavailable":
                    row["deployment_step_ps"] = int(row["deployment_step_ps"])
            rows.append(row)
        return rows


def identity(row: dict) -> tuple:
    return tuple(row[k] for k in IDENTITY)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest_matches(path: Path, expected: str) -> bool:
    """Accept the frozen digest over the raw bytes or over LF-normalized bytes.

    Windows checkouts may materialize CRLF line endings for text inputs; the
    frozen digests were taken over the committed bytes, so a text input is
    also accepted when its CRLF-normalized content matches. An input whose
    committed bytes legitimately contain CR sequences still matches raw. The
    committed-blob check in `chronology_findings` reads git's own bytes and
    needs neither form.
    """
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() == expected:
        return True
    return hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest() == expected


def frozen_findings(frozen: dict) -> list[str]:
    paths = {str(STUDY.relative_to(ROOT) / name): value
             for name, value in FROZEN_DIGESTS.items()}
    paths.update(frozen["pinned_inputs"])
    return [f"input digest: {name}" for name, value in paths.items()
            if not (ROOT / name).is_file() or not digest_matches(ROOT / name, value)]


def chronology_findings() -> list[str]:
    def git(*args):
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, check=False)

    findings = []
    if git("merge-base", "--is-ancestor", EXPECTATIONS_COMMIT, "HEAD").returncode:
        findings.append("expectations commit is not an ancestor")
    names = git("diff-tree", "--no-commit-id", "--name-only", "-r",
                EXPECTATIONS_COMMIT).stdout.decode().splitlines()
    expected = {str(STUDY.relative_to(ROOT) / name) for name in FROZEN_DIGESTS}
    if set(names) != expected:
        findings.append("freeze commit is not expectations-only")
    for name, expected_digest in FROZEN_DIGESTS.items():
        path = str(STUDY.relative_to(ROOT) / name)
        blob = git("show", f"{EXPECTATIONS_COMMIT}:{path}")
        if blob.returncode or hashlib.sha256(blob.stdout).hexdigest() != expected_digest:
            findings.append(f"committed frozen bytes: {name}")
    return findings


def inventory_work(frozen: dict, model: dict) -> tuple[int, int, int]:
    if model["id"] == frozen["external"]["id"]:
        record = json.loads((ROOT / model["source"]).read_text())
        work = record["model_work_derivation"]
        return (work["static_parameter_bytes"] // 4,
                work["decode_flops_per_batch_item_per_rank_tp4"],
                work["decode_logical_kv_bytes_per_batch_item"] // 4)
    path = next(p for p in frozen["pinned_inputs"] if "deployment-projections/" in p)
    inventory = json.loads((ROOT / path).read_text())
    unit = next(u for u in inventory["units"] if u["id"] == "sglang-decode-ep72-dp-attention")
    static = next(r for r in unit["static_rank_classes"] if r["class_id"] == "rank-class-0")
    case = next(c for c in unit["case_projections"]
                if c["case_id"] == "sglang-decode-ep72-b32-c2000")
    rank = next(r for r in case["rank_classes"] if r["class_id"] == "rank-class-0")
    weights = static["base"]["logical_total_hbm_bytes_per_rank"]
    flops, remainder = divmod(rank["base_total_flops_per_rank"], 32)
    cache, byte_remainder = divmod(rank["base_logical_hbm_bytes_per_rank"] - weights, 32)
    if remainder or byte_remainder:
        raise ValueError("EP72 batch-32 anchor does not divide exactly")
    return weights, flops, cache


def kernel_for(frozen: dict, model: dict, context: int, batch: int) -> tuple:
    if model["kind"] == "dense":
        dims = ModelDims(**model["dims"])
        record = StepRecord(0, 0, [ScheduledRequest(
            f"r-{index}", RequestPhase.DECODE, 1, context_length=context,
        ) for index in range(batch)])
        kernel = step_kernel(dims, record, num_sampled=batch)
        weights = dims.weight_bytes + dims.lm_head_bytes
        return kernel, weights
    weights, flops, cache = inventory_work(frozen, model)
    return KernelSpec(model["id"], batch * flops, weights + batch * cache), weights


def deployment_estimate(model: dict, context: int, batch: int, kernel: KernelSpec,
                        weights: int, gpu):
    source = "examples/decode_hbm_crossover_v1/expectations.json"
    sha = FROZEN_DIGESTS["expectations.json"]
    tp = model["tp"]
    candidate = DeploymentCandidate(
        candidate_id=f"{model['id']}-{gpu.name}-c{context}",
        model=ModelRef("declared", model["id"], sha),
        pools=(PoolSpec("decode", 1, tp, tp, 1, 1, 1, gpu.name),),
        fabric=FabricSpec(400_000_000_000, 450_000_000_000),
        workload=WorkloadPoint(None, context, 1, context),
        sla=SlaSpec(None, None), budget=BudgetSpec(tp, 1),
    )
    inputs = EstimatorInputs(
        model_work=ModelWork(kernel.name, int(kernel.flops) // batch, weights,
                             (int(kernel.bytes_moved) - weights) // batch, 0, sha, source),
        envelopes={gpu.name: EnvelopeSpec(gpu.name, gpu.peak_flops,
                    gpu.mem_bandwidth, 1, "simllm/compute/transformer.py; " + source)},
    )
    return estimate_decode_step(candidate, batch, inputs)


def price_cell(frozen: dict, model: dict, expected: dict) -> dict:
    batch, context = expected["batch"], expected["context"]
    base = GPU_ENVELOPES[expected["device"]]
    gpu = replace(base, mem_bandwidth=base.mem_bandwidth * float(expected["hbm_scale"]),
                  peak_flops=base.peak_flops * model.get("peak_scale", 1))
    kernel, weights = kernel_for(frozen, model, context, batch)
    floor = max(Fraction(int(kernel.bytes_moved) * PS, int(gpu.mem_bandwidth)),
                Fraction(int(kernel.flops) * PS, int(gpu.peak_flops)))
    ceiling = (Fraction(int(kernel.bytes_moved) * PS, int(gpu.mem_bandwidth))
               + Fraction(int(kernel.flops) * PS, int(gpu.peak_flops)))
    estimate = RooflineProvider(efficiency=1.0).estimate(kernel, gpu)
    deployment = (deployment_estimate(model, context, batch, kernel, weights, gpu)
                  if model["tp"] in (4, 8) else None)
    tpot = estimate.duration_ps
    speed = Fraction(PS, tpot)
    throughput = Fraction(batch, model["tp"]) * speed
    cache = int(kernel.bytes_moved) - weights
    flops_per_token = int(kernel.flops) // batch
    kv_per_request = cache // batch
    denominator = flops_per_token * int(gpu.mem_bandwidth) - kv_per_request * int(gpu.peak_flops)
    crossover = (Fraction(weights * int(gpu.peak_flops), denominator)
                 if denominator > 0 else None)
    serial_ceiling = -(-ceiling.numerator // ceiling.denominator)
    return {
        **{k: expected[k] for k in ("scope", *IDENTITY)},
        "weight_bytes": weights, "kv_bytes": cache, "flops": int(kernel.flops),
        "memory_ps": int(kernel.bytes_moved / gpu.mem_bandwidth * PS),
        "compute_ps": int(kernel.flops / gpu.peak_flops * PS),
        "tpot_ps": tpot, "rational_floor_ps": int(floor),
        "serial_ceiling_ps": serial_ceiling, "bound": estimate.bound,
        "weight_floor_ps": weights * PS // int(gpu.mem_bandwidth),
        "tpot_ms": tpot / 10**9, "tp": model["tp"], "ep": model["ep"],
        "request_tokens_per_second": str(speed),
        "throughput_tokens_per_second_per_gpu": str(throughput),
        "throughput_decimal": float(throughput),
        "hbm_bytes_per_second": int(gpu.mem_bandwidth), "peak_flops": int(gpu.peak_flops),
        "cache_to_weight_ratio": str(Fraction(cache, weights)),
        "weight_cache_equal_batch": str(Fraction(weights, kv_per_request)),
        "crossover_batch": str(crossover) if crossover is not None else "none",
        "relative_to_crossover": ("no-finite-crossover" if crossover is None else
                                  "below" if batch < crossover else "at-or-above"),
        "evidence_class": "ROOFLINE", "capacity_status": "unchecked",
        "deployment_step_ps": deployment.step_ps if deployment else "unavailable",
        "deployment_guards_ok": deployment is None or (
            deployment.step_ps == tpot and deployment.analytical_step_ps == tpot
            and deployment.fabric_floor.duration_ps == deployment.intra_floor.duration_ps == 0
            and deployment.fabric_excess_ps == deployment.intra_excess_ps == 0
            and deployment.batch_service is None and deployment.handoff is None
            and all(term.estimate.evidence.value in {"ROOFLINE", "DECLARED"}
                    for term in deployment.stamp.terms)),
    }


def audit(rows: list[dict], frozen: dict, expected: list[dict],
          findings: list[str] | None = None) -> tuple[dict, list[dict]]:
    findings = list(findings or [])
    predictions = {identity(row): row for row in expected}
    actual = {identity(row): row for row in rows}
    if len(actual) != len(rows) or set(actual) != set(predictions):
        findings.append("cell identities are missing, duplicated or unexpected")
    models = {m["id"]: m for m in [*frozen["models"], frozen["external"]]}
    mismatches = []
    max_residual = 0
    rational_differences = []
    for key, row in actual.items():
        oracle = predictions.get(key)
        if oracle is None:
            continue
        residual = row["tpot_ps"] - oracle["tpot_ps"]
        max_residual = max(max_residual, abs(residual))
        mismatch = [k for k, v in oracle.items() if row[k] != v]
        if mismatch:
            mismatches.append({"cell": list(key), "fields": mismatch, "residual_ps": residual})
        if any(row[k] != oracle[k] for k in ("weight_bytes", "kv_bytes", "flops")):
            findings.append(f"emitted work differs: {key}")
        t = row["tpot_ps"]
        delta = t - row["rational_floor_ps"]
        if delta:
            rational_differences.append({"cell": list(key), "delta_ps": delta})
        if (abs(delta) > 1 or t + 1 < row["weight_floor_ps"]
                or t > row["serial_ceiling_ps"]):
            findings.append(f"physical bounds: {key}")
        if (row["tp"] != models[row["model"]]["tp"]
                or row["ep"] != models[row["model"]]["ep"]
                or Fraction(row["throughput_tokens_per_second_per_gpu"])
                != Fraction(row["batch"] * PS, row["tp"] * t)
                or Fraction(row["request_tokens_per_second"]) != Fraction(PS, t)):
            findings.append(f"output-token normalization: {key}")
        if row["evidence_class"] != "ROOFLINE" or not row["deployment_guards_ok"]:
            findings.append(f"pricing provenance or disabled terms: {key}")
    relations = []

    def emit(family, keys, passed, **evidence):
        relations.append({"family": family, "cells": [list(k) for k in keys],
                          "passed": bool(passed), **evidence})

    main = {key: row for key, row in actual.items() if row["scope"] == "sweep"}
    bandwidth_identities = 0
    for key, row in main.items():
        m, c, dev, scale, b = key
        next_scale = {"0.5": "1", "1": "2"}.get(scale)
        fast_key = (m, c, dev, next_scale, b)
        if fast_key in main:
            fast = main[fast_key]
            slow_t, fast_t = row["tpot_ps"], fast["tpot_ps"]
            if row["bound"] == fast["bound"] == "compute":
                bandwidth_identities += 1
                if slow_t != fast_t:
                    findings.append(f"inactive bandwidth identity: {key}")
            else:
                memory_pair = row["bound"] == fast["bound"] == "memory"
                emit("B1-bandwidth", [key, fast_key], fast_t <= slow_t <= 2 * fast_t + 2
                     and (not memory_pair or abs(slow_t - 2 * fast_t) <= 2),
                     memory_pair=memory_pair, ratio=slow_t / fast_t)
        double_key = (m, c, dev, scale, 2 * b)
        if double_key in main:
            second = main[double_key]
            t1, t2 = row["tpot_ps"], second["tpot_ps"]
            y1 = Fraction(row["throughput_tokens_per_second_per_gpu"])
            y2 = Fraction(second["throughput_tokens_per_second_per_gpu"])
            emit("B4-frontier", [key, double_key], t2 >= t1
                 and y2 / y1 >= 1 - Fraction(2, min(t1, t2)))
            if row["bound"] == second["bound"] == "memory":
                increment = Fraction(row["kv_bytes"] * PS, row["hbm_bytes_per_second"])
                error = t2 - t1 - increment
                emit("B2-memory-affine", [key, double_key], abs(error) <= 2,
                     increment_residual_ps=str(error))
            if row["bound"] == second["bound"] == "compute":
                emit("B3-compute-linear", [key, double_key], abs(t2 - 2 * t1) <= 2
                     and abs(y2 / y1 - 1) <= Fraction(2, t2),
                     double_residual_ps=t2 - 2 * t1,
                     throughput_rounding_decrease=y2 < y1)
        context_key = (m, 2048, dev, scale, b)
        if c == 1 and context_key in main:
            emit("B6-context", [key, context_key], main[context_key]["tpot_ps"] > row["tpot_ps"])
    for knee in frozen["crossovers"]:
        if knee["model"] == frozen["external"]["id"]:
            continue
        prefix = (knee["model"], knee["context"], knee["device"], knee["hbm_scale"])
        cross = knee["crossover_batch"]
        if cross is None:
            keys = [(*prefix, b) for b in frozen["batches"]]
            passed = all(k in actual and actual[k]["bound"] == "memory" for k in keys)
        else:
            value = Fraction(cross)
            lo, hi = int(value), -(-value.numerator // value.denominator)
            keys = list(dict.fromkeys([(*prefix, lo), (*prefix, hi)]))
            passed = all(k in actual and actual[k]["bound"] ==
                         ("memory" if k[-1] < value else "compute") for k in keys)
        matching = [row for key, row in actual.items() if key[:4] == prefix]
        passed = passed and all(row["crossover_batch"] == (cross or "none") for row in matching)
        emit("B5-crossover", keys, passed, crossover_batch=cross)
    nonvoid = not findings
    scores = {family: {"instances": sum(r["family"] == family for r in relations),
                       "passed": sum(r["family"] == family and r["passed"] for r in relations)}
              for family in sorted({r["family"] for r in relations})}
    result = {
        "schema": "simllm-decode-hbm-result-v1", "expectations_commit": EXPECTATIONS_COMMIT,
        "nonvoid": nonvoid, "fatal_findings": findings,
        "verdict": "VOID" if not nonvoid else
                   "PASS" if not mismatches and all(r["passed"] for r in relations) else "FAIL",
        "configurations": dict(sorted(Counter(r["scope"] for r in rows).items())),
        "exact_oracle": {"rows": len(expected), "matched": len(expected) - len(mismatches)
                         if nonvoid else None, "max_tpot_residual_ps": max_residual,
                         "mismatches": mismatches},
        "behavioral_families": scores if nonvoid else None,
        "inactive_bandwidth_identities_unscored": bandwidth_identities,
        "rational_floor_differences_unscored": rational_differences,
        "crossovers": frozen["crossovers"],
        "capacity_status": "unchecked", "production_changed": False,
        "owning_tasks": ["DEPLOY-4", "DEPLOY-5"], "tasks_closed": [],
    }
    return result, relations


def run_study(extra_findings: list[str] | None = None) -> tuple:
    frozen = json.loads((STUDY / "expectations.json").read_text())
    expected = read_csv(STUDY / "expected_cells.csv")
    findings = frozen_findings(frozen) + list(extra_findings or [])
    rows, attempts = [], []
    models = {m["id"]: m for m in [*frozen["models"], frozen["external"]]}
    for dev, (peak, hbm) in frozen["devices"].items():
        gpu = GPU_ENVELOPES[dev]
        if gpu.peak_flops != peak or gpu.mem_bandwidth != hbm:
            findings.append(f"GPU envelope changed: {dev}")

    def blocked(*args, **kwargs):
        attempts.append("process creation attempted")
        raise RuntimeError("process creation blocked in pricing")

    if not findings:
        with ExitStack() as stack:
            stack.enter_context(patch("subprocess.Popen", blocked))
            if hasattr(os, "posix_spawn"):
                stack.enter_context(patch("os.posix_spawn", blocked))
            try:
                for cell in expected:
                    rows.append(price_cell(frozen, models[cell["model"]], cell))
            except (ValueError, TypeError, KeyError, AssertionError, RuntimeError) as error:
                findings.append(f"pricing exception: {type(error).__name__}: {error}")
    findings.extend(attempts)
    summary, relations = audit(rows, frozen, expected, findings)
    summary["pricing_subprocess_attempts"] = len(attempts)
    summary["input_digests"] = {**frozen["pinned_inputs"], **FROZEN_DIGESTS}
    external = next((r for r in rows if r["scope"] == "external"), None)
    if external:
        reference = json.loads((ROOT / frozen["external"]["source"]).read_text())
        candidates = reference["families"]["X3"]["external_rows"]
        anchor = next(r for r in candidates if r["row"] == frozen["external"]["external_row"])
        external_ps = int(Fraction(anchor["tpot_ms"]) * 10**9)
        summary["external_comparison_unscored"] = {
            "source": frozen["external"]["source"], "row": anchor["row"],
            "roofline_tpot_ps": external["tpot_ps"], "external_tpot_ps": external_ps,
            "floor_over_external": str(Fraction(external["tpot_ps"], external_ps)),
            "roofline_below_external": external["tpot_ps"] <= external_ps,
            "external_evidence": "offline performance-database estimate, not live serving",
        }
    return rows, summary, relations


def plot(rows: list[dict], out: Path) -> None:
    """Render six log-scale views at two-column print width."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FixedLocator, FuncFormatter, LogLocator, NullFormatter

    frozen = json.loads((STUDY / "expectations.json").read_text())
    fig = plt.figure(figsize=(7.2, 9.0), layout="constrained")
    grid = fig.add_gridspec(3, 2, width_ratios=(1.0, 1.0))
    axes = [fig.add_subplot(grid[i, 0]) for i in range(3)]
    colors = dict(zip(frozen["devices"], plt.rcParams["axes.prop_cycle"].by_key()["color"]))

    def readable_log(ax, batch_axis=False):
        for axis in (ax.xaxis, ax.yaxis):
            axis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5), numticks=30))
            axis.set_minor_locator(LogLocator(base=10, subs=(3, 4, 6, 7, 8, 9), numticks=30))
            axis.set_minor_formatter(NullFormatter())
            axis.set_major_formatter(FuncFormatter(
                lambda v, _: f"{v / 1000:g}k" if v >= 1000 else f"{v:g}"))
        if batch_axis:
            ax.xaxis.set_major_locator(FixedLocator([1, 4, 16, 64, 256]))
        else:
            ax.yaxis.set_major_locator(LogLocator(base=10, numticks=10))
        ax.tick_params(axis="both", which="major", labelsize=7, pad=2)
        ax.tick_params(axis="both", which="minor", length=2)
        ax.spines[["top", "right"]].set_visible(False)

    for index, (ax, model) in enumerate(zip(axes, frozen["models"])):
        floors = {}
        for dev in frozen["devices"]:
            for context in model["contexts"]:
                selected = [r for r in rows if r["scope"] == "sweep"
                            and r["model"] == model["id"] and r["device"] == dev
                            and r["context"] == context and r["hbm_scale"] == "1"]
                selected.sort(key=lambda r: r["batch"])
                if not selected:
                    continue
                floors[dev] = int(selected[0]["weight_floor_ps"]) / 1e9
                curve = [(r["batch"], r["tpot_ms"], r["throughput_decimal"])
                         for r in selected]
                first = selected[0]
                if first["crossover_batch"] != "none" and context != 2048:
                    cross = Fraction(first["crossover_batch"])
                    time = cross * Fraction(first["flops"], first["peak_flops"])
                    throughput = cross / (model["tp"] * time)
                    curve.append((float(cross), float(time * 1000), float(throughput)))
                    ax.plot(float(time * 1000), float(throughput), "D", color=colors[dev],
                            markerfacecolor="white", markersize=5, zorder=4)
                curve.sort()
                ax.loglog([p[1] for p in curve], [p[2] for p in curve],
                          "--" if context == 2048 else "-", color=colors[dev], linewidth=1)
                ax.plot([r["tpot_ms"] for r in selected],
                        [r["throughput_decimal"] for r in selected],
                        "o", color=colors[dev], markersize=3)
        for dev, floor in floors.items():
            ax.axvline(floor, color=colors[dev], linewidth=0.5, alpha=0.5)
        floor_labels = [f"{d.upper()} {floors[d]:.2f}" for d in frozen["devices"] if d in floors]
        ax.set_title(f"({chr(97 + 2 * index)}) {model['label']}\n"
                     "Weight floors (ms): " + ", ".join(floor_labels[:2]) + "\n"
                     + ", ".join(floor_labels[2:]), fontsize=7, loc="left", pad=6)
        ax.set_xlabel("Time per output token (ms)", fontsize=8)
        ax.set_ylabel("Output rate (tokens/s/GPU)", fontsize=8)
        ax.grid(True, which="major", alpha=0.25)
        ax.margins(x=0.12, y=0.18)
        readable_log(ax)

    right = [fig.add_subplot(grid[i, 1]) for i in range(3)]
    for index, (ax, model) in enumerate(zip(right, frozen["models"])):
        dev = "h200"
        for context, style in zip(model["contexts"], ("-", "--")):
            for scale, marker in (("0.5", "v"), ("1", "o"), ("2", "^")):
                selected = [r for r in rows if r["scope"] == "sweep" and r["model"] == model["id"]
                            and r["device"] == dev and r["context"] == context
                            and r["hbm_scale"] == scale]
                selected.sort(key=lambda r: r["batch"])
                if not selected:
                    continue
                color = {"0.5": "#d62728", "1": "#1f77b4", "2": "#2ca02c"}[scale]
                curve = [(r["batch"], r["tpot_ms"]) for r in selected]
                first = selected[0]
                if first["crossover_batch"] != "none":
                    cross = Fraction(first["crossover_batch"])
                    if selected[0]["batch"] <= cross <= selected[-1]["batch"]:
                        time_ms = cross * Fraction(first["flops"], first["peak_flops"]) * 1000
                        curve.append((float(cross), float(time_ms)))
                        ax.plot(float(cross), float(time_ms), "D", color=color,
                                markerfacecolor="white", markersize=5, zorder=4)
                curve.sort()
                ax.loglog([p[0] for p in curve], [p[1] for p in curve],
                          style, linewidth=1, color=color)
                ax.plot([r["batch"] for r in selected], [r["tpot_ms"] for r in selected],
                        linestyle="none", marker=marker, markersize=3, color=color)
        ax.set_title(f"({chr(98 + 2 * index)}) {model['label']}\n"
                     "H200 bandwidth sweep", fontsize=7, loc="left", pad=6)
        ax.set_xlabel("Decode batch (requests)", fontsize=8)
        ax.set_ylabel("Time per output token (ms)", fontsize=8)
        ax.grid(True, which="major", alpha=0.25)
        bandwidth_handles = [
            Line2D([], [], color=color, marker=marker, linestyle="none", markersize=4,
                   label=f"{scale} bandwidth")
            for scale, marker, color in (("Half", "v", "#d62728"),
                                         ("Nominal", "o", "#1f77b4"),
                                         ("Double", "^", "#2ca02c"))
        ]
        ax.legend(handles=bandwidth_handles, fontsize=7, loc="upper left",
                  framealpha=1, edgecolor="0.85", handlelength=1, labelspacing=0.25)
        ax.margins(x=0.06, y=0.12)
        readable_log(ax, batch_axis=True)

    handles = [Line2D([], [], color=colors[d], label=d.upper()) for d in reversed(frozen["devices"])]
    handles += [Line2D([], [], color="black", linestyle="-", label="C=1; EP72: C=2000"),
                Line2D([], [], color="black", linestyle="--", label="C=2048"),
                Line2D([], [], color="black", marker="D", markerfacecolor="white",
                       linestyle="none", label="Analytical B*"),
                Line2D([], [], color="gray", linewidth=0.5, label="Left: weight floor")]
    fig.suptitle("Decode roofline\nNominal bandwidth (left); H200 bandwidth sweep (right)",
                 fontsize=10)
    fig.legend(handles=handles, loc="outside lower center", ncols=3, fontsize=7,
               title="Device colors apply to left panels; C is context length (tokens)",
               title_fontsize=7, frameon=False)
    out.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(out / f"decode-hbm-crossover.{suffix}", dpi=240,
                    metadata={"CreationDate": None} if suffix == "pdf" else None)
    plt.close(fig)


def write_results(out: Path, rows: list[dict], summary: dict, relations: list[dict]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    if rows:
        with (out / "results.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    for name, data in (("results.json", summary), ("relation_instances.json", relations)):
        # Bytes, not text: Windows text mode would translate the LF terminators.
        (out / name).write_bytes((json.dumps(data, indent=2, sort_keys=True) + "\n").encode("utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, help="bulk evidence directory; or SIMLLM_S2_BULK_ROOT")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    if args.out is None:
        configured = os.environ.get("SIMLLM_S2_BULK_ROOT")
        if not configured:
            parser.error("set SIMLLM_S2_BULK_ROOT or pass --out")
        args.out = Path(configured)
    rows, summary, relations = run_study(chronology_findings())
    write_results(args.out, rows, summary, relations)
    if args.plot and summary["nonvoid"]:
        plot(rows, args.out / "figures")
    print(f"verdict={summary['verdict']} configurations={summary['configurations']}")
    print(f"exact_oracle={summary['exact_oracle']}")
    print(f"behavioral_families={summary['behavioral_families']}")
    print(f"fatal_findings={summary['fatal_findings']}")
    print(f"pricing_subprocess_attempts={summary['pricing_subprocess_attempts']}")
    print(f"expectations_commit={EXPECTATIONS_COMMIT}")
    if summary["verdict"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
