"""Run the frozen compute-fidelity study.

Three separable parts, in the order that keeps value if a later one stalls:

VAR   re-reads the immutable Turing capture tracked in
      examples/compute_calibration_v1/calibration.json and adds one targeted
      device probe, to decide what the 2 percent coefficient-of-variation
      ceiling that kept COMP-1 open actually measured.
XFER  machine-checks the by-construction facts behind the statement of what a
      Turing anchor transfers to a production envelope.
FIX   bounds the fixed per-step cost the modeled compute path omits, from a
      Turing launch capture and the frozen eager-mode launch bracket of a
      24-layer top-8 MoE decode step.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
EXPECTATIONS_PATH = HERE / "expectations.json"
CALIBRATION_PATH = REPO_ROOT / "examples" / "compute_calibration_v1" / "calibration.json"
PROFILE_TABLE_PATH = REPO_ROOT / "examples" / "compute_calibration_v1" / "profile_table.json"
PROBE_SOURCE = REPO_ROOT / "tools" / "compute_capture" / "gpu_fixed_cost_probe.cu"
EXPECTATION_COMMIT = "62c088e"


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return value


def _validate_registry(expectations: dict[str, Any], out: Path) -> None:
    if expectations["schema"] != "simllm-compute-fidelity-v1-expectations-v1":
        raise AssertionError("expectation schema drifted")
    prior = expectations["prior_observations"]
    if len(prior["published_failing_cells"]) != 3:
        raise AssertionError("published failing cell inventory drifted")
    if prior["total_cell_count"] - len(prior["published_failing_cells"]) != prior[
        "unobserved_cell_count"
    ]:
        raise AssertionError("genuine-risk denominator drifted")
    counts = expectations["launch_count"]
    per_layer = counts["per_layer"]
    layer_low = sum(bounds[0] for bounds in per_layer.values())
    layer_high = sum(bounds[1] for bounds in per_layer.values())
    if [layer_low, layer_high] != counts["per_layer_total"]:
        raise AssertionError("per-layer launch enumeration does not sum to its total")
    per_step = counts["per_step"]
    step_low = sum(bounds[0] for bounds in per_step.values())
    step_high = sum(bounds[1] for bounds in per_step.values())
    if [step_low, step_high] != counts["per_step_total"]:
        raise AssertionError("per-step launch enumeration does not sum to its total")
    layers = counts["num_layers"]
    if [
        layers * layer_low + step_low,
        layers * layer_high + step_high,
    ] != counts["launches_per_step"]:
        raise AssertionError("launch bracket does not follow the enumeration")
    scored = expectations["scored_relations"]
    if sum(scored.values()) != 102:
        raise AssertionError("scored instance inventory drifted")
    if len(expectations["fatal_unscored_guards"]) != 12:
        raise AssertionError("fatal guard inventory drifted")
    run_root = os.environ.get("SIMLLM_WAVE11_RUN_ROOT")
    if not run_root:
        raise RuntimeError("SIMLLM_WAVE11_RUN_ROOT must be configured")
    try:
        out.resolve().relative_to(Path(run_root).resolve())
    except ValueError as error:
        raise ValueError("output must remain under the branch wave-11 root") from error


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args()


def _tool(path: Path, name: str) -> Path:
    candidate = path / "bin" / name
    if not candidate.is_file():
        raise FileNotFoundError(f"CUDA tool {name!r} is unavailable below --cuda-root: {candidate}")
    return candidate


def _run(
    command: Sequence[str | Path],
    *,
    timeout: int = 900,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        [str(item) for item in command],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if check and completed.returncode != 0:
        raise RuntimeError(
            f"command failed with status {completed.returncode}: "
            f"{' '.join(str(item) for item in command)}\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    return completed


def _write_log(path: Path, completed: subprocess.CompletedProcess[str]) -> None:
    path.write_text(
        f"returncode: {completed.returncode}\n"
        f"stdout:\n{completed.stdout}\n"
        f"stderr:\n{completed.stderr}\n",
        encoding="utf-8",
    )


def _git_revision() -> str:
    return _run(("git", "rev-parse", "HEAD"), timeout=60).stdout.strip()


def _nearest_rank(values: Sequence[float], percentile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("percentile of an empty sample")
    index = max(1, min(len(ordered), int(-(-percentile * len(ordered) // 1))))
    return float(ordered[index - 1])


def _population_cv(values: Sequence[float]) -> float:
    mean = statistics.fmean(values)
    return statistics.pstdev(values, mu=mean) / mean


def _cell_statistics(durations: Sequence[int], threshold: float) -> dict[str, Any]:
    median = float(statistics.median(durations))
    ratios = [value / median for value in durations]
    core = [value for value, ratio in zip(durations, ratios, strict=True) if ratio <= threshold]
    excursions = [
        value for value, ratio in zip(durations, ratios, strict=True) if ratio > threshold
    ]
    return {
        "count": len(durations),
        "median_ps": median,
        "minimum_ps": float(min(durations)),
        "maximum_ps": float(max(durations)),
        "all_sample_cv": _population_cv(durations),
        "trimmed_cv": _population_cv(core) if len(core) > 1 else 0.0,
        "core_count": len(core),
        "excursion_count": len(excursions),
        "excursion_fraction": len(excursions) / len(durations),
        "maximum_excursion_ratio": max(ratios),
        "quartile_relative_spread": (
            _nearest_rank(durations, 0.75) - _nearest_rank(durations, 0.25)
        )
        / median,
    }


def _parse_probe(text: str) -> dict[str, Any]:
    scalars: dict[str, str] = {}
    samples: list[dict[str, float]] = []
    in_samples = False
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("samples_begin"):
            in_samples = True
            continue
        if line == "samples_end":
            in_samples = False
            continue
        fields = line.split(",")
        if in_samples and fields[0] == "sample":
            samples.append(
                {
                    "launch": float(fields[1]),
                    "event_duration_ns": float(fields[2]),
                    "kernel_span_ns": float(fields[3]),
                    "block_cycle_sum": float(fields[4]),
                    "block_resident_ns_sum": float(fields[5]),
                    "block_count": float(fields[6]),
                }
            )
            continue
        if len(fields) == 2:
            scalars[fields[0]] = fields[1]
    return {"scalars": scalars, "samples": samples}


def _attribute(
    samples: Sequence[dict[str, float]],
    duration_key: str,
    threshold: float,
    tolerance: float,
) -> dict[str, Any]:
    durations = [sample[duration_key] for sample in samples]
    cycles = [sample["block_cycle_sum"] for sample in samples]
    median_duration = float(statistics.median(durations))
    median_cycles = float(statistics.median(cycles))
    rows = []
    clock_caused = 0
    residency_caused = 0
    unattributed = 0
    for sample in samples:
        duration_ratio = sample[duration_key] / median_duration
        cycle_ratio = sample["block_cycle_sum"] / median_cycles
        if duration_ratio <= threshold:
            continue
        is_clock = abs(cycle_ratio - 1.0) <= tolerance
        is_residency = (
            abs(cycle_ratio - duration_ratio) <= tolerance * duration_ratio
            and duration_ratio > threshold
        )
        if is_clock and not is_residency:
            label = "clock"
            clock_caused += 1
        elif is_residency and not is_clock:
            label = "residency"
            residency_caused += 1
        else:
            label = "unattributed"
            unattributed += 1
        rows.append(
            {
                "launch": int(sample["launch"]),
                "duration_ratio": duration_ratio,
                "cycle_ratio": cycle_ratio,
                "effective_clock_mhz": (
                    1000.0 * sample["block_cycle_sum"] / sample["block_resident_ns_sum"]
                    if sample["block_resident_ns_sum"] > 0
                    else 0.0
                ),
                "attribution": label,
            }
        )
    excursions = len(rows)
    baseline_clock = 1000.0 * sum(cycles) / sum(
        sample["block_resident_ns_sum"] for sample in samples
    )
    # Descriptive only, never scored: the effective SM clock of an excursion
    # relative to the whole run's effective clock separates a clock-state drop
    # from longer block residency at an unchanged clock. The frozen attribution
    # rule above uses cycle counts and is evaluated exactly as registered; this
    # measurement is reported beside it because a memory-bound kernel spends
    # fewer SM cycles when its clock drops, which the frozen rule does not
    # anticipate.
    for row in rows:
        row["effective_clock_ratio"] = (
            row["effective_clock_mhz"] / baseline_clock if baseline_clock else 0.0
        )
    clock_state_rows = [row for row in rows if row["effective_clock_ratio"] < 0.9]
    dominant = max(
        (("clock", clock_caused), ("residency", residency_caused)),
        key=lambda item: item[1],
    )
    return {
        "effective_clock_drop_count": len(clock_state_rows),
        "effective_clock_drop_minimum_ratio": (
            min(row["effective_clock_ratio"] for row in clock_state_rows)
            if clock_state_rows
            else 1.0
        ),
        "duration_key": duration_key,
        "median_duration_ns": median_duration,
        "median_block_cycle_sum": median_cycles,
        "run_effective_clock_mhz": baseline_clock,
        "excursion_count": excursions,
        "excursion_fraction": excursions / len(samples),
        "clock_caused": clock_caused,
        "residency_caused": residency_caused,
        "unattributed": unattributed,
        "dominant_cause": dominant[0] if excursions else "none",
        "dominant_fraction": (dominant[1] / excursions) if excursions else 0.0,
        "rows": rows,
    }


def _xfer_guards(expectations: dict[str, Any]) -> list[dict[str, Any]]:
    from simllm.compute import (
        GPU_ENVELOPES,
        HostInitiationModel,
        KernelSpec,
        ProfileTableProvider,
        RooflineProvider,
    )

    guards: list[dict[str, Any]] = []
    table_json = _load_json(PROFILE_TABLE_PATH)
    envelope_keys = set(GPU_ENVELOPES)
    entry_gpus = sorted({str(entry["gpu"]) for entry in table_json["entries"]})
    guards.append(
        {
            "id": "XFER-G1_table_gpu_identity",
            "passed": (
                "GTX 1660 Ti" in str(table_json["provenance"]["gpu"])
                and all(gpu.startswith("gtx1660-ti-sm75-") for gpu in entry_gpus)
                and not (set(entry_gpus) & envelope_keys)
            ),
            "detail": {
                "provenance_gpu": table_json["provenance"]["gpu"],
                "entry_gpus": entry_gpus,
                "envelope_keys": sorted(envelope_keys),
            },
        }
    )

    provider = ProfileTableProvider.load(PROFILE_TABLE_PATH)
    covered = table_json["entries"][0]
    covered_config = tuple((str(name), int(value)) for name, value in covered["config"])
    probe_kernel = KernelSpec(
        name=str(covered["kernel"]),
        flops=0.0,
        bytes_moved=0.0,
        config=covered_config,
    )
    try:
        provider.estimate(probe_kernel, GPU_ENVELOPES["b100"])
        fails_closed = False
        detail_message = "b100 query returned an estimate"
    except KeyError as error:
        fails_closed = True
        detail_message = f"KeyError: {error}"
    guards.append(
        {
            "id": "XFER-G2_table_fails_closed_on_b100",
            "passed": fails_closed,
            "detail": {"kernel": covered["kernel"], "message": detail_message},
        }
    )

    host = HostInitiationModel()
    guards.append(
        {
            "id": "XFER-G3_host_model_defaults",
            "passed": host.initiation_delay_ps == 0 and host.profile == "ideal",
            "detail": {
                "initiation_delay_ps": host.initiation_delay_ps,
                "profile": host.profile,
            },
        }
    )

    # The first registered run asserted exact integer equality between the
    # doubled kernel and twice the single kernel. That failed by 1 ps, because
    # the true value 793,650,793.65 ps is rounded once per call and not because
    # the provider carries a fixed term. The claim is therefore tested in the
    # rounding-free form as well: a kernel with no work must return exactly
    # zero, which no additive launch, scheduling or sampling constant could
    # survive. The proportionality check keeps its original form with the one
    # picosecond of unavoidable integer rounding allowed, six orders of
    # magnitude below any per-launch cost this study measures.
    roofline = RooflineProvider()
    gpu = GPU_ENVELOPES["b100"]
    zero = roofline.estimate(KernelSpec(name="probe", flops=0.0, bytes_moved=0.0), gpu)
    single = roofline.estimate(
        KernelSpec(name="probe", flops=1.0e12, bytes_moved=5.564e8), gpu
    )
    doubled = roofline.estimate(
        KernelSpec(name="probe", flops=2.0e12, bytes_moved=1.1128e9), gpu
    )
    residual_ps = doubled.duration_ps - 2 * single.duration_ps
    guards.append(
        {
            "id": "XFER-G4_roofline_has_no_additive_term",
            "passed": zero.duration_ps == 0 and abs(residual_ps) <= 1,
            "detail": {
                "zero_work_ps": zero.duration_ps,
                "single_ps": single.duration_ps,
                "doubled_ps": doubled.duration_ps,
                "proportionality_residual_ps": residual_ps,
            },
        }
    )

    envelope = expectations["envelope"]
    weight_bytes = envelope["active_weight_bytes"]
    tolerance = envelope["relative_tolerance"]
    checks = {}
    for key, expected_us in envelope["floors_us"].items():
        observed = weight_bytes / GPU_ENVELOPES[key].mem_bandwidth * 1.0e6
        checks[key] = {"observed_us": observed, "expected_us": expected_us}
    modeled = (
        weight_bytes
        / (GPU_ENVELOPES["b100"].mem_bandwidth * envelope["roofline_derate"])
        * 1.0e6
    )
    checks["modeled_b100_derated"] = {
        "observed_us": modeled,
        "expected_us": envelope["modeled_decode_compute_us"],
    }
    checks["modeled_over_b100_floor"] = {
        "observed_us": modeled / checks["b100"]["observed_us"],
        "expected_us": envelope["modeled_over_b100_floor"],
    }
    checks["h100_floor_over_modeled"] = {
        "observed_us": checks["h100"]["observed_us"] / modeled,
        "expected_us": envelope["h100_floor_over_modeled"],
    }
    guards.append(
        {
            "id": "XFER-G5_envelope_arithmetic",
            "passed": all(
                abs(item["observed_us"] - item["expected_us"])
                <= tolerance * abs(item["expected_us"])
                for item in checks.values()
            ),
            "detail": checks,
        }
    )
    return guards


def _production(args: argparse.Namespace, expectations: dict[str, Any]) -> int:
    from simllm.compute import ComputeCalibrationArtifact, sha256_file

    if args.out.exists():
        raise FileExistsError(f"immutable output directory already exists: {args.out}")
    args.out.mkdir(parents=True)
    build_dir = args.out / "build"
    capture_dir = args.out / "capture"
    build_dir.mkdir()
    capture_dir.mkdir()

    cuda_root = args.cuda_root.resolve()
    nvcc = _tool(cuda_root, "nvcc")
    probe = build_dir / "gpu_fixed_cost_probe"
    compile_run = _run(
        (nvcc, "-std=c++17", "-O3", "-lineinfo", "-arch=sm_75", PROBE_SOURCE, "-o", probe),
        timeout=600,
    )
    _write_log(build_dir / "nvcc.log", compile_run)

    guards: list[dict[str, Any]] = []
    scored: dict[str, dict[str, Any]] = {}

    # -------------------------------------------------------------- VAR part
    artifact = ComputeCalibrationArtifact.load(CALIBRATION_PATH)
    variation = expectations["variation"]
    threshold = variation["excursion_ratio_threshold"]
    total_samples = sum(len(cell.durations_ps) for cell in artifact.cells)
    guards.append(
        {
            "id": "G1_artifact_inventory",
            "passed": (
                len(artifact.cells) == expectations["prior_observations"]["total_cell_count"]
                and total_samples == expectations["prior_observations"]["total_sample_count"]
            ),
            "detail": {"cells": len(artifact.cells), "samples": total_samples},
        }
    )

    published = {
        (item["family"], item["dtype"], item["shape"]): item
        for item in expectations["prior_observations"]["published_failing_cells"]
    }
    cell_rows: list[dict[str, Any]] = []
    for cell in artifact.cells:
        shape = cell.config[-1][1]
        key = (cell.family, cell.dtype, shape)
        stats = _cell_statistics(cell.durations_ps, threshold)
        stats.update(
            {
                "family": cell.family,
                "dtype": cell.dtype,
                "shape": shape,
                "split": cell.split,
                "published": key in published,
            }
        )
        cell_rows.append(stats)

    published_rows = [row for row in cell_rows if row["published"]]
    unobserved_rows = [row for row in cell_rows if not row["published"]]
    cv_ceiling = variation["all_sample_cv_ceiling_percent"] / 100.0
    guards.append(
        {
            "id": "G2_published_cv_reproduction",
            "passed": len(published_rows) == 3
            and all(
                abs(
                    row["all_sample_cv"] * 100.0
                    - published[(row["family"], row["dtype"], row["shape"])][
                        "coefficient_of_variation_percent"
                    ]
                )
                <= 1.0e-3
                for row in published_rows
            ),
            "detail": [
                {
                    "family": row["family"],
                    "dtype": row["dtype"],
                    "shape": row["shape"],
                    "recomputed_cv_percent": row["all_sample_cv"] * 100.0,
                    "published_cv_percent": published[
                        (row["family"], row["dtype"], row["shape"])
                    ]["coefficient_of_variation_percent"],
                }
                for row in published_rows
            ],
        }
    )
    failures = [row for row in cell_rows if row["all_sample_cv"] >= cv_ceiling]
    guards.append(
        {
            "id": "G3_all_sample_cv_failure_count",
            "passed": len(failures) == variation["expected_all_sample_cv_failures"],
            "detail": {
                "observed": len(failures),
                "expected": variation["expected_all_sample_cv_failures"],
                "cells": [
                    {
                        "family": row["family"],
                        "dtype": row["dtype"],
                        "shape": row["shape"],
                        "all_sample_cv_percent": row["all_sample_cv"] * 100.0,
                    }
                    for row in failures
                ],
            },
        }
    )
    guards.append(
        {
            "id": "G4_genuine_risk_denominator",
            "passed": len(unobserved_rows)
            == expectations["prior_observations"]["unobserved_cell_count"],
            "detail": {"unobserved_cells": len(unobserved_rows)},
        }
    )

    trimmed_ceiling = variation["trimmed_cv_ceiling_percent"] / 100.0
    ratio_ceiling = variation["maximum_excursion_ratio_ceiling"]
    s1_pass = [row for row in unobserved_rows if row["trimmed_cv"] < trimmed_ceiling]
    s3_pass = [row for row in unobserved_rows if row["maximum_excursion_ratio"] <= ratio_ceiling]
    aggregate_excursions = sum(row["excursion_count"] for row in cell_rows)
    aggregate_fraction = aggregate_excursions / total_samples
    scored["VAR-S1_trimmed_cv_below_ceiling"] = {
        "passed": len(s1_pass),
        "total": len(unobserved_rows),
        "worst": max(row["trimmed_cv"] for row in unobserved_rows),
    }
    scored["VAR-S3_maximum_excursion_ratio_bounded"] = {
        "passed": len(s3_pass),
        "total": len(unobserved_rows),
        "worst": max(row["maximum_excursion_ratio"] for row in unobserved_rows),
    }
    scored["VAR-S2_aggregate_excursion_fraction"] = {
        "passed": int(aggregate_fraction < variation["aggregate_excursion_fraction_ceiling"]),
        "total": 1,
        "worst": aggregate_fraction,
    }

    # ------------------------------------------------------------ PROBE part
    identity_run = _run((probe, "--identity"), timeout=120)
    _write_log(capture_dir / "probe_identity.log", identity_run)
    identity = _parse_probe(identity_run.stdout)["scalars"]
    guards.append(
        {
            "id": "G5_probe_device_identity",
            "passed": (
                identity.get("device_name") == expectations["probe"]["device_model"]
                and identity.get("compute_capability")
                == expectations["probe"]["compute_capability"]
            ),
            "detail": identity,
        }
    )

    stability_run = _run(
        (probe, "--stability", str(expectations["probe"]["stability_launches"])),
        timeout=1800,
    )
    (capture_dir / "stability.csv").write_text(stability_run.stdout, encoding="utf-8")
    _write_log(capture_dir / "probe_stability.log", stability_run)
    stability = _parse_probe(stability_run.stdout)
    samples = stability["samples"]
    guards.append(
        {
            "id": "G7_probe_positive_samples",
            "passed": len(samples) == expectations["probe"]["stability_launches"]
            and all(
                sample["kernel_span_ns"] > 0
                and sample["event_duration_ns"] > 0
                and sample["block_cycle_sum"] > 0
                and sample["block_count"] == expectations["probe"]["stability_blocks"]
                for sample in samples
            ),
            "detail": {"samples": len(samples)},
        }
    )
    span_median_us = statistics.median(sample["kernel_span_ns"] for sample in samples) / 1000.0
    tracked_median_us = expectations["probe"]["tracked_cell_median_us"]
    guards.append(
        {
            "id": "G6_probe_median_cross_check",
            "passed": abs(span_median_us - tracked_median_us)
            <= expectations["probe"]["median_cross_check_relative_tolerance"]
            * tracked_median_us,
            "detail": {
                "probe_kernel_span_median_us": span_median_us,
                "tracked_cell_median_us": tracked_median_us,
                "relative_difference": (span_median_us - tracked_median_us) / tracked_median_us,
            },
        }
    )

    attributions = {
        key: _attribute(
            samples,
            key,
            threshold,
            expectations["probe"]["attribution_cycle_ratio_tolerance"],
        )
        for key in ("kernel_span_ns", "event_duration_ns")
    }
    primary = attributions["kernel_span_ns"]
    secondary = attributions["event_duration_ns"]
    minimum_fraction = expectations["probe"]["attribution_minimum_fraction"]
    probe_ceiling = expectations["probe"]["excursion_fraction_ceiling"]
    probe_passed = 0
    probe_rows = []
    reproduced = primary["excursion_count"] > 0 and secondary["excursion_count"] > 0
    probe_rows.append({"id": "PROBE-1_excursion_reproduced", "passed": reproduced})
    attributed = (
        primary["dominant_fraction"] >= minimum_fraction
        and secondary["dominant_fraction"] >= minimum_fraction
        and primary["dominant_cause"] == secondary["dominant_cause"]
    )
    probe_rows.append({"id": "PROBE-2_excursions_attributed", "passed": attributed})
    sparse = all(
        0.0 < attribution["excursion_fraction"] < probe_ceiling
        for attribution in attributions.values()
    )
    probe_rows.append({"id": "PROBE-3_excursions_sparse", "passed": sparse})
    probe_passed = sum(1 for row in probe_rows if row["passed"])
    scored["PROBE_reproduction_attribution_sparsity"] = {
        "passed": probe_passed,
        "total": 3,
        "rows": probe_rows,
    }

    # -------------------------------------------------------------- FIX part
    launch_run = _run(
        (
            probe,
            "--launch",
            str(expectations["probe"]["launch_iterations"]),
            str(expectations["probe"]["graph_nodes"]),
            str(expectations["probe"]["graph_replays"]),
            str(expectations["probe"]["long_isolated_iterations"]),
            str(expectations["probe"]["long_backtoback_iterations"]),
        ),
        timeout=1800,
    )
    (capture_dir / "launch.csv").write_text(launch_run.stdout, encoding="utf-8")
    _write_log(capture_dir / "probe_launch.log", launch_run)
    launch = {
        key: float(value)
        for key, value in _parse_probe(launch_run.stdout)["scalars"].items()
        if key
        not in {"device_name", "compute_capability", "mode"}
    }

    fixed = expectations["fixed_cost"]
    band = fixed["cpu_enqueue_band_ns"]
    modeled_us = fixed["modeled_decode_compute_us"]
    launches = expectations["launch_count"]["launches_per_step"]
    fix_rows = [
        {
            "id": "FIX-1_cpu_enqueue_in_band",
            "passed": band[0] <= launch["empty_stream_cpu_enqueue_ns"] <= band[1],
            "observed_ns": launch["empty_stream_cpu_enqueue_ns"],
        },
        {
            "id": "FIX-2_backtoback_exceeds_isolated",
            "passed": launch["long_backtoback_ns"] > launch["long_isolated_ns"],
            "observed_ns": launch["long_backtoback_ns"] - launch["long_isolated_ns"],
        },
        {
            "id": "FIX-3_graph_cheaper_than_eager",
            "passed": launch["empty_graph_ns"] < launch["empty_stream_pipelined_ns"],
            "observed_ns": launch["empty_stream_pipelined_ns"] - launch["empty_graph_ns"],
        },
        {
            "id": "FIX-4_eager_upper_bound_exceeds_modeled_compute",
            "passed": launches[1] * launch["empty_stream_pipelined_ns"] / 1000.0 > modeled_us,
            "observed_us": launches[1] * launch["empty_stream_pipelined_ns"] / 1000.0,
        },
    ]
    scored["FIX_launch_cost_bracket_and_bound"] = {
        "passed": sum(1 for row in fix_rows if row["passed"]),
        "total": 4,
        "rows": fix_rows,
    }

    def _bound(per_launch_ns: float) -> dict[str, float]:
        low_us = launches[0] * per_launch_ns / 1000.0
        high_us = launches[1] * per_launch_ns / 1000.0
        return {
            "per_launch_ns": per_launch_ns,
            "step_fixed_low_us": low_us,
            "step_fixed_high_us": high_us,
            "omitted_low_us": max(0.0, low_us - modeled_us),
            "omitted_high_us": max(0.0, high_us - modeled_us),
        }

    prefill_us = fixed["published_prefill_step0_makespan_ps"] / 1.0e6
    bounds = {
        "graph_replay": _bound(launch["empty_graph_ns"]),
        "device_gap": _bound(launch["stamped_device_gap_ns"]),
        "eager_host_bound": _bound(launch["empty_stream_pipelined_ns"]),
    }
    # A step is not the sum of its service time and its launch cost: with the
    # host running ahead the two overlap, so the step floor is
    # max(kernel service, launches * per-launch cost). The modeled makespan
    # already contains the compute term, so the projection replaces that term
    # rather than adding to it, i.e. it adds only the omitted excess.
    ttft = {
        name: {
            "modeled_ttft_us": prefill_us,
            "ttft_with_fixed_low_us": prefill_us + value["omitted_low_us"],
            "ttft_with_fixed_high_us": prefill_us + value["omitted_high_us"],
            "relative_increase_low": value["omitted_low_us"] / prefill_us,
            "relative_increase_high": value["omitted_high_us"] / prefill_us,
        }
        for name, value in bounds.items()
    }

    guards.extend(_xfer_guards(expectations))

    results = {
        "schema": "simllm-compute-fidelity-v1-results-v1",
        "expectation_commit": EXPECTATION_COMMIT,
        "observed_commit": _git_revision(),
        "probe_source_sha256": sha256_file(PROBE_SOURCE),
        "calibration_sha256": sha256_file(CALIBRATION_PATH),
        "profile_table_sha256": sha256_file(PROFILE_TABLE_PATH),
        "probe_identity": identity,
        "cells": cell_rows,
        "published_cells": published_rows,
        "attribution": attributions,
        "launch": launch,
        "launch_bracket": launches,
        "bounds": bounds,
        "ttft_projection": ttft,
        "scored": scored,
        "fatal_guards": guards,
    }
    (args.out / "results.json").write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (HERE / "results.json").write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    failed_guards = [guard["id"] for guard in guards if not guard["passed"]]
    scored_passed = sum(int(item["passed"]) for item in scored.values())
    scored_total = sum(int(item["total"]) for item in scored.values())
    print(f"fatal guards: {len(guards) - len(failed_guards)} of {len(guards)} held")
    for guard_id in failed_guards:
        print(f"  VIOLATED: {guard_id}")
    for name, item in scored.items():
        print(f"scored {name}: {item['passed']} of {item['total']}")
    print(f"scored total: {scored_passed} of {scored_total}")
    if failed_guards:
        print("run is VOID for closure purposes: a fatal guard was violated")
        return 1
    if scored_passed != scored_total:
        print("registered acceptance not met: a scored relation was refuted")
        return 1
    return 0


def main() -> int:
    args = _parse_args()
    expectations = _load_json(EXPECTATIONS_PATH)
    _validate_registry(expectations, args.out)
    scored = sum(expectations["scored_relations"].values())
    print(
        f"check-only: 50 cells, {expectations['prior_observations']['total_sample_count']} "
        f"tracked samples, {expectations['prior_observations']['unobserved_cell_count']} "
        f"never-observed cells, {scored} scored instances, "
        f"{len(expectations['fatal_unscored_guards'])} fatal guards, launch bracket "
        f"{expectations['launch_count']['launches_per_step']}"
    )
    if args.check_only:
        return 0
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    return _production(args, expectations)


if __name__ == "__main__":
    raise SystemExit(main())
