"""Run the frozen live host-step sensitivity after accepted calibration."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

EXPECTATIONS = HERE / "expectations.json"
CALIBRATION = HERE / "calibration.json"
PRIOR_LIVE_ATTEMPT = HERE / "live_attempt1.json"
RESULTS = HERE / "results.json"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--htsim-rnic", type=Path, required=True)
    parser.add_argument("--baseline-cell", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args()


def _run(command: tuple[object, ...]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(part) for part in command],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _whole_nanosecond_enclosure(value_ps: int) -> int:
    """Return the smallest whole-nanosecond value not below ``value_ps``."""

    return ((value_ps + 999) // 1000) * 1000


def _check(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    values = _load_json(EXPECTATIONS)
    calibration = _load_json(CALIBRATION)
    if values["schema"] != "simllm-host-step-cost-v1-expectations-v1":
        raise AssertionError("expectation schema drifted")
    if calibration["schema"] != "simllm-host-step-calibration-v1":
        raise AssertionError("calibration schema drifted")
    if calibration["run_status"] != "accepted":
        raise RuntimeError("live study requires an accepted calibration")
    if calibration["attempt"] != values["calibration_attempt"]:
        raise RuntimeError("calibration attempt disagrees with the freeze")
    if values["live_attempt"] != 2:
        raise AssertionError("live attempt identity drifted")
    if calibration["fatal_guard_failures"]:
        raise RuntimeError("accepted calibration unexpectedly has fatal findings")
    if not all(row["passed"] for row in calibration["scored_relations"]):
        raise RuntimeError("accepted calibration unexpectedly has a scored miss")
    if values["representative_step"]["acceptance_decode_multiplier"] != [
        1.8,
        7.75,
    ]:
        raise AssertionError("flagship multiplier band drifted")
    if sum(values["scored_live_relations"].values()) != 12:
        raise AssertionError("scored live inventory drifted")
    prior = _load_json(PRIOR_LIVE_ATTEMPT)
    prior_freeze = values["prior_live_attempt"]
    if _sha256(PRIOR_LIVE_ATTEMPT) != prior_freeze["retained_void_sha256"]:
        raise RuntimeError("retained live attempt-one identity drifted")
    if (
        prior["run_status"] != "void"
        or prior["behavioral_score_interpretable"] is not False
        or prior["fatal_guard_failures"] != [prior_freeze["failed_fatal_guard"]]
    ):
        raise RuntimeError("retained live attempt one is not the frozen void run")
    fixture = values["live_fixture"]
    if fixture["network_profile"] != "rnic-nn-fluid":
        raise AssertionError("live network profile drifted")
    if fixture["linkspeed_bps"] != 400_000_000_000:
        raise AssertionError("live link rate drifted")
    if [row["step_index"] for row in fixture["steps"]] != [0, 1, 2]:
        raise AssertionError("live step sequence drifted")
    ideal = fixture["ideal_step_ps"]
    if ideal["tpot"] * 2 != sum(ideal["decode"]):
        raise AssertionError("ideal TPOT arithmetic drifted")
    representative = values["representative_step"]
    if (
        representative["modeled_compute_ps"]
        + representative["modeled_decode_network_ps"]
        != representative["modeled_decode_makespan_ps"]
    ):
        raise AssertionError("representative step does not conserve")
    for bounds in fixture["network_physical_bounds_ps"].values():
        if len(bounds) != 2 or not 0 < bounds[0] <= bounds[1]:
            raise AssertionError("network physical bounds drifted")
    launches = values["launch_count"]
    capture_bounds = values["capture"]["acceptance_ps"]
    launch_floor_bounds = (
        launches["minimum"] * capture_bounds["graph_replay"][0],
        launches["maximum"] * capture_bounds["eager_host_bound"][1],
    )
    budget = values["mission_budget"]
    ratios = [
        (launch_ps + real_ps) / (launch_ps + budget["modeled_network_ps"])
        for launch_ps in launch_floor_bounds
        for real_ps in budget["plausible_collective_network_ps"]
    ]
    acceptance = budget["acceptance_conditional_turing_residual_optimism"]
    if min(ratios) < acceptance[0] or max(ratios) > acceptance[1]:
        raise AssertionError("conditional budget enclosure drifted")
    if not args.htsim_rnic.is_file() or not os.access(args.htsim_rnic, os.X_OK):
        raise FileNotFoundError("htsim_rnic is missing or not executable")
    converter_value = os.environ.get("SIMLLM_TXT2BIN")
    if not converter_value:
        raise RuntimeError("SIMLLM_TXT2BIN must be configured")
    converter = Path(converter_value)
    if not converter.is_file() or not os.access(converter, os.X_OK):
        raise FileNotFoundError("SIMLLM_TXT2BIN is missing or not executable")
    for name, expected in fixture["input_sha256"].items():
        if name == "routed_experts_json":
            path = args.baseline_cell / "routed-experts.json"
        elif name == "steps_jsonl":
            path = args.baseline_cell / "steps.jsonl"
        else:
            raise AssertionError(f"unknown fixture input: {name}")
        if not path.is_file():
            raise FileNotFoundError(f"live fixture input is missing: {path}")
        if _sha256(path) != expected:
            raise RuntimeError(f"live fixture input identity drifted: {path.name}")
    root_value = os.environ.get("SIMLLM_WAVE12_RUN_ROOT")
    if not root_value:
        raise RuntimeError("SIMLLM_WAVE12_RUN_ROOT must be configured")
    args.out.resolve().relative_to(Path(root_value).resolve())
    if args.out.exists():
        raise FileExistsError(f"output already exists: {args.out}")
    return values, calibration


def _clean_head() -> str:
    status = _run(("git", "status", "--porcelain", "--untracked-files=all")).stdout
    if status:
        raise RuntimeError(
            "live evidence requires a clean tracked and untracked worktree before output"
        )
    return _run(("git", "rev-parse", "HEAD")).stdout.strip()


def _expectation_commit() -> str:
    return _run(
        (
            "git",
            "log",
            "-1",
            "--format=%H",
            "--",
            EXPECTATIONS,
            HERE / "expectations.md",
        )
    ).stdout.strip()


def _dims() -> Any:
    from simllm.compute import ModelDims

    return ModelDims(
        num_layers=24,
        hidden_size=1024,
        intermediate_size=512,
        num_heads=16,
        num_kv_heads=8,
        head_size=64,
        vocab_size=49_152,
        dtype_bytes=2,
        num_experts=32,
        top_k=8,
        moe_intermediate_size=512,
        local_num_experts=4,
    )


def _fixed_provider(by_context_ps: dict[int, int]) -> Any:
    """Supply the three frozen source services by their distinct KV shape."""

    from simllm.compute import ComputeProvider, DurationEstimate

    class FixedStepProvider(ComputeProvider):
        precision_compute_level = None

        def estimate(self, kernel: Any, gpu: Any) -> Any:
            config = dict(kernel.config)
            context = int(config["kv_tokens"])
            try:
                duration_ps = by_context_ps[context]
            except KeyError as exc:
                raise KeyError(f"no fixed service for kv_tokens={context}") from exc
            return DurationEstimate(
                duration_ps=duration_ps,
                bound="declared-fixed",
                uncertainty=0.0,
            )

        def estimate_layers(self, kernel: Any, gpu: Any, num_layers: int) -> None:
            return None

    return FixedStepProvider()


def _supply(baseline_cell: Path, step_indices: tuple[int, ...]) -> Any:
    from simllm.preplay import read_routed_experts
    from simllm.traffic import ExpertPlacementSnapshot, RoutedMoeSupply

    routed = read_routed_experts(baseline_cell / "routed-experts.json")
    placement = ExpertPlacementSnapshot(
        placement_epoch=0,
        expert_owners=tuple(
            (layer, expert, expert % 8)
            for layer in range(24)
            for expert in range(32)
        ),
    )
    return RoutedMoeSupply(
        engine_rank=0,
        routed_experts=routed,
        placements=(placement,),
        step_placement_epochs=tuple((step_index, 0) for step_index in step_indices),
    )


def _profile_model(profile: str, launch_count: int) -> Any:
    from simllm.compute import HostInitiationModel

    if profile == "turing-cuda-graph":
        return HostInitiationModel.turing_cuda_graph(launch_count)
    if profile == "turing-eager-host":
        return HostInitiationModel.turing_eager_host(launch_count)
    raise AssertionError(f"unknown host profile: {profile}")


def _model_json(model: Any) -> dict[str, Any]:
    return {
        "profile": model.profile,
        "launch_class": model.launch_class,
        "device_key": model.device_key,
        "device_name": model.device_name,
        "gpu_uuid": model.gpu_uuid,
        "host_cpu": model.host_cpu,
        "driver_version": model.driver_version,
        "cuda_version": model.cuda_version,
        "source_study": model.source_study,
        "launch_count": model.launch_count,
        "point_ps_per_launch": model.initiation_delay_ps,
        "empirical_min_ps_per_launch": model.lower_ps_per_launch,
        "empirical_max_ps_per_launch": model.upper_ps_per_launch,
        "uncertainty_kind": model.uncertainty_kind,
    }


def _run_row(
    args: argparse.Namespace,
    values: dict[str, Any],
    profile: str,
    launch_count: int,
    *,
    cell_name: str,
) -> dict[str, Any]:
    from simllm.adapters.vllm import (
        SimExecutorConfig,
        TranslatedStep,
        VllmBatchSlice,
        build_granite_execution_observations,
    )
    from simllm.adapters.vllm.executor import _SimStepRuntime
    from simllm.backends import (
        HtsimRequestMetricReducer,
        HtsimStepSink,
        HtsimStepSinkConfig,
    )
    from simllm.compute import GPU_ENVELOPES
    from simllm.core import step_records_from_jsonl

    fixture = values["live_fixture"]
    source_records = tuple(
        record
        for record in step_records_from_jsonl(args.baseline_cell / "steps.jsonl")
        if record.step_index in {0, 1, 2}
    )
    if tuple(record.step_index for record in source_records) != (0, 1, 2):
        raise RuntimeError("live fixture does not contain the frozen first three steps")
    if any(
        tuple(request.request_id for request in record.scheduled) != ("r00",)
        for record in source_records
    ):
        raise RuntimeError("live fixture first three steps changed scheduling")
    provider = _fixed_provider(
        dict(zip((19, 20, 21), fixture["fixed_compute_ps"], strict=True))
    )
    host_model = (
        _profile_model(profile, launch_count)
        if profile != "ideal"
        else __import__(
            "simllm.compute", fromlist=["HostInitiationModel"]
        ).HostInitiationModel.ideal()
    )
    supply = _supply(
        args.baseline_cell,
        tuple(record.step_index for record in source_records),
    )
    workdir = args.out / "cells" / cell_name
    dims = _dims()
    ep_ranks = tuple(range(8))
    gpu = GPU_ENVELOPES["gtx1660-ti-sm75"]
    sink = HtsimStepSink(
        HtsimStepSinkConfig(
            profile=fixture["network_profile"],
            tp_ranks=(0,),
            dims=dims,
            workdir=workdir,
            ep_ranks=ep_ranks,
            linkspeed_bps=fixture["linkspeed_bps"],
            provider=provider,
            gpu=gpu,
            host_model=host_model,
            routed_moe_supply=supply,
            num_goal_ranks=8,
        )
    )

    fallback_calls = 0

    def fallback(_translated: Any) -> int:
        nonlocal fallback_calls
        fallback_calls += 1
        raise AssertionError("authoritative packet sink returned no result")

    runtime = _SimStepRuntime(
        config=SimExecutorConfig(),
        step_sink=sink,
        fallback_latency=fallback,
        host_model=host_model,
        gpu=gpu,
    )
    reducer = HtsimRequestMetricReducer({"r00": fixture["arrival_ps"]})
    step_rows = []
    for source in source_records:
        record = replace(source, virtual_time_ps=runtime.clock.now_ps)
        result = runtime.settle(
            TranslatedStep(
                record=record,
                req_ids=["r00"],
                produces_token=[True],
            )
        )
        locality = sink.locality_outcomes[-1]
        metrics = reducer.consume(record, result, locality)
        if len(metrics) != 1:
            raise AssertionError("one-request step did not emit one request metric")
        if (
            locality.compute_service_ps
            + sum(
                composed
                for composed, fabric in zip(
                    locality.composed_phase_service_ps,
                    locality.fabric_phase_service_ps,
                    strict=True,
                )
                if fabric
            )
            != result.step_latency_ps
        ):
            raise AssertionError("step locality partition does not conserve")
        _, observed_timing = build_granite_execution_observations(
            record,
            dims,
            ep_ranks,
            (
                VllmBatchSlice(
                    None,
                    tuple(request.request_id for request in record.scheduled),
                    record.total_new_tokens,
                ),
            ),
            provider,
            gpu,
            host_model,
        )
        outcome = sink.outcomes[-1]
        step_rows.append(
            {
                "step_index": record.step_index,
                "released_at_ps": record.virtual_time_ps,
                "step_latency_ps": result.step_latency_ps,
                "completed_at_ps": result.completed_at_ps,
                "compute_estimate_ps": outcome.compute_estimate_ps,
                "compute_service_ps": locality.compute_service_ps,
                "provider_compute_ps": outcome.provider_compute_ps,
                "host_launch_floor_ps": outcome.host_launch_floor_ps,
                "host_launch_floor_lower_ps": outcome.host_launch_floor_lower_ps,
                "host_launch_floor_upper_ps": outcome.host_launch_floor_upper_ps,
                "observed_schedule_provider_compute_ps": (
                    observed_timing.provider_compute_ps
                ),
                "observed_schedule_represented_compute_ps": (
                    observed_timing.represented_compute_ps
                ),
                "network_service_ps": (
                    result.step_latency_ps - locality.compute_service_ps
                ),
                "total_directed_bytes": locality.total_directed_bytes,
                "num_flows": outcome.num_flows,
                "artifact_operation_ids": [
                    list(operation_ids)
                    for operation_ids in locality.artifact_operation_ids
                ],
            }
        )
    totals = reducer.totals()
    if len(totals) != 1 or totals[0].request_id != "r00":
        raise AssertionError("live request reducer did not emit r00")
    total = totals[0]
    if total.tpot_ps is None:
        raise AssertionError("three-token fixture must produce TPOT")
    if total.ttft_attribution.total_ps != total.ttft_ps:
        raise AssertionError("TTFT attribution does not conserve")
    if total.decode_attribution.total_ps != total.tpot_ps * 2:
        raise AssertionError("TPOT attribution does not conserve")
    return {
        "profile": profile,
        "launch_count": launch_count,
        "host_model": _model_json(host_model),
        "steps": step_rows,
        "ttft_ps": total.ttft_ps,
        "tpot_numerator": total.tpot_ps.numerator,
        "tpot_denominator": total.tpot_ps.denominator,
        "fallback_calls": fallback_calls,
        "runtime_step_count": len(runtime.step_results),
    }


def _mismatch_checks(args: argparse.Namespace) -> dict[str, Any]:
    from simllm.backends import HtsimStepSinkConfig, SerialStepLowererConfig
    from simllm.compute import GPU_ENVELOPES

    rows = []
    for profile in ("turing-cuda-graph", "turing-eager-host"):
        model = _profile_model(profile, 440)
        for device in ("b100", "h100"):
            workdir = args.out / "must-not-exist" / f"{profile}-{device}"
            lowerer_failed = False
            sink_failed = False
            try:
                SerialStepLowererConfig(
                    dims=_dims(),
                    tp_ranks=(0,),
                    gpu=GPU_ENVELOPES[device],
                    host_model=model,
                )
            except ValueError:
                lowerer_failed = True
            try:
                HtsimStepSinkConfig(
                    profile="rnic-nn-fluid",
                    tp_ranks=(0,),
                    dims=_dims(),
                    workdir=workdir,
                    gpu=GPU_ENVELOPES[device],
                    host_model=model,
                )
            except ValueError:
                sink_failed = True
            rows.append(
                {
                    "profile": profile,
                    "device": device,
                    "lowerer_failed": lowerer_failed,
                    "sink_failed": sink_failed,
                    "workdir_absent": not workdir.exists(),
                    "passed": lowerer_failed and sink_failed and not workdir.exists(),
                }
            )
    return {"passed": all(row["passed"] for row in rows), "rows": rows}


def _profile_guard(
    rows: list[dict[str, Any]], calibration: dict[str, Any]
) -> dict[str, Any]:
    failures = []
    for row in rows:
        if row["profile"] == "ideal":
            continue
        expected = calibration["profiles"][row["profile"]]
        observed = row["host_model"]
        checks = {
            "launch_class": expected["launch_class"],
            "point_ps_per_launch": expected["point_ps_per_launch"],
            "empirical_min_ps_per_launch": expected[
                "empirical_min_ps_per_launch"
            ],
            "empirical_max_ps_per_launch": expected[
                "empirical_max_ps_per_launch"
            ],
            "uncertainty_kind": expected["uncertainty_kind"],
        }
        for name, value in checks.items():
            if observed[name] != value:
                failures.append(
                    {
                        "profile": row["profile"],
                        "field": name,
                        "expected": value,
                        "observed": observed[name],
                    }
                )
        device = calibration["device"]
        for name in (
            "device_key",
            "device_name",
            "gpu_uuid",
            "host_cpu",
            "driver_version",
            "cuda_version",
        ):
            if observed[name] != device[name]:
                failures.append(
                    {
                        "profile": row["profile"],
                        "field": name,
                        "expected": device[name],
                        "observed": observed[name],
                    }
                )
    return {"passed": not failures, "failures": failures}


def _physical_checks(
    rows: list[dict[str, Any]],
    values: dict[str, Any],
    calibration: dict[str, Any],
) -> dict[str, Any]:
    fixture = values["live_fixture"]
    network_bounds = fixture["network_physical_bounds_ps"]
    names = ("prefill", "decode_step_1", "decode_step_2")
    result = []
    for row in rows:
        if row["profile"] == "ideal":
            continue
        serialized = calibration["measurements_ps"]["serialized_launch"]
        g = row["host_model"]["point_ps_per_launch"]
        count = row["launch_count"]
        for index, step in enumerate(row["steps"]):
            compute = fixture["fixed_compute_ps"][index]
            network_low, network_high = network_bounds[names[index]]
            lower = network_low + max(compute, count * g)
            upper = network_high + compute + count * serialized
            observed = step["step_latency_ps"]
            result.append(
                {
                    "profile": row["profile"],
                    "launch_count": count,
                    "step_index": index,
                    "floor_ps": lower,
                    "observed_ps": observed,
                    "ceiling_ps": upper,
                    "passed": lower <= observed <= upper,
                }
            )
    return {"passed": all(row["passed"] for row in result), "rows": result}


def _conservation_checks(
    rows: list[dict[str, Any]], values: dict[str, Any]
) -> dict[str, Any]:
    """Check exact component, sequential-clock, and request partitions."""

    fixture = values["live_fixture"]
    ideal = next(row for row in rows if row["profile"] == "ideal")
    details = []
    for row in rows:
        expected_release = fixture["arrival_ps"]
        step_checks = []
        for index, step in enumerate(row["steps"]):
            ideal_step = ideal["steps"][index]
            checks = {
                "release_is_sequential": step["released_at_ps"] == expected_release,
                "completion_conserves": step["completed_at_ps"]
                == step["released_at_ps"] + step["step_latency_ps"],
                "components_conserve": step["step_latency_ps"]
                == step["compute_service_ps"] + step["network_service_ps"],
                "provider_is_frozen_input": step["provider_compute_ps"]
                == fixture["fixed_compute_ps"][index],
                "network_is_host_invariant": step["network_service_ps"]
                == ideal_step["network_service_ps"],
                "bytes_are_host_invariant": step["total_directed_bytes"]
                == ideal_step["total_directed_bytes"],
                "flows_are_host_invariant": step["num_flows"]
                == ideal_step["num_flows"],
                "operation_order_is_host_invariant": step[
                    "artifact_operation_ids"
                ]
                == ideal_step["artifact_operation_ids"],
            }
            step_checks.append(
                {
                    "step_index": step["step_index"],
                    "checks": checks,
                    "passed": all(checks.values()),
                }
            )
            expected_release = step["completed_at_ps"]
        tpot = Fraction(row["tpot_numerator"], row["tpot_denominator"])
        request_checks = {
            "ttft_is_prefill_completion": row["ttft_ps"]
            == row["steps"][0]["completed_at_ps"] - fixture["arrival_ps"],
            "tpot_is_decode_mean": tpot
            == Fraction(
                row["steps"][1]["step_latency_ps"]
                + row["steps"][2]["step_latency_ps"],
                2,
            ),
            "runtime_has_three_steps": row["runtime_step_count"] == 3,
            "fallback_is_unused": row["fallback_calls"] == 0,
        }
        details.append(
            {
                "profile": row["profile"],
                "launch_count": row["launch_count"],
                "step_checks": step_checks,
                "request_checks": request_checks,
                "passed": all(item["passed"] for item in step_checks)
                and all(request_checks.values()),
            }
        )
    return {"passed": all(row["passed"] for row in details), "rows": details}


def _attempt_two_checks(
    rows: list[dict[str, Any]],
    values: dict[str, Any],
    prior: dict[str, Any],
) -> dict[str, Any]:
    """Check the exact live attempt-two repair rows frozen after attempt one."""

    current_by_key = {
        (row["profile"], row["launch_count"]): row for row in rows
    }
    prior_by_key = {
        (row["profile"], row["launch_count"]): row for row in prior["rows"]
    }
    details = []
    for expected in values["live_attempt_two"]["expected_rows"]:
        key = (expected["profile"], expected["launch_count"])
        current = current_by_key[key]
        previous = prior_by_key[key]
        expected_tpot = Fraction(expected["tpot_ps"], 1)
        checks = {
            "service_ps": all(
                step["compute_service_ps"] == expected["service_ps"]
                for step in current["steps"]
            ),
            "step_latency_ps": [
                step["step_latency_ps"] for step in current["steps"]
            ]
            == expected["step_latency_ps"],
            "completion_ps": [
                step["completed_at_ps"] for step in current["steps"]
            ]
            == expected["completion_ps"],
            "release_ps": [step["released_at_ps"] for step in current["steps"]]
            == [0, *expected["completion_ps"][:-1]],
            "ttft_ps": current["ttft_ps"] == expected["ttft_ps"],
            "tpot_ps": Fraction(
                current["tpot_numerator"], current["tpot_denominator"]
            )
            == expected_tpot,
            "service_delta_ps": all(
                step["compute_service_ps"] - old_step["compute_service_ps"]
                == expected["step_delta_ps"]
                for step, old_step in zip(
                    current["steps"], previous["steps"], strict=True
                )
            ),
            "latency_delta_ps": all(
                step["step_latency_ps"] - old_step["step_latency_ps"]
                == expected["step_delta_ps"]
                for step, old_step in zip(
                    current["steps"], previous["steps"], strict=True
                )
            ),
            "completion_delta_ps": all(
                step["completed_at_ps"] - old_step["completed_at_ps"]
                == (index + 1) * expected["step_delta_ps"]
                for index, (step, old_step) in enumerate(
                    zip(current["steps"], previous["steps"], strict=True)
                )
            ),
            "network_unchanged": all(
                step["network_service_ps"] == old_step["network_service_ps"]
                for step, old_step in zip(
                    current["steps"], previous["steps"], strict=True
                )
            ),
            "bytes_unchanged": all(
                step["total_directed_bytes"] == old_step["total_directed_bytes"]
                for step, old_step in zip(
                    current["steps"], previous["steps"], strict=True
                )
            ),
            "flow_count_unchanged": all(
                step["num_flows"] == old_step["num_flows"]
                for step, old_step in zip(
                    current["steps"], previous["steps"], strict=True
                )
            ),
        }
        details.append(
            {
                "profile": expected["profile"],
                "launch_count": expected["launch_count"],
                "checks": checks,
                "passed": all(checks.values()),
            }
        )
    return {"passed": all(row["passed"] for row in details), "rows": details}


def _quantized_compute_checks(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Check that every calibrated GOAL service is the exact enclosure Q."""

    details = []
    for row in rows:
        if row["profile"] == "ideal":
            continue
        for step in row["steps"]:
            floor_ps = max(
                step["provider_compute_ps"], step["host_launch_floor_ps"]
            )
            represented_ps = _whole_nanosecond_enclosure(floor_ps)
            checks = {
                "estimate_equals_floor": step["compute_estimate_ps"] == floor_ps,
                "service_equals_enclosure": step["compute_service_ps"]
                == represented_ps,
                "encloses_floor": floor_ps <= step["compute_service_ps"],
                "is_narrow": step["compute_service_ps"] < floor_ps + 1000,
                "does_not_double_charge": step["compute_service_ps"]
                < step["provider_compute_ps"] + step["host_launch_floor_ps"],
            }
            details.append(
                {
                    "profile": row["profile"],
                    "launch_count": row["launch_count"],
                    "step_index": step["step_index"],
                    "floor_ps": floor_ps,
                    "represented_ps": represented_ps,
                    "checks": checks,
                    "passed": all(checks.values()),
                }
            )
    return {"passed": all(row["passed"] for row in details), "rows": details}


def _observed_schedule_checks(
    rows: list[dict[str, Any]], values: dict[str, Any]
) -> dict[str, Any]:
    """Keep observed-schedule provider attribution separate from service Q."""

    provider_inputs = values["live_attempt_two"]["raw_provider_ps"]
    details = []
    for row in rows:
        if row["profile"] == "ideal":
            continue
        for index, step in enumerate(row["steps"]):
            checks = {
                "provider_is_raw": step["observed_schedule_provider_compute_ps"]
                == provider_inputs[index],
                "represented_matches_serial": step[
                    "observed_schedule_represented_compute_ps"
                ]
                == step["compute_service_ps"],
                "represented_is_enclosure": step[
                    "observed_schedule_represented_compute_ps"
                ]
                == _whole_nanosecond_enclosure(step["compute_estimate_ps"]),
            }
            details.append(
                {
                    "profile": row["profile"],
                    "launch_count": row["launch_count"],
                    "step_index": step["step_index"],
                    "checks": checks,
                    "passed": all(checks.values()),
                }
            )
    return {"passed": all(row["passed"] for row in details), "rows": details}


def _score_rows(
    rows: list[dict[str, Any]], values: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    fixture = values["live_fixture"]
    ideal_prefill = fixture["ideal_step_ps"]["prefill"]
    ideal_decode = fixture["ideal_step_ps"]["decode"]
    ideal_tpot = fixture["ideal_step_ps"]["tpot"]
    band = values["representative_step"]["acceptance_decode_multiplier"]
    scores = []
    for row in rows:
        if row["profile"] == "ideal":
            continue
        decode_multiplier = row["steps"][1]["step_latency_ps"] / ideal_decode[0]
        tpot = Fraction(row["tpot_numerator"], row["tpot_denominator"])
        tpot_multiplier = float(tpot / ideal_tpot)
        ttft_multiplier = row["ttft_ps"] / ideal_prefill
        common = {"profile": row["profile"], "launch_count": row["launch_count"]}
        scores.extend(
            (
                {
                    "id": "LIVE-1_decode_multiplier_in_band",
                    **common,
                    "observed": decode_multiplier,
                    "acceptance": band,
                    "passed": band[0] <= decode_multiplier <= band[1],
                },
                {
                    "id": "LIVE-2_tpot_multiplier_in_band",
                    **common,
                    "observed": tpot_multiplier,
                    "acceptance": band,
                    "passed": band[0] <= tpot_multiplier <= band[1],
                },
                {
                    "id": "LIVE-3_ttft_relative_increase_is_smaller",
                    **common,
                    "ttft_multiplier": ttft_multiplier,
                    "tpot_multiplier": tpot_multiplier,
                    "passed": 1.0 < ttft_multiplier < tpot_multiplier,
                },
            )
        )
    derived_rows = []
    by_key = {(row["profile"], row["launch_count"]): row for row in rows}
    for profile in ("turing-cuda-graph", "turing-eager-host"):
        derived_rows.append(
            {
                "kind": "launch_count",
                "profile": profile,
                "passed": by_key[(profile, 567)]["tpot_numerator"]
                > by_key[(profile, 440)]["tpot_numerator"],
            }
        )
    for count in (440, 567):
        derived_rows.append(
            {
                "kind": "launch_class",
                "launch_count": count,
                "passed": by_key[("turing-eager-host", count)]["tpot_numerator"]
                > by_key[("turing-cuda-graph", count)]["tpot_numerator"],
            }
        )
    return scores, {
        "id": "LIVE-D1_two_parameter_directions",
        "scored": False,
        "passed": all(row["passed"] for row in derived_rows),
        "rows": derived_rows,
    }


def _budget(rows: list[dict[str, Any]], values: dict[str, Any]) -> dict[str, Any]:
    budget = values["mission_budget"]
    point_endpoints = []
    empirical_endpoints = []
    for row in rows:
        if row["profile"] == "ideal":
            continue
        point_floor = (
            row["launch_count"]
            * row["host_model"]["point_ps_per_launch"]
        )
        point_represented = _whole_nanosecond_enclosure(point_floor)
        for real_network in budget["plausible_collective_network_ps"]:
            point_endpoints.append(
                {
                    "profile": row["profile"],
                    "launch_count": row["launch_count"],
                    "per_launch_ps": row["host_model"]["point_ps_per_launch"],
                    "represented_launch_ps": point_represented,
                    "plausible_network_ps": real_network,
                    "ratio": (point_represented + real_network)
                    / (point_represented + budget["modeled_network_ps"]),
                }
            )
        for endpoint_name in (
            "empirical_min_ps_per_launch",
            "empirical_max_ps_per_launch",
        ):
            per_launch = row["host_model"][endpoint_name]
            represented = _whole_nanosecond_enclosure(
                row["launch_count"] * per_launch
            )
            for real_network in budget["plausible_collective_network_ps"]:
                empirical_endpoints.append(
                    {
                        "profile": row["profile"],
                        "launch_count": row["launch_count"],
                        "empirical_endpoint": endpoint_name,
                        "per_launch_ps": per_launch,
                        "represented_launch_ps": represented,
                        "plausible_network_ps": real_network,
                        "ratio": (represented + real_network)
                        / (represented + budget["modeled_network_ps"]),
                    }
                )
    point_observed = [row["ratio"] for row in point_endpoints]
    empirical_observed = [row["ratio"] for row in empirical_endpoints]
    point_range = (min(point_observed), max(point_observed))
    empirical_range = (min(empirical_observed), max(empirical_observed))
    frozen_point = tuple(values["live_attempt_two"]["point_budget_expected"])
    frozen_empirical = tuple(
        values["live_attempt_two"]["empirical_budget_expected"]
    )
    acceptance = budget["acceptance_conditional_turing_residual_optimism"]
    point_matches = all(
        abs(observed - expected) <= 1e-12
        for observed, expected in zip(point_range, frozen_point, strict=True)
    )
    empirical_matches = all(
        abs(observed - expected) <= 1e-12
        for observed, expected in zip(
            empirical_range, frozen_empirical, strict=True
        )
    )
    point_passed = (
        acceptance[0] <= point_range[0] <= point_range[1] <= acceptance[1]
        and point_matches
    )
    empirical_passed = (
        acceptance[0]
        <= empirical_range[0]
        <= empirical_range[1]
        <= acceptance[1]
        and empirical_matches
    )
    return {
        "minimum": point_range[0],
        "maximum": point_range[1],
        "empirical_minimum": empirical_range[0],
        "empirical_maximum": empirical_range[1],
        "acceptance": acceptance,
        "passed": point_passed and empirical_passed,
        "endpoints": point_endpoints,
        "point": {
            "minimum": point_range[0],
            "maximum": point_range[1],
            "frozen_expected": list(frozen_point),
            "matches_frozen": point_matches,
            "passed": point_passed,
            "endpoints": point_endpoints,
        },
        "empirical": {
            "minimum": empirical_range[0],
            "maximum": empirical_range[1],
            "frozen_expected": list(frozen_empirical),
            "matches_frozen": empirical_matches,
            "passed": empirical_passed,
            "endpoints": empirical_endpoints,
        },
        "residual_scheduler_sampler_python_cost": "unknown and omitted",
        "b100_host_step_cost": budget["b100_host_step_cost"],
    }


def _production(
    args: argparse.Namespace,
    values: dict[str, Any],
    calibration: dict[str, Any],
) -> int:
    observed_head = _clean_head()
    source_hash = _sha256(Path(__file__))
    os.environ["SIMLLM_HTSIM_RNIC"] = str(args.htsim_rnic.resolve())
    args.out.mkdir(parents=True)
    rows = [
        _run_row(args, values, "ideal", 0, cell_name="ideal"),
        *(
            _run_row(
                args,
                values,
                profile,
                launch_count,
                cell_name=f"{profile}-n{launch_count}",
            )
            for profile in ("turing-cuda-graph", "turing-eager-host")
            for launch_count in (440, 567)
        ),
    ]
    mismatch = _mismatch_checks(args)
    provenance = _profile_guard(rows, calibration)
    physical = _physical_checks(rows, values, calibration)
    conservation = _conservation_checks(rows, values)
    prior = _load_json(PRIOR_LIVE_ATTEMPT)
    attempt_two = _attempt_two_checks(rows, values, prior)
    quantized_compute = _quantized_compute_checks(rows)
    observed_schedule = _observed_schedule_checks(rows, values)
    scores, derived = _score_rows(rows, values)
    budget = _budget(rows, values)
    fixture = values["live_fixture"]
    ideal = rows[0]
    ideal_exact = (
        [step["step_latency_ps"] for step in ideal["steps"]]
        == [
            fixture["ideal_step_ps"]["prefill"],
            *fixture["ideal_step_ps"]["decode"],
        ]
        and ideal["ttft_ps"] == fixture["ideal_step_ps"]["prefill"]
        and Fraction(ideal["tpot_numerator"], ideal["tpot_denominator"])
        == fixture["ideal_step_ps"]["tpot"]
        and [step["compute_estimate_ps"] for step in ideal["steps"]]
        == fixture["fixed_compute_ps"]
        and [step["compute_service_ps"] for step in ideal["steps"]]
        == fixture["fixed_compute_ps"]
        and [step["provider_compute_ps"] for step in ideal["steps"]]
        == fixture["fixed_compute_ps"]
        and all(step["host_launch_floor_ps"] == 0 for step in ideal["steps"])
        and ideal["fallback_calls"] == 0
        and all(row["fallback_calls"] == 0 for row in rows)
    )
    end_head = _run(("git", "rev-parse", "HEAD")).stdout.strip()
    stable_source = observed_head == end_head and source_hash == _sha256(Path(__file__))
    guards = [
        {
            "id": "LIVE-G1_profile_provenance",
            "passed": provenance["passed"] and stable_source,
            "detail": {
                "profile_check": provenance,
                "observed_head": observed_head,
                "end_head": end_head,
                "source_sha256": source_hash,
            },
        },
        {
            "id": "LIVE-G2_device_mismatch_fails_closed",
            "passed": mismatch["passed"],
            "detail": mismatch,
        },
        {
            "id": "LIVE-G3_step_and_metric_conservation",
            "passed": (
                ideal_exact
                and physical["passed"]
                and conservation["passed"]
                and attempt_two["passed"]
            ),
            "detail": {
                "ideal_exact": ideal_exact,
                "physical": physical,
                "conservation": conservation,
                "attempt_two": attempt_two,
            },
        },
        {
            "id": "LIVE-G4_budget_arithmetic",
            "passed": budget["passed"],
            "detail": budget,
        },
        {
            "id": "LIVE-G5_GOAL_quantized_compute_encloses_host_floor",
            "passed": quantized_compute["passed"],
            "detail": quantized_compute,
        },
        {
            "id": "LIVE-G6_observed_schedule_attribution",
            "passed": observed_schedule["passed"],
            "detail": observed_schedule,
        },
    ]
    fatal_failures = [row["id"] for row in guards if not row["passed"]]
    passed = sum(1 for row in scores if row["passed"])
    if fatal_failures:
        status = "void"
    elif passed != len(scores):
        status = "not_accepted"
    else:
        status = "accepted_pending_offpath"
    result = {
        "schema": "simllm-host-step-cost-v1-results-v1",
        "study": "host_step_cost_v1",
        "task": "COMP-2",
        "live_attempt": values["live_attempt"],
        "expectation_commit": _expectation_commit(),
        "observed_commit": observed_head,
        "calibration_sha256": _sha256(CALIBRATION),
        "run_status": status,
        "behavioral_score_interpretable": not fatal_failures,
        "fatal_guard_failures": fatal_failures,
        "fatal_guards": guards,
        "scored_relations": scores,
        "derived_unscored_checks": [derived],
        "rows": rows,
        "conditional_turing_budget": budget,
        "exact_ideal_offpath": {
            "guard": "OFF-G1_ideal_named_study_exact_identity",
            "status": "pending separate five-cell mission rerun",
        },
    }
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    (args.out / "results.json").write_text(payload, encoding="utf-8", newline="\n")
    RESULTS.write_text(payload, encoding="utf-8", newline="\n")
    if fatal_failures:
        print("run is VOID for closure purposes: fatal guard findings follow")
        for identifier in fatal_failures:
            print(f"  {identifier}")
        print("behavioral score suppressed")
        return 1
    print("all live fatal guards held")
    print(f"scored live relations: {passed} of {len(scores)}")
    if passed != len(scores):
        print("registered live acceptance was not met")
        return 1
    print("live host-step study passed pending the separate ideal compatibility guard")
    return 0


def main() -> int:
    args = _parse_args()
    values, calibration = _check(args)
    if args.check_only:
        print(
            "check-only: live registry, arithmetic, inputs and tools valid; "
            "htsim not invoked; no output created"
        )
        return 0
    return _production(args, values, calibration)


if __name__ == "__main__":
    raise SystemExit(main())
