#!/usr/bin/env python3
"""Run the frozen CORE-62 roofline gate and TRAF-68 network study."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from dataclasses import asdict
from fractions import Fraction
from itertools import pairwise
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from examples.deployment_frontier_v1.frontier import (
    EXPECTATIONS_PATH,
    account_step,
    bottleneck_classification,
    compact_nvlink_ring_service,
    fabric_attribution,
    ideal_network_service,
    intra_node_attribution,
    kernel_service,
    load_expectations,
    operating_point,
    partition_network_bytes,
    sha256_file,
)

EXPECTATIONS_COMMIT = "a7c86086dff5b3039cb84cfe0fa84875404f397d"
EXPECTATIONS_SHA256 = "54295c81cebe36ee32d12b8ab1432c9fc060094ddf98403152b0d619cc37438f"
RESULT_SCHEMA = "simllm-deployment-frontier-result-v1"
BULK_ROOT_ENV = "SIMLLM_CORE62_BULK_ROOT"


def _git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def _require_clean_authority(frozen: dict[str, Any]) -> dict[str, Any]:
    if sha256_file(EXPECTATIONS_PATH) != EXPECTATIONS_SHA256:
        raise SystemExit("the CORE-62 expectations digest moved after its freeze")
    ancestor = _git("merge-base", "--is-ancestor", EXPECTATIONS_COMMIT, "HEAD")
    if ancestor.returncode:
        raise SystemExit("the CORE-62 expectations commit is not an ancestor of HEAD")
    status = _git("status", "--porcelain", "--untracked-files=no")
    if status.returncode or status.stdout.strip():
        raise SystemExit("the CORE-62 gated run requires a clean tracked worktree")
    committed = _git("show", f"{EXPECTATIONS_COMMIT}:examples/deployment_frontier_v1/expectations.json")
    if committed.returncode:
        raise SystemExit("the frozen expectations file cannot be read from its commit")
    import hashlib

    if hashlib.sha256(committed.stdout.encode("utf-8")).hexdigest() != EXPECTATIONS_SHA256:
        raise SystemExit("the expectations commit does not contain the frozen bytes")

    source_checks = [
        (frozen["model_inventory"]["path"], frozen["model_inventory"]["sha256"]),
        (
            frozen["gpu_envelopes"]["source_path"],
            frozen["gpu_envelopes"]["source_sha256"],
        ),
        (
            frozen["gpu_envelopes"]["roofline_source_path"],
            frozen["gpu_envelopes"]["roofline_source_sha256"],
        ),
        (
            frozen["network_inputs"]["fabric"]["source_path"],
            frozen["network_inputs"]["fabric"]["source_sha256"],
        ),
        (
            frozen["network_inputs"]["intra_node"]["profile_path"],
            frozen["network_inputs"]["intra_node"]["profile_sha256"],
        ),
        (
            frozen["network_inputs"]["intra_node"]["implementation_path"],
            frozen["network_inputs"]["intra_node"]["implementation_sha256"],
        ),
        (
            frozen["published_context"]["source_path"],
            frozen["published_context"]["source_sha256"],
        ),
    ]
    for relative, expected in source_checks:
        if sha256_file(REPOSITORY_ROOT / relative) != expected:
            raise SystemExit(f"frozen source digest moved: {relative}")

    lock = frozen["preservation_lock"]
    inherited_path = REPOSITORY_ROOT / lock["inherited"]["path"]
    if sha256_file(inherited_path) != lock["inherited"]["sha256"]:
        raise SystemExit("the inherited preservation authority moved")
    inherited = json.loads(inherited_path.read_text(encoding="utf-8"))[
        "preservation_lock"
    ]["artifacts"]
    if len(inherited) != lock["inherited"]["expected_artifacts"]:
        raise SystemExit("the inherited preservation class changed size")
    artifacts = inherited + lock["additional_artifacts"]
    if len(artifacts) != lock["expected_total_artifacts"]:
        raise SystemExit("the CORE-62 preservation class changed size")
    for artifact in artifacts:
        if sha256_file(REPOSITORY_ROOT / artifact["path"]) != artifact["sha256"]:
            raise SystemExit(f"preservation lock mismatch: {artifact['path']}")

    inventory = frozen["model_inventory"]
    if inventory["flops_per_batch_item"] * inventory["frozen_batch"] != inventory[
        "frozen_flops"
    ]:
        raise SystemExit("the batch-32 FLOP projection does not reconstruct")
    reconstructed_hbm = inventory["static_logical_hbm_bytes"] + (
        inventory["dynamic_hbm_bytes_per_batch_item"] * inventory["frozen_batch"]
    )
    if reconstructed_hbm != inventory["frozen_logical_hbm_bytes"]:
        raise SystemExit("the batch-32 HBM projection does not reconstruct")
    return {
        "expectations_commit": EXPECTATIONS_COMMIT,
        "expectations_sha256": EXPECTATIONS_SHA256,
        "preservation_artifacts_checked": len(artifacts),
        "source_digests_checked": len(source_checks),
    }


def _require_binary(path: Path, expected_sha256: str, name: str) -> Path:
    resolved = path.resolve()
    if not resolved.is_file():
        raise SystemExit(f"{name} is not a file: {resolved.as_posix()}")
    if sha256_file(resolved) != expected_sha256:
        raise SystemExit(f"{name} does not match the frozen binary digest")
    return resolved


def _prepare_run_dir(path: Path) -> Path:
    configured_root = os.environ.get(BULK_ROOT_ENV)
    if not configured_root:
        raise SystemExit(f"set {BULK_ROOT_ENV} to the external CORE-62 bulk root")
    bulk_root = Path(configured_root).resolve()
    resolved = path.resolve()
    if resolved.parent != bulk_root:
        raise SystemExit(f"--run-dir must be one new child of {bulk_root.as_posix()}")
    bulk_root.mkdir(parents=True, exist_ok=True)
    resolved.mkdir(parents=False, exist_ok=False)
    return resolved


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _artifact(path: Path, run_dir: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(run_dir).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _run_fabric_trace(
    *,
    payloads: list[int],
    rank_count: int,
    cell_dir: Path,
    stem: str,
    htsim_rnic: Path,
    txt2bin: Path,
    linkspeed_bps: int,
) -> tuple[int, list[dict[str, Any]], dict[str, Any]]:
    from simllm.backends import HtsimRnicConfig, run_htsim_rnic
    from simllm.goal import GoalTrace, to_binary
    from simllm.traffic import ordered_pairwise_messages

    trace = GoalTrace(rank_count)
    messages = [
        (f"flow-{index}", index + 1, 0, payload)
        for index, payload in enumerate(payloads)
    ]
    ordered_pairwise_messages(
        trace,
        ranks=list(range(rank_count)),
        messages=messages,
        tag=62_068,
        operation_id=f"core62:{stem}",
    )
    if [message.payload_bytes for message in trace.messages] != payloads:
        raise AssertionError("GOAL structured payload ledger differs from the frozen split")
    goal_path = trace.write(cell_dir / f"{stem}.goal")
    bin_path = to_binary(goal_path, tool=txt2bin)
    csv_path = cell_dir / f"{stem}-completion.csv"
    run = run_htsim_rnic(
        HtsimRnicConfig(
            goal_bin=bin_path,
            profile="rnic-nn",
            linkspeed_bps=linkspeed_bps,
            completion_csv=csv_path,
        ),
        binary=htsim_rnic,
    )
    observed_payloads = sorted(flow.payload_bytes for flow in run.flows)
    if observed_payloads != sorted(payloads):
        raise AssertionError("htsim completion payloads differ from the GOAL ledger")
    if not run.quiescent:
        raise AssertionError("htsim did not report physical quiescence")
    flow_rows = [
        {
            "source": flow.source,
            "destination": flow.destination,
            "payload_bytes": flow.payload_bytes,
            "start_time_ps": flow.start_time_ps,
            "completion_time_ps": flow.completion_time_ps,
            "fct_ps": flow.fct_ps,
        }
        for flow in run.flows
    ]
    artifacts = {
        "goal": goal_path,
        "binary_goal": bin_path,
        "completion_csv": csv_path,
    }
    return run.job_completion_time_ps(), flow_rows, artifacts


def _fabric_observation(
    *,
    payloads: list[int],
    cell_dir: Path,
    htsim_rnic: Path,
    txt2bin: Path,
    linkspeed_bps: int,
) -> tuple[dict[str, Any], dict[str, Path]]:
    if not payloads:
        return (
            {
                "concurrent_service_ps": 0,
                "isolated_service_ps": 0,
                "concurrent_flows": [],
                "isolated_flows": [],
                "isolated_trace_reused_concurrent": True,
                "physical_quiescence": True,
            },
            {},
        )
    rank_count = len(payloads) + 1
    concurrent, concurrent_flows, concurrent_artifacts = _run_fabric_trace(
        payloads=payloads,
        rank_count=rank_count,
        cell_dir=cell_dir,
        stem="concurrent",
        htsim_rnic=htsim_rnic,
        txt2bin=txt2bin,
        linkspeed_bps=linkspeed_bps,
    )
    if len(payloads) == 1:
        isolated = concurrent
        isolated_flows = concurrent_flows
        isolated_artifacts: dict[str, Path] = {}
        reused = True
    else:
        isolated_payload = max(payloads)
        isolated, isolated_flows, isolated_artifacts = _run_fabric_trace(
            payloads=[isolated_payload],
            rank_count=rank_count,
            cell_dir=cell_dir,
            stem="isolated",
            htsim_rnic=htsim_rnic,
            txt2bin=txt2bin,
            linkspeed_bps=linkspeed_bps,
        )
        reused = False
    artifacts = {
        **{f"concurrent_{name}": path for name, path in concurrent_artifacts.items()},
        **{f"isolated_{name}": path for name, path in isolated_artifacts.items()},
    }
    return (
        {
            "concurrent_service_ps": concurrent,
            "isolated_service_ps": isolated,
            "concurrent_flows": concurrent_flows,
            "isolated_flows": isolated_flows,
            "isolated_trace_reused_concurrent": reused,
            "physical_quiescence": True,
        },
        artifacts,
    )


def _published_guard(frozen: dict[str, Any], points: list[dict[str, Any]]) -> dict[str, Any]:
    comparable = frozen["published_context"]["paired"][0]
    matching = next(
        point
        for point in points
        if point["configuration_id"] == "h100-nine-node-incast"
        and point["batch_per_gpu"] == comparable["batch_per_node"] // comparable["gpus_per_node"]
    )
    analytical = matching["analytical_operating_point"]
    published_x = Fraction(
        comparable["tokens_per_second_per_node"],
        comparable["batch_per_node"],
    )
    published_y = Fraction(
        comparable["tokens_per_second_per_node"],
        comparable["gpus_per_node"],
    )
    analytical_x = Fraction(
        analytical["x_tokens_per_second_per_request"]["numerator"],
        analytical["x_tokens_per_second_per_request"]["denominator"],
    )
    analytical_y = Fraction(
        analytical["y_tokens_per_second_per_gpu"]["numerator"],
        analytical["y_tokens_per_second_per_gpu"]["denominator"],
    )
    return {
        "published_id": comparable["id"],
        "comparison_configuration_id": matching["configuration_id"],
        "comparison_batch_per_gpu": matching["batch_per_gpu"],
        "published_x": {"numerator": published_x.numerator, "denominator": published_x.denominator},
        "published_y": {"numerator": published_y.numerator, "denominator": published_y.denominator},
        "published_on_or_below_analytical": (
            published_x <= analytical_x and published_y <= analytical_y
        ),
    }


def _direction_checks(points: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_configuration: dict[str, list[dict[str, Any]]] = {}
    for point in points:
        by_configuration.setdefault(point["configuration_id"], []).append(point)
    speeds_nonincreasing = True
    throughput_nondecreasing = True
    for rows in by_configuration.values():
        rows.sort(key=lambda row: row["batch_per_gpu"])
        speeds = [
            row["analytical_operating_point"]["x_tokens_per_second_per_request"]["decimal"]
            for row in rows
        ]
        throughput = [
            row["analytical_operating_point"]["y_tokens_per_second_per_gpu"]["decimal"]
            for row in rows
        ]
        speeds_nonincreasing &= all(left >= right for left, right in pairwise(speeds))
        throughput_nondecreasing &= all(
            left <= right for left, right in pairwise(throughput)
        )
    return [
        {
            "name": "per-request speed is nonincreasing with batch",
            "passed": speeds_nonincreasing,
        },
        {
            "name": "aggregate per-GPU throughput is nondecreasing with batch",
            "passed": throughput_nondecreasing,
        },
        {
            "name": "nine-node incast has positive elapsed inter-node attribution",
            "passed": any(
                point["configuration_id"] == "h100-nine-node-incast"
                and point["accounting"]["inter_node_attributed_ps"] > 0
                for point in points
            ),
        },
        {
            "name": "pass-through candidate switch attribution is zero",
            "passed": all(
                point["intra_node_attribution"]["switch_contention_ps"] == 0
                for point in points
            ),
        },
        {
            "name": "at least one point is roofline-bound with neither network material",
            "passed": any(
                point["bottleneck"]["classification"] == "neither"
                and point["accounting"]["inter_node_attributed_ps"] == 0
                and point["accounting"]["intra_node_attributed_ps"] == 0
                for point in points
            ),
        },
    ]


def _point(
    *,
    frozen: dict[str, Any],
    configuration: dict[str, Any],
    batch_per_gpu: int,
    cell_dir: Path,
    run_dir: Path,
    htsim_rnic: Path,
    txt2bin: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    from simllm.backends.htsim_nvlink import load_nvlink_candidate_profile

    kernel = kernel_service(
        frozen,
        gpu_name=configuration["gpu"],
        batch_per_gpu=batch_per_gpu,
    )
    partition = partition_network_bytes(
        frozen,
        configuration=configuration,
        batch_per_gpu=batch_per_gpu,
    )
    ideal = ideal_network_service(frozen, partition)
    fabric, artifact_paths = _fabric_observation(
        payloads=partition["remote_flow_payload_bytes"],
        cell_dir=cell_dir,
        htsim_rnic=htsim_rnic,
        txt2bin=txt2bin,
        linkspeed_bps=frozen["network_inputs"]["fabric"][
            "nominal_link_rate_bits_per_second"
        ],
    )
    profile = load_nvlink_candidate_profile(
        REPOSITORY_ROOT / frozen["network_inputs"]["intra_node"]["profile_path"]
    )
    compact = compact_nvlink_ring_service(
        profile,
        payload_bytes_per_transfer=partition["local_logical_bytes_per_transfer"],
        transfer_count=configuration["intra_transfer_count"],
    )
    if compact.logical_bytes != (
        partition["local_logical_bytes_per_transfer"]
        * configuration["intra_transfer_count"]
    ):
        raise AssertionError("candidate logical-byte ledger does not conserve")
    if compact.request_wire_bytes < compact.request_payload_bytes:
        raise AssertionError("candidate wire-byte ledger is smaller than its payload")

    fabric_terms = fabric_attribution(
        ideal_wire_ps=ideal["ideal_fabric_wire_ps"],
        isolated_service_ps=fabric["isolated_service_ps"],
        concurrent_service_ps=fabric["concurrent_service_ps"],
    )
    intra_terms = intra_node_attribution(
        ideal_wire_ps=ideal["ideal_intra_node_wire_ps"],
        result=compact,
    )
    if fabric["concurrent_service_ps"] < ideal["ideal_fabric_wire_ps"]:
        raise AssertionError("simulated fabric service is below the ideal wire floor")
    if compact.completion_time_ps < ideal["ideal_intra_node_wire_ps"]:
        raise AssertionError("candidate intra-node service is below the ideal wire floor")
    accounting = account_step(
        kernel_floor_ps=kernel["kernel_floor_ps"],
        ideal_fabric_wire_ps=ideal["ideal_fabric_wire_ps"],
        ideal_intra_node_wire_ps=ideal["ideal_intra_node_wire_ps"],
        simulated_fabric_ps=fabric["concurrent_service_ps"],
        simulated_intra_node_ps=compact.completion_time_ps,
    )
    bottleneck = bottleneck_classification(
        kernel_floor_ps=kernel["kernel_floor_ps"],
        simulated_fabric_ps=fabric["concurrent_service_ps"],
        simulated_intra_node_ps=compact.completion_time_ps,
    )
    point = {
        "configuration_id": configuration["id"],
        "configuration_label": configuration["label"],
        "gpu": configuration["gpu"],
        "gpus_per_node": configuration["gpus_per_node"],
        "node_count": configuration["node_count"],
        "batch_per_gpu": batch_per_gpu,
        "kernel": kernel,
        "byte_partition": partition,
        "ideal_network": ideal,
        "fabric_observation": fabric,
        "fabric_attribution": fabric_terms,
        "intra_node_candidate": asdict(compact),
        "intra_node_attribution": intra_terms,
        "accounting": accounting,
        "bottleneck": bottleneck,
        "analytical_operating_point": operating_point(
            batch_per_gpu=batch_per_gpu,
            step_time_ps=accounting["analytical_step_ps"],
        ),
        "simulated_operating_point": operating_point(
            batch_per_gpu=batch_per_gpu,
            step_time_ps=accounting["simulated_step_ps"],
        ),
        "intra_node_evidence": {
            "status": frozen["network_inputs"]["intra_node"]["status"],
            "profile_id": profile.profile_id,
            "cross_architecture_use": frozen["network_inputs"]["intra_node"][
                "cross_architecture_use"
            ],
            "domain_mapping": "two independent four-endpoint candidate domains for eight transfers",
        },
        "artifacts": {
            name: _artifact(path, run_dir) for name, path in artifact_paths.items()
        },
    }
    return point, list(point["artifacts"].values())


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Execute every frozen point and return the compact result."""

    frozen = load_expectations()
    authority = _require_clean_authority(frozen)
    fabric_freeze = frozen["network_inputs"]["fabric"]
    htsim_rnic = _require_binary(
        args.htsim_rnic,
        fabric_freeze["accepted_htsim_rnic_sha256"],
        "htsim_rnic",
    )
    txt2bin = _require_binary(
        args.txt2bin,
        fabric_freeze["accepted_txt2bin_sha256"],
        "txt2bin",
    )
    run_dir = _prepare_run_dir(args.run_dir)
    points = []
    bulk_artifacts = []
    for configuration in frozen["configurations"]:
        for batch_per_gpu in frozen["batch_per_gpu_sweep"]:
            cell_dir = run_dir / "cells" / configuration["id"] / f"batch-{batch_per_gpu}"
            cell_dir.mkdir(parents=True, exist_ok=False)
            point, artifacts = _point(
                frozen=frozen,
                configuration=configuration,
                batch_per_gpu=batch_per_gpu,
                cell_dir=cell_dir,
                run_dir=run_dir,
                htsim_rnic=htsim_rnic,
                txt2bin=txt2bin,
            )
            points.append(point)
            bulk_artifacts.extend(artifacts)
            print(
                f"{configuration['id']} B={batch_per_gpu}: "
                f"residual={point['accounting']['residual_ps']} ps, "
                f"binds={point['bottleneck']['classification']}"
            )

    if len(points) != len(frozen["configurations"]) * len(
        frozen["batch_per_gpu_sweep"]
    ):
        raise AssertionError("the run did not produce every frozen point")
    published_guard = _published_guard(frozen, points)
    directions = _direction_checks(points)
    residuals_zero = all(point["accounting"]["residual_ps"] == 0 for point in points)
    attributions_nonnegative = all(
        point["accounting"]["inter_node_attributed_ps"] >= 0
        and point["accounting"]["intra_node_attributed_ps"] >= 0
        for point in points
    )
    fatal_guards = {
        "all_residuals_zero": residuals_zero,
        "attributions_nonnegative": attributions_nonnegative,
        "published_comparable_point_on_or_below_analytical": published_guard[
            "published_on_or_below_analytical"
        ],
        "all_coordinates_positive": all(
            point[kind][axis]["decimal"] > 0
            for point in points
            for kind in ("analytical_operating_point", "simulated_operating_point")
            for axis in (
                "x_tokens_per_second_per_request",
                "y_tokens_per_second_per_gpu",
            )
        ),
    }
    direction_pass = all(check["passed"] for check in directions)
    status = "PASS" if all(fatal_guards.values()) and direction_pass else "REFUTED"
    result = {
        "schema": RESULT_SCHEMA,
        "status": status,
        "verdict": (
            "All 18 residuals are exactly zero after the two frozen network terms."
            if residuals_zero
            else "At least one residual is nonzero and remains unexplained."
        ),
        "scope": (
            "Roofline replay of the declared disaggregated-session decode step; "
            "this is not a live SGLang frontend run."
        ),
        "plot_contract": frozen["plot_contract"],
        "published_context": frozen["published_context"],
        "accounting_schema": frozen["accounting_identity"]["schema"],
        "provenance": {
            **authority,
            "head_commit": _git("rev-parse", "HEAD").stdout.strip(),
            "htsim_rnic": {"filename": htsim_rnic.name, "sha256": sha256_file(htsim_rnic)},
            "txt2bin": {"filename": txt2bin.name, "sha256": sha256_file(txt2bin)},
            "kernel_provider": "RooflineProvider",
            "kernel_simulation_enabled": False,
            "roofline_efficiency": frozen["gpu_envelopes"]["efficiency"],
            "bulk_run_id": run_dir.name,
        },
        "intra_node_candidate_disclosure": frozen["network_inputs"]["intra_node"][
            "cross_architecture_use"
        ],
        "points": points,
        "published_guard": published_guard,
        "fatal_guards": fatal_guards,
        "expected_direction_checks": directions,
        "bulk_artifacts": bulk_artifacts,
        "preservation_lock": {
            "class": frozen["preservation_lock"]["class"],
            "artifacts_checked": authority["preservation_artifacts_checked"],
            "all_byte_identical": True,
        },
        "reserved_residual_ids": frozen["study"]["reserved_residual_ids"],
    }
    _write_json(run_dir / "raw-result.json", result)
    _write_csv(run_dir / "points.csv", points)
    return result


def _write_csv(path: Path, points: list[dict[str, Any]]) -> None:
    fieldnames = [
        "configuration_id",
        "batch_per_gpu",
        "analytical_step_ps",
        "simulated_step_ps",
        "inter_node_attributed_ps",
        "intra_node_attributed_ps",
        "residual_ps",
        "fabric_raw_excess_ps",
        "intra_node_raw_excess_ps",
        "fabric_mechanism",
        "intra_node_module",
        "bottleneck",
        "analytical_x_tokens_per_second_per_request",
        "analytical_y_tokens_per_second_per_gpu",
        "simulated_x_tokens_per_second_per_request",
        "simulated_y_tokens_per_second_per_gpu",
    ]
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for point in points:
            accounting = point["accounting"]
            writer.writerow(
                {
                    "configuration_id": point["configuration_id"],
                    "batch_per_gpu": point["batch_per_gpu"],
                    "analytical_step_ps": accounting["analytical_step_ps"],
                    "simulated_step_ps": accounting["simulated_step_ps"],
                    "inter_node_attributed_ps": accounting["inter_node_attributed_ps"],
                    "intra_node_attributed_ps": accounting["intra_node_attributed_ps"],
                    "residual_ps": accounting["residual_ps"],
                    "fabric_raw_excess_ps": point["fabric_attribution"]["raw_excess_ps"],
                    "intra_node_raw_excess_ps": point["intra_node_attribution"][
                        "raw_excess_ps"
                    ],
                    "fabric_mechanism": point["fabric_attribution"][
                        "dominant_mechanism"
                    ],
                    "intra_node_module": point["intra_node_attribution"][
                        "dominant_module"
                    ],
                    "bottleneck": point["bottleneck"]["classification"],
                    "analytical_x_tokens_per_second_per_request": point[
                        "analytical_operating_point"
                    ]["x_tokens_per_second_per_request"]["decimal"],
                    "analytical_y_tokens_per_second_per_gpu": point[
                        "analytical_operating_point"
                    ]["y_tokens_per_second_per_gpu"]["decimal"],
                    "simulated_x_tokens_per_second_per_request": point[
                        "simulated_operating_point"
                    ]["x_tokens_per_second_per_request"]["decimal"],
                    "simulated_y_tokens_per_second_per_gpu": point[
                        "simulated_operating_point"
                    ]["y_tokens_per_second_per_gpu"]["decimal"],
                }
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--htsim-rnic", type=Path, required=True)
    parser.add_argument("--txt2bin", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    result = run(parse_args())
    print(f"CORE-62/TRAF-68 {result['status']}")


if __name__ == "__main__":
    main()
