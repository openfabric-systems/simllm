"""Run the frozen vLLM observed-schedule study."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

CAPTURE_SHA256 = "5f55fccc265ee5519430a0f73d6631e49aa547ec07fcb81034e9fc2b4d9fead6"
CAPTURE_ROWS = 120
REPLAY_RUN_SHA256 = "b4d38a09011caf6de159c22133264d62a2727063496953f4337b17d79cfde93e"
ROUTED_EXPERTS_SHA256 = (
    "24e986e989e21f1bfe7e758d4470928c82c3bbaec06072a839743b9b17d7cf5f"
)
VLLM_VERSION = "0.26.0"
NUM_LAYERS = 24
NUM_MICROBATCHES = 2
SEMANTIC_MOE_SITES = 48
DBO_MOE_INVOCATIONS = 96
EXPECTED_GENUINE_RISK_INSTANCES = 4
DP_SIZE = 8
TP_SIZE = 1
PP_SIZE = 1
DBO_DECODE_THRESHOLD = 2
DBO_PREFILL_THRESHOLD = 512
SERIAL_GRAPH_BYTES = 4_127
SERIAL_GRAPH_SHA256 = (
    "aa3c836fe559973a7bf0940384c2e8a84e6af84e0fbd2c02d3b89774ee0c8e2d"
)
SERIAL_GOAL_BYTES = 1_880
SERIAL_GOAL_SHA256 = (
    "7087db6780f7e34f5a559a6505eeccc15d984c7b478cd8f0bc5838053825d4b6"
)

REDUCTION_BANDS_PS = {
    "single-node": (1_000_000, 5_000_000),
    "cross-node": (20_000_000, 130_000_000),
}
PERTURBATION_MIN_PS = {
    "single-node": 100_000,
    "cross-node": 5_000_000,
}

SOURCE_HASHES = {
    "model_executor/models/granitemoe.py": (
        "b60e452c3f28b25aa104c88869daa25c06a7fb6ed45bd34e908fa6a8395efda1"
    ),
    "config/parallel.py": (
        "a6581c267ab265e24905d2f5caa514482c28359f71380c6f894ceab25aa22541"
    ),
    "v1/worker/gpu_model_runner.py": (
        "81b7627fbe81f7aaa2f77b4bf085faa353c69d03662ebfe369536a9773bb70d0"
    ),
    "v1/worker/dp_utils.py": (
        "2ba84bbf92a25e756576918bfb215c1fb387b006899885d811bdb2f774e843a9"
    ),
    "v1/worker/ubatch_utils.py": (
        "0b727aaa1c7072152e25f684ddc2fc9790c430eddd862e610c97a8e3e9febdc4"
    ),
    "v1/worker/gpu_ubatch_wrapper.py": (
        "4eae50c929f3ba873072c13291c7140be3dd00d4a5b623170dff44754519c021"
    ),
    "v1/worker/ubatching.py": (
        "40391241c564feb5f16c77898ae6ae152ed6e71a4682e2a406387785d8de02d7"
    ),
    "model_executor/layers/fused_moe/modular_kernel.py": (
        "f78ae626babfd69f3c6ba37eef9c8f5186f28cd9064f566e341ca0c9e0fdb9b9"
    ),
    "model_executor/layers/fused_moe/prepare_finalize/deepep_ht.py": (
        "465cdf1d6cee91b2ee8c2e43abbea6e8408976e3048c10f44c089f34b415bc60"
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _line_count(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for _ in handle)


def _check_file(path: Path, expected_sha256: str, label: str) -> None:
    if not path.is_file():
        raise SystemExit(f"{label} is not a file: {path}")
    if _sha256(path) != expected_sha256:
        raise SystemExit(f"{label} SHA-256 does not match the frozen input")


def check_expectation_registry(args: argparse.Namespace) -> None:
    """Validate frozen inputs and literals without importing target code."""

    _check_file(args.capture, CAPTURE_SHA256, "capture")
    if _line_count(args.capture) != CAPTURE_ROWS:
        raise SystemExit("capture row count does not match the frozen input")
    _check_file(args.replay_run, REPLAY_RUN_SHA256, "replay run")
    _check_file(args.routed_experts, ROUTED_EXPERTS_SHA256, "routed experts")

    if not args.vllm_source.is_dir():
        raise SystemExit(f"vLLM source is not a directory: {args.vllm_source}")
    for relative, expected in SOURCE_HASHES.items():
        _check_file(args.vllm_source / relative, expected, f"vLLM source {relative}")

    if not args.vllm_python.is_file():
        raise SystemExit(f"vLLM Python is not a file: {args.vllm_python}")
    observed_version = subprocess.run(
        [str(args.vllm_python), "-c", "import vllm; print(vllm.__version__)"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if observed_version != VLLM_VERSION:
        raise SystemExit(
            f"vLLM version must be {VLLM_VERSION}, got {observed_version!r}"
        )

    if args.output_dir.exists():
        raise SystemExit(f"output directory already exists: {args.output_dir}")

    assert NUM_LAYERS == 24
    assert NUM_MICROBATCHES == 2
    assert SEMANTIC_MOE_SITES == 2 * NUM_LAYERS
    assert DBO_MOE_INVOCATIONS == SEMANTIC_MOE_SITES * NUM_MICROBATCHES
    assert (TP_SIZE, DP_SIZE, PP_SIZE) == (1, 8, 1)
    assert DBO_DECODE_THRESHOLD == 2
    assert DBO_PREFILL_THRESHOLD > 54
    assert EXPECTED_GENUINE_RISK_INSTANCES == 2 * len(REDUCTION_BANDS_PS)
    assert set(REDUCTION_BANDS_PS) == set(PERTURBATION_MIN_PS)
    for low, high in REDUCTION_BANDS_PS.values():
        assert 0 < low <= high
    for placement, minimum in PERTURBATION_MIN_PS.items():
        assert 0 < minimum <= REDUCTION_BANDS_PS[placement][1]
    assert REDUCTION_BANDS_PS["cross-node"][0] >= (
        5 * REDUCTION_BANDS_PS["single-node"][0]
    )
    assert SERIAL_GRAPH_BYTES > SERIAL_GOAL_BYTES > 0
    assert len(SERIAL_GRAPH_SHA256) == len(SERIAL_GOAL_SHA256) == 64


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--replay-run", type=Path, required=True)
    parser.add_argument("--routed-experts", type=Path, required=True)
    parser.add_argument("--vllm-source", type=Path, required=True)
    parser.add_argument("--vllm-python", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args()


def _canonical_identity(value: object) -> dict[str, object]:
    wire = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return {
        "bytes": len(wire),
        "sha256": hashlib.sha256(wire).hexdigest(),
    }


def _fraction_to_json(value: Fraction | None) -> dict[str, int | float] | None:
    if value is None:
        return None
    return {
        "denominator": value.denominator,
        "float_ps": float(value),
        "numerator": value.numerator,
    }


def _fraction_from_json(value: dict[str, object]) -> Fraction:
    return Fraction(int(value["numerator"]), int(value["denominator"]))


def _model_dims():
    from simllm.compute import ModelDims

    return ModelDims(
        num_layers=NUM_LAYERS,
        hidden_size=1_024,
        intermediate_size=512,
        num_heads=16,
        num_kv_heads=8,
        head_size=64,
        vocab_size=49_155,
        dtype_bytes=2,
        num_experts=32,
        top_k=8,
        moe_intermediate_size=512,
        local_num_experts=4,
    )


def _routed_supply(path: Path):
    from simllm.preplay import read_routed_experts
    from simllm.traffic import ExpertPlacementSnapshot, RoutedMoeSupply

    placement = ExpertPlacementSnapshot(
        placement_epoch=0,
        expert_owners=tuple(
            (layer, expert, expert % DP_SIZE)
            for layer in range(NUM_LAYERS)
            for expert in range(32)
        ),
    )
    return RoutedMoeSupply(
        placements=(placement,),
        step_placement_epochs=tuple((step, 0) for step in range(4_096)),
        routed_experts=read_routed_experts(path),
    )


def _profile(placement: str):
    from simllm.core import CoarseDeviceProfile

    if placement == "single-node":
        return CoarseDeviceProfile(nvlink_rate_bps=3_600_000_000_000)
    if placement == "cross-node":
        return CoarseDeviceProfile(
            gpus_per_node=1,
            rnic_rate_bps=400_000_000_000,
        )
    raise ValueError(f"unknown placement {placement!r}")


def _observations_from_source(source: object):
    from simllm.core import ExecutionObservations, execution_graph_from_json

    if source is None:
        return None
    graph = execution_graph_from_json(source)
    return ExecutionObservations(
        operations=graph.operations,
        completion_operation_ids=graph.completion_operation_ids,
    )


def _perturb_observations(observations, step_index: int):
    from simllm.core import ExecutionObservations

    if observations is None:
        return None, 0
    predecessor = f"step-{step_index}:ubatch-0:layer-12:ep-combine"
    target = f"step-{step_index}:ubatch-1:layer-12:rank-0:experts"
    operation_ids = {operation.operation_id for operation in observations.operations}
    if target not in operation_ids:
        return observations, 0
    if predecessor not in operation_ids:
        raise RuntimeError("dependency perturbation predecessor is absent")
    operations = []
    changed = 0
    for operation in observations.operations:
        if operation.operation_id == target:
            if predecessor in operation.depends_on:
                raise RuntimeError("dependency perturbation edge already exists")
            operation = replace(
                operation,
                depends_on=(*operation.depends_on, predecessor),
            )
            changed += 1
        operations.append(operation)
    if changed != 1:
        raise RuntimeError("dependency perturbation did not change exactly one operation")
    return (
        ExecutionObservations(
            operations=tuple(operations),
            completion_operation_ids=observations.completion_operation_ids,
        ),
        changed,
    )


def _schedule_fields_match(source, lowered) -> bool:
    from simllm.core import CollectiveWork, ComputeWork

    if source.completion_operation_ids != lowered.completion_operation_ids:
        return False
    if len(source.operations) != len(lowered.operations):
        return False
    for observed, bound in zip(source.operations, lowered.operations):
        if (
            observed.operation_id != bound.operation_id
            or observed.rank != bound.rank
            or observed.logical_queue != bound.logical_queue
            or observed.depends_on != bound.depends_on
            or observed.participant_local_depends_on
            != bound.participant_local_depends_on
            or observed.not_before_ps != bound.not_before_ps
            or observed.priority != bound.priority
            or observed.correlation != bound.correlation
        ):
            return False
        if isinstance(observed.work, ComputeWork):
            if observed.work != bound.work:
                return False
        elif isinstance(observed.work, CollectiveWork):
            if not isinstance(bound.work, CollectiveWork):
                return False
            if (
                observed.work.collective != bound.work.collective
                or observed.work.ranks != bound.work.ranks
                or observed.work.channel_hint != bound.work.channel_hint
            ):
                return False
        else:
            return False
    return True


def _request_pair_inventory(graph) -> tuple[tuple[object, ...], ...]:
    from simllm.core import CollectiveWork

    return tuple(
        sorted(
            (
                operation.correlation.layer,
                operation.work.channel_hint,
                request_id,
                source,
                destination,
                size,
            )
            for operation in graph.operations
            if isinstance(operation.work, CollectiveWork)
            for request_id, source, destination, size in (
                operation.work.request_pair_payload_bytes
            )
        )
    )


def _mode_run(
    placement: str,
    mode: str,
    live_path: str,
    routed_experts: str,
    raw_output: str,
) -> dict[str, object]:
    from simllm.backends import (
        DeviceRuntimeStepSink,
        SerialStepLowerer,
        SerialStepLowererConfig,
    )
    from simllm.compute import RooflineProvider
    from simllm.core import (
        CoarseDeviceRuntime,
        CompletionReducer,
        VirtualClock,
        execution_graph_from_json,
        execution_graph_to_json,
        execution_result_to_json,
        step_record_from_json,
        step_result_to_json,
    )
    from simllm.traffic import project_execution_graph_goal, render_step_goal

    profile = _profile(placement)
    config = SerialStepLowererConfig(
        _model_dims(),
        (0,),
        ep_ranks=tuple(range(DP_SIZE)),
        provider=RooflineProvider(efficiency=0.7),
        routed_moe_supply=_routed_supply(Path(routed_experts)),
    )
    clock = VirtualClock()
    sink = DeviceRuntimeStepSink(
        config,
        runtime=CoarseDeviceRuntime(profile),
    )
    sink.bind_clock(clock)
    direct_clock = VirtualClock()
    direct_reducer = CompletionReducer(direct_clock)
    direct_runtime = CoarseDeviceRuntime(profile)
    direct_lowerer = SerialStepLowerer(config)
    raw_rows = []
    last_r0 = None
    perturbation_edges = 0
    schedule_checks: list[bool] = []
    serial_checks: list[bool] = []
    live_checks: list[bool] = []
    serial_goal_identities = []
    serial_projection_identities = []
    request_pair_step_identities = []
    request_pair_rows = 0
    request_pair_bytes = 0

    with Path(live_path).open("r", encoding="utf-8") as stream:
        live_rows = [json.loads(line) for line in stream if line.strip()]
    for live_row in live_rows:
        source_record = step_record_from_json(live_row["record"])
        record = replace(source_record, virtual_time_ps=clock.now_ps)
        source_observations = _observations_from_source(live_row["source_graph"])
        observations = source_observations
        if mode == "serial":
            observations = None
        elif mode == "perturbed":
            observations, changed = _perturb_observations(
                source_observations,
                record.step_index,
            )
            perturbation_edges += changed
        elif mode != "observed":
            raise ValueError(f"unknown execution mode {mode!r}")

        step_result = sink(record, observations)
        outcome = sink.outcomes[-1]
        graph_json = execution_graph_to_json(outcome.graph)
        execution_json = execution_result_to_json(outcome.execution_result)
        step_json = step_result_to_json(step_result)
        graph_identity = _canonical_identity(graph_json)
        execution_identity = _canonical_identity(execution_json)
        pair_inventory = _request_pair_inventory(outcome.graph)
        request_pair_step_identities.append(_canonical_identity(pair_inventory))
        request_pair_rows += len(pair_inventory)
        request_pair_bytes += sum(int(row[-1]) for row in pair_inventory)

        if observations is not None:
            observed_graph = execution_graph_from_json(
                {
                    **live_row["source_graph"],
                    "execution_id": f"step-{record.step_index}",
                    "released_at_ps": record.virtual_time_ps,
                }
            )
            if mode == "perturbed":
                observed_graph = replace(
                    observed_graph,
                    operations=observations.operations,
                    completion_operation_ids=observations.completion_operation_ids,
                )
            schedule_checks.append(_schedule_fields_match(observed_graph, outcome.graph))

        if mode == "serial":
            if direct_clock.now_ps != record.virtual_time_ps:
                raise RuntimeError("serial direct clock drifted before comparison")
            direct_graph = direct_lowerer.lower(record)
            direct_execution = direct_runtime.execute(direct_graph)
            direct_step = direct_reducer.reduce(
                record,
                direct_graph,
                direct_execution,
                direct_runtime.last_report,
            )
            direct_graph_json = execution_graph_to_json(direct_graph)
            direct_execution_json = execution_result_to_json(direct_execution)
            direct_step_json = step_result_to_json(direct_step)
            projection = project_execution_graph_goal(direct_graph)
            serial_projection_identities.append(
                {
                    "artifacts": [
                        {
                            "bytes": len(artifact.trace.render().encode()),
                            "sha256": hashlib.sha256(
                                artifact.trace.render().encode()
                            ).hexdigest(),
                        }
                        for artifact in projection.artifacts
                    ],
                    "boundaries": len(projection.boundaries),
                    "serialized_edges": len(projection.serialized_edges),
                    "step_index": record.step_index,
                }
            )
            if record.total_new_tokens > 0:
                goal = render_step_goal(
                    record,
                    config.dims,
                    config.tp_ranks,
                    per_layer_calc_ns=direct_lowerer.timing(record).layer_calc_ns,
                    ep_ranks=config.ep_ranks,
                    routed_supply=config.routed_moe_supply,
                ).render().encode()
                serial_goal_identities.append(
                    {
                        "bytes": len(goal),
                        "sha256": hashlib.sha256(goal).hexdigest(),
                        "step_index": record.step_index,
                    }
                )
            serial_checks.append(
                graph_json == direct_graph_json
                and execution_json == direct_execution_json
                and step_json == direct_step_json
            )

        if placement == "cross-node" and mode == "observed":
            live_checks.append(
                graph_identity == live_row["lowered_graph_identity"]
                and execution_identity == live_row["execution_result_identity"]
                and step_json == live_row["step_result"]
            )

        for metric in step_result.request_metrics:
            if metric.request_id == "r0":
                last_r0 = metric
        raw_rows.append(
            {
                "execution_result_identity": execution_identity,
                "graph_identity": graph_identity,
                "request_metric_ids": [
                    metric.request_id for metric in step_result.request_metrics
                ],
                "request_pair_identity": _canonical_identity(pair_inventory),
                "step_index": record.step_index,
                "step_result": step_json,
            }
        )

    if last_r0 is None or last_r0.tpot_ps is None:
        raise RuntimeError(f"{placement} {mode} produced no r0 TPOT")
    output = Path(raw_output)
    with output.open("w", encoding="utf-8", newline="\n") as stream:
        for row in raw_rows:
            stream.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    return {
        "live_identity_checks": live_checks,
        "mode": mode,
        "perturbation_edges": perturbation_edges,
        "placement": placement,
        "r0_tpot_ps": _fraction_to_json(last_r0.tpot_ps),
        "r0_ttft_ps": last_r0.ttft_ps,
        "raw_rows": len(raw_rows),
        "request_pair_bytes": request_pair_bytes,
        "request_pair_rows": request_pair_rows,
        "request_pair_step_identities": request_pair_step_identities,
        "schedule_preservation_checks": schedule_checks,
        "serial_direct_checks": serial_checks,
        "serial_goal_identities": serial_goal_identities,
        "serial_projection_identities": serial_projection_identities,
    }


def _run_live_driver(args: argparse.Namespace) -> Path:
    repo = Path(__file__).resolve().parents[2]
    driver = Path(__file__).with_name("live_driver.py")
    log_path = args.output_dir / "live_driver.log"
    environment = dict(os.environ)
    previous_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        str(repo)
        if not previous_pythonpath
        else str(repo) + os.pathsep + previous_pythonpath
    )
    environment["HF_HUB_OFFLINE"] = "1"
    environment["TRANSFORMERS_OFFLINE"] = "1"
    command = [
        str(args.vllm_python),
        str(driver),
        "--capture",
        str(args.capture),
        "--replay-run",
        str(args.replay_run),
        "--routed-experts",
        str(args.routed_experts),
        "--output-dir",
        str(args.output_dir),
    ]
    with log_path.open("w", encoding="utf-8", newline="\n") as log:
        completed = subprocess.run(
            command,
            cwd=repo,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode:
        raise RuntimeError(
            f"live vLLM driver failed with {completed.returncode}; see {log_path}"
        )
    live_path = args.output_dir / "live_observations.jsonl"
    if not live_path.is_file():
        raise RuntimeError("live vLLM driver produced no observation stream")
    return live_path


def _run_modes(args: argparse.Namespace, live_path: Path) -> dict[str, dict[str, object]]:
    jobs = [
        (placement, mode)
        for placement in REDUCTION_BANDS_PS
        for mode in ("serial", "observed", "perturbed")
    ]
    results: dict[str, dict[str, object]] = {}
    with ProcessPoolExecutor(max_workers=len(jobs)) as executor:
        pending = {
            executor.submit(
                _mode_run,
                placement,
                mode,
                str(live_path),
                str(args.routed_experts),
                str(args.output_dir / f"raw_{placement}_{mode}.jsonl"),
            ): (placement, mode)
            for placement, mode in jobs
        }
        for future in as_completed(pending):
            placement, mode = pending[future]
            results[f"{placement}:{mode}"] = future.result()
    return results


def _evaluate_scored(mode_results: dict[str, dict[str, object]]) -> dict[str, object]:
    rows = []
    passed = 0
    for placement, (low, high) in REDUCTION_BANDS_PS.items():
        serial = mode_results[f"{placement}:serial"]
        observed = mode_results[f"{placement}:observed"]
        perturbed = mode_results[f"{placement}:perturbed"]
        serial_tpot = _fraction_from_json(serial["r0_tpot_ps"])
        observed_tpot = _fraction_from_json(observed["r0_tpot_ps"])
        perturbed_tpot = _fraction_from_json(perturbed["r0_tpot_ps"])
        reduction = serial_tpot - observed_tpot
        perturbation = perturbed_tpot - observed_tpot
        reduction_passed = low <= reduction <= high
        perturbation_passed = (
            perturbation >= PERTURBATION_MIN_PS[placement]
            and observed_tpot < perturbed_tpot <= serial_tpot
        )
        passed += int(reduction_passed) + int(perturbation_passed)
        rows.append(
            {
                "observed_tpot_ps": _fraction_to_json(observed_tpot),
                "observed_ttft_ps": observed["r0_ttft_ps"],
                "perturbation_increase_ps": _fraction_to_json(perturbation),
                "perturbation_passed": perturbation_passed,
                "perturbed_tpot_ps": _fraction_to_json(perturbed_tpot),
                "placement": placement,
                "reduction_passed": reduction_passed,
                "serial_minus_observed_tpot_ps": _fraction_to_json(reduction),
                "serial_tpot_ps": _fraction_to_json(serial_tpot),
                "serial_ttft_ps": serial["r0_ttft_ps"],
            }
        )
    return {
        "evaluation_order": "raw mode metrics before exact and fatal guards",
        "genuine_risk_executed": EXPECTED_GENUINE_RISK_INSTANCES,
        "genuine_risk_passed": passed,
        "rows": rows,
    }


def _producer_inventory(live_path: Path) -> dict[str, object]:
    from simllm.backends import SerialStepLowerer, SerialStepLowererConfig
    from simllm.compute import RooflineProvider
    from simllm.core import CollectiveWork, ComputeWork, execution_graph_from_json

    rows = [
        json.loads(line)
        for line in live_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    nonempty = 0
    emitted = 0
    unique_site_counts = []
    invocation_counts = []
    ordered_layer_checks = []
    completion_checks = []
    compute_conservation = []
    request_partitions = []
    lowerer = SerialStepLowerer(
        SerialStepLowererConfig(
            _model_dims(),
            (0,),
            ep_ranks=tuple(range(DP_SIZE)),
            provider=RooflineProvider(efficiency=0.7),
        )
    )
    for row in rows:
        from simllm.core import step_record_from_json

        record = step_record_from_json(row["record"])
        if record.total_new_tokens <= 0:
            continue
        nonempty += 1
        source = row["source_graph"]
        if source is None:
            continue
        emitted += 1
        graph = execution_graph_from_json(source)
        collectives = [
            operation
            for operation in graph.operations
            if isinstance(operation.work, CollectiveWork)
        ]
        sites = {
            (operation.correlation.layer, operation.work.channel_hint)
            for operation in collectives
        }
        unique_site_counts.append(len(sites))
        invocation_counts.append(len(collectives))
        ordered_layer_checks.append(
            sorted({layer for layer, _ in sites}) == list(range(NUM_LAYERS))
            and sites
            == {
                (layer, site)
                for layer in range(NUM_LAYERS)
                for site in ("dispatch", "combine")
            }
        )
        microbatches = sorted(
            {
                operation.correlation.microbatch
                for operation in collectives
            },
            key=lambda value: -1 if value is None else value,
        )
        expected_completions = 1 if microbatches == [None] else len(microbatches)
        completion_checks.append(
            len(graph.completion_operation_ids) == expected_completions
            and all(
                any(
                    operation.operation_id == completion_id
                    and operation.correlation.request_ids
                    for operation in graph.operations
                )
                for completion_id in graph.completion_operation_ids
            )
        )
        correlated = {
            microbatch: next(
                operation.correlation.request_ids
                for operation in collectives
                if operation.correlation.microbatch == microbatch
            )
            for microbatch in microbatches
        }
        flattened = tuple(
            request_id
            for microbatch in microbatches
            for request_id in correlated[microbatch]
        )
        request_partitions.append(
            flattened == tuple(request.request_id for request in record.scheduled)
        )
        rank_zero_compute = sum(
            operation.work.nominal_duration_ps or 0
            for operation in graph.operations
            if operation.rank == 0 and isinstance(operation.work, ComputeWork)
        )
        represented_serial = sum(lowerer.timing(record).layer_calc_ns) * 1_000
        compute_conservation.append(rank_zero_compute == represented_serial)

    schedule_source = (
        Path(__file__).resolve().parents[2]
        / "simllm"
        / "adapters"
        / "vllm"
        / "schedule.py"
    ).read_text(encoding="utf-8")
    forbidden_knobs = [
        token
        for token in ("overlap_fraction", "overlap_percent", "overlap_percentage")
        if token in schedule_source
    ]
    return {
        "completion_frontier_checks": completion_checks,
        "compute_conservation_checks": compute_conservation,
        "dbo_invocation_rows": sum(count == DBO_MOE_INVOCATIONS for count in invocation_counts),
        "emitted_observation_steps": emitted,
        "forbidden_overlap_knobs": forbidden_knobs,
        "invocation_counts": invocation_counts,
        "nonempty_translated_steps": nonempty,
        "ordered_layer_checks": ordered_layer_checks,
        "request_partition_checks": request_partitions,
        "unique_semantic_site_counts": unique_site_counts,
    }


def _serial_fixture() -> dict[str, object]:
    from simllm.backends import ObservedStepLowerer, SerialStepLowerer, SerialStepLowererConfig
    from simllm.compute import ComputeProvider, DurationEstimate, ModelDims
    from simllm.core import RequestPhase, ScheduledRequest, StepRecord, execution_graph_to_json
    from simllm.traffic import project_execution_graph_goal, render_step_goal

    class _FlopProvider(ComputeProvider):
        def estimate(self, kernel, gpu):
            return DurationEstimate(duration_ps=int(kernel.flops), bound="compute")

    dims = ModelDims(2, 64, 128, 4, 4, 16, 256, 2)
    record = StepRecord(
        0,
        0,
        [
            ScheduledRequest(
                "p",
                RequestPhase.PREFILL,
                4,
                num_cached_tokens=4,
                context_length=8,
            ),
            ScheduledRequest("d", RequestPhase.DECODE, 1, context_length=32),
        ],
    )
    config = SerialStepLowererConfig(dims, (0, 1), provider=_FlopProvider())
    graph = ObservedStepLowerer(config).lower(record, None)
    direct = SerialStepLowerer(config).lower(record)
    wire = (
        json.dumps(execution_graph_to_json(graph), sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode()
    projection = project_execution_graph_goal(graph)
    projection_identities = [
        {
            "bytes": len(artifact.trace.render().encode()),
            "sha256": hashlib.sha256(artifact.trace.render().encode()).hexdigest(),
        }
        for artifact in projection.artifacts
    ]
    goal = render_step_goal(
        record,
        dims,
        config.tp_ranks,
        per_layer_calc_ns=SerialStepLowerer(config).timing(record).layer_calc_ns,
    ).render().encode()
    return {
        "direct_graph_equal": execution_graph_to_json(graph)
        == execution_graph_to_json(direct),
        "graph_bytes": len(wire),
        "graph_sha256": hashlib.sha256(wire).hexdigest(),
        "goal_bytes": len(goal),
        "goal_sha256": hashlib.sha256(goal).hexdigest(),
        "projection_artifact_identities": projection_identities,
        "projection_boundaries": len(projection.boundaries),
        "projection_serialized_edges": len(projection.serialized_edges),
    }


def _fatal_guards(
    scored: dict[str, object],
    modes: dict[str, dict[str, object]],
    inventory: dict[str, object],
    fixture: dict[str, object],
) -> dict[str, object]:
    rows = {row["placement"]: row for row in scored["rows"]}
    single_reduction = _fraction_from_json(
        rows["single-node"]["serial_minus_observed_tpot_ps"]
    )
    cross_reduction = _fraction_from_json(
        rows["cross-node"]["serial_minus_observed_tpot_ps"]
    )
    all_mode_tpots = [
        _fraction_from_json(result["r0_tpot_ps"])
        for result in modes.values()
    ]
    traffic_inventories = [
        modes[f"{placement}:{mode}"]["request_pair_step_identities"]
        for placement in REDUCTION_BANDS_PS
        for mode in ("serial", "observed", "perturbed")
    ]
    return {
        "cross_node_bandwidth_scaling": (
            cross_reduction > single_reduction
            and cross_reduction >= 5 * single_reduction
        ),
        "live_cross_node_identity": all(
            modes["cross-node:observed"]["live_identity_checks"]
        ),
        "physical_decode_bounds": all(
            69_206_016 <= value <= 332_000_000 for value in all_mode_tpots
        ),
        "producer_completion_frontiers": all(
            inventory["completion_frontier_checks"]
        ),
        "producer_compute_conservation": all(inventory["compute_conservation_checks"]),
        "producer_every_nonempty_step": (
            inventory["nonempty_translated_steps"]
            == inventory["emitted_observation_steps"]
        ),
        "producer_layer_and_site_inventory": (
            all(inventory["ordered_layer_checks"])
            and all(
                count == SEMANTIC_MOE_SITES
                for count in inventory["unique_semantic_site_counts"]
            )
        ),
        "producer_no_overlap_knob": not inventory["forbidden_overlap_knobs"],
        "producer_request_partitions": all(inventory["request_partition_checks"]),
        "schedule_fields_survive_lowering": all(
            check
            for result in modes.values()
            for check in result["schedule_preservation_checks"]
        ),
        "request_pair_rebinding_exact": (
            bool(modes["single-node:serial"]["request_pair_rows"])
            and all(
                inventory == traffic_inventories[0]
                for inventory in traffic_inventories[1:]
            )
        ),
        "serial_direct_all_steps": all(
            check
            for key, result in modes.items()
            if key.endswith(":serial")
            for check in result["serial_direct_checks"]
        ),
        "serial_fixture": (
            fixture["direct_graph_equal"]
            and fixture["graph_bytes"] == SERIAL_GRAPH_BYTES
            and fixture["graph_sha256"] == SERIAL_GRAPH_SHA256
            and fixture["goal_bytes"] == SERIAL_GOAL_BYTES
            and fixture["goal_sha256"] == SERIAL_GOAL_SHA256
        ),
        "ttft_exact_single_batch": all(
            row["serial_ttft_ps"] == row["observed_ttft_ps"]
            for row in scored["rows"]
        ),
    }


def _git_head() -> str | None:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def _run_study(args: argparse.Namespace) -> dict[str, object]:
    live_path = _run_live_driver(args)
    modes = _run_modes(args, live_path)

    # Scored relations are evaluated from the raw mode rows before any later
    # exact-identity or fatal-unscored result is interpreted.
    scored = _evaluate_scored(modes)
    inventory = _producer_inventory(live_path)
    fixture = _serial_fixture()
    fatal = _fatal_guards(scored, modes, inventory, fixture)
    return {
        "behavioral_evidence": scored,
        "capture": {
            "rows": _line_count(args.capture),
            "sha256": _sha256(args.capture),
        },
        "fatal_unscored_guards": fatal,
        "mode_results": modes,
        "producer_inventory": inventory,
        "provenance": {
            "expectation_commit": "a91ac06",
            "repository_commit_observed": _git_head(),
            "vllm_authored_against_commit": (
                "568afb3a13806beb53bb2e6bd518269357b237c0"
            ),
            "vllm_observed_file_sha256": SOURCE_HASHES,
            "vllm_observed_version": VLLM_VERSION,
        },
        "schema": "simllm-vllm-observed-schedule-study-v1",
        "serial_fixture": fixture,
    }


def main() -> None:
    args = _parse_args()
    check_expectation_registry(args)
    if args.check_only:
        print("check-only: frozen expectation registry passed; no artifacts written")
        return
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = _run_study(args)
    result_path = args.output_dir / "results.json"
    result_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    behavior = report["behavioral_evidence"]
    print(
        "genuine-risk instances: "
        f"{behavior['genuine_risk_passed']}/{behavior['genuine_risk_executed']}"
    )
    print(f"fatal guards passed: {sum(report['fatal_unscored_guards'].values())}/"
          f"{len(report['fatal_unscored_guards'])}")
    print(f"wrote {result_path}")


if __name__ == "__main__":
    main()
