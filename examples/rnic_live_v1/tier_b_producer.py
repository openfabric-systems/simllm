"""Produce raw Tier B live-chain observations from composed native evidence."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from simllm._local_config import path_from_env
from simllm.backends.composed_rnic import (
    ComposedRnicCell,
    ComposedRnicObservations,
    ComposedRnicSession,
    invoke_composed_tier_a_producer,
)
from simllm.backends.htsim_dcqcn import HtsimDcqcnConfig, run_htsim_dcqcn
from simllm.backends.htsim_rnic import HtsimRnicConfig, RnicRunResult, run_htsim_rnic
from simllm.core.execution import (
    COMPLETION_EVENT_SCHEMA,
    EXECUTION_RESULT_SCHEMA,
)
from simllm.goal import to_binary

STUDY_DIR = Path(__file__).resolve().parent
TIER_A_EXPECTATIONS = STUDY_DIR / "tier_a_expectations.json"
TIER_B_SCHEMA = "simllm-rnic-tier-b-expectations-v1"
TIER_B_OBSERVATION_SCHEMA = "simllm-rnic-tier-b-observations-v1"
FROZEN_SIMLLM_BASE = "fc282efc91573638de7dcfae2befee1cf022011b"
T0 = 7_000
REQUEST_ID = "tier-b-request"
BYPASS_PROFILES = ("rnic-nn-fluid", "rnic-nn", "rnic-cn", "dcqcn")
TRACKED_BYPASS_TOPOLOGY = (
    REPO_ROOT
    / "third_party"
    / "htsim"
    / "htsim"
    / "sim"
    / "datacenter"
    / "topologies"
    / "leaf_spine_tiny.topo"
)


def _required_path(name: str) -> Path:
    path = path_from_env(name)
    if path is None:
        raise RuntimeError(f"{name} must name the required Tier B executable")
    return path.resolve(strict=True)


def _load_json(path: Path, name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot read {name} {path}: {error}") from error
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be a JSON object")
    return value


def _validate_expectations(value: dict[str, Any], factory: str) -> None:
    if value.get("schema") != TIER_B_SCHEMA:
        raise ValueError("unsupported Tier B expectations schema")
    if value.get("observation_schema") != TIER_B_OBSERVATION_SCHEMA:
        raise ValueError("unsupported Tier B observation schema")
    if value.get("factory") != factory or factory != "htsim":
        raise ValueError("Tier B requires the composed htsim factory")
    if value.get("simllm_base_commit") != FROZEN_SIMLLM_BASE:
        raise ValueError("Tier B SimLLM base commit drifted")
    if value.get("retained_bypass_profiles") != list(BYPASS_PROFILES):
        raise ValueError("Tier B bypass profile set drifted")


def _fraction(value: Any) -> dict[str, int] | None:
    if value is None:
        return None
    return {"numerator": value.numerator, "denominator": value.denominator}


def _additive(value: Any) -> dict[str, int]:
    return {
        "queue_wait_ps": value.queue_wait_ps,
        "service_ps": value.service_ps,
        "visibility_ps": value.visibility_ps,
        "visit_count": value.visit_count,
    }


def _profile(rate_gbps: int):
    from simllm.core import CoarseDeviceProfile

    return CoarseDeviceProfile(
        gpus_per_node=8,
        rnics_per_node=8,
        rnic_rate_bps=rate_gbps * 1_000_000_000,
        nvlink_rate_bps=100_000_000_000,
        launch_service_ps=0,
        control_service_ps=0,
        nccl_channel_service_ps=0,
        completion_delivery_ps=0,
        copy_engines=(),
        goal_base_tag=1000,
    )


def _graph(step_index: int, release_ps: int, payload_bytes: int, wqe_count: int):
    from simllm.core import (
        ControlMode,
        ControlWork,
        ExecutionGraph,
        ExecutionOperation,
        OperationCorrelation,
    )

    correlation = OperationCorrelation(request_ids=(REQUEST_ID,))
    operations = tuple(
        ExecutionOperation(
            operation_id=f"tier-b-wqe-{ordinal}",
            rank=0,
            logical_queue=f"tier-b-control-{ordinal}",
            work=ControlWork(
                message=f"tier-b-send-{ordinal}",
                destination_ranks=(8,),
                payload_bytes=payload_bytes,
                mode=ControlMode.SYNCHRONOUS,
            ),
            correlation=correlation,
        )
        for ordinal in range(wqe_count)
    )
    return ExecutionGraph(
        execution_id=(
            f"tier-b-{payload_bytes}-{wqe_count}-step-{step_index}-at-{release_ps}"
        ),
        step_index=step_index,
        released_at_ps=release_ps,
        operations=operations,
        completion_operation_ids=(operations[-1].operation_id,),
    )


def _step_record(step_index: int, release_ps: int):
    from simllm.core import RequestPhase, ScheduledRequest, StepRecord

    phase = RequestPhase.PREFILL if step_index == 0 else RequestPhase.DECODE
    return StepRecord(
        step_index=step_index,
        virtual_time_ps=release_ps,
        scheduled=[ScheduledRequest(REQUEST_ID, phase, 1)],
        num_sampled=1,
        sampled_request_ids=[REQUEST_ID],
    )


def _event_row(index: int, event: Any) -> dict[str, Any]:
    resource_kind = None
    resource_id = None
    if event.resource is not None:
        resource_kind = event.resource.kind.value
        resource_id = event.resource.resource_id
    return {
        "event_index": index,
        "schema": COMPLETION_EVENT_SCHEMA,
        "execution_id": event.execution_id,
        "operation_id": event.operation_id,
        "phase": event.phase.value,
        "timestamp_ps": event.timestamp_ps,
        "resource_kind": resource_kind,
        "resource_id": resource_id,
        "completed_bytes": event.completed_bytes,
        "subject_object_id": event.subject_object_id,
    }


def _runtime_report_row(report: Any) -> dict[str, Any]:
    operations = [
        {
            "operation_id": operation.operation_id,
            "completed_at_ps": operation.completed_at_ps,
            "critical_predecessor_id": operation.critical_predecessor_id,
            "attribution": asdict(operation.attribution),
        }
        for operation in report.operations
    ]
    visits = [
        {
            "execution_id": visit.execution_id,
            "operation_id": visit.operation_id,
            "stage": visit.stage or "coarse_runtime",
            "resource_kind": visit.resource.kind.value,
            "resource_id": visit.resource.resource_id,
            "subject_object_id": visit.subject_object_id,
            "submitted_at_ps": visit.submitted_at_ps,
            "eligible_at_ps": visit.eligible_at_ps,
            "started_at_ps": visit.started_at_ps,
            "finished_at_ps": visit.finished_at_ps,
            "completed_at_ps": visit.completed_at_ps,
            "service_bytes": visit.service_bytes,
        }
        for visit in report.visits
    ]
    wqes = []
    for ordinal, wqe in enumerate(report.wqes):
        if any(
            value is None
            for value in (
                wqe.doorbell_started_at_ps,
                wqe.doorbell_completed_at_ps,
                wqe.network_eligible_at_ps,
                wqe.network_started_at_ps,
                wqe.network_finished_at_ps,
            )
        ):
            raise RuntimeError("Tier B native projection omitted a stage timestamp")
        wqes.append(
            {
                "ordinal": ordinal,
                "wqe_id": wqe.wqe_id,
                "operation_id": wqe.operation_id,
                "submitted_at_ps": wqe.submitted_at_ps,
                "doorbell_started_at_ps": wqe.doorbell_started_at_ps,
                "doorbell_completed_at_ps": wqe.doorbell_completed_at_ps,
                "network_eligible_at_ps": wqe.network_eligible_at_ps,
                "network_started_at_ps": wqe.network_started_at_ps,
                "network_finished_at_ps": wqe.network_finished_at_ps,
                "completed_at_ps": wqe.completed_at_ps,
                "payload_bytes": wqe.payload_bytes,
                "sq_post_sequence": wqe.sq_post_sequence,
                "cq_post_sequence": wqe.cq_post_sequence,
            }
        )
    return {
        "execution_id": report.execution_id,
        "authority": report.authority,
        "operations": operations,
        "visits": visits,
        "wqes": wqes,
        "sum_visit_wait_ps": report.sum_visit_wait_ps,
        "critical_path_queue_ps": report.critical_path_queue_ps,
        "realized_critical_path_operation_ids": list(
            report.realized_critical_path_operation_ids
        ),
    }


def _metric_row(metric: Any) -> dict[str, Any]:
    return {
        "request_id": metric.request_id,
        "phase": metric.phase.value,
        "token_index": metric.token_index,
        "completed_at_ps": metric.completed_at_ps,
        "latency_ps": metric.latency_ps,
        "ttft_ps": metric.ttft_ps,
        "tpot_ps": _fraction(metric.tpot_ps),
        "attribution": asdict(metric.attribution),
        "additive_visit_totals": _additive(metric.additive_visit_totals),
    }


def _run_structural_cell(cell: ComposedRnicCell) -> dict[str, Any]:
    from simllm.core import (
        CoarseDeviceRuntime,
        CompletionReducer,
        RnicAuthorityMode,
        VirtualClock,
    )

    clock = VirtualClock(T0)
    reducer = CompletionReducer(clock)
    session = ComposedRnicSession(
        cell,
        session_id=(
            f"tier-b-{cell.payload_bytes}-{cell.link_rate_gbps}-"
            f"{cell.doorbell_service_ps}-{cell.wqe_count}wqe"
        ),
    )
    runtime = CoarseDeviceRuntime(
        _profile(cell.link_rate_gbps),
        authority_mode=RnicAuthorityMode.STRUCTURAL,
        native_session=session,
    )
    steps: list[dict[str, Any]] = []
    completion_times: list[int] = []
    for step_index in range(3):
        record = _step_record(step_index, clock.now_ps)
        graph = _graph(step_index, clock.now_ps, cell.payload_bytes, cell.wqe_count)
        callbacks: list[Any] = []
        execution = runtime.execute(graph, on_event=callbacks.append)
        report = runtime.last_report
        if report is None:
            raise RuntimeError("Tier B runtime omitted its report")
        if len(callbacks) != len(execution.events) or any(
            callback is not event
            for callback, event in zip(callbacks, execution.events, strict=True)
        ):
            raise RuntimeError("callback and ExecutionResult events are not identical")
        step = reducer.reduce(record, graph, execution, report)
        event_indices = list(range(len(execution.events)))
        completion_times.append(step.completed_at_ps)
        steps.append(
            {
                "step_index": step_index,
                "execution_graph": {
                    "execution_id": graph.execution_id,
                    "step_index": graph.step_index,
                    "released_at_ps": graph.released_at_ps,
                    "operation_ids": [
                        operation.operation_id for operation in graph.operations
                    ],
                    "completion_operation_ids": list(
                        graph.completion_operation_ids
                    ),
                    "request_ids": [REQUEST_ID],
                },
                "callback_event_indices": event_indices,
                "completion_events": [
                    _event_row(index, event)
                    for index, event in enumerate(execution.events)
                ],
                "execution_result": {
                    "schema": EXECUTION_RESULT_SCHEMA,
                    "execution_id": execution.execution_id,
                    "completed_at_ps": execution.completed_at_ps,
                    "quiesced_at_ps": execution.quiesced_at_ps,
                    "event_indices": event_indices,
                },
                "runtime_report": _runtime_report_row(report),
                "step_result": {
                    "step_index": step.step_index,
                    "step_latency_ps": step.step_latency_ps,
                    "completed_at_ps": step.completed_at_ps,
                    "request_metrics": [
                        _metric_row(metric) for metric in step.request_metrics
                    ],
                    "additive_visit_totals": _additive(
                        step.additive_visit_totals
                    ),
                },
            }
        )
    latest = reducer.latest_request_metrics
    if len(latest) != 1:
        raise RuntimeError("Tier B reducer omitted the latest request metric")
    return {
        "payload_bytes": cell.payload_bytes,
        "link_rate_gbps": cell.link_rate_gbps,
        "doorbell_service_ps": cell.doorbell_service_ps,
        "hardware_mode": "structural",
        "authority": session.authority_name,
        "simllm_profile": "coarse-zero-service",
        "steps": steps,
        "request_summary": {
            "request_id": REQUEST_ID,
            "token_completion_times_ps": completion_times,
            "ttft_ps": latest[0].ttft_ps,
            "tpot_ps": _fraction(latest[0].tpot_ps),
        },
    }


def _goal_text() -> str:
    ranks = []
    for rank in range(32):
        if rank == 0:
            body = "op0: send 4096b to 1 tag 7"
        elif rank == 1:
            body = "op0: recv 4096b from 0 tag 7"
        else:
            body = ""
        ranks.append(f"rank {rank} {{\n{body}\n}}")
    return "num_ranks 32\n\n" + "\n\n".join(ranks) + "\n"


def _write_bypass_topology(temp_root: Path) -> Path:
    source = TRACKED_BYPASS_TOPOLOGY.resolve(strict=True).read_text(encoding="utf-8")
    speed_line = "Downlink_speed_Gbps 100"
    if source.count(speed_line) != 2:
        raise RuntimeError("tracked tiny topology no longer has two equal 100G tiers")
    configured = source.replace(speed_line, "Downlink_speed_Gbps 400")
    topology = temp_root / "leaf_spine_tiny_400g.topo"
    topology.write_text(configured, encoding="utf-8", newline="\n")
    return topology


def _canonical_rows(run: RnicRunResult) -> list[list[Any]]:
    rows = [
        [
            flow.profile,
            flow.flow_id,
            flow.source,
            flow.destination,
            flow.tag,
            flow.payload_bytes,
            flow.start_time_ps,
            flow.completion_time_ps,
            flow.fct_ps,
            flow.wqe_id,
            flow.sq_id,
            flow.rq_id,
            flow.cq_id,
            flow.sq_post_sequence,
            flow.sq_dispatch_sequence,
            flow.cq_post_sequence,
            flow.cq_consume_sequence,
            flow.transport_kind,
            flow.transport_object_id,
        ]
        for flow in run.flows
    ]
    rows.append(["jct_ps", run.job_completion_time_ps()])
    return rows


def _replay_artifacts(run: RnicRunResult) -> tuple[list[list[int]], list[Any]]:
    jct = run.job_completion_time_ps()
    completions = [T0 + (index + 1) * jct for index in range(3)]
    step_tuples = [[index, jct, completed] for index, completed in enumerate(completions)]
    summary = [REQUEST_ID, completions, jct, [jct, 1]]
    return step_tuples, summary


def _run_bypass_profile(
    profile: str,
    *,
    rnic_binary: Path,
    dcqcn_binary: Path,
    goal_bin: Path,
    topology: Path,
    run_dir: Path,
) -> tuple[dict[str, Any], list[str]]:
    completion_csv = run_dir / "completion.csv"
    if profile == "dcqcn":
        flags = {
            "-seed": "1",
            "-ecn_seed": "1",
            "-data_header_bytes": "0",
            "-pfc": "off",
        }
        run = run_htsim_dcqcn(
            HtsimDcqcnConfig(
                goal_bin=goal_bin,
                topology=topology,
                link_bps=400_000_000_000,
                completion_csv=completion_csv,
                extra_flags=flags,
            ),
            binary=dcqcn_binary,
        )
        argv = [
            "-goal",
            "fixture.bin",
            "-topology",
            "leaf_spine_tiny_400g.topo",
            "-link_bps",
            "400000000000",
            "-completion_csv",
            "completion.csv",
            *[item for pair in flags.items() for item in pair],
        ]
    else:
        flags: dict[str, str] = {}
        if profile == "rnic-nn":
            flags = {
                "-rnic_data_header_bytes": "0",
                "-rnic_nn_propagation_ps": "0",
            }
        elif profile == "rnic-cn":
            flags = {
                "-rnic_data_header_bytes": "0",
                "-rnic_cn_prbs_seed": "1",
            }
        run = run_htsim_rnic(
            HtsimRnicConfig(
                goal_bin=goal_bin,
                profile=profile,
                linkspeed_bps=400_000_000_000,
                completion_csv=completion_csv,
                extra_flags=flags,
            ),
            binary=rnic_binary,
        )
        argv = [
            "-goal",
            "fixture.bin",
            "-linkspeed_bps",
            "400000000000",
            "-rnic_profile",
            profile,
            "-completion_csv",
            "completion.csv",
            *[item for pair in flags.items() for item in pair],
        ]
    step_tuples, request_summary = _replay_artifacts(run)
    artifacts = {
        "completion_csv_hex": completion_csv.read_bytes().hex(),
        "canonical_completion_rows": _canonical_rows(run),
        "step_result_tuples": step_tuples,
        "replay_request_summary": request_summary,
    }
    return artifacts, argv


def _bypass_rows(temp_root: Path) -> list[dict[str, Any]]:
    reference_rnic = _required_path("SIMLLM_TIER_B_REFERENCE_RNIC")
    reference_dcqcn = _required_path("SIMLLM_TIER_B_REFERENCE_DCQCN")
    candidate_rnic = _required_path("SIMLLM_TIER_B_BYPASS_RNIC")
    candidate_dcqcn = _required_path("SIMLLM_TIER_B_BYPASS_DCQCN")
    txt2bin = _required_path("SIMLLM_TXT2BIN")
    topology = _write_bypass_topology(temp_root)
    goal_text = _goal_text()
    goal_path = temp_root / "fixture.goal"
    goal_path.write_text(goal_text, encoding="utf-8", newline="\n")
    goal_bin = to_binary(goal_path, temp_root / "fixture.bin", tool=txt2bin)
    rows = []
    for profile in BYPASS_PROFILES:
        reference_dir = temp_root / f"bypass-{profile}-reference"
        candidate_dir = temp_root / f"bypass-{profile}-candidate"
        reference_dir.mkdir()
        candidate_dir.mkdir()
        reference, reference_argv = _run_bypass_profile(
            profile,
            rnic_binary=reference_rnic,
            dcqcn_binary=reference_dcqcn,
            goal_bin=goal_bin,
            topology=topology,
            run_dir=reference_dir,
        )
        candidate, candidate_argv = _run_bypass_profile(
            profile,
            rnic_binary=candidate_rnic,
            dcqcn_binary=candidate_dcqcn,
            goal_bin=goal_bin,
            topology=topology,
            run_dir=candidate_dir,
        )
        if reference_argv != candidate_argv:
            raise RuntimeError(f"{profile} bypass argv changed between builds")
        rows.append(
            {
                "profile": profile,
                "hardware_mode": "bypass",
                "authority": "AtlahsWqeLedger",
                "inputs": {
                    "goal_text_hex": goal_text.encode("utf-8").hex(),
                    "goal_binary_hex": goal_bin.read_bytes().hex(),
                    "topology_hex": (
                        topology.read_bytes().hex() if profile == "dcqcn" else ""
                    ),
                    "seed": 1,
                    "baseline_argv": reference_argv,
                },
                "reference_artifacts": reference,
                "candidate_artifacts": candidate,
            }
        )
    return rows


def _publish(path: Path, value: dict[str, Any]) -> None:
    temporary = Path(f"{path}.tmp")
    if path.exists() or temporary.exists():
        raise FileExistsError("Tier B observation or temporary file already exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(value, indent=2, sort_keys=True) + "\n"
    try:
        temporary.write_text(serialized, encoding="utf-8", newline="\n")
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def produce(factory: str, expectations_path: Path, observations_path: Path) -> None:
    expectations = _load_json(expectations_path, "Tier B expectations")
    _validate_expectations(expectations, factory)
    if observations_path.exists() or Path(f"{observations_path}.tmp").exists():
        raise FileExistsError("Tier B observations already exist")
    tier_a_producer = _required_path("SIMLLM_RNIC_TIER_A_PRODUCER")
    _required_path("SIMLLM_HTSIM_RNIC")

    with tempfile.TemporaryDirectory(
        prefix=".tier-b-producer-",
        dir=observations_path.parent,
    ) as temporary:
        temp_root = Path(temporary)
        native_path = temp_root / "tier_a_observations.json"
        invoke_composed_tier_a_producer(
            tier_a_producer,
            TIER_A_EXPECTATIONS,
            native_path,
        )
        native_raw = _load_json(native_path, "Tier A composed observations")
        from examples.rnic_live_v1.tier_a_acceptance import (
            _validate_expectations as validate_tier_a_expectations,
        )
        from examples.rnic_live_v1.tier_a_acceptance import check_observations

        tier_a_expectations = _load_json(TIER_A_EXPECTATIONS, "Tier A expectations")
        validate_tier_a_expectations(tier_a_expectations)
        check_observations(native_raw, tier_a_expectations, "htsim")
        native = ComposedRnicObservations.from_json(native_raw)
        structural_single = [
            _run_structural_cell(native.single_wqe[key])
            for key in sorted(native.single_wqe)
        ]
        structural_fifo = [
            _run_structural_cell(native.fifo[key]) for key in sorted(native.fifo)
        ]
        bypass = _bypass_rows(temp_root)

    observations = {
        "schema": TIER_B_OBSERVATION_SCHEMA,
        "factory": factory,
        "simllm_base_commit": FROZEN_SIMLLM_BASE,
        "structural_single_wqe": structural_single,
        "structural_fifo": structural_fifo,
        "bypass": bypass,
    }
    _publish(observations_path, observations)


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--factory", required=True)
    parser.add_argument("--expectations", required=True, type=Path)
    parser.add_argument("--observations", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    arguments = _parse_arguments()
    try:
        produce(
            arguments.factory,
            arguments.expectations.resolve(strict=True),
            arguments.observations.resolve(strict=False),
        )
    except BaseException as error:
        print(f"Tier B producer error: {error}", file=sys.stderr)
        raise SystemExit(2) from error


if __name__ == "__main__":
    main()
