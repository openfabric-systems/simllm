"""Produce raw observations for the frozen RNIC authority comparison."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from simllm.backends.composed_rnic import (
    ComposedRnicCell,
    ComposedRnicObservations,
    ComposedRnicSession,
    invoke_composed_tier_a_producer,
)
from simllm.backends.htsim_rnic import (
    HtsimRnicConfig,
    run_htsim_rnic,
)
from simllm.backends.rnic_records import (
    BypassArtifacts,
    canonical_bypass_parameters,
)
from simllm.core import (
    CoarseDeviceProfile,
    CoarseDeviceRuntime,
    CompletionReducer,
    ComputeWork,
    ControlMode,
    ControlWork,
    ExecutionGraph,
    ExecutionOperation,
    OperationCorrelation,
    RequestPhase,
    RnicAuthorityMode,
    ScheduledRequest,
    StepRecord,
    VirtualClock,
    execution_graph_to_json,
    execution_result_to_json,
    step_record_to_json,
)
from simllm.goal import GoalTrace, to_binary

EXPECTATIONS = Path(__file__).with_name("expectations.json")
TIER_A_EXPECTATIONS = ROOT / "examples" / "rnic_live_v1" / "tier_a_expectations.json"


class AuthorityProducerError(RuntimeError):
    """The raw producer could not establish the frozen authority path."""


def _object(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be a JSON object")
    return value


def load_expectations(path: str | Path = EXPECTATIONS) -> dict[str, Any]:
    """Load and validate the literals used by both producer and tests."""

    source = Path(path)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AuthorityProducerError(f"cannot read authority expectations: {error}") from error
    data = _object(value, "authority expectations")
    if data.get("schema") != "simllm-rnic-authority-expectations-v1":
        raise AuthorityProducerError("unsupported authority expectations schema")
    if data.get("observation_schema") != "simllm-rnic-authority-observations-v1":
        raise AuthorityProducerError("authority observation schema drifted")
    graph = _object(data.get("fixed_graph"), "fixed_graph")
    if graph.get("operation_ids") != ["core21-contended-send"]:
        raise AuthorityProducerError("fixed graph operation inventory drifted")
    if graph.get("destination_ranks") != [8, 16] or graph.get("wqe_count") != 2:
        raise AuthorityProducerError("fixed graph contention shape drifted")
    if data.get("sweep") != {
        "link_rate_gbps": [200, 400],
        "doorbell_service_ps": [0, 1000],
    }:
        raise AuthorityProducerError("authority comparison matrix drifted")
    return data


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_head(path: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _profile(rate_gbps: int) -> CoarseDeviceProfile:
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


def build_history_seed_graph(expectations: Mapping[str, Any]) -> ExecutionGraph:
    """Return the common real-reduction seed for the decode request."""

    seed = expectations["history_seed"]
    operation = ExecutionOperation(
        operation_id="core21-history-seed",
        rank=0,
        logical_queue="core21-history-seed",
        work=ComputeWork(
            kernel="core21-history-seed",
            nominal_duration_ps=seed["duration_ps"],
        ),
        correlation=OperationCorrelation(request_ids=(seed["request_id"],)),
    )
    return ExecutionGraph(
        execution_id="core21-history-seed",
        step_index=0,
        released_at_ps=seed["first_release_ps"],
        operations=(operation,),
        completion_operation_ids=(operation.operation_id,),
    )


def build_history_seed_record(expectations: Mapping[str, Any]) -> StepRecord:
    seed = expectations["history_seed"]
    return StepRecord(
        step_index=0,
        virtual_time_ps=seed["first_release_ps"],
        scheduled=[
            ScheduledRequest(
                request_id=seed["request_id"],
                phase=RequestPhase.PREFILL,
                num_new_tokens=1,
            )
        ],
        num_sampled=1,
        sampled_request_ids=[seed["request_id"]],
    )


def build_fixed_graph(expectations: Mapping[str, Any]) -> ExecutionGraph:
    """Return the one operation, two-extent decision graph."""

    frozen = expectations["fixed_graph"]
    operation = ExecutionOperation(
        operation_id=frozen["operation_ids"][0],
        rank=frozen["source_rank"],
        logical_queue=frozen["logical_queue"],
        work=ControlWork(
            message="core21-contended-send",
            destination_ranks=tuple(frozen["destination_ranks"]),
            payload_bytes=frozen["payload_bytes"],
            mode=ControlMode.SYNCHRONOUS,
        ),
        correlation=OperationCorrelation(request_ids=tuple(frozen["request_ids"])),
    )
    return ExecutionGraph(
        execution_id=frozen["execution_id"],
        step_index=frozen["step_index"],
        released_at_ps=frozen["released_at_ps"],
        operations=(operation,),
        completion_operation_ids=tuple(frozen["completion_operation_ids"]),
    )


def build_fixed_step_record(expectations: Mapping[str, Any]) -> StepRecord:
    frozen = expectations["fixed_graph"]
    prefill_id, decode_id = frozen["request_ids"]
    return StepRecord(
        step_index=frozen["step_index"],
        virtual_time_ps=frozen["released_at_ps"],
        scheduled=[
            ScheduledRequest(prefill_id, RequestPhase.PREFILL, 1),
            ScheduledRequest(decode_id, RequestPhase.DECODE, 1),
        ],
        num_sampled=2,
        sampled_request_ids=[prefill_id, decode_id],
    )


def _build_failure_graph(expectations: Mapping[str, Any]) -> ExecutionGraph:
    frozen = expectations["fixed_graph"]
    operation = ExecutionOperation(
        operation_id="core21-incomplete-transaction",
        rank=frozen["source_rank"],
        logical_queue=frozen["logical_queue"],
        work=ControlWork(
            message="core21-incomplete-transaction",
            destination_ranks=(frozen["destination_ranks"][0],),
            payload_bytes=frozen["payload_bytes"],
            mode=ControlMode.SYNCHRONOUS,
        ),
        correlation=OperationCorrelation(request_ids=tuple(frozen["request_ids"])),
    )
    return ExecutionGraph(
        execution_id="core21-incomplete-transaction",
        step_index=frozen["step_index"],
        released_at_ps=frozen["released_at_ps"],
        operations=(operation,),
        completion_operation_ids=(operation.operation_id,),
    )


def _fraction(value: Any) -> dict[str, int] | None:
    if value is None:
        return None
    return {"numerator": value.numerator, "denominator": value.denominator}


def _additive(value: Any) -> dict[str, int] | None:
    if value is None:
        return None
    return {
        "queue_wait_ps": value.queue_wait_ps,
        "service_ps": value.service_ps,
        "visibility_ps": value.visibility_ps,
        "visit_count": value.visit_count,
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


def _step_result_row(step: Any) -> dict[str, Any]:
    return {
        "step_index": step.step_index,
        "step_latency_ps": step.step_latency_ps,
        "completed_at_ps": step.completed_at_ps,
        "request_metrics": [_metric_row(metric) for metric in step.request_metrics],
        "additive_visit_totals": _additive(step.additive_visit_totals),
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
    wqe_fields = (
        "authority",
        "execution_id",
        "operation_id",
        "wqe_id",
        "native_wqe_id",
        "sq_id",
        "rq_id",
        "cq_id",
        "qp_id",
        "rnic_id",
        "source_rank",
        "destination_rank",
        "payload_bytes",
        "goal_tag",
        "extent_index",
        "sq_post_sequence",
        "cq_post_sequence",
        "submitted_at_ps",
        "eligible_at_ps",
        "started_at_ps",
        "finished_at_ps",
        "completed_at_ps",
        "channel_id",
        "nccl_command_id",
        "doorbell_started_at_ps",
        "doorbell_completed_at_ps",
        "network_eligible_at_ps",
        "network_started_at_ps",
        "network_finished_at_ps",
    )
    wqes = [{name: getattr(wqe, name) for name in wqe_fields} for wqe in report.wqes]
    return {
        "execution_id": report.execution_id,
        "authority": report.authority,
        "operations": operations,
        "visits": visits,
        "wqes": wqes,
        "sum_visit_wait_ps": report.sum_visit_wait_ps,
        "critical_path_queue_ps": report.critical_path_queue_ps,
        "realized_critical_path_operation_ids": list(report.realized_critical_path_operation_ids),
        "random_draw_count": report.random_draw_count,
    }


def _request_summary(step: Any, expectations: Mapping[str, Any]) -> dict[str, Any]:
    prefill_id, decode_id = expectations["fixed_graph"]["request_ids"]
    metrics = {metric.request_id: metric for metric in step.request_metrics}
    if set(metrics) != {prefill_id, decode_id}:
        raise AuthorityProducerError("fixed StepResult lost a request metric")
    return {
        "jct_ps": step.step_latency_ps,
        "prefill_ttft_ps": metrics[prefill_id].ttft_ps,
        "decode_tpot_ps": _fraction(metrics[decode_id].tpot_ps),
    }


def _completion_csv(result: Any) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(
        (
            "execution_id",
            "operation_id",
            "phase",
            "timestamp_ps",
            "subject_object_id",
            "completed_bytes",
        )
    )
    for event in result.events:
        writer.writerow(
            (
                event.execution_id,
                event.operation_id,
                event.phase.value,
                event.timestamp_ps,
                event.subject_object_id or "",
                "" if event.completed_bytes is None else event.completed_bytes,
            )
        )
    return output.getvalue().encode("utf-8")


def _make_bypass_artifacts(
    *,
    graph_json: dict[str, Any],
    step_json: dict[str, Any],
    execution: Any,
    step: Any,
    summary: dict[str, Any],
    rate_gbps: int,
) -> BypassArtifacts:
    graph_pretty = (json.dumps(graph_json, indent=2, sort_keys=True) + "\n").encode("utf-8")
    return BypassArtifacts(
        goal_text=graph_pretty,
        goal_binary=_canonical_bytes(graph_json),
        topology=b"",
        profile="core21-fixed-contended-graph",
        seed=0,
        baseline_parameters=canonical_bypass_parameters(
            {
                "authority_mode": "bypass",
                "link_rate_gbps": rate_gbps,
                "payload_bytes": graph_json["operations"][0]["work"]["payload_bytes"],
                "step_record_sha256": hashlib.sha256(_canonical_bytes(step_json)).hexdigest(),
                "wqe_count": 2,
            }
        ),
        completion_csv=_completion_csv(execution),
        canonical_completion=_canonical_bytes(execution_result_to_json(execution)),
        step_results=_canonical_bytes(_step_result_row(step)),
        replay_summary=_canonical_bytes(summary),
    )


def _bypass_artifacts_row(artifacts: BypassArtifacts) -> dict[str, Any]:
    return {
        "goal_text_hex": artifacts.goal_text.hex(),
        "goal_binary_hex": artifacts.goal_binary.hex(),
        "topology_hex": artifacts.topology.hex(),
        "profile": artifacts.profile,
        "seed": artifacts.seed,
        "baseline_parameters": [list(item) for item in artifacts.baseline_parameters],
        "completion_csv_hex": artifacts.completion_csv.hex(),
        "canonical_completion_hex": artifacts.canonical_completion.hex(),
        "step_results_hex": artifacts.step_results.hex(),
        "replay_summary_hex": artifacts.replay_summary.hex(),
    }


def bypass_artifacts_from_mode(mode_row: Mapping[str, Any]) -> BypassArtifacts:
    """Decode the standard bypass bundle emitted by this producer."""

    raw = mode_row.get("bypass_artifacts")
    if not isinstance(raw, dict):
        raise TypeError("mode row does not contain bypass artifacts")
    pairs = raw.get("baseline_parameters")
    if not isinstance(pairs, list):
        raise TypeError("bypass baseline parameters must be an array")
    return BypassArtifacts(
        goal_text=bytes.fromhex(raw["goal_text_hex"]),
        goal_binary=bytes.fromhex(raw["goal_binary_hex"]),
        topology=bytes.fromhex(raw["topology_hex"]),
        profile=raw["profile"],
        seed=raw["seed"],
        baseline_parameters=tuple((item[0], item[1]) for item in pairs),
        completion_csv=bytes.fromhex(raw["completion_csv_hex"]),
        canonical_completion=bytes.fromhex(raw["canonical_completion_hex"]),
        step_results=bytes.fromhex(raw["step_results_hex"]),
        replay_summary=bytes.fromhex(raw["replay_summary_hex"]),
    )


def _common_history(
    expectations: Mapping[str, Any], rate_gbps: int
) -> tuple[CompletionReducer, CompletionReducer, dict[str, Any]]:
    seed_graph = build_history_seed_graph(expectations)
    seed_record = build_history_seed_record(expectations)
    runtime = CoarseDeviceRuntime(
        _profile(rate_gbps),
        authority_mode=RnicAuthorityMode.BYPASS,
    )
    execution = runtime.execute(seed_graph)
    report = runtime.last_report
    if report is None:
        raise AuthorityProducerError("history seed omitted its runtime report")
    reducers = (
        CompletionReducer(VirtualClock(seed_graph.released_at_ps)),
        CompletionReducer(VirtualClock(seed_graph.released_at_ps)),
    )
    steps = tuple(
        reducer.reduce(seed_record, seed_graph, execution, report) for reducer in reducers
    )
    rows = tuple(_step_result_row(step) for step in steps)
    if rows[0] != rows[1]:
        raise AuthorityProducerError("history seed reducers did not agree")
    row = {
        "execution_graph": execution_graph_to_json(seed_graph),
        "step_record": step_record_to_json(seed_record),
        "execution_result": execution_result_to_json(execution),
        "runtime_report": _runtime_report_row(report),
        "step_result": rows[0],
    }
    return reducers[0], reducers[1], row


def _run_bypass(
    expectations: Mapping[str, Any],
    rate_gbps: int,
    reducer: CompletionReducer,
    graph: ExecutionGraph,
    record: StepRecord,
) -> tuple[dict[str, Any], BypassArtifacts]:
    runtime = CoarseDeviceRuntime(
        _profile(rate_gbps),
        authority_mode=RnicAuthorityMode.BYPASS,
    )
    execution = runtime.execute(graph)
    report = runtime.last_report
    if report is None:
        raise AuthorityProducerError("bypass execution omitted its runtime report")
    step = reducer.reduce(record, graph, execution, report)
    summary = _request_summary(step, expectations)
    graph_json = execution_graph_to_json(graph)
    step_json = step_record_to_json(record)
    artifacts = _make_bypass_artifacts(
        graph_json=graph_json,
        step_json=step_json,
        execution=execution,
        step=step,
        summary=summary,
        rate_gbps=rate_gbps,
    )
    row = {
        "hardware_mode": "bypass",
        "authority": runtime.authority_name,
        "execution_result": execution_result_to_json(execution),
        "runtime_report": _runtime_report_row(report),
        "step_result": _step_result_row(step),
        "request_summary": summary,
        "wqe_count": len(report.wqes),
        "bypass_artifacts": _bypass_artifacts_row(artifacts),
    }
    return row, artifacts


def run_bypass_replay(
    expectations: Mapping[str, Any], rate_gbps: int = 400
) -> tuple[dict[str, Any], BypassArtifacts]:
    """Rerun only the real bypass path for the post-negative guard."""

    bypass_reducer, _, _ = _common_history(expectations, rate_gbps)
    return _run_bypass(
        expectations,
        rate_gbps,
        bypass_reducer,
        build_fixed_graph(expectations),
        build_fixed_step_record(expectations),
    )


def run_authority_cell(
    cell: ComposedRnicCell,
    expectations: Mapping[str, Any],
) -> dict[str, Any]:
    """Run one native FIFO cell through both deployed authority paths."""

    if not isinstance(cell, ComposedRnicCell):
        raise TypeError("cell must be a ComposedRnicCell")
    graph = build_fixed_graph(expectations)
    record = build_fixed_step_record(expectations)
    bypass_reducer, structural_reducer, history = _common_history(expectations, cell.link_rate_gbps)
    bypass_row, _ = _run_bypass(
        expectations,
        cell.link_rate_gbps,
        bypass_reducer,
        graph,
        record,
    )

    session = ComposedRnicSession(
        cell,
        session_id=(f"core21-{cell.link_rate_gbps}g-{cell.doorbell_service_ps}ps"),
    )
    runtime = CoarseDeviceRuntime(
        _profile(cell.link_rate_gbps),
        authority_mode=RnicAuthorityMode.STRUCTURAL,
        native_session=session,
    )
    before = {
        "committed_transactions": session.committed_transactions,
        "committed_wqes": session.committed_wqes,
    }
    failure_type = ""
    failure_message = ""
    try:
        runtime.execute(_build_failure_graph(expectations))
    except ValueError as error:
        failure_type = type(error).__name__
        failure_message = str(error)
    else:
        raise AuthorityProducerError("incomplete native transaction unexpectedly committed")
    after = {
        "committed_transactions": session.committed_transactions,
        "committed_wqes": session.committed_wqes,
    }
    failure = {
        "exception_type": failure_type,
        "exception_message": failure_message,
        "before": before,
        "after": after,
        "retry": None,
        "runtime_last_report_is_none": runtime.last_report is None,
        "bypass_ledger_is_none": runtime.bypass_ledger is None,
        "authority": runtime.authority_name,
    }

    execution = runtime.execute(graph)
    report = runtime.last_report
    if report is None:
        raise AuthorityProducerError("structural execution omitted its runtime report")
    step = structural_reducer.reduce(record, graph, execution, report)
    failure["retry"] = {
        "committed_transactions": session.committed_transactions,
        "committed_wqes": session.committed_wqes,
    }
    structural_row = {
        "hardware_mode": "structural",
        "authority": runtime.authority_name,
        "execution_result": execution_result_to_json(execution),
        "runtime_report": _runtime_report_row(report),
        "step_result": _step_result_row(step),
        "request_summary": _request_summary(step, expectations),
        "wqe_count": len(report.wqes),
        "bypass_artifacts": None,
    }
    return {
        "link_rate_gbps": cell.link_rate_gbps,
        "doorbell_service_ps": cell.doorbell_service_ps,
        "execution_graph": execution_graph_to_json(graph),
        "step_record": step_record_to_json(record),
        "history_seed": history,
        "bypass": bypass_row,
        "structural": structural_row,
        "transaction_failure": failure,
    }


def _preflight_goal(path: Path) -> Path:
    trace = GoalTrace(32)
    trace.rank(0).send(4096, to=8, tag=21)
    trace.rank(8).recv(4096, source=0, tag=21)
    return trace.write(path)


def _composed_preflight(*, htsim_rnic: Path, txt2bin: Path, temporary: Path) -> dict[str, Any]:
    goal_text = _preflight_goal(temporary / "authority-preflight.goal")
    goal_binary = to_binary(
        goal_text,
        temporary / "authority-preflight.bin",
        tool=txt2bin,
    )
    completion_csv = temporary / "authority-preflight.csv"
    result = run_htsim_rnic(
        HtsimRnicConfig(
            goal_bin=goal_binary,
            profile="rnic-nn",
            linkspeed_bps=400_000_000_000,
            completion_csv=completion_csv,
            extra_flags={
                "-rnic_data_header_bytes": "0",
                "-rnic_nn_propagation_ps": "0",
            },
        ),
        binary=htsim_rnic,
    )
    structural = any(
        "hardware_mode=structural" in line and "wqe_authority=simllm-native-rnic-session" in line
        for line in result.manifest
    )
    if not structural:
        raise AuthorityProducerError("composed preflight lacks structural native authority")
    return {
        "profile": "rnic-nn",
        "manifest": result.manifest,
        "flow_count": len(result.flows),
        "quiescent": result.quiescent,
    }


def _load_json(path: Path, name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AuthorityProducerError(f"cannot read {name}: {error}") from error
    return _object(value, name)


def _publish(path: Path, value: dict[str, Any]) -> None:
    temporary = Path(f"{path}.tmp")
    if path.exists() or temporary.exists():
        raise FileExistsError("authority observation or temporary file already exists")
    serialized = json.dumps(value, indent=2, sort_keys=True) + "\n"
    try:
        temporary.write_text(serialized, encoding="utf-8", newline="\n")
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def produce(
    *,
    expectations_path: Path,
    observations_path: Path,
    tier_a_producer: Path,
    htsim_rnic: Path,
    txt2bin: Path,
) -> None:
    expectations = load_expectations(expectations_path)
    if observations_path.exists() or Path(f"{observations_path}.tmp").exists():
        raise FileExistsError("authority observations already exist")
    observations_path.parent.mkdir(parents=True, exist_ok=True)
    native_path = observations_path.with_name("native_observations.json")
    if native_path.exists():
        raise FileExistsError("native observations already exist")

    htsim_rnic = htsim_rnic.resolve(strict=True)
    txt2bin = txt2bin.resolve(strict=True)
    with tempfile.TemporaryDirectory(
        prefix=".authority-producer-",
        dir=observations_path.parent,
    ) as temporary_name:
        preflight = _composed_preflight(
            htsim_rnic=htsim_rnic,
            txt2bin=txt2bin,
            temporary=Path(temporary_name),
        )

    tier_a_producer = tier_a_producer.resolve(strict=True)
    invoke_composed_tier_a_producer(
        tier_a_producer,
        TIER_A_EXPECTATIONS,
        native_path,
    )
    native_raw = _load_json(native_path, "native Tier A observations")
    from examples.rnic_live_v1.tier_a_acceptance import (
        _validate_expectations as validate_tier_a_expectations,
    )
    from examples.rnic_live_v1.tier_a_acceptance import check_observations

    tier_a_expectations = _load_json(TIER_A_EXPECTATIONS, "Tier A expectations")
    validate_tier_a_expectations(tier_a_expectations)
    check_observations(native_raw, tier_a_expectations, "htsim")
    native = ComposedRnicObservations.from_json(native_raw)

    cells = [
        run_authority_cell(native.fifo[(rate, doorbell)], expectations)
        for rate in expectations["sweep"]["link_rate_gbps"]
        for doorbell in expectations["sweep"]["doorbell_service_ps"]
    ]
    observations = {
        "schema": expectations["observation_schema"],
        "simllm_source_commit": _git_head(ROOT),
        "htsim_source_commit": expectations["htsim_source_commit"],
        "native_observations_sha256": _sha256(native_path),
        "preflight": preflight,
        "cells": cells,
    }
    _publish(observations_path, observations)


def _absolute(path: Path, name: str) -> None:
    if not path.is_absolute():
        raise ValueError(f"{name} must be an absolute path")


def _parse_arguments(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expectations", required=True, type=Path)
    parser.add_argument("--observations", required=True, type=Path)
    parser.add_argument("--tier-a-producer", required=True, type=Path)
    parser.add_argument("--htsim-rnic", required=True, type=Path)
    parser.add_argument("--txt2bin", required=True, type=Path)
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parse_arguments(argv)
    try:
        expectations = load_expectations(arguments.expectations)
        for name in (
            "observations",
            "tier_a_producer",
            "htsim_rnic",
            "txt2bin",
        ):
            _absolute(getattr(arguments, name), name)
        if arguments.check_only:
            if arguments.observations.exists():
                raise FileExistsError("authority observations already exist")
            if expectations["producer_argument_names"] != [
                "--expectations",
                "--observations",
                "--tier-a-producer",
                "--htsim-rnic",
                "--txt2bin",
            ]:
                raise AuthorityProducerError("producer argument registry drifted")
            print("RNIC authority producer registry check passed; no artifacts produced")
            return 0
        produce(
            expectations_path=arguments.expectations.resolve(strict=True),
            observations_path=arguments.observations.resolve(strict=False),
            tier_a_producer=arguments.tier_a_producer,
            htsim_rnic=arguments.htsim_rnic,
            txt2bin=arguments.txt2bin,
        )
        return 0
    except AuthorityProducerError as error:
        print(str(error), file=sys.stderr)
        return 2
    except (OSError, RuntimeError, TypeError, ValueError, subprocess.SubprocessError) as error:
        print(f"authority producer error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
