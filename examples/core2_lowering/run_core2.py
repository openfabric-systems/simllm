"""Run the pre-registered CORE-2 lowering and graph-replay study.

The three timing paths are independent after lowering: the legacy step sink,
GOAL rendered only from a JSON-round-tripped execution graph, and frozen
closed forms from expectations.md. Raw artifacts belong on a data volume.

Usage:
    SIMLLM_HTSIM_RNIC=... SIMLLM_DATA_ROOT=... \
    python -m examples.core2_lowering.run_core2
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from simllm._local_config import path_from_env
from simllm.backends import (
    FlowCompletion,
    HtsimRnicConfig,
    HtsimStepSink,
    HtsimStepSinkConfig,
    SerialStepLowerer,
    SerialStepLowererConfig,
    parse_completion_csv,
    run_htsim_rnic,
)
from simllm.compute import ModelDims
from simllm.core import (
    CollectiveWork,
    ControlWork,
    DmaWork,
    KvCacheWork,
    RequestPhase,
    ScheduledRequest,
    StepRecord,
    execution_graph_from_json,
    execution_graph_to_json,
    validate_execution_graph,
)
from simllm.goal import to_binary
from simllm.traffic import render_serial_execution_graph_goal

G = 1_000_000_000
L = 2
HIDDEN = 1024
DTYPE_BYTES = 2
PAYLOAD_BYTES = 524_288
COMPUTE_ESTIMATE_PS = 24_061_074
PER_LAYER_CALC_NS = 12_030
CRITICAL_COMPUTE_PS = 24_060_000
PROPAGATION_PS = 2_000_000

SMALL_DIMS = ModelDims(
    num_layers=L,
    hidden_size=HIDDEN,
    intermediate_size=4096,
    num_heads=8,
    num_kv_heads=8,
    head_size=128,
    vocab_size=32000,
    dtype_bytes=DTYPE_BYTES,
)

SMALL_MOE_DIMS = ModelDims(
    num_layers=L,
    hidden_size=HIDDEN,
    intermediate_size=4096,
    num_heads=8,
    num_kv_heads=8,
    head_size=128,
    vocab_size=32000,
    dtype_bytes=DTYPE_BYTES,
    num_experts=8,
    top_k=2,
    moe_intermediate_size=512,
    local_num_experts=2,
)

FROZEN_DENSE: dict[tuple[int, int], tuple[int, int]] = {
    (2, 400): (82_003_040, 16),
    (2, 200): (123_946_080, 16),
    (4, 400): (134_974_560, 96),
    (4, 200): (197_889_120, 96),
}
FROZEN_MOE_JCT_PS = 25_811_524
FROZEN_MOE_FLOWS = 48


def prefill_record() -> StepRecord:
    return StepRecord(
        step_index=0,
        virtual_time_ps=0,
        scheduled=[
            ScheduledRequest(
                "prefill-0",
                RequestPhase.PREFILL,
                num_new_tokens=256,
                context_length=256,
            )
        ],
    )


def decode_record() -> StepRecord:
    return StepRecord(
        step_index=1,
        virtual_time_ps=0,
        scheduled=[
            ScheduledRequest(
                f"decode-{index}",
                RequestPhase.DECODE,
                num_new_tokens=1,
                context_length=128,
            )
            for index in range(2)
        ],
    )


def frozen_dense_jct_ps(world: int, rate_gbps: int) -> int:
    ps_per_byte = 8_000 // rate_gbps
    round_ps = (PAYLOAD_BYTES // world) * ps_per_byte + PROPAGATION_PS
    return CRITICAL_COMPUTE_PS + 2 * L * 2 * (world - 1) * round_ps


def serialization_only_ps(jct_ps: int, world: int) -> int:
    propagation = 2 * L * 2 * (world - 1) * PROPAGATION_PS
    return jct_ps - CRITICAL_COMPUTE_PS - propagation


def completion_ledger(flows: list[FlowCompletion]) -> list[tuple[object, ...]]:
    return sorted(
        (
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
        )
        for flow in flows
    )


def assert_wqe_accounting(flows: list[FlowCompletion], expected_count: int) -> None:
    assert len(flows) == expected_count
    required = (
        "wqe_id",
        "sq_id",
        "rq_id",
        "cq_id",
        "sq_post_sequence",
        "sq_dispatch_sequence",
        "cq_post_sequence",
        "cq_consume_sequence",
        "transport_kind",
        "transport_object_id",
    )
    assert all(getattr(flow, field_name) is not None for flow in flows for field_name in required)
    assert len({flow.flow_id for flow in flows}) == expected_count
    assert len({flow.wqe_id for flow in flows}) == expected_count
    assert len({(flow.flow_id, flow.wqe_id) for flow in flows}) == expected_count
    assert all(flow.sq_post_sequence == flow.sq_dispatch_sequence for flow in flows)
    assert all(flow.cq_post_sequence == flow.cq_consume_sequence for flow in flows)
    assert all(flow.transport_kind == "none" and flow.transport_object_id == 0 for flow in flows)

    for queue_field, sequence_field in (
        ("sq_id", "sq_post_sequence"),
        ("cq_id", "cq_post_sequence"),
    ):
        queue_ids = {getattr(flow, queue_field) for flow in flows}
        for queue_id in queue_ids:
            sequences = sorted(
                getattr(flow, sequence_field)
                for flow in flows
                if getattr(flow, queue_field) == queue_id
            )
            assert sequences == list(range(1, len(sequences) + 1))


def graph_replay(graph, workdir: Path, rate_gbps: int):
    workdir.mkdir(parents=True, exist_ok=True)
    payload = execution_graph_to_json(graph)
    graph_path = workdir / "execution-graph.json"
    graph_path.write_text(json.dumps(payload, indent=2) + "\n")
    replay_graph = execution_graph_from_json(json.loads(graph_path.read_text()))
    assert replay_graph == graph
    validate_execution_graph(replay_graph)
    trace = render_serial_execution_graph_goal(replay_graph)
    goal_path = trace.write(workdir / "graph-replay.goal")
    run = run_htsim_rnic(
        HtsimRnicConfig(
            goal_bin=to_binary(goal_path),
            profile="rnic-nn-fluid",
            linkspeed_bps=rate_gbps * G,
            completion_csv=workdir / "graph-replay.csv",
        )
    )
    return run, json.dumps(payload, sort_keys=True, separators=(",", ":"))


def legacy_replay(
    record: StepRecord,
    dims: ModelDims,
    tp_ranks: tuple[int, ...],
    ep_ranks: tuple[int, ...] | None,
    workdir: Path,
    rate_gbps: int,
):
    sink = HtsimStepSink(
        HtsimStepSinkConfig(
            profile="rnic-nn-fluid",
            tp_ranks=tp_ranks,
            ep_ranks=ep_ranks,
            dims=dims,
            workdir=workdir,
            linkspeed_bps=rate_gbps * G,
        )
    )
    result = sink(record)
    assert result is not None
    outcome = sink.outcomes[-1]
    csv_path = workdir / f"step-{record.step_index:06d}.rnic-nn-fluid.csv"
    return result, outcome, parse_completion_csv(csv_path)


def assert_graph_payloads_are_serial_compatibility_only(graph) -> None:
    for operation in graph.operations:
        assert not isinstance(operation.work, (KvCacheWork, DmaWork, ControlWork))
        if isinstance(operation.work, CollectiveWork):
            assert operation.work.algorithm_hint in {"ring", "pairwise"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=Path,
    )
    args = parser.parse_args()
    if args.out is None:
        data_root = path_from_env("SIMLLM_DATA_ROOT")
        if data_root is None:
            parser.error("--out is required when SIMLLM_DATA_ROOT is not set")
        args.out = data_root / "core2_lowering"
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    graph_json_by_world: dict[int, str] = {}
    dense_jct: dict[tuple[int, int], int] = {}

    def emit(**row: object) -> None:
        rows.append(row)
        print(" ".join(f"{key}={value}" for key, value in row.items()))

    record = prefill_record()
    for world in (2, 4):
        ranks = tuple(range(world))
        lowerer = SerialStepLowerer(SerialStepLowererConfig(dims=SMALL_DIMS, tp_ranks=ranks))
        graph = lowerer.lower(record)
        validate_execution_graph(graph)
        assert_graph_payloads_are_serial_compatibility_only(graph)

        for rate_gbps in (400, 200):
            cell = out / f"dense-w{world}-r{rate_gbps}g"
            legacy, outcome, legacy_flows = legacy_replay(
                record,
                SMALL_DIMS,
                ranks,
                None,
                cell / "legacy",
                rate_gbps,
            )
            replay, canonical_json = graph_replay(graph, cell / "graph", rate_gbps)
            prior_json = graph_json_by_world.setdefault(world, canonical_json)
            assert canonical_json == prior_json
            expected_ps, expected_flows = FROZEN_DENSE[(world, rate_gbps)]
            assert frozen_dense_jct_ps(world, rate_gbps) == expected_ps
            graph_jct_ps = replay.job_completion_time_ps()
            assert outcome.compute_estimate_ps == COMPUTE_ESTIMATE_PS
            assert outcome.per_layer_calc_ns == PER_LAYER_CALC_NS
            assert legacy.step_latency_ps == expected_ps
            assert graph_jct_ps == expected_ps
            assert completion_ledger(legacy_flows) == completion_ledger(replay.flows)
            assert len(legacy_flows) == len(replay.flows) == expected_flows
            assert_wqe_accounting(legacy_flows, expected_flows)
            assert_wqe_accounting(replay.flows, expected_flows)
            assert replay.quiescent
            dense_jct[(world, rate_gbps)] = graph_jct_ps
            emit(
                check="dense",
                world=world,
                rate_gbps=rate_gbps,
                legacy_jct_ps=legacy.step_latency_ps,
                graph_jct_ps=graph_jct_ps,
                expected_jct_ps=expected_ps,
                legacy_residual_ps=legacy.step_latency_ps - expected_ps,
                graph_residual_ps=graph_jct_ps - expected_ps,
                path_residual_ps=graph_jct_ps - legacy.step_latency_ps,
                flows=len(replay.flows),
                ledger_equal=True,
                wqe_ledger_equal=True,
                sq_post_dispatch_exact=True,
                cq_immediate_consume=True,
                transport_kind="none",
                graph_equal_across_rates=True,
                physical_quiescence=replay.quiescent,
            )

    for world in (2, 4):
        q200 = serialization_only_ps(dense_jct[(world, 200)], world)
        q400 = serialization_only_ps(dense_jct[(world, 400)], world)
        assert q200 == 2 * q400
    for rate_gbps in (200, 400):
        q2 = serialization_only_ps(dense_jct[(2, rate_gbps)], 2)
        q4 = serialization_only_ps(dense_jct[(4, rate_gbps)], 4)
        assert 2 * q4 == 3 * q2
    assert all(dense_jct[(world, 400)] * 2 != dense_jct[(world, 200)] for world in (2, 4))

    moe_record = decode_record()
    moe_tp = (0,)
    moe_ep = (0, 1, 2, 3)
    moe_lowerer = SerialStepLowerer(
        SerialStepLowererConfig(dims=SMALL_MOE_DIMS, tp_ranks=moe_tp, ep_ranks=moe_ep)
    )
    moe_graph = moe_lowerer.lower(moe_record)
    assert_graph_payloads_are_serial_compatibility_only(moe_graph)
    moe_dir = out / "moe-ep4-r400g"
    legacy, _, legacy_flows = legacy_replay(
        moe_record,
        SMALL_MOE_DIMS,
        moe_tp,
        moe_ep,
        moe_dir / "legacy",
        400,
    )
    replay, _ = graph_replay(moe_graph, moe_dir / "graph", 400)
    graph_jct_ps = replay.job_completion_time_ps()
    assert legacy.step_latency_ps == graph_jct_ps == FROZEN_MOE_JCT_PS
    assert len(legacy_flows) == len(replay.flows) == FROZEN_MOE_FLOWS
    assert completion_ledger(legacy_flows) == completion_ledger(replay.flows)
    assert_wqe_accounting(legacy_flows, FROZEN_MOE_FLOWS)
    assert_wqe_accounting(replay.flows, FROZEN_MOE_FLOWS)
    emit(
        check="moe",
        world=4,
        rate_gbps=400,
        legacy_jct_ps=legacy.step_latency_ps,
        graph_jct_ps=graph_jct_ps,
        expected_jct_ps=FROZEN_MOE_JCT_PS,
        legacy_residual_ps=legacy.step_latency_ps - FROZEN_MOE_JCT_PS,
        graph_residual_ps=graph_jct_ps - FROZEN_MOE_JCT_PS,
        path_residual_ps=graph_jct_ps - legacy.step_latency_ps,
        flows=len(replay.flows),
        ledger_equal=True,
        wqe_ledger_equal=True,
        sq_post_dispatch_exact=True,
        cq_immediate_consume=True,
        transport_kind="none",
        graph_equal_across_rates="not-applicable",
        physical_quiescence=replay.quiescent,
    )

    fields: list[str] = []
    for row in rows:
        fields.extend(key for key in row if key not in fields)
    with open(out / "summary.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} rows -> {out / 'summary.csv'}")


if __name__ == "__main__":
    main()
