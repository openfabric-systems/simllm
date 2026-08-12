"""Captured-routing and placement-epoch traffic tests."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

import simllm.backends.step_sink as step_sink_module
from simllm.backends import (
    HtsimStepSink,
    HtsimStepSinkConfig,
    ObservedStepLowerer,
    SerialStepLowerer,
    SerialStepLowererConfig,
)
from simllm.compute import ComputeProvider, DurationEstimate, ModelDims
from simllm.core import (
    CollectiveWork,
    ExecutionObservations,
    RequestBookkeeper,
    RequestPhase,
    ScheduledRequest,
    StepRecord,
)
from simllm.placement import PlacementManifest, RankPlacement
from simllm.preplay import (
    PREPLAY_TRACE_SCHEMA,
    ForwardPhase,
    RequestArrival,
    RoutedExperts,
    RoutedLayer,
    RoutedRequest,
    RoutedToken,
    join_preplay_arrivals,
    project_preplay_routing,
)
from simllm.traffic import (
    ExpertPlacementSnapshot,
    RoutedMoeSupply,
    render_step_goal,
    step_moe_alltoalls,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_GRANITE_FIXTURE = _REPO_ROOT / "examples/preplay_trace_v1/granite_length_cap.jsonl"

TINY_MOE_DIMS = ModelDims(
    num_layers=2,
    hidden_size=4,
    intermediate_size=8,
    num_heads=2,
    num_kv_heads=2,
    head_size=2,
    vocab_size=16,
    dtype_bytes=2,
    num_experts=4,
    top_k=2,
    moe_intermediate_size=4,
    local_num_experts=2,
)

GRANITE_MOE_DIMS = ModelDims(
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
    local_num_experts=16,
)

_EPOCH1_LAYER13_RANK0 = {
    2,
    7,
    8,
    10,
    11,
    13,
    14,
    15,
    16,
    18,
    20,
    21,
    22,
    24,
    25,
    27,
}


class FixedProvider(ComputeProvider):
    def __init__(self, duration_ps: int):
        self.duration_ps = duration_ps

    def estimate(self, kernel, gpu):
        return DurationEstimate(duration_ps=self.duration_ps, bound="measured")


def _token(
    phase: ForwardPhase,
    token_index: int,
    token_id: int,
    layer0: tuple[int, int],
    layer1: tuple[int, int],
) -> RoutedToken:
    return RoutedToken(
        phase=phase,
        token_index=token_index,
        token_id=token_id,
        layers=(
            RoutedLayer(layer_index=0, expert_ids=layer0),
            RoutedLayer(layer_index=1, expert_ids=layer1),
        ),
    )


def _routing() -> RoutedExperts:
    return RoutedExperts(
        trace_schema=PREPLAY_TRACE_SCHEMA,
        trace_sha256="a" * 64,
        expert_count=4,
        top_k=2,
        moe_layer_indices=(0, 1),
        requests=(
            RoutedRequest(
                request_id="alpha",
                prompt_token_count=2,
                output_token_count=3,
                tokens=(
                    _token(ForwardPhase.PREFILL, 0, 10, (2, 3), (0, 2)),
                    _token(ForwardPhase.PREFILL, 1, 11, (0, 2), (0, 1)),
                    _token(ForwardPhase.DECODE, 0, 20, (1, 3), (2, 3)),
                    _token(ForwardPhase.DECODE, 1, 21, (0, 1), (1, 2)),
                ),
            ),
        ),
    )


def _tiny_manifest(epoch: int) -> PlacementManifest:
    if epoch == 0:
        rank0 = {0: [0, 1], 1: [0, 1]}
        rank1 = {0: [2, 3], 1: [2, 3]}
    else:
        rank0 = {0: [0, 2], 1: [0, 1]}
        rank1 = {0: [1, 3], 1: [2, 3]}
    return PlacementManifest(
        ranks=[
            RankPlacement(
                global_rank=0,
                hostname="host",
                local_rank=0,
                local_expert_ids=rank0,
                placement_epoch=epoch,
            ),
            RankPlacement(
                global_rank=1,
                hostname="host",
                local_rank=1,
                local_expert_ids=rank1,
                placement_epoch=epoch,
            ),
        ]
    )


def _tiny_supply() -> RoutedMoeSupply:
    return RoutedMoeSupply(
        routed_experts=_routing(),
        placements=tuple(
            ExpertPlacementSnapshot.from_manifest(_tiny_manifest(epoch), (0, 1))
            for epoch in (0, 1)
        ),
        step_placement_epochs=((0, 0), (1, 1), (2, 0), (3, 0)),
    )


def _prefill_record(step_index: int = 0, *, count: int = 2, context: int = 2):
    return StepRecord(
        step_index=step_index,
        virtual_time_ps=0,
        scheduled=[
            ScheduledRequest(
                "alpha",
                RequestPhase.PREFILL,
                count,
                context_length=context,
            )
        ],
        num_sampled=1,
    )


def _granite_manifest(epoch: int) -> PlacementManifest:
    local_by_rank = ({}, {})
    for layer in range(24):
        rank0 = (
            _EPOCH1_LAYER13_RANK0
            if epoch == 1 and layer == 13
            else set(range(16))
        )
        local_by_rank[0][layer] = [expert for expert in range(32) if expert in rank0]
        local_by_rank[1][layer] = [expert for expert in range(32) if expert not in rank0]
    return PlacementManifest(
        ranks=[
            RankPlacement(
                global_rank=rank,
                hostname="host",
                local_rank=rank,
                local_expert_ids=local_by_rank[rank],
                placement_epoch=epoch,
            )
            for rank in (0, 1)
        ]
    )


def _granite_supply() -> RoutedMoeSupply:
    run = join_preplay_arrivals(
        (RequestArrival(request_id="length-cap", arrived_at_ps=0),),
        _GRANITE_FIXTURE,
        RequestBookkeeper(),
    )
    return RoutedMoeSupply(
        routed_experts=project_preplay_routing(run),
        placements=tuple(
            ExpertPlacementSnapshot.from_manifest(
                _granite_manifest(epoch),
                (0, 1),
            )
            for epoch in (0, 1)
        ),
        step_placement_epochs=((0, 0), (1, 1)),
    )


def _granite_record(step_index: int) -> StepRecord:
    return StepRecord(
        step_index=step_index,
        virtual_time_ps=0,
        scheduled=[
            ScheduledRequest(
                "length-cap",
                RequestPhase.PREFILL,
                22,
                context_length=22,
            )
        ],
        num_sampled=1,
    )


def test_manifest_snapshot_and_supply_select_explicit_epochs():
    snapshot = ExpertPlacementSnapshot.from_manifest(_tiny_manifest(0), (0, 1))
    supply = _tiny_supply()

    assert snapshot.placement_epoch == 0
    assert snapshot.owner_map()[(0, 3)] == 1
    assert supply.placement_for_step(0).placement_epoch == 0
    assert supply.placement_for_step(1).placement_epoch == 1
    with pytest.raises(ValueError, match="no placement epoch for step 99"):
        supply.placement_for_step(99)


def test_captured_expansion_deduplicates_destinations_and_transposes_combine():
    ops = step_moe_alltoalls(
        _prefill_record(),
        TINY_MOE_DIMS,
        (0, 1),
        routed_supply=_tiny_supply(),
    )

    assert [(op.layer, op.phase, op.placement_epoch) for op in ops] == [
        (0, "dispatch", 0),
        (0, "combine", 0),
        (1, "dispatch", 0),
        (1, "combine", 0),
    ]
    assert ops[0].per_pair_bytes == 0
    assert ops[0].pair_payload_bytes == ((0, 1, 16), (1, 0, 8))
    assert ops[1].pair_payload_bytes == ((0, 1, 8), (1, 0, 16))
    assert ops[2].pair_payload_bytes == ((0, 1, 8), (1, 0, 16))
    assert ops[3].pair_payload_bytes == ((0, 1, 16), (1, 0, 8))


def test_epoch_change_and_phase_slices_change_only_selected_routes():
    epoch1 = step_moe_alltoalls(
        _prefill_record(1),
        TINY_MOE_DIMS,
        (0, 1),
        routed_supply=_tiny_supply(),
    )
    first_prefill = step_moe_alltoalls(
        _prefill_record(3, count=1, context=1),
        TINY_MOE_DIMS,
        (0, 1),
        routed_supply=_tiny_supply(),
    )
    decode = StepRecord(
        step_index=2,
        virtual_time_ps=0,
        scheduled=[
            ScheduledRequest(
                "alpha",
                RequestPhase.DECODE,
                1,
                context_length=3,
            )
        ],
    )
    first_decode = step_moe_alltoalls(
        decode,
        TINY_MOE_DIMS,
        (0, 1),
        routed_supply=_tiny_supply(),
    )

    assert epoch1[0].placement_epoch == 1
    assert epoch1[0].pair_payload_bytes == ((0, 1, 8), (1, 0, 16))
    assert first_prefill[0].pair_payload_bytes == ((0, 1, 8),)
    assert first_decode[0].pair_payload_bytes == ((0, 1, 8), (1, 0, 8))


def test_granite_fixture_matches_frozen_epoch_pair_tables():
    supply = _granite_supply()
    operations = {
        epoch: step_moe_alltoalls(
            _granite_record(epoch),
            GRANITE_MOE_DIMS,
            (0, 1),
            routed_supply=supply,
        )
        for epoch in (0, 1)
    }

    for epoch, ops in operations.items():
        assert len(ops) == 48
        dispatch = [op for op in ops if op.phase == "dispatch"]
        assert all(op.placement_epoch == epoch for op in ops)
        assert sum(max(entry[2] for entry in op.pair_payload_bytes) for op in dispatch) == (
            1_081_344 if epoch == 0 else 1_077_248
        )
        for layer, op in enumerate(dispatch):
            expected = ((0, 1, 45_056), (1, 0, 45_056))
            if layer == 2:
                expected = ((0, 1, 45_056), (1, 0, 43_008))
            if epoch == 1 and layer == 13:
                expected = ((0, 1, 40_960), (1, 0, 38_912))
            assert op.pair_payload_bytes == expected


def test_render_and_lowerer_carry_the_same_sparse_tables_and_epoch():
    supply = _tiny_supply()
    record = _prefill_record(1)
    trace = render_step_goal(
        record,
        TINY_MOE_DIMS,
        (0,),
        1,
        ep_ranks=(0, 1),
        routed_supply=supply,
    )
    graph = SerialStepLowerer(
        SerialStepLowererConfig(
            TINY_MOE_DIMS,
            (0,),
            ep_ranks=(0, 1),
            provider=FixedProvider(2_000),
            routed_moe_supply=supply,
        )
    ).lower(record)
    collectives = [
        operation
        for operation in graph.operations
        if isinstance(operation.work, CollectiveWork)
    ]

    assert "send 8b to 1 tag 1000" in trace.render()
    assert "send 16b to 0 tag 1000" in trace.render()
    assert [operation.placement_epoch for operation in collectives] == [1, 1, 1, 1]
    assert collectives[0].work.payload_bytes == 0
    assert collectives[0].work.pair_payload_bytes == (
        (0, 1, 8),
        (1, 0, 16),
    )


def test_observed_lowerer_binds_captured_pair_tables_and_epoch():
    supply = _tiny_supply()
    record = _prefill_record(1)
    config = SerialStepLowererConfig(
        TINY_MOE_DIMS,
        (0,),
        ep_ranks=(0, 1),
        provider=FixedProvider(2_000),
        routed_moe_supply=supply,
    )
    serial = SerialStepLowerer(config).lower(record)
    marker_operations = tuple(
        replace(
            operation,
            work=replace(operation.work, algorithm_hint=None),
            placement_epoch=0,
        )
        if isinstance(operation.work, CollectiveWork)
        else operation
        for operation in serial.operations
    )

    observed = ObservedStepLowerer(config).lower(
        record,
        ExecutionObservations(marker_operations, serial.completion_operation_ids),
    )

    assert observed == serial


def test_step_sink_uses_captured_supply_and_reports_authority(tmp_path, monkeypatch):
    monkeypatch.setattr(step_sink_module, "to_binary", lambda path: path)
    monkeypatch.setattr(
        step_sink_module,
        "run_htsim_rnic",
        lambda _config: SimpleNamespace(
            job_completion_time_ps=lambda: 123_456,
            flows=[object()] * 8,
            quiescent=True,
        ),
    )
    sink = HtsimStepSink(
        HtsimStepSinkConfig(
            profile="rnic-nn-fluid",
            tp_ranks=(0,),
            dims=TINY_MOE_DIMS,
            workdir=tmp_path,
            ep_ranks=(0, 1),
            provider=FixedProvider(2_000),
            routed_moe_supply=_tiny_supply(),
        )
    )

    result = sink(_prefill_record(1))

    assert result is not None
    assert result.step_latency_ps == 123_456
    assert sink.outcomes[0].routing_mode == "captured"
    assert sink.outcomes[0].placement_epoch == 1
    goal = (tmp_path / "step-000001.goal").read_text()
    assert "send 8b to 1 tag 1000" in goal
    assert "send 16b to 0 tag 1000" in goal


def test_step_sink_reports_no_realized_moe_traffic_with_supply(tmp_path, monkeypatch):
    monkeypatch.setattr(step_sink_module, "to_binary", lambda path: path)
    monkeypatch.setattr(
        step_sink_module,
        "run_htsim_rnic",
        lambda _config: SimpleNamespace(
            job_completion_time_ps=lambda: 123_456,
            flows=[object()] * 8,
            quiescent=True,
        ),
    )
    dense_dims = replace(
        TINY_MOE_DIMS,
        num_experts=0,
        top_k=0,
        moe_intermediate_size=None,
        local_num_experts=0,
    )
    sink = HtsimStepSink(
        HtsimStepSinkConfig(
            profile="rnic-nn-fluid",
            tp_ranks=(0, 1),
            dims=dense_dims,
            workdir=tmp_path,
            ep_ranks=(0, 1),
            provider=FixedProvider(2_000),
            routed_moe_supply=_tiny_supply(),
        )
    )

    assert sink(_prefill_record(1)) is not None
    assert sink.outcomes[0].routing_mode == "none"
    assert sink.outcomes[0].placement_epoch is None


def test_uniform_path_keeps_frozen_goal_bytes():
    record = _granite_record(0)
    payload = render_step_goal(
        record,
        GRANITE_MOE_DIMS,
        (0,),
        1,
        ep_ranks=(0, 1),
    ).render().encode()
    operations = step_moe_alltoalls(record, GRANITE_MOE_DIMS, (0, 1))

    assert len(payload) == 13_200
    assert hashlib.sha256(payload).hexdigest() == (
        "d708e998685b617478e891b316728d14b8ac6185a62b73817f80af1c5adff518"
    )
    assert all(operation.per_pair_bytes == 180_224 for operation in operations)
    assert all(operation.pair_payload_bytes == () for operation in operations)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ("expert-count", "disagrees with model num_experts"),
        ("top-k", "disagrees with model top_k"),
        ("layers", "disagree with model layers"),
        ("missing-owner", "missing owner"),
        ("outside-rank", "outside ep_ranks"),
        ("missing-request", "absent from routed-experts projection"),
        ("prefill-slice", "outside captured count"),
        ("decode-slice", "outside captured count"),
    ],
)
def test_captured_expansion_rejects_inconsistent_inputs(change, message):
    supply = _tiny_supply()
    dims = TINY_MOE_DIMS
    record = _prefill_record()
    if change == "expert-count":
        dims = replace(dims, num_experts=5)
    elif change == "top-k":
        dims = replace(dims, top_k=3)
    elif change == "layers":
        dims = replace(dims, num_layers=3)
    elif change in {"missing-owner", "outside-rank"}:
        placement = supply.placements[0]
        entries = placement.expert_owners
        if change == "missing-owner":
            entries = entries[1:]
        else:
            entries = (*entries[:-1], (*entries[-1][:2], 2))
        changed = ExpertPlacementSnapshot(
            placement_epoch=0,
            expert_owners=entries,
        )
        supply = replace(supply, placements=(changed, supply.placements[1]))
    elif change == "missing-request":
        record.scheduled[0].request_id = "missing"
    elif change == "prefill-slice":
        record.scheduled[0].context_length = 3
    elif change == "decode-slice":
        record.scheduled[0].phase = RequestPhase.DECODE
        record.scheduled[0].context_length = 99

    with pytest.raises(ValueError, match=message):
        step_moe_alltoalls(
            record,
            dims,
            (0, 1),
            routed_supply=supply,
        )


def test_supply_and_manifest_snapshot_reject_invalid_epoch_state():
    first = ExpertPlacementSnapshot.from_manifest(_tiny_manifest(0), (0, 1))
    second = ExpertPlacementSnapshot.from_manifest(_tiny_manifest(1), (0, 1))

    with pytest.raises(ValueError, match="duplicate placement_epoch"):
        RoutedMoeSupply(
            routed_experts=_routing(),
            placements=(first, first),
            step_placement_epochs=((0, 0),),
        )
    with pytest.raises(ValueError, match="sorted by step index"):
        RoutedMoeSupply(
            routed_experts=_routing(),
            placements=(first, second),
            step_placement_epochs=((1, 1), (0, 0)),
        )
    with pytest.raises(ValueError, match="has no snapshot"):
        RoutedMoeSupply(
            routed_experts=_routing(),
            placements=(first, second),
            step_placement_epochs=((0, 7),),
        )

    manifest = _tiny_manifest(0)
    manifest.ranks[1].placement_epoch = 1
    with pytest.raises(ValueError, match="disagree on placement_epoch"):
        ExpertPlacementSnapshot.from_manifest(manifest, (0, 1))


def test_drain_dense_and_single_rank_bypasses_remain_empty():
    supply = _tiny_supply()
    drain = StepRecord(step_index=99, virtual_time_ps=0)
    dense = replace(
        TINY_MOE_DIMS,
        num_experts=0,
        top_k=0,
        moe_intermediate_size=None,
        local_num_experts=0,
    )

    assert step_moe_alltoalls(
        drain,
        TINY_MOE_DIMS,
        (0, 1),
        routed_supply=supply,
    ) == []
    assert step_moe_alltoalls(
        _prefill_record(),
        dense,
        (0, 1),
        routed_supply=supply,
    ) == []
    assert step_moe_alltoalls(
        _prefill_record(),
        TINY_MOE_DIMS,
        (0,),
        routed_supply=supply,
    ) == []


def test_observed_lowerer_preserves_per_request_attribution():
    """The observed path must carry request attribution, not just pair totals.

    The traffic planner owns collective semantics, so a schedule observed from
    a framework carries no request attribution of its own. If the planner does
    not rebind it, per-request fidelity silently disappears on exactly the path
    a live adapter will use.
    """

    supply = _tiny_supply()
    record = _prefill_record(1)
    config = SerialStepLowererConfig(
        TINY_MOE_DIMS,
        (0,),
        ep_ranks=(0, 1),
        provider=FixedProvider(2_000),
        routed_moe_supply=supply,
    )
    serial = SerialStepLowerer(config).lower(record)
    bare_markers = tuple(
        replace(
            operation,
            work=replace(
                operation.work,
                algorithm_hint=None,
                request_pair_payload_bytes=(),
            ),
            placement_epoch=0,
        )
        if isinstance(operation.work, CollectiveWork)
        else operation
        for operation in serial.operations
    )

    observed = ObservedStepLowerer(config).lower(
        record,
        ExecutionObservations(bare_markers, serial.completion_operation_ids),
    )

    attributed = [
        operation
        for operation in observed.operations
        if isinstance(operation.work, CollectiveWork)
        and operation.work.request_pair_payload_bytes
    ]
    assert attributed, "observed lowering dropped every per-request attribution"
    for expected, actual in zip(serial.operations, observed.operations, strict=True):
        if isinstance(expected.work, CollectiveWork):
            assert (
                actual.work.request_pair_payload_bytes
                == expected.work.request_pair_payload_bytes
            )
