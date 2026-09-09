"""Routed demand, causal dispatch and per-invocation minimum controls."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from examples.routed_compute_v1.inputs import GPU, campaign_config, specimen, step
from simllm.backends import DeviceRuntimeStepSink, ObservedStepLowerer, SerialStepLowerer
from simllm.compute import (
    ComputeProvider,
    DurationEstimate,
    HostInitiationModel,
    KernelSpec,
    ModelDims,
    ProfileTableProvider,
    RooflineProvider,
    RoutedComputeConfig,
    grouped_expert_kernels,
)
from simllm.core import (
    CoarseDeviceProfile,
    CoarseDeviceRuntime,
    CollectiveWork,
    ComputeWork,
    ExecutionObservations,
    JoinProvenance,
    RequestBookkeeper,
    RequestLifetimeRegistry,
    RequestPhase,
    ResourceKind,
    RoutingViewDescriptor,
    ScheduledRequest,
    StepRecord,
    VirtualClock,
)
from simllm.core.execution_io import execution_graph_to_json
from simllm.core.step_io import step_result_to_json
from simllm.preplay import (
    RequestArrival,
    build_routing_arena,
    join_preplay_arrivals,
    project_preplay_routing,
)
from simllm.traffic import (
    ExpertPlacementSnapshot,
    RoutedMoeSupply,
    step_moe_alltoalls,
    step_routed_moe_work,
)


def _sink(config):
    sink = DeviceRuntimeStepSink(config, runtime=CoarseDeviceRuntime(
        CoarseDeviceProfile(rnic_rate_bps=8_000_000_000_000, nvlink_rate_bps=8_000_000_000_000)
    ))
    sink.bind_clock(VirtualClock())
    return sink


def test_two_local_experts_share_one_vector_but_keep_both_compute_rows():
    cfg = specimen(((2, 3), (2, 3), (0, 2)))
    record = step(cfg, 0)
    layers = step_routed_moe_work(record, cfg.dims, cfg.ep_ranks, cfg.routed_moe_supply)
    old = step_moe_alltoalls(record, cfg.dims, cfg.ep_ranks, routed_supply=cfg.routed_moe_supply)
    for layer in layers:
        assert layer.expert_rows(0) == ((0, 1),)
        assert layer.expert_rows(3) == ((2, 3), (3, 2))
        assert len(layer.expert_assignments) == 6
        assert layer.dispatch.pair_payload_bytes == ((0, 3, 96),)
        assert (layer.dispatch, layer.combine) == tuple(old[2 * layer.layer:2 * layer.layer + 2])
        with pytest.raises(ValueError, match="outside"):
            layer.expert_rows(1)


def test_active_weights_and_histogram_are_exact_without_inactive_residents():
    cfg = specimen(((2, 3), (2, 3), (0, 2)))
    up, down = grouped_expert_kernels(cfg.dims, layer=0, rank=3, expert_rows=((2, 3), (3, 2)))
    assert (up.flops, down.flops) == (4 * 16 * 32 * 5, 2 * 16 * 32 * 5)
    assert (up.bytes_moved, down.bytes_moved) == (2 * 16 * 32 * 2 * 2, 16 * 32 * 2 * 2)
    assert dict(up.config)["expert_2_rows"] == 3
    assert dict(up.config)["expert_3_rows"] == 2
    changed_residency = replace(cfg.dims, local_num_experts=4)
    assert grouped_expert_kernels(changed_residency, layer=0, rank=3, expert_rows=((2, 3), (3, 2))) == (up, down)


def test_minimum_applies_to_two_grouped_invocations_not_experts_or_rows():
    cfg = specimen(((0, 1), (0, 1), (0, 1)), minimum_ps=20_000_000)
    kernels = grouped_expert_kernels(cfg.dims, layer=0, rank=0, expert_rows=((0, 3), (1, 3)))
    prices = [cfg.routed_compute.estimate(cfg.provider, kernel, GPU) for kernel in kernels]
    assert [price.duration_ps for price in prices] == [20_000_000, 20_000_000]
    assert all(price.bound == "declared-gemm-minimum" for price in prices)
    off = RoutedComputeConfig()
    assert all(off.estimate(cfg.provider, kernel, GPU) == cfg.provider.estimate(kernel, GPU) for kernel in kernels)


def test_profile_table_uses_same_maximum_and_retains_missing_key_rejection():
    cfg = specimen(((2,),))
    kernels = grouped_expert_kernels(cfg.dims, layer=0, rank=3, expert_rows=((2, 1),))
    provider = ProfileTableProvider({(k.name, k.config, GPU.name): 7_000_000 for k in kernels})
    mode = RoutedComputeConfig(5_000_000, GPU)
    assert mode.estimate(provider, kernels[0], GPU).duration_ps == 7_000_000
    assert RoutedComputeConfig(9_000_000, GPU).estimate(provider, kernels[0], GPU).duration_ps == 9_000_000
    with pytest.raises(KeyError):
        mode.estimate(ProfileTableProvider({}), kernels[0], GPU)


def test_runtime_expert_work_follows_dispatch_and_drives_actual_request_metrics():
    cfg = campaign_config(8, "hot", 5_000_000)
    sink = _sink(cfg)
    for index in range(3):
        result = sink(step(cfg, index, sink.clock.now_ps), None)
        outcome = sink.outcomes[-1]
        timings = {row.operation_id: row for row in outcome.runtime_report.operations}
        operations = {row.operation_id: row for row in outcome.graph.operations}
        assert len(result.request_metrics) == 8
        for layer in range(2):
            prefix = f"step-{index}:layer-{layer}"
            dispatch = timings[f"{prefix}:ep-dispatch"]
            up = timings[f"{prefix}:rank-3:moe_gate_up"]
            down = timings[f"{prefix}:rank-3:moe_down"]
            combine = timings[f"{prefix}:ep-combine"]
            assert up.eligible_at_ps >= dispatch.completed_at_ps
            assert down.eligible_at_ps >= up.completed_at_ps
            assert combine.eligible_at_ps >= down.completed_at_ps
            assert all(
                visit.started_at_ps >= timings[visit.operation_id].eligible_at_ps
                for visit in outcome.runtime_report.visits
                if visit.resource.kind is ResourceKind.GPU_WORK_QUEUE
            )
            assert not any(
                row.rank == 0 and isinstance(row.work, ComputeWork)
                and row.work.kernel in ("moe_gate_up", "moe_down")
                for row in operations.values()
            )
            assert operations[f"{prefix}:rank-3:moe_gate_up"].placement_epoch == 0
        assert all(row.completed_at_ps == result.completed_at_ps for row in result.request_metrics)
        if index:
            assert all(row.tpot_ps is not None for row in result.request_metrics)
        else:
            assert all(row.ttft_ps == result.completed_at_ps for row in result.request_metrics)


def test_local_only_route_has_no_fabric_and_still_computes_every_row():
    cfg = specimen(((0, 1), (0, 1)), minimum_ps=5_000_000)
    sink = _sink(cfg)
    result = sink(step(cfg, 0), None)
    graph = sink.outcomes[0].graph
    assert result.completed_at_ps > 0
    assert not any(isinstance(op.work, CollectiveWork) for op in graph.operations)
    experts = [op for op in graph.operations if isinstance(op.work, ComputeWork) and op.work.kernel.startswith("moe_") and op.work.kernel != "moe_pre_dispatch"]
    assert len(experts) == 4
    assert all(op.rank == 0 and dict(op.work.config)["routed_rows"] == 4 for op in experts)


def test_placement_epoch_drives_traffic_and_compute_together():
    cfg = specimen(((2,), (2,)))
    original = cfg.routed_moe_supply
    moved = ExpertPlacementSnapshot(
        placement_epoch=1,
        expert_owners=tuple((layer, expert, 3 if expert < 2 else 0) for layer in range(2) for expert in range(4)),
    )
    supply = replace(original, placements=(*original.placements, moved), step_placement_epochs=((0, 0), (1, 1)))
    cfg = replace(cfg, routed_moe_supply=supply)
    first = SerialStepLowerer(cfg).lower(step(cfg, 0))
    second = SerialStepLowerer(cfg).lower(step(cfg, 1))
    assert any(isinstance(op.work, CollectiveWork) for op in first.operations)
    assert not any(isinstance(op.work, CollectiveWork) for op in second.operations)
    assert all(op.placement_epoch == 1 for op in second.operations)
    assert all(op.rank == 0 for op in second.operations)


def test_prefill_chunk_and_decode_offsets_select_only_the_current_rows():
    cfg = specimen(((2,),), prompt_tokens=3)
    record = step(cfg, 0)
    record.scheduled[0].num_new_tokens = 1
    record.scheduled[0].context_length = 2
    first = step_routed_moe_work(record, cfg.dims, cfg.ep_ranks, cfg.routed_moe_supply)
    assert [row.token_index for row in first[0].expert_assignments] == [1]
    decode = step_routed_moe_work(step(cfg, 2), cfg.dims, cfg.ep_ranks, cfg.routed_moe_supply)
    assert [row.token_index for row in decode[0].expert_assignments] == [1]
    record.scheduled[0].context_length = 4
    with pytest.raises(ValueError, match="outside"):
        step_routed_moe_work(record, cfg.dims, cfg.ep_ranks, cfg.routed_moe_supply)


def test_no_sampled_tokens_omit_output_projection():
    cfg = specimen(((2,),), prompt_tokens=3)
    record = step(cfg, 0)
    record.num_sampled = 0
    record.sampled_request_ids = []
    graph = SerialStepLowerer(cfg).lower(record)
    assert not any(isinstance(op.work, ComputeWork) and op.work.kernel == "lm_head" for op in graph.operations)


@pytest.mark.parametrize("changes", [
    {"routed_moe_supply": None},
    {"tp_ranks": (0, 3)},
    {"tp_ranks": (3,)},
    {"ep_ranks": (0,)},
    {"host_model": HostInitiationModel(initiation_delay_ps=1)},
    {"routed_compute": True},
    {"gpu": replace(GPU, name="foreign")},
    {"gpu": replace(GPU, peak_flops=2_000_000_000)},
    {"gpu": replace(GPU, mem_bandwidth=2_000_000_000_000_000)},
])
def test_invalid_composition_rejects_before_any_runtime_exists(changes):
    cfg = specimen(((2,),), minimum_ps=1)
    with pytest.raises((TypeError, ValueError)):
        replace(cfg, **changes)


@pytest.mark.parametrize("lowerer", [SerialStepLowerer, ObservedStepLowerer])
def test_observed_conflict_and_scalar_timing_are_explicitly_rejected(lowerer):
    cfg = specimen(((2,),))
    with pytest.raises(ValueError, match="conflicts"):
        lowerer(cfg).lower(step(cfg, 0), ExecutionObservations(()))
    with pytest.raises(ValueError, match="per rank"):
        SerialStepLowerer(cfg).timing(step(cfg, 0))


def test_bad_route_and_provider_leave_prior_completed_state_unchanged():
    cfg = specimen(((2,),))
    sink = _sink(cfg)
    sink(step(cfg, 0), None)
    before = (sink.clock.now_ps, sink.outcomes, sink.runtime.last_report)
    bad = step(cfg, 1, sink.clock.now_ps)
    bad.scheduled[0].request_id = "foreign"
    with pytest.raises(ValueError, match="absent"):
        sink(bad, None)
    assert (sink.clock.now_ps, sink.outcomes, sink.runtime.last_report) == before
    with pytest.raises(ValueError, match="conflicts"):
        sink(step(cfg, 1, sink.clock.now_ps), ExecutionObservations(()))
    assert (sink.clock.now_ps, sink.outcomes, sink.runtime.last_report) == before

    class BadProvider(ComputeProvider):
        def estimate(self, kernel, gpu):
            return DurationEstimate(-1, "declared")

    fresh = _sink(replace(cfg, provider=BadProvider()))
    with pytest.raises(ValueError, match="nonnegative"):
        fresh(step(cfg, 0), None)
    assert fresh.clock.now_ps == 0 and fresh.outcomes == ()
    assert fresh.runtime.last_report is None


def test_explicit_off_reproduces_graph_and_complete_step_results():
    enabled = campaign_config(8, "balanced", 0)
    off = replace(enabled, routed_compute=None)
    first = _sink(off)
    second = _sink(replace(off))
    for index in range(3):
        one = first(step(off, index, first.clock.now_ps), None)
        two = second(step(off, index, second.clock.now_ps), None)
        assert step_result_to_json(one) == step_result_to_json(two)
        graph = first.outcomes[-1].graph
        assert execution_graph_to_json(graph) == execution_graph_to_json(SerialStepLowerer(off).lower(first.outcomes[-1].record))
        assert all("declared_gemm_minimum_ps" not in dict(op.work.config) for op in graph.operations if isinstance(op.work, ComputeWork))


def test_drain_is_empty_and_does_not_charge_the_selected_minimum():
    cfg = specimen(((2,),), minimum_ps=20_000_000)
    enabled = SerialStepLowerer(cfg).lower(StepRecord(0, 0))
    disabled = SerialStepLowerer(replace(cfg, routed_compute=None)).lower(StepRecord(0, 0))
    assert execution_graph_to_json(enabled) == execution_graph_to_json(disabled)
    assert enabled.operations == ()


@pytest.mark.parametrize("value", [-1, True, 1.5])
def test_minimum_requires_a_nonnegative_integer(value):
    with pytest.raises(ValueError):
        RoutedComputeConfig(value, GPU)


@pytest.mark.parametrize("histogram", [(), ((0, 0),), ((4, 1),), ((0, 1), (0, 2)), ((1, 1), (0, 1)), ((True, 1),)])
def test_invalid_expert_histograms_reject(histogram):
    cfg = specimen(((2,),))
    with pytest.raises((ValueError, TypeError)):
        grouped_expert_kernels(cfg.dims, layer=0, rank=3, expert_rows=histogram)


def test_fused_or_empty_work_cannot_be_given_one_kernel_minimum():
    mode = RoutedComputeConfig(10, GPU)
    for kernel in (KernelSpec("llm_step", 1, 1), KernelSpec("moe_gate_up", 0, 0)):
        with pytest.raises(ValueError, match="nonempty"):
            mode.estimate(RooflineProvider(), kernel, GPU)


def test_mixed_prefill_decode_uses_each_requests_phase_slice():
    cfg = specimen(((0, 2), (2, 3)), prompt_tokens=2)
    record = step(cfg, 0)
    record.scheduled[0].num_new_tokens = 1
    record.scheduled[0].context_length = 2
    record.scheduled[1].phase = RequestPhase.DECODE
    record.scheduled[1].num_new_tokens = 1
    record.scheduled[1].context_length = 3
    layers = step_routed_moe_work(record, cfg.dims, cfg.ep_ranks, cfg.routed_moe_supply)
    assert [(row.request_id, row.token_index, row.expert_id) for row in layers[0].expert_assignments] == [
        ("request-0", 1, 0), ("request-0", 1, 2),
        ("request-1", 0, 2), ("request-1", 0, 3),
    ]
    assert layers[0].expert_rows(0) == ((0, 1),)
    assert layers[0].expert_rows(3) == ((2, 2), (3, 1))


@pytest.mark.parametrize("mutation", ["missing-owner", "foreign-owner", "model", "duplicate-request", "padding"])
def test_admission_failures_do_not_execute_a_partial_graph(mutation):
    cfg = specimen(((2,),))
    record = step(cfg, 0)
    if mutation in ("missing-owner", "foreign-owner"):
        owners = cfg.routed_moe_supply.placements[0].expert_owners
        if mutation == "missing-owner":
            owners = owners[:-1]
        else:
            owners = (*owners[:-1], (1, 3, 9))
        placement = ExpertPlacementSnapshot(placement_epoch=0, expert_owners=owners)
        cfg = replace(cfg, routed_moe_supply=replace(cfg.routed_moe_supply, placements=(placement,)))
    elif mutation == "model":
        cfg = replace(cfg, dims=replace(cfg.dims, num_experts=8))
    elif mutation == "duplicate-request":
        record.scheduled.append(replace(record.scheduled[0]))
    else:
        record.num_tokens_after_padding = record.total_new_tokens + 1
    sink = _sink(cfg)
    with pytest.raises((ValueError, TypeError)):
        sink(record, None)
    assert sink.clock.now_ps == 0 and sink.outcomes == ()
    assert sink.runtime.last_report is None


def test_packed_arena_and_strict_projection_share_one_work_inventory(tmp_path):
    trace = Path(__file__).resolve().parents[1] / "examples/preplay_trace_v1/granite_length_cap.jsonl"
    run = join_preplay_arrivals(
        (RequestArrival(request_id="length-cap", arrived_at_ps=0),),
        trace, RequestBookkeeper(),
    )
    routed = project_preplay_routing(run)
    arena = build_routing_arena(run, tmp_path / "routing-arena.json")
    dims = ModelDims(
        24, 1024, 512, 16, 8, 64, 49152, num_experts=32, top_k=8,
        moe_intermediate_size=512, local_num_experts=16,
    )
    placement = ExpertPlacementSnapshot(
        placement_epoch=0,
        expert_owners=tuple(
            (layer, expert, 0 if expert < 16 else 3)
            for layer in range(24) for expert in range(32)
        ),
    )
    strict = RoutedMoeSupply(
        engine_rank=0, placements=(placement,), step_placement_epochs=((0, 0),),
        routed_experts=routed,
    )
    lifetimes = RequestLifetimeRegistry(arena.moe_layer_indices)
    provenance = JoinProvenance(
        run_schema=run.schema, trace_schema=run.trace.schema, trace_sha256=run.trace.sha256,
    )
    for request in arena.requests:
        lifetimes.register(
            request.request_id, provenance, 0,
            RoutingViewDescriptor(
                arena_id=arena.arena_id,
                token_offset=request.token_offset, token_count=request.token_count,
                prompt_token_count=request.prompt_token_count, release_callback=lambda: None,
            ),
        )
    packed = RoutedMoeSupply(
        engine_rank=0, placements=(placement,), step_placement_epochs=((0, 0),),
        routing_arena=arena, lifetimes=lifetimes,
    )
    request = routed.requests[0]
    record = StepRecord(0, 0, [
        ScheduledRequest(
            request.request_id, RequestPhase.PREFILL, request.prompt_token_count,
            context_length=request.prompt_token_count,
        )
    ])
    try:
        expected = step_routed_moe_work(record, dims, (0, 3), strict)
        assert step_routed_moe_work(record, dims, (0, 3), packed) == expected
        assert lifetimes.by_request_id(request.request_id).consumption_cursor == 0
        lifetime = lifetimes.by_request_id(request.request_id)
        lifetime.view_released = True
        with pytest.raises(ValueError, match="released"):
            step_routed_moe_work(record, dims, (0, 3), packed)
        lifetime.view_released = False
        lifetime.routing_view = replace(lifetime.view, token_count=lifetime.view.token_count + 1)
        with pytest.raises(ValueError, match="extent"):
            step_routed_moe_work(record, dims, (0, 3), packed)
    finally:
        arena.close()
    with pytest.raises(ValueError, match="closed"):
        step_routed_moe_work(record, dims, (0, 3), packed)
