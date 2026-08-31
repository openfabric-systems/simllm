"""Micro-scale conformance coverage for the surrogate serving loop."""

from __future__ import annotations

from dataclasses import replace

import pytest

from simllm.backends import (
    DeviceRuntimeStepSink,
    SerialStepLowerer,
    SerialStepLowererConfig,
)
from simllm.compute import ComputeProvider, DurationEstimate, ModelDims
from simllm.core import (
    CoarseDeviceRuntime,
    CompletionReducer,
    KvCacheAction,
    KvLifecycleLedger,
    KvPoolSpec,
    RequestPhase,
    StepRecord,
    StepResult,
    VirtualClock,
    step_record_from_json,
    step_record_to_json,
    step_result_to_json,
)
from simllm.deploy import (
    SURROGATE_LOOP_SCHEMA,
    EstimateStamp,
    EstimatorClass,
    EvidenceClass,
    NamedTermEstimate,
    PointClass,
    SurrogateLoopConfig,
    SurrogateQueuePolicy,
    SurrogateRequest,
    SurrogateReserveMode,
    SurrogateServingLoop,
    SurrogateStopPolicy,
    TermEstimate,
    estimate_stamp_from_json,
    estimate_stamp_to_json,
    surrogate_loop_stamp,
)


def _pricing_stamp() -> EstimateStamp:
    return EstimateStamp(
        candidate_key="1" * 64,
        terms=(
            NamedTermEstimate(
                "step-pricing",
                TermEstimate(1, EvidenceClass.DECLARED, "unit pricing surface"),
            ),
        ),
    )


def _stamp() -> EstimateStamp:
    return surrogate_loop_stamp(_pricing_stamp())


def _config(
    *,
    budget: int = 32,
    max_num_seqs: int = 2,
    chunked: bool = True,
    threshold: int = 0,
    max_model_len: int = 128,
    policy: SurrogateQueuePolicy = SurrogateQueuePolicy.FCFS,
    blocks: int = 16,
    reserve: SurrogateReserveMode = SurrogateReserveMode.NONE,
    watermark: float = 0.0,
    prefix_caching: bool = True,
) -> SurrogateLoopConfig:
    return SurrogateLoopConfig(
        resolved_max_num_scheduled_tokens=budget,
        max_num_seqs=max_num_seqs,
        enable_chunked_prefill=chunked,
        long_prefill_token_threshold=threshold,
        max_model_len=max_model_len,
        queue_policy=policy,
        scheduler_block_size=16,
        num_kv_blocks=blocks,
        reserve_mode=reserve,
        watermark=watermark,
        enable_prefix_caching=prefix_caching,
    )


def _pool(blocks: int = 16) -> KvPoolSpec:
    return KvPoolSpec(
        pool_id="surrogate:test:kv",
        block_bytes=16 * 128,
        block_tokens=16,
        capacity_blocks=blocks,
    )


def _request(
    request_id: str,
    prompt_tokens: int,
    *,
    arrived_at_ps: int = 0,
    output_tokens: int = 1,
    priority: int = 0,
    token_base: int = 0,
    stop_policy: SurrogateStopPolicy | None = None,
) -> SurrogateRequest:
    return SurrogateRequest(
        request_id=request_id,
        arrived_at_ps=arrived_at_ps,
        prompt_token_ids=tuple(range(token_base, token_base + prompt_tokens)),
        max_output_tokens=output_tokens,
        priority=priority,
        stop_policy=stop_policy or SurrogateStopPolicy(default_token_id=10_000),
    )


def _fixed_pricer(latency_ps: int = 1):
    def price(record: StepRecord) -> StepResult:
        return StepResult(
            step_index=record.step_index,
            step_latency_ps=latency_ps,
            completed_at_ps=record.virtual_time_ps + latency_ps,
        )

    return price


def _run(
    config: SurrogateLoopConfig,
    requests: tuple[SurrogateRequest, ...],
    *,
    latency_ps: int = 1,
):
    loop = SurrogateServingLoop(
        config,
        requests,
        _pool(config.num_kv_blocks),
        _stamp(),
    )
    return loop, loop.run(step_sink=_fixed_pricer(latency_ps))


def _nonempty(result):
    return tuple(emission for emission in result.emissions if emission.record.scheduled)


def test_registered_loop_stamp_and_causal_tuple_are_explicit() -> None:
    config = _config(
        budget=7,
        max_num_seqs=3,
        chunked=False,
        threshold=5,
        max_model_len=96,
        policy=SurrogateQueuePolicy.PRIORITY,
        blocks=9,
        reserve=SurrogateReserveMode.FULL_ISL,
        watermark=0.125,
    )
    stamp = _stamp()

    assert stamp.estimator_class is EstimatorClass.ESTIMATE_LOOP
    assert estimate_stamp_from_json(estimate_stamp_to_json(stamp)) == stamp
    assert _pricing_stamp().estimator_class is EstimatorClass.ESTIMATE
    assert config.causal_tuple == (
        ("resolved_max_num_scheduled_tokens", 7),
        ("max_num_seqs", 3),
        ("enable_chunked_prefill", False),
        ("enable_prefix_caching", True),
        ("long_prefill_token_threshold", 5),
        ("max_model_len", 96),
        ("queue_policy", "priority"),
        ("scheduler_block_size", 16),
        ("num_kv_blocks", 9),
        ("reserve_mode", "full-isl"),
        ("watermark", 0.125),
    )

    loop, result = _run(config, (_request("r0", 4),))

    assert result.schema == SURROGATE_LOOP_SCHEMA
    assert result.causal_tuple == config.causal_tuple
    assert result.point_class is PointClass.ESTIMATE_LOOP
    assert loop.point_class is PointClass.ESTIMATE_LOOP
    assert all(
        emission.stamp.estimator_class is EstimatorClass.ESTIMATE_LOOP
        and emission.point_class is PointClass.ESTIMATE_LOOP
        for emission in result.emissions
    )


def test_registered_loop_refuses_an_unregistered_block_or_stamp() -> None:
    with pytest.raises(ValueError, match="scheduler_block_size must be 16"):
        replace(_config(), scheduler_block_size=32)
    with pytest.raises(TypeError, match="enable_prefix_caching must be a boolean"):
        replace(_config(), enable_prefix_caching=1)

    with pytest.raises(ValueError, match="ESTIMATE-LOOP"):
        SurrogateServingLoop(
            _config(),
            (_request("r0", 4),),
            _pool(),
            _pricing_stamp(),
        )


@pytest.mark.parametrize(
    ("budget", "max_num_seqs", "expected_ids", "expected_tokens"),
    [
        (8, 1, ("r0",), (5,)),
        (8, 2, ("r0", "r1"), (5, 3)),
        (10, 1, ("r0",), (5,)),
        (10, 2, ("r0", "r1"), (5, 5)),
    ],
)
def test_f1_budget_subtraction_and_sequence_cap(
    budget: int,
    max_num_seqs: int,
    expected_ids: tuple[str, ...],
    expected_tokens: tuple[int, ...],
) -> None:
    requests = tuple(_request(f"r{index}", 5) for index in range(3))
    _, result = _run(
        _config(budget=budget, max_num_seqs=max_num_seqs),
        requests,
    )

    first = _nonempty(result)[0].record
    assert tuple(row.request_id for row in first.scheduled) == expected_ids
    assert tuple(row.num_new_tokens for row in first.scheduled) == expected_tokens
    assert tuple(row.phase for row in first.scheduled) == (
        RequestPhase.PREFILL,
    ) * len(expected_ids)
    assert tuple(row.context_length for row in first.scheduled) == expected_tokens
    assert first.sampled_request_ids == [
        request_id
        for request_id, tokens in zip(expected_ids, expected_tokens, strict=True)
        if tokens == 5
    ]
    assert all(
        emission.record.total_new_tokens <= budget for emission in result.emissions
    )
    assert result.admission_order == ("r0", "r1", "r2")


@pytest.mark.parametrize(
    ("prompt_tokens", "expected_chunks"),
    [(7, (7,)), (8, (8,)), (9, (8, 1))],
)
def test_f2_chunk_boundaries_cross_the_budget(
    prompt_tokens: int,
    expected_chunks: tuple[int, ...],
) -> None:
    _, result = _run(
        _config(budget=8, max_num_seqs=1, chunked=True),
        (_request("r0", prompt_tokens),),
    )

    chunks = tuple(
        emission.record.scheduled[0].num_new_tokens for emission in _nonempty(result)
    )
    assert chunks == expected_chunks


def test_f2_chunking_off_stops_at_the_waiting_head() -> None:
    loop = SurrogateServingLoop(
        _config(budget=8, max_num_seqs=2, chunked=False),
        (_request("head", 9), _request("follower", 2)),
        _pool(),
        _stamp(),
    )

    with pytest.raises(RuntimeError, match="waiting head cannot be admitted"):
        loop.step(step_sink=_fixed_pricer())

    assert loop.waiting_request_ids == ("head", "follower")
    assert loop.running_request_ids == ()


def test_f2_long_prefill_cap_precedes_the_chunking_off_check() -> None:
    _, result = _run(
        _config(
            budget=32,
            max_num_seqs=1,
            chunked=False,
            threshold=16,
        ),
        (_request("long", 33),),
    )

    assert tuple(
        emission.record.scheduled[0].num_new_tokens for emission in _nonempty(result)
    ) == (16, 16, 1)


def test_negative_control_wrong_budget_order_is_detected() -> None:
    def wrong_budget_order(remaining: int, budget: int, threshold: int) -> int | None:
        if remaining > budget:
            return None
        return min(remaining, threshold, budget)

    def assert_frozen_chunk(value: int | None) -> None:
        assert value == 16

    with pytest.raises(AssertionError):
        assert_frozen_chunk(wrong_budget_order(33, 32, 16))


@pytest.mark.parametrize(
    ("blocks", "max_num_seqs", "expect_preemption"),
    [(3, 2, True), (5, 2, False), (3, 1, False)],
)
def test_f3_capacity_crossing_controls_preemption(
    blocks: int,
    max_num_seqs: int,
    expect_preemption: bool,
) -> None:
    _, result = _run(
        _config(
            budget=32,
            max_num_seqs=max_num_seqs,
            blocks=blocks,
        ),
        (
            _request("a", 16, output_tokens=3),
            _request("b", 16, output_tokens=3, token_base=100),
        ),
    )

    preempted = tuple(
        request_id
        for emission in result.emissions
        for request_id in emission.record.preempted_request_ids
    )
    assert bool(preempted) is expect_preemption
    if expect_preemption:
        assert preempted[0] == "b"
        recompute = next(
            work
            for _, work in result.kv_operations
            if work.action is KvCacheAction.RECOMPUTE
        )
        assert recompute.request_id == "b"
        assert (recompute.token_start, recompute.token_end) == (0, 16)
        resumed = next(
            row
            for emission in result.emissions
            for row in emission.record.scheduled
            if row.request_id == "b" and row.num_new_tokens == 17
        )
        assert resumed.context_length == 17
        assert next(
            row.num_preemptions
            for row in result.request_results
            if row.request_id == "b"
        ) == 1


def _priority_victim(policy: SurrogateQueuePolicy) -> str:
    _, result = _run(
        _config(
            budget=32,
            max_num_seqs=2,
            policy=policy,
            blocks=4,
        ),
        (
            _request("a", 16, output_tokens=4, priority=9),
            _request(
                "b",
                16,
                arrived_at_ps=1,
                output_tokens=4,
                priority=1,
                token_base=100,
            ),
        ),
    )
    return next(
        emission.record.preempted_request_ids[0]
        for emission in result.emissions
        if emission.record.preempted_request_ids
    )


def test_f3_fcfs_tail_and_priority_maximum_choose_different_victims() -> None:
    assert _priority_victim(SurrogateQueuePolicy.FCFS) == "b"
    assert _priority_victim(SurrogateQueuePolicy.PRIORITY) == "a"


def test_f3_priority_victim_keeps_running_order_when_native_key_ties() -> None:
    _, result = _run(
        _config(
            budget=32,
            max_num_seqs=2,
            policy=SurrogateQueuePolicy.PRIORITY,
            blocks=4,
        ),
        (
            _request("a", 16, output_tokens=4, priority=1),
            _request("b", 16, output_tokens=4, priority=1, token_base=100),
        ),
    )

    victim = next(
        emission.record.preempted_request_ids[0]
        for emission in result.emissions
        if emission.record.preempted_request_ids
    )
    assert victim == "a"


def test_negative_control_priority_tail_victim_is_detected() -> None:
    def assert_priority_maximum(request_id: str) -> None:
        assert request_id == "a"

    wrong_running_tail = "b"
    with pytest.raises(AssertionError):
        assert_priority_maximum(wrong_running_tail)


@pytest.mark.parametrize(
    ("prompt_tokens", "expected_cached"),
    [(8, 0), (16, 0), (32, 16), (48, 32)],
)
def test_f4_prefix_hits_use_full_chained_extents_and_recompute_the_tail(
    prompt_tokens: int,
    expected_cached: int,
) -> None:
    prompt = tuple(range(prompt_tokens))
    requests = (
        SurrogateRequest("first", 0, prompt, 1),
        SurrogateRequest("repeat", 1, prompt, 2),
    )
    _, result = _run(
        _config(budget=128, max_num_seqs=1, blocks=12),
        requests,
    )

    repeated_rows = [
        row
        for emission in result.emissions
        for row in emission.record.scheduled
        if row.request_id == "repeat"
    ]
    assert repeated_rows[0].num_cached_tokens == expected_cached
    assert repeated_rows[0].num_new_tokens == prompt_tokens - expected_cached
    assert sum(row.num_cached_tokens for row in repeated_rows[1:]) == 0


def test_f4_reverse_freeing_and_lazy_lru_eviction_order() -> None:
    requests = (
        _request("a", 32, token_base=0),
        _request("b", 16, arrived_at_ps=1, token_base=100),
        _request("c", 32, arrived_at_ps=2, token_base=200),
    )
    _, result = _run(
        _config(budget=64, max_num_seqs=1, blocks=4),
        requests,
    )

    evictions = [
        work.block_ids[0]
        for _, work in result.kv_operations
        if work.action is KvCacheAction.EVICT
    ]
    assert evictions == ["2", "1"]
    c_operations = [
        work.action
        for operation_id, work in result.kv_operations
        if operation_id.startswith("surrogate-kv-000002")
    ]
    assert c_operations == [
        KvCacheAction.RESERVE,
        KvCacheAction.EVICT,
        KvCacheAction.EVICT,
        KvCacheAction.ALLOCATE,
        KvCacheAction.WRITE,
        KvCacheAction.RELEASE,
    ]


@pytest.mark.parametrize(
    ("prefix_caching", "expected_free_blocks"),
    [
        (True, (1, 2, 3)),
        (False, (2, 3, 1)),
    ],
)
def test_0271_cache_disabled_release_appends_for_locality(
    prefix_caching: bool,
    expected_free_blocks: tuple[int, ...],
) -> None:
    loop = SurrogateServingLoop(
        _config(
            budget=4,
            max_num_seqs=1,
            blocks=4,
            prefix_caching=prefix_caching,
        ),
        (_request("short", 1),),
        _pool(4),
        _stamp(),
    )

    loop.step(step_sink=_fixed_pricer())

    assert loop.free_block_ids == expected_free_blocks


def test_f4_content_miss_does_not_alias_equal_length_prompts() -> None:
    _, result = _run(
        _config(budget=64, max_num_seqs=1, blocks=8),
        (
            _request("first", 32, token_base=0),
            _request("different", 32, arrived_at_ps=1, token_base=1_000),
        ),
    )

    different = next(
        row
        for emission in result.emissions
        for row in emission.record.scheduled
        if row.request_id == "different"
    )
    assert different.num_cached_tokens == 0


@pytest.mark.parametrize(
    ("max_num_seqs", "expected_batches", "expected_releases"),
    [
        (1, (("early",), ("late-first",), ("late-second",)), (0, 10, 20)),
        (2, (("early",), ("late-first", "late-second")), (0, 10, 10)),
    ],
)
def test_f5_virtual_admission_order_and_first_release(
    max_num_seqs: int,
    expected_batches: tuple[tuple[str, ...], ...],
    expected_releases: tuple[int, ...],
) -> None:
    _, result = _run(
        _config(budget=8, max_num_seqs=max_num_seqs),
        (
            _request("late-first", 2, arrived_at_ps=10),
            _request("early", 2, arrived_at_ps=0, token_base=10),
            _request("late-second", 2, arrived_at_ps=10, token_base=20),
        ),
        latency_ps=10,
    )

    assert result.admission_order == ("early", "late-first", "late-second")
    assert tuple(
        tuple(row.request_id for row in emission.record.scheduled)
        for emission in _nonempty(result)
    ) == expected_batches
    by_id = {row.request_id: row for row in result.request_results}
    assert tuple(
        by_id[request_id].first_released_at_ps
        for request_id in ("early", "late-first", "late-second")
    ) == expected_releases
    assert tuple(
        by_id[request_id].first_released_at_ps - by_id[request_id].arrived_at_ps
        for request_id in ("early", "late-first", "late-second")
    ) == tuple(
        released - arrived
        for released, arrived in zip(
            expected_releases,
            (0, 10, 10),
            strict=True,
        )
    )


def test_full_isl_reservation_refuses_a_chunk_that_cannot_finish() -> None:
    request = (_request("long", 32),)
    allowed = SurrogateServingLoop(
        _config(budget=16, max_num_seqs=1, blocks=2),
        request,
        _pool(2),
        _stamp(),
    )
    first = allowed.step(step_sink=_fixed_pricer())
    assert first.record.scheduled[0].num_new_tokens == 16

    reserved = SurrogateServingLoop(
        _config(
            budget=16,
            max_num_seqs=1,
            blocks=2,
            reserve=SurrogateReserveMode.FULL_ISL,
        ),
        request,
        _pool(2),
        _stamp(),
    )
    with pytest.raises(RuntimeError, match="waiting head cannot be admitted"):
        reserved.step(step_sink=_fixed_pricer())


def test_watermark_applies_only_after_another_request_is_running() -> None:
    requests = (
        _request("first", 16),
        _request("second", 16, token_base=100),
    )
    _, baseline = _run(
        _config(budget=32, max_num_seqs=2, blocks=4, watermark=0.0),
        requests,
    )
    guarded = SurrogateServingLoop(
        _config(budget=32, max_num_seqs=2, blocks=4, watermark=0.5),
        requests,
        _pool(4),
        _stamp(),
    )
    first_guarded = guarded.step(step_sink=_fixed_pricer())

    assert tuple(row.request_id for row in _nonempty(baseline)[0].record.scheduled) == (
        "first",
        "second",
    )
    assert tuple(row.request_id for row in first_guarded.record.scheduled) == ("first",)


class _FixedProvider(ComputeProvider):
    def estimate(self, kernel, gpu):
        return DurationEstimate(duration_ps=2_000_000, bound="unit-fixed")


_DIMS = ModelDims(
    num_layers=2,
    hidden_size=64,
    intermediate_size=128,
    num_heads=4,
    num_kv_heads=4,
    head_size=16,
    vocab_size=256,
    dtype_bytes=2,
)
_LOWERER_CONFIG = SerialStepLowererConfig(
    _DIMS,
    (0,),
    provider=_FixedProvider(),
)


def test_f6_records_replay_through_the_same_metric_chain_exactly() -> None:
    config = _config(budget=4, max_num_seqs=1)
    loop = SurrogateServingLoop(
        config,
        (_request("r0", 2, output_tokens=2),),
        _pool(),
        _stamp(),
    )
    live_sink = DeviceRuntimeStepSink(_LOWERER_CONFIG)

    result = loop.run(observation_step_sink=live_sink)

    replay_clock = VirtualClock()
    replay_sink = DeviceRuntimeStepSink(_LOWERER_CONFIG)
    replay_sink.bind_clock(replay_clock)
    replayed = []
    for record in result.records:
        assert replay_clock.now_ps == record.virtual_time_ps
        replayed.append(replay_sink(record, None))

    assert [step_result_to_json(value) for value in replayed] == [
        step_result_to_json(value) for value in result.results
    ]
    metrics = [
        metric
        for step_result in result.results
        for metric in step_result.request_metrics
    ]
    assert [metric.token_index for metric in metrics] == [1, 2]
    assert metrics[0].ttft_ps == 2_000_000
    assert metrics[1].tpot_ps == 2_000_000


def test_f7_kv_stream_is_ledger_and_lowerer_ready() -> None:
    prompt = tuple(range(32))
    pool = _pool(6)
    loop = SurrogateServingLoop(
        _config(budget=64, max_num_seqs=1, blocks=6),
        (
            SurrogateRequest("first", 0, prompt, 1),
            SurrogateRequest("repeat", 1, prompt, 1),
        ),
        pool,
        _stamp(),
    )
    result = loop.run(step_sink=_fixed_pricer())

    ledger = KvLifecycleLedger((pool,))
    lowerer = SerialStepLowerer(_LOWERER_CONFIG)
    runtime = CoarseDeviceRuntime(kv_pools=(pool,))
    clock = VirtualClock()
    reducer = CompletionReducer(clock)
    for emission in result.emissions:
        ledger.consume(emission.kv_operations)
        observations = emission.kv_observations()
        graph = lowerer.lower(emission.record, observations)
        clock.advance_to(emission.record.virtual_time_ps)
        execution_result = runtime.execute(graph)
        assert runtime.last_report is not None
        reduced = reducer.reduce(
            emission.record,
            graph,
            execution_result,
            runtime.last_report,
        )
        assert reduced.step_index == emission.record.step_index

    accounting = ledger.report().pool(pool.pool_id)
    assert accounting.prefix_hit_tokens == 16
    assert accounting.recomputed_tokens == 0
    assert accounting.reserved_blocks == 0
    assert accounting.live_blocks == 0
    assert accounting.reclaimable_blocks == 3
    assert [
        work.action for _, work in result.kv_operations if work.request_id == "repeat"
    ][:5] == [
        KvCacheAction.BIND_PREFIX,
        KvCacheAction.RESERVE,
        KvCacheAction.TOUCH,
        KvCacheAction.ALLOCATE,
        KvCacheAction.WRITE,
    ]


def test_f7_multi_request_kv_stream_has_one_completion_per_scheduled_row() -> None:
    _, result = _run(
        _config(budget=32, max_num_seqs=2, blocks=8),
        (
            _request("a", 16),
            _request("b", 16, token_base=100),
        ),
    )

    emission = _nonempty(result)[0]
    observations = emission.kv_observations()
    operations = {
        operation.operation_id: operation for operation in observations.operations
    }
    assert tuple(
        operations[operation_id].correlation.request_ids[0]
        for operation_id in observations.completion_operation_ids
    ) == ("a", "b")

    graph = SerialStepLowerer(_LOWERER_CONFIG).lower(emission.record, observations)
    runtime = CoarseDeviceRuntime(kv_pools=(_pool(8),))
    execution_result = runtime.execute(graph)
    assert runtime.last_report is not None
    reduced = CompletionReducer(VirtualClock()).reduce(
        emission.record,
        graph,
        execution_result,
        runtime.last_report,
    )
    assert reduced.step_index == emission.record.step_index


def test_complete_step_records_round_trip_and_shift_finished_ids() -> None:
    stop = SurrogateStopPolicy(
        sampled_token_ids=(7, 8),
        stop_token_ids=(7,),
        default_token_id=9,
    )
    _, result = _run(
        _config(budget=8, max_num_seqs=1),
        (_request("r0", 3, output_tokens=4, stop_policy=stop),),
    )

    assert len(result.emissions) == 2
    producing, drain = result.emissions
    assert producing.record.sampled_request_ids == ["r0"]
    assert producing.record.num_sampled == 1
    assert producing.record.finished_request_ids == []
    assert drain.record.scheduled == []
    assert drain.record.sampled_request_ids == []
    assert drain.record.num_sampled == 0
    assert drain.record.finished_request_ids == ["r0"]
    assert result.request_results[0].output_token_ids == (7,)
    assert [
        step_record_from_json(step_record_to_json(record)) for record in result.records
    ] == list(result.records)
