from fractions import Fraction

import pytest

from simllm.adapters.vllm.executor import (
    SimExecutorConfig,
    configure,
    reset_configuration,
)
from simllm.adapters.vllm.pd_session import (
    DEPLOYMENT_CURVE_SCHEMA,
    VllmPdCurvePoint,
    VllmPdCurveRecord,
    VllmPdRequest,
    VllmPdRequestResult,
)
from simllm.core import (
    KV_HANDOFF_AUTHORITY,
    DeclaredKvHandoffPolicy,
    DisaggregatedRequestTimeline,
    KvHandoffEvent,
    KvHandoffGeometry,
    VirtualClock,
)


def test_geometry_prices_complete_prompt_kv_bytes():
    geometry = KvHandoffGeometry(
        num_layers=24,
        num_kv_heads=8,
        head_size=64,
        element_bytes=2,
    )

    assert geometry.bytes_per_token == 49_152
    assert geometry.bytes_for_prompt(8) == 393_216
    assert geometry.bytes_for_prompt(16) == 786_432


def test_declared_handoff_is_one_clock_visit():
    clock = VirtualClock(start_ps=17)

    event = DeclaredKvHandoffPolicy(duration_ps=100_000_000).apply(
        clock,
        request_id="request-0",
        kv_bytes=393_216,
    )

    assert event.authority == KV_HANDOFF_AUTHORITY
    assert event.pricing_arm == "declared-constant"
    assert event.submitted_at_ps == 17
    assert event.eligible_at_ps == 17
    assert event.started_at_ps == 17
    assert event.finished_at_ps == 100_000_017
    assert event.completed_at_ps == 100_000_017
    assert event.queue_wait_ps == 0
    assert event.service_ps == 100_000_000
    assert event.visibility_ps == 0
    assert clock.now_ps == event.completed_at_ps


def test_declared_handoffs_can_be_scheduled_independently():
    policy = DeclaredKvHandoffPolicy(duration_ps=100)

    first = policy.schedule(
        submitted_at_ps=17,
        request_id="request-0",
        kv_bytes=10,
    )
    second = policy.schedule(
        submitted_at_ps=17,
        request_id="request-1",
        kv_bytes=20,
    )

    assert first.submitted_at_ps == second.submitted_at_ps == 17
    assert first.completed_at_ps == second.completed_at_ps == 117


def test_handoff_off_arm_is_identity():
    clock = VirtualClock(start_ps=91)

    event = DeclaredKvHandoffPolicy.off().apply(
        clock,
        request_id="request-off",
        kv_bytes=1,
    )

    assert event.pricing_arm == "off"
    assert event.service_ps == 0
    assert clock.now_ps == 91


def test_timeline_reduces_exact_ttft_and_decode_only_tpot():
    handoff = KvHandoffEvent(
        request_id="request-0",
        kv_bytes=393_216,
        submitted_at_ps=110,
        eligible_at_ps=110,
        started_at_ps=110,
        finished_at_ps=130,
        completed_at_ps=130,
        pricing_arm="declared-constant",
    )

    timeline = DisaggregatedRequestTimeline(
        request_id="request-0",
        admitted_at_ps=10,
        prefill_eligible_at_ps=20,
        prefill_completed_at_ps=110,
        handoff=handoff,
        decode_eligible_at_ps=135,
        decode_token_completed_at_ps=(175, 205, 245, 280),
    )

    assert timeline.prefill_queue_ps == 10
    assert timeline.prefill_service_ps == 90
    assert timeline.decode_admission_wait_ps == 5
    assert timeline.decode_first_token_service_ps == 40
    assert timeline.decomposition_total_ps == 165
    assert timeline.ttft_ps == 165
    assert timeline.tpot_ps == Fraction(105, 3)
    assert timeline.to_json()["decomposition"]["total_ps"] == 165


def test_timeline_rejects_a_second_timing_story():
    handoff = KvHandoffEvent(
        request_id="request-0",
        kv_bytes=1,
        submitted_at_ps=100,
        eligible_at_ps=100,
        started_at_ps=100,
        finished_at_ps=101,
        completed_at_ps=101,
        pricing_arm="declared-constant",
    )

    with pytest.raises(ValueError, match="submitted at prefill completion"):
        DisaggregatedRequestTimeline(
            request_id="request-0",
            admitted_at_ps=0,
            prefill_eligible_at_ps=0,
            prefill_completed_at_ps=99,
            handoff=handoff,
            decode_eligible_at_ps=101,
            decode_token_completed_at_ps=(102,),
        )


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"duration_ps": 0}, "positive duration"),
        ({"duration_ps": 1, "enabled": False}, "zero duration"),
        ({"duration_ps": True}, "integer"),
    ],
)
def test_handoff_policy_rejects_ambiguous_arms(kwargs, error):
    with pytest.raises((TypeError, ValueError), match=error):
        DeclaredKvHandoffPolicy(**kwargs)


def test_executor_hooks_carry_one_declared_pool_role_and_clock():
    reset_configuration()
    clock = VirtualClock(start_ps=31)
    try:
        hooks = configure(
            config=SimExecutorConfig(pool_role="prefill"),
            clock=clock,
        )

        assert hooks.config is not None
        assert hooks.config.pool_role == "prefill"
        assert hooks.clock is clock
    finally:
        hooks = reset_configuration()

    assert hooks.config is None
    assert hooks.clock is None


def test_executor_config_rejects_an_undeclared_pool_role():
    with pytest.raises(ValueError, match="POOL_ROLE"):
        SimExecutorConfig(pool_role="worker")


def _curve_request(request_id, admitted_at_ps, completed_at_ps):
    handoff = KvHandoffEvent(
        request_id=request_id,
        kv_bytes=1,
        submitted_at_ps=admitted_at_ps + 10,
        eligible_at_ps=admitted_at_ps + 10,
        started_at_ps=admitted_at_ps + 10,
        finished_at_ps=admitted_at_ps + 20,
        completed_at_ps=admitted_at_ps + 20,
        pricing_arm="declared-constant",
    )
    token_times = tuple(completed_at_ps - offset for offset in (3, 2, 1, 0))
    timeline = DisaggregatedRequestTimeline(
        request_id=request_id,
        admitted_at_ps=admitted_at_ps,
        prefill_eligible_at_ps=admitted_at_ps,
        prefill_completed_at_ps=admitted_at_ps + 10,
        handoff=handoff,
        decode_eligible_at_ps=admitted_at_ps + 20,
        decode_token_completed_at_ps=token_times,
    )
    return VllmPdRequestResult(
        timeline=timeline,
        prefill_engine_id="prefill-0",
        decode_engine_id="decode-0",
        prefill_internal_request_id=f"prefill-{request_id}",
        decode_internal_request_id=f"decode-{request_id}",
        bootstrap_token_id=512,
        decode_token_ids=(1, 2, 3, 4),
        kv_transfer_params={},
        prefill_records=(),
        decode_records=(),
    )


def test_curve_point_uses_exact_terminal_throughput_and_mean_request_delay():
    requests = (
        _curve_request("a", 0, 100),
        _curve_request("b", 10, 130),
    )

    point = VllmPdCurvePoint.from_requests(Fraction(16), requests)

    assert point.aggregated_output_throughput_tokens_per_second == Fraction(
        8 * 1_000_000_000_000,
        130,
    )
    assert point.per_token_request_delay_ps == Fraction(55, 2)
    assert point.request_count == 2
    assert point.output_token_count == 8


def test_curve_record_is_machine_readable_and_load_ordered():
    requests = (_curve_request("a", 0, 100),)
    points = tuple(
        VllmPdCurvePoint.from_requests(Fraction(load), requests)
        for load in (8, 16, 32)
    )

    record = VllmPdCurveRecord("p1-d1-prompt8", 1, 1, 8, points)

    rendered = record.to_json()
    assert rendered["schema"] == DEPLOYMENT_CURVE_SCHEMA
    assert [
        row["offered_load_requests_per_second"]["numerator"]
        for row in rendered["points"]
    ] == [8, 16, 32]
    with pytest.raises(ValueError, match="unique and increasing"):
        VllmPdCurveRecord("unordered", 1, 1, 8, tuple(reversed(points)))


def test_concurrent_request_validates_stable_admission_input():
    request = VllmPdRequest("request-0", (1, 2), 4, 17)

    assert request.prompt_token_ids == (1, 2)
    with pytest.raises(ValueError, match="nonnegative integers"):
        VllmPdRequest("bad", (1, -1), 4, 17)
