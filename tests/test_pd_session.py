from fractions import Fraction

import pytest

from simllm.adapters.vllm.executor import (
    SimExecutorConfig,
    configure,
    reset_configuration,
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
