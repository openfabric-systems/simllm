"""Tests for the closed-loop htsim step sink (simllm.backends.step_sink)."""

import pytest

from simllm.backends import HtsimStepSink, HtsimStepSinkConfig, find_htsim_rnic
from simllm.compute import ModelDims
from simllm.core import RequestPhase, ScheduledRequest, StepRecord
from simllm.goal import find_txt2bin

SMALL_DIMS = ModelDims(
    num_layers=2,
    hidden_size=1024,
    intermediate_size=4096,
    num_heads=8,
    num_kv_heads=8,
    head_size=128,
    vocab_size=32000,
    dtype_bytes=2,
)

SMALL_MOE_DIMS = ModelDims(
    num_layers=2,
    hidden_size=1024,
    intermediate_size=4096,
    num_heads=8,
    num_kv_heads=8,
    head_size=128,
    vocab_size=32000,
    dtype_bytes=2,
    num_experts=8,
    top_k=2,
    moe_intermediate_size=512,
    local_num_experts=2,
)

# M1-calibrated fluid constants (examples/m1/RESULTS.md C1/C2)
PS_PER_BYTE_400G = 20
PROPAGATION_PS = 2_000_000


def decode_record() -> StepRecord:
    return StepRecord(
        step_index=4,
        virtual_time_ps=1_000_000_000,
        scheduled=[
            ScheduledRequest(f"r{i}", RequestPhase.DECODE, num_new_tokens=1,
                             context_length=128)
            for i in range(2)
        ],
    )


def sink(tmp_path, tp_ranks=(0, 1), profile="rnic-nn-fluid", dims=SMALL_DIMS,
         ep_ranks=None) -> HtsimStepSink:
    return HtsimStepSink(HtsimStepSinkConfig(
        profile=profile,
        tp_ranks=tp_ranks,
        dims=dims,
        workdir=tmp_path / "sink",
        ep_ranks=ep_ranks,
    ))


def test_sink_returns_none_without_collectives(tmp_path):
    # TP world of 1: the adapter's own compute estimate stands
    assert sink(tmp_path, tp_ranks=(0,))(decode_record()) is None
    # drain record: zero new tokens, completions only
    drain = StepRecord(step_index=9, virtual_time_ps=5, finished_request_ids=["a"])
    s = sink(tmp_path)
    assert s(drain) is None
    assert s.outcomes == []


def test_sink_returns_none_with_ep_ranks_but_dense_dims(tmp_path):
    # an EP group alone produces no traffic for a dense model at TP=1
    s = sink(tmp_path, tp_ranks=(0,), ep_ranks=(0, 1, 2, 3))
    assert s(decode_record()) is None
    # MoE dims but a drain record: still nothing to simulate
    s = sink(tmp_path, tp_ranks=(0,), dims=SMALL_MOE_DIMS, ep_ranks=(0, 1))
    drain = StepRecord(step_index=9, virtual_time_ps=5, finished_request_ids=["a"])
    assert s(drain) is None
    # MoE dims without a configured EP group: dense behavior (TP=1 -> None)
    s = sink(tmp_path, tp_ranks=(0,), dims=SMALL_MOE_DIMS)
    assert s(decode_record()) is None


def test_sink_config_rejects_unknown_profile(tmp_path):
    with pytest.raises(ValueError, match="profile"):
        sink(tmp_path, profile="rnic-ss")


@pytest.mark.skipif(
    find_txt2bin() is None or find_htsim_rnic() is None,
    reason="backend toolchain not available (submodule not built)",
)
def test_sink_end_to_end_matches_closed_form(tmp_path):
    """One real step through txt2bin + htsim_rnic on the fluid profile.

    The fluid manifold is exact (M1), so the returned makespan must equal
    the closed form L * calc_eff + 2L * 2(W-1) * (chunk * 20 + P) to 0 ps.
    """
    s = sink(tmp_path)
    record = decode_record()
    result = s(record)
    assert result is not None

    estimate_ps = s.compute_estimate_ps(record)
    per_layer_ns = estimate_ps // (SMALL_DIMS.num_layers * 1000)
    payload = record.total_new_tokens * SMALL_DIMS.hidden_size * SMALL_DIMS.dtype_bytes
    chunk = payload // 2
    rounds_per_allreduce = 2 * (2 - 1)
    allreduce_ps = rounds_per_allreduce * (chunk * PS_PER_BYTE_400G + PROPAGATION_PS)
    expected = (
        SMALL_DIMS.num_layers * max(per_layer_ns, 1) * 1000
        + 2 * SMALL_DIMS.num_layers * allreduce_ps
    )

    assert result.step_index == record.step_index
    assert result.step_latency_ps == expected
    assert result.completed_at_ps == record.virtual_time_ps + expected
    assert len(s.outcomes) == 1
    outcome = s.outcomes[0]
    assert outcome.compute_estimate_ps == estimate_ps
    assert outcome.per_layer_calc_ns == per_layer_ns
    assert outcome.makespan_ps == expected
    # 2L allreduces x 2(W-1) rounds x W sends per round
    assert outcome.num_flows == 2 * SMALL_DIMS.num_layers * rounds_per_allreduce * 2
    assert 0.0 < outcome.network_share_for(SMALL_DIMS.num_layers) < 1.0
    # the GOAL and completion CSV land in the configured workdir
    assert (tmp_path / "sink" / "step-000004.goal").is_file()
    assert (tmp_path / "sink" / "step-000004.rnic-nn-fluid.csv").is_file()


@pytest.mark.skipif(
    find_txt2bin() is None or find_htsim_rnic() is None,
    reason="backend toolchain not available (submodule not built)",
)
def test_sink_moe_end_to_end_matches_closed_form(tmp_path):
    """An EP-only MoE step through txt2bin + htsim_rnic on the fluid profile.

    Per layer: calc, then dispatch and combine pairwise all-to-allvs over
    W = 4 EP ranks. Each a2av phase releases W(W-1) equal flows at once
    (every NIC sends and receives W-1), so the fluid manifold grants each
    flow floor(B / (W-1)) whole bps (the exact max-min water level floored,
    rnic_max_min_allocator.cpp) and completes it after
    ceil(S * 8e12 / rate) whole ps (rnic_fluid_manifold.cpp), plus P.
    For W-1 dividing 400e9 this is exactly the M1 20 ps/byte law; W-1 = 3
    is not a divisor, hence the explicit floor/ceil form.
    """
    world = 4
    s = sink(tmp_path, tp_ranks=(0,), dims=SMALL_MOE_DIMS,
             ep_ranks=tuple(range(world)))
    record = decode_record()
    result = s(record)
    assert result is not None

    estimate_ps = s.compute_estimate_ps(record)
    per_layer_ns = estimate_ps // (SMALL_MOE_DIMS.num_layers * 1000)
    per_pair = (record.total_new_tokens * SMALL_MOE_DIMS.top_k
                * SMALL_MOE_DIMS.hidden_size * SMALL_MOE_DIMS.dtype_bytes) // world
    rate_bps = 400_000_000_000 // (world - 1)
    debt = per_pair * 8 * 10**12
    phase_ps = -(-debt // rate_bps) + PROPAGATION_PS
    expected = SMALL_MOE_DIMS.num_layers * (max(per_layer_ns, 1) * 1000 + 2 * phase_ps)

    assert result.step_latency_ps == expected
    # 2L a2avs x W(W-1) flows each
    assert s.outcomes[0].num_flows == 2 * SMALL_MOE_DIMS.num_layers * world * (world - 1)
