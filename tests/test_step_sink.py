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


def sink(tmp_path, tp_ranks=(0, 1), profile="rnic-nn-fluid") -> HtsimStepSink:
    return HtsimStepSink(HtsimStepSinkConfig(
        profile=profile,
        tp_ranks=tp_ranks,
        dims=SMALL_DIMS,
        workdir=tmp_path / "sink",
    ))


def test_sink_returns_none_without_collectives(tmp_path):
    # TP world of 1: the adapter's own compute estimate stands
    assert sink(tmp_path, tp_ranks=(0,))(decode_record()) is None
    # drain record: zero new tokens, completions only
    drain = StepRecord(step_index=9, virtual_time_ps=5, finished_request_ids=["a"])
    s = sink(tmp_path)
    assert s(drain) is None
    assert s.outcomes == []


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
