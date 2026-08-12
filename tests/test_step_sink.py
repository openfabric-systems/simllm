"""Tests for the closed-loop htsim step sink (simllm.backends.step_sink)."""

import hashlib
import threading
import time
from types import SimpleNamespace

import pytest

import simllm.backends.step_sink as step_sink_module
from simllm.backends import (
    HtsimPersistentStepSink,
    HtsimStepSink,
    HtsimStepSinkConfig,
    StepNetworkOutcome,
    find_htsim_rnic,
)
from simllm.compute import (
    ComputeProvider,
    DurationEstimate,
    HostInitiationModel,
    ModelDims,
)
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

SAMPLE_DIMS = ModelDims(
    num_layers=2,
    hidden_size=64,
    intermediate_size=128,
    num_heads=4,
    num_kv_heads=4,
    head_size=16,
    vocab_size=256,
    dtype_bytes=2,
)

# M1-calibrated fluid constants (examples/m1/RESULTS.md C1/C2)
PS_PER_BYTE_400G = 20
PROPAGATION_PS = 2_000_000
# Post-specified re-acceptance: the unchanged graph wire projects at causal
# levels. Compute levels stay analytic, while these four collective artifacts
# reach the backend. The legacy direct diagnostic remains byte-locked in
# test_step_lowerer.py.
DEFAULT_ARTIFACT_MANIFEST = (
    (
        "step-000000.artifact-0001.goal",
        "74122635377dab6e0e605c88ac7e745d126ec6da58f9a6bf2b352d7dedad1d29",
    ),
    (
        "step-000000.artifact-0002.goal",
        "3e6ee8835bc50284cba1f20fb85250f5563b118fc9b121c41a236dad026b8e62",
    ),
    (
        "step-000000.artifact-0004.goal",
        "ffd1d5e61243f5c5d3ffd07d867ed71ae769a26757d7ceb4bc0630fea26d78e2",
    ),
    (
        "step-000000.artifact-0005.goal",
        "c5520d64fe8df27809dd1df8dc2d0e8b0e700657122bb081f51333ec6daa2e76",
    ),
)


class LayerProvider(ComputeProvider):
    def __init__(self, layer_ps, fused_adjustment=0):
        self.layer_ps = tuple(layer_ps)
        self.fused_adjustment = fused_adjustment

    def estimate(self, kernel, gpu):
        return DurationEstimate(
            duration_ps=sum(self.layer_ps) + self.fused_adjustment,
            bound="measured",
        )

    def estimate_layers(self, kernel, gpu, num_layers):
        return tuple(
            DurationEstimate(duration_ps=duration_ps, bound="measured")
            for duration_ps in self.layer_ps
        )


class FlopProvider(ComputeProvider):
    def estimate(self, kernel, gpu):
        return DurationEstimate(duration_ps=int(kernel.flops), bound="compute")


def stub_backend(monkeypatch, makespan_ps=100_000):
    monkeypatch.setattr(step_sink_module, "to_binary", lambda path: path)
    monkeypatch.setattr(
        step_sink_module,
        "run_htsim_rnic",
        lambda _config: SimpleNamespace(
            job_completion_time_ps=lambda: makespan_ps,
            flows=[],
            quiescent=True,
        ),
    )


def goal_artifact_manifest(workdir, step_index):
    return tuple(
        (path.name, hashlib.sha256(path.read_bytes()).hexdigest())
        for path in sorted(
            workdir.glob(f"step-{step_index:06d}.artifact-*.goal")
        )
    )


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


@pytest.mark.parametrize("profile", ["rnic-nn", "rnic-nn-fluid"])
def test_sink_accepts_multi_artifact_stateless_profile(
    tmp_path, monkeypatch, profile
):
    backend_calls = 0

    def tracked_backend(_config):
        nonlocal backend_calls
        backend_calls += 1
        return SimpleNamespace(
            job_completion_time_ps=lambda: 100_000,
            flows=[],
            quiescent=True,
        )

    monkeypatch.setattr(step_sink_module, "to_binary", lambda path: path)
    monkeypatch.setattr(step_sink_module, "run_htsim_rnic", tracked_backend)

    result = sink(tmp_path, profile=profile)(decode_record())

    assert result is not None
    assert backend_calls == 4


def test_sink_rejects_multi_artifact_stateful_profile_before_backend(
    tmp_path, monkeypatch
):
    backend_called = False

    def unexpected_backend(_config):
        nonlocal backend_called
        backend_called = True
        raise AssertionError("stateful backend must not execute")

    monkeypatch.setattr(step_sink_module, "to_binary", lambda path: path)
    monkeypatch.setattr(step_sink_module, "run_htsim_rnic", unexpected_backend)

    with pytest.raises(RuntimeError, match=r"rnic-cn.*BACK-38"):
        sink(tmp_path, profile="rnic-cn")(decode_record())

    assert not backend_called


def test_sink_config_preserves_provider_positional_argument(tmp_path):
    provider = LayerProvider((2_600, 4_600))
    config = HtsimStepSinkConfig(
        "rnic-nn-fluid",
        (0, 1),
        SMALL_DIMS,
        tmp_path,
        None,
        400_000_000_000,
        None,
        provider,
    )
    assert config.provider is provider
    assert config.num_goal_ranks is None


def test_step_network_outcome_preserves_legacy_positional_shape():
    outcome = StepNetworkOutcome(1, 2_000, 1, 4_000, 2)
    assert outcome.layer_calc_ns == ()
    assert outcome.num_sampled == 0
    assert not outcome.sample_count_exact
    assert not outcome.quiescent
    assert outcome.network_share_for(2) == 0.5


def test_sink_passes_explicit_goal_rank_count(tmp_path, monkeypatch):
    stub_backend(monkeypatch)
    s = HtsimStepSink(
        HtsimStepSinkConfig(
            profile="rnic-nn-fluid",
            tp_ranks=(0, 1),
            dims=SMALL_DIMS,
            workdir=tmp_path / "padded",
            num_goal_ranks=8,
        )
    )

    assert s(decode_record()) is not None
    goals = sorted((tmp_path / "padded").glob("step-000004.artifact-*.goal"))
    assert len(goals) == 4
    assert all(
        goal.read_text().startswith("num_ranks 8\n")
        for goal in goals
    )


def test_sink_uses_valid_provider_layer_breakdown(tmp_path, monkeypatch):
    stub_backend(monkeypatch)
    s = HtsimStepSink(
        HtsimStepSinkConfig(
            profile="rnic-nn-fluid",
            tp_ranks=(0, 1),
            dims=SMALL_DIMS,
            workdir=tmp_path / "layered",
            provider=LayerProvider((2_600, 4_600)),
        )
    )

    assert s(decode_record()) is not None
    outcome = s.outcomes[0]
    assert outcome.compute_estimate_ps == 7_200
    assert outcome.per_layer_calc_ns is None
    assert outcome.layer_calc_ns == (2, 5)
    # Post-specified re-acceptance: the unchanged graph wire projects into two
    # analytic compute levels plus four fixed-duration backend collectives.
    assert outcome.makespan_ps == 7_000 + 4 * 100_000
    goals = sorted((tmp_path / "layered").glob("step-000004.artifact-*.goal"))
    assert len(goals) == 4
    goal_text = "\n".join(goal.read_text() for goal in goals)
    assert ": calc " not in goal_text
    locality = s.locality_outcomes[0]
    assert locality.ordering_authority == "execution-graph"
    assert locality.artifact_count == 6
    assert locality.backend_runs == 4
    assert locality.composed_phase_service_ps == (
        2_000,
        100_000,
        100_000,
        5_000,
        100_000,
        100_000,
    )


def test_sink_assigns_host_delay_before_first_layer_boundary(tmp_path, monkeypatch):
    stub_backend(monkeypatch)
    s = HtsimStepSink(
        HtsimStepSinkConfig(
            profile="rnic-nn-fluid",
            tp_ranks=(0, 1),
            dims=SMALL_DIMS,
            workdir=tmp_path / "host-delay",
            provider=LayerProvider((600, 600)),
            host_model=HostInitiationModel(initiation_delay_ps=800),
        )
    )

    assert s(decode_record()) is not None
    assert s.outcomes[0].compute_estimate_ps == 2_000
    assert s.outcomes[0].layer_calc_ns == (1, 1)


@pytest.mark.parametrize(
    ("provider", "message"),
    [
        (LayerProvider((1_000,)), "length"),
        (LayerProvider((-1, 1_001)), "nonnegative"),
        (LayerProvider((1_000, 2_000), fused_adjustment=1), "sum"),
    ],
)
def test_sink_rejects_invalid_provider_layer_breakdown(tmp_path, provider, message):
    s = HtsimStepSink(
        HtsimStepSinkConfig(
            profile="rnic-nn-fluid",
            tp_ranks=(0, 1),
            dims=SMALL_DIMS,
            workdir=tmp_path / message,
            provider=provider,
        )
    )

    with pytest.raises(ValueError, match=message):
        s(decode_record())


def test_sink_default_artifact_manifest_is_reaccepted(tmp_path, monkeypatch):
    stub_backend(monkeypatch, makespan_ps=82_003_040)
    record = StepRecord(
        0,
        0,
        [ScheduledRequest("prefill", RequestPhase.PREFILL, 256, context_length=256)],
    )
    s = HtsimStepSink(
        HtsimStepSinkConfig(
            profile="rnic-nn-fluid",
            tp_ranks=(0, 1),
            dims=SMALL_DIMS,
            workdir=tmp_path / "baseline",
        )
    )

    assert s(record) is not None
    assert goal_artifact_manifest(tmp_path / "baseline", 0) == (
        DEFAULT_ARTIFACT_MANIFEST
    )
    assert s.outcomes[0].per_layer_calc_ns == 12_030
    assert s.outcomes[0].layer_calc_ns == (12_030, 12_030)
    # Post-specified re-acceptance: causal-level projection leaves the graph
    # wire unchanged and adds two analytic compute levels around four backend
    # collective artifacts.
    assert s.outcomes[0].makespan_ps == 24_060_000 + 4 * 82_003_040
    locality = s.locality_outcomes[0]
    assert locality.ordering_authority == "execution-graph"
    assert locality.graph_execution_id == "step-0"
    assert locality.artifact_count == locality.graph_artifact_count == 6
    assert locality.backend_runs == len(DEFAULT_ARTIFACT_MANIFEST)


def test_sink_uses_exact_sample_count_for_chunked_prefill(tmp_path, monkeypatch):
    stub_backend(monkeypatch)
    scheduled = [
        ScheduledRequest("p", RequestPhase.PREFILL, 4, context_length=8),
        ScheduledRequest("d", RequestPhase.DECODE, 1, context_length=32),
    ]
    approximate_record = StepRecord(0, 0, scheduled)
    exact_record = StepRecord(0, 0, scheduled, num_sampled=1)

    approximate = HtsimStepSink(
        HtsimStepSinkConfig(
            profile="rnic-nn-fluid",
            tp_ranks=(0, 1),
            dims=SAMPLE_DIMS,
            workdir=tmp_path / "approximate",
            provider=FlopProvider(),
        )
    )
    exact = HtsimStepSink(
        HtsimStepSinkConfig(
            profile="rnic-nn-fluid",
            tp_ranks=(0, 1),
            dims=SAMPLE_DIMS,
            workdir=tmp_path / "exact",
            provider=FlopProvider(),
        )
    )

    assert approximate.compute_estimate_ps(approximate_record) == 912_896
    assert exact.compute_estimate_ps(exact_record) == 880_128
    assert approximate(approximate_record) is not None
    assert exact(exact_record) is not None
    assert approximate.outcomes[0].num_sampled == 2
    assert not approximate.outcomes[0].sample_count_exact
    assert approximate.outcomes[0].layer_calc_ns == (456, 456)
    assert exact.outcomes[0].num_sampled == 1
    assert exact.outcomes[0].sample_count_exact
    assert exact.outcomes[0].layer_calc_ns == (440, 440)
    assert exact.outcomes[0].quiescent


def test_sink_sample_count_identity_when_every_request_samples(tmp_path, monkeypatch):
    stub_backend(monkeypatch)
    scheduled = [
        ScheduledRequest("d0", RequestPhase.DECODE, 1, context_length=32),
        ScheduledRequest("d1", RequestPhase.DECODE, 1, context_length=32),
    ]
    records = [StepRecord(0, 0, scheduled), StepRecord(0, 0, scheduled, num_sampled=2)]
    artifact_manifests = []
    estimates = []
    for label, record in zip(("absent", "exact"), records, strict=True):
        s = HtsimStepSink(
            HtsimStepSinkConfig(
                profile="rnic-nn-fluid",
                tp_ranks=(0, 1),
                dims=SAMPLE_DIMS,
                workdir=tmp_path / label,
                provider=FlopProvider(),
            )
        )
        assert s(record) is not None
        artifact_manifests.append(goal_artifact_manifest(tmp_path / label, 0))
        estimates.append(s.outcomes[0].compute_estimate_ps)
        assert s.outcomes[0].layer_calc_ns == (212, 212)

    assert estimates == [424_960, 424_960]
    assert artifact_manifests[0] == artifact_manifests[1]


def test_persistent_sink_prepares_concurrently_and_publishes_in_order(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(step_sink_module, "to_binary", lambda path: path)
    active = 0
    maximum_active = 0
    worker_threads = set()
    state_lock = threading.Lock()

    def tracked_backend(config):
        nonlocal active, maximum_active
        goal_bytes = config.goal_bin.read_bytes()
        with state_lock:
            active += 1
            maximum_active = max(maximum_active, active)
            worker_threads.add(threading.get_ident())
        try:
            time.sleep(0.02)
        finally:
            with state_lock:
                active -= 1
        return SimpleNamespace(
            job_completion_time_ps=lambda: 100_000
            + int(hashlib.sha256(goal_bytes).hexdigest()[:8], 16),
            flows=[],
            quiescent=True,
        )

    monkeypatch.setattr(step_sink_module, "run_htsim_rnic", tracked_backend)
    records = [
        StepRecord(
            index,
            index * 1_000,
            [
                ScheduledRequest(
                    f"r{index}",
                    RequestPhase.DECODE,
                    num_new_tokens=1,
                    context_length=128 + index,
                )
            ],
        )
        for index in range(6)
    ]
    diagnostic = HtsimStepSink(
        HtsimStepSinkConfig(
            profile="rnic-nn-fluid",
            tp_ranks=(0, 1),
            dims=SMALL_DIMS,
            workdir=tmp_path / "diagnostic",
        )
    )
    diagnostic_results = [diagnostic(record) for record in records]
    assert maximum_active == 1

    maximum_active = 0
    worker_threads.clear()
    with HtsimPersistentStepSink(
        HtsimStepSinkConfig(
            profile="rnic-nn-fluid",
            tp_ranks=(0, 1),
            dims=SMALL_DIMS,
            workdir=tmp_path / "persistent",
        ),
        max_workers=3,
    ) as persistent:
        persistent.prepare(records)
        assert persistent.outcomes == []
        assert persistent.prepared_steps_remaining == len(records)
        persistent_results = [persistent(record) for record in records]
        assert persistent.prepared_steps_remaining == 0

    assert maximum_active >= 2
    assert len(worker_threads) >= 2
    assert persistent_results == diagnostic_results
    assert persistent.outcomes == diagnostic.outcomes
    for record in records:
        assert goal_artifact_manifest(
            tmp_path / "persistent", record.step_index
        ) == goal_artifact_manifest(tmp_path / "diagnostic", record.step_index)


@pytest.mark.parametrize("max_workers", [True, 1.5, "2"])
def test_persistent_sink_rejects_noninteger_worker_count(tmp_path, max_workers):
    with pytest.raises(TypeError, match="integer"):
        HtsimPersistentStepSink(
            HtsimStepSinkConfig(
                profile="rnic-nn-fluid",
                tp_ranks=(0, 1),
                dims=SMALL_DIMS,
                workdir=tmp_path,
            ),
            max_workers=max_workers,
        )


def test_persistent_sink_enforces_prepared_record_order(tmp_path, monkeypatch):
    stub_backend(monkeypatch)
    records = [
        StepRecord(index, index * 10, decode_record().scheduled)
        for index in range(2)
    ]
    persistent = HtsimPersistentStepSink(
        HtsimStepSinkConfig(
            profile="rnic-nn-fluid",
            tp_ranks=(0, 1),
            dims=SMALL_DIMS,
            workdir=tmp_path / "persistent",
        ),
        max_workers=2,
    )

    with persistent:
        with pytest.raises(RuntimeError, match="prepare"):
            persistent(records[0])
        with pytest.raises(ValueError, match="unique"):
            persistent.prepare([records[0], records[0]])
        persistent.prepare(records)
        with pytest.raises(ValueError, match="next prepared step"):
            persistent(records[1])
        assert persistent.prepared_steps_remaining == 2
        with pytest.raises(RuntimeError, match="consume"):
            persistent.prepare(records)
        assert persistent(records[0]) is not None
        assert persistent(records[1]) is not None
        persistent.prepare([records[0]])
        assert persistent(records[0]) is not None

    with pytest.raises(RuntimeError, match="closed"):
        persistent.prepare(records)


def test_persistent_sink_failed_preparation_publishes_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(step_sink_module, "to_binary", lambda path: path)
    records = [
        StepRecord(index, index * 10, decode_record().scheduled)
        for index in range(2)
    ]

    def failing_backend(config):
        if "step-000001" in config.completion_csv.name:
            raise RuntimeError("injected backend failure")
        return SimpleNamespace(
            job_completion_time_ps=lambda: 100_000,
            flows=[],
            quiescent=True,
        )

    monkeypatch.setattr(step_sink_module, "run_htsim_rnic", failing_backend)
    with HtsimPersistentStepSink(
        HtsimStepSinkConfig(
            profile="rnic-nn-fluid",
            tp_ranks=(0, 1),
            dims=SMALL_DIMS,
            workdir=tmp_path / "persistent",
        ),
        max_workers=2,
    ) as persistent:
        with pytest.raises(RuntimeError, match="injected backend failure"):
            persistent.prepare(records)
        assert persistent.prepared_steps_remaining == 0
        assert persistent.outcomes == []

        stub_backend(monkeypatch)
        persistent.prepare(records)
        results = [persistent(record) for record in records]
        assert [result.step_index for result in results if result is not None] == [0, 1]
        # Post-specified re-acceptance: the unchanged graph wire now has two
        # analytic compute levels and four backend collective artifacts.
        assert [result.step_latency_ps for result in results if result is not None] == [
            24_060_000 + 4 * 100_000,
            24_060_000 + 4 * 100_000,
        ]


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
    assert outcome.layer_calc_ns == (per_layer_ns,) * SMALL_DIMS.num_layers
    assert outcome.makespan_ps == expected
    # 2L allreduces x 2(W-1) rounds x W sends per round
    assert outcome.num_flows == 2 * SMALL_DIMS.num_layers * rounds_per_allreduce * 2
    assert 0.0 < outcome.network_share_for(SMALL_DIMS.num_layers) < 1.0
    # The four collective artifacts and completion CSVs land in the workdir.
    assert len(list((tmp_path / "sink").glob("step-000004.artifact-*.goal"))) == 4
    assert len(
        list((tmp_path / "sink").glob("step-000004.artifact-*.rnic-nn-fluid.csv"))
    ) == 4


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
