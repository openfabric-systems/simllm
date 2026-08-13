"""Focused acceptance tests for the calibrated fixed host-step model."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from simllm.adapters.vllm import (
    VllmBatchSlice,
    build_granite_execution_observations,
)
from simllm.adapters.vllm.executor import SimExecutorConfig, _SimStepRuntime
from simllm.backends import (
    DeviceRuntimeStepSink,
    HtsimStepSinkConfig,
    SerialStepLowerer,
    SerialStepLowererConfig,
)
from simllm.compute import (
    GPU_ENVELOPES,
    ComputeProvider,
    DurationEstimate,
    HostInitiationModel,
    ModelDims,
)
from simllm.core import RequestPhase, ScheduledRequest, StepRecord, VirtualClock

TURING = GPU_ENVELOPES["gtx1660-ti-sm75"]
DIMS = ModelDims(
    num_layers=1,
    hidden_size=64,
    intermediate_size=128,
    num_heads=4,
    num_kv_heads=4,
    head_size=16,
    vocab_size=256,
)
DIMS_24 = ModelDims(
    num_layers=24,
    hidden_size=64,
    intermediate_size=128,
    num_heads=4,
    num_kv_heads=4,
    head_size=16,
    vocab_size=256,
    num_experts=8,
    top_k=1,
    moe_intermediate_size=128,
    local_num_experts=1,
)
RECORD = StepRecord(
    0,
    0,
    [ScheduledRequest("r0", RequestPhase.DECODE, 1, context_length=8)],
    num_sampled=1,
    sampled_request_ids=["r0"],
)


class FixedProvider(ComputeProvider):
    def __init__(
        self,
        duration_ps: int,
        uncertainty: float = 0.0,
        *,
        layer_breakdown: bool = False,
    ) -> None:
        self.duration_ps = duration_ps
        self.uncertainty = uncertainty
        self.layer_breakdown = layer_breakdown

    def estimate(self, kernel, gpu):
        return DurationEstimate(
            duration_ps=self.duration_ps,
            bound="compute",
            uncertainty=self.uncertainty,
        )

    def estimate_layers(self, kernel, gpu, num_layers):
        if not self.layer_breakdown:
            return None
        quotient, remainder = divmod(self.duration_ps, num_layers)
        return tuple(
            DurationEstimate(
                quotient + (index < remainder),
                "compute",
                self.uncertainty,
            )
            for index in range(num_layers)
        )


def test_named_profiles_match_the_accepted_capture_and_provenance():
    graph = HostInitiationModel.turing_cuda_graph(440)
    eager = HostInitiationModel.turing_eager_host(567)

    assert (
        graph.profile,
        graph.launch_class,
        graph.initiation_delay_ps,
        graph.lower_ps_per_launch,
        graph.upper_ps_per_launch,
        graph.launch_floor_ps,
    ) == (
        "turing-cuda-graph",
        "cuda-graph-node",
        809_306,
        624_665,
        809_306,
        356_094_640,
    )
    assert (
        eager.profile,
        eager.launch_class,
        eager.initiation_delay_ps,
        eager.lower_ps_per_launch,
        eager.upper_ps_per_launch,
        eager.launch_floor_ps,
    ) == (
        "turing-eager-host",
        "eager-host-bound",
        2_364_255,
        2_327_730,
        2_544_074,
        1_340_532_585,
    )
    assert graph.device_key == eager.device_key == "gtx1660-ti-sm75"
    assert graph.device_model == "NVIDIA GeForce GTX 1660 Ti"
    assert graph.gpu_uuid == "GPU-a90a812a-41bf-4f2f-c96d-d83e6eae6bd0"
    assert graph.host_cpu == "AMD Ryzen 9 3950X 16-Core Processor"
    assert graph.driver_version == "550.90.07"
    assert graph.cuda_version == "12.4.99"
    assert graph.source_study == "examples/host_step_cost_v1"
    assert graph.uncertainty_kind == (
        "sample-limited empirical range, not a confidence interval"
    )


def test_calibrated_composition_uses_max_and_carries_empirical_uncertainty():
    model = HostInitiationModel.turing_eager_host(50)
    provider = DurationEstimate(100_000_000, "compute", 0.1)

    estimate = model.represented_estimate(provider, TURING)

    assert estimate.duration_ps == 118_212_750
    assert estimate.provider_duration_ps == 100_000_000
    assert estimate.launch_floor_ps == 118_212_750
    assert estimate.launch_floor_lower_ps == 116_386_500
    assert estimate.launch_floor_upper_ps == 127_203_700
    assert estimate.empirical_lower_ps == 116_386_500
    assert estimate.empirical_upper_ps == 127_203_700
    assert estimate.exposed_ps == 18_212_750
    assert estimate.bound == "host-initiation"
    assert estimate.uncertainty_fraction == pytest.approx(
        (127_203_700 - 118_212_750) / 118_212_750
    )


def test_ideal_is_exact_identity_and_legacy_scalar_stays_additive():
    provider = DurationEstimate(123_456, "memory", 0.25)

    ideal = HostInitiationModel.ideal().represented_estimate(provider, TURING)
    assert ideal.duration_ps == provider.duration_ps
    assert ideal.bound == provider.bound
    assert ideal.uncertainty_fraction == provider.uncertainty
    assert ideal.exposed_ps == 0

    legacy = HostInitiationModel(800, "gin").represented_estimate(provider, TURING)
    assert legacy.duration_ps == 124_256
    assert legacy.exposed_ps == 800


def test_lowerer_and_packet_sink_defaults_are_exact_ideal(tmp_path):
    lowerer = SerialStepLowererConfig(DIMS, (0,))
    packet = HtsimStepSinkConfig(
        profile="rnic-nn-fluid",
        tp_ranks=(0,),
        dims=DIMS,
        workdir=tmp_path / "packet",
    )

    assert lowerer.host_model == HostInitiationModel.ideal()
    assert packet.host_model == HostInitiationModel.ideal()
    assert lowerer.host_model.delay_ps() == packet.host_model.delay_ps() == 0


@pytest.mark.parametrize("device", ["b100", "h100"])
def test_calibrated_profile_rejects_unmeasured_devices_before_output(tmp_path, device):
    model = HostInitiationModel.turing_cuda_graph(440)
    target = tmp_path / device

    with pytest.raises(ValueError, match="calibrated only"):
        HtsimStepSinkConfig(
            profile="rnic-nn-fluid",
            tp_ranks=(0,),
            dims=DIMS,
            workdir=target,
            provider=FixedProvider(99_024_000),
            gpu=GPU_ENVELOPES[device],
            host_model=model,
        )

    assert not target.exists()


def test_lowerer_and_device_sink_apply_one_shared_host_term():
    model = HostInitiationModel.turing_cuda_graph(440)
    config = SerialStepLowererConfig(
        DIMS,
        (0,),
        provider=FixedProvider(99_024_000),
        gpu=TURING,
        host_model=model,
    )
    timing = SerialStepLowerer(config).timing(RECORD)
    assert timing.provider_compute_ps == 99_024_000
    assert timing.host_launch_floor_ps == 356_094_640
    assert timing.compute_estimate_ps == 356_094_640
    assert timing.exposed_host_ps == 257_070_640

    sink = DeviceRuntimeStepSink(config)
    clock = VirtualClock()
    sink.bind_clock(clock)
    result = sink(RECORD, None)
    assert sink.host_model is model
    assert result.step_latency_ps == 356_095_000
    assert result.step_latency_ps >= timing.host_launch_floor_ps
    assert result.step_latency_ps < 99_024_000 + 356_094_640


@pytest.mark.parametrize("layer_breakdown", [False, True])
@pytest.mark.parametrize(
    ("profile", "launch_count", "floor_ps", "represented_ps"),
    [
        ("turing-cuda-graph", 440, 356_094_640, 356_095_000),
        ("turing-cuda-graph", 567, 458_876_502, 458_877_000),
        ("turing-eager-host", 440, 1_040_272_200, 1_040_273_000),
        ("turing-eager-host", 567, 1_340_532_585, 1_340_533_000),
    ],
)
def test_calibrated_serial_and_observed_paths_use_the_same_narrow_enclosure(
    layer_breakdown,
    profile,
    launch_count,
    floor_ps,
    represented_ps,
):
    provider = FixedProvider(99_024_000, layer_breakdown=layer_breakdown)
    model = (
        HostInitiationModel.turing_cuda_graph(launch_count)
        if profile == "turing-cuda-graph"
        else HostInitiationModel.turing_eager_host(launch_count)
    )
    config = SerialStepLowererConfig(
        DIMS_24,
        (0,),
        ep_ranks=tuple(range(8)),
        provider=provider,
        gpu=TURING,
        host_model=model,
    )

    timing = SerialStepLowerer(config).timing(RECORD)
    _, observed = build_granite_execution_observations(
        RECORD,
        DIMS_24,
        tuple(range(8)),
        (VllmBatchSlice(None, ("r0",), 1),),
        provider,
        TURING,
        model,
    )

    assert timing.provider_compute_ps == observed.provider_compute_ps == 99_024_000
    assert timing.compute_estimate_ps == floor_ps
    assert sum(timing.layer_calc_ns) * 1000 == represented_ps
    assert observed.represented_compute_ps == represented_ps
    assert floor_ps <= represented_ps < floor_ps + 1000
    assert represented_ps < 99_024_000 + floor_ps


@pytest.mark.parametrize("layer_breakdown", [False, True])
@pytest.mark.parametrize(
    ("model", "delay_ps"),
    [
        (HostInitiationModel.ideal(), 0),
        (HostInitiationModel(800, "gin"), 800),
    ],
)
def test_calibrated_enclosure_does_not_change_ideal_or_legacy_quantization(
    layer_breakdown,
    model,
    delay_ps,
):
    duration_ps = 99_024_321
    provider = FixedProvider(duration_ps, layer_breakdown=layer_breakdown)
    config = SerialStepLowererConfig(
        DIMS_24,
        (0,),
        ep_ranks=tuple(range(8)),
        provider=provider,
        gpu=TURING,
        host_model=model,
    )
    expected = (
        ((duration_ps + delay_ps) // 1000) * 1000
        if layer_breakdown
        else ((duration_ps + delay_ps) // (24 * 1000)) * 24 * 1000
    )

    timing = SerialStepLowerer(config).timing(RECORD)
    _, observed = build_granite_execution_observations(
        RECORD,
        DIMS_24,
        tuple(range(8)),
        (VllmBatchSlice(None, ("r0",), 1),),
        provider,
        TURING,
        model,
    )

    assert sum(timing.layer_calc_ns) * 1000 == expected
    assert observed.represented_compute_ps == expected
    assert observed.provider_compute_ps == duration_ps


@dataclass
class HostAwareSink:
    host_model: HostInitiationModel

    def __call__(self, record):
        return None


def test_coordinator_dispatch_rejects_adapter_sink_host_model_mismatch():
    graph = HostInitiationModel.turing_cuda_graph(440)
    eager = HostInitiationModel.turing_eager_host(440)

    with pytest.raises(RuntimeError, match="same host model"):
        _SimStepRuntime(
            config=SimExecutorConfig(),
            step_sink=HostAwareSink(eager),
            fallback_latency=lambda translated: 0,
            host_model=graph,
            gpu=TURING,
        )
