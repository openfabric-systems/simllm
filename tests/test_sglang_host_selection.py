"""SGLang per-step host cost selection (SGL-23), exercised without SGLang.

The selector is adapter-side glue over :mod:`simllm.compute.host`, so nothing
here imports SGLang or touches a GPU. The tests that matter are the ones a
wrong selector would fail: that the ideal path stays an exact identity, that a
calibrated path composes with the overlap-aware ``max`` rule rather than the
legacy additive branch, and that the device hybrid it forces is visible in
every enabled selection.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from simllm.adapters.sglang.host import (
    SGLANG_HOST_PROFILES,
    SGLANG_HOST_TRANSFER_DISCLOSURE,
    SGLANG_TRANSFERRED_LAUNCH_BRACKET,
    SGLANG_TRANSFERRED_LAUNCH_COUNTS,
    select_sglang_host_model,
)
from simllm.compute import (
    GPU_ENVELOPES,
    HostInitiationModel,
    KernelSpec,
    ModelDims,
    RooflineProvider,
    step_kernel,
)
from simllm.core import RequestPhase, ScheduledRequest, StepRecord

GRAPH_POINT_PS = 809_306
EAGER_POINT_PS = 2_364_255

DIMS = ModelDims(
    num_layers=24,
    hidden_size=1024,
    intermediate_size=512,
    num_heads=16,
    num_kv_heads=8,
    head_size=64,
    vocab_size=49155,
    dtype_bytes=2,
    num_experts=32,
    top_k=8,
    moe_intermediate_size=512,
    local_num_experts=4,
)


def decode_record() -> StepRecord:
    return StepRecord(
        step_index=0,
        virtual_time_ps=0,
        scheduled=[
            ScheduledRequest("r0", RequestPhase.DECODE, 1, context_length=13)
        ],
        num_sampled=1,
        sampled_request_ids=["r0"],
    )


def test_the_default_selection_is_the_exact_pre_seam_configuration():
    selection = select_sglang_host_model()

    assert selection.profile == "ideal"
    assert selection.is_default
    assert selection.host_model == HostInitiationModel.ideal()
    assert selection.host_model.is_ideal
    assert selection.gpu == GPU_ENVELOPES["b100"]
    assert selection.launch_count == 0
    assert selection.launch_floor_ps == 0
    assert selection.transfer_disclosure is None
    assert selection.sink_overrides() == {
        "host_model": selection.host_model,
        "gpu": selection.gpu,
        "provider": selection.provider,
    }
    assert selection.worker_overrides() == {
        "host_model": selection.host_model,
        "gpu": selection.gpu,
        "compute_provider": selection.provider,
    }


def test_the_ideal_provider_prices_exactly_what_a_plain_roofline_prices():
    selection = select_sglang_host_model("ideal", efficiency=0.7)
    kernel = step_kernel(DIMS, decode_record(), num_sampled=1)

    assert selection.provider.estimate(
        kernel, GPU_ENVELOPES["b100"]
    ) == RooflineProvider(0.7).estimate(kernel, GPU_ENVELOPES["b100"])


def test_the_ideal_profile_refuses_a_launch_count():
    with pytest.raises(ValueError, match="takes no launch count"):
        select_sglang_host_model("ideal", launch_count=440)


def test_an_unknown_profile_names_the_ones_that_exist():
    with pytest.raises(ValueError, match="unknown SGLang host profile"):
        select_sglang_host_model("turing-something-else")
    assert SGLANG_HOST_PROFILES == (
        "ideal",
        "turing-cuda-graph",
        "turing-eager-host",
    )


@pytest.mark.parametrize(
    ("profile", "point_ps"),
    [("turing-cuda-graph", GRAPH_POINT_PS), ("turing-eager-host", EAGER_POINT_PS)],
)
def test_a_calibrated_selection_carries_the_turing_device_and_the_disclosure(
    profile, point_ps
):
    selection = select_sglang_host_model(profile, launch_count=100)

    assert not selection.is_default
    assert selection.host_model.is_calibrated
    assert selection.host_model.device_key == "gtx1660-ti-sm75"
    assert selection.gpu.name == "gtx1660-ti-sm75"
    assert selection.provider_envelope == "b100"
    assert selection.launch_count == 100
    assert selection.launch_floor_ps == 100 * point_ps
    assert selection.transfer_disclosure == SGLANG_HOST_TRANSFER_DISCLOSURE
    assert "gtx1660-ti-sm75" in selection.describe()
    assert "SGL-24" in selection.transfer_disclosure


def test_the_disclosure_names_all_three_transferred_sources():
    for fragment in ("GTX 1660 Ti", "vLLM 0.26.0", "b100", "not calibrated"):
        assert fragment in SGLANG_HOST_TRANSFER_DISCLOSURE


def test_an_absent_launch_count_takes_this_profile_s_bracket_endpoint():
    assert SGLANG_TRANSFERRED_LAUNCH_COUNTS == {
        "turing-cuda-graph": 440,
        "turing-eager-host": 567,
    }
    assert select_sglang_host_model("turing-cuda-graph").launch_count == 440
    assert select_sglang_host_model("turing-eager-host").launch_count == 567


def test_either_bracket_endpoint_is_selectable_with_either_launch_class():
    """The bracket is one eager enumeration's min and max, not a per-class pair.

    The defaults above are a study convention that reproduces the composed
    vLLM study's two headline cells. Nothing may refuse the other endpoint.
    """

    assert SGLANG_TRANSFERRED_LAUNCH_BRACKET == (440, 567)
    for profile in ("turing-cuda-graph", "turing-eager-host"):
        for endpoint in SGLANG_TRANSFERRED_LAUNCH_BRACKET:
            selection = select_sglang_host_model(profile, launch_count=endpoint)
            assert selection.launch_count == endpoint
    crossed = select_sglang_host_model("turing-cuda-graph", launch_count=567)
    assert crossed.launch_floor_ps == 567 * GRAPH_POINT_PS


@pytest.mark.parametrize("launch_count", [0, -1])
def test_a_calibrated_profile_refuses_a_nonpositive_launch_count(launch_count):
    with pytest.raises(ValueError, match="positive launch count"):
        select_sglang_host_model("turing-cuda-graph", launch_count=launch_count)


def test_a_calibrated_profile_refuses_a_boolean_launch_count():
    with pytest.raises(TypeError, match="must be an integer"):
        select_sglang_host_model("turing-cuda-graph", launch_count=True)


def test_the_provider_keeps_the_b100_envelope_under_the_turing_device_key():
    """The device key moves for the host model; the compute must not move."""

    selection = select_sglang_host_model("turing-cuda-graph", launch_count=440)
    kernel = step_kernel(DIMS, decode_record(), num_sampled=1)
    plain = RooflineProvider(0.7)

    pinned = selection.provider.estimate(kernel, selection.gpu)
    assert pinned == plain.estimate(kernel, GPU_ENVELOPES["b100"])
    assert pinned != plain.estimate(kernel, GPU_ENVELOPES["gtx1660-ti-sm75"])


def test_the_pinned_provider_forwards_a_layer_breakdown_on_the_same_envelope():
    selection = select_sglang_host_model("turing-eager-host", launch_count=567)
    kernel = step_kernel(DIMS, decode_record(), num_sampled=1)

    assert selection.provider.estimate_layers(kernel, selection.gpu, 24) is None

    kernel_without_families = KernelSpec(
        name="probe", flops=1.0e12, bytes_moved=1.0e9
    )
    assert (
        selection.provider.estimate_layers(kernel_without_families, selection.gpu, 4)
        is None
    )


def test_a_masked_launch_demand_changes_the_step_by_nothing():
    """The overlap-aware max rule, not the legacy additive branch.

    122 CUDA-graph launches are 98,735,332 ps, below this record's provider
    service, so the represented duration must be the provider's own. An
    additive composition would report the sum instead.
    """

    selection = select_sglang_host_model("turing-cuda-graph", launch_count=122)
    kernel = step_kernel(DIMS, decode_record(), num_sampled=1)
    provider_estimate = selection.provider.estimate(kernel, selection.gpu)
    represented = selection.host_model.represented_estimate(
        provider_estimate, selection.gpu
    )

    assert selection.launch_floor_ps == 122 * GRAPH_POINT_PS == 98_735_332
    assert provider_estimate.duration_ps == 98_980_937
    assert represented.duration_ps == provider_estimate.duration_ps
    assert represented.exposed_ps == 0
    assert represented.bound == provider_estimate.bound == "memory"


def test_one_more_launch_replaces_the_compute_entirely():
    selection = select_sglang_host_model("turing-cuda-graph", launch_count=123)
    kernel = step_kernel(DIMS, decode_record(), num_sampled=1)
    provider_estimate = selection.provider.estimate(kernel, selection.gpu)
    represented = selection.host_model.represented_estimate(
        provider_estimate, selection.gpu
    )

    assert selection.launch_floor_ps == 123 * GRAPH_POINT_PS == 99_544_638
    assert represented.duration_ps == 99_544_638
    assert represented.exposed_ps == 99_544_638 - 98_980_937
    assert represented.bound == "host-initiation"


def test_the_transferred_bracket_endpoints_reproduce_the_accepted_floors():
    """Both endpoints must land on the values the vLLM composed study reported."""

    graph = select_sglang_host_model("turing-cuda-graph")
    eager = select_sglang_host_model("turing-eager-host")

    assert graph.launch_floor_ps == 356_094_640
    assert eager.launch_floor_ps == 1_340_532_585


def test_an_unknown_envelope_is_refused():
    with pytest.raises(ValueError, match="unknown compute envelope"):
        select_sglang_host_model("ideal", provider_envelope="b300")


@pytest.mark.parametrize("efficiency", [0.0, 1.5])
def test_an_out_of_range_efficiency_is_refused(efficiency):
    with pytest.raises(ValueError, match=r"efficiency must be in \(0, 1\]"):
        select_sglang_host_model("ideal", efficiency=efficiency)


def test_worker_overrides_splat_into_configure_s_exact_signature():
    """The keyword spellings differ between the two consumers, so both are pinned.

    ``configure`` takes ``compute_provider`` and ``HtsimStepSinkConfig`` takes
    ``provider``. A single accessor cannot serve both, and splatting the wrong
    one raises ``TypeError``. The stub below is asserted to carry configure's
    exact keyword set first, so it cannot drift away from the real function and
    keep passing.
    """

    from simllm.adapters.sglang.worker import configure

    def stub_configure(
        *,
        step_sink=None,
        compute_provider=None,
        gpu=None,
        host_model=None,
        config=None,
    ):
        return {
            "step_sink": step_sink,
            "compute_provider": compute_provider,
            "gpu": gpu,
            "host_model": host_model,
            "config": config,
        }

    real = inspect.signature(configure).parameters
    stub = inspect.signature(stub_configure).parameters
    assert set(real) == set(stub)
    assert all(real[name].kind is inspect.Parameter.KEYWORD_ONLY for name in real)

    selection = select_sglang_host_model("turing-cuda-graph", launch_count=440)
    applied = stub_configure(**selection.worker_overrides())

    assert applied["compute_provider"] is selection.provider
    assert applied["gpu"] is selection.gpu
    assert applied["host_model"] is selection.host_model
    # Binding against the real function is what makes this a contract rather
    # than a restatement of the stub.
    inspect.signature(configure).bind_partial(**selection.worker_overrides())

    with pytest.raises(TypeError):
        stub_configure(**selection.sink_overrides())


def test_sink_overrides_splat_into_the_step_sink_configuration():
    from simllm.backends import HtsimStepSinkConfig

    selection = select_sglang_host_model("turing-eager-host", launch_count=567)
    config = HtsimStepSinkConfig(
        profile="rnic-nn-fluid",
        tp_ranks=(0, 1),
        dims=DIMS,
        workdir=Path("unused-by-construction"),
        **selection.sink_overrides(),
    )

    assert config.provider is selection.provider
    assert config.gpu is selection.gpu
    assert config.host_model is selection.host_model

    with pytest.raises(TypeError):
        HtsimStepSinkConfig(
            profile="rnic-nn-fluid",
            tp_ranks=(0, 1),
            dims=DIMS,
            workdir=Path("unused-by-construction"),
            **selection.worker_overrides(),
        )


def test_the_module_imports_without_sglang():
    import simllm.adapters.sglang.host as host_module

    assert "sglang" not in dir(host_module)
