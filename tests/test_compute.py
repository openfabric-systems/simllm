import pytest

from simllm.compute import (
    GpuSpec,
    HostInitiationModel,
    KernelSpec,
    ProfileTableProvider,
    RooflineProvider,
)

H100ish = GpuSpec(name="h100-bf16", peak_flops=989e12, mem_bandwidth=3.35e12)


def test_roofline_compute_bound():
    # Big GEMM: high arithmetic intensity
    k = KernelSpec(name="gemm", flops=1e15, bytes_moved=1e9)
    est = RooflineProvider(efficiency=1.0).estimate(k, H100ish)
    assert est.bound == "compute"
    assert est.duration_ps == pytest.approx(1e15 / 989e12 * 1e12, rel=1e-6)


def test_roofline_memory_bound():
    # Decode-style: streams weights, few flops per byte
    k = KernelSpec(name="decode_attn", flops=1e9, bytes_moved=1e12)
    est = RooflineProvider(efficiency=1.0).estimate(k, H100ish)
    assert est.bound == "memory"


def test_roofline_rejects_bad_efficiency():
    with pytest.raises(ValueError):
        RooflineProvider(efficiency=0.0)


def test_profile_table():
    cfg = (("batch_tokens", 512), ("hidden", 7168))
    provider = ProfileTableProvider({("moe_forward", cfg, "h100-bf16"): 123_000_000})
    est = provider.estimate(KernelSpec("moe_forward", 0, 0, cfg), H100ish)
    assert est.duration_ps == 123_000_000
    assert est.bound == "measured"
    with pytest.raises(KeyError):
        provider.estimate(KernelSpec("moe_forward", 0, 0, ()), H100ish)


def test_host_model_defaults_to_ideal():
    m = HostInitiationModel()
    assert m.delay_ps() == 0
    assert m.profile == "ideal"
    gin = HostInitiationModel(initiation_delay_ps=800_000_000, profile="gin")
    assert gin.delay_ps() == 800_000_000
