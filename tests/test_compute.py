import pytest

from simllm.compute import (
    GPU_ENVELOPES,
    PROFILE_TABLE_SCHEMA,
    GpuSpec,
    HostInitiationModel,
    KernelSpec,
    ModelDims,
    ProfileTableProvenance,
    ProfileTableProvider,
    RooflineProvider,
    step_kernel,
    step_kernels,
)
from simllm.core import RequestPhase, ScheduledRequest, StepRecord

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


def test_compute_provider_layer_breakdown_is_optional():
    provider = RooflineProvider(efficiency=1.0)
    kernel = KernelSpec(name="gemm", flops=1e9, bytes_moved=1e6)
    assert provider.estimate_layers(kernel, H100ish, num_layers=2) is None


def test_enabled_roofline_needs_family_metadata():
    provider = RooflineProvider(efficiency=1.0, enable_layer_breakdown=True)
    kernel = KernelSpec(name="gemm", flops=1e9, bytes_moved=1e6)
    assert provider.estimate_layers(kernel, H100ish, num_layers=2) is None


def test_profile_table():
    cfg = (("batch_tokens", 512), ("hidden", 7168))
    provider = ProfileTableProvider({("moe_forward", cfg, "h100-bf16"): 123_000_000})
    est = provider.estimate(KernelSpec("moe_forward", 0, 0, cfg), H100ish)
    assert est.duration_ps == 123_000_000
    assert est.bound == "measured"
    assert provider.estimate_layers(KernelSpec("moe_forward", 0, 0, cfg), H100ish, 2) is None
    with pytest.raises(KeyError):
        provider.estimate(KernelSpec("moe_forward", 0, 0, ()), H100ish)


def test_host_model_defaults_to_ideal():
    m = HostInitiationModel()
    assert m.delay_ps() == 0
    assert m.profile == "ideal"
    gin = HostInitiationModel(initiation_delay_ps=800_000_000, profile="gin")
    assert gin.delay_ps() == 800_000_000


# ---- profile-table artifact and interpolation (COMP-1 table half) ----

def _table_provider(overrides=None):
    """A tiny synthetic table for one kernel on one gpu, 1 numeric axis."""
    entries = {
        ("gemm", (("new_tokens", 8),), "h100-bf16"): (100_000, 0.05),
        ("gemm", (("new_tokens", 32),), "h100-bf16"): (400_000, 0.10),
    }
    entries.update(overrides or {})
    return ProfileTableProvider(entries)


def test_profile_table_interpolates_log_linearly():
    provider = _table_provider()
    # new_tokens=16 is the log-space midpoint of 8 and 32, so the duration
    # is the geometric mean sqrt(100e3 * 400e3) = 200e3
    est = provider.estimate(
        KernelSpec("gemm", 0, 0, (("new_tokens", 16),)), H100ish)
    assert est.duration_ps == 200_000
    assert est.bound == "interpolated"
    # inflated to the 0.15 floor (neighbors are 0.05 and 0.10)
    assert est.uncertainty == 0.15


def test_profile_table_interpolation_inherits_worse_neighbor_uncertainty():
    provider = _table_provider({
        ("gemm", (("new_tokens", 32),), "h100-bf16"): (400_000, 0.30),
    })
    est = provider.estimate(
        KernelSpec("gemm", 0, 0, (("new_tokens", 16),)), H100ish)
    assert est.uncertainty == 0.30


def test_profile_table_out_of_range_raises():
    provider = _table_provider()
    for q in (4, 64):
        with pytest.raises(KeyError):
            provider.estimate(
                KernelSpec("gemm", 0, 0, (("new_tokens", q),)), H100ish)


def test_profile_table_multi_axis_miss_raises():
    # entries differ from the query on BOTH axes: 1D-per-axis (COMP-4)
    provider = ProfileTableProvider({
        ("attn", (("new_tokens", 8), ("kv_tokens", 100)), "h100-bf16"): 1_000,
        ("attn", (("new_tokens", 32), ("kv_tokens", 200)), "h100-bf16"): 4_000,
    })
    with pytest.raises(KeyError):
        provider.estimate(
            KernelSpec("attn", 0, 0, (("new_tokens", 16), ("kv_tokens", 150))),
            H100ish)


def test_profile_table_interpolates_one_axis_of_two():
    # second axis pinned: interpolation along new_tokens alone is allowed
    provider = ProfileTableProvider({
        ("attn", (("new_tokens", 8), ("kv_tokens", 100)), "h100-bf16"): 1_000,
        ("attn", (("new_tokens", 32), ("kv_tokens", 100)), "h100-bf16"): 4_000,
    })
    est = provider.estimate(
        KernelSpec("attn", 0, 0, (("new_tokens", 16), ("kv_tokens", 100))),
        H100ish)
    assert est.duration_ps == 2_000
    assert est.bound == "interpolated"


def test_profile_table_json_round_trip(tmp_path):
    provenance = ProfileTableProvenance(
        source="accel-sim", version="v1.3.0", gpu="h100-bf16",
        created="2026-08-04")
    provider = _table_provider()
    provider.provenance = provenance
    path = provider.save(tmp_path / "table.json")

    loaded = ProfileTableProvider.load(path)
    assert loaded.provenance == provenance
    kernel = KernelSpec("gemm", 0, 0, (("new_tokens", 8),))
    original = provider.estimate(kernel, H100ish)
    round_tripped = loaded.estimate(kernel, H100ish)
    assert round_tripped == original
    assert round_tripped.bound == "measured"
    # interpolation works identically on the loaded table
    mid = KernelSpec("gemm", 0, 0, (("new_tokens", 16),))
    assert loaded.estimate(mid, H100ish) == provider.estimate(mid, H100ish)


@pytest.mark.parametrize("reference", (None, 7, True, {}))
def test_profile_table_rejects_non_string_provenance_references(reference):
    payload = {
        "source": "capture",
        "version": "v1",
        "gpu": "h100-bf16",
        "created": "2026-08-07",
        "references": [reference],
    }

    with pytest.raises(ValueError, match=r"references\[0\].*nonblank string"):
        ProfileTableProvenance.from_json(payload)


def test_profile_table_save_requires_provenance(tmp_path):
    with pytest.raises(ValueError, match="provenance"):
        _table_provider().save(tmp_path / "table.json")


def test_profile_table_load_rejects_wrong_schema(tmp_path):
    import json

    path = tmp_path / "bad.json"
    path.write_text(json.dumps({
        "schema": "simllm-profile-table-v0",
        "provenance": {"source": "capture", "version": "x", "gpu": "g",
                       "created": "2026-08-04"},
        "entries": [],
    }))
    with pytest.raises(ValueError, match="schema"):
        ProfileTableProvider.load(path)


def test_profile_table_load_rejects_missing_provenance_field(tmp_path):
    import json

    path = tmp_path / "bad.json"
    path.write_text(json.dumps({
        "schema": PROFILE_TABLE_SCHEMA,
        "provenance": {"source": "capture", "version": "x", "gpu": "g"},
        "entries": [],
    }))
    with pytest.raises(ValueError, match="created"):
        ProfileTableProvider.load(path)


# ---- kernel-family decomposition (COMP-1 groundwork) ----

DENSE_DIMS = ModelDims(
    num_layers=4,
    hidden_size=64,
    intermediate_size=256,
    num_heads=8,
    num_kv_heads=2,
    head_size=8,
    vocab_size=1000,
    dtype_bytes=2,
)

MOE_DIMS = ModelDims(
    num_layers=4,
    hidden_size=64,
    intermediate_size=256,
    num_heads=8,
    num_kv_heads=2,
    head_size=8,
    vocab_size=1000,
    dtype_bytes=2,
    num_experts=16,
    top_k=4,
    moe_intermediate_size=32,
    local_num_experts=2,
)


def mixed_record() -> StepRecord:
    return StepRecord(step_index=0, virtual_time_ps=0, scheduled=[
        ScheduledRequest("p", RequestPhase.PREFILL, num_new_tokens=64,
                         context_length=64),
        ScheduledRequest("d", RequestPhase.DECODE, num_new_tokens=1,
                         context_length=100),
    ])


@pytest.mark.parametrize("dims", [DENSE_DIMS, MOE_DIMS], ids=["dense", "moe"])
def test_step_kernels_sum_to_fused_exactly(dims):
    record = mixed_record()
    fused = step_kernel(dims, record, num_sampled=2)
    families = step_kernels(dims, record, num_sampled=2)
    assert sum(k.flops for k in families) == fused.flops
    assert sum(k.bytes_moved for k in families) == fused.bytes_moved
    assert fused.family_kernels == tuple(families)


def test_step_kernels_sum_exact_with_fractional_weight_bytes():
    # 4-bit quantized weights: the int() truncation of the fused
    # weight_bytes must not lose a unit against the per-family split
    quantized = ModelDims(
        num_layers=3, hidden_size=63, intermediate_size=253, num_heads=7,
        num_kv_heads=7, head_size=9, vocab_size=997, dtype_bytes=2,
        weight_dtype_bytes=0.5,
    )
    record = mixed_record()
    fused = step_kernel(quantized, record, num_sampled=1)
    families = step_kernels(quantized, record, num_sampled=1)
    assert sum(k.bytes_moved for k in families) == fused.bytes_moved
    assert sum(k.flops for k in families) == fused.flops


def test_step_kernels_families_and_configs():
    record = mixed_record()
    families = {k.name: k for k in step_kernels(DENSE_DIMS, record, num_sampled=2)}
    assert set(families) == {"attn_gemm", "attn_score", "mlp_gemm", "lm_head", "kv_read"}
    new_tokens = 65
    kv_tokens = 164
    assert families["attn_gemm"].config == (("new_tokens", new_tokens),)
    assert families["mlp_gemm"].config == (("new_tokens", new_tokens),)
    assert families["attn_score"].config == (
        ("new_tokens", new_tokens), ("kv_tokens", kv_tokens))
    assert families["lm_head"].config == (("sampled", 2),)
    assert families["kv_read"].config == (("kv_tokens", kv_tokens),)
    # kv_read is pure bytes; attn_score carries no weight bytes
    assert families["kv_read"].flops == 0.0
    assert families["kv_read"].bytes_moved > 0.0
    assert families["attn_score"].bytes_moved == 0.0
    # weights are counted once: the two GEMM families split weight_bytes
    assert (families["attn_gemm"].bytes_moved + families["mlp_gemm"].bytes_moved
            == DENSE_DIMS.weight_bytes)
    assert families["lm_head"].bytes_moved == DENSE_DIMS.lm_head_bytes


@pytest.mark.parametrize(
    ("num_layers", "expected_fused_ps", "expected_layer_ps"),
    [
        (2, 35_474, (14_811, 20_663)),
        (4, 65_097, (14_811, 14_811, 14_812, 20_663)),
    ],
)
def test_roofline_layer_breakdown_matches_frozen_memory_bound_rows(
    num_layers, expected_fused_ps, expected_layer_ps
):
    dims = ModelDims(
        num_layers=num_layers,
        hidden_size=64,
        intermediate_size=128,
        num_heads=4,
        num_kv_heads=4,
        head_size=16,
        vocab_size=256,
        dtype_bytes=2,
    )
    record = StepRecord(
        0,
        0,
        [ScheduledRequest("d", RequestPhase.DECODE, 1, context_length=4)],
        num_sampled=1,
    )
    kernel = step_kernel(dims, record, num_sampled=1)
    provider = RooflineProvider(efficiency=0.7, enable_layer_breakdown=True)

    fused = provider.estimate(kernel, GPU_ENVELOPES["b100"])
    estimates = provider.estimate_layers(kernel, GPU_ENVELOPES["b100"], num_layers)

    assert fused.bound == "memory"
    assert fused.duration_ps == expected_fused_ps
    assert estimates is not None
    assert tuple(estimate.duration_ps for estimate in estimates) == expected_layer_ps
    assert sum(estimate.duration_ps for estimate in estimates) == fused.duration_ps
    assert all(estimate.bound == fused.bound for estimate in estimates)
    assert all(estimate.uncertainty == fused.uncertainty for estimate in estimates)


def test_roofline_compute_bound_split_puts_lm_head_on_last_layer():
    families = (
        KernelSpec("body", flops=90.0, bytes_moved=0.0),
        KernelSpec("lm_head", flops=30.0, bytes_moved=0.0),
    )
    kernel = KernelSpec(
        "llm_step",
        flops=120.0,
        bytes_moved=0.0,
        family_kernels=families,
    )
    gpu = GpuSpec("one-flop-per-ps", peak_flops=1e12, mem_bandwidth=1e12)
    provider = RooflineProvider(efficiency=1.0, enable_layer_breakdown=True)

    estimates = provider.estimate_layers(kernel, gpu, num_layers=3)

    assert estimates is not None
    assert tuple(estimate.duration_ps for estimate in estimates) == (30, 30, 60)


def test_roofline_layer_breakdown_rejects_invalid_contract_inputs():
    family = KernelSpec("body", flops=10.0, bytes_moved=10.0)
    kernel = KernelSpec(
        "llm_step",
        flops=11.0,
        bytes_moved=10.0,
        family_kernels=(family,),
    )
    provider = RooflineProvider(efficiency=1.0, enable_layer_breakdown=True)

    with pytest.raises(ValueError, match="conserve"):
        provider.estimate_layers(kernel, H100ish, num_layers=2)
    with pytest.raises(ValueError, match="nonnegative"):
        provider.estimate_layers(
            KernelSpec(
                "llm_step",
                flops=-1.0,
                bytes_moved=1.0,
                family_kernels=(
                    KernelSpec("body", flops=-1.0, bytes_moved=1.0),
                ),
            ),
            H100ish,
            num_layers=2,
        )
    with pytest.raises(ValueError, match="positive"):
        provider.estimate_layers(
            KernelSpec(
                "llm_step",
                flops=10.0,
                bytes_moved=10.0,
                family_kernels=(family,),
            ),
            H100ish,
            num_layers=0,
        )


# ---- MoE geometry on ModelDims ----

def test_moe_dims_defaults_preserve_dense_behavior():
    assert DENSE_DIMS.num_experts == 0
    assert DENSE_DIMS.mlp_active_params == DENSE_DIMS.mlp_params
    assert DENSE_DIMS.mlp_resident_params == DENSE_DIMS.mlp_params
    assert DENSE_DIMS.weight_bytes == int(
        (DENSE_DIMS.attention_params + DENSE_DIMS.mlp_params) * 2)


def test_moe_dims_flops_and_resident_weights():
    # per token: top_k experts of 3 * hidden * moe_intermediate params
    assert MOE_DIMS.mlp_active_params == 3 * 64 * 32 * 4 * 4
    # resident: local_num_experts experts per layer
    assert MOE_DIMS.mlp_resident_params == 3 * 64 * 32 * 2 * 4
    record = mixed_record()
    fused = step_kernel(MOE_DIMS, record, num_sampled=2)
    dense_equivalent = step_kernel(DENSE_DIMS, record, num_sampled=2)
    # attention terms identical; only the MLP terms moved
    delta_flops = fused.flops - dense_equivalent.flops
    assert delta_flops == 2 * 65 * (MOE_DIMS.mlp_active_params - DENSE_DIMS.mlp_params)


def test_moe_dims_local_zero_means_all_experts():
    all_local = ModelDims(
        num_layers=4, hidden_size=64, intermediate_size=256, num_heads=8,
        num_kv_heads=2, head_size=8, vocab_size=1000, dtype_bytes=2,
        num_experts=16, top_k=4, moe_intermediate_size=32,
    )
    assert all_local.resident_experts == 16
    assert all_local.mlp_resident_params == 3 * 64 * 32 * 16 * 4


def test_moe_dims_validation():
    kwargs = {
        "num_layers": 4, "hidden_size": 64, "intermediate_size": 256,
        "num_heads": 8, "num_kv_heads": 2, "head_size": 8,
        "vocab_size": 1000, "dtype_bytes": 2,
    }
    with pytest.raises(ValueError, match="moe_intermediate_size"):
        ModelDims(**kwargs, num_experts=16, top_k=4)
    with pytest.raises(ValueError, match="top_k"):
        ModelDims(**kwargs, num_experts=16, top_k=0, moe_intermediate_size=32)
    with pytest.raises(ValueError, match="top_k"):
        ModelDims(**kwargs, num_experts=16, top_k=17, moe_intermediate_size=32)
    with pytest.raises(ValueError, match="local_num_experts"):
        ModelDims(**kwargs, num_experts=16, top_k=4, moe_intermediate_size=32,
                  local_num_experts=17)
