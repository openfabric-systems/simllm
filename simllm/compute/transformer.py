"""Framework-neutral transformer step-cost model.

One transformer geometry (:class:`ModelDims`, dense or MoE), one
fused-kernel rendering of an engine step (:func:`step_kernel`), a
kernel-family decomposition of the same step (:func:`step_kernels`) and the
GPU envelopes the roofline provider prices it against. Frontend adapters
(vLLM, SGLang) build the :class:`ModelDims` from their own config objects
and share everything else, so the two frontends can never drift apart on
what a step costs.

Why the family decomposition exists (COMP-1): the offline SASS calibration
pipeline (Accel-Sim trace-driven simulation, see docs/modules/compute.md)
produces per-kernel duration tables, and a single fused "llm_step" kernel
is not a unit any tracer or simulator ever sees. :func:`step_kernels`
splits the fused step into named families (attention GEMMs, attention
score/value passes, MLP GEMMs, LM head, KV-cache reads) whose shapes map
onto microbenchmarkable kernels, so offline SASS runs can populate a
``ProfileTableProvider`` table per family and the step loop can sum
per-family estimates instead of pricing one opaque blob. The decomposition
is exact by construction: family flops and bytes sum to the fused kernel's
totals, with weights counted once, in the family that streams them.
Families aggregate over all layers of the step; per-layer (per-invocation)
shapes are COMP-6.
"""

from __future__ import annotations

from dataclasses import dataclass

from simllm.compute.provider import ComputeProvider, GpuSpec, KernelSpec
from simllm.core import StepRecord

from .host import HostInitiationModel

#: Dense (non-sparse) BF16 tensor-core peak FLOP/s and HBM bytes/s.
GPU_ENVELOPES: dict[str, GpuSpec] = {
    "gtx1660-ti-sm75": GpuSpec(
        name="gtx1660-ti-sm75",
        peak_flops=5.437e12,
        mem_bandwidth=288e9,
    ),
    "a100": GpuSpec(name="a100", peak_flops=312e12, mem_bandwidth=2.039e12),
    "h100": GpuSpec(name="h100", peak_flops=989.5e12, mem_bandwidth=3.35e12),
    "h200": GpuSpec(name="h200", peak_flops=989.5e12, mem_bandwidth=4.8e12),
    "b100": GpuSpec(name="b100", peak_flops=1.8e15, mem_bandwidth=8.0e12),
    "b200": GpuSpec(name="b200", peak_flops=2.25e15, mem_bandwidth=8.0e12),
}


@dataclass(frozen=True)
class ModelDims:
    """Per-rank transformer geometry for the analytical step estimate.

    Every field is already sharded, i.e. it describes what one GPU owns:
    ``num_layers`` is the pipeline slice, ``num_heads`` / ``num_kv_heads`` /
    ``intermediate_size`` are the tensor-parallel slice.

    MoE geometry (all default to the dense model, which preserves the
    pre-MoE behavior of every property exactly): when ``num_experts > 0``
    every layer's MLP is a routed mixture. ``moe_intermediate_size`` is the
    intermediate size of ONE expert, already tensor-parallel sharded like
    ``intermediate_size`` (which is ignored for MoE dims). ``top_k`` experts
    activate per token, so the per-token MLP flops multiply by ``top_k``.
    Under expert parallelism a rank does not store all experts:
    ``local_num_experts`` is the number of experts per MoE layer resident
    on THIS rank (what this GPU's HBM actually holds and streams each
    step); the default 0 means "all ``num_experts`` are local", i.e. no
    expert parallelism. Weight bytes count only the resident experts;
    activated-expert flops are counted in full on this rank, which assumes
    perfectly balanced routing (COMP-7): with uniform routing each rank
    computes its own tokens' worth of expert work in aggregate.

    ``dtype_bytes`` is the activation dtype width. Weights and KV cache may
    be narrower (quantized checkpoints, fp8 KV cache), so they carry their
    own element widths; ``None`` means "same as the activation dtype".
    ``defaulted_fields`` names every geometry field the adapter's config
    reader fell back to a default for, so a consumer can see which latency
    terms rest on a guessed geometry.
    """

    num_layers: int
    hidden_size: int
    intermediate_size: int
    num_heads: int
    num_kv_heads: int
    head_size: int
    vocab_size: int
    dtype_bytes: int = 2
    weight_dtype_bytes: float | None = None
    kv_dtype_bytes: float | None = None
    defaulted_fields: tuple[str, ...] = ()
    #: total experts per MoE layer across the EP group; 0 = dense model
    num_experts: int = 0
    #: experts activated per token by the router (MoE only)
    top_k: int = 0
    #: per-expert intermediate size, already TP-sharded (MoE only)
    moe_intermediate_size: int | None = None
    #: experts per MoE layer resident on this rank; 0 = all of them
    local_num_experts: int = 0

    def __post_init__(self) -> None:
        if self.num_experts > 0:
            if self.moe_intermediate_size is None:
                raise ValueError("MoE dims (num_experts > 0) need moe_intermediate_size")
            if not 1 <= self.top_k <= self.num_experts:
                raise ValueError("MoE dims need 1 <= top_k <= num_experts")
            if self.local_num_experts > self.num_experts:
                raise ValueError("local_num_experts cannot exceed num_experts")

    @property
    def weight_element_bytes(self) -> float:
        """Bytes per weight element (quantization-aware)."""
        if self.weight_dtype_bytes is not None:
            return self.weight_dtype_bytes
        return float(self.dtype_bytes)

    @property
    def kv_element_bytes(self) -> float:
        """Bytes per KV-cache element (cache-dtype-aware)."""
        if self.kv_dtype_bytes is not None:
            return self.kv_dtype_bytes
        return float(self.dtype_bytes)

    @property
    def attention_params(self) -> int:
        """QKV and output projection parameters on this rank, all layers."""
        q_dim = self.num_heads * self.head_size
        kv_dim = self.num_kv_heads * self.head_size
        per_layer = self.hidden_size * (q_dim + 2 * kv_dim) + q_dim * self.hidden_size
        return per_layer * self.num_layers

    @property
    def mlp_params(self) -> int:
        """Dense gate/up/down projection parameters on this rank, all layers.

        The dense-model MLP geometry; MoE dims use :attr:`mlp_active_params`
        (flops) and :attr:`mlp_resident_params` (weight bytes) instead.
        """
        return 3 * self.hidden_size * self.intermediate_size * self.num_layers

    @property
    def resident_experts(self) -> int:
        """Experts per MoE layer stored on this rank (0 declared = all)."""
        if self.local_num_experts > 0:
            return self.local_num_experts
        return self.num_experts

    @property
    def mlp_active_params(self) -> int:
        """MLP parameters each new token multiplies against, all layers.

        Dense: the full MLP. MoE: ``top_k`` experts' gate/up/down
        projections per token (balanced-routing assumption, COMP-7).
        """
        if self.num_experts > 0:
            assert self.moe_intermediate_size is not None
            return 3 * self.hidden_size * self.moe_intermediate_size * self.top_k * self.num_layers
        return self.mlp_params

    @property
    def mlp_resident_params(self) -> int:
        """MLP parameters stored (and streamed per step) on this rank.

        Dense: the full MLP. MoE: the :attr:`resident_experts` this rank
        owns under expert parallelism; the roofline streams exactly these
        bytes once per step, matching how a decode step reads every
        resident expert's weights regardless of routing.
        """
        if self.num_experts > 0:
            assert self.moe_intermediate_size is not None
            return (
                3
                * self.hidden_size
                * self.moe_intermediate_size
                * self.resident_experts
                * self.num_layers
            )
        return self.mlp_params

    @property
    def weight_bytes(self) -> int:
        return int((self.attention_params + self.mlp_resident_params) * self.weight_element_bytes)

    @property
    def lm_head_bytes(self) -> int:
        return int(self.hidden_size * self.vocab_size * self.weight_element_bytes)


def step_shape(record: StepRecord) -> tuple[int, int, int]:
    """(new_tokens, kv_read_tokens, attention_pairs) of one step record.

    ``attention_pairs`` counts ``n * (prior_context + n/2)`` query-key pairs
    per request so a prefill chunk pays its triangular share. The fused
    kernel and the family decomposition both read these counters from here,
    so their shapes cannot drift apart.
    """
    new_tokens = 0
    kv_read_tokens = 0
    attention_pairs = 0
    for request in record.scheduled:
        n = request.num_new_tokens
        prior = max(request.context_length - n, 0)
        new_tokens += n
        kv_read_tokens += request.context_length
        attention_pairs += n * prior + n * n // 2
    return new_tokens, kv_read_tokens, attention_pairs


def step_kernel(dims: ModelDims, record: StepRecord, num_sampled: int) -> KernelSpec:
    """One engine step as a single fused kernel for a compute provider.

    FLOPs are 2 per multiply-accumulate over (a) the projections for every
    scheduled token (dense MLP, or ``top_k`` activated experts per token for
    MoE dims), (b) the attention score and value passes over the
    query-key-pair count of :func:`step_shape`, and (c) the LM head for the
    tokens that actually sample. Bytes are the resident weights (read once
    per step; for MoE only the experts resident on this rank), the LM head,
    and the KV cache of every scheduled request's context, which is what
    makes decode steps memory-bound in the roofline.
    """
    new_tokens, kv_read_tokens, attention_pairs = step_shape(record)
    dense_flops = 2 * new_tokens * (dims.attention_params + dims.mlp_active_params)
    attn_flops = 4 * attention_pairs * dims.num_layers * dims.num_heads * dims.head_size
    head_flops = 2 * num_sampled * dims.hidden_size * dims.vocab_size
    kv_bytes = (
        2
        * kv_read_tokens
        * dims.num_layers
        * dims.num_kv_heads
        * dims.head_size
        * dims.kv_element_bytes
    )
    return KernelSpec(
        name="llm_step",
        flops=float(dense_flops + attn_flops + head_flops),
        bytes_moved=float(dims.weight_bytes + dims.lm_head_bytes + kv_bytes),
        config=(
            ("new_tokens", new_tokens),
            ("sampled", num_sampled),
            ("kv_tokens", kv_read_tokens),
        ),
        family_kernels=tuple(step_kernels(dims, record, num_sampled)),
    )


def step_kernels(dims: ModelDims, record: StepRecord, num_sampled: int) -> list[KernelSpec]:
    """The fused step split into named kernel families (COMP-1 groundwork).

    Families, in rough execution order, with the shape key each carries:

    - ``attn_gemm``: QKV and output projections for every new token
      (``new_tokens``); streams the attention weights.
    - ``attn_score``: the attention score and value passes over the
      query-key pairs (``new_tokens``, ``kv_tokens``); zero weight bytes,
      its KV traffic lives in ``kv_read``.
    - ``mlp_gemm``: dense MLP, or ``top_k`` activated experts per token for
      MoE dims (``new_tokens``); streams the MLP weights resident on this
      rank.
    - ``lm_head``: the sampling projection (``sampled``); streams the LM
      head weights.
    - ``kv_read``: the KV-cache read bytes of every scheduled context
      (``kv_tokens``); zero flops.

    Exactness invariant (unit-tested): family flops sum to the fused
    :func:`step_kernel` flops exactly, and family ``bytes_moved`` sum to
    the fused ``bytes_moved`` exactly. Weights are counted once, in the
    family that streams them; the MLP share is derived as
    ``dims.weight_bytes`` minus the attention share so quantized
    (fractional-byte) weights cannot lose a rounding unit against the fused
    total. Families aggregate over all layers of the step; per-layer
    (per-invocation) kernel shapes for SASS table keys are COMP-6.
    """
    new_tokens, kv_read_tokens, attention_pairs = step_shape(record)
    attn_weight_bytes = int(dims.attention_params * dims.weight_element_bytes)
    mlp_weight_bytes = dims.weight_bytes - attn_weight_bytes
    kv_bytes = (
        2
        * kv_read_tokens
        * dims.num_layers
        * dims.num_kv_heads
        * dims.head_size
        * dims.kv_element_bytes
    )
    return [
        KernelSpec(
            name="attn_gemm",
            flops=float(2 * new_tokens * dims.attention_params),
            bytes_moved=float(attn_weight_bytes),
            config=(("new_tokens", new_tokens),),
        ),
        KernelSpec(
            name="attn_score",
            flops=float(4 * attention_pairs * dims.num_layers * dims.num_heads * dims.head_size),
            bytes_moved=0.0,
            config=(("new_tokens", new_tokens), ("kv_tokens", kv_read_tokens)),
        ),
        KernelSpec(
            name="mlp_gemm",
            flops=float(2 * new_tokens * dims.mlp_active_params),
            bytes_moved=float(mlp_weight_bytes),
            config=(("new_tokens", new_tokens),),
        ),
        KernelSpec(
            name="lm_head",
            flops=float(2 * num_sampled * dims.hidden_size * dims.vocab_size),
            bytes_moved=float(dims.lm_head_bytes),
            config=(("sampled", num_sampled),),
        ),
        KernelSpec(
            name="kv_read",
            flops=0.0,
            bytes_moved=float(kv_bytes),
            config=(("kv_tokens", kv_read_tokens),),
        ),
    ]


def estimate_step_latency_ps(
    dims: ModelDims,
    record: StepRecord,
    num_sampled: int,
    provider: ComputeProvider,
    gpu: GpuSpec,
    host_model: HostInitiationModel,
) -> int:
    """Simulated duration after one host-launch composition."""
    kernel = step_kernel(dims, record, num_sampled)
    return host_model.represented_estimate(provider.estimate(kernel, gpu), gpu).duration_ps
