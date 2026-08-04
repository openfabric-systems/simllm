"""Framework-neutral transformer step-cost model.

One dense-transformer geometry (:class:`ModelDims`), one fused-kernel
rendering of an engine step (:func:`step_kernel`) and the GPU envelopes the
roofline provider prices it against. Frontend adapters (vLLM, SGLang) build
the :class:`ModelDims` from their own config objects and share everything
else, so the two frontends can never drift apart on what a step costs.
"""

from __future__ import annotations

from dataclasses import dataclass

from simllm.compute.provider import ComputeProvider, GpuSpec, KernelSpec
from simllm.core import StepRecord

from .host import HostInitiationModel

#: Dense (non-sparse) BF16 tensor-core peak FLOP/s and HBM bytes/s.
GPU_ENVELOPES: dict[str, GpuSpec] = {
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
    ``intermediate_size`` are the tensor-parallel slice. MoE routing is not
    modeled (adapter task VLLM-6): an MoE model is costed as its dense
    attention plus one expert's worth of MLP per token.

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
        """Gate, up and down projection parameters on this rank, all layers."""
        return 3 * self.hidden_size * self.intermediate_size * self.num_layers

    @property
    def weight_bytes(self) -> int:
        return int((self.attention_params + self.mlp_params) * self.weight_element_bytes)

    @property
    def lm_head_bytes(self) -> int:
        return int(self.hidden_size * self.vocab_size * self.weight_element_bytes)


def step_kernel(dims: ModelDims, record: StepRecord, num_sampled: int) -> KernelSpec:
    """One engine step as a single fused kernel for a compute provider.

    FLOPs are 2 per multiply-accumulate over (a) the dense projections for
    every scheduled token, (b) the attention score and value passes, counted
    as ``n * (prior_context + n/2)`` query-key pairs per request so a prefill
    chunk pays its triangular share, and (c) the LM head for the tokens that
    actually sample. Bytes are the resident weights (read once per step), the
    LM head, and the KV cache of every scheduled request's context, which is
    what makes decode steps memory-bound in the roofline.
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

    dense_flops = 2 * new_tokens * (dims.attention_params + dims.mlp_params)
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
    )


def estimate_step_latency_ps(
    dims: ModelDims,
    record: StepRecord,
    num_sampled: int,
    provider: ComputeProvider,
    gpu: GpuSpec,
    host_model: HostInitiationModel,
) -> int:
    """Simulated duration of one step: kernel estimate plus host initiation."""
    kernel = step_kernel(dims, record, num_sampled)
    return provider.estimate(kernel, gpu).duration_ps + host_model.delay_ps()
