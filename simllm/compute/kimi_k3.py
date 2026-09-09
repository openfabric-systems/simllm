"""Typed Kimi K3 text geometry, logical weights and retained state shapes.

These records identify mathematics. Checkpoint encoding, a device's resident
layout and its per-step memory traffic are separate facts. No service estimate
or framework import belongs in this geometry authority.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from math import prod
from typing import Any

KIMI_K3_GEOMETRY_SCHEMA = "simllm-kimi-k3-text-geometry-v1"
MXFP4_GROUP32 = "mxfp4-e2m1-group32-e8m0"


def _positive(name: str, value: object) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonnegative(name: str, value: object) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _fields(value: object, names: set[str], path: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != names:
        raise ValueError(f"{path} must carry exactly {sorted(names)}")
    return value


def _scalars(value: object) -> dict[str, Any]:
    return {field.name: getattr(value, field.name) for field in fields(value)}


@dataclass(frozen=True, slots=True)
class KdaSpec:
    heads: int
    head_dim: int
    convolution_width: int
    full_rank_output_gate: bool
    gate_lower_bound: str
    recurrent_dtype: str
    convolution_state_dtype: str

    def __post_init__(self) -> None:
        for name in ("heads", "head_dim", "convolution_width"):
            _positive(f"KDA.{name}", getattr(self, name))
        if self.convolution_width < 2:
            raise ValueError("KDA convolution must retain at least one prior position")
        if self.full_rank_output_gate is not True or self.gate_lower_bound != "-5":
            raise ValueError("K3 requires the full output gate and bounded per-key forget gate")
        if (self.recurrent_dtype, self.convolution_state_dtype) != ("float32", "bfloat16"):
            raise ValueError("this structural envelope declares FP32 recurrence and BF16 convolution state")

    @property
    def projection_width(self) -> int:
        return self.heads * self.head_dim

    def to_obj(self) -> dict[str, Any]:
        return _scalars(self)

    @classmethod
    def from_obj(cls, value: object) -> KdaSpec:
        return cls(**_fields(value, {f.name for f in fields(cls)}, "kda"))


@dataclass(frozen=True, slots=True)
class MlaSpec:
    heads: int
    query_rank: int
    kv_rank: int
    score_nope_dim: int
    score_extra_dim: int
    value_dim: int
    skip_rotary: bool
    output_gate: bool
    cache_dtype: str

    def __post_init__(self) -> None:
        for name in ("heads", "query_rank", "kv_rank", "score_nope_dim", "score_extra_dim", "value_dim"):
            _positive(f"MLA.{name}", getattr(self, name))
        if self.skip_rotary is not True or self.output_gate is not True:
            raise ValueError("K3 MLA requires NoPE and the full output gate")
        if self.cache_dtype != "bfloat16":
            raise ValueError("this structural envelope declares BF16 compressed MLA state")

    @property
    def expanded_score_dim(self) -> int:
        return self.score_nope_dim + self.score_extra_dim

    @property
    def compressed_width(self) -> int:
        return self.kv_rank + self.score_extra_dim

    def to_obj(self) -> dict[str, Any]:
        return _scalars(self)

    @classmethod
    def from_obj(cls, value: object) -> MlaSpec:
        return cls(**_fields(value, {f.name for f in fields(cls)}, "mla"))


@dataclass(frozen=True, slots=True)
class LatentMoeSpec:
    experts: int
    selected_experts: int
    latent_width: int
    intermediate_size: int
    shared_experts: int
    normalize_latent_output: bool
    router_activation: str
    topk_method: str
    renormalize: bool
    expert_groups: int
    topk_groups: int
    routed_scale: str
    activation: str
    situ_beta: str
    situ_linear_beta: str

    def __post_init__(self) -> None:
        for name in ("experts", "selected_experts", "latent_width", "intermediate_size", "shared_experts", "expert_groups", "topk_groups"):
            _positive(f"MoE.{name}", getattr(self, name))
        if self.selected_experts > self.experts:
            raise ValueError("selected experts exceed the resident expert population")
        if self.normalize_latent_output is not True or self.renormalize is not True:
            raise ValueError("K3 requires latent-output normalization and normalized routing weights")
        if (self.router_activation, self.topk_method, self.routed_scale) != ("sigmoid", "noaux_tc", "1"):
            raise ValueError("unsupported K3 routing operator")
        if (self.expert_groups, self.topk_groups) != (1, 1):
            raise ValueError("this K3 envelope has one expert-selection group")
        if (self.activation, self.situ_beta, self.situ_linear_beta) != ("situ", "4", "25"):
            raise ValueError("unsupported K3 SiTU activation")

    @property
    def expert_parameters(self) -> int:
        return 3 * self.latent_width * self.intermediate_size

    @property
    def expert_encoding_floor_bytes(self) -> int:
        values = self.expert_parameters
        return (values + 1) // 2 + (values + 31) // 32

    def to_obj(self) -> dict[str, Any]:
        return _scalars(self)

    @classmethod
    def from_obj(cls, value: object) -> LatentMoeSpec:
        return cls(**_fields(value, {f.name for f in fields(cls)}, "moe"))


@dataclass(frozen=True, slots=True)
class KimiK3LayerSpec:
    index: int
    attention: str
    feed_forward: str
    residual_bank_before: int
    writes_residual_snapshot: bool


@dataclass(frozen=True, slots=True)
class LogicalWeightShape:
    """A mathematical tensor shape, not a resident or checkpoint allocation."""

    name: str
    dimensions: tuple[int, ...]
    layer: int | None

    @property
    def parameters(self) -> int:
        return prod(self.dimensions)

    def to_obj(self) -> dict[str, Any]:
        return {"name": self.name, "dimensions": list(self.dimensions),
                "layer": self.layer, "parameters": self.parameters,
                "scope": "logical-weight-shape"}


@dataclass(frozen=True, slots=True)
class KimiK3Spec:
    hidden_size: int
    vocab_size: int
    dense_intermediate_size: int
    dense_prefix_layers: int
    layer_types: tuple[str, ...]
    residual_block_size: int
    kda: KdaSpec
    mla: MlaSpec
    moe: LatentMoeSpec
    rms_norm_epsilon: str
    activation_dtype: str
    tied_embeddings: bool
    max_context: int
    checkpoint_quantization: str

    def __post_init__(self) -> None:
        for name in ("hidden_size", "vocab_size", "dense_intermediate_size", "residual_block_size", "max_context"):
            _positive(f"K3.{name}", getattr(self, name))
        _positive("K3.dense_prefix_layers", self.dense_prefix_layers)
        if not isinstance(self.layer_types, tuple) or not self.layer_types:
            raise ValueError("K3 layer types must be a nonempty immutable ordered tuple")
        if any(item not in {"kda", "mla"} for item in self.layer_types):
            raise ValueError("K3 layer types contain an unknown attention mechanism")
        if set(self.layer_types) != {"kda", "mla"} or self.dense_prefix_layers >= self.num_layers:
            raise ValueError("K3 requires both attention mechanisms and a routed suffix")
        for value, kind in ((self.kda, KdaSpec), (self.mla, MlaSpec), (self.moe, LatentMoeSpec)):
            if not isinstance(value, kind):
                raise TypeError(f"K3 geometry requires typed {kind.__name__}")
        if self.activation_dtype != "bfloat16" or self.tied_embeddings is not False:
            raise ValueError("this K3 text envelope requires BF16 activations and untied embeddings")
        if self.rms_norm_epsilon != "1/100000":
            raise ValueError("K3 model RMS normalization epsilon must be 1e-5")
        if self.checkpoint_quantization != MXFP4_GROUP32:
            raise ValueError("K3 checkpoint must identify packed values and group-32 scales")

    @property
    def num_layers(self) -> int:
        return len(self.layer_types)

    @property
    def layers(self) -> tuple[KimiK3LayerSpec, ...]:
        width = self.residual_block_size
        return tuple(KimiK3LayerSpec(i, attention,
                                    "dense" if i < self.dense_prefix_layers else "latent-moe",
                                    (i + width - 1) // width, i % width == 0)
                     for i, attention in enumerate(self.layer_types))

    def to_obj(self) -> dict[str, Any]:
        value = _scalars(self)
        value.update(schema=KIMI_K3_GEOMETRY_SCHEMA, layer_types=list(self.layer_types),
                     kda=self.kda.to_obj(), mla=self.mla.to_obj(), moe=self.moe.to_obj())
        return value

    @classmethod
    def from_obj(cls, value: object) -> KimiK3Spec:
        obj = _fields(value, {f.name for f in fields(cls)} | {"schema"}, "K3 geometry").copy()
        if obj.pop("schema") != KIMI_K3_GEOMETRY_SCHEMA:
            raise ValueError("unknown K3 geometry schema")
        if not isinstance(obj["layer_types"], list):
            raise TypeError("serialized layer types must be an array")
        obj.update(layer_types=tuple(obj["layer_types"]), kda=KdaSpec.from_obj(obj["kda"]),
                   mla=MlaSpec.from_obj(obj["mla"]), moe=LatentMoeSpec.from_obj(obj["moe"]))
        return cls(**obj)

    def retained_state(self, *, sequences: int, cached_tokens: int, current_tokens: int) -> dict[str, int]:
        """Return declared capacity; token counts are totals across all sequences.

        This does not price cache accesses or describe physical allocation.
        """
        for name, value in (("sequences", sequences), ("cached_tokens", cached_tokens), ("current_tokens", current_tokens)):
            _nonnegative(name, value)
        linear = self.layer_types.count("kda")
        full = self.layer_types.count("mla")
        kda = self.kda
        bank = (self.num_layers + self.residual_block_size - 1) // self.residual_block_size
        return {
            "kda_recurrent_bytes": sequences * linear * kda.heads * kda.head_dim ** 2 * 4,
            "kda_convolution_history_bytes": sequences * linear * 3 * kda.projection_width * (kda.convolution_width - 1) * 2,
            "mla_cached_history_bytes": cached_tokens * full * self.mla.compressed_width * 2,
            "mla_new_token_state_bytes": current_tokens * full * self.mla.compressed_width * 2,
            "residual_snapshot_capacity_bytes": current_tokens * bank * self.hidden_size * 2,
            "residual_current_stream_bytes": current_tokens * self.hidden_size * 2,
        }

    def weight_shapes(self) -> tuple[LogicalWeightShape, ...]:
        """Enumerate logical text tensors before packing, padding or absorption."""
        result: list[LogicalWeightShape] = []

        def add(name: str, shape: tuple[int, ...], layer: int | None = None) -> None:
            result.append(LogicalWeightShape(name, shape, layer))

        h, a, m, moe = self.hidden_size, self.kda, self.mla, self.moe
        add("embedding", (self.vocab_size, h))
        for layer in self.layers:
            i = layer.index
            for name in ("input_norm", "post_attention_norm", "attention_residual_norm", "mlp_residual_norm", "attention_residual_query", "mlp_residual_query"):
                add(name, (h,), i)
            if layer.attention == "kda":
                for name in ("q", "k", "v", "output_gate"):
                    add(f"kda.{name}", (h, a.projection_width), i)
                add("kda.output", (a.projection_width, h), i)
                add("kda.forget_a", (h, a.head_dim), i)
                add("kda.forget_b", (a.head_dim, a.projection_width), i)
                add("kda.beta", (h, a.heads), i)
                for name in ("q", "k", "v"):
                    add(f"kda.{name}_convolution", (a.projection_width, a.convolution_width), i)
                add("kda.A_log", (a.heads,), i)
                add("kda.dt_bias", (a.heads, a.head_dim), i)
                add("kda.output_norm", (a.head_dim,), i)
            else:
                for name, shape in (
                    ("q_a", (h, m.query_rank)), ("q_b", (m.query_rank, m.heads * m.expanded_score_dim)),
                    ("kv_a", (h, m.compressed_width)), ("kv_b", (m.kv_rank, m.heads * (m.score_nope_dim + m.value_dim))),
                    ("output_gate", (h, m.heads * m.value_dim)), ("output", (m.heads * m.value_dim, h)),
                    ("q_norm", (m.query_rank,)), ("kv_norm", (m.kv_rank,)),
                ):
                    add(f"mla.{name}", shape, i)
            if layer.feed_forward == "dense":
                width = self.dense_intermediate_size
                for name, shape in (("gate", (h, width)), ("up", (h, width)), ("down", (width, h))):
                    add(f"dense.{name}", shape, i)
            else:
                e, width = moe.latent_width, moe.intermediate_size
                for name, shape in (
                    ("router", (h, moe.experts)), ("router_correction", (moe.experts,)),
                    ("latent_down", (h, e)), ("latent_up", (e, h)), ("latent_norm", (e,)),
                    ("expert_gate", (moe.experts, e, width)), ("expert_up", (moe.experts, e, width)),
                    ("expert_down", (moe.experts, width, e)),
                    ("shared_gate", (h, width * moe.shared_experts)),
                    ("shared_up", (h, width * moe.shared_experts)),
                    ("shared_down", (width * moe.shared_experts, h)),
                ):
                    add(f"moe.{name}", shape, i)
        for name in ("final_norm", "output_residual_norm", "output_residual_query"):
            add(name, (h,))
        add("lm_head", (h, self.vocab_size))
        return tuple(result)
