# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Yifeng Wang
# SPDX-License-Identifier: Apache-2.0
"""Model-configured pass composition over imported external measurements."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from simllm.calibration.external_db import (
    ExternalDatabaseIdentityError,
    ExternalLatency,
    ExternalOperationDatabase,
    ExternalPassResult,
)
from simllm.calibration.external_nccl import ExternalNcclDatabase


@dataclass(frozen=True)
class ExternalModelConfig:
    """Narrow model and parallelism contract used by the pass composer."""

    model_id: str
    architecture: str
    num_hidden_layers: int
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int
    tensor_parallel: int
    pipeline_parallel: int
    expert_parallel: int
    nextn: int
    num_local_experts: int = 0
    num_experts_per_tok: int = 0
    has_shared_expert: bool = False
    workload_distribution: str = "power_law"
    gemm_quant_mode: str = "fp8_block"
    attention_quant_mode: str = "bfloat16"

    def __post_init__(self) -> None:
        positive = {
            "num_hidden_layers": self.num_hidden_layers,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "head_dim": self.head_dim,
            "vocab_size": self.vocab_size,
            "tensor_parallel": self.tensor_parallel,
            "pipeline_parallel": self.pipeline_parallel,
            "expert_parallel": self.expert_parallel,
        }
        invalid = {name: value for name, value in positive.items() if value <= 0}
        if invalid:
            raise ValueError(f"external model dimensions must be positive: {invalid!r}")
        if self.nextn < 0:
            raise ValueError("nextn must be non-negative")
        if self.pipeline_parallel != 1:
            raise ValueError("the external pass composer currently requires pipeline_parallel=1")
        if self.num_attention_heads % self.tensor_parallel != 0:
            raise ValueError("attention heads must divide tensor parallelism")
        if self.num_key_value_heads % self.tensor_parallel != 0:
            raise ValueError("key/value heads must divide tensor parallelism")
        if self.architecture not in {"dense", "moe"}:
            raise ValueError("architecture must be 'dense' or 'moe'")
        if self.architecture == "dense":
            if self.expert_parallel != 1:
                raise ValueError("dense composition requires expert_parallel=1")
        elif (
            self.num_local_experts <= 0
            or self.num_experts_per_tok <= 0
            or self.num_local_experts % self.expert_parallel != 0
        ):
            raise ValueError("MoE experts and top-k must be positive and divide expert parallelism")

    @classmethod
    def from_mapping(
        cls,
        model: Mapping[str, Any],
        *,
        architecture: str,
        tensor_parallel: int,
        pipeline_parallel: int,
        expert_parallel: int,
        workload_distribution: str = "power_law",
        gemm_quant_mode: str = "fp8_block",
        attention_quant_mode: str = "bfloat16",
    ) -> ExternalModelConfig:
        """Build a config while requiring an explicit MTP ``nextn`` value."""

        if "nextn" not in model:
            raise ExternalDatabaseIdentityError(
                "external model composition requires explicit nextn; use_mtp is not a substitute"
            )
        required = (
            "model_id",
            "num_hidden_layers",
            "hidden_size",
            "intermediate_size",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "vocab_size",
        )
        missing = [name for name in required if name not in model]
        if missing:
            raise ExternalDatabaseIdentityError(
                f"external model config is missing required fields {missing!r}"
            )
        return cls(
            model_id=str(model["model_id"]),
            architecture=architecture,
            num_hidden_layers=int(model["num_hidden_layers"]),
            hidden_size=int(model["hidden_size"]),
            intermediate_size=int(model["intermediate_size"]),
            num_attention_heads=int(model["num_attention_heads"]),
            num_key_value_heads=int(model["num_key_value_heads"]),
            head_dim=int(model["head_dim"]),
            vocab_size=int(model["vocab_size"]),
            tensor_parallel=tensor_parallel,
            pipeline_parallel=pipeline_parallel,
            expert_parallel=expert_parallel,
            nextn=int(model["nextn"]),
            num_local_experts=int(model.get("num_local_experts", 0)),
            num_experts_per_tok=int(model.get("num_experts_per_tok", 0)),
            has_shared_expert=bool(model.get("has_shared_expert", False)),
            workload_distribution=workload_distribution,
            gemm_quant_mode=gemm_quant_mode,
            attention_quant_mode=attention_quant_mode,
        )


def qwen3_32b_fp8_config() -> ExternalModelConfig:
    """Return the frozen Qwen3-32B configuration used by the parity study."""

    return ExternalModelConfig(
        model_id="qwen3_32b_fp8",
        architecture="dense",
        num_hidden_layers=64,
        hidden_size=5120,
        intermediate_size=25600,
        num_attention_heads=64,
        num_key_value_heads=8,
        head_dim=128,
        vocab_size=151936,
        tensor_parallel=4,
        pipeline_parallel=1,
        expert_parallel=1,
        nextn=0,
    )


class ExternalPassModel:
    """Compose dense or MoE passes from one explicit model configuration."""

    def __init__(
        self,
        database: ExternalOperationDatabase,
        config: ExternalModelConfig,
        *,
        nccl_database: ExternalNcclDatabase | None = None,
    ) -> None:
        self.database = database
        self.config = config
        self.nccl_database = nccl_database

    @property
    def _layer_repeats(self) -> int:
        return self.config.num_hidden_layers + self.config.nextn

    @property
    def _singleton_scale(self) -> float:
        return self._layer_repeats / self.config.num_hidden_layers

    def _result(self, latency: float, operation: str, rule: str) -> ExternalLatency:
        return self.database._result(latency, operation, rule)

    def _memory_operation(
        self,
        *,
        operation: str,
        mem_bytes: int,
        scale_factor: float,
    ) -> ExternalLatency:
        latency = self.database.query_memory_operation(mem_bytes) * scale_factor
        return self._result(latency, operation, "analytical-h200-empirical-memory")

    def _gemm(
        self,
        *,
        operation: str,
        m: int,
        n: int,
        k: int,
        quant_mode: str,
        scale_factor: float,
    ) -> ExternalLatency:
        base = self.database.query_gemm(m=m, n=n, k=k, quant_mode=quant_mode)
        return self._result(
            base.latency_ms * scale_factor,
            operation,
            "external-gemm-times-repeat-count",
        )

    def _allreduce(
        self,
        *,
        operation: str,
        tokens: int,
        scale_factor: float,
    ) -> ExternalLatency:
        if self.config.tensor_parallel == 1:
            return self._result(0.0, operation, "tensor-parallel-width-one-no-op")
        base = self.database.query_custom_allreduce(
            quant_mode="half",
            tp_size=self.config.tensor_parallel,
            size=tokens * self.config.hidden_size,
        )
        return self._result(
            base.latency_ms * scale_factor,
            operation,
            "external-half-allreduce-times-repeat-count",
        )

    def _context_attention(
        self,
        *,
        batch_size: int,
        sequence: int,
        prefix: int,
    ) -> ExternalLatency:
        config = self.config
        num_heads = config.num_attention_heads // config.tensor_parallel
        num_kv_heads = config.num_key_value_heads // config.tensor_parallel
        result = self.database.query_context_attention(
            b=batch_size,
            s=sequence,
            prefix=prefix,
            n=num_heads,
            n_kv=num_kv_heads,
            kv_quant_mode=config.attention_quant_mode,
            fmha_quant_mode=config.attention_quant_mode,
            window_size=0,
            head_size=config.head_dim,
        ).latency_ms

        query_elements = num_heads * config.head_dim
        key_elements = num_kv_heads * config.head_dim
        value_elements = num_kv_heads * config.head_dim
        qk_norm_latency = (
            2 * self.database.query_memory_operation(query_elements * 2)
            + 2 * self.database.query_memory_operation(key_elements * 2)
        )
        extra_latency = qk_norm_latency * 2
        apply_rope_latency = 2 * self.database.query_memory_operation(
            query_elements * 2 + key_elements * 2
        )
        kv_write_latency = self.database.query_memory_operation(
            key_elements * 2
        ) + self.database.query_memory_operation(value_elements * 2)
        extra_latency += apply_rope_latency + kv_write_latency
        result += extra_latency * 1.1
        return self._result(
            result * config.num_hidden_layers,
            "context_attention",
            "external-context-attention-plus-qk-norm-rope-kv-write",
        )

    def _generation_attention(self, *, batch_size: int, sequence: int) -> ExternalLatency:
        config = self.config
        base = self.database.query_generation_attention(
            b=batch_size,
            s=sequence,
            n=config.num_attention_heads // config.tensor_parallel,
            n_kv=config.num_key_value_heads // config.tensor_parallel,
            kv_quant_mode=config.attention_quant_mode,
            window_size=0,
            head_size=config.head_dim,
        )
        return self._result(
            base.latency_ms * self._layer_repeats,
            "generation_attention",
            "external-generation-attention-times-repeat-count",
        )

    def _dense_context_operations(
        self,
        *,
        batch_size: int,
        effective_isl: int,
        prefix: int,
    ) -> tuple[ExternalLatency, ...]:
        config = self.config
        tokens = batch_size * effective_isl
        vocab_per_rank = config.vocab_size // config.tensor_parallel
        intermediate_per_rank = config.intermediate_size // config.tensor_parallel
        qkv_width = (
            config.num_attention_heads * config.head_dim // config.tensor_parallel
            + config.head_dim
            * (config.num_key_value_heads // config.tensor_parallel)
            * 2
        )
        return (
            self._memory_operation(
                operation="context_embedding",
                mem_bytes=tokens * config.hidden_size * 2,
                scale_factor=1,
            ),
            self._memory_operation(
                operation="context_add_norm_1",
                mem_bytes=tokens * (2 * config.hidden_size + 2 * config.hidden_size) * 2,
                scale_factor=config.num_hidden_layers,
            ),
            self._gemm(
                operation="context_qkv_gemm",
                m=tokens,
                n=qkv_width,
                k=config.hidden_size,
                quant_mode=config.gemm_quant_mode,
                scale_factor=config.num_hidden_layers,
            ),
            self._context_attention(
                batch_size=batch_size,
                sequence=effective_isl,
                prefix=prefix,
            ),
            self._gemm(
                operation="context_proj_gemm",
                m=tokens,
                n=config.hidden_size,
                k=config.num_attention_heads * config.head_dim // config.tensor_parallel,
                quant_mode=config.gemm_quant_mode,
                scale_factor=config.num_hidden_layers,
            ),
            self._memory_operation(
                operation="context_add_norm_2",
                mem_bytes=tokens * (2 * config.hidden_size + 2 * config.hidden_size) * 2,
                scale_factor=config.num_hidden_layers,
            ),
            self._gemm(
                operation="context_gate_ffn1_gemm",
                m=tokens,
                n=2 * intermediate_per_rank,
                k=config.hidden_size,
                quant_mode=config.gemm_quant_mode,
                scale_factor=config.num_hidden_layers,
            ),
            self._memory_operation(
                operation="context_act_gate",
                mem_bytes=tokens * (2 * intermediate_per_rank + intermediate_per_rank) * 2,
                scale_factor=config.num_hidden_layers,
            ),
            self._gemm(
                operation="context_ffn2_gemm",
                m=tokens,
                n=config.hidden_size,
                k=intermediate_per_rank,
                quant_mode=config.gemm_quant_mode,
                scale_factor=config.num_hidden_layers,
            ),
            self._gemm(
                operation="context_logits_gemm",
                m=batch_size,
                n=vocab_per_rank,
                k=config.hidden_size,
                quant_mode="bfloat16",
                scale_factor=1,
            ),
            self._allreduce(operation="context_embedding_ar", tokens=tokens, scale_factor=1),
            self._allreduce(
                operation="context_ar_1",
                tokens=tokens,
                scale_factor=config.num_hidden_layers,
            ),
            self._allreduce(
                operation="context_ar_2",
                tokens=tokens,
                scale_factor=config.num_hidden_layers,
            ),
            self._result(0.0, "context_p2p", "pipeline-width-one-no-op"),
        )

    def _dense_generation_operations(
        self,
        *,
        batch_size: int,
        sequence: int,
    ) -> tuple[ExternalLatency, ...]:
        config = self.config
        tokens = batch_size
        vocab_per_rank = config.vocab_size // config.tensor_parallel
        intermediate_per_rank = config.intermediate_size // config.tensor_parallel
        qkv_width = (
            config.num_attention_heads * config.head_dim // config.tensor_parallel
            + config.head_dim
            * (config.num_key_value_heads // config.tensor_parallel)
            * 2
        )
        repeats = float(self._layer_repeats)
        return (
            self._memory_operation(
                operation="generation_embedding",
                mem_bytes=tokens * config.hidden_size * 2,
                scale_factor=self._singleton_scale,
            ),
            self._memory_operation(
                operation="generation_add_norm_1",
                mem_bytes=tokens * (2 * config.hidden_size + 2 * config.hidden_size) * 2,
                scale_factor=repeats,
            ),
            self._gemm(
                operation="generation_qkv_gemm",
                m=tokens,
                n=qkv_width,
                k=config.hidden_size,
                quant_mode=config.gemm_quant_mode,
                scale_factor=repeats,
            ),
            self._generation_attention(batch_size=batch_size, sequence=sequence),
            self._gemm(
                operation="generation_proj_gemm",
                m=tokens,
                n=config.hidden_size,
                k=config.num_attention_heads * config.head_dim // config.tensor_parallel,
                quant_mode=config.gemm_quant_mode,
                scale_factor=repeats,
            ),
            self._memory_operation(
                operation="generation_add_norm_2",
                mem_bytes=tokens * (2 * config.hidden_size + 2 * config.hidden_size) * 2,
                scale_factor=repeats,
            ),
            self._gemm(
                operation="generation_gate_ffn1_gemm",
                m=tokens,
                n=2 * intermediate_per_rank,
                k=config.hidden_size,
                quant_mode=config.gemm_quant_mode,
                scale_factor=repeats,
            ),
            self._memory_operation(
                operation="generation_act_gate",
                mem_bytes=tokens * (2 * intermediate_per_rank + intermediate_per_rank) * 2,
                scale_factor=repeats,
            ),
            self._gemm(
                operation="generation_ffn2_gemm",
                m=tokens,
                n=config.hidden_size,
                k=intermediate_per_rank,
                quant_mode=config.gemm_quant_mode,
                scale_factor=repeats,
            ),
            self._gemm(
                operation="generation_logits_gemm",
                m=tokens,
                n=vocab_per_rank,
                k=config.hidden_size,
                quant_mode="bfloat16",
                scale_factor=self._singleton_scale,
            ),
            self._allreduce(
                operation="generation_embedding_ar",
                tokens=tokens,
                scale_factor=self._singleton_scale,
            ),
            self._allreduce(
                operation="generation_ar_1",
                tokens=tokens,
                scale_factor=repeats,
            ),
            self._allreduce(
                operation="generation_ar_2",
                tokens=tokens,
                scale_factor=repeats,
            ),
            self._result(0.0, "generation_p2p", "pipeline-width-one-no-op"),
        )

    def _moe_dispatch(
        self,
        *,
        operation: str,
        collective: str,
        tokens: int,
    ) -> ExternalLatency:
        config = self.config
        if self.nccl_database is None:
            raise ExternalDatabaseIdentityError(
                "MoE pass composition requires the imported NCCL database"
            )
        if config.expert_parallel == 1:
            return self._result(0.0, operation, "expert-parallel-width-one-no-op")
        base = self.nccl_database.query(
            dtype="half",
            operation=collective,
            ranks=config.expert_parallel,
            message_size=tokens * config.hidden_size * config.expert_parallel,
        )
        return ExternalLatency(
            latency_ms=base.latency_ms * self._layer_repeats,
            source=base.source,
            operation=operation,
            rule=(
                f"{base.rule};half-message="
                "tokens-times-hidden-times-expert-parallel"
            ),
            evidence_class=base.evidence_class,
        )

    def _moe_generation_operations(
        self,
        *,
        batch_size: int,
        sequence: int,
    ) -> tuple[ExternalLatency, ...]:
        config = self.config
        tokens = batch_size
        repeats = float(self._layer_repeats)
        qkv_width = (
            config.num_attention_heads * config.head_dim // config.tensor_parallel
            + config.head_dim
            * (config.num_key_value_heads // config.tensor_parallel)
            * 2
        )
        workload_distribution = (
            "power_law_1.2"
            if config.workload_distribution == "power_law"
            else config.workload_distribution
        )
        moe = self.database.query_moe(
            num_tokens=tokens * config.expert_parallel,
            hidden_size=config.hidden_size,
            inter_size=config.intermediate_size,
            topk=config.num_experts_per_tok,
            num_experts=config.num_local_experts,
            moe_tp_size=config.tensor_parallel,
            moe_ep_size=config.expert_parallel,
            quant_mode=config.gemm_quant_mode,
            workload_distribution=workload_distribution,
        )
        return (
            self._memory_operation(
                operation="generation_embedding",
                mem_bytes=tokens * config.hidden_size * 2,
                scale_factor=self._singleton_scale,
            ),
            self._memory_operation(
                operation="generation_add_norm_1",
                mem_bytes=tokens * (2 * config.hidden_size + 2 * config.hidden_size) * 2,
                scale_factor=repeats,
            ),
            self._gemm(
                operation="generation_qkv_gemm",
                m=tokens,
                n=qkv_width,
                k=config.hidden_size,
                quant_mode=config.gemm_quant_mode,
                scale_factor=repeats,
            ),
            self._generation_attention(batch_size=batch_size, sequence=sequence),
            self._gemm(
                operation="generation_proj_gemm",
                m=tokens,
                n=config.hidden_size,
                k=config.num_attention_heads * config.head_dim // config.tensor_parallel,
                quant_mode=config.gemm_quant_mode,
                scale_factor=repeats,
            ),
            self._memory_operation(
                operation="generation_add_norm_2",
                mem_bytes=tokens * (2 * config.hidden_size + 2 * config.hidden_size) * 2,
                scale_factor=repeats,
            ),
            self._gemm(
                operation="generation_router_gemm",
                m=tokens,
                n=config.num_local_experts,
                k=config.hidden_size,
                quant_mode="bfloat16",
                scale_factor=repeats,
            ),
            self._moe_dispatch(
                operation="generation_moe_pre_dispatch",
                collective="all_gather",
                tokens=tokens,
            ),
            self._result(
                moe.latency_ms * repeats,
                "generation_moe",
                "external-moe-times-repeat-count",
            ),
            self._moe_dispatch(
                operation="generation_moe_post_dispatch",
                collective="reduce_scatter",
                tokens=tokens,
            ),
            self._gemm(
                operation="generation_logits_gemm",
                m=tokens,
                n=config.vocab_size // config.tensor_parallel,
                k=config.hidden_size,
                quant_mode="bfloat16",
                scale_factor=self._singleton_scale,
            ),
            self._allreduce(
                operation="generation_embedding_ar",
                tokens=tokens,
                scale_factor=self._singleton_scale,
            ),
            self._result(0.0, "generation_p2p", "pipeline-width-one-no-op"),
        )

    def run_context(
        self,
        *,
        batch_size: int,
        isl: int,
        prefix: int = 0,
    ) -> ExternalPassResult:
        """Evaluate one static context pass for a dense model."""

        if self.config.architecture != "dense" or self.config.nextn != 0:
            raise ValueError("context composition is available only for dense nextn=0 configs")
        effective_isl = isl - prefix
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if effective_isl <= 0:
            raise ValueError("isl must remain positive after removing prefix")
        operations = self._dense_context_operations(
            batch_size=batch_size,
            effective_isl=effective_isl,
            prefix=prefix,
        )
        total = math.fsum(entry.latency_ms for entry in operations)
        return ExternalPassResult(
            mode="static_ctx",
            total=self._result(
                total,
                f"{self.config.model_id}_context_pass",
                "ordered-python-phase-sum",
            ),
            operations=operations,
        )

    def run_generation(
        self,
        *,
        batch_size: int,
        isl: int,
        osl: int,
        stride: int = 32,
        beam_width: int = 1,
    ) -> ExternalPassResult:
        """Evaluate a sampled static generation pass."""

        if batch_size <= 0 or isl <= 0:
            raise ValueError("batch_size and isl must be positive")
        if osl <= 1:
            raise ValueError("generation requires osl greater than one")
        if stride <= 0:
            raise ValueError("stride must be positive")
        if beam_width != 1:
            raise ValueError("the external generation composition supports beam_width=1 only")

        effective_batch = batch_size * (self.config.nextn + 1)
        names: list[str] = []
        totals: dict[str, float] = {}
        rules: dict[str, str] = {}
        sources: dict[str, Any] = {}
        evidence_classes: dict[str, str] = {}
        for index in range(0, osl - 1, stride):
            if self.config.architecture == "dense":
                sampled = self._dense_generation_operations(
                    batch_size=effective_batch,
                    sequence=isl + index + 1,
                )
            else:
                sampled = self._moe_generation_operations(
                    batch_size=effective_batch,
                    sequence=isl + index + 1,
                )
            repeat_count = min(stride, osl - 1 - index)
            for entry in sampled:
                if entry.operation not in totals:
                    names.append(entry.operation)
                    totals[entry.operation] = 0.0
                    rules[entry.operation] = entry.rule
                    sources[entry.operation] = entry.source
                    evidence_classes[entry.operation] = entry.evidence_class
                elif (
                    entry.source != sources[entry.operation]
                    or entry.evidence_class != evidence_classes[entry.operation]
                ):
                    raise ExternalDatabaseIdentityError(
                        f"operation {entry.operation!r} changed provenance across strides"
                    )
                totals[entry.operation] += entry.latency_ms * repeat_count
        operations = tuple(
            ExternalLatency(
                latency_ms=totals[name],
                source=sources[name],
                operation=name,
                rule=f"{rules[name]};stride-repeat",
                evidence_class=evidence_classes[name],
            )
            for name in names
        )
        total = math.fsum(entry.latency_ms for entry in operations)
        return ExternalPassResult(
            mode="static_gen",
            total=self._result(
                total,
                f"{self.config.model_id}_generation_pass",
                "ordered-python-phase-sum",
            ),
            operations=operations,
        )


__all__ = [
    "ExternalModelConfig",
    "ExternalPassModel",
    "qwen3_32b_fp8_config",
]
