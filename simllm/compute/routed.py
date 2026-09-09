"""Declared grouped expert projections and their device-service minimum."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import TYPE_CHECKING

from simllm.compute.provider import ComputeProvider, DurationEstimate, GpuSpec, KernelSpec

if TYPE_CHECKING:
    from simllm.compute.transformer import ModelDims


@dataclass(frozen=True)
class RoutedComputeConfig:
    """Select routed work with an optional explicitly declared GEMM minimum.

    The minimum applies to each nonempty grouped projection, independently
    of expert count and host launch mode. It is not a calibrated GPU value.
    """

    gemm_minimum_ps: int = 0
    minimum_gpu: GpuSpec | None = None

    def __post_init__(self) -> None:
        if type(self.gemm_minimum_ps) is not int or self.gemm_minimum_ps < 0:
            raise ValueError("gemm_minimum_ps must be a nonnegative integer")
        if self.minimum_gpu is not None and not isinstance(self.minimum_gpu, GpuSpec):
            raise TypeError("minimum_gpu must be a GpuSpec or None")
        if self.gemm_minimum_ps and self.minimum_gpu is None:
            raise ValueError("a positive declared minimum requires its complete GPU envelope")

    def validate_gpu(self, gpu: GpuSpec) -> None:
        if self.minimum_gpu is not None and self.minimum_gpu != gpu:
            raise ValueError("declared GEMM minimum belongs to a different GPU envelope")

    def estimate(
        self, provider: ComputeProvider, kernel: KernelSpec, gpu: GpuSpec
    ) -> DurationEstimate:
        self.validate_gpu(gpu)
        if (
            kernel.name not in ("moe_gate_up", "moe_down")
            or kernel.family_kernels or kernel.flops <= 0 or kernel.bytes_moved <= 0
        ):
            raise ValueError("routed minimum requires one nonempty grouped expert projection")
        estimate = provider.estimate(kernel, gpu)
        if type(estimate.duration_ps) is not int or estimate.duration_ps < 0:
            raise ValueError("provider must return nonnegative integer service")
        if estimate.duration_ps >= self.gemm_minimum_ps:
            return estimate
        return DurationEstimate(
            self.gemm_minimum_ps,
            "declared-gemm-minimum",
            max(estimate.uncertainty, 0.5),
        )


def grouped_expert_kernels(
    dims: ModelDims,
    *,
    layer: int,
    rank: int,
    expert_rows: tuple[tuple[int, int], ...],
) -> tuple[KernelSpec, KernelSpec]:
    """Price every routed row and one weight stream per active expert."""

    if dims.moe_intermediate_size is None or dims.num_experts <= 0:
        raise ValueError("grouped expert work requires complete MoE geometry")
    if not expert_rows:
        raise ValueError("an idle rank has no grouped expert invocation")
    if type(layer) is not int or not 0 <= layer < dims.num_layers:
        raise ValueError("layer is outside the model")
    if type(rank) is not int or rank < 0:
        raise ValueError("rank must be a nonnegative integer")
    if tuple(sorted(expert_rows)) != expert_rows:
        raise ValueError("expert rows must be sorted by expert identity")
    seen = set()
    for expert, count in expert_rows:
        if type(expert) is not int or not 0 <= expert < dims.num_experts:
            raise ValueError("expert identity is outside the model")
        if expert in seen:
            raise ValueError("expert rows contain a duplicate expert")
        if type(count) is not int or count <= 0:
            raise ValueError("an active expert must have a positive integer row count")
        seen.add(expert)
    rows = sum(count for _, count in expert_rows)
    matrix_elements = dims.hidden_size * dims.moe_intermediate_size
    weight_bytes = (
        matrix_elements * len(expert_rows) * Fraction(dims.weight_element_bytes)
    )
    if weight_bytes.denominator != 1:
        raise ValueError("expert weight extents require whole byte representation")
    config = (
        ("layer", layer),
        ("owner_rank", rank),
        ("routed_rows", rows),
        ("active_experts", len(expert_rows)),
        *((f"expert_{expert}_rows", count) for expert, count in expert_rows),
    )
    return (
        KernelSpec(
            "moe_gate_up", 4 * matrix_elements * rows,
            2 * int(weight_bytes), config,
        ),
        KernelSpec(
            "moe_down", 2 * matrix_elements * rows,
            int(weight_bytes), config,
        ),
    )
