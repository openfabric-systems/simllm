"""SGLang parallel arrangements over disaggregated serving manifests.

The stock CPU scheduler stays width one in the simulated session. This module
projects the deployment's simulated GPUs into the role-local attention data
parallel, dense data parallel and expert parallel groups that SGLang's public
DeepSeek configurations disclose. The groups are structural inputs to
placement and traffic rendering. They do not request distributed tensor work.
"""

from __future__ import annotations

from dataclasses import dataclass

from simllm.core import ServingPoolRole
from simllm.placement.disaggregated import (
    DisaggregatedDeploymentManifests,
    disaggregated_manifests,
)
from simllm.placement.manifest import GroupMembership


def _positive_int(name: str, value: object) -> int:
    if isinstance(value, bool) or type(value) is not int:
        raise TypeError(f"{name} must be an integer")
    if value < 1:
        raise ValueError(f"{name} must be positive")
    return value


@dataclass(frozen=True)
class SglangPoolArrangement:
    """Role-local structural parallel sizes from one SGLang deployment."""

    enable_data_parallel_attention: bool
    attention_data_parallel_size: int
    dense_data_parallel_size: int
    expert_parallel_size: int

    def __post_init__(self) -> None:
        if type(self.enable_data_parallel_attention) is not bool:
            raise TypeError("enable_data_parallel_attention must be a boolean")
        for name in (
            "attention_data_parallel_size",
            "dense_data_parallel_size",
            "expert_parallel_size",
        ):
            _positive_int(name, getattr(self, name))
        if (
            not self.enable_data_parallel_attention
            and self.attention_data_parallel_size != 1
        ):
            raise ValueError(
                "disabled data-parallel attention requires size one"
            )

    @classmethod
    def identity(cls) -> SglangPoolArrangement:
        """The structural single-rank arrangement."""

        return cls(
            enable_data_parallel_attention=False,
            attention_data_parallel_size=1,
            dense_data_parallel_size=1,
            expert_parallel_size=1,
        )

    def to_json(self) -> dict[str, object]:
        return {
            "enable_data_parallel_attention": self.enable_data_parallel_attention,
            "attention_data_parallel_size": self.attention_data_parallel_size,
            "dense_data_parallel_size": self.dense_data_parallel_size,
            "expert_parallel_size": self.expert_parallel_size,
        }


def _partition_membership(
    role_ranks: tuple[int, ...],
    *,
    global_rank: int,
    size: int,
) -> GroupMembership:
    if len(role_ranks) % size:
        raise ValueError(
            f"parallel size {size} does not divide role width {len(role_ranks)}"
        )
    index = role_ranks.index(global_rank)
    start = index // size * size
    members = list(role_ranks[start : start + size])
    return GroupMembership(rank_in_group=index - start, global_ranks=members)


def _apply_arrangement(
    manifests: DisaggregatedDeploymentManifests,
    *,
    role: ServingPoolRole,
    arrangement: SglangPoolArrangement,
) -> None:
    role_ranks = tuple(
        rank.global_rank
        for rank in manifests.placement.ranks
        if rank.pool_role == role.value
    )
    sizes = {
        "attn_dp": arrangement.attention_data_parallel_size,
        "dense_dp": arrangement.dense_data_parallel_size,
        "ep": arrangement.expert_parallel_size,
    }
    for name, size in sizes.items():
        if len(role_ranks) % size:
            raise ValueError(
                f"{role.value} {name} size {size} does not divide "
                f"simulated role width {len(role_ranks)}"
            )
    for rank in manifests.placement.ranks:
        if rank.pool_role != role.value:
            continue
        for name, size in sizes.items():
            rank.groups[name] = _partition_membership(
                role_ranks,
                global_rank=rank.global_rank,
                size=size,
            )


def sglang_disaggregated_manifests(
    *,
    prefill_nodes: int,
    decode_nodes: int,
    gpus_per_node: int,
    prefill_arrangement: SglangPoolArrangement,
    decode_arrangement: SglangPoolArrangement,
    framework_version: str,
) -> DisaggregatedDeploymentManifests:
    """Build role-aware manifests with SGLang's structural parallel groups."""

    if not isinstance(prefill_arrangement, SglangPoolArrangement):
        raise TypeError("prefill_arrangement must be SglangPoolArrangement")
    if not isinstance(decode_arrangement, SglangPoolArrangement):
        raise TypeError("decode_arrangement must be SglangPoolArrangement")
    if not isinstance(framework_version, str) or not framework_version.strip():
        raise ValueError("framework_version must be a nonblank string")
    manifests = disaggregated_manifests(
        prefill_nodes=prefill_nodes,
        decode_nodes=decode_nodes,
        gpus_per_node=gpus_per_node,
        framework="sglang",
        framework_version=framework_version,
    )
    _apply_arrangement(
        manifests,
        role=ServingPoolRole.PREFILL,
        arrangement=prefill_arrangement,
    )
    _apply_arrangement(
        manifests,
        role=ServingPoolRole.DECODE,
        arrangement=decode_arrangement,
    )
    manifests.validate()
    return manifests


__all__ = ["SglangPoolArrangement", "sglang_disaggregated_manifests"]
