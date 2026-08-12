"""Captured MoE routing joined to versioned expert-placement snapshots."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from simllm.core.request_lifetime import RequestLifetimeRegistry
from simllm.placement.manifest import PLACEMENT_SCHEMA, PlacementManifest
from simllm.preplay.arena import RoutingArena
from simllm.preplay.routing import RoutedExperts, validate_routed_experts


def _integer(value: object, path: str, *, nonnegative: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{path}: expected an integer")
    if nonnegative and value < 0:
        raise ValueError(f"{path}: expected a nonnegative integer")
    return value


def _ranks(values: Sequence[int], path: str) -> tuple[int, ...]:
    ranks = tuple(values)
    if not ranks:
        raise ValueError(f"{path}: must not be empty")
    for index, rank in enumerate(ranks):
        _integer(rank, f"{path}[{index}]", nonnegative=True)
    if len(ranks) != len(set(ranks)):
        raise ValueError(f"{path}: contains duplicate ranks")
    return ranks


@dataclass(frozen=True, kw_only=True)
class ExpertPlacementSnapshot:
    """One complete `(layer, expert, owner rank)` table at an EPLB epoch."""

    placement_epoch: int
    expert_owners: tuple[tuple[int, int, int], ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "expert_owners", tuple(self.expert_owners))
        validate_expert_placement_snapshot(self)

    @classmethod
    def from_manifest(
        cls,
        manifest: PlacementManifest,
        ep_ranks: Sequence[int],
    ) -> ExpertPlacementSnapshot:
        """Project one manifest's selected EP group into an immutable snapshot."""

        if not isinstance(manifest, PlacementManifest):
            raise TypeError("manifest: expected PlacementManifest")
        if manifest.schema != PLACEMENT_SCHEMA:
            raise ValueError(f"manifest.schema: unsupported schema {manifest.schema!r}")
        ranks = _ranks(ep_ranks, "ep_ranks")
        placements = []
        for rank in ranks:
            try:
                placements.append(manifest.by_rank(rank))
            except KeyError as exc:
                raise ValueError(f"manifest: missing EP rank {rank}") from exc
        epochs = {placement.placement_epoch for placement in placements}
        if len(epochs) != 1:
            raise ValueError("manifest: EP ranks disagree on placement_epoch")
        epoch = next(iter(epochs))
        entries = []
        for placement in placements:
            for layer, expert_ids in placement.local_expert_ids.items():
                for expert_id in expert_ids:
                    entries.append((layer, expert_id, placement.global_rank))
        return cls(
            placement_epoch=epoch,
            expert_owners=tuple(sorted(entries)),
        )

    def owner_map(self) -> dict[tuple[int, int], int]:
        """Return the validated `(layer, expert)` to rank projection."""

        return {
            (layer, expert): rank
            for layer, expert, rank in self.expert_owners
        }


def validate_expert_placement_snapshot(value: ExpertPlacementSnapshot) -> None:
    """Validate one canonical immutable placement snapshot."""

    if not isinstance(value, ExpertPlacementSnapshot):
        raise TypeError("snapshot: expected ExpertPlacementSnapshot")
    _integer(value.placement_epoch, "snapshot.placement_epoch", nonnegative=True)
    if not isinstance(value.expert_owners, tuple):
        raise TypeError("snapshot.expert_owners: in-memory contract requires a tuple")
    if not value.expert_owners:
        raise ValueError("snapshot.expert_owners: must not be empty")
    keys = []
    for index, entry in enumerate(value.expert_owners):
        path = f"snapshot.expert_owners[{index}]"
        if not isinstance(entry, tuple) or len(entry) != 3:
            raise TypeError(f"{path}: expected a three-item tuple")
        layer, expert, rank = entry
        _integer(layer, f"{path}[0]", nonnegative=True)
        _integer(expert, f"{path}[1]", nonnegative=True)
        _integer(rank, f"{path}[2]", nonnegative=True)
        keys.append((layer, expert))
    if tuple(sorted(value.expert_owners)) != value.expert_owners:
        raise ValueError("snapshot.expert_owners: must be lexicographically sorted")
    if len(keys) != len(set(keys)):
        raise ValueError("snapshot.expert_owners: expert has multiple owners")


@dataclass(frozen=True, kw_only=True)
class RoutedMoeSupply:
    """Captured assignments, placement snapshots and explicit step epochs."""

    placements: tuple[ExpertPlacementSnapshot, ...]
    step_placement_epochs: tuple[tuple[int, int], ...]
    routed_experts: RoutedExperts | None = None
    routing_arena: RoutingArena | None = None
    lifetimes: RequestLifetimeRegistry | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "placements", tuple(self.placements))
        object.__setattr__(
            self,
            "step_placement_epochs",
            tuple(self.step_placement_epochs),
        )
        validate_routed_moe_supply(self)

    def placement_for_step(self, step_index: int) -> ExpertPlacementSnapshot:
        """Return the explicitly selected placement snapshot for one step."""

        _integer(step_index, "step_index", nonnegative=True)
        selected_epoch = None
        for candidate_step, epoch in self.step_placement_epochs:
            if candidate_step == step_index:
                selected_epoch = epoch
                break
        if selected_epoch is None:
            raise ValueError(
                f"step_placement_epochs: no placement epoch for step {step_index}"
            )
        for placement in self.placements:
            if placement.placement_epoch == selected_epoch:
                return placement
        raise AssertionError("validated step epoch has no placement snapshot")


def validate_routed_moe_supply(value: RoutedMoeSupply) -> None:
    """Validate one routed supply without consulting a particular step or model."""

    if not isinstance(value, RoutedMoeSupply):
        raise TypeError("supply: expected RoutedMoeSupply")
    if (value.routed_experts is None) == (value.routing_arena is None):
        raise ValueError(
            "supply: select exactly one routed_experts or routing_arena authority"
        )
    if value.routed_experts is not None:
        validate_routed_experts(value.routed_experts)
        if value.lifetimes is not None:
            raise ValueError(
                "supply.lifetimes: only an enabled routing arena owns live views"
            )
    else:
        arena = value.routing_arena
        assert arena is not None
        if not isinstance(arena, RoutingArena):
            raise TypeError("supply.routing_arena: expected RoutingArena")
        if arena.closed:
            raise ValueError("supply.routing_arena: arena is closed")
        if not isinstance(value.lifetimes, RequestLifetimeRegistry):
            raise TypeError(
                "supply.lifetimes: arena authority requires RequestLifetimeRegistry"
            )
        arena_ids = {request.request_id for request in arena.requests}
        lifetime_ids = {request.request_id for request in value.lifetimes.requests}
        if arena_ids != lifetime_ids:
            raise ValueError(
                "supply.lifetimes: request identities disagree with routing arena"
            )
        for request in arena.requests:
            lifetime = value.lifetimes.by_request_id(request.request_id)
            descriptor = lifetime.view
            if descriptor.arena_id != arena.arena_id:
                raise ValueError(
                    f"supply.lifetimes[{request.request_id!r}]: arena identity disagrees"
                )
            if (
                descriptor.token_offset != request.token_offset
                or descriptor.token_count != request.token_count
                or descriptor.prompt_token_count != request.prompt_token_count
            ):
                raise ValueError(
                    f"supply.lifetimes[{request.request_id!r}]: view extent disagrees"
                )
            if lifetime.moe_layer_indices != arena.moe_layer_indices:
                raise ValueError(
                    f"supply.lifetimes[{request.request_id!r}]: layers disagree"
                )
    if not isinstance(value.placements, tuple):
        raise TypeError("supply.placements: in-memory contract requires a tuple")
    if not value.placements:
        raise ValueError("supply.placements: must not be empty")
    epochs = []
    for index, placement in enumerate(value.placements):
        if not isinstance(placement, ExpertPlacementSnapshot):
            raise TypeError(
                f"supply.placements[{index}]: expected ExpertPlacementSnapshot"
            )
        validate_expert_placement_snapshot(placement)
        epochs.append(placement.placement_epoch)
    if tuple(sorted(epochs)) != tuple(epochs):
        raise ValueError("supply.placements: must be sorted by placement_epoch")
    if len(epochs) != len(set(epochs)):
        raise ValueError("supply.placements: duplicate placement_epoch")

    if not isinstance(value.step_placement_epochs, tuple):
        raise TypeError(
            "supply.step_placement_epochs: in-memory contract requires a tuple"
        )
    if not value.step_placement_epochs:
        raise ValueError("supply.step_placement_epochs: must not be empty")
    steps = []
    known_epochs = set(epochs)
    for index, entry in enumerate(value.step_placement_epochs):
        path = f"supply.step_placement_epochs[{index}]"
        if not isinstance(entry, tuple) or len(entry) != 2:
            raise TypeError(f"{path}: expected a two-item tuple")
        step_index, epoch = entry
        _integer(step_index, f"{path}[0]", nonnegative=True)
        _integer(epoch, f"{path}[1]", nonnegative=True)
        if epoch not in known_epochs:
            raise ValueError(f"{path}: placement epoch {epoch} has no snapshot")
        steps.append(step_index)
    if tuple(sorted(steps)) != tuple(steps):
        raise ValueError("supply.step_placement_epochs: must be sorted by step index")
    if len(steps) != len(set(steps)):
        raise ValueError("supply.step_placement_epochs: duplicate step index")
