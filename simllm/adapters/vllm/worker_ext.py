"""Worker-side placement-manifest exporter for vLLM (the extraction path).

vLLM injects an arbitrary class into the worker's bases, so one
``collective_rpc`` reaches every rank without a fork::

    vllm serve <model> -tp 8 \\
        --worker-extension-cls simllm.adapters.vllm.PlacementExporter

    # driver side (in-process engine)
    entries = llm.collective_rpc("simllm_placement_entry")
    manifest = manifest_from_worker_entries(entries)
    manifest.save("placement.json")

Each rank exports *its own* view: hostname, local rank, GPU identity, the
actual ``GroupCoordinator.ranks`` lists, the model's real pipeline layer
range, per-MoE-layer local expert ids and the EPLB placement epoch. Group
memberships are read rather than recomputed from a rank formula, because
external DP, elastic scaling and layout changes silently break derived
layouts (see :mod:`simllm.placement.manifest`).

Two constraints shape this module:

- vLLM asserts that no attribute name on the extension class collides with
  one on the worker class, so :class:`PlacementExporter` exposes exactly one
  non-dunder name (``simllm_placement_entry``) and all the work lives in
  module-level functions.
- Every field except ``global_rank``, ``hostname`` and ``local_rank`` is
  optional. Discovery is getattr-based and never raises: a renamed internal
  costs one field, it does not fail the capture run.

Nothing here imports vLLM or torch at module import time, so the assembly
side (:func:`manifest_from_worker_entries`) runs anywhere.
"""

from __future__ import annotations

import logging
import re
import socket
from collections.abc import Mapping, Sequence
from typing import Any

from simllm.placement import GroupMembership, PlacementManifest, RankPlacement

logger = logging.getLogger(__name__)

#: Process-group names exported per rank, mapped to the module-level
#: coordinator in ``vllm.distributed.parallel_state``. The public
#: ``get_*_group()`` accessors assert when a group is not initialized, so the
#: private globals are read directly and a missing group is simply skipped.
GROUP_COORDINATORS: dict[str, str] = {
    "tp": "_TP",
    "pp": "_PP",
    "dp": "_DP",
    "ep": "_EP",
    "pcp": "_PCP",
    # Decode context parallelism genuinely shards the KV cache across these
    # ranks (AttentionSpec divides max_num_blocks_per_req by the DCP size),
    # which is exactly the placement fact the manifest exists to record.
    "dcp": "_DCP",
    "eplb": "_EPLB",
}

_LAYER_INDEX = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")


class PlacementExporter:
    """Worker extension exporting this rank's placement-manifest entry.

    Selected with ``--worker-extension-cls
    simllm.adapters.vllm.PlacementExporter``; call it with
    ``collective_rpc("simllm_placement_entry")``.
    """

    def simllm_placement_entry(self) -> dict[str, Any]:
        """This worker's :class:`~simllm.placement.RankPlacement` as a dict."""
        return placement_entry(self)


def _attr(obj: Any, name: str, default: Any = None) -> Any:
    return default if obj is None else getattr(obj, name, default)


def _int_list(values: Any) -> list[int]:
    """Coerce a ranks list (list, tuple or tensor) to plain ints."""
    if values is None:
        return []
    if hasattr(values, "tolist"):
        values = values.tolist()
    try:
        return [int(value) for value in values]
    except (TypeError, ValueError):
        return []


def _worker_rank(worker: Any) -> int:
    rank = _attr(worker, "rank")
    if not isinstance(rank, int):
        rank = _attr(_attr(worker, "parallel_config"), "rank", 0)
    return int(rank) if isinstance(rank, int) else 0


def _worker_model(worker: Any) -> Any:
    """The nn.Module the runner loaded, via ``get_model()`` or ``.model``."""
    runner = _attr(worker, "model_runner")
    getter = _attr(runner, "get_model")
    if callable(getter):
        try:
            model = getter()
        except Exception as exc:  # noqa: BLE001 - vLLM internals differ across releases
            logger.debug("model_runner.get_model() failed: %s", exc)
        else:
            if model is not None:
                return model
    return _attr(runner, "model")


def discover_groups(global_rank: int) -> dict[str, GroupMembership]:
    """Actual process-group memberships of this rank, empty if not distributed."""
    try:
        from vllm.distributed import parallel_state
    except ImportError:
        return {}
    groups: dict[str, GroupMembership] = {}
    for name, attribute in GROUP_COORDINATORS.items():
        coordinator = getattr(parallel_state, attribute, None)
        ranks = _int_list(_attr(coordinator, "ranks"))
        if not ranks:
            continue
        rank_in_group = _attr(coordinator, "rank_in_group")
        if not isinstance(rank_in_group, int):
            rank_in_group = ranks.index(global_rank) if global_rank in ranks else 0
        groups[name] = GroupMembership(rank_in_group=int(rank_in_group), global_ranks=ranks)
    return groups


def discover_gpu_identity(device_index: int) -> tuple[str | None, str | None]:
    """GPU UUID and PCI bus id of ``device_index``, both optional."""
    try:
        import torch

        if not torch.cuda.is_available():
            return None, None
        props = torch.cuda.get_device_properties(device_index)
    except Exception as exc:  # noqa: BLE001 - no CUDA, or a torch build without properties
        logger.debug("GPU identity unavailable: %s", exc)
        return None, None
    uuid = getattr(props, "uuid", None)
    pci_bus_id = getattr(props, "pci_bus_id", None)
    return (str(uuid) if uuid else None, str(pci_bus_id) if pci_bus_id else None)


def discover_layer_range(model: Any) -> tuple[int, int] | None:
    """The model's own ``[start_layer, end_layer)``, not a computed split.

    vLLM decoder models carry the range on the inner module (``model.model``
    for most architectures, ``model.language_model`` for multimodal ones), so
    the direct chain is tried first and a bounded ``named_modules()`` scan is
    the fallback.
    """
    candidates = [
        model,
        _attr(model, "model"),
        _attr(model, "language_model"),
        _attr(_attr(model, "model"), "model"),
    ]
    for candidate in candidates:
        start = _attr(candidate, "start_layer")
        end = _attr(candidate, "end_layer")
        if isinstance(start, int) and isinstance(end, int):
            return (start, end)
    named_modules = _attr(model, "named_modules")
    if not callable(named_modules):
        return None
    try:
        for _name, module in named_modules():
            start = getattr(module, "start_layer", None)
            end = getattr(module, "end_layer", None)
            if isinstance(start, int) and isinstance(end, int):
                return (start, end)
    except Exception as exc:  # noqa: BLE001 - vLLM internals differ across releases
        logger.debug("layer-range scan failed: %s", exc)
    return None


def owned_expert_ids(
    expert_map: Any, local_num_experts: Any = None, global_num_experts: Any = None
) -> list[int]:
    """Global expert ids this rank owns, from a vLLM ``expert_map``.

    ``expert_map[global_id]`` is the local index of an owned expert and -1
    otherwise. It is ``None`` when expert parallelism is off, in which case
    the rank owns every expert.
    """
    if expert_map is not None:
        values = expert_map.tolist() if hasattr(expert_map, "tolist") else list(expert_map)
        if isinstance(global_num_experts, int) and global_num_experts > 0:
            # A mask-carrying map is padded with fused shared experts.
            values = values[:global_num_experts]
        return [index for index, value in enumerate(values) if int(value) >= 0]
    owns_all = local_num_experts in (None, global_num_experts)
    if isinstance(global_num_experts, int) and global_num_experts > 0 and owns_all:
        return list(range(global_num_experts))
    return []


def discover_local_experts(model: Any) -> dict[int, list[int]]:
    """MoE layer index to the global expert ids this rank owns.

    Reads ``expert_map`` off each fused-MoE module, or off its
    ``expert_map_manager`` (v0.26.0 moved the map behind that manager).
    Layer indices come from the module path, so a module that is not under
    ``layers.<n>`` is skipped rather than guessed at.
    """
    named_modules = _attr(model, "named_modules")
    if not callable(named_modules):
        return {}
    owned: dict[int, list[int]] = {}
    try:
        modules = list(named_modules())
    except Exception as exc:  # noqa: BLE001 - vLLM internals differ across releases
        logger.debug("named_modules() failed: %s", exc)
        return {}
    for name, module in modules:
        manager = getattr(module, "expert_map_manager", None)
        expert_map = getattr(module, "expert_map", None)
        if expert_map is None:
            expert_map = _attr(manager, "expert_map")
        local_num = getattr(module, "local_num_experts", None) or _attr(
            manager, "local_num_experts"
        )
        global_num = getattr(module, "global_num_experts", None) or _attr(
            manager, "global_num_experts"
        )
        if expert_map is None and local_num is None:
            continue
        match = _LAYER_INDEX.search(name)
        if match is None:
            continue
        ids = owned_expert_ids(expert_map, local_num, global_num)
        if ids:
            owned[int(match.group(1))] = ids
    return owned


def discover_placement_epoch(worker: Any) -> int:
    """EPLB rearrangement counter, i.e. which expert placement these ids are."""
    state = _attr(_attr(worker, "model_runner"), "eplb_state")
    epoch = _attr(state, "expert_rearrangement_step")
    return int(epoch) if isinstance(epoch, int) else 0


def placement_entry(worker: Any) -> dict[str, Any]:
    """Build one rank's placement dict from a live vLLM worker.

    The returned mapping has exactly the field names of
    :class:`simllm.placement.RankPlacement`, with ``groups`` as plain dicts,
    so it survives vLLM's msgpack RPC without any custom codec.
    """
    global_rank = _worker_rank(worker)
    local_rank = _attr(worker, "local_rank", 0)
    local_rank = int(local_rank) if isinstance(local_rank, int) else 0
    gpu_uuid, pci_bus_id = discover_gpu_identity(local_rank)
    model = _worker_model(worker)
    layer_range = discover_layer_range(model)
    return {
        "global_rank": global_rank,
        "hostname": socket.gethostname(),
        "local_rank": local_rank,
        "gpu_uuid": gpu_uuid,
        "pci_bus_id": pci_bus_id,
        "groups": {
            name: {
                "rank_in_group": membership.rank_in_group,
                "global_ranks": membership.global_ranks,
            }
            for name, membership in discover_groups(global_rank).items()
        },
        "pipeline_layer_range": list(layer_range) if layer_range else None,
        "local_expert_ids": discover_local_experts(model),
        "placement_epoch": discover_placement_epoch(worker),
    }


def vllm_version() -> str | None:
    """Installed vLLM version, or ``None`` when vLLM is not importable."""
    try:
        import vllm

        version = getattr(vllm, "__version__", None)
        if version:
            return str(version)
    except ImportError:
        pass
    try:
        from importlib.metadata import PackageNotFoundError
        from importlib.metadata import version as package_version

        return package_version("vllm")
    except (ImportError, PackageNotFoundError):
        return None


def _rank_placement(entry: Mapping[str, Any]) -> RankPlacement:
    for required in ("global_rank", "hostname"):
        if required not in entry:
            raise ValueError(f"placement entry is missing {required!r}: {dict(entry)!r}")
    groups = {}
    for name, membership in (entry.get("groups") or {}).items():
        groups[name] = GroupMembership(
            rank_in_group=int(membership["rank_in_group"]),
            global_ranks=[int(rank) for rank in membership["global_ranks"]],
        )
    layer_range = entry.get("pipeline_layer_range")
    expert_ids = {
        int(layer): [int(expert) for expert in experts]
        for layer, experts in (entry.get("local_expert_ids") or {}).items()
    }
    return RankPlacement(
        global_rank=int(entry["global_rank"]),
        hostname=str(entry["hostname"]),
        local_rank=int(entry.get("local_rank", 0)),
        gpu_uuid=entry.get("gpu_uuid"),
        pci_bus_id=entry.get("pci_bus_id"),
        groups=groups,
        pipeline_layer_range=(int(layer_range[0]), int(layer_range[1])) if layer_range else None,
        local_expert_ids=expert_ids,
        placement_epoch=int(entry.get("placement_epoch", 0)),
    )


def manifest_from_worker_entries(
    entries: Sequence[Mapping[str, Any]],
    framework_version: str | None = None,
    source: str = "extracted",
) -> PlacementManifest:
    """Assemble the per-worker dicts from one ``collective_rpc`` into a manifest.

    Ranks are sorted by ``global_rank`` and duplicates are rejected: a
    manifest with two entries for one rank would make ``by_rank`` ambiguous.
    The framework version is recorded because the extraction surface is
    internal vLLM API.
    """
    if not entries:
        raise ValueError("no placement entries: collective_rpc returned nothing")
    ranks = [_rank_placement(entry) for entry in entries]
    seen = {rank.global_rank for rank in ranks}
    if len(seen) != len(ranks):
        raise ValueError("duplicate global_rank in placement entries")
    ranks.sort(key=lambda rank: rank.global_rank)
    return PlacementManifest(
        ranks=ranks,
        source=source,
        framework="vllm",
        framework_version=framework_version or vllm_version(),
    )
