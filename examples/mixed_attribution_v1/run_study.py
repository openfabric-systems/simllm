"""Run the frozen BACK-43 mixed NVLink and fabric attribution study.

Every literal below is quoted from `expectations.md`, which landed before the
behavior existed. Nothing here is edited to match an observation: a relation
that fails is reported as failed.

The study replays one fixed step-record set, three prefill tokens of the
tracked Granite routing capture, under five cells that vary placement locality
and NVLink bandwidth, and takes each cell all the way to per-request TTFT and
TPOT with the medium partition that BACK-43 introduces.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from fractions import Fraction
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Frozen literals. Quoted from expectations.md; never edited to fit a result.
# ---------------------------------------------------------------------------

EXPECTATIONS_COMMIT = "4e7bcb9a356f9660b7f708c6e9f5d53735e264a3"
EXPECTATIONS_SHA256 = (
    "906ac794c3a2567bb8f07f57f4a1756e1001219a071dd9e581293b028899d1a6"
)
RUN_ROOT_ENV = "SIMLLM_MIXED_ATTRIBUTION_RUN_ROOT"

REQUEST_ID = "length-cap"
NUM_LAYERS = 24
NUM_EXPERTS = 32
TOP_K = 8
VECTOR_BYTES = 2_048
EP_RANKS = (0, 1, 2, 3)
STEP_COUNT = 3
COMPUTE_ESTIMATE_PS = 24_000
LINKSPEED_BPS = 400_000_000_000
NVLINK_RATE_BPS = 450_000_000_000
NVLINK_HALF_RATE_BPS = 225_000_000_000

#: napkin bounds for one all-local step's NVLink service, in picoseconds
NVLINK_SERVICE_FLOOR_PS = 240_000
NVLINK_SERVICE_CEILING_PS = 672_000
#: napkin bound for one all-remote step's whole service, in picoseconds
ALL_REMOTE_SERVICE_FLOOR_PS = 96_000_000
ALL_REMOTE_SERVICE_CEILING_PS = 110_000_000
#: 24 fully intra-node phases, one 2,048-byte vector each direction, at
#: ceil(2048 * 1e9 / 450e9) = 5 ns per phase
MIXED_LOCAL_ONLY_ARTIFACTS = 24
MIXED_NVLINK_TTFT_PS = 120_000
#: the same 24 phases at half the rate, ceil(2048 * 1e9 / 225e9) = 10 ns
MIXED_NVLINK_TTFT_HALF_PS = 240_000
MIXED_NVLINK_DELTA_PS = MIXED_NVLINK_TTFT_HALF_PS - MIXED_NVLINK_TTFT_PS
#: whole-nanosecond quantization slack over 48 phases, in picoseconds
QUANTIZATION_WINDOW_PS = 48_000

HOSTS = {
    "AAAA": ("a", "a", "a", "a"),
    "AABB": ("a", "a", "b", "b"),
    "ABCD": ("a", "b", "c", "d"),
}
UNIFORM_LAYOUT = "uniform"
NODE_LOCAL_LAYOUT = "node-local-even"

#: label -> (host layout, expert layout, NVLink bytes per second)
CELLS: dict[str, tuple[str, str, int]] = {
    "all-remote-450": ("ABCD", UNIFORM_LAYOUT, NVLINK_RATE_BPS),
    "all-local-450": ("AAAA", UNIFORM_LAYOUT, NVLINK_RATE_BPS),
    "all-local-225": ("AAAA", UNIFORM_LAYOUT, NVLINK_HALF_RATE_BPS),
    "mixed-450": ("AABB", NODE_LOCAL_LAYOUT, NVLINK_RATE_BPS),
    "mixed-225": ("AABB", NODE_LOCAL_LAYOUT, NVLINK_HALF_RATE_BPS),
}

# Evidence classes. They are never summed with each other.
FATAL_GUARD = "fatal-guard"
EXACT_ORACLE = "exact-oracle"
GENUINE_RISK = "behavioral-relation"

GUARD_IDS = ("G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8")
ORACLE_IDS = ("E1",)
BEHAVIORAL_IDS = ("F1", "F2", "F3", "F4")


def _check_frozen_registry() -> None:
    if len(EXPECTATIONS_COMMIT) != 40 or len(EXPECTATIONS_SHA256) != 64:
        raise AssertionError("the frozen provenance literals are malformed")
    if set(CELLS) != {
        "all-remote-450",
        "all-local-450",
        "all-local-225",
        "mixed-450",
        "mixed-225",
    }:
        raise AssertionError("the frozen cell matrix changed")
    for label, (hosts, layout, rate) in CELLS.items():
        if hosts not in HOSTS:
            raise AssertionError(f"cell {label} names an unknown host layout")
        if layout not in (UNIFORM_LAYOUT, NODE_LOCAL_LAYOUT):
            raise AssertionError(f"cell {label} names an unknown expert layout")
        if rate not in (NVLINK_RATE_BPS, NVLINK_HALF_RATE_BPS):
            raise AssertionError(f"cell {label} names an unfrozen NVLink rate")
    if NVLINK_RATE_BPS != 2 * NVLINK_HALF_RATE_BPS:
        raise AssertionError("the bandwidth axis is no longer a factor of two")
    if NVLINK_SERVICE_FLOOR_PS >= NVLINK_SERVICE_CEILING_PS:
        raise AssertionError("the NVLink napkin interval is empty")
    if ALL_REMOTE_SERVICE_FLOOR_PS >= ALL_REMOTE_SERVICE_CEILING_PS:
        raise AssertionError("the all-remote napkin interval is empty")
    #: the frozen NVLink interval is the 48-phase span of the per-phase bounds
    if NVLINK_SERVICE_FLOOR_PS != 48 * _ceil_div(VECTOR_BYTES * 10**9, NVLINK_RATE_BPS) * 1_000:
        raise AssertionError("the NVLink floor no longer matches its closed form")
    if NVLINK_SERVICE_CEILING_PS != 48 * _ceil_div(
        3 * VECTOR_BYTES * 10**9, NVLINK_RATE_BPS
    ) * 1_000:
        raise AssertionError("the NVLink ceiling no longer matches its closed form")
    if MIXED_NVLINK_TTFT_PS != MIXED_LOCAL_ONLY_ARTIFACTS * _ceil_div(
        VECTOR_BYTES * 10**9, NVLINK_RATE_BPS
    ) * 1_000:
        raise AssertionError("the mixed NVLink component left its closed form")
    if MIXED_NVLINK_TTFT_HALF_PS != MIXED_LOCAL_ONLY_ARTIFACTS * _ceil_div(
        VECTOR_BYTES * 10**9, NVLINK_HALF_RATE_BPS
    ) * 1_000:
        raise AssertionError("the halved NVLink component left its closed form")
    if MIXED_NVLINK_DELTA_PS != MIXED_NVLINK_TTFT_PS:
        raise AssertionError("the frozen bandwidth delta is not the frozen component")


def _ceil_div(numerator: int, denominator: int) -> int:
    return (numerator + denominator - 1) // denominator


# ---------------------------------------------------------------------------
# Fixture construction
# ---------------------------------------------------------------------------


def _trace_path() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "preplay_trace_v1/granite_length_cap.jsonl"
    )


def _routed_projection():
    from simllm.core import RequestBookkeeper
    from simllm.preplay import (
        RequestArrival,
        join_preplay_arrivals,
        project_preplay_routing,
    )

    joined = join_preplay_arrivals(
        (RequestArrival(request_id=REQUEST_ID, arrived_at_ps=0),),
        _trace_path(),
        RequestBookkeeper(),
    )
    return project_preplay_routing(joined)


def _expert_layout(projection, layout: str) -> dict[int, dict[int, list[int]]]:
    """Per-layer expert ownership for the four expert-parallel ranks.

    ``uniform`` gives rank ``r`` the experts ``[8r, 8r+8)`` at every layer.
    ``node-local-even`` moves, at every even layer, the eight experts the
    capture's first prefill token selects onto rank 1, so that phase stays
    inside the first node of the two-node placement. Both are declared
    fixtures, not calibrations.
    """

    uniform = {
        layer: {rank: list(range(rank * 8, rank * 8 + 8)) for rank in EP_RANKS}
        for layer in range(NUM_LAYERS)
    }
    if layout == UNIFORM_LAYOUT:
        return uniform
    token = projection.by_request_id(REQUEST_ID).prefill_tokens[0]
    per_layer = {}
    for layer in range(NUM_LAYERS):
        if layer % 2:
            per_layer[layer] = uniform[layer]
            continue
        selected = sorted(token.layers[layer].expert_ids)
        if len(selected) != TOP_K:
            raise AssertionError(f"layer {layer} did not select {TOP_K} experts")
        rest = [expert for expert in range(NUM_EXPERTS) if expert not in set(selected)]
        per_layer[layer] = {
            0: rest[:8],
            1: selected,
            2: rest[8:16],
            3: rest[16:24],
        }
    return per_layer


def _routed_supply(projection, layout: str):
    from simllm.placement import PlacementManifest, RankPlacement
    from simllm.traffic import ExpertPlacementSnapshot, RoutedMoeSupply

    per_layer = _expert_layout(projection, layout)
    manifest = PlacementManifest(
        ranks=[
            RankPlacement(
                global_rank=rank,
                hostname="expert-owner",
                local_rank=rank,
                local_expert_ids={
                    layer: per_layer[layer][rank] for layer in range(NUM_LAYERS)
                },
                placement_epoch=0,
            )
            for rank in EP_RANKS
        ]
    )
    return RoutedMoeSupply(
        engine_rank=0,
        routed_experts=projection,
        placements=(ExpertPlacementSnapshot.from_manifest(manifest, EP_RANKS),),
        step_placement_epochs=tuple((step, 0) for step in range(STEP_COUNT)),
    )


def _dims():
    from simllm.compute import ModelDims

    return ModelDims(
        num_layers=NUM_LAYERS,
        hidden_size=VECTOR_BYTES // 2,
        intermediate_size=512,
        num_heads=8,
        num_kv_heads=4,
        head_size=64,
        vocab_size=49_152,
        dtype_bytes=2,
        num_experts=NUM_EXPERTS,
        top_k=TOP_K,
        moe_intermediate_size=512,
        local_num_experts=8,
    )


def _physical_manifest(hosts: tuple[str, ...]):
    from simllm.placement import PlacementManifest, RankPlacement

    counts: dict[str, int] = {}
    ranks = []
    for global_rank, hostname in enumerate(hosts):
        local_rank = counts.get(hostname, 0)
        counts[hostname] = local_rank + 1
        ranks.append(
            RankPlacement(
                global_rank=global_rank,
                hostname=hostname,
                local_rank=local_rank,
            )
        )
    return PlacementManifest(ranks=ranks)


def _record(step_index: int, virtual_time_ps: int):
    """One step of the fixed record set: the capture's next prefill token."""

    from simllm.core import RequestPhase, ScheduledRequest, StepRecord

    return StepRecord(
        step_index=step_index,
        virtual_time_ps=virtual_time_ps,
        scheduled=[
            ScheduledRequest(
                REQUEST_ID,
                RequestPhase.PREFILL,
                1,
                context_length=step_index + 1,
            )
        ],
        num_sampled=1,
    )


# ---------------------------------------------------------------------------
# Independent recomputation, standard library only
# ---------------------------------------------------------------------------


def _recompute_step(locality: dict[str, Any]) -> dict[str, Any]:
    """Recompute one step's owner table and medium partition from its tuples."""

    kernel_ps = nvlink_ps = fabric_ps = co_critical_ps = base_ps = 0
    masked_nvlink_ps = masked_fabric_ps = 0
    owners = []
    for composed, fabric, local, base, medium in zip(
        locality["composed_phase_service_ps"],
        locality["fabric_phase_service_ps"],
        locality["local_phase_service_ps"],
        locality["base_phase_latency_ps"],
        locality["local_phase_medium"],
        strict=True,
    ):
        realized = max(local, fabric)
        if composed != base + realized:
            raise AssertionError("an artifact's composed service left its own terms")
        base_ps += base
        if realized == 0:
            owners.append("unserviced")
        elif medium == "gpu-compute":
            kernel_ps += realized
            owners.append("kernel")
        elif local == fabric:
            co_critical_ps += realized
            owners.append("co-critical")
        elif local > fabric:
            nvlink_ps += realized
            masked_fabric_ps += fabric
            owners.append("nvlink")
        else:
            fabric_ps += realized
            masked_nvlink_ps += local
            owners.append("fabric")
    return {
        "owners": owners,
        "kernel_ps": kernel_ps,
        "nvlink_ps": nvlink_ps,
        "fabric_ps": fabric_ps,
        "co_critical_ps": co_critical_ps,
        "collective_base_ps": base_ps,
        "masked_nvlink_ps": masked_nvlink_ps,
        "masked_fabric_ps": masked_fabric_ps,
        "total_ps": kernel_ps + nvlink_ps + fabric_ps + co_critical_ps + base_ps,
    }


def _recompute_request(cell: dict[str, Any]) -> dict[str, Any]:
    """Recompute TTFT, TPOT and both partitions from the steps and arrivals."""

    accounted_through_ps = 0
    first_token_at_ps = None
    last_token_at_ps = None
    ttft = {"queue_ps": 0}
    decode = {"queue_ps": 0}
    inter_token_ps = []
    for step in cell["steps"]:
        recomputed = _recompute_step(step["locality"])
        queue_ps = step["virtual_time_ps"] - accounted_through_ps
        interval = dict(recomputed)
        interval["queue_ps"] = queue_ps
        interval["total_ps"] = recomputed["total_ps"] + queue_ps
        accounted_through_ps = step["completed_at_ps"]
        if first_token_at_ps is None:
            first_token_at_ps = step["completed_at_ps"]
            ttft = interval
        else:
            inter_token_ps.append(step["completed_at_ps"] - last_token_at_ps)
            for key in (
                "queue_ps",
                "kernel_ps",
                "nvlink_ps",
                "fabric_ps",
                "co_critical_ps",
                "collective_base_ps",
                "total_ps",
            ):
                decode[key] = decode.get(key, 0) + interval[key]
        last_token_at_ps = step["completed_at_ps"]
    return {
        "ttft_ps": first_token_at_ps,
        "tpot_ps": (
            Fraction(sum(inter_token_ps), len(inter_token_ps))
            if inter_token_ps
            else None
        ),
        "token_count": len(cell["steps"]),
        "first_token_at_ps": first_token_at_ps,
        "last_token_at_ps": last_token_at_ps,
        "ttft_media": ttft,
        "decode_media": decode,
    }


def _media_json(media) -> dict[str, int]:
    return {
        "queue_ps": media.queue_ps,
        "kernel_ps": media.kernel_ps,
        "nvlink_ps": media.nvlink_ps,
        "fabric_ps": media.fabric_ps,
        "co_critical_ps": media.co_critical_ps,
        "collective_base_ps": media.collective_base_ps,
        "control_ps": media.control_ps,
        "total_ps": media.total_ps,
    }


def _attribution_json(attribution) -> dict[str, int]:
    return {
        "queue_ps": attribution.queue_ps,
        "kv_ps": attribution.kv_ps,
        "kernel_ps": attribution.kernel_ps,
        "dma_ps": attribution.dma_ps,
        "collective_ps": attribution.collective_ps,
        "nic_ps": attribution.nic_ps,
        "control_ps": attribution.control_ps,
        "total_ps": attribution.total_ps,
    }


def _locality_json(locality) -> dict[str, Any]:
    return {
        "authority": locality.authority,
        "artifact_count": locality.artifact_count,
        "backend_runs": locality.backend_runs,
        "total_directed_bytes": locality.total_directed_bytes,
        "fabric_directed_bytes": locality.fabric_directed_bytes,
        "nvlink_directed_bytes": locality.nvlink_directed_bytes,
        "nvlink_service_ps": locality.nvlink_service_ps,
        "compute_service_ps": locality.compute_service_ps,
        "nvlink_bandwidth_bytes_per_second": (
            locality.nvlink_bandwidth_bytes_per_second
        ),
        "composed_phase_service_ps": list(locality.composed_phase_service_ps),
        "fabric_phase_service_ps": list(locality.fabric_phase_service_ps),
        "local_phase_service_ps": list(locality.local_phase_service_ps),
        "base_phase_latency_ps": list(locality.base_phase_latency_ps),
        "local_phase_medium": list(locality.local_phase_medium),
    }


# ---------------------------------------------------------------------------
# Cell execution
# ---------------------------------------------------------------------------


def _run_cell(out: Path, label: str, projection) -> dict[str, Any]:
    from dataclasses import replace

    from simllm.backends import (
        HtsimRequestMetricReducer,
        HtsimStepSink,
        HtsimStepSinkConfig,
        attribute_step_detail,
    )
    from simllm.compute import ComputeProvider, DurationEstimate

    class FixedProvider(ComputeProvider):
        def estimate(self, kernel, gpu):
            return DurationEstimate(
                duration_ps=COMPUTE_ESTIMATE_PS,
                bound="declared-fixed",
            )

    host_layout, expert_layout, nvlink_rate = CELLS[label]
    sink = HtsimStepSink(
        HtsimStepSinkConfig(
            profile="rnic-nn-fluid",
            tp_ranks=(0,),
            dims=_dims(),
            workdir=out / label,
            ep_ranks=EP_RANKS,
            linkspeed_bps=LINKSPEED_BPS,
            provider=FixedProvider(),
            routed_moe_supply=_routed_supply(projection, expert_layout),
            placement_manifest=_physical_manifest(HOSTS[host_layout]),
            nvlink_bandwidth_bytes_per_second=nvlink_rate,
            num_goal_ranks=len(EP_RANKS),
        )
    )

    started = time.time()
    reducer = HtsimRequestMetricReducer({REQUEST_ID: 0})
    stripped_reducer = HtsimRequestMetricReducer({REQUEST_ID: 0})
    virtual_time_ps = 0
    steps: list[dict[str, Any]] = []
    stripped_steps: list[dict[str, Any]] = []
    for step_index in range(STEP_COUNT):
        record = _record(step_index, virtual_time_ps)
        result = sink(record)
        if result is None:
            raise AssertionError(f"cell {label} step {step_index} simulated nothing")
        virtual_time_ps = result.completed_at_ps
        locality = sink.locality_outcomes[step_index]
        network = sink.outcomes[step_index]
        step = attribute_step_detail(result, locality)
        metrics = reducer.consume(record, result, locality)
        #: the same step in the locality shape recorded before the projection
        stripped = replace(
            locality,
            local_phase_service_ps=(),
            base_phase_latency_ps=(),
            local_phase_medium=(),
        )
        stripped_ok = True
        stripped_rows: list[dict[str, Any]] = []
        try:
            stripped_step = attribute_step_detail(result, stripped)
            stripped_metrics = stripped_reducer.consume(record, result, stripped)
        except ValueError:
            stripped_ok = False
        else:
            stripped_rows = [
                {
                    "request_id": metric.request_id,
                    "latency_ps": metric.latency_ps,
                    "ttft_ps": metric.ttft_ps,
                    "tpot_ps": None if metric.tpot_ps is None else str(metric.tpot_ps),
                    "attribution": _attribution_json(metric.attribution),
                }
                for metric in stripped_metrics
            ]
            stripped_steps.append(
                {
                    "attribution": _attribution_json(stripped_step.attribution),
                    "media": _media_json(stripped_step.media),
                    "metrics": stripped_rows,
                }
            )
        steps.append(
            {
                "step_index": step_index,
                "virtual_time_ps": record.virtual_time_ps,
                "step_latency_ps": result.step_latency_ps,
                "completed_at_ps": result.completed_at_ps,
                "routing_mode": network.routing_mode,
                "placement_epoch": network.placement_epoch,
                "quiescent": network.quiescent,
                "attribution": _attribution_json(step.attribution),
                "media": _media_json(step.media),
                "masked": {
                    "nvlink_ps": step.masked.nvlink_ps,
                    "fabric_ps": step.masked.fabric_ps,
                },
                "locality": _locality_json(locality),
                "legacy_shape_accepted": stripped_ok,
                "metrics": [
                    {
                        "request_id": metric.request_id,
                        "latency_ps": metric.latency_ps,
                        "ttft_ps": metric.ttft_ps,
                        "tpot_ps": (
                            None if metric.tpot_ps is None else str(metric.tpot_ps)
                        ),
                        "attribution": _attribution_json(metric.attribution),
                    }
                    for metric in metrics
                ],
            }
        )

    (totals,) = reducer.totals()
    stripped_totals = stripped_reducer.totals()
    return {
        "name": label,
        "hosts": list(HOSTS[host_layout]),
        "host_layout": host_layout,
        "expert_layout": expert_layout,
        "nvlink_bandwidth_bytes_per_second": nvlink_rate,
        "wall_seconds": round(time.time() - started, 2),
        "steps": steps,
        "stripped_steps": stripped_steps,
        "totals": {
            "request_id": totals.request_id,
            "ttft_ps": totals.ttft_ps,
            "tpot_ps": None if totals.tpot_ps is None else str(totals.tpot_ps),
            "token_count": totals.token_count,
            "first_token_at_ps": totals.first_token_at_ps,
            "last_token_at_ps": totals.last_token_at_ps,
            "ttft_attribution": _attribution_json(totals.ttft_attribution),
            "decode_attribution": _attribution_json(totals.decode_attribution),
            "ttft_media": _media_json(totals.ttft_media),
            "decode_media": _media_json(totals.decode_media),
        },
        "stripped_totals": [
            {
                "request_id": row.request_id,
                "ttft_ps": row.ttft_ps,
                "tpot_ps": None if row.tpot_ps is None else str(row.tpot_ps),
                "token_count": row.token_count,
                "first_token_at_ps": row.first_token_at_ps,
                "last_token_at_ps": row.last_token_at_ps,
                "ttft_attribution": _attribution_json(row.ttft_attribution),
                "decode_attribution": _attribution_json(row.decode_attribution),
            }
            for row in stripped_totals
        ],
    }


# ---------------------------------------------------------------------------
# Guards and relations
# ---------------------------------------------------------------------------


def _guard_rows(cells: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []

    remote = cells["all-remote-450"]
    identity = {
        "steps_compared": len(remote["stripped_steps"]),
        "step_differences": [],
        "totals_differences": [],
    }
    for index, (step, stripped) in enumerate(
        zip(remote["steps"], remote["stripped_steps"], strict=False)
    ):
        if step["attribution"] != stripped["attribution"]:
            identity["step_differences"].append({"step": index, "field": "attribution"})
        if step["media"] != stripped["media"]:
            identity["step_differences"].append({"step": index, "field": "media"})
        if step["metrics"] != stripped["metrics"]:
            identity["step_differences"].append({"step": index, "field": "metrics"})
    stripped_totals = remote["stripped_totals"]
    if len(stripped_totals) != 1:
        identity["totals_differences"].append("stripped replay published no totals")
    else:
        for field, value in stripped_totals[0].items():
            if remote["totals"][field] != value:
                identity["totals_differences"].append(field)
    rows.append(
        {
            "id": "G1",
            "class": FATAL_GUARD,
            "claim": "the all-remote path is byte-identical without the projection",
            "observed": identity,
            "held": (
                identity["steps_compared"] == STEP_COUNT
                and not identity["step_differences"]
                and not identity["totals_differences"]
            ),
        }
    )

    conservation = []
    for label, cell in cells.items():
        for step in cell["steps"]:
            conservation.append(
                step["attribution"]["total_ps"] == step["step_latency_ps"]
                and step["media"]["total_ps"] == step["step_latency_ps"]
                and all(
                    metric["attribution"]["total_ps"] == metric["latency_ps"]
                    for metric in step["metrics"]
                )
            )
        totals = cell["totals"]
        conservation.append(
            totals["ttft_media"]["total_ps"] == totals["ttft_ps"]
            and totals["decode_media"]["total_ps"]
            == totals["decode_attribution"]["total_ps"]
        )
        del label
    rows.append(
        {
            "id": "G2",
            "class": FATAL_GUARD,
            "claim": "every partition conserves its own realized interval",
            "observed": {"checks": len(conservation), "failures": conservation.count(False)},
            "held": all(conservation),
        }
    )

    roll_up = []
    for cell in cells.values():
        for step in cell["steps"]:
            media = step["media"]
            attribution = step["attribution"]
            roll_up.append(
                media["kernel_ps"] == attribution["kernel_ps"]
                and media["queue_ps"] == attribution["queue_ps"]
                and media["control_ps"] == attribution["control_ps"]
                and media["nvlink_ps"]
                + media["fabric_ps"]
                + media["co_critical_ps"]
                + media["collective_base_ps"]
                == attribution["collective_ps"]
            )
    rows.append(
        {
            "id": "G3",
            "class": FATAL_GUARD,
            "claim": "medium components roll up to the coarse partition",
            "observed": {"checks": len(roll_up), "failures": roll_up.count(False)},
            "held": all(roll_up),
        }
    )

    inactive = []
    for cell in cells.values():
        for step in cell["steps"]:
            for metric in step["metrics"]:
                attribution = metric["attribution"]
                inactive.append(
                    not attribution["kv_ps"]
                    and not attribution["dma_ps"]
                    and not attribution["nic_ps"]
                    and not attribution["control_ps"]
                )
    rows.append(
        {
            "id": "G4",
            "class": FATAL_GUARD,
            "claim": "kv, dma, nic and control stay exactly zero",
            "observed": {"intervals": len(inactive), "violations": inactive.count(False)},
            "held": all(inactive),
        }
    )

    health = []
    for cell in cells.values():
        health.append(len(cell["steps"]) == STEP_COUNT)
        health.append(cell["totals"]["ttft_ps"] > 0)
        for step in cell["steps"]:
            health.append(
                step["routing_mode"] == "captured"
                and step["placement_epoch"] == 0
                and step["quiescent"]
            )
    rows.append(
        {
            "id": "G5",
            "class": FATAL_GUARD,
            "claim": "every cell reduces three captured, quiescent steps",
            "observed": {"checks": len(health), "failures": health.count(False)},
            "held": all(health),
        }
    )

    projection = []
    for cell in cells.values():
        for step in cell["steps"]:
            locality = step["locality"]
            lengths = {
                len(locality["composed_phase_service_ps"]),
                len(locality["fabric_phase_service_ps"]),
                len(locality["local_phase_service_ps"]),
                len(locality["base_phase_latency_ps"]),
                len(locality["local_phase_medium"]),
            }
            declared = sum(
                local
                for local, medium in zip(
                    locality["local_phase_service_ps"],
                    locality["local_phase_medium"],
                    strict=True,
                )
                if medium == "nvlink"
            )
            projection.append(
                len(lengths) == 1
                and declared == locality["nvlink_service_ps"]
                and lengths == {locality["artifact_count"]}
            )
    rows.append(
        {
            "id": "G6",
            "class": FATAL_GUARD,
            "claim": "the per-artifact projection agrees with the step totals",
            "observed": {"steps": len(projection), "failures": projection.count(False)},
            "held": all(projection),
        }
    )

    remote_components = [
        (step["media"]["nvlink_ps"], step["masked"]["nvlink_ps"])
        for step in cells["all-remote-450"]["steps"]
    ]
    rows.append(
        {
            "id": "G7",
            "class": FATAL_GUARD,
            "claim": "the all-remote cell charges no NVLink component",
            "observed": {"per_step_nvlink_and_masked_ps": remote_components},
            "held": all(
                owned == 0 and masked == 0 for owned, masked in remote_components
            ),
        }
    )

    local_backends = [
        (
            step["locality"]["backend_runs"],
            step["locality"]["fabric_directed_bytes"],
            step["media"]["fabric_ps"],
        )
        for label in ("all-local-450", "all-local-225")
        for step in cells[label]["steps"]
    ]
    rows.append(
        {
            "id": "G8",
            "class": FATAL_GUARD,
            "claim": "the all-local cells run no backend and carry no fabric work",
            "observed": {"runs_bytes_service": local_backends},
            "held": all(entry == (0, 0, 0) for entry in local_backends),
        }
    )
    return rows


def _oracle_rows(cells: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    mismatches = []
    compared = 0
    for label, cell in cells.items():
        for step in cell["steps"]:
            recomputed = _recompute_step(step["locality"])
            media = step["media"]
            compared += 1
            for key in (
                "kernel_ps",
                "nvlink_ps",
                "fabric_ps",
                "co_critical_ps",
                "collective_base_ps",
            ):
                if recomputed[key] != media[key]:
                    mismatches.append({"cell": label, "step": step["step_index"], "field": key})
            if recomputed["masked_nvlink_ps"] != step["masked"]["nvlink_ps"]:
                mismatches.append(
                    {"cell": label, "step": step["step_index"], "field": "masked_nvlink_ps"}
                )
            if recomputed["masked_fabric_ps"] != step["masked"]["fabric_ps"]:
                mismatches.append(
                    {"cell": label, "step": step["step_index"], "field": "masked_fabric_ps"}
                )
        recomputed_request = _recompute_request(cell)
        totals = cell["totals"]
        compared += 1
        published_tpot = totals["tpot_ps"]
        expected_tpot = (
            None
            if recomputed_request["tpot_ps"] is None
            else str(recomputed_request["tpot_ps"])
        )
        if published_tpot != expected_tpot:
            mismatches.append({"cell": label, "field": "tpot_ps"})
        for key in ("ttft_ps", "token_count", "first_token_at_ps", "last_token_at_ps"):
            if recomputed_request[key] != totals[key]:
                mismatches.append({"cell": label, "field": key})
        for view in ("ttft_media", "decode_media"):
            for key in (
                "queue_ps",
                "kernel_ps",
                "nvlink_ps",
                "fabric_ps",
                "co_critical_ps",
                "collective_base_ps",
                "total_ps",
            ):
                if recomputed_request[view].get(key, 0) != totals[view][key]:
                    mismatches.append({"cell": label, "field": f"{view}.{key}"})
    return [
        {
            "id": "E1",
            "class": EXACT_ORACLE,
            "claim": "an independent recomputation reproduces every published value",
            "observed": {"comparisons": compared, "mismatches": mismatches},
            "passed": not mismatches,
        }
    ]


def _behavioral_rows(cells: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []

    f1 = []
    for label in ("all-local-450", "all-local-225"):
        for step in cells[label]["steps"]:
            media = step["media"]
            expected_floor = NVLINK_SERVICE_FLOOR_PS
            expected_ceiling = NVLINK_SERVICE_CEILING_PS
            if label.endswith("225"):
                expected_floor *= 2
                expected_ceiling *= 2
            f1.append(
                {
                    "cell": label,
                    "step": step["step_index"],
                    "fabric_ps": media["fabric_ps"],
                    "kernel_ps": media["kernel_ps"],
                    "nvlink_ps": media["nvlink_ps"],
                    "interval_ps": [expected_floor, expected_ceiling],
                    "passed": (
                        media["fabric_ps"] == 0
                        and media["kernel_ps"] == COMPUTE_ESTIMATE_PS
                        and expected_floor <= media["nvlink_ps"] <= expected_ceiling
                        and media["kernel_ps"] + media["nvlink_ps"]
                        == step["step_latency_ps"]
                    ),
                }
            )
    rows.append(
        {
            "id": "F1",
            "class": GENUINE_RISK,
            "claim": "an all-local step is kernel plus NVLink, inside the napkin interval",
            "instances": f1,
            "passed": all(row["passed"] for row in f1),
        }
    )

    mixed = cells["mixed-450"]["steps"][0]
    owners = _recompute_step(mixed["locality"])["owners"]
    f2 = {
        "nvlink_ps": mixed["media"]["nvlink_ps"],
        "fabric_ps": mixed["media"]["fabric_ps"],
        "kernel_ps": mixed["media"]["kernel_ps"],
        "expected_nvlink_ps": MIXED_NVLINK_TTFT_PS,
        "nvlink_owned_artifacts": owners.count("nvlink"),
        "fabric_owned_artifacts": owners.count("fabric"),
        "ttft_ps": cells["mixed-450"]["totals"]["ttft_ps"],
        "ttft_media_total_ps": cells["mixed-450"]["totals"]["ttft_media"]["total_ps"],
    }
    f2["passed"] = (
        f2["nvlink_ps"] == MIXED_NVLINK_TTFT_PS
        and f2["fabric_ps"] > 0
        and f2["kernel_ps"] == COMPUTE_ESTIMATE_PS
        and f2["nvlink_owned_artifacts"] >= MIXED_LOCAL_ONLY_ARTIFACTS
        and f2["fabric_owned_artifacts"] >= MIXED_LOCAL_ONLY_ARTIFACTS
        and f2["ttft_media_total_ps"] == f2["ttft_ps"]
    )
    rows.append(
        {
            "id": "F2",
            "class": GENUINE_RISK,
            "claim": "one step carries both owners and its components total its TTFT",
            "instances": [f2],
            "passed": f2["passed"],
        }
    )

    f3 = []
    fast = cells["mixed-450"]
    slow = cells["mixed-225"]
    fast_owners = [
        _recompute_step(step["locality"])["owners"] for step in fast["steps"]
    ]
    slow_owners = [
        _recompute_step(step["locality"])["owners"] for step in slow["steps"]
    ]
    f3.append(
        {
            "cell_pair": "mixed",
            "ttft_delta_ps": slow["totals"]["ttft_ps"] - fast["totals"]["ttft_ps"],
            "nvlink_delta_ps": (
                slow["totals"]["ttft_media"]["nvlink_ps"]
                - fast["totals"]["ttft_media"]["nvlink_ps"]
            ),
            "expected_delta_ps": MIXED_NVLINK_DELTA_PS,
            "fabric_identical": (
                slow["totals"]["ttft_media"]["fabric_ps"]
                == fast["totals"]["ttft_media"]["fabric_ps"]
            ),
            "owners_identical": fast_owners == slow_owners,
            "passed": (
                slow["totals"]["ttft_ps"] - fast["totals"]["ttft_ps"]
                == MIXED_NVLINK_DELTA_PS
                and slow["totals"]["ttft_media"]["nvlink_ps"]
                - fast["totals"]["ttft_media"]["nvlink_ps"]
                == MIXED_NVLINK_DELTA_PS
                and slow["totals"]["ttft_media"]["fabric_ps"]
                == fast["totals"]["ttft_media"]["fabric_ps"]
                and fast_owners == slow_owners
            ),
        }
    )
    for fast_step, slow_step in zip(
        cells["all-local-450"]["steps"],
        cells["all-local-225"]["steps"],
        strict=True,
    ):
        fast_nvlink = fast_step["media"]["nvlink_ps"]
        slow_nvlink = slow_step["media"]["nvlink_ps"]
        f3.append(
            {
                "cell_pair": "all-local",
                "step": fast_step["step_index"],
                "nvlink_450_ps": fast_nvlink,
                "nvlink_225_ps": slow_nvlink,
                "bracket_ps": [2 * fast_nvlink - QUANTIZATION_WINDOW_PS, 2 * fast_nvlink],
                "passed": (
                    2 * fast_nvlink - QUANTIZATION_WINDOW_PS
                    <= slow_nvlink
                    <= 2 * fast_nvlink
                    and slow_step["step_latency_ps"] - fast_step["step_latency_ps"]
                    == slow_nvlink - fast_nvlink
                ),
            }
        )
    rows.append(
        {
            "id": "F3",
            "class": GENUINE_RISK,
            "claim": "halving the NVLink rate moves only the NVLink-owned service",
            "instances": f3,
            "passed": all(row["passed"] for row in f3),
        }
    )

    ordering = {
        label: cells[label]["totals"]["ttft_ps"]
        for label in ("all-local-450", "mixed-450", "all-remote-450")
    }
    fabric_components = {
        label: cells[label]["totals"]["ttft_media"]["fabric_ps"]
        for label in ("all-local-450", "mixed-450", "all-remote-450")
    }
    remote_service_ps = cells["all-remote-450"]["steps"][0]["step_latency_ps"]
    f4 = {
        "ttft_ps": ordering,
        "fabric_component_ps": fabric_components,
        "all_remote_step_service_ps": remote_service_ps,
        "all_remote_interval_ps": [
            ALL_REMOTE_SERVICE_FLOOR_PS,
            ALL_REMOTE_SERVICE_CEILING_PS,
        ],
    }
    f4["passed"] = (
        ordering["all-local-450"] < ordering["mixed-450"] < ordering["all-remote-450"]
        and fabric_components["all-local-450"] == 0
        and fabric_components["all-local-450"]
        < fabric_components["mixed-450"]
        < fabric_components["all-remote-450"]
        and ALL_REMOTE_SERVICE_FLOOR_PS
        <= remote_service_ps
        <= ALL_REMOTE_SERVICE_CEILING_PS
    )
    rows.append(
        {
            "id": "F4",
            "class": GENUINE_RISK,
            "claim": "locality orders TTFT and the all-remote step meets its bound",
            "instances": [f4],
            "passed": f4["passed"],
        }
    )
    return rows


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(*args: str) -> str:
    return subprocess.run(
        ("git", *args),
        cwd=Path(__file__).resolve().parents[2],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def check_only(args: argparse.Namespace) -> None:
    _check_frozen_registry()
    print(
        f"check-only run-dir={args.run_dir}; validated the frozen BACK-43 cell "
        "matrix and its closed-form literals, and produced no artifacts"
    )


def _provenance(args: argparse.Namespace) -> dict[str, str]:
    expectations = Path(__file__).with_name("expectations.md")
    if _sha256(expectations) != EXPECTATIONS_SHA256:
        raise RuntimeError("expectations.md differs from the final freeze")
    frozen = _git("show", f"{EXPECTATIONS_COMMIT}:examples/mixed_attribution_v1/expectations.md")
    if frozen != expectations.read_text(encoding="utf-8").rstrip("\n"):
        raise RuntimeError("the checked-out expectations do not match their freeze")
    configured_root = os.environ.get(RUN_ROOT_ENV)
    if not configured_root:
        raise RuntimeError(f"{RUN_ROOT_ENV} must name this branch's external run root")
    run_root = Path(configured_root).resolve()
    try:
        args.run_dir.resolve().relative_to(run_root)
    except ValueError as error:
        raise ValueError(f"study output must remain under {RUN_ROOT_ENV}") from error
    if args.run_dir.exists():
        raise FileExistsError(f"study output already exists: {args.run_dir}")
    htsim = args.htsim_rnic.resolve()
    if not htsim.is_file() or not os.access(htsim, os.X_OK):
        raise FileNotFoundError(f"htsim_rnic is not an executable file: {htsim}")
    txt2bin_value = os.environ.get("SIMLLM_TXT2BIN")
    if not txt2bin_value:
        raise RuntimeError("SIMLLM_TXT2BIN must name the registered converter")
    txt2bin = Path(txt2bin_value)
    if not txt2bin.is_file() or not os.access(txt2bin, os.X_OK):
        raise FileNotFoundError(f"SIMLLM_TXT2BIN is not executable: {txt2bin}")
    os.environ["SIMLLM_HTSIM_RNIC"] = str(htsim)
    return {
        "simllm_revision": _git("rev-parse", "HEAD"),
        "expectations_commit": EXPECTATIONS_COMMIT,
        "expectations_sha256": EXPECTATIONS_SHA256,
        "htsim_gitlink": _git("rev-parse", "HEAD:third_party/htsim"),
        "htsim_rnic_sha256": _sha256(htsim),
        "txt2bin_sha256": _sha256(txt2bin),
        "trace_sha256": _sha256(_trace_path()),
        "python": platform.python_version(),
        "platform": platform.platform(),
    }


def run(args: argparse.Namespace) -> int:
    _check_frozen_registry()
    provenance = _provenance(args)
    args.run_dir.mkdir(parents=True)
    projection = _routed_projection()
    cells = {
        label: _run_cell(args.run_dir, label, projection) for label in CELLS
    }
    guards = _guard_rows(cells)
    oracles = _oracle_rows(cells)
    behavioral = _behavioral_rows(cells)
    reported = (
        tuple(row["id"] for row in guards),
        tuple(row["id"] for row in oracles),
        tuple(row["id"] for row in behavioral),
    )
    if reported != (GUARD_IDS, ORACLE_IDS, BEHAVIORAL_IDS):
        raise AssertionError("the reported evidence set is not the frozen one")
    void = [row["id"] for row in guards if not row["held"]]
    summary = {
        "study": "mixed_attribution_v1",
        "provenance": provenance,
        "cells": cells,
        "fatal_guards": guards,
        "exact_oracle_relations": oracles,
        "behavioral_relations": behavioral,
        "verdict": {
            "void": bool(void),
            "violated_guards": void,
            "exact_oracle_passed": sum(1 for row in oracles if row["passed"]),
            "exact_oracle_total": len(oracles),
            "behavioral_passed": sum(1 for row in behavioral if row["passed"]),
            "behavioral_total": len(behavioral),
        },
    }
    (args.run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    for row in guards:
        print(f"{row['id']} {'held' if row['held'] else 'VIOLATED'}: {row['claim']}")
    for row in oracles + behavioral:
        print(f"{row['id']} {'passed' if row['passed'] else 'FAILED'}: {row['claim']}")
    if void:
        print(f"VOID: fatal guards violated: {', '.join(void)}")
        return 1
    failed = [row["id"] for row in oracles + behavioral if not row["passed"]]
    print(
        f"scored exact-oracle {summary['verdict']['exact_oracle_passed']} of "
        f"{summary['verdict']['exact_oracle_total']}, behavioral "
        f"{summary['verdict']['behavioral_passed']} of "
        f"{summary['verdict']['behavioral_total']}"
    )
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help=f"output directory, which must sit under {RUN_ROOT_ENV}",
    )
    parser.add_argument(
        "--htsim-rnic",
        type=Path,
        default=Path("htsim_rnic"),
        help="path to the htsim_rnic binary this run executes",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="validate the frozen registry, write nothing and import nothing",
    )
    args = parser.parse_args(argv)
    if args.check_only:
        check_only(args)
        return 0
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
