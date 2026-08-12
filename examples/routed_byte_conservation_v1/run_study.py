"""Run the frozen VLLM-24 independent routed-MoE byte conservation study."""

from __future__ import annotations

import argparse
from pathlib import Path

EVIDENCE_AUTHORED_AGAINST = "aeb40ac95cdd8163942297335948c94df0376e04"
VLLM_AUTHORED_AGAINST = "568afb3a13806beb53bb2e6bd518269357b237c0"

#: the nine frozen conservation rules, in evaluation order
ALWAYS_RULES = (
    "source-attribution",
    "destination-legality",
    "owner-egress",
    "transpose-symmetry",
    "step-hop-bound",
)
CAPTURED_ONLY_RULES = (
    "vector-granularity",
    "request-identity",
    "per-request-hop-bound",
    "per-layer-hop-bound",
)

#: captured Granite geometry, read off the tracked preplay trace header
EXPERT_COUNT = 32
TOP_K = 8
NUM_LAYERS = 24
HIDDEN_SIZE = 1_024
DTYPE_BYTES = 2
VECTOR_BYTES = HIDDEN_SIZE * DTYPE_BYTES

#: frozen steps: name -> total_new_tokens
STEP_TOKENS = {"prefill": 22, "decode": 1}
EP_WORLDS = (2, 8)
ARMS = ("owner-attributed", "source-replicated")

#: exact closed-form oracles, hops
EXPECTED_STEP_HOP_BOUND = {"prefill": 8_448, "decode": 384}
EXPECTED_PER_LAYER_HOP_BOUND = {
    ("prefill", 2): 22,
    ("prefill", 8): 154,
    ("decode", 2): 1,
    ("decode", 8): 7,
}
#: hops_A ceilings, i.e. T * min(top_k, W - 1) * num_layers * 2
EXPECTED_ARM_A_HOP_CEILING = {
    ("prefill", 2): 1_056,
    ("prefill", 8): 7_392,
    ("decode", 2): 48,
    ("decode", 8): 336,
}
#: hops_A floors from the block structure: at W = 8 every token touches at
#: least ceil(top_k / (32 // 8)) = 2 owner blocks, so at least one is remote
EXPECTED_ARM_A_HOP_FLOOR = {
    ("prefill", 2): 0,
    ("prefill", 8): 1_056,
    ("decode", 2): 0,
    ("decode", 8): 48,
}

#: which rule detects the replicated arm in which cell
EXPECTED_DETECTION = {
    ("prefill", 2, "source-attribution"): True,
    ("prefill", 8, "source-attribution"): True,
    ("decode", 2, "source-attribution"): True,
    ("decode", 8, "source-attribution"): True,
    ("prefill", 2, "step-hop-bound"): False,
    ("prefill", 8, "step-hop-bound"): True,
    ("decode", 2, "step-hop-bound"): False,
    ("decode", 8, "step-hop-bound"): True,
}

EXPECTED_SCORED_INSTANCES = 9
EXPECTED_FATAL_GUARDS = ("arm-a-conserves", "replication-multiplier")


def _check_frozen_registry() -> None:
    if len(ALWAYS_RULES) + len(CAPTURED_ONLY_RULES) != 9:
        raise AssertionError("the frozen rule registry is not the nine named rules")
    if set(ALWAYS_RULES) & set(CAPTURED_ONLY_RULES):
        raise AssertionError("a rule appears in both applicability classes")
    if VECTOR_BYTES != 2_048:
        raise AssertionError("hidden vector byte arithmetic drifted")

    for step, tokens in STEP_TOKENS.items():
        bound = tokens * TOP_K * NUM_LAYERS * 2
        if EXPECTED_STEP_HOP_BOUND[step] != bound:
            raise AssertionError(f"step hop bound arithmetic drifted at {step}")
        for world in EP_WORLDS:
            per_layer = tokens * min(TOP_K, world - 1)
            if EXPECTED_PER_LAYER_HOP_BOUND[(step, world)] != per_layer:
                raise AssertionError(
                    f"per-layer hop bound arithmetic drifted at {step} W={world}"
                )
            ceiling = per_layer * NUM_LAYERS * 2
            if EXPECTED_ARM_A_HOP_CEILING[(step, world)] != ceiling:
                raise AssertionError(
                    f"arm A hop ceiling arithmetic drifted at {step} W={world}"
                )
            floor = EXPECTED_ARM_A_HOP_FLOOR[(step, world)]
            if floor < 0 or floor > ceiling:
                raise AssertionError(
                    f"arm A physical interval is empty at {step} W={world}"
                )

    # The bound cannot detect an unbiased W-fold replication when even the
    # ceiling of the correct arm, multiplied by W, stays inside the bound.
    for step, tokens in STEP_TOKENS.items():
        for world in EP_WORLDS:
            ceiling = EXPECTED_ARM_A_HOP_CEILING[(step, world)]
            floor = EXPECTED_ARM_A_HOP_FLOOR[(step, world)]
            bound = EXPECTED_STEP_HOP_BOUND[step]
            certain_miss = world * ceiling <= bound
            certain_hit = world * floor > bound
            expected = EXPECTED_DETECTION[(step, world, "step-hop-bound")]
            if certain_miss and expected:
                raise AssertionError(
                    f"frozen detection claims a hit the bound cannot produce at "
                    f"{step} W={world}"
                )
            if certain_hit and not expected:
                raise AssertionError(
                    f"frozen detection claims a miss the bound cannot produce at "
                    f"{step} W={world}"
                )
    if any(
        not detected
        for (_, _, rule), detected in EXPECTED_DETECTION.items()
        if rule == "source-attribution"
    ):
        raise AssertionError("structural source replication must be detected everywhere")
    if EXPECTED_SCORED_INSTANCES != 2 + 2 + 4 + 1:
        raise AssertionError("scored instance arithmetic drifted")
    if len(EXPECTED_FATAL_GUARDS) != 2:
        raise AssertionError("fatal guard registry drifted")
    if set(ARMS) != {"owner-attributed", "source-replicated"}:
        raise AssertionError("arm registry drifted")
    if EXPERT_COUNT % max(EP_WORLDS) or EXPERT_COUNT % min(EP_WORLDS):
        raise AssertionError("contiguous owner blocks do not divide the expert count")


def check_only(args: argparse.Namespace) -> None:
    _check_frozen_registry()
    print(
        f"check-only out={args.out}; validated the frozen VLLM-24 rule registry, "
        f"{len(STEP_TOKENS) * len(EP_WORLDS) * len(ARMS)} cells and "
        f"{EXPECTED_SCORED_INSTANCES} scored instances, and produced no artifacts"
    )



# --- production run ---------------------------------------------------------
#
# Everything above this line is the frozen expectation. Everything below reads
# the implementation and reports what it observed. The two fatal guards are
# evaluated first: arm A must conserve, and the injected fault must be the
# exact W-fold replication the freeze describes. Only then are the detection
# expectations scored, because they are only interpretable once the fault is
# the one that was registered.

_FIXTURE = "examples/preplay_trace_v1/granite_length_cap.jsonl"
_REQUEST_ID = "length-cap"


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _observed_provenance() -> dict:
    import subprocess

    def _git(*arguments: str) -> str | None:
        try:
            completed = subprocess.run(
                ["git", *arguments],
                cwd=_repository_root(),
                capture_output=True,
                text=True,
                check=True,
            )
        except Exception:  # noqa: BLE001 - provenance is best effort
            return None
        return completed.stdout.strip() or None

    return {
        "repository_commit": _git("rev-parse", "HEAD"),
        "repository_dirty": bool(_git("status", "--porcelain")),
        "evidence_authored_against": EVIDENCE_AUTHORED_AGAINST,
        "vllm_authored_against": VLLM_AUTHORED_AGAINST,
        "capture": _FIXTURE,
    }


def _granite_dims():
    from simllm.compute import ModelDims

    return ModelDims(
        num_layers=NUM_LAYERS,
        hidden_size=HIDDEN_SIZE,
        intermediate_size=512,
        num_heads=16,
        num_kv_heads=8,
        head_size=64,
        vocab_size=49_155,
        dtype_bytes=DTYPE_BYTES,
        num_experts=EXPERT_COUNT,
        top_k=TOP_K,
        moe_intermediate_size=512,
        local_num_experts=EXPERT_COUNT // 8,
    )


def _manifest(world: int):
    """Contiguous owner blocks of ``32 // world`` experts on every layer."""

    from simllm.placement import PlacementManifest, RankPlacement

    block = EXPERT_COUNT // world
    return PlacementManifest(
        ranks=[
            RankPlacement(
                global_rank=rank,
                hostname="host",
                local_rank=rank,
                local_expert_ids={
                    layer: list(range(rank * block, (rank + 1) * block))
                    for layer in range(NUM_LAYERS)
                },
                placement_epoch=0,
            )
            for rank in range(world)
        ]
    )


def _supply(world: int):
    from simllm.core import RequestBookkeeper
    from simllm.preplay import (
        RequestArrival,
        join_preplay_arrivals,
        project_preplay_routing,
    )
    from simllm.traffic import ExpertPlacementSnapshot, RoutedMoeSupply

    run = join_preplay_arrivals(
        (RequestArrival(request_id=_REQUEST_ID, arrived_at_ps=0),),
        _repository_root() / _FIXTURE,
        RequestBookkeeper(),
    )
    ranks = tuple(range(world))
    return RoutedMoeSupply(
        engine_rank=0,
        routed_experts=project_preplay_routing(run),
        placements=(
            ExpertPlacementSnapshot.from_manifest(_manifest(world), ranks),
        ),
        step_placement_epochs=((0, 0), (1, 0)),
    )


def _record(tokens: int, step_index: int = 0):
    from simllm.core import RequestPhase, ScheduledRequest, StepRecord

    return StepRecord(
        step_index=step_index,
        virtual_time_ps=0,
        scheduled=[
            ScheduledRequest(
                _REQUEST_ID,
                RequestPhase.PREFILL,
                tokens,
                context_length=tokens,
            )
        ],
        num_sampled=1,
    )


def _ownership(record, dims):
    from simllm.traffic import RoutedTokenOwnership

    return RoutedTokenOwnership(
        engine_rank=0,
        request_token_counts=tuple(
            (request.request_id, request.num_new_tokens)
            for request in record.scheduled
            if request.num_new_tokens > 0
        ),
        num_layers=dims.num_layers,
        top_k=dims.top_k,
        vector_bytes=dims.hidden_size * dims.dtype_bytes,
    )


def _arm_a_tables(record, dims, world):
    from simllm.traffic import RoutedPhaseTable, step_moe_alltoalls

    operations = step_moe_alltoalls(
        record,
        dims,
        tuple(range(world)),
        routed_supply=_supply(world),
    )
    return tuple(
        RoutedPhaseTable(
            layer=operation.layer,
            phase=operation.phase,
            pair_payload_bytes=operation.pair_payload_bytes,
            request_pair_payload_bytes=operation.request_pair_payload_bytes,
        )
        for operation in operations
    )


def _replicate_sources(tables, world: int):
    """The pre-TRAF-25 shape: every EP rank is a source of the same tokens.

    Each replica keeps the owner's per-destination distribution and sends its
    one self-addressed share back to the owner instead, so a replica moves
    exactly the owner's bytes and the whole table moves ``world`` times them.
    """

    from simllm.traffic import RoutedPhaseTable

    replicated = []
    for table in tables:
        if table.phase != "dispatch":
            continue
        pairs: dict[tuple[int, int], int] = {}
        requests: dict[tuple[str, int, int], int] = {}
        for source in range(world):
            for owner, destination, size in table.pair_payload_bytes:
                target = destination if destination != source else owner
                key = (source, target)
                pairs[key] = pairs.get(key, 0) + size
            for request_id, owner, destination, size in (
                table.request_pair_payload_bytes
            ):
                target = destination if destination != source else owner
                key = (request_id, source, target)
                requests[key] = requests.get(key, 0) + size
        dispatch_pairs = tuple(
            (source, destination, size)
            for (source, destination), size in sorted(pairs.items())
        )
        dispatch_requests = tuple(
            (request_id, source, destination, size)
            for (request_id, source, destination), size in sorted(requests.items())
        )
        replicated.append(
            RoutedPhaseTable(
                layer=table.layer,
                phase="dispatch",
                pair_payload_bytes=dispatch_pairs,
                request_pair_payload_bytes=dispatch_requests,
            )
        )
        replicated.append(
            RoutedPhaseTable(
                layer=table.layer,
                phase="combine",
                pair_payload_bytes=tuple(
                    sorted(
                        (destination, source, size)
                        for source, destination, size in dispatch_pairs
                    )
                ),
                request_pair_payload_bytes=tuple(
                    sorted(
                        (request_id, destination, source, size)
                        for request_id, source, destination, size in dispatch_requests
                    )
                ),
            )
        )
    return tuple(replicated)


def _dispatch_sources(tables) -> tuple[int, ...]:
    """The source ranks of the dispatch phase only.

    ``RoutedConservationReport.source_ranks`` spans both phases, and combine
    legitimately leaves the peer expert owners, so the single-owner claim is
    read off dispatch.
    """

    return tuple(
        sorted(
            {
                source
                for table in tables
                if table.phase == "dispatch"
                for source, _, _ in table.pair_payload_bytes
            }
        )
    )


def _report_for(tables, record, dims, world):
    from simllm.traffic import ROUTED_EVIDENCE_CAPTURED, routed_moe_conservation_report

    return routed_moe_conservation_report(
        tables,
        _ownership(record, dims),
        tuple(range(world)),
        evidence_mode=ROUTED_EVIDENCE_CAPTURED,
    )


def _observed_evidence_cell(dims) -> list[dict]:
    """E6: the semantic marker is named, and the guard runs on the plan anyway."""

    from dataclasses import replace

    from simllm.adapters.vllm import (
        VllmBatchSlice,
        build_granite_execution_observations,
    )
    from simllm.compute import GPU_ENVELOPES, HostInitiationModel, RooflineProvider
    from simllm.core import CollectiveWork, ExecutionObservations
    from simllm.traffic import (
        OBSERVED_NO_BYTE_EVIDENCE,
        lower_step_observations,
        observed_routed_byte_evidence,
    )

    record = _record(STEP_TOKENS["prefill"])
    ranks = tuple(range(8))
    observations, _ = build_granite_execution_observations(
        record,
        dims,
        ranks,
        (VllmBatchSlice(None, (_REQUEST_ID,), STEP_TOKENS["prefill"]),),
        RooflineProvider(),
        GPU_ENVELOPES["b100"],
        HostInitiationModel(initiation_delay_ps=0),
    )
    evidence = observed_routed_byte_evidence(observations)
    graph = lower_step_observations(
        record,
        dims,
        (0,),
        observations,
        ep_ranks=ranks,
        routed_supply=_supply(8),
    )
    lowered_routed = [
        operation
        for operation in graph.operations
        if isinstance(operation.work, CollectiveWork)
        and operation.work.collective == "all-to-allv"
    ]
    bound_bytes = all(
        bool(operation.work.pair_payload_bytes) for operation in lowered_routed
    )

    # a disagreeing observed table must still fail exactly as before
    corrupted = []
    already_corrupted = False
    for operation in observations.operations:
        work = operation.work
        if (
            isinstance(work, CollectiveWork)
            and work.collective == "all-to-allv"
            and not already_corrupted
            and operation.correlation.layer == 0
            and work.channel_hint == "dispatch"
        ):
            already_corrupted = True
            corrupted.append(
                replace(
                    operation,
                    work=replace(work, pair_payload_bytes=((0, 1, 12_345),)),
                )
            )
        else:
            corrupted.append(operation)
    rejected = False
    try:
        lower_step_observations(
            record,
            dims,
            (0,),
            ExecutionObservations(
                operations=tuple(corrupted),
                completion_operation_ids=observations.completion_operation_ids,
            ),
            ep_ranks=ranks,
            routed_supply=_supply(8),
        )
    except ValueError:
        rejected = True

    return [
        {
            "family": "E6",
            "cell": "granite-marker-at-w8",
            "observed_evidence_mode": evidence,
            "lowered_routed_sites": len(lowered_routed),
            "every_lowered_site_carries_plan_bytes": bound_bytes,
            "disagreeing_observed_table_rejected": rejected,
            "passed": (
                evidence == OBSERVED_NO_BYTE_EVIDENCE
                and len(lowered_routed) == NUM_LAYERS * 2
                and bound_bytes
                and rejected
            ),
        }
    ]


def production(args: argparse.Namespace) -> None:
    import json

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dims = _granite_dims()

    cells: dict[str, dict] = {}
    executed_steps: dict[str, int] = {}
    not_executed: list[dict] = []

    from simllm.core import RequestBookkeeper
    from simllm.preplay import (
        RequestArrival,
        join_preplay_arrivals,
        project_preplay_routing,
    )

    run = join_preplay_arrivals(
        (RequestArrival(request_id=_REQUEST_ID, arrived_at_ps=0),),
        _repository_root() / _FIXTURE,
        RequestBookkeeper(),
    )
    captured = project_preplay_routing(run).by_request_id(_REQUEST_ID)
    decode_tokens = len(captured.decode_tokens)
    executed_steps["prefill"] = STEP_TOKENS["prefill"]
    if decode_tokens >= STEP_TOKENS["decode"]:
        executed_steps["decode"] = STEP_TOKENS["decode"]
    else:
        not_executed.append(
            {
                "step": "decode",
                "reason": (
                    "the frozen capture carries 22 prefill tokens and "
                    f"{decode_tokens} decode tokens, so the frozen DECODE cell "
                    "cannot be built from it"
                ),
            }
        )
    # post-specified substitute for the unbuildable decode cell, labeled as such
    substitute_steps = {"prefill-chunk-1": 1}

    fatal: list[dict] = []
    scored: list[dict] = []

    for label, tokens in list(executed_steps.items()) + list(substitute_steps.items()):
        frozen_cell = label in STEP_TOKENS
        for world in EP_WORLDS:
            record = _record(tokens)
            arm_a = _arm_a_tables(record, dims, world)
            arm_b = _replicate_sources(arm_a, world)
            report_a = _report_for(arm_a, record, dims, world)
            report_b = _report_for(arm_b, record, dims, world)
            key = f"{label}/W{world}"
            dispatch_sources_a = _dispatch_sources(arm_a)
            dispatch_sources_b = _dispatch_sources(arm_b)
            cells[key] = {
                "frozen_cell": frozen_cell,
                "tokens": tokens,
                "arm_a": {
                    "source_ranks": list(report_a.source_ranks),
                    "dispatch_source_ranks": list(dispatch_sources_a),
                    "total_directed_bytes": report_a.total_directed_bytes,
                    "owner_egress_bytes": report_a.owner_egress_bytes,
                    "owner_ingress_bytes": report_a.owner_ingress_bytes,
                    "emitted_hops": report_a.emitted_hops,
                    "step_hop_bound": report_a.step_hop_bound,
                    "per_layer_hop_bound": report_a.per_layer_hop_bound,
                    "violations": list(report_a.violations),
                },
                "arm_b": {
                    "source_ranks": list(report_b.source_ranks),
                    "dispatch_source_ranks": list(dispatch_sources_b),
                    "total_directed_bytes": report_b.total_directed_bytes,
                    "emitted_hops": report_b.emitted_hops,
                    "violations": list(report_b.violations),
                },
            }
            fatal.append(
                {
                    "guard": "arm-a-conserves",
                    "cell": key,
                    "held": report_a.conserved
                    and dispatch_sources_a == (0,)
                    and report_a.owner_egress_bytes == report_a.owner_ingress_bytes,
                }
            )
            fatal.append(
                {
                    "guard": "replication-multiplier",
                    "cell": key,
                    "held": report_b.emitted_hops
                    == world * report_a.emitted_hops
                    and report_b.total_directed_bytes
                    == world * report_a.total_directed_bytes,
                }
            )
            if not frozen_cell:
                continue
            bound_fired = "step-hop-bound" in report_b.violations
            source_fired = "source-attribution" in report_b.violations
            scored.append(
                {
                    "family": "E3" if world == 8 else "E4",
                    "cell": key,
                    "expected_step_hop_bound_detection": EXPECTED_DETECTION[
                        (label, world, "step-hop-bound")
                    ],
                    "observed_step_hop_bound_detection": bound_fired,
                    "passed": bound_fired
                    == EXPECTED_DETECTION[(label, world, "step-hop-bound")],
                }
            )
            scored.append(
                {
                    "family": "E5",
                    "cell": key,
                    "expected_source_attribution_detection": EXPECTED_DETECTION[
                        (label, world, "source-attribution")
                    ],
                    "observed_source_attribution_detection": source_fired,
                    "passed": source_fired
                    == EXPECTED_DETECTION[(label, world, "source-attribution")],
                }
            )

    scored.extend(_observed_evidence_cell(dims))

    # physical sanity: the frozen floors and ceilings on arm A
    sanity = []
    for label in executed_steps:
        for world in EP_WORLDS:
            key = f"{label}/W{world}"
            hops = cells[key]["arm_a"]["emitted_hops"]
            floor = EXPECTED_ARM_A_HOP_FLOOR[(label, world)]
            ceiling = EXPECTED_ARM_A_HOP_CEILING[(label, world)]
            sanity.append(
                {
                    "cell": key,
                    "hops": hops,
                    "floor": floor,
                    "ceiling": ceiling,
                    "inside": floor <= hops <= ceiling,
                }
            )
    for label in executed_steps:
        low = cells[f"{label}/W2"]["arm_a"]["total_directed_bytes"]
        high = cells[f"{label}/W8"]["arm_a"]["total_directed_bytes"]
        sanity.append(
            {
                "cell": f"{label}/scaling",
                "bytes_w2": low,
                "bytes_w8": high,
                "inside": high >= low,
            }
        )

    void = [finding for finding in fatal if not finding["held"]]
    passed = sum(1 for row in scored if row["passed"])
    report = {
        "study": "routed_byte_conservation_v1",
        "task": "VLLM-24",
        "provenance": _observed_provenance(),
        "cells": cells,
        "fatal_guards": fatal,
        "void": bool(void),
        "not_executed": not_executed,
        "physical_sanity": sanity,
        "scored_rows": scored,
        "scored": None if void else f"{passed}/{len(scored)}",
        "frozen_scored_denominator": EXPECTED_SCORED_INSTANCES,
    }
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    if void:
        print("VOID: fatal guard violated")
        for finding in void:
            print(f"  {finding['guard']} at {finding['cell']}")
    else:
        print(f"scored {passed}/{len(scored)} of a frozen {EXPECTED_SCORED_INSTANCES}")
    for row in scored:
        if not row["passed"]:
            print(f"  MISS {row['family']} {row['cell']}: {row}")
    for row in sanity:
        if not row["inside"]:
            print(f"  PHYSICAL SANITY OUTSIDE BOUNDS: {row}")
    for entry in not_executed:
        print(f"  NOT EXECUTED {entry['step']}: {entry['reason']}")
    print(f"report written to {out / 'report.json'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="routed_byte_conservation_v1")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if args.check_only:
        check_only(args)
        return
    _check_frozen_registry()
    production(args)


if __name__ == "__main__":
    main()
