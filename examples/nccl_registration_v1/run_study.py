"""Run the frozen NCCL/RCCL registration study.

Every acceptance clause, bound and relation this script evaluates is frozen in
``expectations.md`` and ``expectations.json``, both committed before the
registration mechanism, the live wiring and this runner existed.

The chain is the repository's own: a ``StepRecord`` into ``HtsimStepSink``, an
authoritative ``ExecutionGraph``, the locality projection, ``StepResult``,
``attribute_step_detail`` and ``HtsimRequestMetricReducer`` into per-request
TTFT and TPOT. The ``mixed-tp4`` cell additionally runs the real ``htsim_rnic``
binary for every fabric phase; set ``SIMLLM_HTSIM_RNIC`` or build the backend
where the README documents it.

    python examples/nccl_registration_v1/run_study.py --check
    python examples/nccl_registration_v1/run_study.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

STUDY_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = STUDY_DIR.parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from simllm.backends import (
    HtsimRequestMetricReducer,
    HtsimStepSink,
    HtsimStepSinkConfig,
    attribute_step_detail,
)
from simllm.compute import GPU_ENVELOPES, ModelDims, RooflineProvider
from simllm.core import RequestPhase, ScheduledRequest, StepRecord
from simllm.placement import PlacementManifest, RankPlacement
from simllm.traffic import (
    DECLARED_CHANNEL_REGISTRATION_COST_PS,
    DECLARED_NCCL_CHANNEL_REGISTRATION_COST,
    CollectiveRegistrationModel,
)

EXPECTATIONS = json.loads((STUDY_DIR / "expectations.json").read_text(encoding="utf-8"))
COST_PS = DECLARED_CHANNEL_REGISTRATION_COST_PS
REGISTRATION_MODEL_ID = "nccl-channel-registration-v1"

TWO_CHANNEL_MODEL = CollectiveRegistrationModel(
    model_id="nccl-channel-registration-two-channel-v1",
    cost=DECLARED_NCCL_CHANNEL_REGISTRATION_COST,
    channels_per_communicator=2,
)

#: arm name, registration selection, step index a rebuild is issued before
ARMS: dict[str, tuple[Any, int | None]] = {
    "off": (None, None),
    "on": (REGISTRATION_MODEL_ID, None),
    "on-2ch": (TWO_CHANNEL_MODEL, None),
    "on-rebuild": (REGISTRATION_MODEL_ID, 2),
}


def _dims(spec: dict[str, int]) -> ModelDims:
    return ModelDims(
        num_layers=spec["num_layers"],
        hidden_size=spec["hidden_size"],
        intermediate_size=spec["intermediate_size"],
        num_heads=spec["num_heads"],
        num_kv_heads=spec["num_kv_heads"],
        head_size=spec["head_size"],
        vocab_size=spec["vocab_size"],
        dtype_bytes=spec["dtype_bytes"],
    )


def _manifest(hosts: list[str]) -> PlacementManifest:
    counts: dict[str, int] = {}
    ranks = []
    for global_rank, host in enumerate(hosts):
        local_rank = counts.get(host, 0)
        counts[host] = local_rank + 1
        ranks.append(
            RankPlacement(global_rank=global_rank, hostname=host, local_rank=local_rank)
        )
    return PlacementManifest(ranks=ranks)


def _records(cell: dict[str, Any]) -> list[StepRecord]:
    prompt_tokens = cell["prompt_tokens"]
    records = [
        StepRecord(
            step_index=0,
            virtual_time_ps=0,
            scheduled=[
                ScheduledRequest(
                    "r0",
                    RequestPhase.PREFILL,
                    prompt_tokens,
                    context_length=prompt_tokens,
                )
            ],
            num_sampled=1,
        )
    ]
    for step_index in range(1, cell["steps"]):
        records.append(
            StepRecord(
                step_index=step_index,
                virtual_time_ps=0,
                scheduled=[
                    ScheduledRequest(
                        "r0",
                        RequestPhase.DECODE,
                        1,
                        context_length=prompt_tokens + step_index,
                    )
                ],
                num_sampled=1,
            )
        )
    return records


def _goal_digests(workdir: Path) -> list[str]:
    return sorted(
        hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(workdir.rglob("*.goal"))
    )


def _run_arm(
    cell_name: str,
    cell: dict[str, Any],
    arm: str,
    workdir: Path,
) -> dict[str, Any]:
    """Replay one cell under one arm and return everything it observed."""

    selection, rebuild_before = ARMS[arm]
    kwargs: dict[str, Any] = {}
    if selection is not None:
        kwargs["collective_registration"] = selection
    sink = HtsimStepSink(
        HtsimStepSinkConfig(
            profile=cell["profile"],
            tp_ranks=tuple(cell["tp_ranks"]),
            dims=_dims(cell["dims"]),
            workdir=workdir / f"{cell_name}.{arm}",
            placement_manifest=_manifest(list(cell["hosts"])),
            provider=RooflineProvider(efficiency=cell["roofline_efficiency"]),
            gpu=GPU_ENVELOPES[cell["gpu"]],
            **kwargs,
        )
    )
    reducer = HtsimRequestMetricReducer({"r0": 0})
    virtual_time_ps = 0
    steps: list[dict[str, Any]] = []
    for record in _records(cell):
        if rebuild_before is not None and record.step_index == rebuild_before:
            sink.rebuild_collective_communicators()
        record.virtual_time_ps = virtual_time_ps
        result = sink(record)
        if result is None:
            raise RuntimeError(f"{cell_name}/{arm}: step {record.step_index} was not simulated")
        locality = sink.locality_outcomes[record.step_index]
        attribution = attribute_step_detail(result, locality)
        registration = (
            sink.collective_registration_outcomes[record.step_index]
            if sink.collective_registration_outcomes
            else None
        )
        reducer.consume(record, result, locality)
        steps.append(
            {
                "step_index": record.step_index,
                "makespan_ps": result.step_latency_ps,
                "completed_at_ps": result.completed_at_ps,
                "artifact_count": locality.artifact_count,
                "backend_runs": locality.backend_runs,
                "fabric_directed_bytes": locality.fabric_directed_bytes,
                "nvlink_directed_bytes": locality.nvlink_directed_bytes,
                "registration_phase_cost_ps": list(
                    locality.registration_phase_cost_ps
                ),
                "charged_registration_ps": (
                    registration.charged_ps if registration is not None else 0
                ),
                "registration_event_count": (
                    len(registration.events) if registration is not None else 0
                ),
                "media": {
                    "kernel_ps": attribution.media.kernel_ps,
                    "nvlink_ps": attribution.media.nvlink_ps,
                    "fabric_ps": attribution.media.fabric_ps,
                    "co_critical_ps": attribution.media.co_critical_ps,
                    "collective_base_ps": attribution.media.collective_base_ps,
                    "collective_registration_ps": (
                        attribution.media.collective_registration_ps
                    ),
                    "total_ps": attribution.media.total_ps,
                },
            }
        )
        virtual_time_ps = result.completed_at_ps
    (row,) = reducer.totals()
    return {
        "arm": arm,
        "steps": steps,
        "ttft_ps": row.ttft_ps,
        "tpot_ps": float(row.tpot_ps) if row.tpot_ps is not None else None,
        "ttft_registration_ps": row.ttft_media.collective_registration_ps,
        "decode_registration_ps": row.decode_media.collective_registration_ps,
        "decode_span_ps": row.last_token_at_ps - row.first_token_at_ps,
        "goal_sha256": _goal_digests(sink.config.workdir),
        "ledger_charged_ps": sink.collective_registration_ledger.charged_ps,
        "registered_identities": [
            identity.to_json()
            for identity in sink.collective_registration_ledger.registered_identities
        ],
    }


def _default_construction_arm(
    cell_name: str,
    cell: dict[str, Any],
    workdir: Path,
) -> dict[str, Any]:
    """Replay the cell with the registration parameter entirely absent."""

    sink = HtsimStepSink(
        HtsimStepSinkConfig(
            profile=cell["profile"],
            tp_ranks=tuple(cell["tp_ranks"]),
            dims=_dims(cell["dims"]),
            workdir=workdir / f"{cell_name}.default",
            placement_manifest=_manifest(list(cell["hosts"])),
            provider=RooflineProvider(efficiency=cell["roofline_efficiency"]),
            gpu=GPU_ENVELOPES[cell["gpu"]],
        )
    )
    reducer = HtsimRequestMetricReducer({"r0": 0})
    virtual_time_ps = 0
    makespans = []
    for record in _records(cell):
        record.virtual_time_ps = virtual_time_ps
        result = sink(record)
        assert result is not None
        reducer.consume(record, result, sink.locality_outcomes[record.step_index])
        makespans.append(result.step_latency_ps)
        virtual_time_ps = result.completed_at_ps
    (row,) = reducer.totals()
    return {
        "makespans_ps": makespans,
        "ttft_ps": row.ttft_ps,
        "goal_sha256": _goal_digests(sink.config.workdir),
        "registration_outcomes": len(sink.collective_registration_outcomes),
        "registration_phase_projection_empty": all(
            outcome.registration_phase_cost_ps == ()
            for outcome in sink.locality_outcomes
        ),
    }


def _guards(results: dict[str, Any]) -> list[dict[str, Any]]:
    """Evaluate the frozen fatal guards. Any failure voids the run."""

    guards: list[dict[str, Any]] = []

    def record(guard_id: str, claim: str, held: bool, detail: str = "") -> None:
        guards.append(
            {"id": guard_id, "claim": claim, "held": bool(held), "detail": detail}
        )

    g1 = True
    g1_detail = []
    for cell_name, cell in results["cells"].items():
        default = cell["default_construction"]
        off = cell["arms"]["off"]
        same = (
            default["makespans_ps"] == [step["makespan_ps"] for step in off["steps"]]
            and default["ttft_ps"] == off["ttft_ps"]
            and default["goal_sha256"] == off["goal_sha256"]
            and default["registration_outcomes"] == 0
            and default["registration_phase_projection_empty"]
        )
        g1 = g1 and same
        g1_detail.append(f"{cell_name}={'equal' if same else 'DIFFERENT'}")
    record(
        "G1",
        "the default-constructed arm and the explicitly disabled arm agree",
        g1,
        ", ".join(g1_detail),
    )

    g2 = True
    g2_detail = []
    for cell_name, cell in results["cells"].items():
        off_digests = cell["arms"]["off"]["goal_sha256"]
        for arm, payload in cell["arms"].items():
            if arm == "off":
                continue
            same = payload["goal_sha256"] == off_digests
            g2 = g2 and same
            g2_detail.append(f"{cell_name}/{arm}={'equal' if same else 'DIFFERENT'}")
    record(
        "G2",
        "every GOAL artifact SHA-256 is identical between the off and on arms",
        g2,
        f"{len(results['cells']['mixed-tp4']['arms']['off']['goal_sha256'])} "
        "artifacts in mixed-tp4; " + ", ".join(g2_detail),
    )

    g3 = all(
        cell["arms"]["off"]["ledger_charged_ps"] == 0
        and all(
            step["charged_registration_ps"] == 0
            and step["registration_phase_cost_ps"] == []
            and step["media"]["collective_registration_ps"] == 0
            for step in cell["arms"]["off"]["steps"]
        )
        for cell in results["cells"].values()
    )
    record(
        "G3",
        "the off arm charges nothing and publishes no registration term",
        g3,
    )

    g4 = True
    for cell in results["cells"].values():
        for payload in cell["arms"].values():
            for step in payload["steps"]:
                g4 = g4 and step["media"]["total_ps"] == step["makespan_ps"]
    record("G4", "every step's medium partition conserves its makespan", g4)

    g5 = results["fail_closed"]["calibrated_request_raises"] and results[
        "fail_closed"
    ]["unknown_selector_raises"]
    record(
        "G5",
        "a calibrated request against a declared cost fails closed",
        g5,
        results["fail_closed"]["calibrated_message"],
    )

    g6 = True
    g6_detail = []
    for cell_name, cell in results["cells"].items():
        for arm, payload in cell["arms"].items():
            if arm == "off":
                continue
            for step in payload["steps"]:
                charges = [
                    value for value in step["registration_phase_cost_ps"] if value
                ]
                g6 = (
                    g6
                    and sum(charges) == step["charged_registration_ps"]
                    and len(charges) * COST_PS * cell["channels"][arm]
                    == step["charged_registration_ps"]
                )
            g6_detail.append(f"{cell_name}/{arm}")
    record(
        "G6",
        "no collective is charged registration on more than one artifact",
        g6,
        ", ".join(g6_detail),
    )

    record(
        "G7",
        "the mirrored nccl_stack event stream is unchanged when no registration "
        "is requested",
        results["seam"]["ungated_stream_identical"],
        f"{results['seam']['ungated_event_count']} events, "
        f"{results['seam']['gated_registration_events']} registration events "
        "when the gate is requested",
    )

    record(
        "G8",
        "the results identity block equals the freeze",
        results["identity"] == results["frozen_identity"],
    )

    record(
        "G9",
        "ledger construction facts hold",
        all(results["ledger_facts"].values()),
        ", ".join(
            f"{name}={'ok' if held else 'FAILED'}"
            for name, held in results["ledger_facts"].items()
        ),
    )
    return guards


def _scored(results: dict[str, Any]) -> dict[str, Any]:
    """Evaluate the frozen exact-oracle rows and behavioral families."""

    local = results["cells"]["local-tp2"]
    mixed = results["cells"]["mixed-tp4"]
    local_identities = local["identities_one_channel"]
    mixed_identities = mixed["identities_one_channel"]

    def makespans(cell: dict[str, Any], arm: str) -> list[int]:
        return [step["makespan_ps"] for step in cell["arms"][arm]["steps"]]

    rows: list[dict[str, Any]] = []

    observed = local["arms"]["on"]["ttft_ps"] - local["arms"]["off"]["ttft_ps"]
    expected = local_identities * COST_PS
    rows.append(
        {
            "id": "O1",
            "claim": "TTFT_on - TTFT_off equals identities times the declared cost",
            "expected_ps": expected,
            "observed_ps": observed,
            "passed": observed == expected,
        }
    )

    off_tail = makespans(local, "off")[1:]
    on_tail = makespans(local, "on")[1:]
    rows.append(
        {
            "id": "O2",
            "claim": "decode-step makespans are unchanged between on and off",
            "expected_ps": 0,
            "observed_ps": [
                on - off for on, off in zip(on_tail, off_tail, strict=True)
            ],
            "passed": on_tail == off_tail,
        }
    )

    observed = local["arms"]["on-2ch"]["ttft_ps"] - local["arms"]["off"]["ttft_ps"]
    expected = 2 * local_identities * COST_PS
    rows.append(
        {
            "id": "O3",
            "claim": "two channels move TTFT by exactly twice the one-channel amount",
            "expected_ps": expected,
            "observed_ps": observed,
            "passed": observed == expected,
        }
    )

    rebuild_step = makespans(local, "on-rebuild")[2] - makespans(local, "off")[2]
    expected = local_identities * COST_PS
    ttft_matches = (
        local["arms"]["on-rebuild"]["ttft_ps"] == local["arms"]["on"]["ttft_ps"]
    )
    rows.append(
        {
            "id": "O4",
            "claim": "a rebuild re-pays exactly once and leaves TTFT alone",
            "expected_ps": expected,
            "observed_ps": rebuild_step,
            "ttft_matches_plain_on_arm": ttft_matches,
            "passed": rebuild_step == expected and ttft_matches,
        }
    )

    observed = mixed["arms"]["on"]["ttft_ps"] - mixed["arms"]["off"]["ttft_ps"]
    expected = mixed_identities * COST_PS
    rows.append(
        {
            "id": "O5",
            "claim": "a phase-split collective on a real backend still pays once",
            "expected_ps": expected,
            "observed_ps": observed,
            "passed": observed == expected,
        }
    )

    o6 = True
    o6_detail = []
    for cell in (local, mixed):
        for arm, payload in cell["arms"].items():
            if arm == "off":
                continue
            for step in payload["steps"]:
                matched = (
                    step["media"]["collective_registration_ps"]
                    == step["charged_registration_ps"]
                )
                o6 = o6 and matched
                if step["step_index"] > 0 and arm != "on-rebuild":
                    o6 = o6 and step["charged_registration_ps"] == 0
            o6_detail.append(f"{arm}:{[s['charged_registration_ps'] for s in payload['steps']]}")
    rows.append(
        {
            "id": "O6",
            "claim": "the published registration component equals the ledger charge",
            "observed": o6_detail,
            "passed": o6,
        }
    )

    families: list[dict[str, Any]] = []

    b1_instances = [
        {
            "identities": mixed_identities,
            "charged_ps": mixed["arms"]["on"]["ledger_charged_ps"],
            "expected_ps": mixed_identities * COST_PS,
        },
        {
            "identities": local_identities,
            "charged_ps": local["arms"]["on"]["ledger_charged_ps"],
            "expected_ps": local_identities * COST_PS,
        },
        {
            "identities": 2 * local_identities,
            "charged_ps": local["arms"]["on-2ch"]["ledger_charged_ps"],
            "expected_ps": 2 * local_identities * COST_PS,
        },
    ]
    families.append(
        {
            "id": "B1",
            "claim": "charged registration is exactly identities times the cost",
            "instances": b1_instances,
            "passed": all(
                instance["charged_ps"] == instance["expected_ps"]
                for instance in b1_instances
            ),
        }
    )

    b2_instances = []
    for name, cell in (("local-tp2", local), ("mixed-tp4", mixed)):
        first_off = makespans(cell, "off")[0]
        first_on = makespans(cell, "on")[0]
        charged = cell["arms"]["on"]["steps"][0]["charged_registration_ps"]
        b2_instances.append(
            {
                "cell": name,
                "off_ps": first_off,
                "on_ps": first_on,
                "charged_ps": charged,
                "additive": first_on == first_off + charged,
            }
        )
    families.append(
        {
            "id": "B2",
            "claim": "the charge is additive and never absorbed by the max term",
            "instances": b2_instances,
            "passed": all(instance["additive"] for instance in b2_instances),
        }
    )

    b3_instances = []
    for name, cell in (("local-tp2", local), ("mixed-tp4", mixed)):
        off_tail = makespans(cell, "off")[1:]
        on_tail = makespans(cell, "on")[1:]
        b3_instances.append(
            {
                "cell": name,
                "off_ps": off_tail,
                "on_ps": on_tail,
                "unchanged": off_tail == on_tail
                and cell["arms"]["off"]["goal_sha256"]
                == cell["arms"]["on"]["goal_sha256"],
            }
        )
    families.append(
        {
            "id": "B3",
            "claim": "steps after the registering step are untouched",
            "instances": b3_instances,
            "passed": all(instance["unchanged"] for instance in b3_instances),
        }
    )

    return {
        "exact_oracle_rows": {
            "denominator": len(rows),
            "passed": sum(1 for row in rows if row["passed"]),
            "rows": rows,
        },
        "behavioral_families": {
            "denominator_families": len(families),
            "denominator_instances": sum(
                len(family["instances"]) for family in families
            ),
            "passed_families": sum(1 for family in families if family["passed"]),
            "families": families,
        },
    }


def _ledger_facts() -> dict[str, bool]:
    """Re-derive the by-construction ledger facts as guards, not as scores."""

    from simllm.core import CollectiveWork
    from simllm.traffic import (
        CollectiveRegistrationLedger,
        collective_communicator_id,
    )

    def work(ranks: tuple[int, ...]) -> CollectiveWork:
        return CollectiveWork(
            collective="all-reduce",
            ranks=ranks,
            payload_bytes=1024,
            algorithm_hint="ring",
        )

    disabled = CollectiveRegistrationLedger()
    facts = {
        "disabled_charges_zero": disabled.charge_collective(
            work((0, 1)), "step-0:layer-0:tp-attention"
        )
        == 0
        and disabled.events == (),
    }

    ledger = CollectiveRegistrationLedger(
        model=CollectiveRegistrationModel(
            model_id="guard",
            cost=DECLARED_NCCL_CHANNEL_REGISTRATION_COST,
        )
    )
    facts["first_charge_pays"] = (
        ledger.charge_collective(work((0, 1)), "step-0:layer-0:tp-attention")
        == COST_PS
    )
    facts["second_charge_is_zero"] = (
        ledger.charge_collective(work((0, 1)), "step-1:layer-0:tp-attention") == 0
    )
    facts["new_buffer_re_pays"] = (
        ledger.charge_collective(work((0, 1)), "step-1:layer-0:tp-mlp") == COST_PS
    )
    facts["new_peer_re_pays"] = (
        ledger.charge_collective(work((0, 1, 2, 3)), "step-1:layer-0:tp-attention")
        == COST_PS
    )
    ledger.rebuild_communicator(collective_communicator_id(work((0, 1))))
    facts["rebuild_re_pays"] = (
        ledger.charge_collective(work((0, 1)), "step-2:layer-0:tp-attention")
        == COST_PS
    )
    return facts


def _seam_evidence() -> dict[str, Any]:
    """Check the mirrored seam's off path and its gated registration events."""

    from simllm.compute import (
        NcclRoute,
        NcclStack,
        NcclStackConfig,
        nccl_stack_events_to_json,
        ncclAllReduce,
        ncclCommInitRank,
        ncclNetRegMr,
    )
    from simllm.core import VirtualClock

    config = NcclStackConfig(
        channel_count=1,
        chunk_bytes=1_024,
        fifo_slots_per_channel=2,
    )

    def ungated() -> list[dict[str, Any]]:
        stack = NcclStack(clock=VirtualClock(), config=config)
        communicator = ncclCommInitRank(
            stack,
            nranks=4,
            communicator_id="ring-4",
            rank=0,
        )
        ncclAllReduce(
            communicator,
            payload_bytes=16_384,
            operation_id="op",
            route=NcclRoute.INTER_NODE,
        )
        return nccl_stack_events_to_json(stack.events)

    def explicitly_ungated() -> list[dict[str, Any]]:
        stack = NcclStack(clock=VirtualClock(), config=config)
        communicator = ncclCommInitRank(
            stack,
            nranks=4,
            communicator_id="ring-4",
            rank=0,
            require_buffer_registration=False,
        )
        ncclAllReduce(
            communicator,
            payload_bytes=16_384,
            operation_id="op",
            route=NcclRoute.INTER_NODE,
        )
        return nccl_stack_events_to_json(stack.events)

    gated_stack = NcclStack(clock=VirtualClock(start_ps=17), config=config)
    gated = ncclCommInitRank(
        gated_stack,
        nranks=4,
        communicator_id="ring-4",
        rank=0,
        require_buffer_registration=True,
    )
    registration = ncclNetRegMr(
        gated,
        buffer_id="layer-0:tp-attention",
        buffer_bytes=16_384,
        declared_cost_ps=COST_PS,
    )
    gate_refused = False
    try:
        ncclAllReduce(
            gated,
            payload_bytes=16_384,
            operation_id="op",
            route=NcclRoute.INTER_NODE,
            buffer_id="layer-1:tp-mlp",
        )
    except ValueError:
        gate_refused = True

    default_events = ungated()
    return {
        "ungated_stream_identical": default_events == explicitly_ungated(),
        "ungated_event_count": len(default_events),
        "gated_registration_events": len(registration.events),
        "gated_declared_total_ps": registration.total_declared_cost_ps,
        "clock_unmoved": gated_stack.clock.now_ps == 17,
        "unregistered_buffer_refused": gate_refused,
    }


def _fail_closed_evidence() -> dict[str, Any]:
    from simllm.traffic import resolve_collective_registration

    message = ""
    calibrated_raises = False
    try:
        DECLARED_NCCL_CHANNEL_REGISTRATION_COST.calibrated_cost_ps()
    except ValueError as error:
        calibrated_raises = True
        message = str(error)

    unknown_raises = False
    try:
        resolve_collective_registration("no-such-model")
    except ValueError:
        unknown_raises = True

    return {
        "calibrated_request_raises": calibrated_raises,
        "calibrated_message": message,
        "unknown_selector_raises": unknown_raises,
    }


def run(workdir: Path) -> dict[str, Any]:
    cells: dict[str, Any] = {}
    for cell_name, cell in EXPECTATIONS["cells"].items():
        arms: dict[str, Any] = {}
        channels: dict[str, int] = {}
        for arm in cell["arms"]:
            arms[arm] = _run_arm(cell_name, cell, arm, workdir)
            selection = ARMS[arm][0]
            channels[arm] = (
                selection.channels_per_communicator
                if isinstance(selection, CollectiveRegistrationModel)
                else 1
            )
        cells[cell_name] = {
            "identities_one_channel": cell["identities_one_channel"],
            "collectives_per_step": cell["collectives_per_step"],
            "channels": channels,
            "arms": arms,
            "default_construction": _default_construction_arm(
                cell_name, cell, workdir
            ),
        }

    identity = {
        "registration_cost_ps": COST_PS,
        "evidence_class": DECLARED_NCCL_CHANNEL_REGISTRATION_COST.evidence_class,
        "cells": {
            name: {
                "dims": cell["dims"],
                "tp_ranks": cell["tp_ranks"],
                "hosts": cell["hosts"],
                "steps": cell["steps"],
                "arms": cell["arms"],
                "identities_one_channel": cell["identities_one_channel"],
            }
            for name, cell in EXPECTATIONS["cells"].items()
        },
    }
    frozen_identity = {
        "registration_cost_ps": EXPECTATIONS["declared"]["registration_cost_ps"],
        "evidence_class": EXPECTATIONS["declared"]["evidence_class"],
        "cells": {
            name: {
                "dims": cell["dims"],
                "tp_ranks": cell["tp_ranks"],
                "hosts": cell["hosts"],
                "steps": cell["steps"],
                "arms": cell["arms"],
                "identities_one_channel": cell["identities_one_channel"],
            }
            for name, cell in EXPECTATIONS["cells"].items()
        },
    }

    results: dict[str, Any] = {
        "study": "nccl_registration_v1",
        "freeze": "examples/nccl_registration_v1/expectations.md",
        "identity": identity,
        "frozen_identity": frozen_identity,
        "cells": cells,
        "ledger_facts": _ledger_facts(),
        "seam": _seam_evidence(),
        "fail_closed": _fail_closed_evidence(),
    }
    results["fatal_guards"] = _guards(results)
    results["scored"] = _scored(results)
    results["verdict"] = (
        "void"
        if any(not guard["held"] for guard in results["fatal_guards"])
        else "interpretable"
    )
    return results


def check_only() -> None:
    """Validate the frozen inputs without running anything."""

    for name, cell in EXPECTATIONS["cells"].items():
        expected = cell["dims"]["num_layers"] * 2
        if cell["collectives_per_step"] != expected:
            raise SystemExit(f"{name}: frozen collective count disagrees with its dims")
        if cell["identities_one_channel"] != expected:
            raise SystemExit(f"{name}: frozen identity count disagrees with its dims")
        for arm in cell["arms"]:
            if arm not in ARMS:
                raise SystemExit(f"{name}: unknown arm {arm!r}")
    if EXPECTATIONS["declared"]["registration_cost_ps"] != COST_PS:
        raise SystemExit("the frozen cost disagrees with the shipped declared cost")
    print("frozen inputs are consistent with the shipped mechanism")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="validate the frozen inputs and exit",
    )
    parser.add_argument(
        "--workdir",
        type=Path,
        default=None,
        help="directory for GOAL artifacts and backend outputs",
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=STUDY_DIR / "results.json",
        help="where to write the results document",
    )
    arguments = parser.parse_args()
    if arguments.check:
        check_only()
        return
    if arguments.workdir is None:
        raise SystemExit(
            "--workdir is required: bulk GOAL and backend outputs must not land "
            "in the repository, so name a directory on a data volume"
        )
    arguments.workdir.mkdir(parents=True, exist_ok=True)
    results = run(arguments.workdir)
    arguments.results.write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"verdict: {results['verdict']}")
    for guard in results["fatal_guards"]:
        print(f"  {guard['id']}: {'held' if guard['held'] else 'VIOLATED'}")
    oracle = results["scored"]["exact_oracle_rows"]
    families = results["scored"]["behavioral_families"]
    print(f"  exact-oracle rows: {oracle['passed']} of {oracle['denominator']}")
    print(
        "  behavioral families: "
        f"{families['passed_families']} of {families['denominator_families']} "
        f"over {families['denominator_instances']} instances"
    )
    print(f"wrote {arguments.results}")


if __name__ == "__main__":
    main()
