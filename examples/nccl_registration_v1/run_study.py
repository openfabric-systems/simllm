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


#: Post-specified, added in the correction round and named as such everywhere.
#: The frozen cells never put a per-artifact fabric term above the 20 us
#: charge, so B2's counterfactual (folding the charge inside the existing
#: max(local, fabric) composition) stays partly visible there. This cell is
#: mixed-tp4 at a 10 Gbit/s link, where one 32,768-byte flow serializes for
#: 26.2 us on top of the 2.000 us propagation reference, so a folded charge
#: would be swallowed completely and the observed delta would be zero.
POST_SPECIFIED_CELLS: dict[str, Any] = {
    "slow-fabric-tp4": {
        "dims": EXPECTATIONS["cells"]["mixed-tp4"]["dims"],
        "tp_ranks": EXPECTATIONS["cells"]["mixed-tp4"]["tp_ranks"],
        "hosts": EXPECTATIONS["cells"]["mixed-tp4"]["hosts"],
        "profile": "rnic-nn-fluid",
        "roofline_efficiency": 0.7,
        "gpu": "b100",
        "prompt_tokens": 64,
        "steps": 2,
        "collectives_per_step": 4,
        "identities_one_channel": 4,
        "linkspeed_bps": 10_000_000_000,
        "arms": ["off", "on"],
    }
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
    #: The off arm names the selection explicitly. Omitting it would make the
    #: arm identical to the feature-absent construction, which would leave G1
    #: comparing one construction with itself.
    sink = HtsimStepSink(
        HtsimStepSinkConfig(
            profile=cell["profile"],
            tp_ranks=tuple(cell["tp_ranks"]),
            dims=_dims(cell["dims"]),
            workdir=workdir / f"{cell_name}.{arm}",
            placement_manifest=_manifest(list(cell["hosts"])),
            provider=RooflineProvider(efficiency=cell["roofline_efficiency"]),
            gpu=GPU_ENVELOPES[cell["gpu"]],
            linkspeed_bps=cell.get("linkspeed_bps", 400_000_000_000),
            collective_registration=selection,
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
                #: the exact per-artifact fields the frozen G3 names
                "fabric_phase_service_ps": list(locality.fabric_phase_service_ps),
                "local_phase_service_ps": list(locality.local_phase_service_ps),
                "base_phase_latency_ps": list(locality.base_phase_latency_ps),
                "composed_phase_service_ps": list(
                    locality.composed_phase_service_ps
                ),
                "local_phase_medium": list(locality.local_phase_medium),
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
    decode_span_ps = row.last_token_at_ps - row.first_token_at_ps
    return {
        "arm": arm,
        "steps": steps,
        "ttft_ps": row.ttft_ps,
        "tpot_ps": float(row.tpot_ps) if row.tpot_ps is not None else None,
        "ttft_registration_ps": row.ttft_media.collective_registration_ps,
        "decode_registration_ps": row.decode_media.collective_registration_ps,
        "decode_span_ps": decode_span_ps,
        #: the request half of the frozen G4, which the step half does not cover
        "ttft_partition_conserves": (
            row.ttft_media.total_ps == row.ttft_ps
            and row.ttft_attribution.total_ps == row.ttft_ps
        ),
        "decode_partition_conserves": (
            row.decode_media.total_ps == decode_span_ps
            and row.decode_attribution.total_ps == decode_span_ps
        ),
        "goal_sha256": _goal_digests(sink.config.workdir),
        "executed_geometry": _executed_geometry(sink, len(steps)),
        "ledger_charged_ps": sink.collective_registration_ledger.charged_ps,
        "registered_identities": [
            identity.to_json()
            for identity in sink.collective_registration_ledger.registered_identities
        ],
    }


def _executed_geometry(sink: HtsimStepSink, executed_steps: int) -> dict[str, Any]:
    """Read the geometry back off the sink that actually ran.

    G8 compares the freeze against what executed. Deriving both sides from the
    freeze would compare the freeze with itself, so every field here comes from
    the constructed configuration rather than from ``expectations.json``.
    """

    config = sink.config
    dims = config.dims
    manifest = config.placement_manifest
    return {
        "dims": {
            "num_layers": dims.num_layers,
            "hidden_size": dims.hidden_size,
            "intermediate_size": dims.intermediate_size,
            "num_heads": dims.num_heads,
            "num_kv_heads": dims.num_kv_heads,
            "head_size": dims.head_size,
            "vocab_size": dims.vocab_size,
            "dtype_bytes": dims.dtype_bytes,
        },
        "tp_ranks": list(config.tp_ranks),
        "hosts": [placement.hostname for placement in manifest.ranks],
        "profile": config.profile,
        "steps": executed_steps,
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
    steps: list[dict[str, Any]] = []
    for record in _records(cell):
        record.virtual_time_ps = virtual_time_ps
        result = sink(record)
        assert result is not None
        locality = sink.locality_outcomes[record.step_index]
        attribution = attribute_step_detail(result, locality)
        reducer.consume(record, result, locality)
        makespans.append(result.step_latency_ps)
        steps.append(
            {
                "step_index": record.step_index,
                "makespan_ps": result.step_latency_ps,
                "completed_at_ps": result.completed_at_ps,
                "fabric_phase_service_ps": list(locality.fabric_phase_service_ps),
                "local_phase_service_ps": list(locality.local_phase_service_ps),
                "base_phase_latency_ps": list(locality.base_phase_latency_ps),
                "composed_phase_service_ps": list(
                    locality.composed_phase_service_ps
                ),
                "local_phase_medium": list(locality.local_phase_medium),
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
        "makespans_ps": makespans,
        "ttft_ps": row.ttft_ps,
        "steps": steps,
        "goal_sha256": _goal_digests(sink.config.workdir),
        "executed_geometry": _executed_geometry(sink, len(steps)),
        "registration_outcomes": len(sink.collective_registration_outcomes),
        "registration_phase_projection_empty": all(
            outcome.registration_phase_cost_ps == ()
            for outcome in sink.locality_outcomes
        ),
    }


FROZEN_GUARD_CLAIMS = {
    guard["id"]: guard["claim"] for guard in EXPECTATIONS["fatal_guards"]
}

#: the exact per-artifact fields the frozen G3 names, plus the composed value
#: and the medium projection they have to agree on for the comparison to mean
#: anything
G3_STEP_FIELDS = (
    "makespan_ps",
    "completed_at_ps",
    "fabric_phase_service_ps",
    "local_phase_service_ps",
    "base_phase_latency_ps",
    "composed_phase_service_ps",
    "local_phase_medium",
    "media",
)


def _guards(results: dict[str, Any]) -> list[dict[str, Any]]:
    """Evaluate the frozen fatal guards. Any failure voids the run.

    Every entry carries the frozen claim string verbatim and a separate
    ``evaluated`` sentence saying what this runner actually checked. When the
    two differ the entry is marked ``partially_evaluated`` and names what
    covers the rest, so a weaker check can never be published under a frozen
    guard identity.
    """

    guards: list[dict[str, Any]] = []

    def record(
        guard_id: str,
        held: bool,
        *,
        evaluated: str,
        detail: str = "",
        partially_evaluated: bool = False,
        remainder_covered_by: str = "",
    ) -> None:
        guards.append(
            {
                "id": guard_id,
                "claim": FROZEN_GUARD_CLAIMS[guard_id],
                "evaluated": evaluated,
                "held": bool(held),
                "partially_evaluated": bool(partially_evaluated),
                "remainder_covered_by": remainder_covered_by,
                "detail": detail,
            }
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
        g1,
        evaluated=(
            "the off arm passes collective_registration=None explicitly and the "
            "default arm omits the parameter, so the two constructions differ; "
            "compared on step makespans, TTFT, the GOAL digest list, the "
            "registration outcome count and the emptiness of the per-artifact "
            "registration projection"
        ),
        detail=", ".join(g1_detail),
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
            g2_detail.append(
                f"{cell_name}/{arm}={'equal' if same else 'DIFFERENT'}"
                f"({len(off_digests)} artifacts)"
            )
    record(
        "G2",
        g2,
        evaluated=(
            "every arm's GOAL digest list compared against its cell's off arm. "
            "Vacuous in local-tp2, which writes no GOAL artifact at all; the "
            "48 mixed-tp4 artifacts carry the whole guard"
        ),
        detail=", ".join(g2_detail),
    )

    g3 = True
    g3_detail = []
    for cell_name, cell in results["cells"].items():
        off_steps = cell["arms"]["off"]["steps"]
        absent_steps = cell["default_construction"]["steps"]
        same = len(off_steps) == len(absent_steps)
        mismatched: list[str] = []
        if same:
            for off_step, absent_step in zip(off_steps, absent_steps, strict=True):
                for field in G3_STEP_FIELDS:
                    if off_step[field] != absent_step[field]:
                        mismatched.append(f"step{off_step['step_index']}.{field}")
            same = not mismatched
        g3 = g3 and same
        g3_detail.append(
            f"{cell_name}={'identical' if same else 'DIFFERENT:' + ','.join(mismatched)}"
        )
    record(
        "G3",
        g3,
        evaluated=(
            "field by field against the feature-absent computation: for every "
            "step of both cells, the off arm's makespan, completion, "
            "per-artifact fabric service, per-artifact local service, "
            "per-artifact base latency, per-artifact composed service, "
            "per-artifact medium and complete medium partition each equal the "
            "value the arm built without the parameter produced"
        ),
        detail=", ".join(g3_detail),
    )

    g4_steps = 0
    g4 = True
    for cell in results["cells"].values():
        for payload in cell["arms"].values():
            for step in payload["steps"]:
                g4 = g4 and step["media"]["total_ps"] == step["makespan_ps"]
                g4_steps += 1
            g4 = (
                g4
                and payload["ttft_partition_conserves"]
                and payload["decode_partition_conserves"]
            )
    request_rows = sum(len(cell["arms"]) for cell in results["cells"].values())
    record(
        "G4",
        g4,
        evaluated=(
            f"both halves of the frozen claim: {g4_steps} evaluated step "
            f"partitions conserve their makespans, and {request_rows} per-request "
            "rows conserve their TTFT and decode spans in both the coarse and "
            "the medium view"
        ),
        detail=f"{g4_steps} steps, {request_rows} request rows",
    )

    fail_closed = results["fail_closed"]
    g5 = (
        fail_closed["calibrated_request_raises"]
        and fail_closed["unknown_selector_raises"]
        and fail_closed["calibrated_without_measurement_raises"]
    )
    record(
        "G5",
        g5,
        evaluated=(
            "all three clauses: a calibrated request against the declared cost "
            "raises, a calibrated provenance without a measurement locator "
            "cannot be constructed, and an unknown model selector raises"
        ),
        detail=fail_closed["calibrated_message"],
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
        g6,
        evaluated=(
            "for every enabled arm and every step, the nonzero per-artifact "
            "charges sum to the ledger charge and their count times the "
            "per-identity cost times the channel count reproduces it"
        ),
        detail=", ".join(g6_detail),
    )

    seam = results["seam"]
    record(
        "G7",
        seam["accepted_sequences_reproduced"] and seam["ungated_stream_identical"],
        evaluated=(
            "against the accepted nccl_stack_v1 sequences: all "
            f"{seam['accepted_sequence_count']} tracked per-rank event-sequence "
            "rows of examples/nccl_stack_v1/results.csv were regenerated through "
            "an ungated communicator and reproduce the tracked event count and "
            "SHA-256 exactly; a default and an explicitly ungated communicator "
            "also produce identical JSON"
        ),
        detail=(
            f"{seam['ungated_event_count']} events in the reference inter-node "
            f"rank 0 stream, {seam['gated_registration_events']} registration "
            "events when the gate is requested"
        ),
    )

    record(
        "G8",
        results["identity"] == results["frozen_identity"],
        evaluated=(
            "the geometry read back off the executed sinks, plus the declared "
            "cost and evidence class read off the shipped model, compared "
            "against expectations.json"
        ),
        detail=(
            "both sides derived independently: the results side from the "
            "constructed HtsimStepSinkConfig objects, the frozen side from the "
            "freeze document"
        ),
    )

    record(
        "G9",
        all(results["ledger_facts"].values()),
        evaluated="all six by-construction ledger facts re-derived in this runner",
        detail=", ".join(
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
                "goal_artifacts": len(cell["arms"]["off"]["goal_sha256"]),
                #: the digest conjunct carries nothing in a cell that writes no
                #: GOAL artifact; only the makespan conjunct does there
                "goal_conjunct_vacuous": not cell["arms"]["off"]["goal_sha256"],
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
        "overlap_note": (
            "O1, B1's 64-identity instance, B2's local-tp2 instance and O6's "
            "step-0 equality are four views of one event: the 64 charges of "
            "local-tp2 step 0. Once G9 holds, identity count times declared "
            "cost is entailed, so those four do not multiply the evidence. The "
            "genuinely independent risks the scored set covers are the charging "
            "site, the phase-zero gate, the dedup across steps, the channel "
            "multiplicity and the rebuild"
        ),
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


def _counterfactual(results: dict[str, Any]) -> dict[str, Any]:
    """Quantify what B2 would have seen had the charge been folded into the max.

    The composition is per executed artifact, so the alternative this family
    exists to reject is `max(local, fabric, registration)` on each charged
    artifact rather than `registration + max(local, fabric)`. The observable
    difference is the per-artifact fabric term summed over the charged
    artifacts, which is what makes the discrimination weak when that term is
    small and complete when it exceeds the charge.
    """

    rows = []
    sources = [("frozen", name, cell) for name, cell in results["cells"].items()]
    sources += [
        ("post-specified", name, cell)
        for name, cell in results["post_specified_cells"].items()
    ]
    for provenance, name, cell in sources:
        on_step = cell["arms"]["on"]["steps"][0]
        off_step = cell["arms"]["off"]["steps"][0]
        charged_indexes = [
            index
            for index, value in enumerate(on_step["registration_phase_cost_ps"])
            if value
        ]
        realized = [
            max(
                off_step["local_phase_service_ps"][index],
                off_step["fabric_phase_service_ps"][index],
            )
            for index in charged_indexes
        ]
        observed_delta = on_step["makespan_ps"] - off_step["makespan_ps"]
        folded_delta = sum(
            max(service_ps, COST_PS) - service_ps for service_ps in realized
        )
        rows.append(
            {
                "provenance": provenance,
                "cell": name,
                "charged_artifacts": len(charged_indexes),
                "per_artifact_realized_service_ps": realized[:1],
                "observed_delta_ps": observed_delta,
                "folded_into_max_delta_ps": folded_delta,
                "discrimination_ps": observed_delta - folded_delta,
                "discrimination_fraction": (
                    (observed_delta - folded_delta) / observed_delta
                    if observed_delta
                    else None
                ),
                "fully_hidden_under_fold": folded_delta == 0,
            }
        )
    return {"rows": rows}


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


def _accepted_nccl_stack_sequences() -> dict[str, Any]:
    """Regenerate the accepted nccl_stack_v1 sequences through an ungated stack.

    The frozen G7 names those sequences, so the check reruns the accepted
    study's own reference routes with the current, gate-carrying module and
    compares each event count and SHA-256 against the tracked
    ``examples/nccl_stack_v1/results.csv``. The digest is built by the accepted
    study's own helpers, imported by path, so the two sides cannot drift on how
    a sequence is serialized.
    """

    import csv
    import importlib.util

    accepted_dir = STUDY_DIR.parent / "nccl_stack_v1"
    spec = importlib.util.spec_from_file_location(
        "_accepted_nccl_stack_v1",
        accepted_dir / "run_nccl_stack_v1.py",
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("the accepted nccl_stack_v1 runner could not be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    tracked: dict[tuple[str, str], dict[str, Any]] = {}
    with (accepted_dir / "results.csv").open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["family"].endswith("_event_sequence"):
                tracked[(row["family"], row["check"])] = json.loads(row["expected"])

    rows = []
    reproduced = True
    for (family, check), expected in sorted(tracked.items()):
        route = (
            module.NcclRoute.INTER_NODE
            if family.startswith("inter_node")
            else module.NcclRoute.INTRA_NODE
        )
        rank = int(check.rsplit("_", 1)[1])
        _, stack, _, _ = module._run_reference_route(route, rank)
        measured = tuple(module._event_tuple(event) for event in stack.events)
        observed = {
            "event_count": len(measured),
            "sha256": module._digest(measured),
        }
        matched = observed == expected
        reproduced = reproduced and matched
        rows.append(
            {
                "family": family,
                "check": check,
                "expected": expected,
                "observed": observed,
                "matched": matched,
            }
        )
    return {"reproduced": reproduced, "row_count": len(rows), "rows": rows}


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

    accepted = _accepted_nccl_stack_sequences()

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
        "accepted_sequences_reproduced": accepted["reproduced"],
        "accepted_sequence_count": accepted["row_count"],
        "accepted_sequence_rows": accepted["rows"],
        "ungated_event_count": len(default_events),
        "gated_registration_events": len(registration.events),
        "gated_declared_total_ps": registration.total_declared_cost_ps,
        "clock_unmoved": gated_stack.clock.now_ps == 17,
        "unregistered_buffer_refused": gate_refused,
    }


def _fail_closed_evidence() -> dict[str, Any]:
    from simllm.traffic import (
        CollectiveRegistrationProvenance,
        resolve_collective_registration,
    )

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

    #: the frozen G5's third clause: a cost cannot claim the calibrated class
    #: without naming the measurement it was fitted to
    construction_raises = False
    try:
        CollectiveRegistrationProvenance(
            evidence_class="calibrated",
            source="a capture that does not exist",
            locator="a row that does not exist",
            basis="none",
        )
    except ValueError:
        construction_raises = True

    return {
        "calibrated_request_raises": calibrated_raises,
        "calibrated_message": message,
        "unknown_selector_raises": unknown_raises,
        "calibrated_without_measurement_raises": construction_raises,
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

    #: Every field of the results side comes from what executed: the geometry
    #: is read back off the constructed sinks and the cost off the shipped
    #: model, so G8 compares the freeze against the run rather than against
    #: itself.
    identity = {
        "registration_cost_ps": (
            DECLARED_NCCL_CHANNEL_REGISTRATION_COST.registration_cost_ps
        ),
        "evidence_class": DECLARED_NCCL_CHANNEL_REGISTRATION_COST.evidence_class,
        "cells": {
            name: {
                "dims": cell["arms"]["off"]["executed_geometry"]["dims"],
                "tp_ranks": cell["arms"]["off"]["executed_geometry"]["tp_ranks"],
                "hosts": cell["arms"]["off"]["executed_geometry"]["hosts"],
                "steps": cell["arms"]["off"]["executed_geometry"]["steps"],
                "arms": sorted(cell["arms"]),
                "identities_one_channel": (
                    len(cell["arms"]["on"]["registered_identities"])
                ),
            }
            for name, cell in cells.items()
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
                "arms": sorted(cell["arms"]),
                "identities_one_channel": cell["identities_one_channel"],
            }
            for name, cell in EXPECTATIONS["cells"].items()
        },
    }

    post_specified: dict[str, Any] = {}
    for cell_name, cell in POST_SPECIFIED_CELLS.items():
        arms = {
            arm: _run_arm(cell_name, cell, arm, workdir) for arm in cell["arms"]
        }
        post_specified[cell_name] = {
            "identities_one_channel": cell["identities_one_channel"],
            "linkspeed_bps": cell["linkspeed_bps"],
            "arms": arms,
        }

    results: dict[str, Any] = {
        "study": "nccl_registration_v1",
        "freeze": "examples/nccl_registration_v1/expectations.md",
        "identity": identity,
        "frozen_identity": frozen_identity,
        "cells": cells,
        "post_specified_cells": post_specified,
        "ledger_facts": _ledger_facts(),
        "seam": _seam_evidence(),
        "fail_closed": _fail_closed_evidence(),
    }
    results["fatal_guards"] = _guards(results)
    results["scored"] = _scored(results)
    results["counterfactual"] = _counterfactual(results)
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
