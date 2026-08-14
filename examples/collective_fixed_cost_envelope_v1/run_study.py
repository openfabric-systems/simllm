"""Run the frozen per-collective fixed-cost envelope study.

The study replays two deterministic step records through ``HtsimStepSink``
under four named fixed-cost arms, two link rates and two expert-parallel
widths, then reports the bracket the arms span. Every predicted value comes
from ``expectations.json``, which was frozen before any arm existed.

Bulk artifacts are written under ``--run-dir`` or the directory named by the
``SIMLLM_FIXED_COST_ENVELOPE_RUN_ROOT`` environment variable. There is no
built-in default: a run outside the repository is a deliberate choice the
caller makes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from dataclasses import asdict
from fractions import Fraction
from itertools import pairwise
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

STUDY_DIR = Path(__file__).resolve().parent
EXPECTATIONS_PATH = STUDY_DIR / "expectations.json"
RUN_ROOT_ENV = "SIMLLM_FIXED_COST_ENVELOPE_RUN_ROOT"
EXPECTATIONS_COMMIT = "9c568b5"
PICOSECONDS_PER_SECOND = 10**12
LINK_RATES_BPS = (400_000_000_000, 200_000_000_000)
EXPERT_PARALLEL_WIDTHS = (4, 8)
PHASES = ("prefill", "decode")
STUDY_ARMS = ("off", "floor", "local", "cross")
EXPECTED_SCORED_FAMILIES = 3
SCORED_RELATION_NAMES = ("S1", "S2", "S3")
FATAL_GUARD_NAMES = ("G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8", "G9")
EXACT_UNSCORED_ROW_NAMES = ("E1", "E2", "E3", "E4")
UNSUPPORTED_WIDTH = 16


def load_expectations() -> dict[str, Any]:
    """Return the frozen expectations document."""

    return json.loads(EXPECTATIONS_PATH.read_text(encoding="utf-8"))


def _cell_name(width: int, phase: str, link_bps: int, arm: str) -> str:
    return f"ep{width}-{phase}-{link_bps // 10**9}g-{arm}"


def _fabric_key(width: int, phase: str, link_bps: int) -> str:
    return f"ep{width}-{phase}-{link_bps // 10**9}g"


def check_arithmetic(frozen: dict[str, Any]) -> dict[str, Any]:
    """Rederive the frozen predictions from the installed calibrations.

    Every predicted step latency is ``compute + 48 * (propagation +
    serialization) + 48 * surcharge``. Recomputing it here proves the freeze is
    arithmetic on constants that live in the repository rather than a table
    typed in by hand.
    """

    from simllm.traffic import (
        B200_NCCL_2_27_CROSS_NODE_PROVISIONAL_PROFILE,
        B200_NCCL_2_27_LOCAL_PROFILE,
        COLLECTIVE_FIXED_COST_FLOOR_PROFILE,
        COLLECTIVE_PROPAGATION_REFERENCE_PS,
    )

    constants = frozen["constants_ps"]
    if COLLECTIVE_PROPAGATION_REFERENCE_PS != constants["propagation_reference_ps"]:
        raise AssertionError("the installed propagation reference moved")
    surcharge = {
        "off": {width: 0 for width in EXPERT_PARALLEL_WIDTHS},
        "floor": {
            width: COLLECTIVE_FIXED_COST_FLOOR_PROFILE.base_latency_ps(width)
            for width in EXPERT_PARALLEL_WIDTHS
        },
        "local": {
            width: B200_NCCL_2_27_LOCAL_PROFILE.base_latency_ps(width)
            for width in EXPERT_PARALLEL_WIDTHS
        },
        "cross": {
            width: B200_NCCL_2_27_CROSS_NODE_PROVISIONAL_PROFILE.base_latency_ps(width)
            for width in EXPERT_PARALLEL_WIDTHS
        },
    }
    collectives = constants["collectives_per_step"]
    derived_fabric: dict[str, int] = {}
    derived_latency: dict[str, int] = {}
    for width in EXPERT_PARALLEL_WIDTHS:
        for phase in PHASES:
            structural = frozen["structural_inputs"][f"ep{width}-{phase}"]
            endpoint_bytes = structural["endpoint_bytes"]
            compute_ps = structural["provider_compute_ps"]
            for link_bps in LINK_RATES_BPS:
                rate = link_bps // 8
                serialization_ps = -(-endpoint_bytes * PICOSECONDS_PER_SECOND // rate)
                fabric_ps = COLLECTIVE_PROPAGATION_REFERENCE_PS + serialization_ps
                derived_fabric[_fabric_key(width, phase, link_bps)] = fabric_ps
                for arm in STUDY_ARMS:
                    derived_latency[_cell_name(width, phase, link_bps, arm)] = (
                        compute_ps
                        + collectives * (fabric_ps + surcharge[arm][width])
                    )
    if derived_fabric != frozen["predicted_fabric_service_ps"]:
        raise AssertionError("the frozen fabric predictions are not reproducible")
    if derived_latency != frozen["predicted_step_latency_ps"]:
        raise AssertionError("the frozen step predictions are not reproducible")
    return {
        "surcharge_ps": surcharge,
        "predicted_fabric_service_ps": derived_fabric,
        "predicted_step_latency_ps": derived_latency,
    }


def _dims(ep_world: int) -> Any:
    from simllm.compute import ModelDims

    return ModelDims(
        num_layers=24,
        hidden_size=1_024,
        intermediate_size=512,
        num_heads=16,
        num_kv_heads=8,
        head_size=64,
        vocab_size=49_152,
        dtype_bytes=2,
        num_experts=32,
        top_k=8,
        moe_intermediate_size=512,
        local_num_experts=32 // ep_world,
    )


def _records() -> tuple[Any, ...]:
    from simllm.core import RequestPhase, ScheduledRequest, StepRecord

    specs = (
        (0, RequestPhase.PREFILL, 8, 8),
        (1, RequestPhase.DECODE, 1, 9),
    )
    return tuple(
        StepRecord(
            step_index=step_index,
            virtual_time_ps=0,
            scheduled=[
                ScheduledRequest(
                    request_id="envelope",
                    phase=phase,
                    num_new_tokens=new_tokens,
                    context_length=context_length,
                )
            ],
            num_sampled=1,
        )
        for step_index, phase, new_tokens, context_length in specs
    )


def _arm_selection(arm: str) -> dict[str, Any]:
    frozen_arms = load_expectations()["arms"]
    if arm == "default":
        return {}
    selection = frozen_arms[arm]
    return {
        "collective_fixed_cost_envelope": selection["envelope"],
        "collective_fixed_cost_arm": selection["arm"],
    }


def _artifact_manifest(workdir: Path) -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "name": path.name,
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in sorted(workdir.glob("*.goal"))
    )


def _structural_inventory(ep_world: int) -> dict[str, Any]:
    """Return the collective inventory of the fixture, timing excluded."""

    from simllm.backends.step_lowerer import SerialStepLowerer, SerialStepLowererConfig
    from simllm.core import CollectiveWork
    from simllm.traffic import critical_collective_endpoint_bytes

    rows: dict[str, Any] = {}
    for record, phase in zip(_records(), PHASES, strict=True):
        lowerer = SerialStepLowerer(
            SerialStepLowererConfig(
                dims=_dims(ep_world),
                tp_ranks=(0,),
                ep_ranks=tuple(range(ep_world)),
            )
        )
        graph, timing = lowerer.lower_with_timing(record)
        collectives = tuple(
            operation
            for operation in graph.operations
            if isinstance(operation.work, CollectiveWork)
        )
        rows[phase] = {
            "collective_count": len(collectives),
            "kinds": [
                list(kind)
                for kind in sorted(
                    {
                        (operation.work.collective, operation.work.algorithm_hint)
                        for operation in collectives
                    }
                )
            ],
            "participant_counts": sorted(
                {len(operation.work.ranks) for operation in collectives}
            ),
            "endpoint_bytes": sorted(
                {
                    critical_collective_endpoint_bytes(operation.work)
                    for operation in collectives
                }
            ),
            "provider_compute_ps": timing.compute_estimate_ps,
        }
    return rows


def _cell(args: argparse.Namespace, *, arm: str, link_bps: int, ep_world: int) -> dict[str, Any]:
    from simllm.backends import HtsimStepSink, HtsimStepSinkConfig

    name = f"ep{ep_world}-{link_bps // 10**9}g-{arm}"
    workdir = args.run_dir / "cells" / name
    sink = HtsimStepSink(
        HtsimStepSinkConfig(
            profile="rnic-nn-fluid",
            tp_ranks=(0,),
            ep_ranks=tuple(range(ep_world)),
            dims=_dims(ep_world),
            workdir=workdir,
            linkspeed_bps=link_bps,
            **_arm_selection(arm),
        )
    )
    results = []
    virtual_time_ps = 0
    for record in _records():
        record.virtual_time_ps = virtual_time_ps
        result = sink(record)
        if result is None:
            raise AssertionError(f"cell {name} produced a step with no collective work")
        results.append(result)
        virtual_time_ps = result.completed_at_ps
    return {
        "name": name,
        "arm": arm,
        "linkspeed_bps": link_bps,
        "ep_world": ep_world,
        "step_latency_ps": [result.step_latency_ps for result in results],
        "step_results": [asdict(result) for result in results],
        "network_outcomes": [asdict(outcome) for outcome in sink.outcomes],
        "locality_outcomes": [asdict(outcome) for outcome in sink.locality_outcomes],
        "collective_timing_outcomes": [
            asdict(outcome) for outcome in sink.collective_timing_outcomes
        ],
        "artifact_manifest": _artifact_manifest(workdir),
    }


def _unsupported_width_probe(args: argparse.Namespace) -> dict[str, Any]:
    from simllm.backends import HtsimStepSink, HtsimStepSinkConfig

    workdir = args.run_dir / "cells" / "unsupported-width"
    sink = HtsimStepSink(
        HtsimStepSinkConfig(
            profile="rnic-nn-fluid",
            tp_ranks=(0,),
            ep_ranks=tuple(range(UNSUPPORTED_WIDTH)),
            dims=_dims(UNSUPPORTED_WIDTH),
            workdir=workdir,
            collective_fixed_cost_envelope="cross-node-fixed-cost-provisional-v1",
            collective_fixed_cost_arm="upper",
        )
    )
    raised = ""
    try:
        sink(_records()[1])
    except ValueError as error:
        raised = str(error)
    return {
        "participant_count": UNSUPPORTED_WIDTH,
        "raised": raised,
        "goal_artifacts": len(tuple(workdir.glob("*.goal"))),
        "completion_artifacts": len(tuple(workdir.glob("*.csv"))),
        "published_outcomes": len(sink.outcomes),
    }


def _envelope_inventory() -> dict[str, Any]:
    from simllm.traffic import (
        CROSS_NODE_COLLECTIVE_FIXED_COST_ENVELOPE,
        INTRA_NODE_COLLECTIVE_FIXED_COST_ENVELOPE,
    )

    rows: dict[str, Any] = {}
    for envelope in (
        INTRA_NODE_COLLECTIVE_FIXED_COST_ENVELOPE,
        CROSS_NODE_COLLECTIVE_FIXED_COST_ENVELOPE,
    ):
        arms = {}
        for arm in ("lower", "upper"):
            profile = envelope.arm_profile(arm)
            provenance = profile.require_provenance()
            arms[arm] = {
                "profile_id": profile.profile_id,
                "evidence_class": provenance.evidence_class,
                "applied_evidence_class": envelope.arm_evidence_class(arm),
                "applied_evidence_note": envelope.arm_evidence_note(arm),
                "source": provenance.source,
                "locator": provenance.locator,
                "transfer": provenance.transfer,
                "bandwidth_bytes_per_second": profile.bandwidth_bytes_per_second,
                "source_payload_bytes": [
                    profile.source_payload_bytes_min,
                    profile.source_payload_bytes_max,
                ],
                "participant_latency_ps": [
                    list(entry) for entry in profile.participant_latency_ps
                ],
                "participant_latency_band_ps": [
                    list(entry) for entry in provenance.participant_latency_band_ps
                ],
            }
        rows[envelope.envelope_id] = {
            "claim": envelope.claim,
            "arms": arms,
            "bracket_ps": {
                str(width): list(envelope.bracket_ps(width))
                for width in envelope.supported_participant_counts
            },
            "realized_bracket_ps": {
                str(width): list(envelope.realized_bracket_ps(width))
                for width in envelope.supported_participant_counts
            },
            "strictly_brackets": all(
                envelope.bracket_ps(width)[0] < envelope.bracket_ps(width)[1]
                for width in envelope.supported_participant_counts
            ),
            "isolates_fixed_cost": (
                envelope.lower_profile.bandwidth_bytes_per_second
                == envelope.upper_profile.bandwidth_bytes_per_second
                and envelope.lower_profile.source_payload_bytes_min
                == envelope.upper_profile.source_payload_bytes_min
                and envelope.lower_profile.source_payload_bytes_max
                == envelope.upper_profile.source_payload_bytes_max
                and envelope.lower_profile.propagation_reference_ps
                == envelope.upper_profile.propagation_reference_ps
            ),
        }
    return rows


def _relative_error(measured: int, predicted: int) -> float:
    return abs(measured - predicted) / predicted


def _score_s1(cells: dict[str, dict[str, Any]], frozen: dict[str, Any]) -> dict[str, Any]:
    tolerance = frozen["scored"]["S1"]["relative_tolerance"]
    predicted = frozen["predicted_step_latency_ps"]
    rows = []
    for arm in STUDY_ARMS:
        for link_bps in LINK_RATES_BPS:
            for width in EXPERT_PARALLEL_WIDTHS:
                cell = cells[f"ep{width}-{link_bps // 10**9}g-{arm}"]
                for index, phase in enumerate(PHASES):
                    key = _cell_name(width, phase, link_bps, arm)
                    measured = cell["step_latency_ps"][index]
                    error = _relative_error(measured, predicted[key])
                    rows.append(
                        {
                            "row": key,
                            "measured_ps": measured,
                            "predicted_ps": predicted[key],
                            "relative_error": error,
                            "passed": error <= tolerance,
                        }
                    )
    return {
        "name": "S1",
        "tolerance": tolerance,
        "rows": rows,
        "passed": all(row["passed"] for row in rows),
    }


def _ratio(numerator_ps: int, denominator_ps: int) -> float:
    return float(Fraction(numerator_ps, denominator_ps))


def _score_s2(cells: dict[str, dict[str, Any]], frozen: dict[str, Any]) -> dict[str, Any]:
    from simllm.traffic import arm_ratio_envelope

    tolerance = frozen["scored"]["S2"]["relative_tolerance"]
    expected = frozen["scored"]["S2"]["values"]
    rows = []
    envelopes = {}
    for link_bps in LINK_RATES_BPS:
        link = f"{link_bps // 10**9}g"
        arm_values = []
        for arm in STUDY_ARMS:
            numerator = cells[f"ep4-{link}-{arm}"]["step_latency_ps"][1]
            denominator = cells[f"ep8-{link}-{arm}"]["step_latency_ps"][1]
            arm_values.append((arm, numerator, denominator))
            measured = _ratio(numerator, denominator)
            predicted = expected[link][arm]
            error = abs(measured - predicted) / predicted
            rows.append(
                {
                    "row": f"{link}-{arm}",
                    "measured": measured,
                    "predicted": predicted,
                    "relative_error": error,
                    "passed": error <= tolerance,
                }
            )
        envelope = arm_ratio_envelope(
            f"ep4 over ep8 decode at {link}",
            "ep4-decode",
            "ep8-decode",
            tuple(arm_values),
        )
        envelopes[link] = asdict(envelope)
    return {
        "name": "S2",
        "tolerance": tolerance,
        "rows": rows,
        "arm_ratio_envelopes": envelopes,
        "passed": all(row["passed"] for row in rows),
    }


def _score_s3(cells: dict[str, dict[str, Any]], frozen: dict[str, Any]) -> dict[str, Any]:
    from simllm.traffic import arm_ratio_envelope

    tolerance = frozen["scored"]["S3"]["relative_tolerance"]
    expected = frozen["scored"]["S3"]["values"]
    rows = []
    envelopes = {}
    for width in EXPERT_PARALLEL_WIDTHS:
        for index, phase in enumerate(PHASES):
            label = f"ep{width}-{phase}"
            arm_values = []
            for arm in STUDY_ARMS:
                numerator = cells[f"ep{width}-200g-{arm}"]["step_latency_ps"][index]
                denominator = cells[f"ep{width}-400g-{arm}"]["step_latency_ps"][index]
                arm_values.append((arm, numerator, denominator))
                measured = _ratio(numerator, denominator)
                predicted = expected[label][arm]
                error = abs(measured - predicted) / predicted
                rows.append(
                    {
                        "row": f"{label}-{arm}",
                        "measured": measured,
                        "predicted": predicted,
                        "relative_error": error,
                        "passed": error <= tolerance,
                    }
                )
            envelopes[label] = asdict(
                arm_ratio_envelope(
                    f"200G over 400G for {label}",
                    f"{label}-200g",
                    f"{label}-400g",
                    tuple(arm_values),
                )
            )
    return {
        "name": "S3",
        "tolerance": tolerance,
        "rows": rows,
        "arm_ratio_envelopes": envelopes,
        "passed": all(row["passed"] for row in rows),
    }


def _exact_rows(cells: dict[str, dict[str, Any]], derived: dict[str, Any]) -> dict[str, Any]:
    """Check the entailed closed-form rows; a violation is fatal, never scored."""

    collectives = load_expectations()["constants_ps"]["collectives_per_step"]
    additivity = []
    ordering = []
    for link_bps in LINK_RATES_BPS:
        link = f"{link_bps // 10**9}g"
        for width in EXPERT_PARALLEL_WIDTHS:
            baseline = cells[f"ep{width}-{link}-off"]["step_latency_ps"]
            for arm in STUDY_ARMS:
                measured = cells[f"ep{width}-{link}-{arm}"]["step_latency_ps"]
                expected_delta = collectives * derived["surcharge_ps"][arm][width]
                additivity.append(
                    {
                        "row": f"ep{width}-{link}-{arm}",
                        "expected_delta_ps": expected_delta,
                        "observed_delta_ps": [
                            measured[index] - baseline[index]
                            for index in range(len(PHASES))
                        ],
                        "held": all(
                            measured[index] - baseline[index] == expected_delta
                            for index in range(len(PHASES))
                        ),
                    }
                )
            latencies = [
                cells[f"ep{width}-{link}-{arm}"]["step_latency_ps"] for arm in STUDY_ARMS
            ]
            ordering.append(
                {
                    "row": f"ep{width}-{link}",
                    "held": all(
                        latencies[0][index] == latencies[1][index]
                        < latencies[2][index]
                        < latencies[3][index]
                        for index in range(len(PHASES))
                    ),
                }
            )
    compression = []
    for width in EXPERT_PARALLEL_WIDTHS:
        for index, phase in enumerate(PHASES):
            ratios = [
                _ratio(
                    cells[f"ep{width}-200g-{arm}"]["step_latency_ps"][index],
                    cells[f"ep{width}-400g-{arm}"]["step_latency_ps"][index],
                )
                for arm in STUDY_ARMS
            ]
            compression.append(
                {
                    "row": f"ep{width}-{phase}",
                    "ratios": ratios,
                    "held": all(
                        earlier >= later
                        for earlier, later in pairwise(ratios)
                    ),
                }
            )
    flip = []
    for link_bps in LINK_RATES_BPS:
        link = f"{link_bps // 10**9}g"
        ratios = {
            arm: _ratio(
                cells[f"ep4-{link}-{arm}"]["step_latency_ps"][1],
                cells[f"ep8-{link}-{arm}"]["step_latency_ps"][1],
            )
            for arm in STUDY_ARMS
        }
        surcharge = derived["surcharge_ps"]
        premise = ratios["off"] > 1.0 and all(
            surcharge[arm][4] < surcharge[arm][8] for arm in ("local", "cross")
        )
        flip.append(
            {
                "row": link,
                "ratios": ratios,
                "premise_held": premise,
                "held": (not premise)
                or min(ratios[arm] for arm in ("local", "cross")) < ratios["off"],
            }
        )
    return {
        "E1": {"rows": additivity, "held": all(row["held"] for row in additivity)},
        "E2": {"rows": ordering, "held": all(row["held"] for row in ordering)},
        "E3": {"rows": compression, "held": all(row["held"] for row in compression)},
        "E4": {"rows": flip, "held": all(row["held"] for row in flip)},
    }


def _guards(
    cells: dict[str, dict[str, Any]],
    guard_cells: dict[str, dict[str, Any]],
    structural: dict[int, dict[str, Any]],
    frozen: dict[str, Any],
    derived: dict[str, Any],
    unsupported: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    from simllm.traffic import (
        CROSS_NODE_COLLECTIVE_FIXED_COST_ENVELOPE,
        INTRA_NODE_COLLECTIVE_FIXED_COST_ENVELOPE,
    )

    collectives = frozen["constants_ps"]["collectives_per_step"]
    guards: dict[str, dict[str, Any]] = {}

    default_rows = []
    floor_rows = []
    for link_bps in LINK_RATES_BPS:
        link = f"{link_bps // 10**9}g"
        for width in EXPERT_PARALLEL_WIDTHS:
            off = cells[f"ep{width}-{link}-off"]
            default = guard_cells[f"ep{width}-{link}-default"]
            floor = cells[f"ep{width}-{link}-floor"]
            default_rows.append(
                {
                    "row": f"ep{width}-{link}",
                    "held": (
                        default["step_results"] == off["step_results"]
                        and default["artifact_manifest"] == off["artifact_manifest"]
                        and default["collective_timing_outcomes"] == []
                        and off["collective_timing_outcomes"] == []
                    ),
                }
            )
            floor_timing = floor["collective_timing_outcomes"]
            floor_rows.append(
                {
                    "row": f"ep{width}-{link}",
                    "held": (
                        floor["step_results"] == off["step_results"]
                        and floor["artifact_manifest"] == off["artifact_manifest"]
                        and len(floor_timing) == len(PHASES)
                        and all(
                            outcome["envelope_id"] == "intra-node-fixed-cost-v1"
                            and outcome["arm"] == "lower"
                            and outcome["evidence_class"] == "structural-floor"
                            and outcome["propagation_reference_ps"]
                            == frozen["constants_ps"]["propagation_reference_ps"]
                            and all(
                                row["collective_base_latency_ps"] == 0
                                for row in outcome["artifacts"]
                            )
                            for outcome in floor_timing
                        )
                    ),
                }
            )
    guards["G1"] = {"rows": default_rows, "held": all(r["held"] for r in default_rows)}
    guards["G2"] = {"rows": floor_rows, "held": all(r["held"] for r in floor_rows)}

    all_cells = list(cells.values()) + list(guard_cells.values())
    guards["G3"] = {
        "held": all(
            outcome["nvlink_directed_bytes"] == 0 and outcome["nvlink_service_ps"] == 0
            for cell in all_cells
            for outcome in cell["locality_outcomes"]
        )
    }

    structural_rows = []
    for width in EXPERT_PARALLEL_WIDTHS:
        for phase in PHASES:
            observed = structural[width][phase]
            expected = frozen["structural_inputs"][f"ep{width}-{phase}"]
            structural_rows.append(
                {
                    "row": f"ep{width}-{phase}",
                    "observed": observed,
                    "held": (
                        observed["collective_count"] == collectives
                        and observed["kinds"] == [["all-to-allv", "pairwise"]]
                        and observed["participant_counts"] == [width]
                        and observed["endpoint_bytes"] == [expected["endpoint_bytes"]]
                        and observed["provider_compute_ps"]
                        == expected["provider_compute_ps"]
                    ),
                }
            )
    guards["G4"] = {
        "rows": structural_rows,
        "held": all(row["held"] for row in structural_rows),
    }

    base_rows = []
    for arm in ("floor", "local", "cross"):
        for link_bps in LINK_RATES_BPS:
            link = f"{link_bps // 10**9}g"
            for width in EXPERT_PARALLEL_WIDTHS:
                cell = cells[f"ep{width}-{link}-{arm}"]
                expected_total = collectives * derived["surcharge_ps"][arm][width]
                base_rows.append(
                    {
                        "row": f"ep{width}-{link}-{arm}",
                        "expected_total_ps": expected_total,
                        "held": all(
                            sum(
                                row["collective_base_latency_ps"]
                                for row in outcome["artifacts"]
                            )
                            == expected_total
                            for outcome in cell["collective_timing_outcomes"]
                        ),
                    }
                )
    guards["G5"] = {"rows": base_rows, "held": all(row["held"] for row in base_rows)}

    inventory = _envelope_inventory()
    guards["G6"] = {
        "inventory": inventory,
        "held": all(
            row["strictly_brackets"] and row["isolates_fixed_cost"]
            for row in inventory.values()
        ),
    }

    guards["G7"] = {
        "probe": unsupported,
        "held": (
            "participant count 16" in unsupported["raised"]
            and unsupported["goal_artifacts"] == 0
            and unsupported["completion_artifacts"] == 0
            and unsupported["published_outcomes"] == 0
        ),
    }

    guards["G8"] = {
        "held": all(
            outcome["quiescent"] and outcome["routing_mode"] == "uniform"
            for cell in all_cells
            for outcome in cell["network_outcomes"]
        )
    }

    envelope_rows = []
    for envelope in (
        INTRA_NODE_COLLECTIVE_FIXED_COST_ENVELOPE,
        CROSS_NODE_COLLECTIVE_FIXED_COST_ENVELOPE,
    ):
        for arm in ("lower", "upper"):
            profile = envelope.arm_profile(arm)
            for width in EXPERT_PARALLEL_WIDTHS:
                minimum, maximum = profile.endpoint_byte_bounds(width)
                for phase in PHASES:
                    load = frozen["structural_inputs"][f"ep{width}-{phase}"][
                        "endpoint_bytes"
                    ]
                    envelope_rows.append(
                        {
                            "row": f"{profile.profile_id}-ep{width}-{phase}",
                            "endpoint_bytes": load,
                            "bounds": [minimum, maximum],
                            "held": minimum <= load <= maximum,
                        }
                    )
    guards["G9"] = {
        "rows": envelope_rows,
        "held": all(row["held"] for row in envelope_rows),
    }

    return guards


def _partition_registered_relations(
    registered: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split the registered relations into scored families and fatal guards."""

    scored = {
        name: value for name, value in registered.items() if name in SCORED_RELATION_NAMES
    }
    guards = {
        name: value for name, value in registered.items() if name in FATAL_GUARD_NAMES
    }
    overlap = set(scored) & set(guards)
    if overlap:
        raise AssertionError(f"a relation cannot be both scored and fatal: {overlap}")
    return scored, guards


def _environment() -> dict[str, Any]:
    def git(*arguments: str) -> str:
        return subprocess.run(
            ["git", *arguments],
            cwd=REPOSITORY_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()

    return {
        "git_head": git("rev-parse", "HEAD"),
        "git_status_clean": git("status", "--porcelain") == "",
        "expectations_commit": EXPECTATIONS_COMMIT,
        "expectations_sha256": hashlib.sha256(
            EXPECTATIONS_PATH.read_bytes()
        ).hexdigest(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "htsim_rnic": os.environ.get("SIMLLM_HTSIM_RNIC", ""),
    }


def run_study(args: argparse.Namespace) -> dict[str, Any]:
    """Execute the frozen matrix and return the scored result document."""

    frozen = load_expectations()
    derived = check_arithmetic(frozen)
    args.run_dir.mkdir(parents=True, exist_ok=True)

    cells: dict[str, dict[str, Any]] = {}
    guard_cells: dict[str, dict[str, Any]] = {}
    for arm in STUDY_ARMS:
        for link_bps in LINK_RATES_BPS:
            for ep_world in EXPERT_PARALLEL_WIDTHS:
                cell = _cell(args, arm=arm, link_bps=link_bps, ep_world=ep_world)
                cells[cell["name"]] = cell
    for link_bps in LINK_RATES_BPS:
        for ep_world in EXPERT_PARALLEL_WIDTHS:
            cell = _cell(args, arm="default", link_bps=link_bps, ep_world=ep_world)
            guard_cells[cell["name"]] = cell

    structural = {width: _structural_inventory(width) for width in EXPERT_PARALLEL_WIDTHS}
    unsupported = _unsupported_width_probe(args)
    guards = _guards(cells, guard_cells, structural, frozen, derived, unsupported)
    registered = {
        "S1": _score_s1(cells, frozen),
        "S2": _score_s2(cells, frozen),
        "S3": _score_s3(cells, frozen),
        **guards,
    }
    scored, fatal = _partition_registered_relations(registered)
    if len(scored) != EXPECTED_SCORED_FAMILIES:
        raise AssertionError("the scored family count moved away from the freeze")
    exact = _exact_rows(cells, derived)
    void = any(not guard["held"] for guard in fatal.values()) or any(
        not row["held"] for row in exact.values()
    )
    return {
        "schema": "simllm-collective-fixed-cost-envelope-results-v1",
        "study": frozen["study"],
        "environment": _environment(),
        "run_configurations": {
            "frozen_cells": len(cells),
            "frozen_simulated_steps": len(cells) * len(PHASES),
            "guard_cells": len(guard_cells),
            "guard_simulated_steps": len(guard_cells) * len(PHASES),
        },
        "derived": {
            "surcharge_ps": {
                arm: {str(width): value for width, value in widths.items()}
                for arm, widths in derived["surcharge_ps"].items()
            }
        },
        "structural_inventory": {
            str(width): rows for width, rows in structural.items()
        },
        "scored": scored,
        "fatal_guards": fatal,
        "exact_unscored_rows": exact,
        "verdict": {
            "void": void,
            "scored_families_passed": sum(
                1 for family in scored.values() if family["passed"]
            ),
            "scored_families": len(scored),
        },
        "cells": cells,
        "guard_cells": guard_cells,
    }


def check_only(args: argparse.Namespace) -> None:
    """Verify the frozen arithmetic without running the backend."""

    frozen = load_expectations()
    derived = check_arithmetic(frozen)
    payload = {
        "expectations_sha256": hashlib.sha256(
            EXPECTATIONS_PATH.read_bytes()
        ).hexdigest(),
        "surcharge_ps": {
            arm: {str(width): value for width, value in widths.items()}
            for arm, widths in derived["surcharge_ps"].items()
        },
        "envelopes": _envelope_inventory(),
    }
    json.dump(payload, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")


def results_summary(results: dict[str, Any]) -> dict[str, Any]:
    """Return the trackable projection of a result document.

    The per-cell artifact manifests and outcome records are bulk run output and
    stay outside the repository; everything a reader needs to check the verdict
    stays here, including the measured step latencies themselves. The backend
    binary's location is replaced by a boolean, because a trackable document
    must not carry a machine-specific filesystem path.
    """

    summary = {
        key: value
        for key, value in results.items()
        if key not in ("cells", "guard_cells")
    }
    environment = dict(summary["environment"])
    environment["htsim_rnic_configured"] = bool(environment.pop("htsim_rnic", ""))
    summary["environment"] = environment
    summary["measured_step_latency_ps"] = {
        _cell_name(cell["ep_world"], phase, cell["linkspeed_bps"], cell["arm"]): (
            cell["step_latency_ps"][index]
        )
        for cell in results["cells"].values()
        for index, phase in enumerate(PHASES)
    }
    summary["measured_fabric_service_ps"] = {
        f"{cell['name']}-{phase}": sorted(
            {
                service_ps
                for service_ps in cell["locality_outcomes"][index][
                    "fabric_phase_service_ps"
                ]
                if service_ps
            }
        )
        for cell in results["cells"].values()
        for index, phase in enumerate(PHASES)
    }
    summary["measured_compute_service_ps"] = {
        f"{cell['name']}-{phase}": cell["locality_outcomes"][index][
            "compute_service_ps"
        ]
        for cell in results["cells"].values()
        for index, phase in enumerate(PHASES)
    }
    return summary


def _resolve_run_dir(value: str | None) -> Path:
    resolved = value or os.environ.get(RUN_ROOT_ENV)
    if not resolved:
        raise SystemExit(
            "no run directory: pass --run-dir or export "
            f"{RUN_ROOT_ENV} to a directory outside the repository"
        )
    return Path(resolved).expanduser().resolve()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--results", default=None)
    parser.add_argument("--summary", default=None)
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()
    if args.check_only:
        check_only(args)
        return 0
    args.run_dir = _resolve_run_dir(args.run_dir)
    results = run_study(args)
    destination = (
        Path(args.results).expanduser().resolve()
        if args.results
        else args.run_dir / "results.json"
    )
    destination.write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    summary_destination = (
        Path(args.summary).expanduser().resolve()
        if args.summary
        else args.run_dir / "results-summary.json"
    )
    summary_destination.write_text(
        json.dumps(results_summary(results), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    verdict = results["verdict"]
    sys.stdout.write(
        f"void={verdict['void']} "
        f"scored={verdict['scored_families_passed']}/{verdict['scored_families']} "
        f"results={destination}\n"
    )
    return 1 if verdict["void"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
