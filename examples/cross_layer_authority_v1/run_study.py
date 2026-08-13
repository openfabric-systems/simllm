"""Run the frozen CORE-8 cross-layer authority study.

The study executes one registered fixture across a two-parameter sweep, checks
the frozen projection clauses on every cell, and submits the thirteen
registered contradictions to the enforcing consumers. Evidence classes are kept
separate in the printed report and in the written artifacts.

``--check-only`` validates the frozen literal table and its arithmetic. That
path imports no SimLLM module, reads no input path and writes no artifact.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

FIXTURE_EXECUTION_ID = "cross-layer-authority"

#: WQE payloads on the single shared RNIC, in bytes.
WQE_PAYLOAD_BYTES = (4096, 2048, 8192, 128)

#: Picoseconds before the first RNIC grant: five 1,000 ps launches place
#: ``xfer`` on its NCCL channel at 2,000 ps, and 5,000 ps of channel service
#: releases it to the NIC at 7,000 ps.
FIRST_GRANT_PS = 7000

PS_PER_SECOND = 1_000_000_000_000

#: (rate_bps, completion_delivery_ps) -> frozen JCT in picoseconds.
FROZEN_JCT_PS = {
    (400_000_000_000, 700): 296_980,
    (200_000_000_000, 700): 586_260,
    (400_000_000_000, 1500): 297_780,
    (200_000_000_000, 1500): 587_060,
}

#: The baseline cell was observed while designing the fixture. It is a
#: preservation baseline, not a scored prediction.
BASELINE_CELL = (400_000_000_000, 700)

CONTRADICTIONS = (
    "C1_queued_is_submission_not_eligibility",
    "C2_dropped_submitted_event",
    "C3_duplicated_started_event",
    "C4_progress_bytes_disagree",
    "C5_wqe_completion_timestamp_disagrees",
    "C6_phantom_subject_object",
    "C7_eligibility_and_grant_swapped",
    "C8_class_service_bytes_invented",
    "D1_wqe_object_created_at_disagrees",
    "D2_wqe_object_bytes_disagree",
    "D3_report_wqe_absent_from_ledger",
    "D4_ledger_event_absent_from_result",
    "D5_completion_stage_timestamp_disagrees",
)


def expected_jct_ps(rate_bps: int, completion_delivery_ps: int) -> int:
    """Return the frozen closed form for one sweep cell."""

    total_bits = sum(WQE_PAYLOAD_BYTES) * 8
    serialization_ps = total_bits * PS_PER_SECOND // rate_bps
    return FIRST_GRANT_PS + serialization_ps + completion_delivery_ps


def check_only() -> int:
    """Validate the frozen table without importing SimLLM."""

    failures: list[str] = []
    for (rate_bps, delivery_ps), frozen in sorted(FROZEN_JCT_PS.items()):
        derived = expected_jct_ps(rate_bps, delivery_ps)
        if derived != frozen:
            failures.append(
                f"cell rate={rate_bps} delivery={delivery_ps}: "
                f"frozen {frozen} but closed form gives {derived}"
            )
    for delivery_ps in (700, 1500):
        slow = FROZEN_JCT_PS[(200_000_000_000, delivery_ps)]
        fast = FROZEN_JCT_PS[(400_000_000_000, delivery_ps)]
        if slow - fast != 289_280:
            failures.append(
                f"halving the rate at delivery={delivery_ps} must add exactly "
                f"289,280 ps, table gives {slow - fast}"
            )
    for rate_bps in (200_000_000_000, 400_000_000_000):
        slow = FROZEN_JCT_PS[(rate_bps, 1500)]
        fast = FROZEN_JCT_PS[(rate_bps, 700)]
        if slow - fast != 800:
            failures.append(
                f"adding 800 ps of delivery at rate={rate_bps} must add exactly "
                f"800 ps, table gives {slow - fast}"
            )
    if len(set(CONTRADICTIONS)) != len(CONTRADICTIONS):
        failures.append("registered contradiction IDs are not unique")
    if len(CONTRADICTIONS) != 13:
        failures.append(f"expected 13 registered contradictions, found {len(CONTRADICTIONS)}")
    if sum(WQE_PAYLOAD_BYTES) != 14_464:
        failures.append("registered WQE payload total disagrees with the freeze")

    for failure in failures:
        print(f"check-only failure: {failure}")
    if failures:
        return 1
    print("check-only: frozen table, closed form and both scaling relations agree")
    print(f"check-only: {len(CONTRADICTIONS)} registered contradictions")
    return 0


def _fixture(rate_bps: int, completion_delivery_ps: int):
    from simllm.core.execution import (
        CollectiveWork,
        ComputeWork,
        ControlWork,
        ExecutionGraph,
        ExecutionOperation,
        OperationCorrelation,
    )
    from simllm.core.runtime import CoarseDeviceProfile

    correlation = OperationCorrelation(request_ids=("request",))

    def transfer(name: str, queue: str, payload_bytes: int) -> ExecutionOperation:
        return ExecutionOperation(
            name,
            0,
            queue,
            CollectiveWork(
                "all-to-allv",
                (0, 8),
                0,
                "pairwise",
                pair_payload_bytes=((0, 8, payload_bytes),),
            ),
            correlation=correlation,
        )

    graph = ExecutionGraph(
        execution_id=FIXTURE_EXECUTION_ID,
        step_index=0,
        released_at_ps=0,
        operations=(
            ExecutionOperation(
                "compute-a",
                0,
                "cuda:0:compute",
                ComputeWork("a", nominal_duration_ps=100_000, hbm_bytes=1024),
                correlation=correlation,
            ),
            transfer("xfer", "cuda:0:nccl", 4096),
            transfer("xfer-fifo", "cuda:0:nccl", 2048),
            transfer("xfer-rival", "cuda:0:nccl-b", 8192),
            ExecutionOperation(
                "ctrl",
                0,
                "cuda:0:ctrl",
                ControlWork("sync", (8,), 128),
                correlation=correlation,
            ),
        ),
    )
    profile = CoarseDeviceProfile(
        rnic_rate_bps=rate_bps,
        launch_service_ps=1000,
        nccl_channel_service_ps=5000,
        control_service_ps=2000,
        completion_delivery_ps=completion_delivery_ps,
    )
    return graph, profile


def _step_record():
    from simllm.core.step import RequestPhase, ScheduledRequest, StepRecord

    return StepRecord(
        step_index=0,
        virtual_time_ps=0,
        scheduled=[ScheduledRequest("request", RequestPhase.PREFILL, 1)],
        num_sampled=1,
        sampled_request_ids=["request"],
    )


def _run_cell(rate_bps: int, completion_delivery_ps: int) -> dict:
    from simllm.core.authority import (
        check_bookkeeping_projection,
        check_completion_event_projection,
    )
    from simllm.core.bookkeeping import RequestBookkeeper
    from simllm.core.clock import VirtualClock
    from simllm.core.completion import CompletionReducer
    from simllm.core.runtime import CoarseDeviceRuntime

    graph, profile = _fixture(rate_bps, completion_delivery_ps)
    bookkeeper = RequestBookkeeper()
    runtime = CoarseDeviceRuntime(profile)
    result = runtime.execute(graph, bookkeeping=bookkeeper)
    report = runtime.last_report
    if report is None:
        raise AssertionError("coarse runtime returned no RuntimeReport")
    ledger = bookkeeper.snapshot()

    check_completion_event_projection(graph, result, report)
    check_bookkeeping_projection(ledger, graph, result, report)

    clock = VirtualClock(0)
    step_result = CompletionReducer(clock).reduce(_step_record(), graph, result, report)
    metric = step_result.request_metrics[0]

    gated = sum(
        1
        for visit in report.visits
        if visit.subject_object_id is None
        and visit.eligible_at_ps > visit.submitted_at_ps
    )
    contended = sum(
        1
        for visit in report.visits
        if visit.subject_object_id is None and visit.queue_wait_ps > 0
    )
    return {
        "rate_bps": rate_bps,
        "completion_delivery_ps": completion_delivery_ps,
        "completed_at_ps": result.completed_at_ps,
        "quiesced_at_ps": result.quiesced_at_ps,
        "step_latency_ps": step_result.step_latency_ps,
        "ttft_ps": metric.ttft_ps,
        "event_count": len(result.events),
        "visit_count": len(report.visits),
        "wqe_count": len(report.wqes),
        "ledger_entry_count": len(ledger.entries),
        "class_service_bytes": [list(pair) for pair in report.class_service_bytes],
        "gated_visit_count": gated,
        "contended_visit_count": contended,
    }


def _base_evidence(rate_bps: int, completion_delivery_ps: int):
    from simllm.core.bookkeeping import RequestBookkeeper
    from simllm.core.runtime import CoarseDeviceRuntime

    graph, profile = _fixture(rate_bps, completion_delivery_ps)
    bookkeeper = RequestBookkeeper()
    runtime = CoarseDeviceRuntime(profile)
    result = runtime.execute(graph, bookkeeping=bookkeeper)
    report = runtime.last_report
    if report is None:
        raise AssertionError("coarse runtime returned no RuntimeReport")
    return graph, result, report, bookkeeper.snapshot()


def _submit_events(graph, result, report, events) -> str:
    """Submit one event stream to the enforcing consumer, reporting its verdict."""

    from dataclasses import replace

    from simllm.core.clock import VirtualClock
    from simllm.core.completion import CompletionReducer

    mutated = replace(result, events=tuple(sorted(events, key=lambda e: e.timestamp_ps)))
    clock = VirtualClock(0)
    reducer = CompletionReducer(clock)
    try:
        reducer.reduce(_step_record(), graph, mutated, report)
    except Exception as error:  # noqa: BLE001 - the verdict is the evidence
        verdict = f"rejected: {type(error).__name__}: {error}"
    else:
        verdict = "ACCEPTED"
    if verdict != "ACCEPTED" and (clock.now_ps != 0 or reducer.latest_request_metrics):
        return "REFUSAL MUTATED CONSUMER STATE"
    return verdict


def _submit_ledger(ledger, graph, result, report) -> str:
    """Submit one ledger to the structural validator and the authority join."""

    from simllm.core.authority import check_bookkeeping_projection
    from simllm.core.bookkeeping import validate_bookkeeping_ledger

    try:
        validate_bookkeeping_ledger(ledger)
        check_bookkeeping_projection(ledger, graph, result, report)
    except Exception as error:  # noqa: BLE001
        return f"rejected: {type(error).__name__}: {error}"
    return "ACCEPTED"


def _relabelled(facts):
    from simllm.core.bookkeeping import BookkeepingEntry, BookkeepingLedger

    return BookkeepingLedger(
        tuple(BookkeepingEntry(index, fact) for index, fact in enumerate(facts))
    )


def _event_contradictions(graph, result, report) -> dict[str, str]:
    from dataclasses import replace

    from simllm.core.execution import EventPhase

    events = list(result.events)
    outcomes: dict[str, str] = {}

    gated = next(
        visit
        for visit in report.visits
        if visit.subject_object_id is None
        and visit.eligible_at_ps > visit.submitted_at_ps
    )
    contended = next(
        visit
        for visit in report.visits
        if visit.subject_object_id is None and visit.queue_wait_ps > 0
    )

    outcomes["C1_queued_is_submission_not_eligibility"] = _submit_events(
        graph,
        result,
        report,
        [
            replace(event, timestamp_ps=gated.submitted_at_ps)
            if (
                event.phase is EventPhase.QUEUED
                and event.subject_object_id is None
                and event.operation_id == gated.operation_id
                and event.resource == gated.resource
                and event.timestamp_ps == gated.eligible_at_ps
            )
            else event
            for event in events
        ],
    )

    dropped = next(
        event
        for event in events
        if event.phase is EventPhase.SUBMITTED and event.subject_object_id is None
    )
    outcomes["C2_dropped_submitted_event"] = _submit_events(
        graph, result, report, [event for event in events if event is not dropped]
    )

    duplicated = next(
        event
        for event in events
        if event.phase is EventPhase.STARTED and event.subject_object_id is None
    )
    outcomes["C3_duplicated_started_event"] = _submit_events(
        graph, result, report, [*events, duplicated]
    )

    heavier = []
    changed = False
    for event in events:
        if (
            not changed
            and event.phase is EventPhase.PROGRESS
            and event.subject_object_id is None
            and event.completed_bytes
        ):
            event = replace(event, completed_bytes=event.completed_bytes + 1)
            changed = True
        heavier.append(event)
    if not changed:
        raise AssertionError("fixture carries no byte-bearing PROGRESS event")
    outcomes["C4_progress_bytes_disagree"] = _submit_events(
        graph, result, report, heavier
    )

    wqe = report.wqes[0]
    outcomes["C5_wqe_completion_timestamp_disagrees"] = _submit_events(
        graph,
        result,
        report,
        [
            replace(event, timestamp_ps=event.timestamp_ps - 1)
            if (
                event.subject_object_id == wqe.wqe_id
                and event.phase is EventPhase.COMPLETED
            )
            else event
            for event in events
        ],
    )

    phantom = replace(
        next(event for event in events if event.subject_object_id == wqe.wqe_id),
        subject_object_id="wqe:phantom",
    )
    outcomes["C6_phantom_subject_object"] = _submit_events(
        graph, result, report, [*events, phantom]
    )

    swapped = []
    for event in events:
        if (
            event.subject_object_id is None
            and event.operation_id == contended.operation_id
            and event.resource == contended.resource
        ):
            if (
                event.phase is EventPhase.QUEUED
                and event.timestamp_ps == contended.eligible_at_ps
            ):
                event = replace(event, timestamp_ps=contended.started_at_ps)
            elif (
                event.phase is EventPhase.STARTED
                and event.timestamp_ps == contended.started_at_ps
            ):
                event = replace(event, timestamp_ps=contended.eligible_at_ps)
        swapped.append(event)
    outcomes["C7_eligibility_and_grant_swapped"] = _submit_events(
        graph, result, report, swapped
    )

    outcomes["C8_class_service_bytes_invented"] = _submit_events(
        graph,
        result,
        replace(report, class_service_bytes=((0, 4096),)),
        events,
    )

    outcomes["R0_unmutated_events"] = _submit_events(graph, result, report, events)
    return outcomes


def _ledger_contradictions(graph, result, report, ledger) -> dict[str, str]:
    from dataclasses import replace

    from simllm.core.bookkeeping import (
        CreatedObjectKind,
        CreatedObjectRecord,
        ProcessingStage,
        StagePhase,
        StageRecord,
    )
    from simllm.core.execution import CompletionEvent, EventPhase

    facts = [entry.fact for entry in ledger.entries]
    index = next(
        position
        for position, fact in enumerate(facts)
        if isinstance(fact, CreatedObjectRecord)
        and fact.ref.kind is CreatedObjectKind.NETWORK_WQE
    )
    outcomes: dict[str, str] = {}

    earlier = list(facts)
    earlier[index] = replace(
        earlier[index], created_at_ps=earlier[index].created_at_ps - 1
    )
    outcomes["D1_wqe_object_created_at_disagrees"] = _submit_ledger(
        _relabelled(earlier), graph, result, report
    )

    heavier = list(facts)
    heavier[index] = replace(
        heavier[index],
        metadata=tuple(
            (name, value + 1 if name == "bytes" else value)
            for name, value in heavier[index].metadata
        ),
    )
    outcomes["D2_wqe_object_bytes_disagree"] = _submit_ledger(
        _relabelled(heavier), graph, result, report
    )

    dropped_id = facts[index].ref.object_id
    without = [
        fact
        for fact in facts
        if not (
            isinstance(fact, CreatedObjectRecord) and fact.ref.object_id == dropped_id
        )
        and not (
            isinstance(fact, CompletionEvent) and fact.subject_object_id == dropped_id
        )
    ]
    outcomes["D3_report_wqe_absent_from_ledger"] = _submit_ledger(
        _relabelled(without), graph, result, report
    )

    extra = next(
        fact
        for fact in facts
        if isinstance(fact, CompletionEvent)
        and fact.subject_object_id is None
        and fact.phase is EventPhase.PROGRESS
    )
    outcomes["D4_ledger_event_absent_from_result"] = _submit_ledger(
        _relabelled([*facts, replace(extra, timestamp_ps=extra.timestamp_ps + 5)]),
        graph,
        result,
        report,
    )

    later = list(facts)
    for position, fact in enumerate(later):
        if (
            isinstance(fact, StageRecord)
            and fact.stage is ProcessingStage.COMPLETION
            and fact.phase is StagePhase.COMPLETED
        ):
            later[position] = replace(fact, timestamp_ps=fact.timestamp_ps + 1)
            break
    outcomes["D5_completion_stage_timestamp_disagrees"] = _submit_ledger(
        _relabelled(later), graph, result, report
    )

    outcomes["R1_unmutated_ledger"] = _submit_ledger(ledger, graph, result, report)
    return outcomes


def _rollback_guards(rate_bps: int, completion_delivery_ps: int) -> dict[str, str]:
    """Run a drifting runtime through the live path and check nothing moved."""

    from simllm.core.bookkeeping import RequestBookkeeper
    from simllm.core.runtime import CoarseDeviceRuntime

    class _DriftingRuntime(CoarseDeviceRuntime):
        """A runtime whose ledger claims one byte more than its WQE authority."""

        def _bookkeeping_objects(self, bookkeeper, graph, scheduled, wqes):
            from dataclasses import replace

            from simllm.core.bookkeeping import CreatedObjectKind

            records = super()._bookkeeping_objects(bookkeeper, graph, scheduled, wqes)
            drifted = []
            bumped = False
            for record in records:
                if not bumped and record.ref.kind is CreatedObjectKind.NETWORK_WQE:
                    record = replace(
                        record,
                        metadata=tuple(
                            (name, value + 1 if name == "bytes" else value)
                            for name, value in record.metadata
                        ),
                    )
                    bumped = True
                drifted.append(record)
            return tuple(drifted)

    graph, profile = _fixture(rate_bps, completion_delivery_ps)
    bookkeeper = RequestBookkeeper()
    runtime = _DriftingRuntime(profile)
    before = bookkeeper.snapshot()
    try:
        runtime.execute(graph, bookkeeping=bookkeeper)
    except ValueError:
        rolled_back = bookkeeper.snapshot().entries == before.entries
        return {
            "S1_ledger_unchanged_after_refusal": (
                "PRESERVED" if rolled_back else "LEDGER MUTATED BY A REFUSED APPEND"
            ),
        }
    return {"S1_ledger_unchanged_after_refusal": "DRIFTING RUNTIME WAS ACCEPTED"}


def _contradiction_outcomes() -> dict[str, str]:
    from simllm.core.clock import VirtualClock
    from simllm.core.completion import CompletionReducer
    from simllm.core.execution import EventPhase

    graph, result, report, ledger = _base_evidence(*BASELINE_CELL)
    outcomes = _event_contradictions(graph, result, report)
    outcomes.update(_ledger_contradictions(graph, result, report, ledger))

    from dataclasses import replace

    clock = VirtualClock(0)
    reducer = CompletionReducer(clock)
    refused = replace(
        result,
        events=tuple(
            event
            for event in result.events
            if not (
                event.phase is EventPhase.SUBMITTED and event.subject_object_id is None
            )
        ),
    )
    try:
        reducer.reduce(_step_record(), graph, refused, report)
    except ValueError:
        preserved = clock.now_ps == 0 and not reducer.latest_request_metrics
        outcomes["S0_clock_unchanged_after_refusal"] = (
            "PRESERVED" if preserved else "REFUSAL MOVED THE CLOCK OR A METRIC"
        )
    else:
        outcomes["S0_clock_unchanged_after_refusal"] = "REFUSED STREAM WAS ACCEPTED"
    outcomes.update(_rollback_guards(*BASELINE_CELL))
    return outcomes


def _provenance() -> dict[str, str]:
    import subprocess

    def capture(*command: str) -> str:
        try:
            return subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        except Exception:  # noqa: BLE001 - provenance is best effort
            return "unavailable"

    return {
        "observed_simllm_revision": capture("git", "rev-parse", "HEAD"),
        "authored_against_simllm_revision": "4e1be35af5327c27db53ed002dc420e1de6f613b",
    }


def run(out_root: Path) -> int:
    out_root.mkdir(parents=True, exist_ok=True)
    failures: list[str] = []

    cells = []
    matching_cells = 0
    for (rate_bps, delivery_ps), frozen in sorted(FROZEN_JCT_PS.items()):
        cell = _run_cell(rate_bps, delivery_ps)
        cell["frozen_jct_ps"] = frozen
        cell["scored"] = (rate_bps, delivery_ps) != BASELINE_CELL
        disagreements = [
            name
            for name in ("completed_at_ps", "quiesced_at_ps", "step_latency_ps", "ttft_ps")
            if cell[name] != frozen
        ]
        for name in disagreements:
            failures.append(
                f"cell rate={rate_bps} delivery={delivery_ps}: {name} "
                f"{cell[name]} disagrees with the frozen {frozen}"
            )
        if not disagreements:
            matching_cells += 1
        if cell["gated_visit_count"] <= 0 or cell["contended_visit_count"] <= 0:
            failures.append(
                f"cell rate={rate_bps} delivery={delivery_ps}: the fixture lost "
                "its discriminating visit shapes"
            )
        cells.append(cell)

    for delivery_ps in (700, 1500):
        slow = next(
            cell["completed_at_ps"]
            for cell in cells
            if cell["rate_bps"] == 200_000_000_000
            and cell["completion_delivery_ps"] == delivery_ps
        )
        fast = next(
            cell["completed_at_ps"]
            for cell in cells
            if cell["rate_bps"] == 400_000_000_000
            and cell["completion_delivery_ps"] == delivery_ps
        )
        if slow - fast != 289_280:
            failures.append(
                f"measured rate relation at delivery={delivery_ps} is {slow - fast} ps, "
                "not the registered 289,280 ps"
            )
    for rate_bps in (200_000_000_000, 400_000_000_000):
        slow = next(
            cell["completed_at_ps"]
            for cell in cells
            if cell["rate_bps"] == rate_bps and cell["completion_delivery_ps"] == 1500
        )
        fast = next(
            cell["completed_at_ps"]
            for cell in cells
            if cell["rate_bps"] == rate_bps and cell["completion_delivery_ps"] == 700
        )
        if slow - fast != 800:
            failures.append(
                f"measured delivery relation at rate={rate_bps} is {slow - fast} ps, "
                "not the registered 800 ps"
            )

    outcomes = _contradiction_outcomes()
    rejected = 0
    for name in CONTRADICTIONS:
        outcome = outcomes.get(name)
        if outcome is None:
            failures.append(f"contradiction {name} was not submitted")
        elif outcome == "ACCEPTED":
            failures.append(f"contradiction {name} is still accepted")
        else:
            rejected += 1
    for name in ("R0_unmutated_events", "R1_unmutated_ledger"):
        if outcomes.get(name) != "ACCEPTED":
            failures.append(f"reference {name} is no longer accepted: {outcomes.get(name)}")
    for name in ("S0_clock_unchanged_after_refusal", "S1_ledger_unchanged_after_refusal"):
        if outcomes.get(name) != "PRESERVED":
            failures.append(f"consumer state guard {name} failed: {outcomes.get(name)}")

    summary = {
        "provenance": _provenance(),
        "cells": cells,
        "contradictions": outcomes,
        "scored": {
            "CLA-B1_contradictions_rejected": f"{rejected}/{len(CONTRADICTIONS)}",
            "CLA-B2_sweep_cells_matching_frozen_jct": f"{matching_cells}/{len(cells)}",
            "CLA-B3_discriminating_shapes": {
                "gated_visits": cells[0]["gated_visit_count"],
                "contended_visits": cells[0]["contended_visit_count"],
            },
        },
        "failures": failures,
    }
    (out_root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("evidence class: run configuration")
    for cell in cells:
        print(
            f"  rate={cell['rate_bps'] // 1_000_000_000}G "
            f"delivery={cell['completion_delivery_ps']}ps "
            f"visits={cell['visit_count']} wqes={cell['wqe_count']} "
            f"events={cell['event_count']} ledger={cell['ledger_entry_count']}"
        )
    print("evidence class: exact-oracle rows")
    for cell in cells:
        role = "scored" if cell["scored"] else "preservation baseline"
        print(
            f"  rate={cell['rate_bps'] // 1_000_000_000}G "
            f"delivery={cell['completion_delivery_ps']}ps "
            f"jct={cell['completed_at_ps']} frozen={cell['frozen_jct_ps']} ({role})"
        )
    print("evidence class: rejected contradictions")
    for name in CONTRADICTIONS:
        print(f"  {name}: {outcomes.get(name)}")
    print("evidence class: structural invariants")
    print(f"  gated visits: {cells[0]['gated_visit_count']}")
    print(f"  contended visits: {cells[0]['contended_visit_count']}")
    for name in ("R0_unmutated_events", "R1_unmutated_ledger"):
        print(f"  {name}: {outcomes.get(name)}")
    for name in ("S0_clock_unchanged_after_refusal", "S1_ledger_unchanged_after_refusal"):
        print(f"  {name}: {outcomes.get(name)}")

    if failures:
        for failure in failures:
            print(f"FAILURE: {failure}")
        return 1
    print(f"wrote {out_root / 'summary.json'}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, help="directory for run artifacts")
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="validate the frozen table and exit without importing SimLLM",
    )
    args = parser.parse_args(argv)
    if args.check_only:
        return check_only()
    if args.out is None:
        parser.error("--out is required unless --check-only is given")
    return run(args.out)


if __name__ == "__main__":
    sys.exit(main())
