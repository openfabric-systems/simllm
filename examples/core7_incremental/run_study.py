"""Measure CORE-7 append scaling against the former full-ledger path."""

from __future__ import annotations

import argparse
import gc
import json
import platform
import statistics
import sys
import time
from collections.abc import Callable
from itertools import pairwise
from pathlib import Path
from typing import Any

from simllm.core.bookkeeping import (
    BookkeepingEntry,
    BookkeepingFact,
    BookkeepingLedger,
    BookkeepingScope,
    CreatedObjectKind,
    CreatedObjectRecord,
    CreatedObjectRef,
    ObjectOwner,
    ProcessingStage,
    RequestBookkeeper,
    StagePhase,
    StageRecord,
    validate_bookkeeping_ledger,
)
from simllm.core.execution import (
    CompletionEvent,
    EventPhase,
    OperationCorrelation,
    ResourceKind,
    ResourceRef,
)

EXPECTATIONS_COMMIT = "d487a69"
INCREMENTAL_SIZES = (1_000, 4_000, 16_000)
REFERENCE_SIZES = (1_000, 2_000, 4_000)
INCREMENTAL_REPETITIONS = 5
REFERENCE_REPETITIONS = 3


def _stage_facts(count: int) -> tuple[BookkeepingFact, ...]:
    return tuple(
        StageRecord(
            ProcessingStage.REQUEST,
            StagePhase.ENTERED,
            index,
            BookkeepingScope(
                correlation=OperationCorrelation(request_ids=(f"request:{index}",))
            ),
        )
        for index in range(count)
    )


def _wqe_facts(count: int) -> tuple[BookkeepingFact, ...]:
    sq = CreatedObjectRecord(
        CreatedObjectRef(CreatedObjectKind.SEND_QUEUE, "sq:shared"),
        ObjectOwner.DEVICE_RUNTIME,
        0,
    )
    rq = CreatedObjectRecord(
        CreatedObjectRef(CreatedObjectKind.RECEIVE_QUEUE, "rq:shared"),
        ObjectOwner.DEVICE_RUNTIME,
        0,
    )
    cq = CreatedObjectRecord(
        CreatedObjectRef(CreatedObjectKind.COMPLETION_QUEUE, "cq:shared"),
        ObjectOwner.DEVICE_RUNTIME,
        0,
    )
    facts: list[BookkeepingFact] = [sq, rq, cq]
    cycle = 0
    while len(facts) < count:
        timestamp = 100 + cycle * 16
        execution_id = f"execution:{cycle}"
        operation_id = f"operation:{cycle}"
        scope = BookkeepingScope(
            correlation=OperationCorrelation(request_ids=(f"request:{cycle}",)),
            step_index=cycle,
            execution_id=execution_id,
            operation_id=operation_id,
        )
        operation = CreatedObjectRecord(
            CreatedObjectRef(
                CreatedObjectKind.EXECUTION_OPERATION,
                f"operation-object:{cycle}",
            ),
            ObjectOwner.CORE,
            timestamp,
            scope,
        )
        nccl = CreatedObjectRecord(
            CreatedObjectRef(CreatedObjectKind.NCCL_COMMAND, f"nccl:{cycle}"),
            ObjectOwner.NCCL,
            timestamp + 1,
            scope,
            parent_refs=(operation.ref,),
        )
        wqe = CreatedObjectRecord(
            CreatedObjectRef(CreatedObjectKind.NETWORK_WQE, f"wqe:{cycle}"),
            ObjectOwner.DEVICE_RUNTIME,
            timestamp + 2,
            scope,
            parent_refs=(nccl.ref, sq.ref, rq.ref, cq.ref),
            metadata=(("bytes", 4096), ("transport_kind", "none")),
        )
        send_resource = ResourceRef(ResourceKind.NIC_SEND_QUEUE, sq.ref.object_id)
        facts.extend(
            (
                operation,
                nccl,
                wqe,
                StageRecord(
                    ProcessingStage.NETWORK,
                    StagePhase.ENTERED,
                    timestamp + 3,
                    scope,
                    (wqe.ref,),
                ),
                CompletionEvent(
                    execution_id,
                    operation_id,
                    EventPhase.SUBMITTED,
                    timestamp + 4,
                    send_resource,
                    subject_object_id=wqe.ref.object_id,
                ),
                CompletionEvent(
                    execution_id,
                    operation_id,
                    EventPhase.QUEUED,
                    timestamp + 5,
                    send_resource,
                    subject_object_id=wqe.ref.object_id,
                ),
                CompletionEvent(
                    execution_id,
                    operation_id,
                    EventPhase.STARTED,
                    timestamp + 6,
                    send_resource,
                    subject_object_id=wqe.ref.object_id,
                ),
                CompletionEvent(
                    execution_id,
                    operation_id,
                    EventPhase.COMPLETED,
                    timestamp + 7,
                    ResourceRef(ResourceKind.COMPLETION_QUEUE, cq.ref.object_id),
                    completed_bytes=4096,
                    subject_object_id=wqe.ref.object_id,
                ),
            )
        )
        cycle += 1
    return tuple(facts[:count])


def _append_incrementally(facts: tuple[BookkeepingFact, ...]) -> int:
    bookkeeper = RequestBookkeeper()
    for fact in facts:
        bookkeeper.append(fact)
    return len(bookkeeper.snapshot().entries)


def _append_with_full_candidates(facts: tuple[BookkeepingFact, ...]) -> int:
    entries: list[BookkeepingEntry] = []
    for fact in facts:
        entry = BookkeepingEntry(len(entries), fact)
        candidate = BookkeepingLedger((*entries, entry))
        validate_bookkeeping_ledger(candidate)
        entries.append(entry)
    return len(entries)


def _measure(
    function: Callable[[tuple[BookkeepingFact, ...]], int],
    facts: tuple[BookkeepingFact, ...],
) -> tuple[int, int]:
    gc.collect()
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        started_ns = time.perf_counter_ns()
        ledger_length = function(facts)
        elapsed_ns = time.perf_counter_ns() - started_ns
    finally:
        if gc_was_enabled:
            gc.enable()
    return elapsed_ns, ledger_length


def _run_sweep(
    *,
    mode: str,
    function: Callable[[tuple[BookkeepingFact, ...]], int],
    fact_streams: dict[str, tuple[BookkeepingFact, ...]],
    sizes: tuple[int, ...],
    repetitions: int,
) -> list[dict[str, int | str]]:
    for fact_mix, stream in fact_streams.items():
        for size in sizes:
            _, ledger_length = _measure(function, stream[:size])
            if ledger_length != size:
                raise AssertionError(
                    f"{mode}/{fact_mix}/{size}: ledger length {ledger_length}"
                )

    measurements: list[dict[str, int | str]] = []
    for repetition in range(repetitions):
        for fact_mix, stream in fact_streams.items():
            for size in sizes:
                elapsed_ns, ledger_length = _measure(function, stream[:size])
                if ledger_length != size:
                    raise AssertionError(
                        f"{mode}/{fact_mix}/{size}: ledger length {ledger_length}"
                    )
                measurements.append(
                    {
                        "mode": mode,
                        "fact_mix": fact_mix,
                        "size": size,
                        "repetition": repetition,
                        "elapsed_ns": elapsed_ns,
                        "ledger_length": ledger_length,
                    }
                )
    return measurements


def _median_rows(
    measurements: list[dict[str, int | str]],
) -> list[dict[str, int | str]]:
    keys = sorted(
        {
            (str(row["mode"]), str(row["fact_mix"]), int(row["size"]))
            for row in measurements
        }
    )
    rows: list[dict[str, int | str]] = []
    for mode, fact_mix, size in keys:
        values = [
            int(row["elapsed_ns"])
            for row in measurements
            if row["mode"] == mode
            and row["fact_mix"] == fact_mix
            and row["size"] == size
        ]
        rows.append(
            {
                "mode": mode,
                "fact_mix": fact_mix,
                "size": size,
                "median_elapsed_ns": int(statistics.median(values)),
            }
        )
    return rows


def _median_lookup(
    medians: list[dict[str, int | str]],
    mode: str,
    fact_mix: str,
    size: int,
) -> int:
    return next(
        int(row["median_elapsed_ns"])
        for row in medians
        if row["mode"] == mode
        and row["fact_mix"] == fact_mix
        and row["size"] == size
    )


def _relations(medians: list[dict[str, int | str]]) -> list[dict[str, Any]]:
    relations: list[dict[str, Any]] = []
    for fact_mix in ("stage", "wqe"):
        for smaller, larger in pairwise(INCREMENTAL_SIZES):
            ratio = _median_lookup(
                medians, "incremental", fact_mix, larger
            ) / _median_lookup(medians, "incremental", fact_mix, smaller)
            relations.append(
                {
                    "relation": "incremental_quadrupling",
                    "fact_mix": fact_mix,
                    "smaller_size": smaller,
                    "larger_size": larger,
                    "ratio": ratio,
                    "bound": 6.0,
                    "passed": ratio <= 6.0,
                }
            )
        reference_ratio = _median_lookup(
            medians, "reference", fact_mix, 4_000
        ) / _median_lookup(medians, "reference", fact_mix, 1_000)
        relations.append(
            {
                "relation": "reference_endpoint_growth",
                "fact_mix": fact_mix,
                "smaller_size": 1_000,
                "larger_size": 4_000,
                "ratio": reference_ratio,
                "bound": 8.0,
                "passed": reference_ratio >= 8.0,
            }
        )
    return relations


def run(out_dir: Path) -> dict[str, Any]:
    maximum_size = max(*INCREMENTAL_SIZES, *REFERENCE_SIZES)
    fact_streams = {
        "stage": _stage_facts(maximum_size),
        "wqe": _wqe_facts(maximum_size),
    }
    generated_streams_reference_valid = True
    generated_stream_lengths_match = True
    for stream in fact_streams.values():
        try:
            validate_bookkeeping_ledger(_ledger_from_facts(stream))
        except (TypeError, ValueError):
            generated_streams_reference_valid = False
        generated_stream_lengths_match &= len(stream) == maximum_size
    if not generated_streams_reference_valid:
        raise AssertionError("a generated timing stream failed reference validation")
    if not generated_stream_lengths_match:
        raise AssertionError("a generated timing stream has the wrong length")

    incremental_fact_streams = fact_streams
    reference_fact_streams = fact_streams
    same_fact_tuples_used_by_both_modes = all(
        incremental_fact_streams[fact_mix] is reference_fact_streams[fact_mix]
        for fact_mix in fact_streams
    )

    measurements = _run_sweep(
        mode="incremental",
        function=_append_incrementally,
        fact_streams=incremental_fact_streams,
        sizes=INCREMENTAL_SIZES,
        repetitions=INCREMENTAL_REPETITIONS,
    )
    measurements.extend(
        _run_sweep(
            mode="reference",
            function=_append_with_full_candidates,
            fact_streams=reference_fact_streams,
            sizes=REFERENCE_SIZES,
            repetitions=REFERENCE_REPETITIONS,
        )
    )
    medians = _median_rows(measurements)
    relations = _relations(medians)
    all_ledger_lengths_equal_requested_size = all(
        int(row["ledger_length"]) == int(row["size"]) for row in measurements
    )
    structural_checks = {
        "generated_streams_reference_valid": generated_streams_reference_valid,
        "generated_stream_lengths_match": generated_stream_lengths_match,
        "same_fact_tuples_used_by_both_modes": same_fact_tuples_used_by_both_modes,
        "all_ledger_lengths_equal_requested_size": all_ledger_lengths_equal_requested_size,
    }
    report = {
        "expectations_commit": EXPECTATIONS_COMMIT,
        "clock": "time.perf_counter_ns",
        "host": {
            "python": sys.version,
            "platform": platform.platform(),
            "processor": platform.processor(),
        },
        "configuration": {
            "incremental_sizes": INCREMENTAL_SIZES,
            "reference_sizes": REFERENCE_SIZES,
            "incremental_repetitions": INCREMENTAL_REPETITIONS,
            "reference_repetitions": REFERENCE_REPETITIONS,
            "warmup_sweeps": 1,
            "garbage_collection_during_measurement": False,
        },
        "measurements": measurements,
        "medians": medians,
        "relations": relations,
        "structural_checks": structural_checks,
        "passed": all(bool(relation["passed"]) for relation in relations)
        and all(structural_checks.values()),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / "measurements.json"
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def _ledger_from_facts(facts: tuple[BookkeepingFact, ...]) -> BookkeepingLedger:
    return BookkeepingLedger(
        tuple(BookkeepingEntry(index, fact) for index, fact in enumerate(facts))
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.out)
    summary = {
        "expectations_commit": report["expectations_commit"],
        "medians": report["medians"],
        "relations": report["relations"],
        "passed": report["passed"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
