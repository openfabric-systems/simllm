"""Finite service-only oracles independent of the new completion implementation."""

from __future__ import annotations

from collections import deque

from examples.independent_engine_completion_v1.common import json_value


def native_reference(process, cell, services):
    """Derive request times from frozen service, arrival and ready-order rules."""
    p, d = process["prefill_engines"], process["decode_engines"]
    engines = [("prefill", index) for index in range(p)] + [("decode", index) for index in range(d)]
    queues = {engine: deque() for engine in engines}
    rows = [{"arrival": arrival, "prompt": prompt, "prefill_engine": ("prefill", index % p),
             "decode_engine": ("decode", index % d), "prefill_start": None, "prefill_end": None,
             "handoff_end": None, "decode_start": None, "tokens": []}
            for index, (arrival, prompt) in enumerate(zip(cell["arrival_offsets_ps"], cell["prompt_tokens"], strict=True))]
    admitted, transferred, pending, steps, advances = set(), set(), {}, [], []
    now = cursor = sequence = 0

    def finish(engine, at, index, start):
        row = rows[index]
        role, ordinal = engine
        steps.append({"engine_id": f"simllm-{role}-{ordinal}", "request_index": index,
                      "start_ps": start, "end_ps": at, "role": role})
        if role == "prefill":
            row["prefill_end"] = at
            row["handoff_end"] = at + cell["handoff_ps"]
            queues[engine].popleft()
        else:
            row["tokens"].append(at)
            if len(row["tokens"]) == cell["decode_output_tokens"]:
                queues[engine].popleft()

    while any(len(row["tokens"]) != cell["decode_output_tokens"] for row in rows):
        for engine, (end, order, index, start) in sorted(pending.copy().items(), key=lambda item: item[1][:2]):
            if end == now:
                finish(engine, end, index, start)
                del pending[engine]
        for index, row in enumerate(rows):
            if index not in admitted and row["arrival"] <= now:
                queues[row["prefill_engine"]].append(index)
                admitted.add(index)
        for index, row in enumerate(rows):
            if index not in transferred and row["handoff_end"] is not None and row["handoff_end"] <= now:
                queues[row["decode_engine"]].append(index)
                transferred.add(index)
        ready = [engine for engine in engines if queues[engine] and engine not in pending]
        if process["mode"] == "serialized":
            ready = sorted(ready, key=lambda engine: (engines.index(engine) - cursor) % len(engines))[:1]
        for engine in ready:
            index = queues[engine][0]
            row = rows[index]
            role = engine[0]
            service = services[str(row["prompt"])][role]
            if role == "decode":
                service = service[len(row["tokens"])] if isinstance(service, list) else service
            start_key = role + "_start"
            if row[start_key] is None:
                row[start_key] = now
            if process["mode"] == "serialized":
                finish(engine, now + service, index, now)
                advances.append({"before_ps": now, "after_ps": now + service})
                now += service
                cursor = (engines.index(engine) + 1) % len(engines)
            else:
                pending[engine] = (now + service, sequence, index, now)
                sequence += 1
        if ready and process["mode"] == "serialized":
            continue
        if all(len(row["tokens"]) == cell["decode_output_tokens"] for row in rows):
            break
        future = [value[0] for value in pending.values()]
        future += [row["arrival"] for index, row in enumerate(rows) if index not in admitted and row["arrival"] > now]
        future += [row["handoff_end"] for index, row in enumerate(rows)
                   if index not in transferred and row["handoff_end"] is not None and row["handoff_end"] > now]
        if not future:
            raise ValueError("frozen reference has unfinished work without a causal event")
        advances.append({"before_ps": now, "after_ps": min(future)})
        now = min(future)
    return json_value({"requests": rows, "service_steps": steps, "makespan_ps": now, "clock_advances": advances})


def component_run(spec):
    """Run each frozen component configuration through the actual core owner."""
    from dataclasses import asdict

    from simllm.core import RequestPhase, ScheduledRequest, StepRecord, StepResult, VirtualClock
    from simllm.core.engine_steps import EngineStepRuntime
    from simllm.core.step import step_record_to_json

    runtime = EngineStepRuntime(VirtualClock())
    output, jobs = [], []
    if spec["kind"] == "equal-work":
        jobs = [{"id": str(index), "engine": str(index) if spec["resource_mode"] == "independent" else "serial",
                 "arrival_ps": 0, "service_ps": spec["service_ps"]} for index in range(spec["work_items"])]
    elif spec["kind"] == "unequal-arrival":
        jobs = [{"id": "long", "engine": "long", "arrival_ps": 0, "service_ps": spec["long_service_ps"]},
                {"id": "short", "engine": "short", "arrival_ps": spec["short_arrival_ps"], "service_ps": spec["short_service_ps"]}]
    elif spec["kind"] == "tie":
        jobs = spec["jobs"]
    elif spec["kind"] == "drain":
        jobs = [{"id": "drain", "engine": "drain", "arrival_ps": spec["at_ps"], "service_ps": 0}]
    else:
        raise ValueError("unknown frozen component kind")
    submitted, before_due = set(), set()
    receipts, inputs, checkpoints = [], [], []

    def snapshot(phase, receipt):
        checkpoints.append({"phase": phase, "sequence": receipt.sequence, "at_ps": runtime.clock.now_ps,
                            "outputs": [dict(row) for row in output],
                            "event_count": len(runtime.events), "visit_count": len(runtime.visits),
                            "result_sequences": [item.sequence for item, result in runtime.results]})

    def pending_receipts():
        return [receipt for receipt in receipts if runtime.pending_for(receipt.engine_id) is receipt]
    while len(output) < len(jobs):
        for receipt in sorted(pending_receipts(), key=lambda row: (row.completed_at_ps, row.sequence)):
            if receipt.completed_at_ps == runtime.clock.now_ps:
                if receipt.sequence not in before_due:
                    snapshot("before-due", receipt)
                    before_due.add(receipt.sequence)
                runtime.complete(receipt)
                snapshot("retired", receipt)
        for index, job in enumerate(jobs):
            if index in submitted or job["arrival_ps"] > runtime.clock.now_ps or runtime.pending_for(job["engine"]) is not None:
                continue
            record = StepRecord(index, runtime.clock.now_ps,
                                [] if spec["kind"] == "drain" else [ScheduledRequest(job["id"], RequestPhase.DECODE, 1)],
                                finished_request_ids=spec.get("finished_request_ids", []))
            result = StepResult(index, job["service_ps"], runtime.clock.now_ps + job["service_ps"])
            receipt = runtime.submit(job["engine"], record, result, guard=lambda: None,
                           publish=lambda result, job=job: output.append({"id": job["id"], "at_ps": result.completed_at_ps}))
            receipts.append(receipt)
            inputs.append({"job_id": job["id"], "engine_id": job["engine"], "record": step_record_to_json(record),
                           "result": asdict(result), "receipt": asdict(receipt)})
            snapshot("submitted", receipt)
            submitted.add(index)
            if runtime.next_completion_ps == runtime.clock.now_ps:
                break
        if len(output) == len(jobs):
            break
        future = [job["arrival_ps"] for index, job in enumerate(jobs) if index not in submitted and job["arrival_ps"] > runtime.clock.now_ps]
        if runtime.next_completion_ps is not None:
            future.append(runtime.next_completion_ps)
        if not future:
            raise ValueError("component has no progress event")
        next_time = min(future)
        if runtime.next_completion_ps == next_time and next_time > runtime.clock.now_ps:
            if next_time - 1 > runtime.clock.now_ps:
                runtime.advance_to(next_time - 1)
            for receipt in pending_receipts():
                if receipt.completed_at_ps == next_time:
                    snapshot("before-due", receipt)
                    before_due.add(receipt.sequence)
            if runtime.complete_due():
                raise ValueError("component published before its due event")
        runtime.advance_to(next_time)
    value = {"id": spec["id"], "outputs": output, "end_ps": runtime.clock.now_ps,
             "inputs": inputs, "checkpoints": checkpoints,
             "completed": [{"receipt": asdict(receipt), "result": asdict(result)} for receipt, result in runtime.results],
             "visits": [asdict(row) for row in runtime.visits], "events": [asdict(row) for row in runtime.events],
             "failure": runtime.failure}
    runtime.close()
    return json_value(value)
