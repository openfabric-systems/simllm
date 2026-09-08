"""Replay the frozen coarse receiver study against baseline or candidate code."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter, defaultdict
from dataclasses import asdict
from enum import Enum
from fractions import Fraction
from itertools import pairwise
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASELINE_COMMIT = "429acf0fbd733006c8978d00cf18c0090584dc86"
MECHANISM_FREEZE = "5de4db26abb64188e27b56c3e3700588cc0df03d"
PRE_RUN_CLARIFICATION = "4226cb3ec398d8536c8f4ff61ce7cedf1715a865"
INPUT_IDENTITY_CLARIFICATION = "859af6a03e8929c715e97573f9b97290055c920c"
RATES = (200, 400)


def json_default(value):
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Fraction):
        return {"numerator": value.numerator, "denominator": value.denominator}
    raise TypeError(f"unsupported canonical value {type(value).__name__}")


def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       default=json_default) + "\n").encode()


def digest(value):
    return hashlib.sha256(value).hexdigest()


def write_once(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(canonical(value))


def cases():
    result = []

    def add(family, pattern, width, payload, rate, channel=0):
        result.append({"family": family, "pattern": pattern, "width": width,
                       "payload_bytes": payload, "rate_gbps": rate,
                       "channel_service_ps": channel,
                       "name": f"{family}-{pattern}-w{width}-b{payload}-r{rate}-c{channel}"})

    for pattern in ("combine", "dispatch", "duplex"):
        for fan_in in (1, 2, 4, 8):
            for payload in (3, 4096, 65536):
                for rate in RATES:
                    add("main", pattern, fan_in, payload, rate)
    for width in (2, 4):
        for payload in (4, 4096):
            for rate in RATES:
                for channel in (0, 7000):
                    add("ring", "ring", width, payload, rate, channel)
    for rate in RATES:
        add("sentinel", "ring", 4, 3, rate)
        for pattern in ("combine", "dispatch"):
            add("asymmetric", pattern, 2, 0, rate)
    for width in (2, 4):
        for payload in (3, 4096):
            for rate in RATES:
                add("complete", "all-to-all", width, payload, rate)
    return result


def traffic(case):
    width, payload, pattern = case["width"], case["payload_bytes"], case["pattern"]
    if case["family"] == "asymmetric":
        pairs = ((0, 8, 3), (0, 16, 5))
        if pattern == "combine":
            pairs = tuple((dst, src, size) for src, dst, size in pairs)
        return (0, 8, 16, 24), pairs
    if pattern in ("combine", "dispatch"):
        ranks = tuple(8*i for i in range(width+1))
        pairs = tuple((rank, 0, payload) if pattern == "combine" else (0, rank, payload)
                      for rank in ranks[1:])
    elif pattern == "duplex":
        ranks = tuple(8*i for i in range(2*width))
        pairs = tuple(pair for left, right in zip(ranks[::2], ranks[1::2], strict=True)
                      for pair in ((left, right, payload), (right, left, payload)))
    elif pattern == "all-to-all":
        ranks = tuple(8*i for i in range(width))
        pairs = tuple((src, dst, payload) for src in ranks for dst in ranks if src != dst)
    else:
        ranks = tuple(8*i for i in range(width))
        chunk = max(1, payload//width)
        pairs = tuple((src, ranks[(index+1) % width], chunk)
                      for _ in range(2*(width-1)) for index, src in enumerate(ranks))
    return ranks, pairs


def serialization(byte_count, rate_gbps):
    numerator = byte_count*8000
    return (numerator + rate_gbps - 1)//rate_gbps


def bounds(case):
    ranks, pairs = traffic(case)
    tx, rx = Counter(), Counter()
    for src, dst, size in pairs:
        tx[src] += size
        rx[dst] += size
    rate, channel = case["rate_gbps"], case["channel_service_ps"]
    wire = [serialization(size, rate) for _, _, size in pairs]
    floor = serialization(max((*tx.values(), *rx.values())), rate)
    ceiling = sum(wire) + len(pairs)*channel
    d = serialization(case["payload_bytes"], rate)
    if case["pattern"] == "ring":
        rounds = 2*(len(ranks)-1)
        expected_off = expected_on = rounds*(wire[0]+channel)
    elif case["family"] == "asymmetric":
        expected_on = sum(wire)
        expected_off = max(wire) if case["pattern"] == "combine" else sum(wire)
    elif case["pattern"] == "combine":
        expected_off, expected_on = d, case["width"]*d
    elif case["pattern"] == "dispatch":
        expected_off = expected_on = case["width"]*d
    elif case["pattern"] == "duplex":
        expected_off = expected_on = d
    else:
        expected_off = (case["width"]-1)*d
        expected_on = (1 if case["width"] == 2 else 5)*d
    return {"phase_floor_ps": floor, "fixture_ceiling_ps": ceiling,
            "unconditional_ceiling_ps": None,
            "expected_disabled_ps": expected_off, "expected_enabled_ps": expected_on,
            "tx_bytes_per_step": dict(sorted(tx.items())),
            "rx_bytes_per_step": dict(sorted(rx.items())),
            "expected_wqes_per_step": len(pairs), "directed_bytes_per_step": sum(tx.values())}


def make_graph(case, step_index, released_at_ps):
    from simllm.core import CollectiveWork, ExecutionGraph, ExecutionOperation, OperationCorrelation
    from simllm.traffic import plan_execution_graph_collectives

    ranks, pairs = traffic(case)
    if case["pattern"] == "ring":
        work = CollectiveWork("all-reduce", ranks, case["payload_bytes"], "ring",
                              channel_hint="tp")
    else:
        work = CollectiveWork("all-to-allv", ranks, 0, "pairwise", channel_hint="ep",
                              pair_payload_bytes=pairs)
    operation = ExecutionOperation(
        "transfer", ranks[0], "cuda:0:nccl:study", work,
        correlation=OperationCorrelation(request_ids=("request",), batch_id="batch", layer=0))
    return plan_execution_graph_collectives(ExecutionGraph(
        f"{case['name']}-step-{step_index}", step_index, released_at_ps,
        (operation,), (operation.operation_id,)))


def observe(case, mode, output):
    from simllm.core import (
        CoarseDeviceProfile,
        CoarseDeviceRuntime,
        CompletionReducer,
        RequestBookkeeper,
        RequestPhase,
        ScheduledRequest,
        StepRecord,
        VirtualClock,
        bookkeeping_ledger_to_json,
        execution_graph_to_json,
    )
    from simllm.traffic import render_serial_execution_graph_goal

    profile = CoarseDeviceProfile(rnic_rate_bps=case["rate_gbps"]*10**9,
                                  nccl_channel_service_ps=case["channel_service_ps"])
    options = {} if mode == "baseline" else {"receiver_ingress": mode == "enabled"}
    runtime = CoarseDeviceRuntime(profile, **options)
    clock = VirtualClock(0)
    reducer, bookkeeper = CompletionReducer(clock), RequestBookkeeper()
    snapshot, steps, graph_hashes, goal_hashes, workload_hashes = [], [], [], [], []
    failure = None
    for index in range(3):
        try:
            record = StepRecord(index, clock.now_ps, [ScheduledRequest(
                "request", RequestPhase.PREFILL if index == 0 else RequestPhase.DECODE,
                1, context_length=index+1)], num_sampled=1, sampled_request_ids=["request"])
            graph = make_graph(case, index, clock.now_ps)
            graph_json = execution_graph_to_json(graph)
            goal = render_serial_execution_graph_goal(graph).render().encode()
            streamed = []
            result = runtime.execute(graph, on_event=streamed.append, bookkeeping=bookkeeper)
            report = runtime.last_report
            step = reducer.reduce(record, graph, result, report)
            snapshot.append({"record": asdict(record), "graph": graph_json,
                             "result": asdict(result), "report": asdict(report),
                             "step": asdict(step), "streamed": [asdict(e) for e in streamed],
                             "bookkeeping": bookkeeping_ledger_to_json(bookkeeper.snapshot())})
            graph_hashes.append(digest(canonical(graph_json)))
            workload_hashes.append(digest(canonical({**graph_json, "released_at_ps": 0})))
            goal_hashes.append(digest(goal))
            write_once(output / f"step-{index}-graph.json", graph_json)
            (output / f"step-{index}.goal").write_bytes(goal)
            receiver_visits = [v for v in report.visits if v.stage == "coarse_receiver_service"]
            source_visits = [v for v in report.visits if v.stage == "coarse_source_admission"]
            steps.append({"latency_ps": step.step_latency_ps,
                          "released_at_ps": graph.released_at_ps,
                          "completed_at_ps": result.completed_at_ps,
                          "ttft_ps": step.request_metrics[0].ttft_ps,
                          "tpot_ps": step.request_metrics[0].tpot_ps,
                          "stream_matches_result": tuple(streamed) == result.events,
                          "wqes": [asdict(w) for w in report.wqes],
                          "receiver_visits": [asdict(v) for v in receiver_visits],
                          "source_visits": [asdict(v) for v in source_visits],
                          "sum_receiver_wait_ps": sum(v.queue_wait_ps for v in receiver_visits),
                          "sum_source_wait_ps": sum(v.queue_wait_ps for v in source_visits),
                          "critical_queue_ps": report.critical_path_queue_ps,
                          "critical_breakdown": asdict(report.operations[0].breakdown),
                          "request_attribution": asdict(step.request_metrics[0].attribution),
                          "event_count": len(result.events)})
        except Exception as error:  # noqa: BLE001 - retain every failed execution as evidence
            failure = f"{type(error).__name__}: {error}"
            break
    write_once(output / "snapshot.json", snapshot)
    ledgers = {}
    for name in ("source_byte_ledger", "receiver_byte_ledger",
                 "source_port_reservations", "receiver_port_reservations"):
        value = getattr(runtime.bypass_ledger, name, None)
        ledgers[name] = ([asdict(v) for v in value] if value is not None and
                         name.endswith("reservations") else value)
    row = {"case": case, "mode": mode, "status": "failed" if failure else "complete",
           "error": failure, "snapshot_sha256": digest((output / "snapshot.json").read_bytes()),
           "graph_sha256": graph_hashes, "goal_sha256": goal_hashes,
           "workload_graph_sha256": workload_hashes,
           "steps": steps, "endpoint_projections": ledgers,
           "ttft_ps": steps[-1]["ttft_ps"] if not failure else None,
           "tpot_ps": steps[-1]["tpot_ps"] if not failure else None,
           "jct_ps": clock.now_ps if not failure else None}
    write_once(output / "observations.json", row)
    return json.loads(canonical(row))


def integral(value):
    if isinstance(value, dict) and set(value) == {"numerator", "denominator"}:
        return Fraction(value["numerator"], value["denominator"])
    return value


def physical_signature(row):
    fields = ("wqe_id", "source_rank", "destination_rank", "payload_bytes",
              "started_at_ps", "finished_at_ps", "completed_at_ps")
    return [tuple(tuple(w[name] for name in fields) for w in step["wqes"])
            for step in row["steps"]]


def apply_guards(row, declared, baseline=None, disabled=None):
    findings, oracles = [], []
    case, mode = row["case"], row["mode"]
    enabled = mode == "enabled"
    expected = declared["expected_enabled_ps" if enabled else "expected_disabled_ps"]
    if row["status"] != "complete" or len(row["steps"]) != 3:
        findings.append("incomplete three-step request")
        return {**row, "fatal_findings": findings, "exact_oracles": oracles}
    if mode == "disabled":
        same = baseline is not None and row["snapshot_sha256"] == baseline["snapshot_sha256"]
        oracles.append({"family": "legacy_snapshot_identity", "matches": same})
        if baseline is None:
            findings.append("disabled configuration has no immutable baseline evidence")
        elif not same:
            findings.append("disabled snapshot differs from immutable baseline")
    if enabled and disabled is not None:
        if row["goal_sha256"] != disabled["goal_sha256"]:
            findings.append("receiver selection changed GOAL bytes")
        if row["workload_graph_sha256"] != disabled["workload_graph_sha256"]:
            findings.append("receiver selection changed the normalized workload graph")
        if declared["expected_enabled_ps"] == declared["expected_disabled_ps"]:
            if physical_signature(row) != physical_signature(disabled):
                findings.append("protected enabled physical timestamps changed")
            if row["graph_sha256"] != disabled["graph_sha256"]:
                findings.append("identity graph bytes changed")
    previous = 0
    _, pairs = traffic(case)
    expected_population = Counter(pairs)
    all_wqes = []
    for step in row["steps"]:
        if not step["stream_matches_result"] or step["released_at_ps"] != previous:
            findings.append("stream or predecessor-release mismatch")
        previous = step["completed_at_ps"]
        all_wqes.extend(step["wqes"])
        if Counter((w["source_rank"], w["destination_rank"], w["payload_bytes"])
                   for w in step["wqes"]) != expected_population:
            findings.append("input extent population or byte mismatch")
        if len({w["wqe_id"] for w in step["wqes"]}) != len(step["wqes"]):
            findings.append("duplicate WQE identity")
        if enabled and not declared["phase_floor_ps"] <= step["latency_ps"] <= declared["fixture_ceiling_ps"]:
            findings.append("enabled duration violates a physical bound")
        if step["latency_ps"] != expected:
            findings.append("exact fixture latency differs from its frozen relation")
        for w in step["wqes"]:
            if w["finished_at_ps"]-w["started_at_ps"] != serialization(w["payload_bytes"], case["rate_gbps"]):
                findings.append("wire service disagrees with bytes over rate")
            if not w["eligible_at_ps"] <= w["started_at_ps"] <= w["finished_at_ps"] <= w["completed_at_ps"]:
                findings.append("noncausal WQE timestamps")
        if enabled:
            if len(step["receiver_visits"]) != len(pairs) or len(step["source_visits"]) != len(pairs):
                findings.append("missing or duplicated admission visit")
            for visits in (step["receiver_visits"], step["source_visits"]):
                if len({v["subject_object_id"] for v in visits}) != len(pairs):
                    findings.append("admission visit identity mismatch")
            if case["family"] == "main" and case["pattern"] == "combine":
                d, fan_in = serialization(case["payload_bytes"], case["rate_gbps"]), case["width"]
                if step["sum_receiver_wait_ps"] != fan_in*(fan_in-1)*d//2:
                    findings.append("additive receiver wait does not conserve visits")
                if step["critical_queue_ps"] != (fan_in-1)*d:
                    findings.append("critical receiver wait was duplicated or lost")
                if step["critical_breakdown"]["service_ps"] != d:
                    findings.append("joint wire service was counted more than once")
    if enabled:
        for direction, rank_key in (("source", "source_rank"), ("receiver", "destination_rank")):
            expected_bytes = Counter()
            intervals = defaultdict(list)
            for w in all_wqes:
                rank = w[rank_key]
                rnic = f"node-{rank//8}:rnic-{rank%8}"
                expected_bytes[rnic] += w["payload_bytes"]
                intervals[rnic].append((w["started_at_ps"], w["finished_at_ps"]))
            observed = dict(row["endpoint_projections"][f"{direction}_byte_ledger"] or ())
            if observed != expected_bytes:
                findings.append(f"{direction} endpoint bytes differ from WQE authority")
            reservations = row["endpoint_projections"][f"{direction}_port_reservations"] or []
            observed_reservations = Counter((v["wqe_id"], v["rnic_id"], v["started_at_ps"],
                                             v["finished_at_ps"], v["service_bytes"]) for v in reservations)
            expected_reservations = Counter((w["wqe_id"], f"node-{w[rank_key]//8}:rnic-{w[rank_key]%8}",
                                             w["started_at_ps"], w["finished_at_ps"], w["payload_bytes"])
                                            for w in all_wqes)
            if observed_reservations != expected_reservations:
                findings.append(f"{direction} port projection lost or duplicated a reservation")
            for visits in intervals.values():
                visits.sort()
                if any(left[1] > right[0] for left, right in pairwise(visits)):
                    findings.append(f"overlapping {direction} port service")
    if integral(row["ttft_ps"]) != expected or integral(row["tpot_ps"]) != expected or row["jct_ps"] != 3*expected:
        findings.append("live request metrics disagree with the declared three-step chain")
    oracles.append({"family": "fixture_step_and_request_timing", "matches": all(
        s["latency_ps"] == expected for s in row["steps"])})
    return {**row, "fatal_findings": sorted(set(findings)), "exact_oracles": oracles}


def behavioral_relations(rows):
    indexed = {(r["case"]["name"], r["mode"]): r for r in rows}
    relations = []

    def add(family, case, observed, expected):
        relations.append({"family": family, "case": case, "observed": observed,
                          "expected": expected, "within_band": observed == expected})

    enabled = [r for r in rows if r["mode"] == "enabled" and r["status"] == "complete"]
    for row in enabled:
        case = row["case"]
        if case["family"] != "main":
            continue
        off = indexed.get((case["name"], "disabled"))
        if case["pattern"] == "combine" and case["width"] > 1 and off and off["status"] == "complete":
            add("receiver_latency_increase", case["name"],
                integral(row["ttft_ps"])-integral(off["ttft_ps"]),
                (case["width"]-1)*serialization(case["payload_bytes"], case["rate_gbps"]))
        key_fields = ("family", "pattern", "width", "payload_bytes", "channel_service_ps")
        if case["rate_gbps"] == 200:
            fast = next((r for r in enabled if r["case"]["rate_gbps"] == 400 and
                         all(r["case"][k] == case[k] for k in key_fields)), None)
            if fast:
                add("inverse_bandwidth", case["name"], integral(row["ttft_ps"]),
                    2*integral(fast["ttft_ps"]))
        if case["payload_bytes"] == 65536:
            small = next((r for r in enabled if r["case"]["payload_bytes"] == 4096 and
                          all(r["case"][k] == case[k] for k in
                              ("family", "pattern", "width", "rate_gbps", "channel_service_ps"))), None)
            if small:
                add("payload_scaling", case["name"], integral(row["ttft_ps"]),
                    16*integral(small["ttft_ps"]))
        if case["width"] > 1 and case["pattern"] in ("combine", "dispatch"):
            half = next((r for r in enabled if r["case"]["width"]*2 == case["width"] and
                         all(r["case"][k] == case[k] for k in
                             ("family", "pattern", "payload_bytes", "rate_gbps", "channel_service_ps"))), None)
            if half:
                add("fan_in_scaling", case["name"], integral(row["ttft_ps"]),
                    2*integral(half["ttft_ps"]))
    return relations


def summarize(rows, relations, expected_modes):
    fatal = [{"case": r["case"]["name"], "mode": r["mode"], "finding": finding}
             for r in rows for finding in r["fatal_findings"]]
    identities = {(r["case"]["name"], r["mode"]) for r in rows}
    expected = {(c["name"], mode) for c in cases() for mode in expected_modes}
    if len(rows) != len(expected) or identities != expected:
        fatal.append({"case": None, "mode": None, "finding": "matrix missing or duplicated"})
    misses = [r for r in relations if not r["within_band"]]
    return {"verdict": "void" if fatal else "relation-finding" if misses else "valid",
            "executions": len(rows), "request_steps": sum(len(r["steps"]) for r in rows),
            "fatal_findings": fatal, "behavioral_findings": misses,
            "behavioral_relation_families": sorted({r["family"] for r in relations}),
            "behavioral_relations": relations, "cells": rows}


def provenance(mode):
    import simllm

    source = Path(simllm.__file__).resolve().parents[1]
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain", "--", "simllm"], cwd=source, text=True)
    if dirty:
        raise ValueError("runtime source has tracked changes; commit before execution")
    if mode == "baseline" and revision != BASELINE_COMMIT:
        raise ValueError("baseline must import the immutable pre-change runtime")
    return {"runtime_commit": revision, "mode": mode, "baseline_commit": BASELINE_COMMIT,
            "mechanism_freeze_commit": MECHANISM_FREEZE,
            "pre_run_clarification_commit": PRE_RUN_CLARIFICATION,
            "input_identity_clarification_commit": INPUT_IDENTITY_CLARIFICATION,
            "ring_clarification_status": "after first implementation edit, before any unit or consumer execution",
            "input_identity_clarification_status": "after initial unit execution, before first consumer execution",
            "script_sha256": digest(Path(__file__).read_bytes()),
            "expectations_sha256": digest((HERE / "expectations.md").read_bytes()),
            "source_sha256": {name: digest((source/name).read_bytes()) for name in
                              ("simllm/core/runtime.py", "simllm/core/authority.py",
                              "simllm/core/completion.py", "simllm/traffic/collective_plan.py")}}


def load_baseline(directory, run_provenance):
    raw = (directory / "results.json").read_bytes()
    baseline = json.loads(raw)
    expected = {case["name"]: case for case in cases()}
    if (baseline["schema"] != "receiver-ingress-v1" or baseline["verdict"] != "valid" or
            baseline["provenance"]["runtime_commit"] != BASELINE_COMMIT or
            baseline["provenance"]["mode"] != "baseline"):
        raise ValueError("baseline provenance or verdict is invalid")
    for field in ("script_sha256", "expectations_sha256"):
        if baseline["provenance"][field] != run_provenance[field]:
            raise ValueError(f"baseline {field} differs from candidate")
    rows = baseline["cells"]
    names = [row["case"]["name"] for row in rows]
    if (len(names) != len(expected) or set(names) != set(expected) or
            baseline["executions"] != len(expected) or baseline["request_steps"] != 3*len(expected)):
        raise ValueError("baseline population is missing, duplicated or unexpected")
    for row in rows:
        name = row["case"]["name"]
        if (row["case"] != expected[name] or row["mode"] != "baseline" or
                row["status"] != "complete" or len(row["steps"]) != 3 or row["fatal_findings"]):
            raise ValueError("baseline configuration or outcome is invalid")
        if digest((directory / name / "baseline" / "snapshot.json").read_bytes()) != row["snapshot_sha256"]:
            raise ValueError("baseline snapshot digest mismatch")
    return {row["case"]["name"]: row for row in rows}, digest(raw)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("baseline", "candidate"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    args = parser.parse_args()
    if args.mode == "candidate" and args.baseline is None:
        parser.error("candidate requires --baseline with the immutable baseline results")
    if args.out.resolve().is_relative_to(HERE.parents[1]):
        parser.error("--out must be outside the repository")
    declared = {case["name"]: bounds(case) for case in cases()}
    run_provenance = provenance(args.mode)
    args.out.mkdir(parents=True, exist_ok=False)
    write_once(args.out / "pre_run_bounds.json", declared)
    baseline_rows = {}
    if args.baseline is not None:
        baseline_rows, baseline_digest = load_baseline(args.baseline, run_provenance)
        run_provenance["baseline_results_sha256"] = baseline_digest
    write_once(args.out / "provenance.json", run_provenance)
    rows = []
    for case in cases():
        disabled = None
        for mode in (("baseline",) if args.mode == "baseline" else ("disabled", "enabled")):
            output = args.out / case["name"] / mode
            output.mkdir(parents=True)
            write_once(output / "inputs.json", {"case": case, "bounds": declared[case["name"]],
                                                 "mode": mode, "provenance": run_provenance})
            row = observe(case, mode, output)
            # Raw behavior is persisted before any compatibility or exact oracle.
            prospective = [*rows, {**row, "fatal_findings": []}]
            relations = behavioral_relations(prospective)
            write_once(output / "behavior_before_guards.json", relations)
            row = apply_guards(row, declared[case["name"]], baseline_rows.get(case["name"]), disabled)
            write_once(output / "cell.json", row)
            rows.append(row)
            if mode == "disabled":
                disabled = row
            print(f"{case['name']} {mode}: {row['status']} TTFT={row['ttft_ps']} "
                  f"fatal={row['fatal_findings']}", flush=True)
    result = summarize(rows, behavioral_relations(rows),
                       ("baseline",) if args.mode == "baseline" else ("disabled", "enabled"))
    result["schema"] = "receiver-ingress-v1"
    result["provenance"] = run_provenance
    write_once(args.out / "results.json", result)
    print(f"verdict={result['verdict']} executions={result['executions']} "
          f"fatal_findings={len(result['fatal_findings'])}", flush=True)
    return 0 if result["verdict"] == "valid" else 2


if __name__ == "__main__":
    raise SystemExit(main())
