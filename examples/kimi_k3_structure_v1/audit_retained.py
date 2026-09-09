"""Post-specified strict audit of retained K3 evidence; never clears a void run."""

from __future__ import annotations

import argparse
import gzip
import json
import subprocess
from collections import Counter, defaultdict
from dataclasses import replace
from itertools import pairwise
from pathlib import Path

from simllm.backends.kimi_k3_lowerer import (
    KimiK3Lowerer,
    KimiK3LowererConfig,
    bind_synthetic_kimi_k3_graph,
)
from simllm.calibration.canonical import canonical_sha256
from simllm.calibration.extraction import case_records_from_suite
from simllm.calibration.kimi_k3 import operator_family
from simllm.calibration.model_inventory import ModelKernelInventory
from simllm.compute.kimi_k3 import KimiK3Spec
from simllm.core import CoarseDeviceRuntime
from simllm.core.bottleneck import classify_runtime
from simllm.core.execution import ComputeWork, ExecutionGraph, ExecutionOperation
from simllm.core.execution_io import (
    execution_graph_from_json,
    execution_graph_to_json,
    execution_result_from_json,
)
from simllm.core.step import RequestPhase, ScheduledRequest, StepRecord, step_record_to_json
from simllm.core.step_io import step_result_from_json
from simllm.traffic.execution_goal import render_serial_execution_graph_goal

from .run_study import FREEZE, HERE, ROOT, SUITE_ID, Evidence, digest, graph_depth, write


def frozen_grid(frozen):
    matrix = frozen["shape_matrix"]
    return sorted(
        [
            ("prefill", batch, n, n)
            for batch in matrix["batch"]
            for n in matrix["cold_prefill_new_tokens_per_sequence"]
        ]
        + [
            ("decode", batch, 1, context)
            for batch in matrix["batch"]
            for context in matrix["decode_context_including_current_token"]
        ]
    )


def partition_guards(graph, phase, frozen, evidence, key):
    partition = frozen["operator_partition"]
    evidence.check(
        key + ":partition-total",
        len(graph.operations),
        frozen["exact_oracles"][phase + "_logical_visits"],
    )
    evidence.check(
        key + ":owned-layer-indices",
        all(
            op.correlation.layer is None or op.correlation.layer in range(93)
            for op in graph.operations
        ),
        True,
    )
    for layer in range(93):
        names = [
            op.work.kernel.removeprefix("kimi_k3.")
            for op in graph.operations
            if op.correlation.layer == layer
        ]
        attention = "mla" if (layer + 1) % 4 == 0 or layer == 92 else "kda"
        expected = (
            partition["kda"]
            if attention == "kda"
            else (
                partition["mla_common"]
                + partition["mla_expanded_extra" if phase == "prefill" else "mla_absorbed_extra"]
            )
        )
        evidence.check(
            f"{key}:layer{layer}:attention",
            sorted(name.split(".", 1)[1] for name in names if name.startswith(attention + ".")),
            sorted(expected),
        )
        ffn = "dense" if layer == 0 else "moe"
        evidence.check(
            f"{key}:layer{layer}:ffn",
            sorted(name.split(".", 1)[1] for name in names if name.startswith(ffn + ".")),
            sorted(partition["dense_ffn" if layer == 0 else "latent_moe"]),
        )
        residual = [name for name in names if name.startswith("residual.")]
        evidence.check(
            f"{key}:layer{layer}:residual",
            len(residual) == 3
            and residual[0].startswith("residual.attention_sources_")
            and residual[1].startswith("residual.mlp_sources_")
            and residual[2] == "residual.ffn_delta_prefix_add",
            True,
        )
    evidence.check(
        key + ":outer",
        [
            op.work.kernel.removeprefix("kimi_k3.")
            for op in graph.operations
            if op.correlation.layer is None
        ],
        partition["outer"],
    )


def projection_guards(graph, case, inventory, evidence, key):
    schemas = {schema.shape_schema_id: schema for schema in inventory.shape_schemas}
    for projection in case.kernel_projections:
        operations = [op for op in graph.operations if operator_family(op) == projection.family_id]
        schema = schemas[projection.shape_vector.shape_schema_id]
        evidence.check(
            key + ":visits:" + projection.family_id,
            projection.logical_launch_count,
            len(operations),
        )
        expected = [0] * len(schema.axes)
        if operations:
            work = operations[0].work
            axes = {name: int(value) for name, value in work.config if type(value) in (int, bool)}
            evidence.check(
                key + ":axis-inventory:" + projection.family_id,
                [axis.axis_id for axis in schema.axes],
                ["logical_visits", *axes],
            )
            expected = [len(operations), *axes.values()]
            evidence.check(
                key + ":homogeneous:" + projection.family_id,
                all(dict(op.work.config) == dict(work.config) for op in operations),
                True,
            )
        evidence.check(
            key + ":shape:" + projection.family_id, list(projection.shape_vector.values), expected
        )
        values = [op.work.flops for op in operations]
        evidence.check(
            key + ":flops:" + projection.family_id,
            projection.aggregate_flops,
            None if any(value is None for value in values) else sum(values),
        )
        evidence.check(
            key + ":memory:" + projection.family_id,
            projection.aggregate_hbm_bytes,
            None if operations else 0,
        )
    for operation in graph.operations:
        axes = dict(operation.work.config)
        if "input_width" in axes:
            evidence.check(
                key + ":matrix:" + operation.operation_id,
                operation.work.flops,
                2 * axes["tokens"] * axes["input_width"] * axes["output_width"] * axes["groups"],
                kind="oracle",
            )


def event_guards(graph, result, duration, evidence, key):
    """Check every lifecycle and serial interval directly from retained events."""
    by_operation = defaultdict(list)
    identities = Counter()
    resources = {
        "host-launch-queue": "node-0:framework-launch",
        "gpu-work-queue": "node-0:gpu-0:work",
    }
    for event in result.events:
        by_operation[event.operation_id].append(event)
        kind = None if event.resource is None else event.resource.kind.value
        identities[(event.operation_id, event.subject_object_id, event.phase, kind)] += 1
    evidence.check(
        key + ":fixed-resource-ownership",
        all(
            event.resource is not None
            and resources.get(event.resource.kind.value) == event.resource.resource_id
            for event in result.events
        ),
        True,
    )
    evidence.check(
        key + ":synthetic-event-bytes",
        all(
            event.completed_bytes == (0 if event.phase.value in ("progress", "completed") else None)
            for event in result.events
        ),
        True,
    )
    evidence.check(
        key + ":unique-lifecycle-events", all(count == 1 for count in identities.values()), True
    )
    evidence.check(
        key + ":complete-operation-events",
        sorted(by_operation),
        sorted(op.operation_id for op in graph.operations),
    )
    required = {
        ("host-launch-queue", phase) for phase in ("submitted", "queued", "started", "progress")
    } | {
        ("gpu-work-queue", phase)
        for phase in ("submitted", "queued", "started", "progress", "completed")
    }
    completed = {}
    intervals = []
    for operation in graph.operations:
        events = by_operation[operation.operation_id]
        local = {
            (event.resource.kind.value, event.phase.value): event
            for event in events
            if event.resource is not None
        }
        name = key + ":" + operation.operation_id
        evidence.check(name + ":lifecycle", sorted(local), sorted(required))
        evidence.check(
            name + ":event-owner",
            all(
                e.execution_id == graph.execution_id and e.subject_object_id is None for e in events
            ),
            True,
        )
        if set(local) != required:
            continue
        times = {pair: event.timestamp_ps for pair, event in local.items()}
        evidence.check(
            name + ":zero-host-service",
            all(
                times[("host-launch-queue", phase)] == graph.released_at_ps
                for phase in ("submitted", "queued", "started", "progress")
            ),
            True,
        )
        submitted, eligible, started, finished, visible = [
            times[("gpu-work-queue", phase)]
            for phase in ("submitted", "queued", "started", "progress", "completed")
        ]
        evidence.check(name + ":submission", submitted, graph.released_at_ps)
        predecessors_complete = all(parent in completed for parent in operation.depends_on)
        evidence.check(name + ":parents-owned", predecessors_complete, True)
        if predecessors_complete:
            evidence.check(
                name + ":causal-eligibility",
                eligible,
                max(
                    [graph.released_at_ps, *[completed[parent] for parent in operation.depends_on]]
                ),
            )
        evidence.check(name + ":grant-after-eligibility", started >= eligible, True)
        evidence.check(name + ":exact-service", finished - started, duration, kind="oracle")
        evidence.check(name + ":visibility", visible, finished)
        completed[operation.operation_id] = visible
        intervals.append((started, finished))
    intervals.sort()
    evidence.check(
        key + ":serial-no-overlap-or-idle",
        bool(intervals)
        and intervals[0][0] == graph.released_at_ps
        and all(a[1] == b[0] for a, b in pairwise(intervals)),
        True,
    )
    evidence.check(
        key + ":exact-step-end",
        result.completed_at_ps,
        graph.released_at_ps + len(graph.operations) * duration,
        kind="oracle",
    )
    evidence.check(key + ":quiescence", result.quiesced_at_ps, result.completed_at_ps)
    frontier = graph.completion_operation_ids or tuple(op.operation_id for op in graph.operations)
    evidence.check(
        key + ":completion-frontier-owned", all(op in completed for op in frontier), True
    )
    if all(op in completed for op in frontier):
        evidence.check(
            key + ":completion-frontier-time",
            max(completed[op] for op in frontier),
            result.completed_at_ps,
        )


def record_guards(raw, cases, records, evidence, key):
    """Join the retained native input and inventory to the frozen suite grid."""
    expected = [step_record_to_json(record) for record in records]
    evidence.check(
        key + ":retained-input-records", [json.loads(line) for line in raw.splitlines()], expected
    )
    evidence.check(key + ":case-count", len(cases), len(records))
    if len(cases) != len(records):
        return
    for case, record in zip(cases, records, strict=True):
        evidence.check(
            key + ":step-hash:" + case.case_id,
            case.step_record_sha256,
            canonical_sha256(step_record_to_json(record)),
        )
        evidence.check(key + ":phase:" + case.case_id, case.phase, record.scheduled[0].phase.value)


def request_guards(step, events, index, release, prompt, service, evidence, key):
    """Join the single request's token boundary to the retained completion."""
    evidence.check(key + ":step-index", step.step_index, index)
    evidence.check(key + ":step-latency", step.step_latency_ps, step.completed_at_ps - release)
    evidence.check(key + ":completion-join", step.completed_at_ps, events.completed_at_ps)
    evidence.check(key + ":request-count", len(step.request_metrics), 1)
    if len(step.request_metrics) != 1:
        return
    metric = step.request_metrics[0]
    evidence.check(
        key + ":request-identity",
        [metric.request_id, metric.phase.value, metric.token_index],
        ["request", "prefill" if index == 0 else "decode", index + 1],
    )
    evidence.check(key + ":request-completion", metric.completed_at_ps, events.completed_at_ps)
    evidence.check(key + ":request-latency", metric.latency_ps, events.completed_at_ps - release)
    evidence.check(key + ":request-ttft", metric.ttft_ps, 3244 * prompt * service, kind="oracle")
    evidence.check(
        key + ":rational-tpot",
        None if metric.tpot_ps is None else [metric.tpot_ps.numerator, metric.tpot_ps.denominator],
        None if index == 0 else [3268 * service, 1],
        kind="oracle",
    )


def rejection_guards(evidence):
    work = ComputeWork(
        "unbound", flops=None, hbm_bytes=None, nominal_duration_ps=7, scope="logical-operator"
    )
    graph = ExecutionGraph("reject", 0, 0, (ExecutionOperation("op", 0, "queue", work),))
    for name, action in (
        ("runtime", lambda: CoarseDeviceRuntime().execute(graph)),
        ("GOAL", lambda: render_serial_execution_graph_goal(graph)),
    ):
        try:
            action()
        except ValueError as error:
            evidence.check("reject-logical:" + name, "logical-operator" in str(error), True)
        else:
            evidence.check("reject-logical:" + name, False, True)
    executable = replace(
        graph,
        operations=(replace(graph.operations[0], work=ComputeWork("unit", nominal_duration_ps=7)),),
    )
    runtime = CoarseDeviceRuntime()
    runtime.execute(executable)
    try:
        classify_runtime(graph, runtime.last_report, {}, None)
    except ValueError as error:
        evidence.check("reject-logical:bottleneck", "logical-operator" in str(error), True)
    else:
        evidence.check("reject-logical:bottleneck", False, True)


def audit(source, output, native_sources):
    output.mkdir(parents=True, exist_ok=False)
    evidence = Evidence()
    source_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip()
    evidence.check(
        "audit-committed-source",
        subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT
        ).decode(),
        "",
    )
    for name in ("expectations.json", "expectations.md"):
        frozen_blob = subprocess.check_output(
            ["git", "show", f"{FREEZE}:examples/kimi_k3_structure_v1/{name}"], cwd=ROOT
        )
        evidence.check(
            "unchanged-freeze:" + name, digest((HERE / name).read_bytes()), digest(frozen_blob)
        )
    frozen = json.loads((HERE / "expectations.json").read_bytes())
    suite = json.loads((ROOT / "offline/calibration/suites" / SUITE_ID / "suite.json").read_bytes())
    summary_raw = (source / "summary.json").read_bytes()
    retained = json.loads(summary_raw)
    evidence.check("original-freeze", retained["freeze_commit"], FREEZE)
    evidence.check(
        "original-void-retained",
        (retained["verdict"], retained["behavioral_score"]),
        ("VOID", None),
    )
    records = case_records_from_suite(suite)
    grid = sorted(
        (
            r.scheduled[0].phase.value,
            len(r.scheduled),
            r.scheduled[0].num_new_tokens,
            r.scheduled[0].context_length,
        )
        for r in records
    )
    evidence.check("exact-frozen-grid", grid, frozen_grid(frozen))
    evidence.check(
        "relation-family-coverage",
        sorted({row["family"] for row in json.loads((source / "relations.json").read_bytes())}),
        sorted(frozen["behavioral_families"]),
    )
    spec = KimiK3Spec.from_obj(suite["reference_model"]["geometry"])
    corrected = []
    evidence.check(
        "retained-framework-inventory", sorted(retained["inventories"]), ["sglang", "vllm"]
    )
    for framework, identity in retained["inventories"].items():
        for repeat in range(2):
            raw = (
                source
                / f"{framework}-repeat-{repeat}"
                / "objects"
                / f"{identity['record_sha256']}.json"
            ).read_bytes()
            inventory = ModelKernelInventory.from_obj(json.loads(raw))
            evidence.check(
                f"{framework}:repeat{repeat}:canonical", raw == inventory.record.canonical, True
            )
            evidence.check(
                f"{framework}:repeat{repeat}:address",
                inventory.record.record_id,
                identity["record_sha256"],
            )
            evidence.check(
                f"{framework}:repeat{repeat}:suite",
                inventory.suite.suite_sha256,
                digest(
                    (ROOT / "offline/calibration/suites" / SUITE_ID / "suite.json").read_bytes()
                ),
            )
            evidence.check(
                f"{framework}:repeat{repeat}:model",
                inventory.model.geometry.to_obj(),
                spec.to_obj(),
            )
            evidence.check(
                f"{framework}:repeat{repeat}:case-identities",
                [case.case_id for case in inventory.cases],
                [cell["id"] for cell in suite["graph_cells"]],
            )
            steps_raw = (source / f"{framework}-repeat-{repeat}" / "steps.jsonl").read_bytes()
            evidence.check(
                f"{framework}:repeat{repeat}:steps", digest(steps_raw), identity["steps_sha256"]
            )
            record_guards(
                steps_raw, inventory.cases, records, evidence, f"{framework}:repeat{repeat}"
            )
        observed_families = set()
        for case, record in zip(inventory.cases, records, strict=True):
            key = framework + ":" + case.case_id
            raw = gzip.decompress(
                (source / f"{framework}-{case.case_id}.graph.json.gz").read_bytes()
            )
            evidence.check(
                key + ":retained-graph-hash",
                digest(raw),
                retained["graph_cells"][framework][case.case_id]["graph_sha256"],
            )
            graph = execution_graph_from_json(json.loads(raw))
            observed_families.update(operator_family(op) for op in graph.operations)
            from simllm.calibration.graph_identity import (
                execution_graph_template_record,
                unbound_execution_graph_record,
            )

            evidence.check(
                key + ":instance-inventory-join",
                unbound_execution_graph_record(graph).record_id,
                case.instance_graph_sha256,
            )
            evidence.check(
                key + ":template-inventory-join",
                execution_graph_template_record(graph).record_id,
                case.template_graph_sha256,
            )
            partition_guards(graph, case.phase, frozen, evidence, key)
            projection_guards(graph, case, inventory, evidence, key)
            batch = len(record.scheduled)
            context = record.scheduled[0].context_length
            n = record.scheduled[0].num_new_tokens
            state = retained["graph_cells"][framework][case.case_id]["state"]
            evidence.check(
                key + ":kda-history-invariance",
                state["kda_recurrent_bytes"] + state["kda_convolution_history_bytes"],
                batch * 449372160,
                kind="oracle",
            )
            evidence.check(
                key + ":mla-history-growth",
                state["mla_cached_history_bytes"],
                batch * (context - n) * 27648,
                kind="oracle",
            )
            actual = graph_depth(graph, slow_shared=True)
            expected = 3744000 if case.phase == "prefill" else 3792000
            evidence.check(key + ":post-specified-shared-addition", actual, expected, kind="oracle")
            corrected.append(
                {
                    "framework": framework,
                    "case": case.case_id,
                    "actual_ps": actual,
                    "original_frozen_ps": frozen["diagnostic"]["shared_branch_control"][
                        case.phase + "_completion_ps"
                    ],
                    "post_specified_ps": expected,
                }
            )
        evidence.check(
            framework + ":total-family-inventory",
            sorted(observed_families),
            sorted(f.family_id for f in inventory.kernel_families),
        )
        for prompt in (1, 4):
            for service in (1000, 2000):
                release = 0
                for index in range(3):
                    name = f"{framework}-p{prompt}-s{service}-step{index}"
                    step = step_result_from_json(
                        json.loads((source / f"{name}.step.json").read_bytes())
                    )
                    events = execution_result_from_json(
                        json.loads(
                            gzip.decompress((source / f"{name}.events.json.gz").read_bytes())
                        )
                    )
                    record = StepRecord(
                        index,
                        release,
                        scheduled=[
                            ScheduledRequest(
                                "request",
                                RequestPhase.PREFILL if index == 0 else RequestPhase.DECODE,
                                prompt if index == 0 else 1,
                                context_length=prompt + index,
                            )
                        ],
                        num_sampled=1,
                    )
                    graph = KimiK3Lowerer(KimiK3LowererConfig(spec, framework)).lower(record)
                    bound = bind_synthetic_kimi_k3_graph(graph, service)
                    event_guards(
                        bound, events, (prompt if index == 0 else 1) * service, evidence, name
                    )
                    evidence.check(name + ":event-graph", events.execution_id, bound.execution_id)
                    evidence.check(
                        name + ":event-inventory",
                        sorted({e.operation_id for e in events.events}),
                        sorted(op.operation_id for op in graph.operations),
                    )
                    for before, after in zip(graph.operations, bound.operations, strict=True):
                        evidence.check(
                            name + ":binding:" + before.operation_id,
                            (
                                before.operation_id,
                                before.logical_queue,
                                before.depends_on,
                                before.correlation,
                            )
                            == (
                                after.operation_id,
                                after.logical_queue,
                                after.depends_on,
                                after.correlation,
                            ),
                            True,
                        )
                        axes = dict(after.work.config)
                        evidence.check(
                            name + ":scope:" + before.operation_id,
                            after.work.scope == "synthetic-operator"
                            and after.work.hbm_bytes == 0
                            and axes.get("synthetic_memory_arbitration") == "zero",
                            True,
                        )
                    request_guards(step, events, index, release, prompt, service, evidence, name)
                    write(
                        output / f"{name}.graph-identities.json",
                        {
                            "logical_sha256": canonical_sha256(execution_graph_to_json(graph)),
                            "synthetic_sha256": canonical_sha256(execution_graph_to_json(bound)),
                            "reconstruction": "post-specified-from-retained-step-and-service-inputs",
                        },
                    )
                    release = step.completed_at_ps
    rejection_guards(evidence)
    for variable in ("sglang_mamba_ssm_dtype", "sglang_mamba_conv_dtype"):
        stderr = (source / f"reject-{variable}.stderr.log").read_bytes()
        evidence.check(
            variable + ":exact-native-rejection",
            stderr.splitlines()[-1:]
            == [
                b"ValueError: this structural envelope declares FP32 recurrence and BF16 convolution state"
            ],
            True,
        )
    native = json.loads(native_sources.read_bytes())
    expected_files = {
        name: value for name, value in frozen["source_files"].items() if name != "config.json"
    }
    source_rows = [row for framework in native["frameworks"] for row in framework["rows"]]
    evidence.check(
        "native-source-file-inventory",
        sorted(row["name"] for row in source_rows),
        sorted(expected_files),
    )
    for row in source_rows:
        evidence.check(
            "native-source:" + row["name"], row["actual_sha256"], expected_files[row["name"]]
        )
        evidence.check(
            "native-source-live:" + row["name"],
            digest(Path(row["file"]).read_bytes()),
            expected_files[row["name"]],
        )
    evidence.check(
        "native-framework-inventory",
        sorted(row["framework"] for row in native["frameworks"]),
        ["sglang", "vllm"],
    )
    for framework in native["frameworks"]:
        expected_framework = next(
            row for row in frozen["frameworks"] if row["id"] == framework["framework"]
        )
        evidence.check(
            "native-pin:" + framework["framework"],
            framework["source_commit"],
            expected_framework["source_commit"],
        )
        evidence.check(
            "native-origin-nonempty:" + framework["framework"], bool(framework["origins"]), True
        )
        for origin in framework["origins"]:
            actual_file = Path(origin["file"]).resolve()
            evidence.check(
                "native-origin:" + origin["module"],
                actual_file.is_relative_to(Path(framework["package"]).resolve()),
                True,
            )
            evidence.check(
                "native-origin-bytes:" + origin["module"],
                digest(actual_file.read_bytes()),
                origin["sha256"],
            )
    result = {
        "schema": "simllm-kimi-k3-retained-audit-v1",
        "evidence_class": "post-specified-regression-audit",
        "source_commit": source_commit,
        "original_summary_sha256": digest(summary_raw),
        "native_sources_sha256": digest(native_sources.read_bytes()),
        "original_verdict": retained["verdict"],
        "owning_task": "COMP-54",
        "closes_task": False,
        "audit_verdict": "PASS" if evidence.valid else "VOID",
        "original_behavioral_score": None,
        "post_specified_shared_oracles": corrected,
        "findings": [row for row in evidence.guards + evidence.oracles if not row["passed"]],
    }
    write(output / "guards.json", evidence.guards)
    write(output / "oracles.json", evidence.oracles)
    write(output / "summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--native-sources", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = audit(args.source_root, args.output_root, args.native_sources)
    except (OSError, ValueError, TypeError, RuntimeError, KeyError) as error:
        result = {
            "schema": "simllm-kimi-k3-retained-audit-v1",
            "audit_verdict": "VOID",
            "evidence_class": "post-specified-regression-audit",
            "closes_task": False,
            "findings": [{"error": f"{type(error).__name__}: {error}"}],
        }
        # Never replace a prior output directory on an admission failure.
        if not isinstance(error, FileExistsError):
            write(args.output_root / "summary.json", result)
    print(json.dumps(result))
    return 0 if result["audit_verdict"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
