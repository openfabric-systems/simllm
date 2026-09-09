"""Retain the frozen K3 structural study, including refuted expectations."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from simllm.backends.kimi_k3_lowerer import (
    KimiK3Lowerer,
    KimiK3LowererConfig,
    bind_synthetic_kimi_k3_graph,
    kimi_k3_step_shape,
)
from simllm.calibration.canonical import canonical_bytes, canonical_sha256
from simllm.calibration.extraction import case_records_from_suite
from simllm.calibration.graph_identity import (
    execution_graph_template_record,
    unbound_execution_graph_record,
)
from simllm.calibration.kimi_k3 import operator_family
from simllm.calibration.model_inventory import ModelKernelInventory
from simllm.compute.kimi_k3 import KimiK3Spec
from simllm.core import CoarseDeviceRuntime, CompletionReducer, VirtualClock
from simllm.core.execution_io import execution_graph_to_json, execution_result_to_json
from simllm.core.step import RequestPhase, ScheduledRequest, StepRecord
from simllm.core.step_io import step_result_to_json

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
FREEZE = "727d78c68e38c4f0eecb6e0e75fe37f13932a0b3"
SUITE_ID = "kimi-k3-text-v1-frameworks-2026-09-09"


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_bytes(value))


def graph_depth(graph, *, slow_shared=False):
    """Independent logical longest path; this is not runtime concurrency."""
    finished = {}
    for operation in graph.operations:
        shared = (
            operation.work.kernel.startswith("kimi_k3.moe.shared_")
            and operation.work.kernel != "kimi_k3.moe.shared_routed_add"
        )
        service = 10000 if slow_shared and shared else 1000
        finished[operation.operation_id] = service + max(
            (finished[parent] for parent in operation.depends_on), default=0
        )
    return max(finished[name] for name in graph.completion_operation_ids)


class Evidence:
    def __init__(self):
        self.guards = []
        self.oracles = []
        self.relations = []

    def check(self, name, actual, expected, *, kind="guard", family=None):
        row = {"name": name, "actual": actual, "expected": expected, "passed": actual == expected}
        if family is not None:
            row["family"] = family
        getattr(
            self, {"guard": "guards", "oracle": "oracles", "relation": "relations"}[kind]
        ).append(row)

    @property
    def valid(self):
        return all(row["passed"] for row in self.guards + self.oracles + self.relations)


def environment(framework):
    env = dict(os.environ)
    env.update(
        PYTHONPATH=str(ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""),
        PYTHONDONTWRITEBYTECODE="1",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
    )
    if framework == "vllm":
        env.update(SIMLLM_VLLM_WORKER_MODE="skeleton", VLLM_ENABLE_V1_MULTIPROCESSING="0")
    else:
        env["SIMLLM_SGLANG_ENABLE"] = "1"
    return env


def native_run(python, framework, code, args, output, name, *, overrides=None, reject=False):
    env = environment(framework)
    env.update(overrides or {})
    try:
        process = subprocess.run(
            [str(python), "-c", code, *map(str, args)],
            cwd=ROOT,
            env=env,
            capture_output=True,
            timeout=600,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        output.joinpath(f"{name}.stdout.log").write_bytes(error.stdout or b"")
        output.joinpath(f"{name}.stderr.log").write_bytes(error.stderr or b"")
        raise
    output.joinpath(f"{name}.stdout.log").write_bytes(process.stdout)
    output.joinpath(f"{name}.stderr.log").write_bytes(process.stderr)
    if reject:
        return process.returncode != 0 and process.stderr.splitlines()[-1:] == [
            b"ValueError: this structural envelope declares FP32 recurrence and BF16 convolution state"
        ]
    if process.returncode:
        raise RuntimeError(f"{name} failed with exit {process.returncode}; raw logs retained")
    records = [json.loads(line) for line in process.stdout.splitlines() if line.startswith(b"{")]
    if len(records) != 1:
        raise ValueError(f"{name} must emit exactly one JSON object")
    return records[0]


def inspect(python, framework, checkpoint, output, name, **kwargs):
    code = (
        "import json,sys; from pathlib import Path; "
        f"from simllm.adapters.{framework}.extraction import inspect_configuration; "
        "print(json.dumps(inspect_configuration(Path(sys.argv[1])),sort_keys=True))"
    )
    return native_run(python, framework, code, [checkpoint], output, name, **kwargs)


def extract(python, framework, checkpoint, output, name, suite_id=SUITE_ID):
    directory = output / name
    directory.mkdir()
    code = "from simllm.calibration.cli import main; raise SystemExit(main())"
    result = native_run(
        python,
        framework,
        code,
        [
            "extract",
            "--framework",
            framework,
            "--suite",
            suite_id,
            "--suite-root",
            ROOT / "offline/calibration",
            "--checkpoint-root",
            checkpoint,
            "--step-records",
            directory / "steps.jsonl",
            "--output-root",
            directory / "objects",
        ],
        output,
        name,
    )
    raw = (directory / "objects" / f"{result['record_sha256']}.json").read_bytes()
    inventory = ModelKernelInventory.from_obj(json.loads(raw))
    if inventory.record.canonical != raw or inventory.record.record_id != result["record_sha256"]:
        raise ValueError("native inventory content address is inconsistent")
    return inventory, digest((directory / "steps.jsonl").read_bytes())


def storage_oracles(spec, frozen, evidence):
    weights = spec.weight_shapes()
    total = lambda names: sum(w.parameters for w in weights if w.name in names)
    state = spec.retained_state(sequences=1, cached_tokens=1, current_tokens=1)
    routed = total({"moe.expert_gate", "moe.expert_up", "moe.expert_down"})
    values = {
        "routed_values_per_expert": spec.moe.expert_parameters,
        "routed_encoded_bytes_per_expert": spec.moe.expert_encoding_floor_bytes,
        "routed_values_all_layers": routed,
        "routed_encoded_bytes_all_layers": routed // 2 + routed // 32,
        "shared_bf16_bytes_all_layers": 2
        * total({"moe.shared_gate", "moe.shared_up", "moe.shared_down"}),
        "latent_projection_values_per_layer": total({"moe.latent_down", "moe.latent_up"}) // 92,
        "router_values_per_layer": total({"moe.router"}) // 92,
        "dense_values_first_layer": total({"dense.gate", "dense.up", "dense.down"}),
        "kda_recurrent_bytes_per_sequence": state["kda_recurrent_bytes"],
        "kda_convolution_history_bytes_per_sequence": state["kda_convolution_history_bytes"],
        "kda_total_persistent_bytes_per_sequence": state["kda_recurrent_bytes"]
        + state["kda_convolution_history_bytes"],
        "mla_history_bytes_per_cached_token": state["mla_cached_history_bytes"],
        "residual_snapshot_bytes_per_current_token": state["residual_snapshot_capacity_bytes"],
    }
    for name, actual in values.items():
        evidence.check(name, actual, frozen["exact_oracles"][name], kind="oracle")
    return values


def check_graphs(inventory, suite, frozen, output, evidence):
    framework = inventory.framework.framework_id
    spec = inventory.model.geometry
    lowerer = KimiK3Lowerer(KimiK3LowererConfig(spec, framework))
    rows = {}
    for case, record in zip(inventory.cases, case_records_from_suite(suite), strict=True):
        graph = lowerer.lower(record)
        shape = kimi_k3_step_shape(record, spec)
        key = f"{framework}:{case.case_id}"
        raw = canonical_bytes(execution_graph_to_json(graph))
        with gzip.GzipFile(
            filename=str(output / f"{key.replace(':', '-')}.graph.json.gz"), mode="wb", mtime=0
        ) as stream:
            stream.write(raw)
        evidence.check(
            key + ":instance-join",
            unbound_execution_graph_record(graph).record_id,
            case.instance_graph_sha256,
        )
        evidence.check(
            key + ":template-join",
            execution_graph_template_record(graph).record_id,
            case.template_graph_sha256,
        )
        evidence.check(
            key + ":original-layers",
            sorted(
                {
                    op.correlation.layer
                    for op in graph.operations
                    if op.correlation.layer is not None
                }
            ),
            list(range(93)),
        )
        evidence.check(
            key + ":unbound-demand",
            all(
                op.work.hbm_bytes is None
                and op.work.scope == "logical-operator"
                and op.work.nominal_duration_ps is None
                for op in graph.operations
            ),
            True,
        )
        counts = {}
        for operation in graph.operations:
            family = operator_family(operation)
            counts[family] = counts.get(family, 0) + 1
        evidence.check(
            key + ":family-join",
            counts,
            {
                p.family_id: p.logical_launch_count
                for p in case.kernel_projections
                if p.logical_launch_count
            },
        )
        evidence.check(
            key + ":visits",
            len(graph.operations),
            frozen["exact_oracles"][case.phase + "_logical_visits"],
            kind="oracle",
        )
        evidence.check(
            key + ":depth",
            graph_depth(graph) // 1000,
            frozen["exact_oracles"][case.phase + "_dependency_depth"],
            kind="oracle",
        )
        evidence.check(
            key + ":residual-sources",
            sum(dict(op.work.config).get("source_vectors", 0) for op in graph.operations),
            frozen["exact_oracles"]["residual_source_evaluations_per_current_token"],
            kind="oracle",
        )
        caches = {
            op.operation_id
            for op in graph.operations
            if op.work.kernel == "kimi_k3.mla.compressed_cache_commit"
        }
        evidence.check(
            key + ":cache-frontier", len(caches & set(graph.completion_operation_ids)), 24
        )
        scores = [
            op for op in graph.operations if op.work.kernel == "kimi_k3.mla.query_key_product"
        ]
        evidence.check(
            key + ":cache-causality",
            all(bool(caches & set(op.depends_on)) == (case.phase == "decode") for op in scores),
            True,
        )
        score_flops = sum(
            op.work.flops
            for op in graph.operations
            if op.work.kernel
            in {"kimi_k3.mla.query_key_product", "kimi_k3.mla.probability_value_product"}
        )
        n = record.scheduled[0].num_new_tokens
        context = record.scheduled[0].context_length
        expected_pairs = len(record.scheduled) * (n * (context - n) + n * (n + 1) // 2)
        family = (
            "cold-prefill-inclusive-pairs" if case.phase == "prefill" else "decode-history-growth"
        )
        evidence.check(
            key + ":attention-work",
            score_flops,
            expected_pairs * 24 * (61440 if case.phase == "prefill" else 208896),
            kind="relation",
            family=family,
        )
        final_norm = next(op for op in graph.operations if op.work.kernel == "kimi_k3.final_norm")
        evidence.check(
            key + ":final-norm-axis",
            dict(final_norm.work.config)["tokens"],
            len(record.scheduled) if framework == "vllm" else shape.new_tokens,
        )
        rows[case.case_id] = {
            "graph_sha256": digest(raw),
            "known_flops": sum(op.work.flops or 0 for op in graph.operations),
            "state": spec.retained_state(
                sequences=shape.sequences,
                cached_tokens=shape.cached_tokens,
                current_tokens=shape.new_tokens,
            ),
            "slow_shared_ps": graph_depth(graph, slow_shared=True),
        }
    for case_id, row in rows.items():
        if "-b3-" in case_id:
            baseline = rows[case_id.replace("-b3-", "-b1-")]
            evidence.check(
                framework + ":batch-work:" + case_id,
                row["known_flops"],
                3 * baseline["known_flops"],
                kind="relation",
                family="batch-linear-work-and-capacity",
            )
            evidence.check(
                framework + ":batch-state:" + case_id,
                row["state"],
                {name: 3 * value for name, value in baseline["state"].items()},
                kind="relation",
                family="batch-linear-work-and-capacity",
            )
    for phase, case_id in (("prefill", "prefill-b1-n1"), ("decode", "decode-b1-c1")):
        evidence.check(
            framework + ":frozen-shared-join:" + phase,
            rows[case_id]["slow_shared_ps"],
            frozen["diagnostic"]["shared_branch_control"][phase + "_completion_ps"],
        )
    return rows


def request_loop(spec, framework, output, evidence):
    rows = []
    for prompt in (1, 4):
        for service in (1000, 2000):
            clock = VirtualClock(0)
            runtime = CoarseDeviceRuntime(serial_compute=True)
            reducer = CompletionReducer(clock)
            results = []
            for index in range(3):
                tokens = prompt if index == 0 else 1
                record = StepRecord(
                    index,
                    clock.now_ps,
                    scheduled=[
                        ScheduledRequest(
                            "request",
                            RequestPhase.PREFILL if index == 0 else RequestPhase.DECODE,
                            tokens,
                            context_length=prompt + index,
                        )
                    ],
                    num_sampled=1,
                )
                graph = KimiK3Lowerer(KimiK3LowererConfig(spec, framework)).lower(record)
                bound = bind_synthetic_kimi_k3_graph(graph, service)
                result = runtime.execute(bound)
                reduced = reducer.reduce(record, bound, result, runtime.last_report)
                name = f"{framework}-p{prompt}-s{service}-step{index}"
                write(output / f"{name}.step.json", step_result_to_json(reduced))
                with gzip.GzipFile(
                    filename=str(output / f"{name}.events.json.gz"), mode="wb", mtime=0
                ) as stream:
                    stream.write(canonical_bytes(execution_result_to_json(result)))
                expected = (3244 if index == 0 else 3268) * tokens * service
                evidence.check(
                    name + ":serial-completion", reduced.step_latency_ps, expected, kind="oracle"
                )
                evidence.check(
                    name + ":event-inventory",
                    len({event.operation_id for event in result.events}),
                    len(graph.operations),
                )
                evidence.check(
                    name + ":synthetic-identity", bound.execution_id != graph.execution_id, True
                )
                metric = reduced.request_metrics[0]
                evidence.check(
                    name + ":ttft", metric.ttft_ps, 3244 * prompt * service, kind="oracle"
                )
                evidence.check(
                    name + ":tpot",
                    None
                    if metric.tpot_ps is None
                    else [metric.tpot_ps.numerator, metric.tpot_ps.denominator],
                    None if index == 0 else [3268 * service, 1],
                    kind="oracle",
                )
                results.append(reduced.step_latency_ps)
            row = {
                "framework": framework,
                "prompt_tokens": prompt,
                "service_ps_per_new_token": service,
                "ttft_ps": results[0],
                "tpot_ps": results[1],
                "jct_ps": clock.now_ps,
                "evidence_class": "synthetic-serial-logical-operator-runtime",
            }
            rows.append(row)
            print(json.dumps(row), flush=True)
    for field in ("ttft_ps", "tpot_ps", "jct_ps"):
        for a, b in ((rows[0], rows[1]), (rows[2], rows[3])):
            evidence.check(
                f"{framework}:double-service:p{a['prompt_tokens']}:{field}",
                b[field],
                2 * a[field],
                kind="relation",
                family="synthetic-request-service-scaling",
            )
    evidence.check(
        framework + ":prompt-ttft-scaling",
        rows[2]["ttft_ps"],
        4 * rows[0]["ttft_ps"],
        kind="relation",
        family="synthetic-request-service-scaling",
    )
    evidence.check(
        framework + ":decode-prompt-independence",
        rows[2]["tpot_ps"],
        rows[0]["tpot_ps"],
        kind="relation",
        family="synthetic-request-service-scaling",
    )
    return rows


def compatibility(arguments, frozen, evidence, output):
    baseline = frozen["compatibility"]["baseline_commit"]
    paths = (
        subprocess.check_output(["git", "ls-tree", "-r", "--name-only", baseline], cwd=ROOT)
        .decode()
        .splitlines()
    )
    selected = [
        p
        for p in paths
        if p.startswith("offline/calibration/model-inventories/")
        and p.endswith(".json")
        or p.startswith("offline/calibration/suites/")
        and p.endswith("/suite.json")
    ]
    for path in selected:
        raw = subprocess.check_output(["git", "show", f"{baseline}:{path}"], cwd=ROOT)
        evidence.check("legacy-bytes:" + path, digest((ROOT / path).read_bytes()), digest(raw))
    checkpoints = json.loads(arguments.legacy_checkpoints.read_bytes())
    if set(checkpoints) != set(frozen["compatibility"]["native_controls"]):
        raise ValueError("legacy checkpoint map must cover all four frozen controls")
    rows = []
    accepted = [
        ModelKernelInventory.from_obj(json.loads((ROOT / path).read_bytes()))
        for path in selected
        if path.startswith("offline/calibration/model-inventories/")
    ]
    for suite_id, checkpoint in checkpoints.items():
        for framework in ("vllm", "sglang"):
            inventory, steps = extract(
                getattr(arguments, framework + "_python"),
                framework,
                Path(checkpoint),
                output,
                f"legacy-{suite_id}-{framework}",
                suite_id,
            )
            matches = [
                item
                for item in accepted
                if item.suite == inventory.suite and item.framework == inventory.framework
            ]
            evidence.check(
                f"native-legacy:{suite_id}:{framework}",
                inventory.record.record_id in {item.record.record_id for item in matches},
                True,
            )
            rows.append(
                {
                    "suite": suite_id,
                    "framework": framework,
                    "inventory_sha256": inventory.record.record_id,
                    "steps_sha256": steps,
                }
            )
    return {"tracked_artifacts": len(selected), "native_records": rows}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vllm-python", type=Path, required=True)
    parser.add_argument("--sglang-python", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--legacy-checkpoints", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    args.output_root.mkdir(parents=True, exist_ok=False)
    evidence = Evidence()
    summary = {
        "schema": "simllm-kimi-k3-structure-result-v1",
        "freeze_commit": FREEZE,
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT)
        .decode()
        .strip(),
        "hardware_measurements": 0,
        "inventories": {},
        "graph_cells": {},
        "request_cells": [],
    }
    frozen = json.loads((HERE / "expectations.json").read_bytes())
    suite = json.loads((ROOT / "offline/calibration/suites" / SUITE_ID / "suite.json").read_bytes())
    try:
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT
        ).decode()
        evidence.check("committed-source", dirty, "")
        for name in ("expectations.json", "expectations.md"):
            blob = subprocess.check_output(
                ["git", "show", f"{FREEZE}:examples/kimi_k3_structure_v1/{name}"], cwd=ROOT
            )
            evidence.check(
                "unchanged-freeze:" + name, digest((HERE / name).read_bytes()), digest(blob)
            )
        for name, expected in frozen["source_files"].items():
            evidence.check(
                "source:" + name, digest((args.source_root / name).read_bytes()), expected
            )
        spec = KimiK3Spec.from_obj(suite["reference_model"]["geometry"])
        summary["storage_oracles"] = storage_oracles(spec, frozen, evidence)
        projections = {}
        for framework in ("vllm", "sglang"):
            python = getattr(args, framework + "_python")
            projection = inspect(
                python, framework, args.checkpoint_root, args.output_root, framework + "-inspect"
            )
            projections[framework] = projection
            evidence.check(
                framework + ":native-geometry", projection["kimi_k3_stack"], spec.to_obj()
            )
            repeats = []
            for repetition in range(2):
                inventory, step_sha = extract(
                    python,
                    framework,
                    args.checkpoint_root,
                    args.output_root,
                    f"{framework}-repeat-{repetition}",
                )
                repeats.append((inventory.record.record_id, step_sha))
            evidence.check(framework + ":repeat-identity", list(repeats[1]), list(repeats[0]))
            summary["inventories"][framework] = {
                "record_sha256": repeats[0][0],
                "steps_sha256": repeats[0][1],
            }
            summary["graph_cells"][framework] = check_graphs(
                inventory, suite, frozen, args.output_root, evidence
            )
            summary["request_cells"].extend(
                request_loop(spec, framework, args.output_root, evidence)
            )
        for variable in ("SGLANG_MAMBA_SSM_DTYPE", "SGLANG_MAMBA_CONV_DTYPE"):
            evidence.check(
                "native-state-override:" + variable,
                inspect(
                    args.sglang_python,
                    "sglang",
                    args.checkpoint_root,
                    args.output_root,
                    "reject-" + variable.lower(),
                    overrides={variable: "float16"},
                    reject=True,
                ),
                True,
            )
        summary["compatibility"] = compatibility(args, frozen, evidence, args.output_root)
    except (
        OSError,
        ValueError,
        TypeError,
        RuntimeError,
        KeyError,
        subprocess.SubprocessError,
    ) as error:
        evidence.check("execution-exception", f"{type(error).__name__}: {error}", None)
    summary.update(
        verdict="PASS" if evidence.valid else "VOID",
        behavioral_score=(
            {
                "families": sorted({row["family"] for row in evidence.relations}),
                "instances": len(evidence.relations),
            }
            if evidence.valid
            else None
        ),
        fatal_findings=[
            row
            for row in evidence.guards + evidence.oracles + evidence.relations
            if not row["passed"]
        ],
    )
    write(args.output_root / "guards.json", evidence.guards)
    write(args.output_root / "oracles.json", evidence.oracles)
    write(args.output_root / "relations.json", evidence.relations)
    summary["evidence_sha256"] = {
        name: canonical_sha256(rows)
        for name, rows in (
            ("guards", evidence.guards),
            ("oracles", evidence.oracles),
            ("relations", evidence.relations),
        )
    }
    write(args.output_root / "summary.json", summary)
    print(
        json.dumps(
            {
                "verdict": summary["verdict"],
                "behavioral_score": summary["behavioral_score"],
                "fatal_findings": summary["fatal_findings"],
            }
        ),
        flush=True,
    )
    return 0 if evidence.valid else 2


if __name__ == "__main__":
    sys.exit(main())
