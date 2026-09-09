"""Qualify fresh K3 structural inventories without revising the original void run."""

from __future__ import annotations

import argparse
import gzip
import json
import os
import subprocess
import traceback
from pathlib import Path

from examples.kimi_k3_structure_v1 import audit_retained as audit
from examples.kimi_k3_structure_v1 import run_study as prior
from examples.kimi_k3_structure_v1.run_study import ROOT, SUITE_ID, digest, write
from simllm.backends.kimi_k3_lowerer import (
    KimiK3Lowerer,
    KimiK3LowererConfig,
    bind_synthetic_kimi_k3_graph,
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
from simllm.core.execution_io import execution_graph_from_json, execution_result_from_json
from simllm.core.step import RequestPhase, ScheduledRequest, StepRecord
from simllm.core.step_io import step_result_from_json

from .checks import (
    KINDS,
    Evidence,
    Scoped,
    causal_cell,
    causal_guards,
    discriminate,
    inherited_families,
    inherited_ids,
    scoped,
    successor_ids,
    validate_evidence,
)

HERE = Path(__file__).resolve().parent
FREEZE = "a3c1bd5c6709f7a07c356638e5e94aeec39ff469"


def git(*arguments):
    return subprocess.check_output(["git", *arguments], cwd=ROOT)


def extract(python, framework, checkpoint, output, repetition):
    name = f"{framework}-repeat-{repetition}"
    directory = output / name
    directory.mkdir()
    env = prior.environment(framework)
    env.update(CUDA_VISIBLE_DEVICES="", HIP_VISIBLE_DEVICES="")
    process = subprocess.Popen(
        [str(python), "-m", "examples.kimi_k3_qualification_v1.native_capture",
         "--framework", framework, "--checkpoint-root", str(checkpoint),
         "--output-root", str(directory)],
        cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    try:
        stdout, stderr = process.communicate(timeout=600)
    except subprocess.TimeoutExpired:
        process.kill()
        stdout, stderr = process.communicate()
        (output / f"{name}.stdout.log").write_bytes(stdout)
        (output / f"{name}.stderr.log").write_bytes(stderr)
        raise
    (output / f"{name}.stdout.log").write_bytes(stdout)
    (output / f"{name}.stderr.log").write_bytes(stderr)
    if process.returncode:
        raise RuntimeError(f"{name} failed with exit {process.returncode}; logs retained")
    rows = [json.loads(line) for line in stdout.splitlines() if line.startswith(b"{")]
    if len(rows) != 1:
        raise ValueError("native extraction must emit exactly one JSON receipt")
    result = rows[0]
    raw = (directory / "objects" / f"{result['record_sha256']}.json").read_bytes()
    inventory = ModelKernelInventory.from_obj(json.loads(raw))
    proof_raw = (directory / "native-proof.json").read_bytes()
    if digest(proof_raw) != result["proof_sha256"]:
        raise ValueError("native receipt does not identify retained source proof")
    return {
        "framework": framework, "repetition": repetition, "directory": directory,
        "pid": process.pid, "python": python, "receipt": result,
        "inventory": inventory, "raw": raw, "proof": json.loads(proof_raw),
    }


def proof_guards(capture, frozen, suite, config, evidence):
    framework, proof = capture["framework"], capture["proof"]
    directory, inventory = capture["directory"], capture["inventory"]
    retained_proof = (directory / "native-proof.json").read_bytes()
    retained_inventory = (directory / "objects" / f"{inventory.record.record_id}.json").read_bytes()
    row = next(f for f in frozen["frameworks"] if f["id"] == framework)
    binding = next(f for f in json.loads(suite)["frameworks"] if f["id"] == framework)
    check = scoped(evidence, f"native:{framework}:{capture['repetition']}")
    check.check("capture-process", type(proof["pid"]) is int and proof["pid"] > 0
                and proof["pid"] == capture["pid"] == capture["receipt"]["pid"], True)
    check.check("interpreter", os.path.abspath(proof["interpreter"]),
                os.path.abspath(capture["python"]))
    check.check("framework", (
        proof["framework"], proof["installed_version"], inventory.framework.framework_id,
        inventory.framework.version, inventory.framework.source_commit,
        inventory.framework.source_tree, proof["schema"], sorted(proof),
    ), (inventory.framework.to_obj(), row["version"], row["id"], row["version"],
        row["source_commit"], row["source_tree"], "simllm-kimi-k3-native-source-proof-v1",
        sorted({"schema", "pid", "interpreter", "framework", "installed_version", "package",
                "projection", "inputs_before", "inputs_after", "sources_before",
                "sources_after", "origins", "inventory_sha256", "steps_sha256",
                "gpu_initialized"})))
    check.check("projection-geometry", proof["projection"], {
        "schema": "simllm-framework-text-config-projection-v1",
        "framework": {key: row[key] for key in ("id", "version", "source_commit", "source_tree")},
        "configuration_seam": row["native_configuration"],
        "architecture_binding": binding["architecture_binding"],
        "text_implementation": binding["text_implementation"],
        "kimi_k3_stack": inventory.model.geometry.to_obj(),
    })
    check.check("suite-hash", (proof["inputs_before"]["suite"],
                               proof["inputs_after"]["suite"], inventory.suite.suite_sha256),
                (digest(suite),) * 3)
    check.check("config-hash", (proof["inputs_before"]["config"],
                                proof["inputs_after"]["config"], digest(config.read_bytes())),
                (frozen["model"]["config_sha256"],) * 3)
    check.check("inventory-hash", (proof["inventory_sha256"],
                                   capture["receipt"]["record_sha256"], digest(retained_inventory)),
                (inventory.record.record_id,) * 3)
    check.check("steps-hash", (proof["steps_sha256"], capture["receipt"]["steps_sha256"]),
                (digest((directory / "steps.jsonl").read_bytes()),) * 2)
    check.check("canonical-inventory",
                retained_inventory == capture["raw"] == inventory.record.canonical, True)
    before, after = proof["sources_before"], proof["sources_after"]
    expected_files = {n: value for n, value in frozen["source_files"].items()
                      if n.startswith(framework + "-")}
    check.check("exact-source-files", (sorted(r["name"] for r in before),
                                       sorted(r["name"] for r in after)),
                (sorted(expected_files), sorted(expected_files)))
    package = Path(proof["package"]).resolve(strict=True)
    for name, expected in expected_files.items():
        rows = [r for r in after if r["name"] == name]
        actual = None
        if len(rows) == 1:
            path = Path(rows[0]["file"]).resolve(strict=True)
            actual = (rows[0]["sha256"], digest(path.read_bytes()), path.is_relative_to(package))
        check.check("file:" + name, actual, (expected, expected, True))
    origins = proof["origins"]
    required = frozen["required_import_origins"][framework]
    names = [r["module"] for r in origins]
    check.check("required-origins", len(set(names)) == len(names)
                and set(required).issubset(names), True)
    check.check("all-origins-inside-package", all(
        Path(r["file"]).resolve(strict=True).is_relative_to(package) for r in origins
    ), True)
    check.check("all-origins-rehashed", all(
        digest(Path(r["file"]).read_bytes()) == r["sha256"] for r in origins
    ), True)
    for name, expected in required.items():
        check.check("origin:" + name,
                    [r["sha256"] for r in origins if r["module"] == name], [expected])
    check.check("source-stable", digest(retained_proof) == capture["receipt"]["proof_sha256"]
                and retained_proof == canonical_bytes(proof) and before == after
                and proof["inputs_before"] == proof["inputs_after"]
                and proof["gpu_initialized"] is False, True)
    evidence.finish_stage(f"native:{framework}:{capture['repetition']}")


def audit_repeat(capture, suite, spec, evidence):
    framework, repeat = capture["framework"], capture["repetition"]
    inventory, raw = capture["inventory"], capture["raw"]
    check = Scoped(evidence, "audit:")
    key = f"{framework}:repeat{repeat}"
    check.check(key + ":canonical", raw == inventory.record.canonical, True)
    check.check(key + ":address", inventory.record.record_id,
                capture["receipt"]["record_sha256"])
    check.check(key + ":suite", inventory.suite.suite_sha256, digest((ROOT / "offline/calibration/suites" / SUITE_ID / "suite.json").read_bytes()))
    check.check(key + ":model", inventory.model.geometry.to_obj(), spec.to_obj())
    check.check(key + ":case-identities", [c.case_id for c in inventory.cases],
                [c["id"] for c in suite["graph_cells"]])
    steps = (capture["directory"] / "steps.jsonl").read_bytes()
    check.check(key + ":steps", digest(steps), capture["receipt"]["steps_sha256"])
    audit.record_guards(steps, inventory.cases, case_records_from_suite(suite), check, key)


def audit_graphs(inventory, suite, inherited, rows, output, evidence):
    framework = inventory.framework.framework_id
    check, families, graphs = Scoped(evidence, "audit:"), set(), {}
    for case, record in zip(inventory.cases, case_records_from_suite(suite), strict=True):
        key = framework + ":" + case.case_id
        raw = gzip.decompress((output / f"{framework}-{case.case_id}.graph.json.gz").read_bytes())
        graph = execution_graph_from_json(json.loads(raw))
        graphs[case.case_id] = graph
        families.update(operator_family(op) for op in graph.operations)
        check.check(key + ":retained-graph-hash", digest(raw), rows[case.case_id]["graph_sha256"])
        check.check(key + ":instance-inventory-join",
                    unbound_execution_graph_record(graph).record_id, case.instance_graph_sha256)
        check.check(key + ":template-inventory-join",
                    execution_graph_template_record(graph).record_id, case.template_graph_sha256)
        audit.partition_guards(graph, case.phase, inherited, check, key)
        audit.projection_guards(graph, case, inventory, check, key)
        batch, request = len(record.scheduled), record.scheduled[0]
        state = rows[case.case_id]["state"]
        check.check(key + ":kda-history-invariance",
                    state["kda_recurrent_bytes"] + state["kda_convolution_history_bytes"],
                    batch * 449372160, kind="oracle")
        check.check(key + ":mla-history-growth", state["mla_cached_history_bytes"],
                    batch * (request.context_length - request.num_new_tokens) * 27648,
                    kind="oracle")
        stage = f"graph:{framework}:{case.case_id}"
        causal_guards(graph, case.phase, scoped(evidence, stage))
        evidence.finish_stage(stage)
    check.check(framework + ":total-family-inventory", sorted(families),
                sorted(f.family_id for f in inventory.kernel_families))
    return graphs


def audit_requests(spec, framework, output, evidence):
    check = Scoped(evidence, "audit:")
    for prompt in (1, 4):
        for service in (1000, 2000):
            release, identities = 0, []
            for index in range(3):
                name = f"{framework}-p{prompt}-s{service}-step{index}"
                step = step_result_from_json(json.loads((output / f"{name}.step.json").read_bytes()))
                events = execution_result_from_json(json.loads(gzip.decompress(
                    (output / f"{name}.events.json.gz").read_bytes()
                )))
                record = StepRecord(index, release, scheduled=[ScheduledRequest(
                    "request", RequestPhase.PREFILL if index == 0 else RequestPhase.DECODE,
                    prompt if index == 0 else 1, context_length=prompt + index,
                )], num_sampled=1)
                graph = KimiK3Lowerer(KimiK3LowererConfig(spec, framework)).lower(record)
                bound = bind_synthetic_kimi_k3_graph(graph, service)
                audit.event_guards(bound, events, (prompt if index == 0 else 1) * service,
                                   check, name)
                check.check(name + ":event-graph", events.execution_id, bound.execution_id)
                check.check(name + ":event-inventory", sorted({e.operation_id for e in events.events}),
                            sorted(op.operation_id for op in graph.operations))
                for before, after in zip(graph.operations, bound.operations, strict=True):
                    check.check(name + ":binding:" + before.operation_id, (
                        before.operation_id, before.logical_queue, before.depends_on,
                        before.correlation,
                    ) == (after.operation_id, after.logical_queue, after.depends_on,
                          after.correlation), True)
                    check.check(name + ":scope:" + before.operation_id,
                                after.work.scope == "synthetic-operator"
                                and after.work.hbm_bytes == 0
                                and dict(after.work.config).get("synthetic_memory_arbitration")
                                == "zero", True)
                audit.request_guards(step, events, index, release, prompt, service, check, name)
                release = step.completed_at_ps
                identities.append((step.step_index, events.execution_id))
            stage = f"request:{framework}:p{prompt}:s{service}"
            scoped(evidence, stage).check("three-retained-step-identities", identities,
                                         [(i, f"kimi-k3-step-{i}:synthetic:{service}")
                                          for i in range(3)])
            evidence.finish_stage(stage)


def protocol_checks(frozen, inherited, suite, historical, evidence):
    check = scoped(evidence, "protocol")
    original = json.loads((historical / "frozen-v1/summary.json").read_bytes())
    check.check("original-void", original["verdict"], "VOID")
    check.check("original-null-score", original["behavioral_score"], None)
    suite_path = ROOT / "offline/calibration/suites" / SUITE_ID / "suite.json"
    check.check("suite-hash", digest(suite_path.read_bytes()), frozen["suite"]["sha256"])
    check.check("suite-state", suite["state"], frozen["suite"]["state"])
    check.check("suite-case-identities", [c["id"] for c in suite["graph_cells"]],
                frozen["suite"]["case_ids"])
    records = case_records_from_suite(suite)
    check.check("frozen-grid", sorted((r.scheduled[0].phase.value, len(r.scheduled),
                                       r.scheduled[0].num_new_tokens,
                                       r.scheduled[0].context_length) for r in records),
                audit.frozen_grid(inherited))
    paths = git("ls-tree", "-r", "--name-only", FREEZE,
                "examples/kimi_k3_structure_v1").decode().splitlines()
    check.check("old-study-bytes", bool(paths) and all(
        (ROOT / p).read_bytes() == git("show", f"{FREEZE}:{p}") for p in paths
    ), True)
    check.check("historical-input-hashes", {
        name: digest((historical / name).read_bytes()) for name in frozen["historical_inputs"]
    }, frozen["historical_inputs"])
    evidence.finish_stage("protocol")


def execute(arguments, frozen, inherited, evidence, summary):
    output = arguments.output_root
    suite_raw = (ROOT / "offline/calibration/suites" / SUITE_ID / "suite.json").read_bytes()
    suite = json.loads(suite_raw)
    spec = KimiK3Spec.from_obj(suite["reference_model"]["geometry"])
    evidence.check("committed-source", git("status", "--porcelain", "--untracked-files=no").decode(), "")
    subprocess.run(["git", "merge-base", "--is-ancestor", FREEZE, "HEAD"], cwd=ROOT, check=True)
    for name in ("expectations.json", "expectations.md"):
        blob = git("show", f"{FREEZE}:examples/kimi_k3_qualification_v1/{name}")
        evidence.check("unchanged-freeze:" + name, digest((HERE / name).read_bytes()), digest(blob))
    for name, expected in inherited["source_files"].items():
        evidence.check("source:" + name, digest((arguments.source_root / name).read_bytes()), expected)
    if not evidence.valid:
        raise ValueError("source admission failed before native execution")
    summary["storage_oracles"] = prior.storage_oracles(spec, inherited, evidence)
    captures = []
    for framework in ("vllm", "sglang"):
        python = getattr(arguments, framework + "_python")
        repeats = []
        for repetition in range(2):
            capture = extract(python, framework, arguments.checkpoint_root, output, repetition)
            captures.append(capture)
            inventory = capture["inventory"]
            repeats.append((inventory.record.record_id, capture["receipt"]["steps_sha256"]))
            audit_repeat(capture, suite, spec, evidence)
        evidence.check(framework + ":native-geometry", capture["proof"]["projection"]["kimi_k3_stack"],
                       spec.to_obj())
        evidence.check(framework + ":repeat-identity", list(repeats[1]), list(repeats[0]))
        summary["inventories"][framework] = {"record_sha256": repeats[0][0],
                                             "steps_sha256": repeats[0][1]}
        rows = prior.check_graphs(inventory, suite, inherited, output, evidence)
        summary["graph_cells"][framework] = rows
        graphs = audit_graphs(inventory, suite, inherited, rows, output, evidence)
        for phase, case in frozen["causal_grid"]["canonical_case"].items():
            for shared in frozen["causal_grid"]["shared_region_ps"]:
                for addition in frozen["causal_grid"]["final_add_ps"]:
                    stage = f"causal:{framework}:{phase}:s{shared}:j{addition}"
                    row = causal_cell(graphs[case], phase, shared, addition,
                                      frozen, scoped(evidence, stage))
                    summary["causal_cells"].append({"framework": framework, **row})
                    evidence.finish_stage(stage)
        for mode in frozen["corruption_controls"]:
            phase = "decode" if mode == "decode-remove-cache-wait" else "prefill"
            stage = f"mutation:{framework}:{mode}"
            discriminate(graphs[frozen["causal_grid"]["canonical_case"][phase]],
                         phase, mode, scoped(evidence, stage))
            evidence.finish_stage(stage)
        summary["request_cells"].extend(prior.request_loop(spec, framework, output, evidence))
        audit_requests(spec, framework, output, evidence)
    state_rejections = []
    for variable in ("SGLANG_MAMBA_SSM_DTYPE", "SGLANG_MAMBA_CONV_DTYPE"):
        result = prior.inspect(arguments.sglang_python, "sglang", arguments.checkpoint_root, output,
                               "reject-" + variable.lower(), overrides={variable: "float16"},
                               reject=True)
        evidence.check("native-state-override:" + variable, result, True)
        state_rejections.append(result)
    before = len(evidence.guards)
    audit.rejection_guards(Scoped(evidence, "audit:"))
    scoped(evidence, "rejections").check("two-native-state-overrides", state_rejections, [True, True])
    scoped(evidence, "rejections").check("three-physical-consumer-rejections",
                                       [r["passed"] for r in evidence.guards[before:before + 3]],
                                       [True, True, True])
    evidence.finish_stage("rejections")
    summary["compatibility"] = prior.compatibility(arguments, inherited, evidence, output)
    expected = {(suite_id, framework) for suite_id in inherited["compatibility"]["native_controls"]
                for framework in ("vllm", "sglang")}
    scoped(evidence, "legacy").check("eight-exact-native-controls",
                                    sorted((r["suite"], r["framework"])
                                           for r in summary["compatibility"]["native_records"]),
                                    sorted(expected))
    evidence.finish_stage("legacy")
    for capture in captures:
        proof_guards(capture, frozen, suite_raw, arguments.checkpoint_root / "config.json", evidence)
    pids = [c["pid"] for c in captures]
    scoped(evidence, "native-distinct-processes").check("four-distinct-positive-pids",
                                                     len(set(pids)) == 4
                                                     and all(type(p) is int and p > 0 for p in pids),
                                                     True)
    evidence.finish_stage("native-distinct-processes")
    summary["native_processes"] = [
        {"framework": c["framework"], "repetition": c["repetition"], "pid": c["pid"],
         "proof_sha256": c["receipt"]["proof_sha256"]} for c in captures
    ]
    protocol_checks(frozen, inherited, suite, arguments.historical_root, evidence)


def finalize(output, frozen, evidence, summary, findings, families):
    if evidence is None:
        findings.append("missing evidence collector")
    else:
        rows = {key: getattr(evidence, key) for key in KINDS.values()}
        findings.extend(evidence.completeness())
        findings.extend(validate_evidence(rows, evidence.expected, frozen["successor_stage_ids"],
                                          sorted(evidence.finished_stages),
                                          families))
        if (not isinstance(evidence.relations, list) or
                len(evidence.relations) != frozen["coverage"]["behavioral_instances"]):
            findings.append("behavioral instance count differs")
        summary["finished_stages"] = sorted(evidence.finished_stages)
        summary["evidence_counts"] = {key: len(value) if isinstance(value, list) else None
                                      for key, value in rows.items()}
        summary["evidence_sha256"] = {}
        for name, values in rows.items():
            try:
                write(output / f"{name}.json", values)
                summary["evidence_sha256"][name] = canonical_sha256(values)
            except (TypeError, ValueError) as error:
                findings.append(f"noncanonical {name}: {type(error).__name__}: {error}")
                raw_path = output / f"{name}.noncanonical.txt"
                raw_path.write_text(repr(values), encoding="utf-8")
                summary.setdefault("noncanonical_evidence", {})[raw_path.name] = digest(
                    raw_path.read_bytes()
                )
    summary.update(verdict="VOID" if findings else "PASS", fatal_findings=findings,
                   behavioral_score=None if findings else {
                       "instances": len(evidence.relations),
                       "families": sorted({r["family"] for r in evidence.relations}),
                   })
    try:
        write(output / "summary.json", summary)
    except (TypeError, ValueError) as error:
        raw = repr(summary).encode("utf-8")
        (output / "summary.noncanonical.txt").write_bytes(raw)
        summary = {
            "schema": "simllm-kimi-k3-qualification-result-v1", "freeze_commit": FREEZE,
            "verdict": "VOID", "behavioral_score": None,
            "fatal_findings": [f"noncanonical summary: {type(error).__name__}: {error}"],
            "raw_summary_sha256": digest(raw),
        }
        write(output / "summary.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("vllm-python", "sglang-python", "checkpoint-root", "source-root",
                 "legacy-checkpoints", "historical-root", "output-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args(argv)
    args.output_root.mkdir(parents=True, exist_ok=False)
    frozen = json.loads((HERE / "expectations.json").read_bytes())
    summary = {
        "schema": "simllm-kimi-k3-qualification-result-v1", "owning_task": "COMP-54",
        "freeze_commit": FREEZE, "source_commit": git("rev-parse", "HEAD").decode().strip(),
        "chronology": frozen["chronology"], "hardware_measurements": 0,
        "original_verdict": "VOID", "original_rescored": False,
        "inventories": {}, "graph_cells": {}, "request_cells": [], "causal_cells": [],
    }
    evidence, findings, families = None, [], {}
    try:
        inherited_raw = (ROOT / frozen["inherited_contract"]["path"]).read_bytes()
        if digest(inherited_raw) != frozen["inherited_contract"]["sha256"]:
            raise ValueError("inherited contract changed")
        inherited = json.loads(inherited_raw)
        for phase, value in frozen["causal_grid"]["slow_shared_reference_ps"].items():
            inherited["diagnostic"]["shared_branch_control"][phase + "_completion_ps"] = value
        expected = inherited_ids(frozen, args.historical_root)
        expected.update(successor_ids(frozen))
        families = inherited_families(frozen, args.historical_root)
        if set(families.values()) != set(frozen["coverage"]["behavioral_families"]):
            raise ValueError("historical behavioral family set differs")
        evidence = Evidence(expected, frozen["successor_stage_ids"])
        execute(args, frozen, inherited, evidence, summary)
    except Exception as error:  # noqa: BLE001
        # Every ordinary execution failure must leave a void record and raw traceback.
        findings.append(f"{type(error).__name__}: {error}")
        raw = traceback.format_exc().encode("utf-8")
        (args.output_root / "exception.txt").write_bytes(raw)
        summary["exception_sha256"] = digest(raw)
    summary = finalize(args.output_root, frozen, evidence, summary, findings, families)
    print(json.dumps({"verdict": summary["verdict"], "fatal_finding_count": len(summary["fatal_findings"]),
                      "behavioral_score": summary["behavioral_score"]}), flush=True)
    return 2 if summary["verdict"] == "VOID" else 0


if __name__ == "__main__":
    raise SystemExit(main())
