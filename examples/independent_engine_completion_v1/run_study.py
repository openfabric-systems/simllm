"""Run the frozen native completion campaign and retain every failed attempt."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

from examples.independent_engine_completion_v1.checks import (
    check_baselines,
    check_cells,
    check_checkpoints,
    check_components,
    check_engines,
    check_native_relations,
    check_projections,
    check_sinks_and_domains,
    check_sources,
)
from examples.independent_engine_completion_v1.common import (
    FREEZE_COMMIT,
    HERE,
    ROOT,
    SOURCES,
    Evidence,
    GuardFailure,
    exact_json_bytes,
    parse_value,
    read,
    sha,
    step_stream,
    write,
)
from examples.independent_engine_completion_v1.reference import component_run, native_reference
from examples.pd_session_v1 import run_study as baseline


def file_receipts(root):
    return {path.relative_to(root).as_posix(): {"sha256": sha(path.read_bytes()), "bytes": path.stat().st_size}
            for path in sorted(root.rglob("*")) if path.is_file()}


def reread_tree(root, receipts, evidence, label):
    evidence.equal(label + ":raw-bytes-and-domain", file_receipts(root), receipts)


def progress_rows(path):
    raw = path.read_bytes()
    rows = [parse_value(line, path.name) for line in raw.splitlines()]
    if raw != b"".join(exact_json_bytes(row) + b"\n" for row in rows):
        raise GuardFailure("progress writer bytes differ: " + path.name)
    return rows


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT)


def file_locks(args, frozen, sources, evidence, label):
    for category, directory, expected in (
        ("repository", ROOT, sources), ("historical", ROOT, frozen["historical_files_sha256"]),
        ("native", args.vllm_source.parent, frozen["native_source_sha256"]),
    ):
        for name, digest in expected.items():
            evidence.equal(label + ":" + category + ":" + name, sha((directory / name).read_bytes()), digest)
    historical = json.loads((ROOT / "examples/pd_session_v1/expectations.json").read_bytes())
    evidence.equal(label + ":prompt-fixture", sha(baseline.TRACE_PATH.read_bytes()), historical["frontend"]["fixture_sha256"])
    native = frozen["native_environment"]
    config = args.hf_hub_cache / ("models--" + native["checkpoint"].replace("/", "--")) / "snapshots" / native["checkpoint_revision"] / "config.json"
    evidence.equal(label + ":model-config", sha(config.read_bytes()), native["model_config_sha256"])
    return config.resolve()


def protocol(args, frozen, evidence):
    git("merge-base", "--is-ancestor", FREEZE_COMMIT, "HEAD")
    git("diff", "--quiet", "HEAD", "--")
    relative = HERE.relative_to(ROOT)
    for name in SOURCES:
        git("ls-files", "--error-unmatch", name)
    for name in ("expectations.json", "expectations.md"):
        evidence.equal("protocol:freeze:" + name, sha((HERE / name).read_bytes()),
                       sha(git("show", f"{FREEZE_COMMIT}:{(relative / name).as_posix()}")))
    for name, digest in frozen["preimplementation_source_sha256"].items():
        evidence.equal("protocol:before-implementation:" + name, sha(git("show", frozen["as_of_commit"] + ":" + name)), digest)
    sources = {name: sha((ROOT / name).read_bytes()) for name in SOURCES}
    config = file_locks(args, frozen, sources, evidence, "before")
    evidence.equal("protocol:known-receipt", sha(args.known_native_receipt.read_bytes()),
                   frozen["chronology"]["known_native_receipt_sha256"])
    bounds = frozen["bounds"]
    floor = bounds["resident_per_rank_weight_bytes"] * 10**12 // bounds["memory_bytes_per_second"]
    evidence.equal("protocol:conditional-streaming-floor", floor, bounds["resident_streaming_floor_ps"])
    physical = {}
    for prompt in (8, 16):
        bits = prompt * bounds["cache_bytes_per_prompt_token"] * 8
        lower = bits * 10**12 // bounds["link_floor_bps"]
        upper = bits * 10**12 // bounds["declared_ceiling_link_bps"] + bounds["declared_ceiling_propagation_ps"]
        physical[str(prompt)] = {"handoff_floor_ps": lower, "handoff_ceiling_ps": upper, "conditional_resident_floor_ps": floor}
        evidence.check("protocol:handoff-bounds:" + str(prompt), all(lower <= handoff <= upper for handoff in (100000000, 200000000)))
        service = frozen["known_services_ps"][str(prompt)]
        evidence.check("protocol:resident-surrogate:" + str(prompt), all(value >= floor for value in (service["prefill"], *service["decode"])))
    references = {process["id"]: {cell["id"]: native_reference(process, cell, frozen["known_services_ps"])
                                for cell in process["cells"]} for process in frozen["native_processes"]}
    # These files are written before any campaign output is inspected.
    write(args.output_root / "source-manifest.json", sources)
    write(args.output_root / "physical-bounds.json", physical)
    write(args.output_root / "reference-timelines.json", references)
    evidence.finish("protocol")
    return sources, config, references


def launch(args, spec, frozen, monitors):
    path = args.output_root / spec["id"]
    path.mkdir()
    env = os.environ.copy()
    for name in ("VLLM_CACHE_ROOT", "XDG_CACHE_HOME", "TMPDIR"):
        cache = args.output_root / "process-caches" / spec["id"] / name.lower()
        cache.mkdir(parents=True)
        env[name] = str(cache)
    env.update(PYTHONPATH=str(ROOT), PYTHONDONTWRITEBYTECODE="1", HF_HUB_OFFLINE="1",
               TRANSFORMERS_OFFLINE="1", HF_HUB_CACHE=str(args.hf_hub_cache),
               VLLM_ENABLE_V1_MULTIPROCESSING="0", SIMLLM_VLLM_WORKER_MODE="skeleton",
               CUDA_VISIBLE_DEVICES="", HIP_VISIBLE_DEVICES="")
    command = [str(args.native_python), "-m", "examples.independent_engine_completion_v1.native",
               "--process", spec["id"], "--vllm-source", str(args.vllm_source),
               "--source-manifest", str(args.output_root / "source-manifest.json"), "--output-root", str(path)]
    monitor = {"command": command, "pid": None, "sampled_max_current_rss_kib": 0,
               "exit_code": None, "stopping_reason": None}
    monitors[spec["id"]] = monitor
    limits = frozen["limits"]
    started = time.monotonic()
    process = None
    try:
        with (path / "native.log").open("wb") as log:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            monitor["pid"] = process.pid
            while process.poll() is None:
                try:
                    rows = Path(f"/proc/{process.pid}/status").read_text().splitlines()
                except FileNotFoundError:
                    rows = []
                current = next((int(row.split()[1]) for row in rows if row.startswith("VmRSS:")), 0)
                monitor["sampled_max_current_rss_kib"] = max(current, monitor["sampled_max_current_rss_kib"])
                if current >= limits["native_rss_cap_kib"]:
                    raise GuardFailure("frozen resident-memory stop reached")
                if time.monotonic() - started >= limits["native_timeout_seconds"]:
                    raise GuardFailure("frozen native timeout reached")
                try:
                    process.wait(timeout=limits["native_rss_sample_seconds"])
                except subprocess.TimeoutExpired:
                    pass
            if process.returncode:
                raise GuardFailure("native process failed: " + spec["id"])
    except BaseException as error:
        monitor["stopping_reason"] = str(error)
        if process is not None and process.poll() is None:
            process.kill()
            process.wait()
        raise
    finally:
        monitor["exit_code"] = None if process is None else process.returncode
        monitor["wall_seconds"] = time.monotonic() - started
        write(path / "process.json", monitor)
    return monitor


def capture_native(args, spec, frozen, monitors, raw_receipts, root_receipts):
    """Lock the first complete receipt before any native output is admitted."""
    try:
        launch(args, spec, frozen, monitors)
    finally:
        name, path = spec["id"], args.output_root / spec["id"]
        if path.exists():
            raw_receipts[name] = file_receipts(path)
            receipt_path = args.output_root / (name + "-raw-receipts.json")
            write(receipt_path, raw_receipts[name])
            root_receipts[receipt_path.name] = {"sha256": sha(receipt_path.read_bytes()), "bytes": receipt_path.stat().st_size}


def admit(data, spec, frozen, sources, args, monitor, references, evidence):
    label = spec["id"]
    evidence.fields(label + ":fields", data, "schema process_id authority sources_before sources_after identity retained_before retained_after "
                    "selected_before selected_after construction controls cells steps clock_advances checkpoints projections sinks final_clock_ps runtime_failure")
    evidence.equal(label + ":schema", data["schema"], "simllm-independent-engine-completion-native-v1")
    evidence.equal(label + ":process", data["process_id"], label)
    check_sources(data, spec, frozen, sources, args, monitor, evidence)
    check_engines(data, spec, frozen, evidence)
    if spec["baseline_controls_first"]:
        check_baselines(data, frozen, evidence)
    else:
        evidence.equal(label + ":no-extra-controls", data["controls"], [])
    steps, owners = check_cells(data, spec, frozen, references, evidence)
    check_projections(data, spec, steps, evidence)
    check_sinks_and_domains(data, spec, steps, evidence)
    check_checkpoints(data, spec, steps, owners, evidence)
    evidence.finish(label + ":admission")


def admit_disk(path, data, spec, monitor, receipt_map, evidence):
    label = spec["id"] + ":disk"
    reread_tree(path, receipt_map, evidence, label + ":before")
    evidence.equal(label + ":native", read(path / "native.json"), data)
    evidence.equal(label + ":receipt", read(path / "receipt.json"),
                   {"pid": monitor["pid"], "native_sha256": receipt_map["native.json"]["sha256"]})
    evidence.equal(label + ":monitor", read(path / "process.json"), monitor)
    evidence.equal(label + ":construction", progress_rows(path / "construction-progress.jsonl"), data["construction"])
    expected_progress = [*data["controls"], *[row for cell in data["cells"] for row in cell["requests"]]]
    evidence.equal(label + ":requests", progress_rows(path / "request-progress.jsonl"), expected_progress)
    for cell in data["cells"]:
        evidence.equal(label + ":cell:" + cell["id"], read(path / (cell["id"] + ".json")), cell)
    if spec["mode"] == "independent":
        evidence.equal(label + ":checkpoints", progress_rows(path / "native-checkpoints.jsonl"), data["checkpoints"])
    else:
        evidence.check(label + ":checkpoint-off", not (path / "native-checkpoints.jsonl").exists())
    streams = sorted(path.rglob("step-records.jsonl"))
    evidence.equal(label + ":stream-count", len(streams), len(data["retained_before"]))
    for engine in data["retained_before"]:
        stream = next(item for item in streams if item.parent.name == engine["engine_id"])
        evidence.equal(label + ":steps:" + engine["engine_id"], step_stream(stream),
                       [step["record"] for step in data["steps"] if step["engine_id"] == engine["engine_id"]])


def runtime_corruption(name, path):
    """Exercise the actual pending authority, including coherent public forgeries."""
    from simllm.backends import HtsimStepSink, HtsimStepSinkConfig
    from simllm.compute import ModelDims, RooflineProvider
    from simllm.core import RequestPhase, ScheduledRequest, StepRecord, VirtualClock
    from simllm.core.engine_steps import EngineStepRuntime
    from simllm.core.value_snapshot import value_snapshot
    from simllm.placement import declared_manifest

    clock = VirtualClock()
    runtime = EngineStepRuntime(clock)
    if name == "two-authorities-one-step":
        try:
            EngineStepRuntime(clock)
        except ValueError as error:
            if runtime.failure is not None or len(clock) or runtime.events or runtime.results:
                raise GuardFailure("second authority altered the original owner") from error
            runtime.close()
            return str(error)
        raise GuardFailure("second timing authority was accepted")
    sink = HtsimStepSink(HtsimStepSinkConfig(
        profile="rnic-nn-fluid", tp_ranks=(0, 1), dims=ModelDims(
            num_layers=1, hidden_size=64, intermediate_size=128, num_heads=4, num_kv_heads=4,
            head_size=16, vocab_size=256, dtype_bytes=2), workdir=path, provider=RooflineProvider(),
        placement_manifest=declared_manifest(tp=2, nodes=1, gpus_per_node=2),
        collective_fixed_cost_envelope="intra-node-fixed-cost-v1", collective_fixed_cost_arm="lower"))
    record = StepRecord(0, 0, [ScheduledRequest("request", RequestPhase.DECODE, 1, context_length=16)], num_sampled=1)
    price = sink.prepare_deferred(record)
    receipt = None

    def publish(result):
        sink.publish_deferred(price, runtime, receipt)

    receipt = runtime.submit("engine", record, price.result, guard=price.validate, publish=publish,
                             publications=(sink.deferred_publication(price),))
    if name == "changed-input-after-submit":
        record.scheduled.append(ScheduledRequest("foreign", RequestPhase.DECODE, 1))
    elif name == "same-object-provider-mutation":
        sink.config.provider.efficiency = 0.8
    elif name == "resolved-configuration-field-mutation":
        sink.config.resolved_collective_evidence_class = "changed"
    elif name == "coherent-prepared-payload-mutation":
        changed = replace(price.simulation, outcome=replace(
            price.simulation.outcome, makespan_ps=price.simulation.outcome.makespan_ps + 1))
        object.__setattr__(price, "simulation", changed)
        object.__setattr__(price, "simulation_state", value_snapshot(changed))
        price.validate()
    elif name != "early-sink-publication":
        raise GuardFailure("unknown runtime corruption: " + name)
    try:
        if name == "early-sink-publication":
            sink.publish_deferred(price, runtime, receipt)
        else:
            runtime.advance_to(price.result.completed_at_ps)
            runtime.complete_due()
    except (RuntimeError, ValueError) as error:
        if not runtime.failure or runtime.results or runtime.visits or sink.outcomes:
            raise GuardFailure("corrupt pending state did not remain unpublished and poisoned") from error
        try:
            runtime.complete_due()
        except RuntimeError:
            return str(error)
        raise GuardFailure("poisoned state retried completion") from error
    raise GuardFailure("runtime corruption was accepted: " + name)


RUNTIME_MUTATIONS = {
    "changed-input-after-submit", "two-authorities-one-step", "same-object-provider-mutation",
    "resolved-configuration-field-mutation", "coherent-prepared-payload-mutation", "early-sink-publication",
}


def mutate_data(name, data):
    if name == "early-native-token":
        row = next(row for row in data["checkpoints"] if row["phase"] == "submitted" and row["native"]["visible"])
        next(iter(row["native"]["visible"].values()))["output_token_ids"].append(512)
    elif name == "early-producer-handoff":
        data["cells"][0]["requests"][0]["result"]["handoff"]["completed_at_ps"] -= 1
    elif name == "duplicate-step-retirement":
        data["projections"]["completed"].append(deepcopy(data["projections"]["completed"][0]))
    elif name == "foreign-engine-receipt":
        data["projections"]["completed"][0]["receipt"]["engine_id"] = "foreign-engine"
    elif name == "missing-completion-event":
        data["projections"]["events"].pop()
    elif name == "changed-selected-configuration":
        data["selected_before"][0]["host_model"]["profile_id"] = "changed"
        data["selected_after"] = deepcopy(data["selected_before"])
    elif name == "unexplained-wall-idle":
        data["clock_advances"][0]["after_ps"] += 1
    elif name == "wrong-baseline-comparison":
        data["controls"][0]["comparison_sha256"] = "0" * 64
    elif name == "wrong-native-source-receipt":
        key = next(iter(data["sources_before"]["native"]["sha256"]))
        data["sources_before"]["native"]["sha256"][key] = "0" * 64
        data["sources_after"] = deepcopy(data["sources_before"])
    else:
        raise GuardFailure("unknown data corruption: " + name)


def semantic_controls(all_data, frozen, sources, args, monitors, references, evidence):
    controls = []
    specs = {row["id"]: row for row in frozen["native_processes"]}
    for name in frozen["semantic_mutations"]:
        if name in RUNTIME_MUTATIONS:
            with TemporaryDirectory(dir=args.output_root, prefix="semantic-control-") as scratch:
                reason = runtime_corruption(name, Path(scratch))
        elif name == "disk-only-raw-receipt-change":
            with TemporaryDirectory(dir=args.output_root, prefix="raw-control-") as scratch:
                path = Path(scratch)
                write(path / "receipt.json", {"pid": 1, "sha256": "a" * 64})
                receipt = file_receipts(path)
                write(path / "receipt.json", {"pid": 1, "sha256": "b" * 64})
                try:
                    reread_tree(path, receipt, Evidence([]), "corrupt-disk")
                except GuardFailure as error:
                    reason = str(error)
                else:
                    raise GuardFailure("changed disk receipt was accepted")
        else:
            selected = "serialized-p1-d1" if name == "wrong-baseline-comparison" else "independent-p1-d1"
            data = deepcopy(all_data[selected])
            mutate_data(name, data)
            try:
                admit(data, specs[selected], frozen, sources, args, monitors[selected], references,
                      Evidence(frozen["required_stages"]))
            except GuardFailure as error:
                reason = str(error)
            else:
                raise GuardFailure("semantic corruption was accepted: " + name)
        evidence.check("semantic-control:" + name, bool(reason))
        controls.append({"name": name, "rejected_by": reason})
    evidence.equal("semantic-control:domain", [row["name"] for row in controls], frozen["semantic_mutations"])
    evidence.finish("semantic-corruptions")
    return controls


def complete(evidence, frozen):
    observed = {}
    for row in evidence.relations:
        observed.setdefault(row["family"], []).append(row["name"])
    expected = {family: sorted(spec["instances"]) for family, spec in frozen["behavioral_families"].items()}
    evidence.equal("complete:behavior-domain", {family: sorted(rows) for family, rows in observed.items()}, expected)
    evidence.equal("complete:behavior-count", len(evidence.relations), frozen["behavioral_instance_total"])
    evidence.equal("complete:oracle-count", len(evidence.oracles), sum(
        value for value in frozen["exact_oracles"].values() if type(value) is int))
    evidence.finish("final-identities")
    evidence.equal("complete:stages", sorted(evidence.finished_stages), sorted(frozen["required_stages"]))


def summary(evidence, frozen, *, valid=False, **extra):
    return {
        "schema": "simllm-independent-engine-completion-summary-v1", "task": "CORE-68",
        "verdict": "PASS" if valid else "VOID", "expectations_commit": FREEZE_COMMIT,
        "fatal_guards_violated": [row for row in evidence.guards if not row["passed"]],
        "unscored_guard_count": len(evidence.guards), "exact_oracle_count": len(evidence.oracles),
        "behavioral_instance_count": len(evidence.relations),
        "behavioral_family_count": len({row["family"] for row in evidence.relations}),
        "behavioral_score": 1 if valid else None, "finished_stages": sorted(evidence.finished_stages),
        "required_stage_count": len(frozen["required_stages"]), **extra,
    }


def execute(args):
    frozen = json.loads((HERE / "expectations.json").read_bytes())
    evidence = Evidence(frozen["required_stages"])
    if args.output_root == ROOT or ROOT in args.output_root.parents:
        raise ValueError("raw native output must be outside the repository")
    args.output_root.mkdir(parents=True, exist_ok=False)
    args.repository_root = ROOT
    source = git("rev-parse", "HEAD").decode().strip()
    all_data, monitors, raw_receipts, controls, root_receipts = {}, {}, {}, [], {}
    try:
        sources, config, references = protocol(args, frozen, evidence)
        components = [component_run(spec) for spec in frozen["component_cases"]]
        write(args.output_root / "component-cases.json", components)
        root_receipts = {name: {"sha256": sha((args.output_root / name).read_bytes()), "bytes": (args.output_root / name).stat().st_size}
                         for name in ("source-manifest.json", "physical-bounds.json", "reference-timelines.json", "component-cases.json")}
        evidence.equal("root:source-manifest", read(args.output_root / "source-manifest.json"), sources)
        evidence.equal("root:reference-timelines", read(args.output_root / "reference-timelines.json"), references)
        evidence.equal("root:components", read(args.output_root / "component-cases.json"), components)
        check_components(components, frozen, evidence)
        for spec in frozen["native_processes"]:
            name, path = spec["id"], args.output_root / spec["id"]
            print("Starting " + name, flush=True)
            capture_native(args, spec, frozen, monitors, raw_receipts, root_receipts)
            evidence.finish(name + ":capture")
            data = read(path / "native.json")
            evidence.equal(name + ":config-path", data["identity"]["config_path"], str(config))
            admit_disk(path, data, spec, monitors[name], raw_receipts[name], evidence)
            admit(data, spec, frozen, sources, args, monitors[name], references, evidence)
            all_data[name] = data
            print("Qualified " + name, flush=True)
        check_native_relations(all_data, frozen, evidence)
        controls = semantic_controls(all_data, frozen, sources, args, monitors, references, evidence)
        evidence.equal("complete:fresh-processes", len({row["pid"] for row in monitors.values()}), len(frozen["native_processes"]))
        evidence.equal("complete:native-requests", sum(
            len(data["controls"]) + sum(len(cell["requests"]) for cell in data["cells"]) for data in all_data.values()),
                       frozen["native_request_total"])
        evidence.equal("complete:native-cells", sum(len(data["controls"]) + len(data["cells"]) for data in all_data.values()),
                       frozen["native_cell_total_including_baseline"])
        for name, receipts in raw_receipts.items():
            reread_tree(args.output_root / name, receipts, evidence, "final:" + name)
        evidence.equal("root:published-domain", sorted(path.name for path in args.output_root.iterdir() if path.is_file()),
                       sorted(root_receipts))
        for name, receipt in root_receipts.items():
            path = args.output_root / name
            evidence.equal("root:reread:" + name, {"sha256": sha(path.read_bytes()), "bytes": path.stat().st_size}, receipt)
        file_locks(args, frozen, sources, evidence, "after")
        evidence.equal("after:known-receipt", sha(args.known_native_receipt.read_bytes()),
                       frozen["chronology"]["known_native_receipt_sha256"])
        git("diff", "--quiet", "HEAD", "--")
        evidence.equal("after:source-commit", git("rev-parse", "HEAD").decode().strip(), source)
        complete(evidence, frozen)
        result = summary(evidence, frozen, valid=True, failure=None)
    except Exception as error:  # noqa: BLE001
        (args.output_root / "exception.txt").write_text(traceback.format_exc(), encoding="utf-8")
        result = summary(evidence, frozen, failure=str(error))
    # Even a failed subprocess has its log and partial progress retained.
    retained = {spec["id"]: file_receipts(args.output_root / spec["id"]) for spec in frozen["native_processes"]
                if (args.output_root / spec["id"]).exists()}
    result.update(source_commit=source, admitted_processes=list(all_data), monitors=monitors,
                  initial_raw_receipts=raw_receipts, raw_receipts=retained, root_receipts=root_receipts, corruption_controls=controls,
                  cells=[{"process": name, "id": cell["id"], "makespan_ps": cell["end_ps"] - cell["start_ps"],
                          "requests": len(cell["requests"])}
                         for name, data in all_data.items() for cell in data["cells"]])
    write(args.output_root / "checks.json", {"guards": evidence.guards, "oracles": evidence.oracles, "relations": evidence.relations})
    write(args.output_root / "summary.json", result)
    print(json.dumps({key: result[key] for key in ("verdict", "failure", "admitted_processes", "behavioral_score")}), flush=True)
    return 0 if result["verdict"] == "PASS" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("native-python", "vllm-source", "hf-hub-cache", "known-native-receipt", "output-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    for name in ("vllm_source", "hf_hub_cache", "known_native_receipt", "output_root"):
        setattr(args, name, getattr(args, name).resolve())
    args.native_python = args.native_python.absolute()
    return execute(args)


if __name__ == "__main__":
    sys.exit(main())
