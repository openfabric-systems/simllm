"""Freeze actual sources, runtimes and physical bounds before any serving run."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

from examples.publication_snapshot_v1.common import study_sources

from .common import (
    DEADLINE_FREEZE,
    FREEZE,
    HERE,
    ROOT,
    git,
    packages,
    read,
    sha,
    validate_sources,
    write,
)
from .network_checks import geometry

ALIAS_FREEZE = "4c0cf5f4494d4534f4cdd481543827bbb3528032"
SERVICE_VECTOR_FREEZE = "9b520748dfe9a7ba9d17c72a98e71148fa0660be"


def inputs():
    frozen, deadline, aliases = (json.loads((HERE / name).read_bytes()) for name in
                                ("expectations.json", "deadline-expectations.json", "alias-expectations.json"))
    frozen["native_source_sha256"].update(aliases["native_source_sha256"])
    dependency = json.loads((ROOT / "examples/independent_engine_completion_v1/expectations.json").read_bytes())
    return frozen, deadline, aliases, dependency


def roots(args):
    return {"before": args.before_repository.resolve(), "after": ROOT}


def selected_root(args, spec):
    return roots(args)[spec.get("source", "after")]


def command(args, spec):
    arm = spec.get("source", "after")
    return [str(args.native_python), str(HERE / "native.py"), "--process", spec["id"],
            "--repository", str(selected_root(args, spec)), "--vllm-source", str(args.vllm_source),
            "--source-manifest", str(args.output_root / (arm + "-packages.json")),
            "--htsim-rnic", str(args.htsim_rnic), "--output-root", str(args.output_root / spec["id"])]


def environment(args, spec):
    cache = args.output_root / "process-caches" / spec["id"]
    values = {name: str(cache / name.lower()) for name in ("VLLM_CACHE_ROOT", "XDG_CACHE_HOME", "TMPDIR")}
    values.update(PYTHONPATH=os.pathsep.join(map(str, (selected_root(args, spec), ROOT))),
                  PYTHONDONTWRITEBYTECODE="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                  HF_HUB_CACHE=str(args.hf_hub_cache), VLLM_ENABLE_V1_MULTIPROCESSING="0",
                  SIMLLM_VLLM_WORKER_MODE="skeleton", CUDA_VISIBLE_DEVICES="", HIP_VISIBLE_DEVICES="",
                  SIMLLM_CHILD_LIFETIME_MARKER_DIR=str(args.output_root / spec["id"] / "markers"),
                  SIMLLM_CHILD_LIFETIME_RUN_NONCE=spec["id"])
    return values


def tracked_clean(root):
    git(root, "diff", "--quiet", "HEAD", "--")
    if git(root, "ls-files", "--others", "--exclude-standard", "simllm", "examples", "tests"):
        raise ValueError("untracked campaign source remains")


def runtime_probe(args):
    """Capture runtime identity or its failed output before checking success."""
    command = [str(args.native_python), "-c", "import json; from examples.snapshot_dispatch_v1.common import runtime_identity; print(json.dumps(runtime_identity(), sort_keys=True, separators=(',', ':')))"]
    write(args.output_root / "runtime-probe-command.json", command)
    result, primary, capture_errors = None, None, []
    stdout = stderr = b""
    try:
        result = subprocess.run(command, cwd=ROOT, env=dict(os.environ, PYTHONPATH=str(ROOT), PYTHONDONTWRITEBYTECODE="1"),
                                capture_output=True, timeout=60, check=False)
        stdout, stderr = result.stdout, result.stderr
        result.check_returncode()
    except BaseException as error:  # noqa: BLE001, preserve failed and interrupted probe output.
        primary = error
        if isinstance(error, subprocess.TimeoutExpired):
            stdout, stderr = error.output or b"", error.stderr or b""
    outcome = {"exit_code": None if result is None else result.returncode,
               "failure": None if primary is None else {"type": type(primary).__name__, "message": str(primary)}}
    surfaces = (("stdout", lambda: (args.output_root / "runtime-probe.stdout").write_bytes(stdout)),
                ("stderr", lambda: (args.output_root / "runtime-probe.stderr").write_bytes(stderr)),
                ("outcome", lambda: write(args.output_root / "runtime-probe-outcome.json", outcome)))
    for name, callback in surfaces:
        try:
            callback()
        except BaseException as error:  # noqa: BLE001, attempt every independent evidence surface.
            capture_errors.append({"surface": name, "type": type(error).__name__, "message": str(error)})
            primary = primary or error
    if primary is not None:
        if capture_errors:
            primary.__dict__["receipt_failure"] = capture_errors
        raise primary
    return read(args.output_root / "runtime-probe.stdout")


def locks(args, frozen, sources, evidence, label):
    for arm, root in roots(args).items():
        tracked_clean(root)
        evidence.equal(label + ":commit:" + arm, git(root, "rev-parse", "HEAD"), sources[arm]["commit"])
        evidence.equal(label + ":packages:" + arm, packages(root), sources[arm]["packages"])
        evidence.equal(label + ":helpers:" + arm, study_sources(root), sources[arm]["helpers"])
        evidence.equal(label + ":tests:" + arm, {name: sha(root / name) for name in sources[arm]["tests"]}, sources[arm]["tests"])
    for name, digest in sources["inputs"].items():
        evidence.equal(label + ":input:" + name, sha(ROOT / name), digest)
    for name, digest in frozen["native_source_sha256"].items():
        evidence.equal(label + ":native:" + name, sha(args.vllm_source.parent / name), digest)
    for name, row in sources["external"].items():
        evidence.equal(label + ":external:" + name, sha(Path(row["path"])), row["sha256"])
    git(args.backend_repository, "diff", "--quiet", "HEAD", "--")
    evidence.equal(label + ":backend-commit", git(args.backend_repository, "rev-parse", "HEAD"), frozen["backend"]["source_commit"])
    evidence.equal(label + ":backend-tree", git(args.backend_repository, "rev-parse", "HEAD^{tree}"), sources["backend_tree"])


def freeze(args, frozen, deadline, aliases, dependency, evidence):
    relative = HERE.relative_to(ROOT)
    for commit, stem in ((FREEZE, "expectations"), (DEADLINE_FREEZE, "deadline-expectations"),
                         (ALIAS_FREEZE, "alias-expectations"), (SERVICE_VECTOR_FREEZE, "service-vector-expectations")):
        git(ROOT, "merge-base", "--is-ancestor", commit, "HEAD")
        for extension in ("json", "md"):
            name = stem + "." + extension
            raw = subprocess.check_output(["git", "show", commit + ":" + (relative / name).as_posix()], cwd=ROOT)
            evidence.equal("freeze:" + name, sha(HERE / name), hashlib.sha256(raw).hexdigest())
    service_vector = json.loads((HERE / "service-vector-expectations.json").read_bytes())
    for name, digest in service_vector["preservation_sha256"].items():
        evidence.equal("freeze:service-vector-preservation:" + name, sha(ROOT / name), digest)
    for name, actual in (("native_processes", frozen["native_process_count"]), ("native_requests", frozen["native_request_count"]),
                         ("exact_oracle_vectors", sum(frozen["exact_oracle_vectors"].values())),
                         ("behavioral_instances", frozen["behavioral_instance_total"]),
                         ("behavioral_families", len(frozen["behavioral_families"]))):
        evidence.equal("freeze:service-vector-inventory:" + name, actual, service_vector[name])
    evidence.equal("freeze:before-commit", git(args.before_repository, "rev-parse", "HEAD"), frozen["as_of_commit"])
    for name, digest in frozen["input_dependencies_sha256"].items():
        evidence.equal("freeze:dependency:" + name, sha(ROOT / name), digest)
    prerequisite = json.loads((ROOT / "examples/independent_engine_completion_v1/results.json").read_bytes())
    evidence.check("freeze:qualified-prerequisite", prerequisite["task"] == "CORE-68"
                   and prerequisite["task_closed"] is True and prerequisite["verdict"] == "PASS")
    native = frozen["native_environment"]
    config = args.hf_hub_cache / ("models--" + native["checkpoint"].replace("/", "--")) / "snapshots" / native["checkpoint_revision"] / "config.json"
    evidence.equal("freeze:model-config", sha(config), native["model_config_sha256"])
    evidence.equal("freeze:known-native-receipt", sha(args.known_native_receipt), dependency["chronology"]["known_native_receipt_sha256"])
    known_backend = read(args.known_backend_receipt)
    evidence.equal("freeze:known-backend-source", known_backend["sources"]["candidate_backend"]["commit"], frozen["backend"]["source_commit"])
    evidence.equal("freeze:known-backend-bytes", known_backend["executables"]["candidate_binary"]["sha256"], frozen["backend"]["binary_sha256"])
    evidence.equal("freeze:backend-binary", sha(args.htsim_rnic), frozen["backend"]["binary_sha256"])
    hardware = {"schema": "simllm-rnic-effective-hardware-v1", "network": {"enabled": True},
                "dma": {"enabled": False}, "qpc": {"enabled": False},
                "work_queue": {"cq_depth": 64, "sq_depth": 64, "cqe_write_service_ps": 0,
                               "doorbell_service_ps": 0, "scheduler_service_ps": 0, "wqe_fetch_service_ps": 0}}
    evidence.equal("freeze:effective-hardware", known_backend["hardware"]["effective_hardware"], hardware)
    evidence.equal("freeze:hardware-digest", known_backend["hardware"]["effective_hardware_sha256"], frozen["backend"]["effective_hardware_sha256"])
    evidence.equal("freeze:hardware-bytes", hashlib.sha256(json.dumps(hardware, sort_keys=True, separators=(",", ":")).encode()).hexdigest(), frozen["backend"]["effective_hardware_sha256"])
    sources = {}
    for arm, root in roots(args).items():
        tracked_clean(root)
        sources[arm] = {"commit": git(root, "rev-parse", "HEAD"), "packages": packages(root), "helpers": study_sources(root),
                        "tests": {name: sha(root / name) for name in git(root, "ls-files", "tests").splitlines()}}
        write(args.output_root / (arm + "-packages.json"), sources[arm]["packages"])
    sources["inputs"] = {name: sha(ROOT / name) for name in (
        *frozen["input_dependencies_sha256"], *service_vector["preservation_sha256"],
        *(str(relative / (stem + "." + extension)) for stem in
        ("expectations", "deadline-expectations", "alias-expectations", "service-vector-expectations") for extension in ("json", "md")),
        "examples/pd_session_v1/expectations.json", "examples/preplay_trace_v1/granite_length_cap.jsonl",
        "examples/independent_engine_completion_v1/results.json")}
    sources["external"] = {name: {"path": str(path.resolve()), "sha256": sha(path)} for name, path in (
        ("interpreter", args.native_python), ("model-config", config), ("htsim-rnic", args.htsim_rnic),
        ("known-native-receipt", args.known_native_receipt), ("known-backend-receipt", args.known_backend_receipt))}
    sources["backend_tree"] = git(args.backend_repository, "rev-parse", "HEAD^{tree}")
    sources["runtime"] = runtime_probe(args)
    evidence.equal("freeze:runtime-executable", sources["runtime"]["executable_sha256"], sources["external"]["interpreter"]["sha256"])
    for name, row in sources["runtime"]["libraries"].items():
        sources["external"]["runtime-library:" + name] = row
    bounds = dependency["bounds"]
    floor = bounds["resident_per_rank_weight_bytes"] * 10**12 // bounds["memory_bytes_per_second"]
    evidence.equal("freeze:resident-floor", floor, 40108032)
    physical = []
    for spec in frozen["native_processes"]:
        service = frozen["known_services_ps"][str(spec["prompt_tokens"])]
        evidence.check("freeze:resident-service:" + spec["id"], all(value >= floor for value in [service["prefill"], *service["decode"]]))
        if spec["kind"] != "compatibility":
            total, shard, n, q = geometry(spec, frozen)
            fixed = frozen["backend"]["pcie_submission_ps"] + frozen["backend"]["propagation_ps"]
            upper = service["prefill"] + fixed + (16 * n + 1) * q + 2 * sum(service["decode"])
            physical.append({"process": spec["id"], "cache_bytes": total, "shard_bytes": shard, "packets_per_shard": n,
                "packet_serialization_ps": q, "conditional_compute_floor_ps": floor,
                "request_floor_ps": service["prefill"] + fixed + (n + 1) * q + sum(service["decode"]), "request_ceiling_ps": upper})
            if len(spec["admission_times_ps"]) > 1:
                evidence.check("freeze:idle-gap:" + spec["id"], spec["admission_times_ps"][1] > upper
                               and spec["admission_times_ps"][1] % q == 0)
    write(args.output_root / "physical-bounds.json", physical)
    write(args.output_root / "source-manifest.json", sources)
    locks(args, frozen, sources, evidence, "pre-run")
    return sources


def admit_sources(data, spec, args, frozen, sources, monitor, evidence):
    label, arm, root = spec["id"] + ":source", spec.get("source", "after"), selected_root(args, spec)
    for phase in ("before", "after"):
        snapshot = data["sources_" + phase]
        evidence.fields(label + ":fields:" + phase, snapshot, "repository commit packages origins helpers native runtime entry prompt")
        validate_sources(snapshot, root, sources[arm]["packages"], args.vllm_source, frozen)
        evidence.equal(label + ":commit:" + phase, snapshot["commit"], sources[arm]["commit"])
        evidence.equal(label + ":runtime:" + phase, snapshot["runtime"], sources["runtime"])
        for name, row in snapshot["helpers"].items():
            path = Path(row["origin"])
            selected = next((key for key, directory in ((arm, root), ("after", ROOT)) if path.is_relative_to(directory)), None)
            evidence.check(label + ":helper-root:" + phase + ":" + name, selected is not None)
            relative = path.relative_to(roots(args)[selected]).as_posix()
            evidence.equal(label + ":helper-bytes:" + phase + ":" + name, row["sha256"], sources[selected]["helpers"].get(relative))
        for name, row in data["sources_before"]["origins"].items():
            evidence.equal(label + ":origin-retained:" + phase + ":" + name, snapshot["origins"].get(name), row)
        for name, row in data["sources_before"]["helpers"].items():
            evidence.equal(label + ":helper-retained:" + phase + ":" + name, snapshot["helpers"].get(name), row)
    identity = data["identity"]
    evidence.fields(label + ":identity-fields", identity, "pid interpreter package version config_sha256 config_path clock_object_id "
                    "cuda_before cuda_after packet_backend_runs all_observed_alive unfinished_after native_has_work_after")
    evidence.equal(label + ":pid", identity["pid"], monitor["pid"])
    evidence.equal(label + ":interpreter", identity["interpreter"], str(args.native_python))
    evidence.equal(label + ":native-package", identity["package"], str(args.vllm_source.resolve()))
    evidence.equal(label + ":native-version", identity["version"], frozen["native_environment"]["vllm_version"])
    evidence.equal(label + ":model-config", [identity["config_path"], identity["config_sha256"]],
                   [sources["external"]["model-config"]["path"], frozen["native_environment"]["model_config_sha256"]])
    evidence.check(label + ":no-gpu-backend", identity["cuda_before"] is False and identity["cuda_after"] is False
                   and type(identity["packet_backend_runs"]) is int and identity["packet_backend_runs"] == 0)
    evidence.equal(label + ":command", monitor["command"], command(args, spec))
    evidence.equal(label + ":cwd", monitor["cwd"], str(root))
    evidence.equal(label + ":environment-overrides", monitor["environment_overrides"], environment(args, spec))
    evidence.check(label + ":host-envelope", monitor["exit_code"] == 0 and monitor["stopping_reason"] is None
                   and 0 < monitor["wall_seconds"] < frozen["limits"]["wall_seconds_per_process"]
                   and 0 < monitor["sampled_max_current_rss_kib"] * 1024 < frozen["limits"]["max_current_rss_bytes"])
