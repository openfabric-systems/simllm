"""Capture full reader values and actual local jobs from the selected package."""

import argparse
import cProfile
import json
import os
import sys
import traceback
from pathlib import Path

from examples.publication_snapshot_v1 import worker as helper
from examples.publication_snapshot_v1.common import (
    pack,
    packages,
    profile_rows,
    sha,
    study_origins,
    write,
)
from examples.snapshot_dispatch_v1.common import binding_witnesses, runtime_identity
from examples.snapshot_dispatch_v1.semantics import observe, primitive_input
from simllm.core.value_snapshot import value_snapshot

HERE = Path(__file__).resolve().parent


def profiled(function, path):
    if sys.getprofile() is not None:
        raise ValueError("profile hook already occupied")
    profiler = cProfile.Profile()
    try:
        return profiler.runcall(function)
    finally:
        profiler.dump_stats(str(path))
        if sys.getprofile() is not None:
            raise ValueError("profile hook was not restored")


def reader(spec, frozen, workdir, output, root):
    sink = helper.make_sink(frozen, workdir)
    for name in frozen["synthetic_populated_names"]:
        getattr(sink, name).extend(helper.SyntheticRow(tuple(range(spec["width"])))
                                 for _ in range(spec["history"]))
    price = sink.prepare_deferred(helper.record(0, 0))
    binding = sink.deferred_publication(price)
    raw = binding.read_values(binding.payload)
    expected_ids = (id(sink), id(sink.config), id(sink.config.provider))
    expected_bindings = tuple(tuple(id(part) for part in entry) for entry in sink._deferred_bindings())
    if raw[2:5] != expected_ids or raw[7] != expected_bindings:
        raise ValueError("actual reader identity slots changed")
    initial = pack(value_snapshot(raw))
    write(output / (spec["id"] + "-reader-before.json"), initial)
    profile = output / (spec["id"] + "-reader.pstats")
    result = profiled(lambda: value_snapshot(binding.read_values(binding.payload)), profile)
    row = {"id": spec["id"], "spec": spec, "snapshot": pack(result), "before": initial,
           "after": pack(value_snapshot(binding.read_values(binding.payload))),
           "identities": list(expected_ids), "binding_identities": [list(entry) for entry in expected_bindings],
           "bindings": helper.binding_names(sink),
           "binding_witnesses": binding_witnesses(sink, helper.binding_names(sink), root), "profiles": profile_rows(profile)}
    write(output / (spec["id"] + "-reader.json"), row)
    return row


def execute(args):
    root, output = args.repository.resolve(), args.output.resolve()
    frozen = json.loads((HERE / "expectations.json").read_bytes())
    old = json.loads((HERE.parent / "publication_snapshot_v1/expectations.json").read_bytes())
    source_before = packages(root)
    runtime_before = runtime_identity()
    result = {"arm": args.arm, "mode": args.mode, "pid": os.getpid(), "interpreter": sys.executable,
              "repository": str(root), "worker_path": str(Path(__file__).resolve()), "worker_sha256": sha(__file__),
              "sources_before": source_before, "runtime_before": runtime_before}
    if args.mode != "main":
        result["semantics"] = [observe(args.mode, value_snapshot)]
    else:
        result.update(primitives=[], readers=[], jobs=[], semantics=[], mutations=[], helpers=[])
        workdir = output.parent / "sink-work"
        for spec in frozen["cases"]:
            value = primitive_input(spec, frozen)
            before = pack(value_snapshot(value))
            write(output / (spec["id"] + "-primitive-before.json"), before)
            profile = output / (spec["id"] + "-primitive.pstats")
            snapshot = profiled(lambda value=value: value_snapshot(value), profile)
            row = {"id": spec["id"], "spec": spec, "snapshot": pack(snapshot), "before": before,
                   "after": pack(value_snapshot(value)), "profiles": profile_rows(profile)}
            write(output / (spec["id"] + "-primitive.json"), row)
            result["primitives"].append(row)
            result["readers"].append(reader(spec, old, workdir, output, root))
        for spec in old["jobs"]:
            profile = output / (spec["id"] + "-job.pstats")
            job = profiled(lambda spec=spec: helper.capture_job(spec, old, workdir), profile)
            write(output / (spec["id"] + "-job.json"), job)
            result["jobs"].append({"job": job, "profiles": profile_rows(profile)})
        for name in frozen["semantics"]:
            row = observe(name, value_snapshot)
            write(output / (name + "-semantic.json"), row)
            result["semantics"].append(row)
        for group, names in (("mutations", old["common_corruptions"]), ("helpers", old["new_helper_corruptions"])):
            for name in names:
                result[group].append(helper.capture_mutation(name, old, workdir, output))
    result.update(sources_after=packages(root), runtime_after=runtime_identity(), study_origins=study_origins(),
                  origins={name: str(Path(module.__file__).resolve()) for name, module in sys.modules.items()
                           if (name == "simllm" or name.startswith("simllm.")) and getattr(module, "__file__", None)})
    write(output / "worker.json", result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", choices=("before", "after"), required=True)
    parser.add_argument("--mode", choices=("main", "fraction-before", "fraction-during"), default="main")
    args = parser.parse_args()
    try:
        execute(args)
    except BaseException:
        (args.output / "exception.txt").write_text(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
