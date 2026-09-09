"""Run one exact source arm and flush each original result before admission."""

# ruff: noqa: BLE001

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent


def source(root):
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=root).decode().strip()
    if git("status", "--porcelain", "--untracked-files=all"):
        raise ValueError("capture requires a clean source tree")
    files = {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in git("ls-files").splitlines()
        if (root / name).is_file()
    }
    return {"commit": git("rev-parse", "HEAD"), "files": files}


def run(args):
    root = args.code_root.resolve()
    # Select the code domain before importing either model or shared helpers.
    sys.path.insert(0, str(root))
    sys.path.insert(1, str(HERE))
    import inputs as study_inputs

    from examples.routed_compute_v1.run_study import Evidence, primitive
    from simllm.backends.packet_breakdown import packet_step_to_json
    from simllm.backends.step_attribution import HtsimRequestMetricReducer
    from simllm.backends.step_sink import HtsimStepSink
    from simllm.compute.gpu_packet_port import GpuPeerPacketSession
    from simllm.core.execution_io import execution_graph_to_json, execution_result_to_json
    from simllm.core.step import step_record_to_json
    from simllm.core.step_io import step_result_to_json

    writer = Evidence(args.output)
    failure = None
    reporting = []
    def seal_generated():
        for path in sorted(args.output.rglob("*")):
            if not path.is_file() or path == writer.journal:
                continue
            name = path.relative_to(args.output).as_posix()
            row = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size}
            if name in writer.receipts:
                if writer.receipts[name] != row:
                    raise ValueError("producer changed a retained first file: " + name)
                continue
            writer.receipts[name] = row
            with writer.journal.open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(json.dumps({"path": name, **row}, sort_keys=True) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
    try:
        if Path(study_inputs.__file__).resolve() != HERE / "inputs.py":
            raise ValueError("capture loaded a different shared input helper")
        before = source(root)
        writer.write("source-before.json", before)
        if before["commit"] != args.expected_commit:
            raise ValueError("capture source differs from the selected source commit")
        reported = args.arm.endswith("-on")
        library = args.native_library if args.arm == "candidate-native-on" else None
        config, profile, binding, options, transfers = study_inputs.inputs(
            args.topology, args.rate, args.donors, args.payload, args.output / "serving",
            reported=reported, library=library,
        )
        writer.write("inputs.json", study_inputs.input_record(
            config, profile, binding, options, transfers, arm=args.arm,
            topology=args.topology, rate=args.rate, donors=args.donors, payload=args.payload,
        ))
        optional = {"capture_critical_path": True} if reported else {}
        component = GpuPeerPacketSession(
            "component", profile, binding, options=options,
            native_switch_library=library, **optional,
        )
        component.admit("component", "incast", transfers)
        component.advance_until_visible(tuple(row.extent_id for row in transfers))
        writer.write("component-visible.json", component.evidence())
        component.drain()
        writer.write("component-drained.json", component.evidence())
        reducer = HtsimRequestMetricReducer({"peer-star": 0})
        sink = HtsimStepSink(config, request_metric_reducer=reducer)
        cursor = 0
        for index in range(3):
            step = study_inputs.record(index, cursor)
            writer.write(f"step-{index}-input.json", step_record_to_json(step))
            result = sink(step)
            peer = sink.peer_evidence[-1]
            writer.write(f"step-{index}-output.json", {
                "record": step_record_to_json(step), "result": step_result_to_json(result),
                "locality": sink.locality_outcomes[-1], "outcome": sink.outcomes[-1],
                "graph": execution_graph_to_json(peer.graph),
                "execution_result": execution_result_to_json(peer.execution_result),
                "artifacts": peer.artifacts, "sessions": peer.session_observations,
            })
            if reported:
                writer.write(f"step-{index}-breakdown.json",
                             packet_step_to_json(result, sink.packet_breakdowns[-1]))
            seal_generated()
            cursor = result.completed_at_ps
        totals, = reducer.totals()
        writer.write("metrics.json", {"request_totals": totals, "job_completion_ps": cursor})
        if reported:
            writer.write("request-breakdowns.json", reducer.critical_path_breakdowns())
        writer.write("serving-drained.json", sink.close_peer_packets())
        after = source(root)
        writer.write("source-after.json", after)
        if after != before:
            raise ValueError("capture source changed during execution")
        origins = {
            name: str(Path(module.__file__).resolve())
            for name, module in sys.modules.items()
            if name.startswith(("simllm", "examples")) and getattr(module, "__file__", None)
        }
        if any(not Path(path).is_relative_to(root) for path in origins.values()):
            raise ValueError("loaded model source escapes the selected source domain")
        if any(before["files"].get(Path(path).relative_to(root).as_posix())
               != hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in origins.values()):
            raise ValueError("loaded source is missing from the frozen source inventory")
        writer.write("loaded-origins.json", {
            "python": sys.version, "executable": sys.executable, "modules": origins,
            "sha256": {name: hashlib.sha256(Path(path).read_bytes()).hexdigest()
                       for name, path in origins.items()},
        })
    except Exception as error:
        failure = {"type": type(error).__name__, "message": str(error),
                   "traceback": traceback.format_exc()}
    try:
        seal_generated()
    except Exception as error:
        if failure is None:
            failure = {"type": type(error).__name__, "message": str(error),
                       "traceback": traceback.format_exc()}
        else:
            reporting.append({"type": type(error).__name__, "message": str(error)})
    summary = {"verdict": "CAPTURED" if failure is None else "VOID", "failure": failure,
               "first_receipts": writer.receipts, "receipt_order": list(writer.receipts),
               "reporting_failures": reporting}
    try:
        writer.write("capture.json", summary, record=False)
    except Exception as error:
        reporting.append({"type": type(error).__name__, "message": str(error)})
        summary["verdict"] = "VOID"
        if summary["failure"] is None:
            summary["failure"] = {"type": type(error).__name__, "message": str(error),
                                  "traceback": traceback.format_exc()}
    print(json.dumps(primitive(summary), sort_keys=True), flush=True)
    return 0 if summary["verdict"] == "CAPTURED" else 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", choices=("baseline-off", "candidate-off", "candidate-on", "candidate-native-on"), required=True)
    parser.add_argument("--topology", choices=("direct", "switched"), required=True)
    parser.add_argument("--rate", type=int, required=True)
    parser.add_argument("--donors", type=int, required=True)
    parser.add_argument("--payload", type=int, required=True)
    parser.add_argument("--native-library")
    raise SystemExit(run(parser.parse_args()))


if __name__ == "__main__":
    main()
