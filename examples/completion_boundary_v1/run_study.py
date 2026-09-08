"""Run only the frozen completion-boundary populations, retaining raw evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import struct
import subprocess
import sys
import traceback
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from enum import Enum
from fractions import Fraction
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
FREEZE = "4d75849189df4a4b9113ee36101b246559cbf47b"
COMPARISON_CHECKS_FREEZE = "5b19d24c56d34353ac5a1a539da3bb0e0eb2117f"
BASE_SIM = "b7d64e1ced5a9e52b71ab89285932a17498a1998"
BASE_NATIVE = "cc1c80f434600ec977fdb5916fc5ff74be63231d"
BASE_BINARY_SHA256 = "c864c0a8cae31756124366033829a92df1271954aeb908da77fe7044a8692e0e"
SCHEMA = "simllm-htsim-flow-session-v1"
RATES = (200_000_000_000, 400_000_000_000)
BATCHES = (16, 32)
FRAME_LIMIT = 1 << 20
MAX_EVENTS = 1_000_000
NATIVE_BUDGET_PS = 1_000_000_000
LIVE_BUDGET_PS = 10_000_000_000
WALL_TIMEOUT_S = 60
TOKEN = 9001
DIMS = {"num_layers": 2, "hidden_size": 256, "intermediate_size": 1024,
        "num_heads": 4, "num_kv_heads": 4, "head_size": 64,
        "vocab_size": 1024, "dtype_bytes": 2}


def plain(value: Any) -> Any:
    """Retain dataclass fields and non-string mapping keys without loss."""
    if is_dataclass(value):
        return {field.name: plain(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        if all(isinstance(key, str) for key in value):
            return {key: plain(item) for key, item in value.items()}
        return [{"key": plain(key), "value": plain(item)} for key, item in value.items()]
    if isinstance(value, (tuple, list)):
        return [plain(item) for item in value]
    if isinstance(value, Fraction):
        return {"numerator": value.numerator, "denominator": value.denominator}
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, bytes):
        return {"hex": value.hex()}
    if isinstance(value, Path):
        return str(value)
    return value


def canonical(value: Any) -> bytes:
    return json.dumps(plain(value), sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical(value) + b"\n")


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest() if hasattr(hashlib, "file_digest") else hashlib.sha256(stream.read()).hexdigest()


def frame(value: dict) -> bytes:
    body = canonical(value)
    if not 0 < len(body) <= FRAME_LIMIT:
        raise ValueError("invalid frozen frame length")
    return struct.pack(">I", len(body)) + body


def q_ps(rate: int) -> int:
    return 4160 * 8 * 10**12 // rate


def flow_oracle(payload: int, rate: int) -> int:
    return (payload // 4096 + 1) * q_ps(rate) + 2_000_000


def flow_floor(payload: int, rate: int) -> int:
    return payload * 8 * 10**12 // rate + 2_000_000


def live_oracle(batch: int, rate: int) -> int:
    return 32_000_000 + 8 * flow_oracle(batch * 256, rate)


def live_floor(batch: int, rate: int) -> int:
    return 32_000_000 + 8 * (batch * 256 * 8 * 10**12 // rate + 4_000_000)


def population() -> dict:
    return {
        "chain": [{"payload": b, "length": k, "rate": r}
                  for b in (4096, 8192) for k in (2, 4) for r in RATES],
        "retention": [{"payload": b, "rate": r} for b in (8192, 16384) for r in RATES],
        "old_transcripts": [{"payload": b, "rate": r} for b in (4096, 8192) for r in RATES],
        "live": [{"profile": p, "batch": b, "rate": r}
                 for p in ("rnic-nn", "rnic-cn") for b in BATCHES for r in RATES],
        "off": [{"profile": p, "batch": b, "rate": r}
                for p in ("rnic-nn", "rnic-nn-fluid") for b in BATCHES for r in RATES],
        "rejections": [{"profile": "rnic-cn", "batch": b, "rate": r}
                       for b in BATCHES for r in RATES],
        "declared_inputs": {
            "dims": DIMS, "tp_ranks": [0, 4], "live_nodes": 8, "native_nodes": 2,
            "seed": 1, "policy_context_token": TOKEN, "compute_per_layer_ps": 16_000_000,
            "host_model": "ideal", "step_context_lengths": [1, 2, 3],
            "tokens_per_request_per_step": 1, "first_step_release_ps": 0,
            "later_step_release": "preceding returned StepResult.completed_at_ps",
            "fresh_release": "matched retained trigger completion_time_ps",
            "new_graph_counts": {"operations": 8, "artifacts": 6, "messages": 16,
                                 "network_rounds": 8, "active_calc_joins": 0},
            "max_events_per_await": MAX_EVENTS, "child_wall_timeout_s": WALL_TIMEOUT_S,
            "native_absolute_max_time_ps": NATIVE_BUDGET_PS,
            "live_absolute_max_time": "checked step release + 10000000000 ps",
            "live_step_ceiling_ps": NATIVE_BUDGET_PS,
            "old_advance_through_ps": 10_000_000,
            "profile_overrides": None, "topology_override": None,
            "placement_remapping": None, "calibrated_surcharges": None,
            "registration": None, "aggregate_floor": None, "dependency_cross_check": None,
        },
    }


class Evidence:
    def __init__(self) -> None:
        self.configurations: list[dict] = []
        self.oracles: list[dict] = []
        self.relations: list[dict] = []
        self.guards: list[dict] = []
        self.controls: list[dict] = []
        self.metrics: list[dict] = []
        self.flows: list[dict] = []

    def guard(self, name: str, passed: bool, **detail: Any) -> None:
        self.guards.append({"name": name, "passed": bool(passed), **plain(detail)})

    def exact(self, family: str, name: str, observed: int, expected: int) -> None:
        self.oracles.append({"family": family, "name": name, "observed_ps": observed,
                             "expected_ps": expected, "passed": observed == expected})

    def relation(self, family: str, name: str, passed: bool, **detail: Any) -> None:
        self.relations.append({"family": family, "name": name, "passed": bool(passed), **plain(detail)})

    def control(self, family: str, name: str, passed: bool, **detail: Any) -> None:
        self.controls.append({"family": family, "name": name, "passed": bool(passed), **plain(detail)})
        self.guard(f"compatibility:{name}", passed)

    def capture(self, name: str, operation, *args, configuration=True, **kwargs):
        try:
            result = operation(*args, **kwargs)
            if configuration:
                self.configurations.append({"name": name, "status": "collected"})
            return result
        except Exception as error:  # noqa: BLE001 (retain the original population after a fatal failure)
            if configuration:
                self.configurations.append({"name": name, "status": "failed", "error": repr(error)})
            self.guard(f"execution:{name}", False, error=repr(error))
            return None

    def summary(self) -> dict:
        void = any(not row["passed"] for row in self.guards)
        failed = [row for row in (*self.oracles, *self.relations) if not row["passed"]]
        status = "void" if void else "passed" if not failed else "refuted" if all(
            row.get("family") == "retention" for row in failed) else "failed"
        families = sorted({row["family"] for row in self.relations})
        return {"schema": "simllm-completion-boundary-study-v1", "freeze_commit": FREEZE,
                "post_specified_comparison_checks_commit": COMPARISON_CHECKS_FREEZE,
                "status": status, "population": population(),
                "configurations": self.configurations, "exact_oracle_rows": self.oracles,
                "behavioral_relations": self.relations, "fatal_guards": self.guards,
                "unscored_compatibility_controls": self.controls,
                "published_step_results": self.metrics,
                "native_flow_metrics": self.flows,
                "behavioral_score": None if void else {
                    family: {"instances": sum(row["family"] == family for row in self.relations),
                             "passed": sum(row["family"] == family and row["passed"] for row in self.relations)}
                    for family in families},
                "native_test_executables": "separate parent integration gate; not scored here"}


def git(root: Path, *args: str) -> str:
    result = subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True, text=True)
    return result.stdout.strip()


def source_identity(root: Path) -> dict:
    dirty = git(root, "status", "--porcelain", "--untracked-files=no")
    if dirty:
        raise ValueError(f"source has tracked changes: {root}: {dirty}")
    return {"root": str(root), "commit": git(root, "rev-parse", "HEAD"),
            "tree": git(root, "rev-parse", "HEAD^{tree}"), "tracked_clean": True}


def binary_identity(path: Path) -> dict:
    if not path.is_file() or not os.access(path, os.X_OK):
        raise ValueError(f"configure an executable file: {path}")
    return {"path": str(path), "sha256": digest(path), "bytes": path.stat().st_size}


def save_streams(out: Path, result) -> None:
    for label, value in (("stdout", getattr(result, "stdout", getattr(result, "output", None))),
                         ("stderr", getattr(result, "stderr", None))):
        if value is not None:
            raw = value if isinstance(value, bytes) else value.encode("utf-8")
            out.with_suffix(f".{label}.bin").write_bytes(raw)


def run_command(command: list[str], out: Path, *, timeout: float = WALL_TIMEOUT_S,
                environment: dict | None = None):
    from simllm.backends._child_process import run_owned_process

    write_json(out.with_suffix(".command.json"), {"argv": command, "wall_timeout_s": timeout})
    try:
        result = run_owned_process(command, timeout_s=timeout, environment=environment)
    except Exception as error:
        save_streams(out, error)
        out.with_suffix(".error.txt").write_text(traceback.format_exc())
        raise
    save_streams(out, result)
    out.with_suffix(".stdout.txt").write_text(result.stdout)
    out.with_suffix(".stderr.txt").write_text(result.stderr)
    if result.returncode:
        raise RuntimeError(f"command failed ({result.returncode}); retained at {out}")
    return result


def hardware_identity(tool: Path, out: Path) -> dict:
    result = run_command([str(tool)], out / "hardware-helper")
    lines = result.stdout.splitlines()
    if len(lines) != 2 or len(lines[0]) != 64:
        raise ValueError("hardware helper must emit its SHA and one native JSON line")
    hardware = json.loads(lines[1])
    if hashlib.sha256(lines[1].encode()).hexdigest() != lines[0]:
        raise ValueError("native hardware JSON/hash disagreement")
    return {"effective_hardware_sha256": lines[0], "effective_hardware": hardware,
            "acquisition": "idle native device constructor; no EventList or execution",
            "port": "actual HtsimNetworkPort ABI v1, traffic class 3",
            "checked_node_counts": [2, 8], "checked_link_rates_bps": list(RATES)}


def save_transcript(directory: Path, transcript, stderr: bytes = b"") -> None:
    directory.mkdir(parents=True, exist_ok=True)
    index = []
    for number, (direction, raw) in enumerate(transcript):
        name = f"{number:05d}-{direction}.bin"
        (directory / name).write_bytes(raw)
        index.append({"file": name, "direction": direction, "bytes": len(raw),
                      "sha256": hashlib.sha256(raw).hexdigest()})
    write_json(directory / "index.json", index)
    (directory / "stderr.bin").write_bytes(stderr)


def session_config(hardware: str, *, rate: int, profile: str = "rnic-nn", nodes: int = 2,
                   budget: int = NATIVE_BUDGET_PS):
    from simllm.backends.flow_session import FlowSessionConfig

    return FlowSessionConfig(profile, nodes, rate, hardware, TOKEN, seed=1,
                             wall_timeout_s=WALL_TIMEOUT_S, max_events=MAX_EVENTS,
                             simulation_budget_ps=budget)


def run_native(binary: Path, hardware: str, out: Path, name: str, point: dict,
               *, fresh_release: int | None = None) -> dict:
    from simllm.backends.flow_session import FlowSession

    out.mkdir(parents=True)
    config = session_config(hardware, rate=point["rate"])
    write_json(out / "input.json", {"name": name, "point": point, "config": config,
                                    "fresh_release_ps": fresh_release,
                                    "absolute_max_time_ps": NATIVE_BUDGET_PS})
    session = FlowSession(config, (str(binary), "--flow-session"), session_id=name)
    updates = []
    try:
        with session:
            def inject(label, source, payload, eligible, predecessors=()):
                return session.inject(execution_id=name, operation_id=label, flow_id=label,
                                      source=source, destination=1-source, tag=session.last_accepted_sequence + 1000,
                                      payload_bytes=payload, eligible_at_ps=eligible,
                                      predecessor_sequences=predecessors)

            def wait(sequence):
                while sequence not in session.completed_sequences:
                    updates.append(session.await_completion((sequence,)))
                return next(row for row in session.completion_rows if row["sequence"] == sequence)

            if "length" in point:
                release, predecessors = 0, ()
                for index in range(point["length"]):
                    sequence = inject(f"chain-{index}", index % 2, point["payload"], release, predecessors)
                    row = wait(sequence)
                    release, predecessors = row["completion_time_ps"], (sequence,)
            elif fresh_release is None:
                trigger = inject("trigger", 0, 4096, 0)
                inject("background", 1, 1_048_576, 0)
                trigger_row = wait(trigger)
                successor = inject("successor", 1, point["payload"], trigger_row["completion_time_ps"], (trigger,))
                wait(successor)
            else:
                wait(inject("successor", 1, point["payload"], fresh_release))
            drain = session.close()
        result = {"rows": plain(session.completion_rows), "events": plain(session.events),
                  "drain": plain(drain), "updates": plain(updates)}
        write_json(out / "result.json", result)
        return result
    except Exception:
        (out / "failure.txt").write_text(traceback.format_exc())
        raise
    finally:
        save_transcript(out / "frames", session.transcript, session.stderr)


def authority_guards(evidence: Evidence, name: str, result: dict, count: int, nodes: int) -> None:
    rows, drain = result["rows"], result["drain"]
    expected = {"native_session_constructed": 1, "native_posts": count,
                "legacy_ledger_constructed": 0, "legacy_posts": 0,
                "legacy_mutations": 0, "legacy_aborts": 0}
    evidence.guard(f"authority:{name}", drain["authority_counters"] == expected,
                   observed=drain["authority_counters"], expected=expected)
    evidence.guard(f"completion-population:{name}",
                   len(rows) == count and [row["sequence"] for row in rows] == list(range(1, count+1))
                   and all(row["completion_status"] == "success" for row in rows))
    evidence.guard(f"quiescence:{name}", len(drain["sq_high_watermarks"]) == nodes
                   and drain["quiesced_at_ps"] >= max(row["completion_time_ps"] for row in rows))


def check_chain(evidence: Evidence, name: str, point: dict, result: dict) -> int:
    authority_guards(evidence, name, result, point["length"], 2)
    release = 0
    for row in result["rows"]:
        evidence.guard(f"dependency:{name}:{row['sequence']}", row["start_time_ps"] == release)
        evidence.guard(f"physics:{name}:{row['sequence']}",
                       flow_floor(point["payload"], point["rate"]) <= row["fct_ps"]
                       <= flow_oracle(point["payload"], point["rate"]))
        evidence.exact("native-flow", f"{name}:{row['sequence']}", row["fct_ps"],
                       flow_oracle(point["payload"], point["rate"]))
        release = row["completion_time_ps"]
    evidence.exact("native-chain", name, release,
                   point["length"] * flow_oracle(point["payload"], point["rate"]))
    return release


def check_action_evidence(evidence: Evidence, name: str, native: dict) -> None:
    """Audit immutable GOAL/action/native joins independently of their producer."""
    rows = {row["sequence"]: row for row in native["native_drain"]["completion_rows"]}
    seen = []
    previous_end = native["graph"]["released_at_ps"]
    for index, artifact in enumerate(native["artifacts"]):
        prefix = f"actions:{name}:{index}"
        snapshot = artifact["snapshot"]
        actions = {(a["rank"], a["label"]): a for a in artifact["actions"]}
        operations = {(op["rank"], op["label"]): op for op in snapshot["operations"]}
        evidence.guard(prefix + ":inventory", len(actions) == len(artifact["actions"])
                       and actions.keys() == operations.keys())
        evidence.guard(prefix + ":release", artifact["released_at_ps"] == previous_end)
        evidence.guard(prefix + ":snapshot", hashlib.sha256(
            bytes.fromhex(snapshot["rendered_bytes"]["hex"])).hexdigest() == snapshot["sha256"])
        dependencies = {key: [] for key in actions}
        for edge in snapshot["dependencies"]:
            evidence.guard(prefix + ":dependency-kind", edge["relation"] == "requires")
            dependencies[edge["rank"], edge["operation_label"]].append(
                actions[edge["rank"], edge["predecessor_label"]]["completed_at_ps"])
        for key, action in actions.items():
            expected = max([artifact["released_at_ps"], *dependencies[key]])
            evidence.guard(prefix + f":eligibility:{key}", action["eligible_at_ps"] == expected
                           and action["started_at_ps"] >= expected
                           and action["operation_id"] == operations[key]["operation_id"])
            if action["kind"] == "calc":
                cost_ns = int(operations[key]["text"].split()[1])
                evidence.guard(prefix + f":calc:{key}", action["completed_at_ps"]
                               == action["started_at_ps"] + max(cost_ns, 1) * 1000)
                if key[0] in (0, 4):
                    evidence.guard(prefix + f":no-active-join:{key}", cost_ns > 0)
        for message in snapshot["messages"]:
            send = actions[message["source_rank"], message["send_label"]]
            receive = actions[message["destination_rank"], message["receive_label"]]
            row = rows[send["sequence"]]
            seen.append(send["sequence"])
            identity = (message["operation_id"], message["source_rank"], message["destination_rank"],
                        message["payload_bytes"], message["tag"])
            observed = tuple(row[key] for key in ("operation_id", "source", "destination", "payload_bytes", "tag"))
            evidence.guard(prefix + f":message:{send['sequence']}", identity == observed
                           and receive["sequence"] == send["sequence"]
                           and row["start_time_ps"] == send["started_at_ps"]
                           and send["completed_at_ps"] == receive["completed_at_ps"]
                           == max(row["completion_time_ps"], receive["started_at_ps"]))
        evidence.guard(prefix + ":completion", artifact["completed_at_ps"]
                       == max(action["completed_at_ps"] for action in actions.values()))
        previous_end = artifact["completed_at_ps"]
    evidence.guard(f"message-join:{name}", sorted(seen) == sorted(rows) and len(seen) == len(set(seen)))
    projected = native["execution_result"]["events"]
    for row in rows.values():
        events = [event for event in native["native_events"] if event["sequence"] == row["sequence"]]
        evidence.guard(f"lifecycle:{name}:{row['sequence']}", [event["kind"] for event in events]
                       == ["accepted", "queued", "started", "completed"])
        for event in events:
            phase = "submitted" if event["kind"] == "accepted" else event["kind"]
            matches = [value for value in projected if value["subject_object_id"] == row["flow_id"]
                       and value["phase"] == phase and value["operation_id"] == row["operation_id"]]
            evidence.guard(f"completion-event:{name}:{row['sequence']}:{phase}", len(matches) == 1
                           and matches[0]["timestamp_ps"] == event["timestamp_ps"])


def check_retention(evidence: Evidence, name: str, point: dict, retained: dict, fresh: dict) -> None:
    ceiling = 2 * (1 + 256 + point["payload"] // 4096) * q_ps(point["rate"]) + 8_000_000
    for label, result, count in (("retained", retained, 3), ("fresh", fresh, 1)):
        authority_guards(evidence, name + label, result, count, 2)
        evidence.guard(f"retention-ceiling:{name}:{label}", result["drain"]["quiesced_at_ps"] <= ceiling,
                       ceiling_ps=ceiling, observed_ps=result["drain"]["quiesced_at_ps"])
        for row in result["rows"]:
            evidence.guard(f"retention-floor:{name}:{label}:{row['sequence']}",
                           row["fct_ps"] >= flow_floor(row["payload_bytes"], point["rate"]))
    trigger, background, successor = retained["rows"]
    diagnostic = fresh["rows"][0]
    evidence.guard(f"retention-release:{name}", successor["start_time_ps"] == trigger["completion_time_ps"]
                   == diagnostic["start_time_ps"])
    evidence.guard(f"background-serialization:{name}", background["completion_time_ps"] >= 256 * q_ps(point["rate"]))
    evidence.relation("retention", name, successor["fct_ps"] > diagnostic["fct_ps"]
                      and retained["drain"]["sq_high_watermarks"][1] == 2
                      and fresh["drain"]["sq_high_watermarks"][1] == 1,
                      retained_fct_ps=successor["fct_ps"], fresh_fct_ps=diagnostic["fct_ps"],
                      retained_high_water=retained["drain"]["sq_high_watermarks"][1],
                      fresh_high_water=fresh["drain"]["sq_high_watermarks"][1])


def legacy_requests(hardware: str, point: dict) -> list[dict]:
    common = {"schema": SCHEMA}
    return [
        {**common, "verb": "open", "session_id": "completion-boundary-old-verbs",
         "profile": "rnic-nn", "node_count": 2, "link_rate_bps": point["rate"], "seed": 1,
         "topology_identity": "rnic-nn:nodes=2", "wqe_authority": "simllm-native-rnic-session",
         "effective_hardware_sha256": hardware},
        {**common, "verb": "inject", "sequence": 1, "execution_id": "old-execution",
         "operation_id": "old-operation", "flow_id": "old-flow", "source": 0, "destination": 1,
         "tag": 1001, "payload_bytes": point["payload"], "eligible_at_ps": 0, "policy_context_token": TOKEN},
        {**common, "verb": "advance", "through_sequence": 1, "through_ps": 10_000_000},
        {**common, "verb": "drain", "through_sequence": 1},
        {**common, "verb": "close", "through_sequence": 1},
    ]


def run_legacy(binary: Path, requests: list[dict], out: Path) -> bytes:
    from simllm.backends._child_process import OwnedBinaryProcess, OwnedBinaryReadError

    write_json(out / "input.json", {"requests": requests, "max_time_ps": NATIVE_BUDGET_PS,
                                    "wall_timeout_s": WALL_TIMEOUT_S})
    transcript, responses = [], []
    process = OwnedBinaryProcess((str(binary), "--flow-session"), timeout_s=WALL_TIMEOUT_S)
    try:
        for request in requests:
            raw = frame(request)
            transcript.append(("request", raw))
            process.write(raw)
            header = b""
            try:
                header = process.read_exact(4)
                size = struct.unpack(">I", header)[0]
                if not 0 < size <= FRAME_LIMIT:
                    transcript.append(("response", header))
                    raise ValueError("old response exceeds the frame bound")
                body = process.read_exact(size)
            except OwnedBinaryReadError as error:
                transcript.append(("response", header + error.partial))
                raise
            transcript.append(("response", header + body))
            response = json.loads(body)
            if (canonical(response) != body or response.get("schema") != SCHEMA
                    or response.get("status") != "ok" or response.get("verb") != request["verb"]):
                raise ValueError("old response is malformed or unsuccessful")
            responses.append(header + body)
        if process.finish() != 0:
            raise ValueError("old native child exited unsuccessfully")
        return b"".join(responses)
    finally:
        process.abort()
        save_transcript(out / "frames", transcript, process.stderr)


def artifacts(directory: Path) -> list[dict]:
    return [{"path": str(path.relative_to(directory)), "bytes": path.stat().st_size, "sha256": digest(path)}
            for path in sorted(directory.rglob("*")) if path.is_file()]


def scenario_worker(spec: dict) -> dict:
    """Run in the selected source tree; the baseline never imports new APIs."""
    root = Path(spec["code_root"]).resolve()
    sys.path.insert(0, str(root))
    import simllm
    import simllm.backends.htsim_rnic as native_module
    import simllm.backends.step_sink as sink_module
    from simllm.backends.step_attribution import HtsimRequestMetricReducer
    from simllm.compute import ComputeProvider, DurationEstimate, HostInitiationModel, ModelDims
    from simllm.core import RequestPhase, ScheduledRequest, StepRecord
    from simllm.core.execution_io import execution_graph_to_json, execution_result_to_json
    from simllm.core.step import step_record_to_json
    from simllm.core.step_io import step_result_to_json
    from simllm.traffic import project_execution_graph_goal

    if not Path(simllm.__file__).resolve().is_relative_to(root):
        raise ValueError("scenario imported the wrong source tree")
    if not Path(spec["identity_manifest"]).is_file():
        raise ValueError("scenario has no pre-execution identity manifest")
    out = Path(spec["out"])
    out.mkdir(parents=True, exist_ok=True)
    work = out / "artifacts"
    markers = out / "child-markers"
    markers.mkdir()
    os.environ.update(SIMLLM_HTSIM_RNIC=spec["binary"], SIMLLM_TXT2BIN=spec["txt2bin"],
                      SIMLLM_CHILD_LIFETIME_MARKER_DIR=str(markers),
                      SIMLLM_CHILD_LIFETIME_RUN_NONCE=spec["name"])
    original_run = native_module.run_owned_process
    commands, sessions = [], []

    def observed_run(command, **kwargs):
        index = len(commands)
        commands.append(list(command))
        log = out / f"native-{index:03d}"
        write_json(log.with_suffix(".command.json"), {"argv": command, "wall_timeout_s": WALL_TIMEOUT_S})
        kwargs["timeout_s"] = WALL_TIMEOUT_S
        try:
            result = original_run(command, **kwargs)
        except Exception as error:
            save_streams(log, error)
            raise
        save_streams(log, result)
        log.with_suffix(".stdout.txt").write_text(result.stdout)
        log.with_suffix(".stderr.txt").write_text(result.stderr)
        return result

    native_module.run_owned_process = observed_run

    class FixedProvider(ComputeProvider):
        def estimate(self, kernel, gpu):
            return DurationEstimate(duration_ps=32_000_000, bound="declared")

    values = {"profile": spec["profile"], "tp_ranks": (0, 4), "dims": ModelDims(**DIMS), "workdir": work,
              "provider": FixedProvider(), "host_model": HostInitiationModel.ideal(), "num_goal_ranks": 8,
              "linkspeed_bps": spec["rate"]}
    if spec["session"]:
        from simllm.backends.flow_session import FlowSession

        def observed_session(config, command, **kwargs):
            session = FlowSession(config, command, **kwargs)
            directory = out / f"session-{len(sessions):03d}"
            write_json(directory / "input.json", {"config": config, "command": command, **kwargs,
                                                  "absolute_max_time_ps": session.max_time_ps})
            sessions.append((session, directory))
            return session

        sink_module.FlowSession = observed_session
        values["flow_session"] = session_config(spec["hardware"], rate=spec["rate"],
                                                profile=spec["profile"], nodes=8, budget=LIVE_BUDGET_PS)
    config = sink_module.HtsimStepSinkConfig(**values)
    kwargs = {"request_metric_reducer": HtsimRequestMetricReducer(
        {f"r{i}": 0 for i in range(spec["batch"])})} if spec["session"] else {}
    sink = sink_module.HtsimStepSink(config, **kwargs)
    result = {"status": "ok", "steps": [], "native_commands": commands}
    release = 0
    try:
        for index in range(3):
            record = StepRecord(index, release, [ScheduledRequest(
                f"r{i}", RequestPhase.PREFILL if index == 0 else RequestPhase.DECODE,
                num_new_tokens=1, context_length=index+1) for i in range(spec["batch"])])
            write_json(out / f"step-{index}.input.json", step_record_to_json(record))
            graph, timing = sink._serial_lowerer().lower_with_timing(record)
            projection = project_execution_graph_goal(graph, num_goal_ranks=8, base_tag=config.base_tag)
            write_json(out / f"step-{index}.graph.json", execution_graph_to_json(graph))
            inputs = []
            for number, artifact in enumerate(projection.artifacts):
                trace = artifact.trace
                raw = trace.render().encode()
                (out / f"step-{index}.artifact-{number}.input.goal").write_bytes(raw)
                inputs.append({"operation_ids": artifact.operation_ids, "operations": trace.operations,
                               "messages": trace.messages, "dependencies": trace.dependencies,
                               "goal_sha256": hashlib.sha256(raw).hexdigest()})
            write_json(out / f"step-{index}.projection.json", {"artifacts": inputs, "timing": timing,
                       "boundaries": projection.boundaries, "serialized_edges": projection.serialized_edges})
            step = sink(record)
            if step is None:
                raise ValueError("frozen collective step returned no result")
            serialized = step_result_to_json(step)
            write_json(work / f"step-{index:06d}.step-result.json", serialized)
            entry = {"result": serialized, "released_at_ps": release,
                     "graph_operations": len(graph.operations), "artifact_count": len(inputs),
                     "message_count": sum(len(artifact.trace.messages) for artifact in projection.artifacts),
                     "outcome": plain(sink.outcomes[-1]), "locality": plain(sink.locality_outcomes[-1])}
            if spec["session"]:
                evidence = sink.session_evidence[-1]
                evidence.validate_result(step)
                write_json(out / f"step-{index}.execution-result.json", execution_result_to_json(evidence.execution_result))
                write_json(out / f"step-{index}.session-evidence.json", evidence)
                entry["session_evidence"] = plain(evidence)
            result["steps"].append(entry)
            release = step.completed_at_ps
    except Exception as error:  # noqa: BLE001 (retain failed raw evidence before returning to the population runner)
        result.update(status="rejected" if spec["profile"] == "rnic-cn" and not spec["session"] else "failed",
                      exception_type=type(error).__name__, error=str(error))
        (out / "failure.txt").write_text(traceback.format_exc())
    finally:
        for session, directory in sessions:
            save_transcript(directory / "frames", session.transcript, session.stderr)
        result["session_count"] = len(sessions)
        result["child_markers"] = artifacts(markers)
        result["artifact_inventory"] = artifacts(work)
        write_json(out / "result.json", result)
    return result


def launch_scenario(args, identity: dict, point: dict, name: str, *, session: bool, baseline: bool) -> dict:
    out = args.out / name
    out.mkdir()
    spec = {**point, "name": name, "session": session, "out": str(out),
            "hardware": identity["hardware"]["effective_hardware_sha256"],
            "identity_manifest": str(args.out / "identity.json"),
            "code_root": str(args.baseline_code if baseline else args.candidate_code),
            "binary": str(args.baseline_binary if baseline else args.candidate_binary),
            "txt2bin": str(args.baseline_txt2bin if baseline else args.candidate_txt2bin)}
    write_json(out / "scenario-input.json", spec)
    # The scenario owns three sequential 60-second sessions. The outer deadline
    # also accommodates pure graph rendering and the stateless artifact controls.
    run_command([sys.executable, str(HERE / "run_study.py"), "--scenario-input", str(out / "scenario-input.json")],
                out / "worker", timeout=900)
    return json.loads((out / "result.json").read_text())


def check_live(evidence: Evidence, name: str, point: dict, result: dict) -> list[int]:
    evidence.guard(f"live-status:{name}", result["status"] == "ok", status=result["status"])
    evidence.guard(f"live-population:{name}", len(result["steps"]) == 3 and result["session_count"] == 3)
    latencies, release = [], 0
    for index, step in enumerate(result["steps"]):
        row = step["result"]
        latency = row["step_latency_ps"]
        native = step["session_evidence"]
        execution = native["execution_result"]
        check_action_evidence(evidence, f"{name}:{index}", native)
        evidence.metrics.append({"configuration": name, "step_result": row})
        evidence.guard(f"graph-population:{name}:{index}", step["graph_operations"] == 8
                       and step["artifact_count"] == 6 and step["message_count"] == 16)
        evidence.guard(f"live-conservation:{name}:{index}", row["completed_at_ps"] == release + latency
                       == execution["completed_at_ps"] and step["released_at_ps"] == release)
        drain = native["native_drain"]
        authority_guards(evidence, f"{name}:{index}", {"rows": drain["completion_rows"], "drain": drain}, 16, 8)
        for flow in drain["completion_rows"]:
            propagation = 4_000_000 if point["profile"] == "rnic-cn" else 2_000_000
            floor = flow["payload_bytes"] * 8 * 10**12 // point["rate"] + propagation
            evidence.guard(f"live-flow-floor:{name}:{index}:{flow['sequence']}", flow["fct_ps"] >= floor,
                           observed_ps=flow["fct_ps"], floor_ps=floor)
            evidence.flows.append({**point, "step": index, **flow})
        evidence.guard(f"live-budget:{name}:{index}", latency <= NATIVE_BUDGET_PS
                       and row["completed_at_ps"] <= execution["quiesced_at_ps"]
                       <= release + LIVE_BUDGET_PS)
        metrics = row["request_metrics"]
        evidence.guard(f"request-population:{name}:{index}", len(metrics) == point["batch"]
                       and {metric["request_id"] for metric in metrics} == {f"r{i}" for i in range(point["batch"])})
        for metric in metrics:
            tpot = None if index == 0 else Fraction(sum(latencies[1:]) + latency, index)
            evidence.guard(f"request-conservation:{name}:{index}:{metric['request_id']}",
                           metric["completed_at_ps"] == row["completed_at_ps"] and metric["latency_ps"] == latency
                           and metric["ttft_ps"] == (latencies[0] if latencies else latency)
                           and metric["tpot_ps"] == plain(tpot))
        if point["profile"] == "rnic-nn":
            evidence.exact("live-nn-step", f"{name}:{index}", latency, live_oracle(point["batch"], point["rate"]))
            floor = 32_000_000 + 8 * flow_floor(point["batch"] * 256, point["rate"])
        else:
            floor = live_floor(point["batch"], point["rate"])
        evidence.guard(f"live-physics:{name}:{index}", latency >= floor, observed_ps=latency, floor_ps=floor)
        latencies.append(latency)
        release = row["completed_at_ps"]
    return latencies


def check_relations(evidence: Evidence, chains: dict, live: dict) -> None:
    for payload in (4096, 8192):
        for length in (2, 4):
            keys = [(payload, length, rate) for rate in RATES]
            if all(key in chains for key in keys):
                slow, fast = (chains[key] - length * 2_000_000 for key in keys)
                evidence.relation("chain-bandwidth", f"B{payload}-K{length}", slow == 2*fast,
                                  slow_serialization_ps=slow, fast_serialization_ps=fast)
    for length in (2, 4):
        for rate in RATES:
            keys = [(payload, length, rate) for payload in (4096, 8192)]
            if all(key in chains for key in keys):
                delta = chains[keys[1]] - chains[keys[0]]
                evidence.relation("chain-payload", f"K{length}-R{rate}", delta == length*q_ps(rate),
                                  observed_delta_ps=delta, expected_delta_ps=length*q_ps(rate))
    for index in range(3):
        for batch in BATCHES:
            keys = [("rnic-nn", batch, rate) for rate in RATES]
            if all(len(live.get(key, [])) == 3 for key in keys):
                slow, fast = (live[key][index] - 48_000_000 for key in keys)
                evidence.relation("live-bandwidth", f"B{batch}-step{index}", slow == 2*fast,
                                  slow_serialization_ps=slow, fast_serialization_ps=fast)
        for rate in RATES:
            keys = [("rnic-nn", batch, rate) for batch in BATCHES]
            if all(len(live.get(key, [])) == 3 for key in keys):
                delta = live[keys[1]][index] - live[keys[0]][index]
                evidence.relation("live-payload", f"R{rate}-step{index}", delta == 8*q_ps(rate),
                                  observed_delta_ps=delta, expected_delta_ps=8*q_ps(rate))
            for batch in BATCHES:
                keys = [(profile, batch, rate) for profile in ("rnic-nn", "rnic-cn")]
                if all(len(live.get(key, [])) == 3 for key in keys):
                    delta = live[keys[1]][index] - live[keys[0]][index]
                    floor = live_floor(batch, rate) - live_oracle(batch, rate)
                    evidence.relation("live-physical-excess", f"B{batch}-R{rate}-step{index}", delta >= floor,
                                      observed_delta_ps=delta, minimum_delta_ps=floor)
                    evidence.guard(f"physical-excess:B{batch}-R{rate}-step{index}", delta >= floor)


def compare_off(evidence: Evidence, name: str, before: dict, after: dict, *, rejection: bool) -> None:
    if rejection:
        passed = (before["status"] == after["status"] == "rejected"
                  and before["exception_type"] == after["exception_type"] == "RuntimeError"
                  and before["error"] == after["error"]
                  and "starts a fresh backend process" in before["error"]
                  and not before["native_commands"] and not after["native_commands"]
                  and not before["child_markers"] and not after["child_markers"]
                  and before["session_count"] == after["session_count"] == 0)
        evidence.control("absent-physical-refusal", name, passed, before=before.get("error"), after=after.get("error"))
    else:
        passed = (before["status"] == after["status"] == "ok"
                  and len(before["steps"]) == len(after["steps"]) == 3
                  and before["artifact_inventory"] == after["artifact_inventory"]
                  and [row["result"] for row in before["steps"]] == [row["result"] for row in after["steps"]])
        evidence.control("absent-ideal-bytes", name, passed,
                         before_inventory=before["artifact_inventory"], after_inventory=after["artifact_inventory"])


def normalize_flow_metrics(evidence: Evidence) -> None:
    """Join declared messages across profiles without equating local cursors."""
    fields = ("batch", "rate", "step", "operation_id", "flow_id", "source",
              "destination", "tag", "payload_bytes")
    inventory = {}
    valid = True
    for profile in ("rnic-nn", "rnic-cn"):
        rows = [row for row in evidence.flows if row["profile"] == profile]
        indexed = {tuple(row.get(key) for key in fields): row for row in rows}
        complete = (len(rows) == len(indexed) == 192
                    and all(all(key in row for key in fields) for row in rows))
        evidence.guard(f"normalized-flow-inventory:{profile}", complete,
                       rows=len(rows), unique_messages=len(indexed), expected=192)
        valid &= complete
        inventory[profile] = indexed
    same_keys = inventory["rnic-nn"].keys() == inventory["rnic-cn"].keys()
    evidence.guard("normalized-flow-key-sets", same_keys)
    if not valid or not same_keys:
        return
    for key, row in inventory["rnic-cn"].items():
        baseline = inventory["rnic-nn"][key]
        row["matched_nn_sequence"] = baseline["sequence"]
        row["normalized_fct_to_matched_nn"] = plain(Fraction(row["fct_ps"], baseline["fct_ps"]))


def execute(args, identity: dict) -> dict:
    evidence, chains, live = Evidence(), {}, {}
    hardware = identity["hardware"]["effective_hardware_sha256"]
    points = population()
    for point in points["chain"]:
        name = f"chain-B{point['payload']}-K{point['length']}-R{point['rate']}"
        result = evidence.capture(name, run_native, args.candidate_binary, hardware, args.out / name, name, point)
        if result is not None:
            value = evidence.capture(name + "-checks", check_chain, evidence, name, point, result, configuration=False)
            if value is not None:
                chains[point["payload"], point["length"], point["rate"]] = value
    for point in points["retention"]:
        name = f"retention-B{point['payload']}-R{point['rate']}"
        retained = evidence.capture(name, run_native, args.candidate_binary, hardware, args.out / name, name, point)
        if retained is None:
            evidence.configurations.append({"name": name + "-fresh", "status": "unavailable"})
            evidence.guard(f"fresh-unavailable:{name}", False, reason="retained trigger has no authoritative completion")
            continue
        release = retained["rows"][0]["completion_time_ps"]
        fresh = evidence.capture(name + "-fresh", run_native, args.candidate_binary, hardware,
                                 args.out / (name + "-fresh"), name + "-fresh", point, fresh_release=release)
        if fresh is not None:
            evidence.capture(name + "-checks", check_retention, evidence, name, point, retained, fresh, configuration=False)
    for point in points["old_transcripts"]:
        name = f"old-B{point['payload']}-R{point['rate']}"
        requests = legacy_requests(hardware, point)
        before = evidence.capture(name + "-base", run_legacy, args.baseline_binary, requests, args.out / (name + "-base"))
        after = evidence.capture(name + "-candidate", run_legacy, args.candidate_binary, requests, args.out / (name + "-candidate"))
        evidence.control("old-verb-bytes", name, before is not None and before == after,
                         before_sha256=hashlib.sha256(before).hexdigest() if before else None,
                         after_sha256=hashlib.sha256(after).hexdigest() if after else None)
    for point in points["live"]:
        name = f"live-{point['profile']}-B{point['batch']}-R{point['rate']}"
        result = evidence.capture(name, launch_scenario, args, identity, point, name, session=True, baseline=False)
        if result is not None:
            value = evidence.capture(name + "-checks", check_live, evidence, name, point, result, configuration=False)
            if value is not None:
                live[point["profile"], point["batch"], point["rate"]] = value
    for family, rejection in (("off", False), ("rejections", True)):
        for point in points[family]:
            name = f"{family}-{point['profile']}-B{point['batch']}-R{point['rate']}"
            before = evidence.capture(name + "-base", launch_scenario, args, identity, point, name + "-base", session=False, baseline=True)
            after = evidence.capture(name + "-candidate", launch_scenario, args, identity, point, name + "-candidate", session=False, baseline=False)
            if before is not None and after is not None:
                evidence.capture(name + "-checks", compare_off, evidence, name, before, after,
                                 rejection=rejection, configuration=False)
    check_relations(evidence, chains, live)
    normalize_flow_metrics(evidence)
    expected = {"chain-bandwidth": 4, "chain-payload": 4, "retention": 4,
                "live-bandwidth": 6, "live-payload": 6, "live-physical-excess": 12}
    observed = {family: sum(row["family"] == family for row in evidence.relations) for family in expected}
    evidence.guard("relation-population", observed == expected, observed=observed, expected=expected)
    evidence.guard("oracle-population", {family: sum(row["family"] == family for row in evidence.oracles)
                   for family in ("native-flow", "native-chain", "live-nn-step")}
                   == {"native-flow": 24, "native-chain": 8, "live-nn-step": 12})
    evidence.guard("compatibility-population", len(evidence.controls) == 16)
    report = evidence.summary()
    write_json(args.out / "summary.json", report)
    return report


def prepare_identity(args) -> dict:
    for name in ("candidate_code", "baseline_code", "candidate_backend"):
        setattr(args, name, getattr(args, name).resolve())
    sys.path.insert(0, str(args.candidate_code))
    sources = {name: source_identity(getattr(args, name))
               for name in ("candidate_code", "baseline_code", "candidate_backend")}
    if git(args.baseline_code, "diff", BASE_SIM, "HEAD", "--", "simllm"):
        raise ValueError("baseline Python implementation differs from the frozen source")
    git(args.candidate_code, "merge-base", "--is-ancestor", FREEZE, "HEAD")
    freeze_path = args.candidate_code / "examples/completion_boundary_v1/expectations.md"
    frozen = subprocess.run(["git", "-C", str(args.candidate_code), "show",
                             f"{FREEZE}:examples/completion_boundary_v1/expectations.md"],
                            check=True, capture_output=True).stdout
    if freeze_path.read_bytes() != frozen:
        raise ValueError("expectations changed after the final freeze")
    git(args.candidate_code, "merge-base", "--is-ancestor", COMPARISON_CHECKS_FREEZE, "HEAD")
    comparison = args.candidate_code / "examples/completion_boundary_v1/post_run_checks.md"
    frozen_comparison = subprocess.run(["git", "-C", str(args.candidate_code), "show",
        f"{COMPARISON_CHECKS_FREEZE}:examples/completion_boundary_v1/post_run_checks.md"],
        check=True, capture_output=True).stdout
    if comparison.read_bytes() != frozen_comparison:
        raise ValueError("post-specified comparison checks changed after their freeze")
    binaries = {}
    for name in ("candidate_binary", "baseline_binary", "candidate_txt2bin", "baseline_txt2bin", "hardware_helper"):
        path = getattr(args, name).resolve()
        setattr(args, name, path)
        binaries[name] = binary_identity(path)
    if binaries["baseline_binary"]["sha256"] != BASE_BINARY_SHA256:
        raise ValueError("baseline binary differs from the independently retained executable")
    cache = args.candidate_binary.parent.parent / "CMakeCache.txt"
    cache_text = cache.read_text()
    if ("HTSIM_ENABLE_SIMLLM_RNIC:BOOL=ON" not in cache_text
            or f"SIMLLM_REPOSITORY_ROOT:PATH={args.candidate_code}" not in cache_text):
        raise ValueError("candidate build must enable composition and bind the candidate SimLLM tree")
    return {"schema": "simllm-completion-boundary-identities-v1", "freeze_commit": FREEZE,
            "post_specified_comparison_checks_commit": COMPARISON_CHECKS_FREEZE,
            "comparison_checks_sha256": hashlib.sha256(frozen_comparison).hexdigest(),
            "freeze_sha256": hashlib.sha256(frozen).hexdigest(), "sources": sources,
            "frozen_source_bases": {"simllm": BASE_SIM, "htsim": BASE_NATIVE},
            "runner": {"path": str(Path(__file__).resolve()), "sha256": digest(Path(__file__))},
            "hardware_helper_source_sha256": digest(HERE / "hardware_identity.cpp"),
            "executables": binaries, "candidate_cmake_cache_sha256": digest(cache),
            "python": {"executable": sys.executable, "version": sys.version},
            "platform": platform.platform(), "inputs": population(), "hardware": None}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario-input", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--out", type=Path)
    for name in ("candidate-code", "baseline-code", "candidate-backend", "candidate-binary",
                 "baseline-binary", "candidate-txt2bin", "baseline-txt2bin", "hardware-helper"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--plan-only", action="store_true", help="record identities and inputs without executing any native code")
    args = parser.parse_args(argv)
    if args.scenario_input is not None:
        scenario_worker(json.loads(args.scenario_input.read_text()))
        return 0
    if any(getattr(args, name) is None for name in ("out", "candidate_code", "baseline_code", "candidate_backend",
           "candidate_binary", "baseline_binary", "candidate_txt2bin", "baseline_txt2bin", "hardware_helper")):
        parser.error("all source, executable and output paths are required")
    args.out = args.out.resolve()
    for root in (args.candidate_code.resolve(), args.baseline_code.resolve(), args.candidate_backend.resolve()):
        if args.out == root or args.out.is_relative_to(root):
            parser.error("raw evidence must be outside the source trees")
    if args.out.exists():
        parser.error("use a new output directory to preserve prior evidence")
    identity = prepare_identity(args)
    args.out.mkdir(parents=True)
    write_json(args.out / "identity.json", identity)
    if args.plan_only:
        print("Frozen identities and populations recorded; no native code executed.")
        return 0
    try:
        identity["hardware"] = hardware_identity(args.hardware_helper, args.out)
        write_json(args.out / "identity.json", identity)
        report = execute(args, identity)
    except Exception:
        (args.out / "fatal-runner-error.txt").write_text(traceback.format_exc())
        write_json(args.out / "summary.json", {"status": "void", "behavioral_score": None,
                   "freeze_commit": FREEZE, "error": "runner failed; retain original population and chronology"})
        raise
    print(f"Study status: {report['status']}; evidence: {args.out}")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
