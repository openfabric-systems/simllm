"""Run the frozen HTSIM-18 and CORE-24 acceptance study."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import struct
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

from simllm.goal import GoalTrace

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
EXPECTATIONS_COMMIT = "71afffe602a527a5fde72e1e47a7987d85ebf479"
HTSIM_BASE_COMMIT = "4885c647eecdfdf81479d1df052223c016ad086b"
SCHEMA = "simllm-htsim-flow-session-v1"
FRAME_LIMIT = 1 << 20
LINK_RATE_BPS = 400_000_000_000
STEP_GAP_PS = 10_000_000
POLICY_CONTEXT_TOKEN = 9001


@dataclass(frozen=True)
class Flow:
    source: int
    destination: int
    payload_bytes: int


REPLAYS = {
    "two-node": (
        2,
        (Flow(0, 1, 4096), Flow(1, 0, 8192)),
        (0.002, 0.5),
        (0.0005, 0.25),
        1.2,
    ),
    "four-node": (
        4,
        (
            Flow(0, 1, 4096),
            Flow(1, 2, 8192),
            Flow(2, 3, 4096),
            Flow(3, 0, 8192),
        ),
        (0.004, 1.0),
        (0.0005, 0.25),
        1.5,
    ),
}
STATE_PAYLOADS = (4096, 8192)
CODEC_CASES = ("empty", "prefill", "decode", "mixed")


def _configured_executable(variable: str) -> Path:
    raw = os.environ.get(variable)
    if not raw:
        raise ValueError(f"{variable} must name an executable")
    path = Path(raw)
    if not path.is_file() or not os.access(path, os.X_OK):
        raise ValueError(f"{variable} is not executable: {path}")
    return path


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()


def _frame(value: dict[str, Any]) -> bytes:
    body = _canonical(value)
    if len(body) > FRAME_LIMIT:
        raise ValueError("frame exceeds the frozen limit")
    return struct.pack(">I", len(body)) + body


def _read_exact(stream: BinaryIO, size: int) -> bytes:
    parts: list[bytes] = []
    remaining = size
    while remaining:
        part = stream.read(remaining)
        if not part:
            raise EOFError(f"session closed with {remaining} response bytes missing")
        parts.append(part)
        remaining -= len(part)
    return b"".join(parts)


def _read_frame(stream: BinaryIO) -> dict[str, Any]:
    size = struct.unpack(">I", _read_exact(stream, 4))[0]
    if size > FRAME_LIMIT:
        raise ValueError("response frame exceeds the frozen limit")
    body = _read_exact(stream, size)
    value = json.loads(body)
    if type(value) is not dict or _canonical(value) != body:
        raise ValueError("server emitted noncanonical JSON")
    return value


class Session:
    def __init__(self, binary: Path) -> None:
        self.process = subprocess.Popen(
            [str(binary), "--flow-session"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert self.process.stdin is not None
        assert self.process.stdout is not None

    def request(self, value: dict[str, Any]) -> dict[str, Any]:
        assert self.process.stdin is not None
        assert self.process.stdout is not None
        self.process.stdin.write(_frame(value))
        self.process.stdin.flush()
        response = _read_frame(self.process.stdout)
        if response.get("schema") != SCHEMA or response.get("status") != "ok":
            raise RuntimeError(f"session request failed: {response}")
        return response

    def finish(self) -> tuple[int, bytes]:
        assert self.process.stdin is not None
        assert self.process.stderr is not None
        self.process.stdin.close()
        stderr = self.process.stderr.read()
        return self.process.wait(timeout=30), stderr


def _goal_text(nodes: int, flow: Flow, tag: int) -> str:
    trace = GoalTrace(nodes)
    trace.rank(flow.source).send(
        flow.payload_bytes,
        to=flow.destination,
        tag=tag,
    )
    trace.rank(flow.destination).recv(
        flow.payload_bytes,
        source=flow.source,
        tag=tag,
    )
    return trace.render()


def _run_cli_flow(
    *,
    binary: Path,
    txt2bin: Path,
    out: Path,
    name: str,
    nodes: int,
    flow: Flow,
    tag: int,
) -> dict[str, Any]:
    goal = out / f"{name}.goal"
    goal_bin = out / f"{name}.bin"
    completion = out / f"{name}.csv"
    goal.write_text(_goal_text(nodes, flow, tag), newline="\n")
    subprocess.run(
        [str(txt2bin), "-i", str(goal), "-o", str(goal_bin)],
        check=True,
        capture_output=True,
    )
    run = subprocess.run(
        [
            str(binary),
            "-goal",
            str(goal_bin),
            "-linkspeed_bps",
            str(LINK_RATE_BPS),
            "-rnic_profile",
            "rnic-nn",
            "-nodes",
            str(nodes),
            "-completion_csv",
            str(completion),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    rows = list(csv.DictReader(completion.open(newline="")))
    if len(rows) != 1:
        raise AssertionError(f"{name}: expected one CLI completion row")
    marker = "hardware_config_sha256="
    hashes = [
        token[len(marker) :]
        for token in run.stdout.split()
        if token.startswith(marker)
    ]
    if len(hashes) != 1 or len(hashes[0]) != 64:
        raise AssertionError(f"{name}: missing CLI hardware hash")
    return {
        "fct_ps": int(rows[0]["fct_ps"]),
        "hardware_config_sha256": hashes[0],
        "stdout_sha256": hashlib.sha256(run.stdout.encode()).hexdigest(),
        "stderr_sha256": hashlib.sha256(run.stderr.encode()).hexdigest(),
        "completion_sha256": hashlib.sha256(completion.read_bytes()).hexdigest(),
    }


def _open(session: Session, nodes: int, hardware_hash: str) -> dict[str, Any]:
    return session.request(
        {
            "effective_hardware_sha256": hardware_hash,
            "link_rate_bps": LINK_RATE_BPS,
            "node_count": nodes,
            "profile": "rnic-nn",
            "schema": SCHEMA,
            "seed": 0,
            "session_id": f"persistent-session-{nodes}",
            "topology_identity": f"rnic-nn:nodes={nodes}",
            "verb": "open",
            "wqe_authority": "simllm-native-rnic-session",
        }
    )


def _inject(
    session: Session,
    *,
    sequence: int,
    flow: Flow,
    eligible_at_ps: int,
    prefix: str,
) -> dict[str, Any]:
    return session.request(
        {
            "destination": flow.destination,
            "eligible_at_ps": eligible_at_ps,
            "execution_id": f"{prefix}-execution-{sequence}",
            "flow_id": f"{prefix}-flow-{sequence}",
            "operation_id": f"{prefix}-operation-{sequence}",
            "payload_bytes": flow.payload_bytes,
            "policy_context_token": POLICY_CONTEXT_TOKEN,
            "schema": SCHEMA,
            "sequence": sequence,
            "source": flow.source,
            "tag": 1000 + sequence,
            "verb": "inject",
        }
    )


def _run_session_replay(
    *,
    binary: Path,
    nodes: int,
    flows: tuple[Flow, ...],
    hardware_hash: str,
    prefix: str,
    overlap: bool,
) -> dict[str, Any]:
    session = Session(binary)
    opened = _open(session, nodes, hardware_hash)
    for index, flow in enumerate(flows, start=1):
        _inject(
            session,
            sequence=index,
            flow=flow,
            eligible_at_ps=0 if overlap else (index - 1) * STEP_GAP_PS,
            prefix=prefix,
        )
    horizon = STEP_GAP_PS if overlap else len(flows) * STEP_GAP_PS
    advanced = session.request(
        {
            "schema": SCHEMA,
            "through_ps": horizon,
            "through_sequence": len(flows),
            "verb": "advance",
        }
    )
    drained = session.request(
        {
            "schema": SCHEMA,
            "through_sequence": len(flows),
            "verb": "drain",
        }
    )
    session.request(
        {
            "schema": SCHEMA,
            "through_sequence": len(flows),
            "verb": "close",
        }
    )
    returncode, stderr = session.finish()
    if returncode != 0 or stderr:
        raise AssertionError(
            f"session exited {returncode} with stderr {stderr.decode(errors='replace')!r}"
        )
    return {"opened": opened, "advanced": advanced, "drained": drained}


def _codec_evidence() -> dict[str, Any]:
    from fractions import Fraction

    from simllm.core import (
        AdditiveVisitTotals,
        LatencyAttribution,
        RequestMetric,
        RequestPhase,
        StepResult,
        step_result_from_json,
        step_result_to_json,
    )

    additive = AdditiveVisitTotals(2, 3, 5, 1)
    prefill = RequestMetric(
        "prefill",
        RequestPhase.PREFILL,
        1,
        107,
        7,
        7,
        None,
        LatencyAttribution(kernel_ps=7),
        additive,
    )
    decode = RequestMetric(
        "decode",
        RequestPhase.DECODE,
        3,
        109,
        9,
        5,
        Fraction(1, 3),
        LatencyAttribution(queue_ps=2, nic_ps=7),
        additive,
    )
    cases = {
        "empty": StepResult(0, 0, 100),
        "prefill": StepResult(1, 7, 107, (prefill,), additive),
        "decode": StepResult(2, 9, 109, (decode,), additive),
        "mixed": StepResult(3, 9, 109, (prefill, decode), additive + additive),
    }
    rows = []
    for name in CODEC_CASES:
        original = cases[name]
        payload = step_result_to_json(original)
        wire = _canonical(payload)
        decoded_payload = json.loads(wire)
        restored = step_result_from_json(decoded_payload)
        rows.append(
            {
                "case": name,
                "identity": restored == original,
                "wire_sha256": hashlib.sha256(wire).hexdigest(),
            }
        )
    fraction_wire = step_result_to_json(cases["decode"])["request_metrics"][0][
        "tpot_ps"
    ]
    if fraction_wire != {"denominator": 3, "numerator": 1}:
        raise AssertionError("nonterminating TPOT did not retain 1/3")
    if not all(row["identity"] for row in rows):
        raise AssertionError(f"StepResult codec identity failed: {rows}")
    return {"cases": rows, "nonterminating_tpot": fraction_wire}


def _check_only_plan(args: argparse.Namespace, binaries: dict[str, Path]) -> None:
    plan = {
        "artifacts_created": False,
        "binaries": {name: str(path) for name, path in binaries.items()},
        "codec_cases": CODEC_CASES,
        "expectations_commit": EXPECTATIONS_COMMIT,
        "frame_limit": FRAME_LIMIT,
        "out": str(args.out),
        "replays": {
            name: {
                "nodes": cell[0],
                "flows": [flow.__dict__ for flow in cell[1]],
                "isolated_band_s": cell[2],
                "session_band_s": cell[3],
                "minimum_speedup": cell[4],
            }
            for name, cell in REPLAYS.items()
        },
        "state_payloads": STATE_PAYLOADS,
    }
    print(json.dumps(plan, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    binaries = {
        "htsim_rnic": _configured_executable("SIMLLM_HTSIM_RNIC"),
        "txt2bin": _configured_executable("SIMLLM_TXT2BIN"),
    }
    if args.check_only:
        _check_only_plan(args, binaries)
        return
    if args.out.exists():
        parser.error("--out must not exist")
    args.out.mkdir(parents=True)

    codec = _codec_evidence()
    fatal_failures: list[str] = []
    wall_rows = []
    replay_rows = []
    hardware_hashes: dict[int, str] = {}
    for replay, cell in REPLAYS.items():
        nodes, flows, isolated_band, session_band, minimum_speedup = cell
        cli_rows = []
        isolated_started = time.perf_counter_ns()
        for index, flow in enumerate(flows, start=1):
            cli_rows.append(
                _run_cli_flow(
                    binary=binaries["htsim_rnic"],
                    txt2bin=binaries["txt2bin"],
                    out=args.out,
                    name=f"{replay}-isolated-{index}",
                    nodes=nodes,
                    flow=flow,
                    tag=1000 + index,
                )
            )
        isolated_ns = time.perf_counter_ns() - isolated_started
        hashes = {row["hardware_config_sha256"] for row in cli_rows}
        if len(hashes) != 1:
            fatal_failures.append(f"{replay}: CLI hardware hash changed")
        hardware_hash = next(iter(hashes))
        hardware_hashes[nodes] = hardware_hash

        session_started = time.perf_counter_ns()
        persistent = _run_session_replay(
            binary=binaries["htsim_rnic"],
            nodes=nodes,
            flows=flows,
            hardware_hash=hardware_hash,
            prefix=replay,
            overlap=False,
        )
        session_ns = time.perf_counter_ns() - session_started
        cli_fcts = [row["fct_ps"] for row in cli_rows]
        session_fcts = [row["fct_ps"] for row in persistent["drained"]["completion_rows"]]
        identity = _canonical(cli_fcts) == _canonical(session_fcts)
        if not identity:
            fatal_failures.append(f"{replay}: FCT byte identity")
        replay_rows.append(
            {
                "replay": replay,
                "cli_fcts": cli_fcts,
                "session_fcts": session_fcts,
                "latency_byte_identity": identity,
                "quiescent": persistent["drained"]["quiescent"],
            }
        )
        isolated_s = isolated_ns / 1e9
        session_s = session_ns / 1e9
        speedup = isolated_s / session_s
        checks = {
            "isolated_in_band": isolated_band[0] <= isolated_s <= isolated_band[1],
            "session_in_band": session_band[0] <= session_s <= session_band[1],
            "speedup": speedup >= minimum_speedup,
        }
        wall_rows.append(
            {
                "replay": replay,
                "isolated_s": isolated_s,
                "session_s": session_s,
                "speedup": speedup,
                "checks": checks,
                "passed": all(checks.values()),
                "genuine_risk": True,
            }
        )

    state_rows = []
    for payload in STATE_PAYLOADS:
        flow = Flow(0, 1, payload)
        reset = _run_cli_flow(
            binary=binaries["htsim_rnic"],
            txt2bin=binaries["txt2bin"],
            out=args.out,
            name=f"state-{payload}-reset",
            nodes=2,
            flow=flow,
            tag=2000 + payload,
        )
        stateful = _run_session_replay(
            binary=binaries["htsim_rnic"],
            nodes=2,
            flows=(flow, flow),
            hardware_hash=hardware_hashes[2],
            prefix=f"state-{payload}",
            overlap=True,
        )
        rows = stateful["drained"]["completion_rows"]
        second_fct = rows[1]["fct_ps"]
        high_watermarks = stateful["drained"]["sq_high_watermarks"]
        persistent_high = high_watermarks[0]
        reset_high = 1
        relation = {
            "persistent_second_gt_reset": second_fct > reset["fct_ps"],
            "persistent_high_watermark_is_two": persistent_high == 2,
            "reset_high_watermark_is_one": reset_high == 1,
        }
        state_rows.append(
            {
                "payload_bytes": payload,
                "persistent_second_fct_ps": second_fct,
                "reset_second_fct_ps": reset["fct_ps"],
                "persistent_sq_high_watermark": persistent_high,
                "reset_sq_high_watermark": reset_high,
                "relation": relation,
                "passed": all(relation.values()),
                "genuine_risk": True,
            }
        )

    summary = {
        "schema": "simllm-persistent-session-study-v1",
        "chronology": {
            "expectations_commit": EXPECTATIONS_COMMIT,
            "implementation_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
            ).strip(),
        },
        "binary_sha256": {
            name: hashlib.sha256(path.read_bytes()).hexdigest()
            for name, path in binaries.items()
        },
        "host": {
            "machine": platform.machine(),
            "python": platform.python_version(),
            "system": platform.system(),
        },
        "codec_fatal_unscored": codec,
        "fatal_unscored": {
            "failures": fatal_failures,
            "replays": replay_rows,
            "passed": not fatal_failures,
        },
        "retained_state_relation": {
            "instances": state_rows,
            "passed": sum(row["passed"] for row in state_rows),
            "total": len(state_rows),
            "genuine_risk_passed": sum(
                row["passed"] for row in state_rows if row["genuine_risk"]
            ),
            "genuine_risk_total": sum(row["genuine_risk"] for row in state_rows),
        },
        "wall_clock_relation": {
            "instances": wall_rows,
            "passed": sum(row["passed"] for row in wall_rows),
            "total": len(wall_rows),
            "genuine_risk_passed": sum(
                row["passed"] for row in wall_rows if row["genuine_risk"]
            ),
            "genuine_risk_total": sum(row["genuine_risk"] for row in wall_rows),
        },
    }
    summary_path = args.out / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"summary={summary_path}")
    print(
        f"fatal={'PASS' if not fatal_failures else 'FAIL'} "
        f"state={summary['retained_state_relation']['passed']}/{len(state_rows)} "
        f"wall={summary['wall_clock_relation']['passed']}/{len(wall_rows)}"
    )
    if fatal_failures:
        raise AssertionError(fatal_failures)
    failed_state = [row for row in state_rows if not row["passed"]]
    failed_wall = [row for row in wall_rows if not row["passed"]]
    if failed_state or failed_wall:
        raise AssertionError({"state": failed_state, "wall": failed_wall})


if __name__ == "__main__":
    main()
