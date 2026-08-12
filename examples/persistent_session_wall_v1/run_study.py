"""Calibrate and qualify the frozen HTSIM-24 held-out wall family."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import statistics
import struct
import subprocess
import time
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, BinaryIO

from simllm.goal import GoalTrace

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
EXPECTATIONS_COMMIT = "522f1fdc7830fd378b15cc9177b764b299d21fec"
AUTHORED_SIMLLM_COMMIT = "76223875557a552deb5aa2c2c529a07f000135ba"
AUTHORED_HTSIM_COMMIT = "fc4400e4ca619223481536632074045cb6af2756"
SCHEMA = "simllm-htsim-flow-session-v1"
CALIBRATION_SCHEMA = "simllm-persistent-session-wall-calibration-v1"
BANDS_SCHEMA = "simllm-persistent-session-wall-bands-v1"
RESULT_SCHEMA = "simllm-persistent-session-wall-study-v1"
FRAME_LIMIT = 1 << 20
LINK_RATE_BPS = 400_000_000_000
STEP_GAP_PS = 1_000_000_000
POLICY_CONTEXT_TOKEN = 9001
CALIBRATION_REPETITIONS = 7


@dataclass(frozen=True)
class Flow:
    source: int
    destination: int
    payload_bytes: int


def _bidirectional_ring(nodes: int) -> tuple[Flow, ...]:
    flows = []
    for source in range(nodes):
        flows.append(Flow(source, (source + 1) % nodes, 4096))
        flows.append(Flow(source, (source - 1) % nodes, 8192))
    return tuple(flows)


REPLAYS = {
    "odd-3-biring": (3, _bidirectional_ring(3)),
    "odd-5-biring": (5, _bidirectional_ring(5)),
}


def _canonical(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()


def _configured_executable(variable: str) -> Path:
    raw = os.environ.get(variable)
    if not raw:
        raise ValueError(f"{variable} must name an executable")
    path = Path(raw).resolve()
    if not path.is_file() or not os.access(path, os.X_OK):
        raise ValueError(f"{variable} is not executable: {path}")
    return path


def _htsim_source_root() -> Path:
    raw = os.environ.get("HTSIM_SOURCE_ROOT")
    if not raw:
        raise ValueError("HTSIM_SOURCE_ROOT must name the backend source checkout")
    root = Path(raw).resolve()
    if not (root / "htsim" / "sim" / "datacenter" / "main_rnic.cpp").is_file():
        raise ValueError(f"HTSIM_SOURCE_ROOT is not an HTSIM checkout: {root}")
    return root


def _git_commit(root: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _replay_registry() -> dict[str, Any]:
    return {
        name: {
            "flows": [flow.__dict__ for flow in flows],
            "nodes": nodes,
        }
        for name, (nodes, flows) in REPLAYS.items()
    }


def _goal_text(nodes: int, flow: Flow, tag: int) -> str:
    trace = GoalTrace(nodes)
    trace.rank(flow.source).send(flow.payload_bytes, to=flow.destination, tag=tag)
    trace.rank(flow.destination).recv(flow.payload_bytes, source=flow.source, tag=tag)
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
    with completion.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 1:
        raise AssertionError(f"{name}: expected one CLI completion row")
    marker = "hardware_config_sha256="
    hashes = [token[len(marker) :] for token in run.stdout.split() if token.startswith(marker)]
    if len(hashes) != 1 or len(hashes[0]) != 64:
        raise AssertionError(f"{name}: missing CLI hardware hash")
    row = rows[0]
    return {
        "completion": {
            "destination": int(row["destination"]),
            "fct_ps": int(row["fct_ps"]),
            "payload_bytes": int(row["payload_bytes"]),
            "source": int(row["source"]),
            "tag": int(row["tag"]),
        },
        "hardware_config_sha256": hashes[0],
    }


def _run_isolated_replay(
    *,
    binary: Path,
    txt2bin: Path,
    out: Path,
    replay: str,
    sample: str,
    nodes: int,
    flows: tuple[Flow, ...],
) -> dict[str, Any]:
    started_ns = time.perf_counter_ns()
    rows = [
        _run_cli_flow(
            binary=binary,
            txt2bin=txt2bin,
            out=out,
            name=f"{replay}-{sample}-{index}",
            nodes=nodes,
            flow=flow,
            tag=1000 + index,
        )
        for index, flow in enumerate(flows, start=1)
    ]
    elapsed_ns = time.perf_counter_ns() - started_ns
    hashes = {row["hardware_config_sha256"] for row in rows}
    if len(hashes) != 1:
        raise AssertionError(f"{replay}: CLI hardware hash changed within replay")
    return {
        "completions": [row["completion"] for row in rows],
        "elapsed_ns": elapsed_ns,
        "hardware_config_sha256": next(iter(hashes)),
    }


def _frame(value: dict[str, Any]) -> bytes:
    body = _canonical(value).rstrip(b"\n")
    if len(body) > FRAME_LIMIT:
        raise ValueError("frame exceeds the frozen limit")
    return struct.pack(">I", len(body)) + body


def _read_exact(stream: BinaryIO, size: int) -> bytes:
    parts = []
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
    if type(value) is not dict or _canonical(value).rstrip(b"\n") != body:
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


def _run_session_replay(
    *,
    binary: Path,
    replay: str,
    nodes: int,
    flows: tuple[Flow, ...],
    hardware_hash: str,
) -> dict[str, Any]:
    started_ns = time.perf_counter_ns()
    session = Session(binary)
    session.request(
        {
            "effective_hardware_sha256": hardware_hash,
            "link_rate_bps": LINK_RATE_BPS,
            "node_count": nodes,
            "profile": "rnic-nn",
            "schema": SCHEMA,
            "seed": 0,
            "session_id": f"held-out-{replay}",
            "topology_identity": f"rnic-nn:nodes={nodes}",
            "verb": "open",
            "wqe_authority": "simllm-native-rnic-session",
        }
    )
    for index, flow in enumerate(flows, start=1):
        session.request(
            {
                "destination": flow.destination,
                "eligible_at_ps": (index - 1) * STEP_GAP_PS,
                "execution_id": f"{replay}-execution-{index}",
                "flow_id": f"{replay}-flow-{index}",
                "operation_id": f"{replay}-operation-{index}",
                "payload_bytes": flow.payload_bytes,
                "policy_context_token": POLICY_CONTEXT_TOKEN,
                "schema": SCHEMA,
                "sequence": index,
                "source": flow.source,
                "tag": 1000 + index,
                "verb": "inject",
            }
        )
    session.request(
        {
            "schema": SCHEMA,
            "through_ps": len(flows) * STEP_GAP_PS,
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
    elapsed_ns = time.perf_counter_ns() - started_ns
    return {
        "completion_rows": drained["completion_rows"],
        "elapsed_ns": elapsed_ns,
        "quiescent": drained.get("quiescent") is True,
        "returncode": returncode,
        "stderr": stderr.decode(errors="replace"),
    }


def _binary_manifest(binaries: dict[str, Path]) -> dict[str, str]:
    return {name: _sha256(path) for name, path in binaries.items()}


def _provenance(source_root: Path) -> dict[str, str]:
    return {
        "authored_against_htsim_commit": AUTHORED_HTSIM_COMMIT,
        "authored_against_simllm_commit": AUTHORED_SIMLLM_COMMIT,
        "expectations_commit": EXPECTATIONS_COMMIT,
        "observed_htsim_commit": _git_commit(source_root),
        "observed_simllm_commit": _git_commit(REPO_ROOT),
    }


def _physical_checks(completions: list[dict[str, int]]) -> dict[str, Any]:
    rows = []
    for completion in completions:
        floor_ps = completion["payload_bytes"] * 8 * 1_000_000_000_000 // LINK_RATE_BPS
        fct_ps = completion["fct_ps"]
        rows.append(
            {
                "fct_ps": fct_ps,
                "payload_bytes": completion["payload_bytes"],
                "serialization_floor_ps": floor_ps,
                "within_bounds": floor_ps <= fct_ps < STEP_GAP_PS,
            }
        )
    return {"passed": all(row["within_bounds"] for row in rows), "rows": rows}


def _calibrate(
    args: argparse.Namespace,
    binaries: dict[str, Path],
    source_root: Path,
) -> None:
    if args.out.exists():
        raise ValueError("--out must not exist")
    args.out.mkdir(parents=True)
    help_ns = []
    for _ in range(CALIBRATION_REPETITIONS):
        started_ns = time.perf_counter_ns()
        subprocess.run(
            [str(binaries["htsim_rnic"]), "--help"],
            check=True,
            capture_output=True,
        )
        help_ns.append(time.perf_counter_ns() - started_ns)

    replays = {}
    fatal_failures = []
    for replay, (nodes, flows) in REPLAYS.items():
        samples = [
            _run_isolated_replay(
                binary=binaries["htsim_rnic"],
                txt2bin=binaries["txt2bin"],
                out=args.out,
                replay=replay,
                sample=f"calibration-{sample}",
                nodes=nodes,
                flows=flows,
            )
            for sample in range(CALIBRATION_REPETITIONS)
        ]
        reference = samples[0]["completions"]
        physical = _physical_checks(reference)
        if not physical["passed"]:
            fatal_failures.append(f"{replay}: physical FCT bounds")
        if any(sample["completions"] != reference for sample in samples[1:]):
            fatal_failures.append(f"{replay}: calibration FCT identity")
        if len({sample["hardware_config_sha256"] for sample in samples}) != 1:
            fatal_failures.append(f"{replay}: hardware hash identity")
        replays[replay] = {
            "completions": reference,
            "hardware_config_sha256": samples[0]["hardware_config_sha256"],
            "isolated_ns": [sample["elapsed_ns"] for sample in samples],
            "physical_sanity": physical,
        }
    summary = {
        "binary_sha256": _binary_manifest(binaries),
        "fatal_unscored": {
            "failures": fatal_failures,
            "valid": not fatal_failures,
        },
        "help_process_ns": help_ns,
        "host": {
            "machine": platform.machine(),
            "python": platform.python_version(),
            "system": platform.system(),
        },
        "provenance": _provenance(source_root),
        "repetitions": CALIBRATION_REPETITIONS,
        "replay_registry": _replay_registry(),
        "replays": replays,
        "schema": CALIBRATION_SCHEMA,
        "session_invocations": 0,
    }
    (args.out / "summary.json").write_bytes(_canonical(summary))
    print(json.dumps(summary, indent=2, sort_keys=True))
    if fatal_failures:
        raise SystemExit(1)


def _load_canonical(path: Path, schema: str) -> tuple[dict[str, Any], bytes]:
    raw = path.read_bytes()
    value = json.loads(raw)
    if type(value) is not dict or _canonical(value) != raw:
        raise ValueError(f"{path} is not canonical JSON")
    if value.get("schema") != schema:
        raise ValueError(f"{path} has unsupported schema")
    return value, raw


def _fraction(value: Fraction) -> dict[str, int]:
    return {"denominator": value.denominator, "numerator": value.numerator}


def _derive_bands(calibration: dict[str, Any], calibration_raw: bytes) -> dict[str, Any]:
    if calibration.get("repetitions") != CALIBRATION_REPETITIONS:
        raise ValueError("calibration repetition count drifted")
    if calibration.get("replay_registry") != _replay_registry():
        raise ValueError("calibration replay registry drifted")
    if calibration.get("session_invocations") != 0:
        raise ValueError("calibration is not base-CLI-only")
    if not calibration.get("fatal_unscored", {}).get("valid"):
        raise ValueError("calibration fatal guards did not pass")
    help_ns = calibration["help_process_ns"]
    if len(help_ns) != CALIBRATION_REPETITIONS:
        raise ValueError("help calibration count drifted")
    rows = {}
    for replay in REPLAYS:
        isolated = calibration["replays"][replay]["isolated_ns"]
        if len(isolated) != CALIBRATION_REPETITIONS:
            raise ValueError(f"{replay}: isolated calibration count drifted")
        isolated_lower = min(isolated) // 2
        isolated_upper = 2 * max(isolated)
        session_lower = max(1, min(help_ns) // 4)
        session_upper = 9 * int(statistics.median(isolated)) // 10
        if not 0 < session_lower < session_upper:
            raise ValueError(f"{replay}: derived session band is empty")
        rows[replay] = {
            "isolated_ns": {"lower": isolated_lower, "upper": isolated_upper},
            "session_ns": {"lower": session_lower, "upper": session_upper},
            "speedup": {
                "maximum": _fraction(Fraction(isolated_upper, session_lower)),
                "minimum": {"denominator": 10, "numerator": 11},
            },
        }
    return {
        "binary_sha256": calibration["binary_sha256"],
        "calibration_sha256": hashlib.sha256(calibration_raw).hexdigest(),
        "expectations_commit": EXPECTATIONS_COMMIT,
        "formulas": {
            "isolated_lower": "floor(min(isolated_ns)/2)",
            "isolated_upper": "2*max(isolated_ns)",
            "maximum_speedup": "isolated_upper/session_lower",
            "minimum_speedup": "11/10",
            "session_lower": "max(1,floor(min(help_process_ns)/4))",
            "session_upper": "floor(9*median(isolated_ns)/10)",
        },
        "provenance": calibration["provenance"],
        "replay_registry": _replay_registry(),
        "replays": rows,
        "schema": BANDS_SCHEMA,
    }


def _lock(args: argparse.Namespace) -> None:
    calibration, raw = _load_canonical(args.calibration, CALIBRATION_SCHEMA)
    bands = _derive_bands(calibration, raw)
    if args.bands.exists():
        raise ValueError("--bands must not exist")
    args.bands.write_bytes(_canonical(bands))
    print(json.dumps(bands, indent=2, sort_keys=True))


def _validate_bands(
    calibration: dict[str, Any],
    calibration_raw: bytes,
    bands: dict[str, Any],
) -> None:
    expected = _derive_bands(calibration, calibration_raw)
    if bands != expected:
        raise ValueError("band lock differs from frozen formulas and calibration")


def _session_completions(rows: list[dict[str, Any]]) -> list[dict[str, int]]:
    return [
        {
            "destination": int(row["destination"]),
            "fct_ps": int(row["fct_ps"]),
            "payload_bytes": int(row["payload_bytes"]),
            "source": int(row["source"]),
            "tag": int(row["tag"]),
        }
        for row in rows
    ]


def _ratio_at_least(value_num: int, value_den: int, bound: dict[str, int]) -> bool:
    return value_num * bound["denominator"] >= value_den * bound["numerator"]


def _ratio_at_most(value_num: int, value_den: int, bound: dict[str, int]) -> bool:
    return value_num * bound["denominator"] <= value_den * bound["numerator"]


def _qualify(
    args: argparse.Namespace,
    binaries: dict[str, Path],
    source_root: Path,
) -> None:
    if args.out.exists():
        raise ValueError("--out must not exist")
    calibration, calibration_raw = _load_canonical(args.calibration, CALIBRATION_SCHEMA)
    bands, bands_raw = _load_canonical(args.bands, BANDS_SCHEMA)
    _validate_bands(calibration, calibration_raw, bands)
    if bands["binary_sha256"] != _binary_manifest(binaries):
        raise ValueError("configured executable digests differ from the band lock")
    args.out.mkdir(parents=True)

    wall_rows = []
    identity_rows = []
    fatal_failures = []
    for replay, (nodes, flows) in REPLAYS.items():
        isolated = _run_isolated_replay(
            binary=binaries["htsim_rnic"],
            txt2bin=binaries["txt2bin"],
            out=args.out,
            replay=replay,
            sample="qualification",
            nodes=nodes,
            flows=flows,
        )
        session = _run_session_replay(
            binary=binaries["htsim_rnic"],
            replay=replay,
            nodes=nodes,
            flows=flows,
            hardware_hash=isolated["hardware_config_sha256"],
        )

        isolated_ns = isolated["elapsed_ns"]
        session_ns = session["elapsed_ns"]
        locked = bands["replays"][replay]
        wall_checks = {
            "isolated_in_band": (
                locked["isolated_ns"]["lower"] <= isolated_ns <= locked["isolated_ns"]["upper"]
            ),
            "session_in_band": (
                locked["session_ns"]["lower"] <= session_ns <= locked["session_ns"]["upper"]
            ),
            "speedup_at_least_minimum": _ratio_at_least(
                isolated_ns, session_ns, locked["speedup"]["minimum"]
            ),
            "speedup_at_most_maximum": _ratio_at_most(
                isolated_ns, session_ns, locked["speedup"]["maximum"]
            ),
        }
        wall_rows.append(
            {
                "checks": wall_checks,
                "genuine_risk": True,
                "isolated_ns": isolated_ns,
                "passed": all(wall_checks.values()),
                "replay": replay,
                "session_ns": session_ns,
                "speedup": isolated_ns / session_ns,
            }
        )

        isolated_completions = isolated["completions"]
        session_completions = _session_completions(session["completion_rows"])
        physical = _physical_checks(isolated_completions)
        identity_checks = {
            "clean_session_exit": session["returncode"] == 0 and not session["stderr"],
            "completion_identity": isolated_completions == session_completions,
            "hardware_hash_matches_calibration": (
                isolated["hardware_config_sha256"]
                == calibration["replays"][replay]["hardware_config_sha256"]
            ),
            "latency_byte_identity": (
                _canonical([row["fct_ps"] for row in isolated_completions])
                == _canonical([row["fct_ps"] for row in session_completions])
            ),
            "physical_sanity": physical["passed"],
            "quiescent": session["quiescent"],
        }
        for name, passed in identity_checks.items():
            if not passed:
                fatal_failures.append(f"{replay}: {name}")
        identity_rows.append(
            {
                "checks": identity_checks,
                "isolated_completions": isolated_completions,
                "physical_sanity": physical,
                "replay": replay,
                "session_completions": session_completions,
            }
        )

    valid = not fatal_failures
    summary = {
        "band_lock_sha256": hashlib.sha256(bands_raw).hexdigest(),
        "binary_sha256": _binary_manifest(binaries),
        "calibration_sha256": hashlib.sha256(calibration_raw).hexdigest(),
        "fatal_unscored": {
            "failures": fatal_failures,
            "identity_replays": identity_rows,
            "valid": valid,
        },
        "host": {
            "machine": platform.machine(),
            "python": platform.python_version(),
            "system": platform.system(),
        },
        "provenance": _provenance(source_root),
        "schema": RESULT_SCHEMA,
        "wall_clock_relation": {
            "genuine_risk_passed": (sum(row["passed"] for row in wall_rows) if valid else None),
            "genuine_risk_total": len(wall_rows) if valid else None,
            "instances": wall_rows,
            "void": not valid,
        },
    }
    (args.out / "summary.json").write_bytes(_canonical(summary))
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not valid or not all(row["passed"] for row in wall_rows):
        raise SystemExit(1)


def _placeholder_calibration(binaries: dict[str, Path], source_root: Path) -> dict[str, Any]:
    replays = {}
    for replay, (_, flows) in REPLAYS.items():
        completions = [
            {
                "destination": flow.destination,
                "fct_ps": 2_000_000 + index * 10_000,
                "payload_bytes": flow.payload_bytes,
                "source": flow.source,
                "tag": 1000 + index,
            }
            for index, flow in enumerate(flows, start=1)
        ]
        base = 30_000_000 if replay == "odd-3-biring" else 50_000_000
        replays[replay] = {
            "completions": completions,
            "hardware_config_sha256": "0" * 64,
            "isolated_ns": [base + index * 100_000 for index in range(7)],
            "physical_sanity": _physical_checks(completions),
        }
    return {
        "binary_sha256": _binary_manifest(binaries),
        "fatal_unscored": {"failures": [], "valid": True},
        "help_process_ns": [4_000_000 + index * 10_000 for index in range(7)],
        "host": {"machine": "check-only", "python": "check-only", "system": "check-only"},
        "provenance": _provenance(source_root),
        "repetitions": CALIBRATION_REPETITIONS,
        "replay_registry": _replay_registry(),
        "replays": replays,
        "schema": CALIBRATION_SCHEMA,
        "session_invocations": 0,
    }


def _check_only(
    args: argparse.Namespace,
    binaries: dict[str, Path],
    source_root: Path,
) -> None:
    derived = None
    if args.phase in {"lock", "qualify"}:
        if args.calibration.is_file():
            calibration, calibration_raw = _load_canonical(args.calibration, CALIBRATION_SCHEMA)
        else:
            calibration = _placeholder_calibration(binaries, source_root)
            calibration_raw = _canonical(calibration)
        derived = _derive_bands(calibration, calibration_raw)
        if args.phase == "qualify":
            if args.bands.is_file():
                bands, _ = _load_canonical(args.bands, BANDS_SCHEMA)
            else:
                bands = derived
            _validate_bands(calibration, calibration_raw, bands)
    plan = {
        "artifacts_created": False,
        "binary_sha256": _binary_manifest(binaries),
        "calibration_repetitions": CALIBRATION_REPETITIONS,
        "derived_bands_validated": derived is not None,
        "expectations_commit": EXPECTATIONS_COMMIT,
        "out": str(args.out) if args.out is not None else None,
        "phase": args.phase,
        "provenance": _provenance(source_root),
        "replay_registry": _replay_registry(),
        "session_invoked": False,
    }
    print(json.dumps(plan, indent=2, sort_keys=True))


def _validate_arguments(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.phase == "calibrate":
        if args.out is None or args.calibration is not None or args.bands is not None:
            parser.error("calibrate requires only --out")
    elif args.phase == "lock":
        if args.calibration is None or args.bands is None or args.out is not None:
            parser.error("lock requires --calibration and --bands")
    elif args.phase == "qualify" and (
        args.calibration is None or args.bands is None or args.out is None
    ):
        parser.error("qualify requires --calibration, --bands and --out")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("calibrate", "lock", "qualify"), required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--bands", type=Path)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    _validate_arguments(parser, args)
    binaries = {
        "htsim_rnic": _configured_executable("SIMLLM_HTSIM_RNIC"),
        "txt2bin": _configured_executable("SIMLLM_TXT2BIN"),
    }
    source_root = _htsim_source_root()
    if args.check_only:
        _check_only(args, binaries, source_root)
    elif args.phase == "calibrate":
        _calibrate(args, binaries, source_root)
    elif args.phase == "lock":
        _lock(args)
    else:
        _qualify(args, binaries, source_root)


if __name__ == "__main__":
    main()
