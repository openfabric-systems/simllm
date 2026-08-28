"""Execute the frozen LogGOPSim ideal-network study."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import re
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from threading import Barrier, Lock
from typing import Any

from simllm.backends import (
    LogGopsimConfig,
    LogGopsimStepSink,
    LogGopsimStepSinkConfig,
    build_loggopsim_command,
    derive_loggp_params,
)
from simllm.compute import ComputeProvider, DurationEstimate, ModelDims
from simllm.core import RequestPhase, ScheduledRequest, StepRecord
from simllm.goal import GoalTrace, to_binary
from simllm.preplay import (
    PREPLAY_TRACE_SCHEMA,
    ForwardPhase,
    RoutedExperts,
    RoutedLayer,
    RoutedRequest,
    RoutedToken,
)
from simllm.traffic import ExpertPlacementSnapshot, RoutedMoeSupply
from simllm.traffic.patterns import pairwise_all_to_allv, ring_allreduce

STUDY_DIR = Path(__file__).resolve().parent
REPO_ROOT = STUDY_DIR.parents[1]
GOALS_DIR = STUDY_DIR / "goals"
EXPECTATIONS_COMMIT = "6523a62"
PINNED_BINARY_SHA256 = (
    "7e0f13ee3c87a20e9d2e94dbbd74c46075fd03df2f1b04d1ed9739c43ee0a2bf"
)
PINNED_BULK_GOALS = {
    "alltoall-p64-s1048576": (
        "e96a335a11061bc2191834e02b11bd937a992dc2c43af4628454de506fb1235f"
    ),
    "chain-p64-l24-s4096": (
        "9ba680241997c0e1af2913e85fba48184b9a02d2d08c2c7f1ad88c9a9ea29338"
    ),
}
HTSIM_DIAGNOSTIC_BASELINE_SECONDS = 7.252
MAX_FINISH_RE = re.compile(
    rb"^Maximum finishing time at host (\d+): (\d+)", re.MULTILINE
)
HOST_TIME_RE = re.compile(rb"^Host (\d+): (\d+)\s*$", re.MULTILINE)
UNMATCHED_RE = re.compile(
    rb"^(?:unexpected|receive) queue on host \d+ contains \d+ elements!$",
    re.MULTILINE,
)


@dataclass(frozen=True)
class CellSpec:
    cell_id: str
    family: str
    goal: str
    expected_ns: int
    latency_ns: int
    overhead_ns: int
    message_gap_ns: int
    exact_g: str
    byte_overhead_ns: int
    threshold_bytes: int
    expected_host_finish_ns: tuple[tuple[int, int], ...] = ()


@dataclass(frozen=True)
class NativeExecution:
    stdout: bytes
    stderr: bytes
    elapsed_seconds: float
    max_finish_ns: int
    host_finish_ns: dict[int, int]


class NativeEvidenceRecorder:
    """Append native stdout, stderr and resolved argv to one attempt."""

    def __init__(self, attempt_dir: Path) -> None:
        self._root = attempt_dir / "native"
        self._counts: dict[str, int] = defaultdict(int)
        self._lock = Lock()

    def capture(
        self,
        label: str,
        argv: list[str] | tuple[str, ...],
        completed: subprocess.CompletedProcess[bytes],
        elapsed_seconds: float,
        *,
        max_finish_ns: int | None,
        host_finish_ns: dict[int, int] | None,
    ) -> None:
        safe_label = re.sub(r"[^A-Za-z0-9_.-]", "_", label)
        with self._lock:
            self._counts[safe_label] += 1
            execution = self._counts[safe_label]
            directory = self._root / safe_label
            directory.mkdir(parents=True, exist_ok=True)
        stem = f"execution-{execution:04d}"
        stdout_name = f"{stem}.stdout"
        stderr_name = f"{stem}.stderr"
        (directory / stdout_name).write_bytes(completed.stdout)
        (directory / stderr_name).write_bytes(completed.stderr)
        provenance = {
            "argv": list(argv),
            "elapsed_seconds": elapsed_seconds,
            "host_finish_ns": host_finish_ns,
            "max_finish_ns": max_finish_ns,
            "returncode": completed.returncode,
            "stderr_file": stderr_name,
            "stderr_sha256": _sha256_bytes(completed.stderr),
            "stdout_file": stdout_name,
            "stdout_sha256": _sha256_bytes(completed.stdout),
        }
        (directory / f"{stem}.json").write_text(
            json.dumps(provenance, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_path(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _parse_max_finish_ns(stdout: bytes) -> int:
    match = MAX_FINISH_RE.search(stdout)
    if match is not None:
        return int(match.group(2))
    host_rows = list(_parse_host_finish_ns(stdout).items())
    if not host_rows:
        raise RuntimeError("LogGOPSim stdout has no maximum host finishing time")
    return max(host_rows, key=lambda row: (row[1], -row[0]))[1]


def _parse_host_finish_ns(stdout: bytes) -> dict[int, int]:
    return {int(host): int(value) for host, value in HOST_TIME_RE.findall(stdout)}


def _stdout_byte_identical(left: bytes, right: bytes) -> bool:
    return left == right


def _execute(
    argv: list[str] | tuple[str, ...],
    recorder: NativeEvidenceRecorder,
    label: str,
) -> NativeExecution:
    started = time.perf_counter_ns()
    completed = subprocess.run(argv, capture_output=True, check=False)
    elapsed = (time.perf_counter_ns() - started) / 1_000_000_000
    if completed.returncode != 0:
        recorder.capture(
            label,
            argv,
            completed,
            elapsed,
            max_finish_ns=None,
            host_finish_ns=None,
        )
        raise RuntimeError(
            f"LogGOPSim exited {completed.returncode}: "
            f"{completed.stderr.decode(errors='replace').strip()}"
        )
    max_finish_ns = _parse_max_finish_ns(completed.stdout)
    host_finish_ns = _parse_host_finish_ns(completed.stdout)
    recorder.capture(
        label,
        argv,
        completed,
        elapsed,
        max_finish_ns=max_finish_ns,
        host_finish_ns=host_finish_ns,
    )
    if UNMATCHED_RE.search(completed.stdout):
        raise RuntimeError("LogGOPSim reported unmatched messages")
    return NativeExecution(
        stdout=completed.stdout,
        stderr=completed.stderr,
        elapsed_seconds=elapsed,
        max_finish_ns=max_finish_ns,
        host_finish_ns=host_finish_ns,
    )


def _execute_pair(
    argv: list[str] | tuple[str, ...],
    recorder: NativeEvidenceRecorder,
    label: str,
) -> tuple[NativeExecution, NativeExecution]:
    """Launch the FG-4 pair against one shared coarse-clock boundary.

    The pinned binary prints an integer-second performance diagnostic as part
    of stdout. Starting immediately after a wall-second rollover prevents two
    subsecond executions from landing on opposite sides of that diagnostic's
    boundary while leaving the simulated schedule and argv unchanged.
    """

    barrier = Barrier(3)

    def worker() -> NativeExecution:
        barrier.wait()
        return _execute(argv, recorder, label)

    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="loggopsim-fg4") as pool:
        futures = (pool.submit(worker), pool.submit(worker))
        while time.time() % 1.0 > 0.02:
            time.sleep(0.002)
        barrier.wait()
        return futures[0].result(), futures[1].result()


def _config(spec: CellSpec, goal_bin: Path) -> LogGopsimConfig:
    return LogGopsimConfig(
        goal_bin=goal_bin,
        latency_ns=spec.latency_ns,
        overhead_ns=spec.overhead_ns,
        message_gap_ns=spec.message_gap_ns,
        byte_gap_ns=float(spec.exact_g),
        byte_gap_ns_string=spec.exact_g,
        byte_overhead_ns=spec.byte_overhead_ns,
        rendezvous_threshold_bytes=spec.threshold_bytes,
        network_type="LogGP",
    )


def _portable_path(path: Path, attempt_dir: Path) -> str:
    try:
        return path.resolve().relative_to(attempt_dir.resolve()).as_posix()
    except ValueError as exc:
        raise RuntimeError(f"recorded GOAL path escapes the attempt directory: {path}") from exc


def _portable_argv(
    argv: list[str] | tuple[str, ...], attempt_dir: Path
) -> list[str]:
    recorded = list(argv)
    recorded[0] = "LogGOPSim"
    goal_index = recorded.index("-f") + 1
    recorded[goal_index] = _portable_path(Path(recorded[goal_index]), attempt_dir)
    return recorded


def _run_exact_cell(
    spec: CellSpec,
    binary: Path,
    goal_bin: Path,
    attempt_dir: Path,
    recorder: NativeEvidenceRecorder,
) -> tuple[dict[str, Any], tuple[bytes, bytes]]:
    argv = build_loggopsim_command(binary, _config(spec, goal_bin))
    first, second = _execute_pair(argv, recorder, f"exact-{spec.cell_id}")
    deterministic = _stdout_byte_identical(first.stdout, second.stdout)
    expected_hosts = dict(spec.expected_host_finish_ns)
    if expected_hosts:
        observed_hosts = {
            host: first.host_finish_ns.get(host) for host in sorted(expected_hosts)
        }
        score_denominator = len(expected_hosts)
        score_passed = sum(
            observed_hosts[host] == expected
            for host, expected in expected_hosts.items()
        )
    else:
        observed_hosts = {}
        score_denominator = 1
        score_passed = int(first.max_finish_ns == spec.expected_ns)
    row = {
        "id": spec.cell_id,
        "family": spec.family,
        "evidence_class": "exact-oracle",
        "goal": spec.goal,
        "parameters": {
            "L_ns": spec.latency_ns,
            "o_ns": spec.overhead_ns,
            "g_ns": spec.message_gap_ns,
            "G_ns_per_byte": spec.exact_g,
            "O_ns_per_byte": spec.byte_overhead_ns,
            "S_bytes": spec.threshold_bytes,
        },
        "argv": _portable_argv(argv, attempt_dir),
        "expected_max_finish_ns": spec.expected_ns,
        "observed_max_finish_ns": first.max_finish_ns,
        "expected_host_finish_ns": expected_hosts,
        "observed_host_finish_ns": observed_hosts,
        "score_denominator": score_denominator,
        "score_passed": score_passed,
        "stdout_sha256": _sha256_bytes(first.stdout),
        "stdout_byte_identical": deterministic,
        "executions": 2,
        "passed": score_passed == score_denominator and deterministic,
    }
    return row, (first.stdout, second.stdout)


def _exact_specs() -> tuple[CellSpec, ...]:
    return (
        CellSpec("E1-base", "E1", "pair-17", 175, 100, 10, 7, "3.0", 0, 50),
        CellSpec("E1-L", "E1", "pair-17", 205, 130, 10, 7, "3.0", 0, 50),
        CellSpec("E1-o", "E1", "pair-17", 181, 100, 13, 7, "3.0", 0, 50),
        CellSpec("E1-g", "E1", "pair-17", 179, 100, 10, 11, "3.0", 0, 50),
        CellSpec("E1-G", "E1", "pair-17", 207, 100, 10, 7, "5.0", 0, 50),
        CellSpec("E1-O", "E1", "pair-17", 239, 100, 10, 7, "3.0", 2, 50),
        CellSpec(
            "E1-S50",
            "E1",
            "pair-51",
            277,
            100,
            10,
            7,
            "3.0",
            0,
            50,
            ((0, 277), (1, 277)),
        ),
        CellSpec(
            "E1-S51",
            "E1",
            "pair-51",
            277,
            100,
            10,
            7,
            "3.0",
            0,
            51,
            ((0, 10), (1, 277)),
        ),
        CellSpec("E1-held", "E1", "pair-64", 579, 137, 19, 11, "0.25", 3, 64),
        CellSpec("E2-s50", "E2", "pair-50", 0, 0, 0, 0, "0.02", 0, 10_000_000),
        CellSpec("E2-s51", "E2", "pair-51", 1, 0, 0, 0, "0.02", 0, 10_000_000),
        CellSpec("E2-s64", "E2", "pair-64", 1, 0, 0, 0, "0.02", 0, 10_000_000),
        CellSpec("E2-s101", "E2", "pair-101", 2, 0, 0, 0, "0.02", 0, 10_000_000),
        CellSpec("E2-s4096", "E2", "pair-4096", 81, 0, 0, 0, "0.02", 0, 10_000_000),
        CellSpec("E2-s65536", "E2", "pair-65536", 1310, 0, 0, 0, "0.02", 0, 10_000_000),
        CellSpec("E2-s1048576", "E2", "pair-1048576", 20971, 0, 0, 0, "0.02", 0, 10_000_000),
        CellSpec("E3-requires", "E3", "dep-requires", 140, 101, 999, 13, "0.02", 7, 5),
        CellSpec("E3-irequires", "E3", "dep-irequires", 123, 101, 999, 13, "0.02", 7, 5),
        CellSpec("E4-A-round", "E4", "ring-round-s17", 175, 100, 10, 7, "3.0", 0, 50),
        CellSpec("E4-A-chain", "E4", "ring-chain-s17", 700, 100, 10, 7, "3.0", 0, 50),
        CellSpec("E4-A-allreduce", "E4", "ring-allreduce-p4-s17", 1050, 100, 10, 7, "3.0", 0, 50),
        CellSpec("E4-B-round", "E4", "ring-round-s4096", 6581, 2500, 1500, 1000, "0.02", 0, 65535),
        CellSpec("E4-B-chain", "E4", "ring-chain-s4096", 26324, 2500, 1500, 1000, "0.02", 0, 65535),
        CellSpec("E4-B-allreduce", "E4", "ring-allreduce-p4-s4096", 39486, 2500, 1500, 1000, "0.02", 0, 65535),
        CellSpec("E5-A", "E5", "alltoall-p4-s17", 1462, 1000, 80, 7, "3.0", 0, 50),
        CellSpec("E5-B", "E5", "alltoall-p4-s4096", 13569, 10000, 100, 50, "0.25", 0, 65535),
        CellSpec("E6-chain", "E6", "chain-p64-l24-s4096", 4422432, 2500, 1500, 1000, "0.02", 0, 65535),
        CellSpec("E6-alltoall", "E6", "alltoall-p64-s1048576", 1408731, 2500, 1500, 1000, "0.02", 0, 65535),
    )


def _render_alltoall_64(size: int) -> GoalTrace:
    trace = GoalTrace(64)
    send_bytes = {
        (source, destination): size
        for source in range(64)
        for destination in range(64)
        if source != destination
    }
    pairwise_all_to_allv(trace, list(range(64)), send_bytes, 500)
    return trace


def _render_chain_64(size: int, layers: int) -> GoalTrace:
    trace = GoalTrace(64)
    previous: dict[int, str] = {}
    for layer in range(layers):
        for site in range(2):
            site_done: dict[int, str] = {}
            for group in range(8):
                ranks = list(range(group * 8, (group + 1) * 8))
                done = ring_allreduce(
                    trace,
                    ranks,
                    8 * size,
                    base_tag=1000 + (2 * layer + site) * 14,
                    after=previous,
                )
                site_done.update(done)
            previous = site_done
    return trace


def _tracked_goal_hashes() -> dict[str, str]:
    rows: dict[str, str] = {}
    for line in (STUDY_DIR / "goals.sha256").read_text(encoding="utf-8").splitlines():
        digest, relative = line.split(maxsplit=1)
        rows[Path(relative).stem] = digest
    return rows


def _prepare_goals(run_dir: Path, txt2bin: Path) -> tuple[dict[str, Path], dict[str, Any]]:
    output = run_dir / "goals"
    output.mkdir(parents=True, exist_ok=True)
    tracked_hashes = _tracked_goal_hashes()
    goal_bins: dict[str, Path] = {}
    tracked_rows = []
    for name, expected in sorted(tracked_hashes.items()):
        source = GOALS_DIR / f"{name}.goal"
        observed = _sha256_path(source)
        tracked_rows.append(
            {"name": name, "expected_sha256": expected, "observed_sha256": observed}
        )
        if observed != expected:
            raise RuntimeError(f"tracked GOAL hash mismatch: {name}")
        goal_bins[name] = to_binary(source, output / f"{name}.bin", tool=txt2bin)

    generated = {
        "alltoall-p64-s1048576": _render_alltoall_64(1_048_576),
        "chain-p64-l24-s4096": _render_chain_64(4096, 24),
    }
    generated_rows = []
    for name, trace in generated.items():
        target = trace.write(output / f"{name}.goal")
        observed = _sha256_path(target)
        expected = PINNED_BULK_GOALS[name]
        generated_rows.append(
            {"name": name, "expected_sha256": expected, "observed_sha256": observed}
        )
        if observed != expected:
            raise RuntimeError(f"generated GOAL hash mismatch: {name}")
        goal_bins[name] = to_binary(target, output / f"{name}.bin", tool=txt2bin)
    return goal_bins, {"tracked": tracked_rows, "generated": generated_rows}


class _FixedProvider(ComputeProvider):
    def estimate(self, kernel, gpu):
        return DurationEstimate(duration_ps=1_000, bound="declared")


def _live_dims() -> ModelDims:
    return ModelDims(
        num_layers=1,
        hidden_size=64,
        intermediate_size=128,
        num_heads=4,
        num_kv_heads=4,
        head_size=16,
        vocab_size=256,
        dtype_bytes=2,
        num_experts=2,
        top_k=1,
        moe_intermediate_size=64,
        local_num_experts=1,
    )


def _live_record() -> StepRecord:
    return StepRecord(
        step_index=0,
        virtual_time_ps=0,
        scheduled=[
            ScheduledRequest(
                "request-0",
                RequestPhase.PREFILL,
                num_new_tokens=1,
                context_length=1,
            )
        ],
    )


def _live_supply() -> RoutedMoeSupply:
    token = RoutedToken(
        phase=ForwardPhase.PREFILL,
        token_index=0,
        token_id=1,
        layers=(RoutedLayer(layer_index=0, expert_ids=(0,)),),
    )
    routed = RoutedExperts(
        trace_schema=PREPLAY_TRACE_SCHEMA,
        trace_sha256="a" * 64,
        expert_count=2,
        top_k=1,
        moe_layer_indices=(0,),
        requests=(
            RoutedRequest(
                request_id="request-0",
                prompt_token_count=1,
                output_token_count=1,
                tokens=(token,),
            ),
        ),
    )
    return RoutedMoeSupply(
        engine_rank=0,
        routed_experts=routed,
        placements=(
            ExpertPlacementSnapshot(
                placement_epoch=0,
                expert_owners=((0, 0, 0), (0, 1, 1)),
            ),
        ),
        step_placement_epochs=((0, 0),),
    )


def _live_ttft(sink: LogGopsimStepSink, record: StepRecord) -> tuple[int, Any]:
    from simllm.adapters.vllm import SimExecutorConfig, TranslatedStep
    from simllm.adapters.vllm.executor import _SimStepRuntime
    from simllm.backends import HtsimRequestMetricReducer

    runtime = _SimStepRuntime(
        config=SimExecutorConfig(),
        step_sink=sink,
        fallback_latency=lambda translated: 0,
    )
    translated = TranslatedStep(
        record=record,
        req_ids=["request-0"],
        produces_token=[True],
    )
    result = runtime.settle(translated)
    reducer = HtsimRequestMetricReducer({"request-0": 0})
    metrics = reducer.consume(record, result, sink.locality_outcomes[-1])
    if len(metrics) != 1:
        raise RuntimeError("live fixture did not produce one request metric")
    return metrics[0].ttft_ps, result


def _run_live_family(
    binary: Path,
    txt2bin: Path,
    run_dir: Path,
    recorder: NativeEvidenceRecorder,
) -> tuple[dict[str, Any], tuple[bytes, bytes]]:
    common = {
        "ep_ranks": (0, 1),
        "dims": _live_dims(),
        "latency_ns": 100,
        "binary": binary,
        "txt2bin": txt2bin,
        "provider": _FixedProvider(),
        "routed_moe_supply": _live_supply(),
    }
    record = _live_record()
    remote = LogGopsimStepSink(
        LogGopsimStepSinkConfig(
            **common,
            tp_ranks=(0, 1),
            workdir=run_dir / "live" / "remote",
        )
    )
    control = LogGopsimStepSink(
        LogGopsimStepSinkConfig(
            **common,
            tp_ranks=(0,),
            workdir=run_dir / "live" / "control",
        )
    )
    remote_ttft, remote_result = _live_ttft(remote, record)
    control_ttft, control_result = _live_ttft(control, record)
    provenance = remote.provenance
    if len(provenance.invocations) != 1:
        raise RuntimeError("live fixture must emit exactly one fabric artifact")
    if control.provenance.invocations:
        raise RuntimeError("live control must emit zero fabric artifacts")
    invocation = provenance.invocations[0]
    first, second = _execute_pair(invocation.argv, recorder, "live-L1-independent")
    independent_ps = first.max_finish_ns * 1000
    locality = remote.locality_outcomes[-1]
    nonzero_fabric_services = tuple(
        value for value in locality.fabric_phase_service_ps if value
    )
    if len(nonzero_fabric_services) != 1:
        raise RuntimeError("live fixture must report one nonzero fabric service")
    sink_network_ps = sum(locality.fabric_phase_service_ps)
    network_delta_ps = remote_ttft - control_ttft
    parameter_400 = derive_loggp_params(
        rate_bits_per_second=400_000_000_000,
        latency_ns=100,
    )
    parameter_200 = derive_loggp_params(
        rate_bits_per_second=200_000_000_000,
        latency_ns=100,
    )
    rows = [
        {
            "id": "L1a",
            "evidence_class": "live-identity",
            "expected_delta_ps": 0,
            "observed_delta_ps": sink_network_ps - independent_ps,
            "passed": sink_network_ps == independent_ps,
        },
        {
            "id": "L1b",
            "evidence_class": "live-identity",
            "expected_network_delta_ps": independent_ps,
            "observed_ttft_delta_ps": network_delta_ps,
            "passed": network_delta_ps == independent_ps,
        },
        {
            "id": "L1c",
            "evidence_class": "live-identity",
            "expected": {"400000000000": "0.02", "200000000000": "0.04"},
            "observed": {
                "400000000000": parameter_400.exact_g_string,
                "200000000000": parameter_200.exact_g_string,
            },
            "passed": (
                parameter_400.exact_g_string == "0.02"
                and parameter_200.exact_g_string == "0.04"
            ),
        },
    ]
    result = {
        "class": "live-identity",
        "denominator": 3,
        "passed": sum(row["passed"] for row in rows),
        "rows": rows,
        "remote_step_result": {
            "step_latency_ps": remote_result.step_latency_ps,
            "completed_at_ps": remote_result.completed_at_ps,
            "ttft_ps": remote_ttft,
        },
        "control_step_result": {
            "step_latency_ps": control_result.step_latency_ps,
            "completed_at_ps": control_result.completed_at_ps,
            "ttft_ps": control_ttft,
        },
        "network_makespan_ps": sink_network_ps,
        "independent_max_finish_ps": independent_ps,
        "binary_sha256": provenance.binary_sha256,
        "parameters": provenance.parameters.to_json(),
        "invocation": {
            "goal_path": _portable_path(invocation.goal_path, run_dir),
            "goal_sha256": invocation.goal_sha256,
            "goal_binary_sha256": invocation.goal_binary_sha256,
            "argv": _portable_argv(invocation.argv, run_dir),
            "exact_g_string": invocation.exact_g_string,
            "sink_max_finish_ps": invocation.max_finish_ps,
            "independent_max_finish_ps": independent_ps,
            "stdout_sha256": _sha256_bytes(first.stdout),
            "stdout_byte_identical": _stdout_byte_identical(
                first.stdout, second.stdout
            ),
            "executions": 2,
        },
    }
    return result, (first.stdout, second.stdout)


def _wall_specs() -> tuple[tuple[str, CellSpec, float], ...]:
    return (
        (
            "W1",
            CellSpec("W1", "W", "ring-allreduce-p4-s4096", 39486, 2500, 1500, 1000, "0.02", 0, 65535),
            1.0,
        ),
        (
            "W2",
            CellSpec("W2", "W", "alltoall-p64-s1048576", 1408731, 2500, 1500, 1000, "0.02", 0, 65535),
            5.0,
        ),
        (
            "W3",
            CellSpec("W3", "W", "chain-p64-l24-s4096", 4422432, 2500, 1500, 1000, "0.02", 0, 65535),
            30.0,
        ),
    )


def _machine_disclosure() -> dict[str, Any]:
    fields: dict[str, str] = {}
    completed = subprocess.run(["lscpu"], capture_output=True, text=True, check=False)
    if completed.returncode == 0:
        for line in completed.stdout.splitlines():
            if ":" in line:
                key, value = line.split(":", 1)
                fields[key.strip()] = value.strip()
    return {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "logical_cpus": os.cpu_count(),
        "cpu_model": fields.get("Model name", platform.processor() or "unknown"),
        "cpu_sockets": fields.get("Socket(s)", "unknown"),
        "cores_per_socket": fields.get("Core(s) per socket", "unknown"),
        "threads_per_core": fields.get("Thread(s) per core", "unknown"),
    }


def _run_wall_family(
    binary: Path,
    goal_bins: dict[str, Path],
    attempt_dir: Path,
    recorder: NativeEvidenceRecorder,
) -> tuple[dict[str, Any], list[tuple[bytes, bytes]]]:
    rows = []
    stdout_pairs = []
    for row_id, spec, ceiling in _wall_specs():
        argv = build_loggopsim_command(binary, _config(spec, goal_bins[spec.goal]))
        observations = [
            _execute(argv, recorder, f"wall-{row_id}-sample") for _ in range(7)
        ]
        determinism_first, determinism_second = _execute_pair(
            argv, recorder, f"wall-{row_id}-determinism"
        )
        stdout_pairs.append((determinism_first.stdout, determinism_second.stdout))
        samples = [observation.elapsed_seconds for observation in observations]
        median = statistics.median(samples)
        deterministic = _stdout_byte_identical(
            determinism_first.stdout, determinism_second.stdout
        )
        finish_values = {observation.max_finish_ns for observation in observations}
        passed = (
            median <= ceiling
            and deterministic
            and finish_values == {spec.expected_ns}
        )
        rows.append(
            {
                "id": row_id,
                "evidence_class": "wall-time",
                "goal": spec.goal,
                "argv": _portable_argv(argv, attempt_dir),
                "exact_g_string": spec.exact_g,
                "expected_max_finish_ns": spec.expected_ns,
                "observed_max_finish_ns": observations[0].max_finish_ns,
                "samples_seconds": samples,
                "median_seconds": median,
                "ceiling_seconds": ceiling,
                "stdout_sha256": _sha256_bytes(determinism_first.stdout),
                "stdout_byte_identical": deterministic,
                "wall_executions": 7,
                "determinism_executions": 2,
                "unscored_ratio_to_htsim_diagnostic": (
                    median / HTSIM_DIAGNOSTIC_BASELINE_SECONDS
                ),
                "ratio_context": (
                    "unscored because schedules and machine conditions are not identical"
                ),
                "passed": passed,
            }
        )
    result = {
        "class": "wall-time",
        "denominator": 3,
        "passed": sum(row["passed"] for row in rows),
        "machine": _machine_disclosure(),
        "htsim_diagnostic_baseline_seconds": HTSIM_DIAGNOSTIC_BASELINE_SECONDS,
        "rows": rows,
    }
    return result, stdout_pairs


def _fg5_holds(spec: CellSpec, payload_bytes: int) -> bool:
    d_g = int((payload_bytes - 1) * float(spec.exact_g))
    return spec.latency_ns + spec.overhead_ns + d_g >= (
        2 * max(spec.overhead_ns, spec.message_gap_ns + d_g) + spec.overhead_ns
    )


def _git_hash(revision: str) -> str:
    return subprocess.run(
        ["git", "rev-parse", revision],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def _is_ancestor(older: str, newer: str) -> bool:
    return subprocess.run(
        ["git", "merge-base", "--is-ancestor", older, newer],
        cwd=REPO_ROOT,
        capture_output=True,
        check=False,
    ).returncode == 0


def _binary_hash_matches(binary_hash: str) -> bool:
    return binary_hash == PINNED_BINARY_SHA256


def _goal_hashes_match(goal_hashes: dict[str, Any]) -> bool:
    return all(
        row["expected_sha256"] == row["observed_sha256"]
        for group in goal_hashes.values()
        for row in group
    )


def _scored_invocation_valid(row: dict[str, Any]) -> bool:
    argv = row["argv"]
    required_flags = ("-f", "-L", "-o", "-g", "-G", "-O", "-S", "-n")
    if len(argv) != 1 + 2 * len(required_flags):
        return False
    if any(argv.count(flag) != 1 for flag in required_flags):
        return False
    exact_g = row.get("exact_g_string", row.get("parameters", {}).get("G_ns_per_byte"))
    if argv[argv.index("-G") + 1] != exact_g:
        return False
    recorded_maximum = row.get(
        "observed_max_finish_ns", row.get("sink_max_finish_ps")
    )
    return isinstance(recorded_maximum, int) and recorded_maximum >= 0


def _mutation_control(mutant: str, rejected: bool) -> dict[str, Any]:
    return {
        "kind": "predicate-exercised",
        "mutant": mutant,
        "rejected": rejected,
    }


def _fatal_guards(
    *,
    binary_hash: str,
    goal_hashes: dict[str, Any],
    exact_rows: list[dict[str, Any]],
    live: dict[str, Any],
    wall: dict[str, Any],
    stdout_pairs: list[tuple[bytes, bytes]],
    expectations_hash: str,
    implementation_hash: str,
) -> list[dict[str, Any]]:
    all_goals_match = _goal_hashes_match(goal_hashes)
    scored_invocations = exact_rows + wall["rows"]
    scored_invocations.append(live["invocation"])
    argv_complete = all(_scored_invocation_valid(row) for row in scored_invocations)
    deterministic = all(
        _stdout_byte_identical(first, second) for first, second in stdout_pairs
    )
    e5_specs = [spec for spec in _exact_specs() if spec.family == "E5"]
    e5_payloads = {"E5-A": 17, "E5-B": 4096}
    e5_holds = all(_fg5_holds(spec, e5_payloads[spec.cell_id]) for spec in e5_specs)
    chronology = _is_ancestor(expectations_hash, implementation_hash)
    mutant_row = dict(exact_rows[0])
    mutant_argv = list(mutant_row["argv"])
    gap_index = mutant_argv.index("-G")
    del mutant_argv[gap_index : gap_index + 2]
    mutant_row["argv"] = mutant_argv
    mutant_goal_hashes = json.loads(json.dumps(goal_hashes))
    mutant_goal_hashes["tracked"][0]["observed_sha256"] = "0" * 64
    deterministic_reference = stdout_pairs[0][0]
    deterministic_mutant = deterministic_reference + b"\nmutation-control"
    mutant_binary_hash = _sha256_bytes(bytes.fromhex(binary_hash) + b"\0")
    return [
        {
            "id": "FG-1",
            "held": _binary_hash_matches(binary_hash),
            "enforcement": "runtime",
            "evaluated": "binary SHA-256 equals the frozen literal before any cell",
            "mutation_control": _mutation_control(
                "SHA-256 of the observed digest bytes plus one zero byte",
                not _binary_hash_matches(mutant_binary_hash),
            ),
        },
        {
            "id": "FG-2",
            "held": all_goals_match,
            "enforcement": "runtime",
            "evaluated": "every tracked and generated GOAL equals its frozen digest",
            "mutation_control": _mutation_control(
                "first observed GOAL digest replaced with 64 zeroes",
                not _goal_hashes_match(mutant_goal_hashes),
            ),
        },
        {
            "id": "FG-3",
            "held": argv_complete,
            "enforcement": "runtime",
            "evaluated": "every native scored row records full argv, exact G and maximum host finish",
            "mutation_control": _mutation_control(
                "one scored argv passed to the validator with -G and its value removed",
                not _scored_invocation_valid(mutant_row),
            ),
        },
        {
            "id": "FG-4",
            "held": deterministic,
            "enforcement": "runtime",
            "evaluated": "every native scored cell ran at least twice with byte-identical stdout",
            "mutation_control": _mutation_control(
                "one real captured stdout passed to the comparator with a perturbed copy",
                not _stdout_byte_identical(
                    deterministic_reference, deterministic_mutant
                ),
            ),
        },
        {
            "id": "FG-5",
            "held": e5_holds,
            "enforcement": "runtime",
            "evaluated": "the frozen separated-domain inequality holds in both E5 cells",
            "mutation_control": _mutation_control(
                "an E5 cell outside the frozen separated domain",
                not _fg5_holds(
                    CellSpec(
                        "mutant", "E5", "mutant", 0, 0, 80, 7, "3.0", 0, 50
                    ),
                    17,
                ),
            ),
        },
        {
            "id": "FG-6",
            "held": chronology,
            "enforcement": "runtime",
            "evaluated": "the frozen expectations commit is an ancestor of the implementation commit",
            "mutation_control": _mutation_control(
                "implementation and expectations commits reversed",
                not _is_ancestor(implementation_hash, expectations_hash),
            ),
        },
    ]


def _family_tallies(exact_rows: list[dict[str, Any]]) -> dict[str, Any]:
    result = {}
    for family in ("E1", "E2", "E3", "E4", "E5", "E6"):
        rows = [row for row in exact_rows if row["family"] == family]
        result[family] = {
            "evidence_class": "exact-oracle",
            "denominator": sum(row["score_denominator"] for row in rows),
            "passed": sum(row["score_passed"] for row in rows),
        }
    return result


def _csv_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for row in result["exact_oracles"]["rows"]:
        expected_hosts = row["expected_host_finish_ns"]
        if expected_hosts:
            for host, expected in sorted(expected_hosts.items()):
                observed = row["observed_host_finish_ns"][host]
                rows.append(
                    {
                        "evidence_class": "exact-oracle",
                        "family": row["family"],
                        "id": f"{row['id']}-host-{host}",
                        "expected": expected,
                        "observed": observed,
                        "units": "ns",
                        "passed": str(observed == expected).lower(),
                    }
                )
        else:
            rows.append(
                {
                    "evidence_class": "exact-oracle",
                    "family": row["family"],
                    "id": row["id"],
                    "expected": row["expected_max_finish_ns"],
                    "observed": row["observed_max_finish_ns"],
                    "units": "ns",
                    "passed": str(row["passed"]).lower(),
                }
            )
    for row in result["live_chain"]["rows"]:
        rows.append(
            {
                "evidence_class": "live-identity",
                "family": "L1",
                "id": row["id"],
                "expected": json.dumps(row.get("expected", row.get("expected_delta_ps", row.get("expected_network_delta_ps"))), sort_keys=True),
                "observed": json.dumps(row.get("observed", row.get("observed_delta_ps", row.get("observed_ttft_delta_ps"))), sort_keys=True),
                "units": "identity",
                "passed": str(row["passed"]).lower(),
            }
        )
    for row in result["wall_time"]["rows"]:
        rows.append(
            {
                "evidence_class": "wall-time",
                "family": "W",
                "id": row["id"],
                "expected": row["ceiling_seconds"],
                "observed": f"{row['median_seconds']:.9f}",
                "units": "s",
                "passed": str(row["passed"]).lower(),
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("evidence_class", "family", "id", "expected", "observed", "units", "passed"),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def _outside_repository(path: Path) -> bool:
    try:
        path.relative_to(REPO_ROOT)
    except ValueError:
        return True
    return False


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _begin_attempt(run_root: Path, args: argparse.Namespace) -> Path:
    run_root = run_root.resolve()
    if not _outside_repository(run_root):
        raise SystemExit("--run-dir must be outside the repository")
    run_root.mkdir(parents=True, exist_ok=True)
    attempts: list[tuple[int, Path]] = []
    for path in run_root.glob("attempt-*"):
        match = re.fullmatch(r"attempt-(\d+)", path.name)
        if match is not None and path.is_dir():
            attempts.append((int(match.group(1)), path))
    incomplete = [path for _, path in attempts if not (path / "verdict.json").is_file()]
    if incomplete:
        names = ", ".join(path.name for path in sorted(incomplete))
        raise SystemExit(
            f"cannot start a later attempt while verdict records are missing: {names}"
        )
    number = max((number for number, _ in attempts), default=0) + 1
    attempt_dir = run_root / f"attempt-{number}"
    attempt_dir.mkdir(parents=False, exist_ok=False)
    _write_json(
        attempt_dir / "attempt.json",
        {
            "binary": str(args.binary.resolve()),
            "results_csv": None if args.results_csv is None else str(args.results_csv),
            "results_json": None if args.results_json is None else str(args.results_json),
            "run_root": str(run_root),
            "started_unix_time_ns": time.time_ns(),
            "txt2bin": str(args.txt2bin.resolve()),
        },
    )
    return attempt_dir


def run_study(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = args.run_dir.resolve()
    if not _outside_repository(run_dir):
        raise SystemExit("--run-dir must be outside the repository")
    run_dir.mkdir(parents=True, exist_ok=True)
    binary = args.binary.resolve()
    txt2bin = args.txt2bin.resolve()

    binary_hash = _sha256_path(binary)
    if not _binary_hash_matches(binary_hash):
        mutant_binary_hash = _sha256_bytes(binary.read_bytes() + b"\0")
        result = {
            "schema": "simllm-loggopsim-ideal-study-v1",
            "verdict": "VOID",
            "findings": ["FG-1 binary hash mismatch before any cell ran"],
            "binary_sha256": binary_hash,
            "fatal_guards": [
                {
                    "id": "FG-1",
                    "held": False,
                    "enforcement": "runtime",
                    "evaluated": "binary SHA-256 equals the frozen literal before any cell",
                    "mutation_control": _mutation_control(
                        "one zero byte appended to the actual binary bytes",
                        not _binary_hash_matches(mutant_binary_hash),
                    ),
                }
            ],
        }
        return result

    goal_bins, goal_hashes = _prepare_goals(run_dir, txt2bin)
    recorder = NativeEvidenceRecorder(run_dir)
    exact_results = [
        _run_exact_cell(spec, binary, goal_bins[spec.goal], run_dir, recorder)
        for spec in _exact_specs()
    ]
    exact_rows = [row for row, _ in exact_results]
    stdout_pairs = [pair for _, pair in exact_results]
    live, live_pair = _run_live_family(binary, txt2bin, run_dir, recorder)
    stdout_pairs.append(live_pair)
    wall, wall_pairs = _run_wall_family(binary, goal_bins, run_dir, recorder)
    stdout_pairs.extend(wall_pairs)
    expectations_hash = _git_hash(EXPECTATIONS_COMMIT)
    implementation_hash = _git_hash("HEAD")
    guards = _fatal_guards(
        binary_hash=binary_hash,
        goal_hashes=goal_hashes,
        exact_rows=exact_rows,
        live=live,
        wall=wall,
        stdout_pairs=stdout_pairs,
        expectations_hash=expectations_hash,
        implementation_hash=implementation_hash,
    )
    void = not all(guard["held"] for guard in guards)
    exact_denominator = sum(row["score_denominator"] for row in exact_rows)
    exact_passed = sum(row["score_passed"] for row in exact_rows)
    alltoall_floor_ns = 1_321_173
    alltoall_ceiling_ns = 1_730_673
    alltoall_observed_ns = next(
        row["observed_max_finish_ns"]
        for row in exact_rows
        if row["id"] == "E6-alltoall"
    )
    result = {
        "schema": "simllm-loggopsim-ideal-study-v1",
        "verdict": "VOID" if void else "PASS",
        "findings": [guard["id"] for guard in guards if not guard["held"]],
        "chronology": {
            "expectations_commit": expectations_hash,
            "implementation_commit": implementation_hash,
        },
        "binary": {
            "sha256": binary_hash,
            "expected_sha256": PINNED_BINARY_SHA256,
        },
        "goal_inputs": goal_hashes,
        "run_history": {
            "prior_void_guard": "FG-4",
            "prior_void_stdout_bytes_retained": False,
            "prior_void_stdout_hashes_only": True,
            "legacy_attempts_without_own_verdict": 5,
            "finding": (
                "the chain's integer-second performance banner printed 0 s and "
                "1 s across subsecond repetitions while maximum finish stayed exact"
            ),
            "correction": (
                "the prior stdout-retention claim is withdrawn; the earlier run "
                "stays void and is not rescored"
            ),
        },
        "attempt_evidence": {
            "full_resolved_argv": "native/*/execution-*.json",
            "stdout_bytes": "native/*/execution-*.stdout",
            "verdict": "verdict.json",
            "policy": (
                "each run uses a fresh attempt-N directory and a later attempt is "
                "refused until every earlier attempt has its own verdict"
            ),
        },
        "determinism_protocol": {
            "native_executions_per_exact_cell": 2,
            "native_executions_per_live_identity": 2,
            "wall_samples_per_cell": 7,
            "additional_wall_determinism_executions_per_cell": 2,
            "shared_start_boundary": "within 20 ms after wall-second rollover",
        },
        "physical_sanity": {
            "alltoall_serialization_floor_ns": alltoall_floor_ns,
            "alltoall_serial_ceiling_ns": alltoall_ceiling_ns,
            "observed_alltoall_ns": alltoall_observed_ns,
            "inside_bounds": (
                alltoall_floor_ns <= alltoall_observed_ns <= alltoall_ceiling_ns
            ),
            "bandwidth_scaling_check": {
                "400G_G_ns_per_byte": "0.02",
                "200G_G_ns_per_byte": "0.04",
                "ratio": 2,
            },
            "end_to_end_plausibility": (
                "the 64-rank 1 MiB pairwise schedule is 1.409 ms, above its "
                "1.321 ms serialization floor and below its 1.731 ms serial ceiling"
            ),
        },
        "exact_oracles": {
            "class": "exact-oracle",
            "denominator": exact_denominator,
            "passed": exact_passed,
            "families": _family_tallies(exact_rows),
            "rows": exact_rows,
        },
        "live_chain": live,
        "wall_time": wall,
        "fatal_guards": guards,
        "evidence_separation": (
            "exact-oracle, live-identity and wall-time denominators are reported "
            "separately; fatal guards are unscored and void the run on violation"
        ),
    }
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--binary",
        type=Path,
        default=os.environ.get("SIMLLM_LOGGOPSIM"),
        required="SIMLLM_LOGGOPSIM" not in os.environ,
    )
    parser.add_argument(
        "--txt2bin",
        type=Path,
        default=os.environ.get("SIMLLM_TXT2BIN"),
        required="SIMLLM_TXT2BIN" not in os.environ,
    )
    parser.add_argument("--results-json", type=Path)
    parser.add_argument("--results-csv", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    attempt_dir = _begin_attempt(args.run_dir, args)
    attempt_args = argparse.Namespace(**vars(args))
    attempt_args.run_dir = attempt_dir
    verdict_path = attempt_dir / "verdict.json"
    try:
        result = run_study(attempt_args)
    except BaseException as exc:
        _write_json(
            verdict_path,
            {
                "error": str(exc),
                "error_type": type(exc).__name__,
                "schema": "simllm-loggopsim-ideal-attempt-v1",
                "verdict": "ERROR",
            },
        )
        raise
    _write_json(verdict_path, result)
    json_path = args.results_json
    if json_path is not None and json_path.resolve() != verdict_path.resolve():
        _write_json(json_path, result)
    csv_path = args.results_csv or attempt_dir / "results.csv"
    if result["verdict"] != "VOID" or "exact_oracles" in result:
        _write_csv(csv_path, _csv_rows(result))
    sys.stdout.write(f"verdict={result['verdict']} results={verdict_path}\n")
    return 1 if result["verdict"] == "VOID" else 0


if __name__ == "__main__":
    raise SystemExit(main())
