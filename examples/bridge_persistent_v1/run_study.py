"""Run the frozen BRIDGE-1 prepared-replay wall-clock study."""

from __future__ import annotations

import argparse
import dataclasses
import enum
import hashlib
import json
import os
import platform
import subprocess
import time
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

from simllm.backends import (
    HtsimPersistentStepSink,
    HtsimStepSink,
    HtsimStepSinkConfig,
    StepNetworkOutcome,
)
from simllm.compute import ModelDims
from simllm.core import StepRecord, StepResult, step_records_from_jsonl
from simllm.placement import declared_manifest

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
EXPECTATIONS_COMMIT = "aa6e92f882e4a2091493ebee68e117547fe60d53"
HTSIM_COMMIT = "edb28c3015c173b4251abc5858c587df325e1ebc"
FIXTURES = {
    "vllm": (
        REPO_ROOT / "examples/m4/fixtures/vllm-m2-steps.jsonl",
        8,
        "a226fcc17908844ba080587fe6607c5c8f34b178d17111fbd384819731b26fb7",
    ),
    "sglang": (
        REPO_ROOT / "examples/m4/fixtures/sglang-m3-steps.jsonl",
        9,
        "656772148cd8fbda71a25af08215d806f38f3886abb068f72c9e0ddc8cb7c26f",
    ),
}
WORKERS = (4, 8)
WALL_BANDS_S = {
    ("vllm", "diagnostic"): (45.0, 85.0),
    ("vllm", 4): (10.0, 45.0),
    ("vllm", 8): (6.0, 35.0),
    ("sglang", "diagnostic"): (50.0, 100.0),
    ("sglang", 4): (12.0, 50.0),
    ("sglang", 8): (7.0, 38.0),
}
MIN_SPEEDUP = {4: 1.5, 8: 2.0}


@dataclass(frozen=True)
class CellResult:
    fixture: str
    mode: str
    workers: int | None
    elapsed_ns: int
    results: tuple[StepResult, ...]
    outcomes: tuple[StepNetworkOutcome, ...]
    result_bytes: tuple[bytes, ...]
    outcome_bytes: tuple[bytes, ...]
    artifact_bytes: dict[str, tuple[bytes, ...]]


def _csv_names(value: str, supported: tuple[str, ...], option: str) -> tuple[str, ...]:
    names = tuple(value.split(","))
    if not names or any(name not in supported for name in names):
        raise ValueError(f"{option} must contain only {supported}")
    if len(names) != len(set(names)):
        raise ValueError(f"{option} entries must be unique")
    return names


def _worker_values(value: str) -> tuple[int, ...]:
    try:
        workers = tuple(int(item) for item in value.split(","))
    except ValueError as exc:
        raise ValueError("--workers must contain integers") from exc
    if workers != WORKERS:
        raise ValueError(f"--workers must be exactly {WORKERS}")
    return workers


def _configured_executable(variable: str) -> Path:
    raw = os.environ.get(variable)
    if not raw:
        raise ValueError(f"{variable} must name the configured pinned executable")
    path = Path(raw)
    if not path.is_file() or not os.access(path, os.X_OK):
        raise ValueError(f"{variable} is not an executable file: {path}")
    return path


def _fixture_plan(names: tuple[str, ...]) -> list[dict[str, object]]:
    rows = []
    for name in names:
        path, records, expected_sha256 = FIXTURES[name]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected_sha256:
            raise ValueError(f"{path}: SHA-256 {digest}, expected {expected_sha256}")
        line_count = sum(bool(line.strip()) for line in path.read_text().splitlines())
        if line_count != records:
            raise ValueError(f"{path}: {line_count} records, expected {records}")
        rows.append(
            {
                "fixture": name,
                "path": str(path.relative_to(REPO_ROOT)),
                "records": records,
                "sha256": digest,
            }
        )
    return rows


def _dims_tp8() -> ModelDims:
    return ModelDims(
        num_layers=32,
        hidden_size=4096,
        intermediate_size=14336 // 8,
        num_heads=32 // 8,
        num_kv_heads=1,
        head_size=128,
        vocab_size=128256,
        dtype_bytes=2,
    )


def _sink_config(workdir: Path) -> HtsimStepSinkConfig:
    manifest = declared_manifest(tp=8, pp=1, dp=1)
    return HtsimStepSinkConfig(
        profile="rnic-nn-fluid",
        tp_ranks=manifest.group_ranks(0, "tp"),
        dims=_dims_tp8(),
        workdir=workdir,
        linkspeed_bps=400_000_000_000,
    )


def _json_ready(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _json_ready(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
    if isinstance(value, enum.Enum):
        return value.value
    if isinstance(value, Fraction):
        return {"numerator": value.numerator, "denominator": value.denominator}
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        _json_ready(value),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()


def _read_artifacts(workdir: Path, records: tuple[StepRecord, ...]) -> dict[str, tuple[bytes, ...]]:
    paths = {
        "goal_text": tuple(
            workdir / f"step-{record.step_index:06d}.goal" for record in records
        ),
        "goal_binary": tuple(
            workdir / f"step-{record.step_index:06d}.bin" for record in records
        ),
        "completion_csv": tuple(
            workdir / f"step-{record.step_index:06d}.rnic-nn-fluid.csv"
            for record in records
        ),
    }
    return {
        artifact: tuple(path.read_bytes() for path in artifact_paths)
        for artifact, artifact_paths in paths.items()
    }


def _run_cell(
    fixture: str,
    records: tuple[StepRecord, ...],
    out: Path,
    workers: int | None,
) -> CellResult:
    mode = "diagnostic" if workers is None else f"prepared-{workers}"
    workdir = out / fixture / mode
    if workdir.exists():
        raise FileExistsError(f"study cell already exists: {workdir}")
    config = _sink_config(workdir)

    started_ns = time.perf_counter_ns()
    if workers is None:
        sink: HtsimStepSink = HtsimStepSink(config)
        raw_results = tuple(sink(record) for record in records)
    else:
        with HtsimPersistentStepSink(config, max_workers=workers) as persistent:
            persistent.prepare(records)
            raw_results = tuple(persistent(record) for record in records)
        sink = persistent
    elapsed_ns = time.perf_counter_ns() - started_ns

    if any(result is None for result in raw_results):
        raise AssertionError(f"{fixture}/{mode}: every recorded step must return a result")
    results = tuple(result for result in raw_results if result is not None)
    if len(results) != len(records) or len(sink.outcomes) != len(records):
        raise AssertionError(f"{fixture}/{mode}: result or outcome count mismatch")
    cell = CellResult(
        fixture=fixture,
        mode=mode,
        workers=workers,
        elapsed_ns=elapsed_ns,
        results=results,
        outcomes=tuple(sink.outcomes),
        result_bytes=tuple(_canonical_bytes(result) for result in results),
        outcome_bytes=tuple(_canonical_bytes(outcome) for outcome in sink.outcomes),
        artifact_bytes=_read_artifacts(workdir, records),
    )
    print(
        f"fixture={fixture} mode={mode} elapsed_s={elapsed_ns / 1e9:.6f} "
        f"steps={len(results)}"
    )
    return cell


def _stream_digest(items: tuple[bytes, ...]) -> str:
    digest = hashlib.sha256()
    for item in items:
        digest.update(len(item).to_bytes(8, "big"))
        digest.update(item)
    return digest.hexdigest()


def _cell_summary(cell: CellResult) -> dict[str, Any]:
    return {
        "fixture": cell.fixture,
        "mode": cell.mode,
        "workers": cell.workers,
        "elapsed_ns": cell.elapsed_ns,
        "elapsed_s": cell.elapsed_ns / 1e9,
        "step_count": len(cell.results),
        "step_latencies_ps": [result.step_latency_ps for result in cell.results],
        "all_quiescent": all(outcome.quiescent for outcome in cell.outcomes),
        "result_stream_sha256": _stream_digest(cell.result_bytes),
        "outcome_stream_sha256": _stream_digest(cell.outcome_bytes),
        "artifact_stream_sha256": {
            artifact: _stream_digest(values)
            for artifact, values in cell.artifact_bytes.items()
        },
    }


def _git_revision(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args],
        cwd=REPO_ROOT,
        text=True,
    ).strip()


def _check_only_plan(
    args: argparse.Namespace,
    fixtures: tuple[str, ...],
    workers: tuple[int, ...],
    binaries: dict[str, Path],
) -> None:
    plan = {
        "artifacts_created": False,
        "binaries": {name: str(path) for name, path in binaries.items()},
        "expectations_commit": EXPECTATIONS_COMMIT,
        "fixtures": _fixture_plan(fixtures),
        "matrix": [
            {
                "fixture": fixture,
                "modes": ["diagnostic", *(f"prepared-{worker}" for worker in workers)],
            }
            for fixture in fixtures
        ],
        "minimum_speedup": MIN_SPEEDUP,
        "out": str(args.out),
        "wall_bands_s": {str(key): value for key, value in WALL_BANDS_S.items()},
    }
    print(json.dumps(plan, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fixtures", default="vllm,sglang")
    parser.add_argument("--workers", default="4,8")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    fixtures = _csv_names(args.fixtures, tuple(FIXTURES), "--fixtures")
    workers = _worker_values(args.workers)
    binaries = {
        variable: _configured_executable(variable)
        for variable in ("SIMLLM_HTSIM_RNIC", "SIMLLM_TXT2BIN")
    }
    fixture_plan = _fixture_plan(fixtures)
    if args.check_only:
        _check_only_plan(args, fixtures, workers, binaries)
        return
    if args.out.exists():
        parser.error("--out must not exist; choose a fresh external study directory")
    args.out.mkdir(parents=True)

    cells: dict[tuple[str, int | None], CellResult] = {}
    for fixture in fixtures:
        path, _, _ = FIXTURES[fixture]
        records = tuple(step_records_from_jsonl(path))
        cells[(fixture, None)] = _run_cell(fixture, records, args.out, None)
        for worker_count in workers:
            cells[(fixture, worker_count)] = _run_cell(
                fixture,
                records,
                args.out,
                worker_count,
            )

    identity = {
        evidence: {"passed": 0, "total": 0}
        for evidence in (
            "step_result",
            "step_outcome",
            "latency_stream",
            "goal_text",
            "goal_binary",
            "completion_csv",
        )
    }
    fatal_failures = []
    for fixture in fixtures:
        diagnostic = cells[(fixture, None)]
        diagnostic_latencies = _canonical_bytes(
            tuple(result.step_latency_ps for result in diagnostic.results)
        )
        for worker_count in workers:
            prepared = cells[(fixture, worker_count)]
            comparisons = {
                "step_result": tuple(
                    left == right
                    for left, right in zip(
                        diagnostic.result_bytes,
                        prepared.result_bytes,
                        strict=True,
                    )
                ),
                "step_outcome": tuple(
                    left == right
                    for left, right in zip(
                        diagnostic.outcome_bytes,
                        prepared.outcome_bytes,
                        strict=True,
                    )
                ),
                "latency_stream": (
                    diagnostic_latencies
                    == _canonical_bytes(
                        tuple(result.step_latency_ps for result in prepared.results)
                    ),
                ),
                **{
                    artifact: tuple(
                        left == right
                        for left, right in zip(
                            diagnostic.artifact_bytes[artifact],
                            prepared.artifact_bytes[artifact],
                            strict=True,
                        )
                    )
                    for artifact in ("goal_text", "goal_binary", "completion_csv")
                },
            }
            for evidence, matches in comparisons.items():
                identity[evidence]["passed"] += sum(matches)
                identity[evidence]["total"] += len(matches)
                if not all(matches):
                    fatal_failures.append(
                        f"{fixture}/prepared-{worker_count}: {evidence} identity"
                    )
    quiescence_passed = sum(
        all(outcome.quiescent for outcome in cell.outcomes) for cell in cells.values()
    )
    if quiescence_passed != len(cells):
        fatal_failures.append("one or more cells did not report physical quiescence")

    scored = []
    for fixture in fixtures:
        diagnostic_s = cells[(fixture, None)].elapsed_ns / 1e9
        for worker_count in workers:
            prepared_s = cells[(fixture, worker_count)].elapsed_ns / 1e9
            diagnostic_band = WALL_BANDS_S[(fixture, "diagnostic")]
            prepared_band = WALL_BANDS_S[(fixture, worker_count)]
            speedup = diagnostic_s / prepared_s
            checks = {
                "diagnostic_in_band": diagnostic_band[0]
                <= diagnostic_s
                <= diagnostic_band[1],
                "prepared_in_band": prepared_band[0]
                <= prepared_s
                <= prepared_band[1],
                "speedup_at_least_bound": speedup >= MIN_SPEEDUP[worker_count],
            }
            scored.append(
                {
                    "fixture": fixture,
                    "workers": worker_count,
                    "diagnostic_s": diagnostic_s,
                    "prepared_s": prepared_s,
                    "diagnostic_band_s": diagnostic_band,
                    "prepared_band_s": prepared_band,
                    "speedup": speedup,
                    "minimum_speedup": MIN_SPEEDUP[worker_count],
                    "checks": checks,
                    "passed": all(checks.values()),
                    "genuine_risk": True,
                }
            )

    summary = {
        "schema": "simllm-bridge-persistent-study-v1",
        "chronology": {
            "expectations_commit": EXPECTATIONS_COMMIT,
            "implementation_commit": _git_revision("rev-parse", "HEAD"),
        },
        "pinned_htsim_commit": HTSIM_COMMIT,
        "binary_sha256": {
            name: hashlib.sha256(path.read_bytes()).hexdigest()
            for name, path in binaries.items()
        },
        "host": {
            "logical_cpus": os.cpu_count(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "system": platform.system(),
        },
        "fixtures": fixture_plan,
        "cells": [_cell_summary(cell) for cell in cells.values()],
        "fatal_unscored": {
            "identity": identity,
            "quiescence": {"passed": quiescence_passed, "total": len(cells)},
            "failures": fatal_failures,
            "passed": not fatal_failures,
        },
        "scored_relation_family": {
            "name": "R1 live wall-clock relation",
            "instances": scored,
            "passed": sum(row["passed"] for row in scored),
            "total": len(scored),
            "genuine_risk_passed": sum(
                row["passed"] for row in scored if row["genuine_risk"]
            ),
            "genuine_risk_total": sum(row["genuine_risk"] for row in scored),
        },
    }
    summary_path = args.out / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"summary={summary_path}")
    print(
        "fatal_identity="
        f"{'PASS' if not fatal_failures else 'FAIL'} "
        f"scored={summary['scored_relation_family']['passed']}/{len(scored)}"
    )
    if fatal_failures:
        raise AssertionError("; ".join(fatal_failures))
    failed_scored = [row for row in scored if not row["passed"]]
    if failed_scored:
        raise AssertionError(f"scored relation failures: {failed_scored}")


if __name__ == "__main__":
    main()
