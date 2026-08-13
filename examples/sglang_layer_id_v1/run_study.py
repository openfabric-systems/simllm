"""SGL-16 study: SGLang-supplied dispatch layer identity.

Runs the frozen cells of `expectations.md` twice, once with the model-order
surrogate that the registered clause names and once with SGLang's own explicit
layer identity, then scores the frozen relations against the two artifact
sets. The engine work happens in a child interpreter that owns SGLang and
torch; this module never imports either.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

SGLANG_AUTHORED_AGAINST = "8f2a3ad6d7d68c58ae65b61a75bb2115449addca"
MODEL_ID = "ibm-granite/granite-3.0-1b-a400m-instruct"
MODEL_REVISION = "ffec3c35bdfd97a06f0b4cd5fcc92cd9b1584445"
MODEL_DTYPE = "float32"
ENGINE_SEED = 173
TORCH_NUM_THREADS = 8
MOE_LAYER_COUNT = 24

PHASES = ("baseline", "treatment")
EXPECTED_SCORED = 3 + 3 + 3
ROTATION_TOKEN_FRACTION = 0.90

SGLANG_SOURCE_HASHES = {
    "python/sglang/srt/layers/moe/topk.py": (
        "45f61df269e2b17f411cf6a2bede0188cd980a2c5f110bafc653c170f2e639c7"
    ),
    "python/sglang/srt/model_executor/model_runner.py": (
        "2a06cabe589c2f0f33e6cbfc97a84a50b2d4aca1c44aa51d4f04063631a222c6"
    ),
    "python/sglang/srt/models/granitemoe.py": (
        "f1a465f47fa472801ba0c8cefb9227254ea6309273e5ba6394054d329ad03910"
    ),
    "python/sglang/srt/plugins/hook_registry.py": (
        "fe214acac324cb2f019aa1dffe1ddcb6e4691862fdd1b6a0ad5033d8167e31e9"
    ),
    "python/sglang/srt/state_capturer/base.py": (
        "2260da69a1f9de5e35a7784142ab32f6a26d5df41a9b1c0784fc1a73e1f77e9f"
    ),
    "python/sglang/srt/state_capturer/routed_experts.py": (
        "49ef692f7d468f297d9aeef5a30657908d4a353bd8a229be8ebc40266b64c669"
    ),
}


def _short_prompt(index: int) -> tuple[int, ...]:
    del index
    return tuple(100 * (step + 1) for step in range(8))


def _long_prompt(index: int) -> tuple[int, ...]:
    del index
    return tuple(1500 + step for step in range(96))


def _pressure_prompt(index: int) -> tuple[int, ...]:
    return tuple(1000 + 100 * index + step for step in range(8))


CELLS: dict[str, dict[str, Any]] = {
    "short": {
        "request_count": 1,
        "request_prefix": "s",
        "prompt": _short_prompt,
        "max_new_tokens": 4,
        "max_total_tokens": 256,
        "context_length": 512,
        "max_running_requests": 2,
        "expect_preemption": False,
    },
    "long": {
        "request_count": 1,
        "request_prefix": "l",
        "prompt": _long_prompt,
        "max_new_tokens": 8,
        "max_total_tokens": 256,
        "context_length": 512,
        "max_running_requests": 2,
        "expect_preemption": False,
    },
    "preempt": {
        "request_count": 8,
        "request_prefix": "p",
        "prompt": _pressure_prompt,
        "max_new_tokens": 20,
        "max_total_tokens": 96,
        "context_length": 512,
        "max_running_requests": 8,
        "expect_preemption": True,
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_hashes(root: Path) -> None:
    for relative, digest in SGLANG_SOURCE_HASHES.items():
        source = root / relative
        if not source.is_file():
            raise SystemExit(f"SGLang source is missing: {relative}")
        observed = _sha256(source)
        if observed != digest:
            raise SystemExit(
                f"SGLang source changed: {relative}; "
                f"expected {digest}, observed {observed}"
            )


def _git_head(root: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _python_version(executable: Path) -> tuple[int, ...]:
    result = subprocess.run(
        [
            str(executable),
            "-c",
            "import sys; print('.'.join(map(str, sys.version_info[:3])))",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return tuple(int(part) for part in result.stdout.strip().split("."))


def _cell_requests(name: str) -> list[dict[str, Any]]:
    cell = CELLS[name]
    prompt = cell["prompt"]
    return [
        {
            "request_id": f"{cell['request_prefix']}{index}",
            "input_token_ids": list(prompt(index)),
            "max_new_tokens": cell["max_new_tokens"],
        }
        for index in range(cell["request_count"])
    ]


# ---------------------------------------------------------------- child side


def _run_child(args: argparse.Namespace) -> None:
    sys.path.insert(0, str(REPOSITORY_ROOT))
    from simllm.preplay import (
        FrameworkOracleRequest,
        PromptFormat,
        SglangCpuRunner,
    )

    name = args.child_cell
    cell = CELLS[name]
    cell_dir = args.run_dir / args.phase / name
    cell_dir.mkdir(parents=True, exist_ok=True)
    runner_kwargs: dict[str, Any] = {}
    if args.layer_audit:
        runner_kwargs["layer_audit"] = True
    runner = SglangCpuRunner(
        model_id=MODEL_ID,
        model_revision=MODEL_REVISION,
        model_path=args.model_path,
        tokenizer_sha256=_sha256(args.model_path / "tokenizer.json"),
        observed_source=_git_head(args.sglang_source),
        authored_against_source=SGLANG_AUTHORED_AGAINST,
        dtype=MODEL_DTYPE,
        torch_num_threads=TORCH_NUM_THREADS,
        engine_seed=ENGINE_SEED,
        **runner_kwargs,
    )
    requests = [
        FrameworkOracleRequest(
            request_id=row["request_id"],
            prompt_sha256=hashlib.sha256(
                json.dumps(row["input_token_ids"]).encode("utf-8")
            ).hexdigest(),
            prompt_format=PromptFormat.TEXT,
            input_token_ids=tuple(row["input_token_ids"]),
            max_new_tokens=row["max_new_tokens"],
        )
        for row in _cell_requests(name)
    ]
    runner.capture(
        requests,
        cell_dir / "trace.jsonl",
        max_total_tokens=cell["max_total_tokens"],
        context_length=cell["context_length"],
        max_running_requests=cell["max_running_requests"],
        page_size=1,
        overwrite=True,
        observation_path=cell_dir / "observations.jsonl",
        raw_response_path=cell_dir / "responses.json",
    )


# --------------------------------------------------------------- parent side


def _launch_cell(args: argparse.Namespace, name: str) -> None:
    command = [
        str(args.sglang_python),
        str(Path(__file__).resolve()),
        "--run-dir",
        str(args.run_dir),
        "--sglang-source",
        str(args.sglang_source),
        "--sglang-python",
        str(args.sglang_python),
        "--model-path",
        str(args.model_path),
        "--phase",
        args.phase,
        "--child-cell",
        name,
    ]
    if args.layer_audit:
        command.append("--layer-audit")
    log_path = args.run_dir / args.phase / f"{name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("wb") as handle:
        subprocess.run(command, check=True, stdout=handle, stderr=subprocess.STDOUT)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _rows_of_type(trace: list[dict[str, Any]], row_type: str) -> list[dict[str, Any]]:
    return [row for row in trace if row.get("row_type") == row_type]


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _observation_rows(cell_dir: Path, kind: str) -> list[dict[str, Any]]:
    return [
        row
        for row in _read_jsonl(cell_dir / "observations.jsonl")
        if row.get("kind") == kind
    ]


def _rotation_changes(trace: list[dict[str, Any]]) -> tuple[int, int]:
    """Tokens whose layer-to-expert map changes under a one-layer rotation."""

    changed = 0
    total = 0
    for row in _rows_of_type(trace, "observed-dispatch"):
        layers = row["routing"]
        experts = [tuple(entry["expert_ids"]) for entry in layers]
        rotated = experts[-1:] + experts[:-1]
        total += 1
        if experts != rotated:
            changed += 1
    return changed, total


def _guard(results: list[dict[str, Any]], name: str, ok: bool, detail: str) -> None:
    results.append({"guard": name, "passed": bool(ok), "detail": detail})


def _score(results: list[dict[str, Any]], family: str, cell: str, ok: bool, detail: str) -> None:
    results.append(
        {"family": family, "cell": cell, "passed": bool(ok), "detail": detail}
    )


def compare(args: argparse.Namespace) -> int:
    guards: list[dict[str, Any]] = []
    scored: list[dict[str, Any]] = []
    for name in CELLS:
        directories = {
            phase: args.run_dir / phase / name for phase in PHASES
        }
        traces = {
            phase: _read_jsonl(directory / "trace.jsonl")
            for phase, directory in directories.items()
        }
        responses = {
            phase: (directory / "responses.json").read_bytes()
            for phase, directory in directories.items()
        }

        for phase, directory in directories.items():
            worker_rows = _observation_rows(directory, "worker-qualified")
            ok = bool(worker_rows) and all(
                row.get("worker_class") == "TpModelWorker"
                and row.get("model_runner_class") == "ModelRunner"
                and row.get("model_class") == "GraniteMoeForCausalLM"
                and row.get("parameter_devices") == ["cpu"]
                for row in worker_rows
            )
            _guard(guards, f"G1[{name}/{phase}]", ok, f"{len(worker_rows)} rows")
            storage_rows = _observation_rows(directory, "capture-storage-qualified")
            ok = bool(storage_rows) and all(
                row.get("device") == "cpu" and row.get("pinned") is False
                for row in storage_rows
            )
            _guard(guards, f"G2[{name}/{phase}]", ok, f"{len(storage_rows)} rows")

        headers = {
            phase: _rows_of_type(trace, "header")[0]["provenance"]
            for phase, trace in traces.items()
        }
        _guard(
            guards,
            f"G3[{name}]",
            headers["baseline"]["dispatch_layer_mapping"] == "granite-model-order"
            and headers["treatment"]["dispatch_layer_mapping"] == "framework-layer-id",
            _canonical(
                {
                    phase: header["dispatch_layer_mapping"]
                    for phase, header in headers.items()
                }
            ),
        )

        layer_rows = _observation_rows(directories["treatment"], "dispatch-layer-qualified")
        expected_ids = list(range(MOE_LAYER_COUNT))
        ok = bool(layer_rows) and all(
            row.get("mapping") == "framework-layer-id"
            and row.get("selected_experts_unchanged") is True
            and row.get("layer_ids") == expected_ids
            for row in layer_rows
        )
        _guard(
            guards,
            f"G4[{name}]",
            ok,
            _canonical([row.get("layer_ids") for row in layer_rows]),
        )

        if CELLS[name]["expect_preemption"]:
            for phase, trace in traces.items():
                kv_rows = _rows_of_type(trace, "kv-event")
                preemptions = [row for row in kv_rows if row.get("kind") == "preemption"]
                request_rows = {
                    row["request_id"]: row for row in _rows_of_type(trace, "request")
                }
                resumed = [
                    row["request_id"]
                    for row in preemptions
                    if request_rows.get(row["request_id"], {}).get(
                        "framework_preemption_count"
                    )
                    == 1
                    and request_rows.get(row["request_id"], {}).get("stop_reason")
                    == "length-cap"
                ]
                _guard(
                    guards,
                    f"G5[{name}/{phase}]",
                    bool(preemptions) and len(resumed) == len(preemptions),
                    f"{len(preemptions)} preemptions, {len(resumed)} resumed",
                )

        audit_rows = _observation_rows(directories["treatment"], "dispatch-layer-agreement")
        disagreements = [row for row in audit_rows if row.get("agrees") is not True]
        _guard(
            guards,
            f"G6[{name}]",
            bool(audit_rows) and not disagreements,
            f"{len(audit_rows)} audited captures, {len(disagreements)} disagreements",
        )

        derived_equal = all(
            _canonical(_rows_of_type(traces["baseline"], row_type))
            == _canonical(_rows_of_type(traces["treatment"], row_type))
            for row_type in ("request", "observed-dispatch")
        )
        _guard(guards, f"G8[{name}]", derived_equal, "derived trace rows")

        _score(
            scored,
            "R1",
            name,
            responses["baseline"] == responses["treatment"],
            f"{len(responses['baseline'])} vs {len(responses['treatment'])} bytes",
        )
        _score(
            scored,
            "R2",
            name,
            _canonical(_rows_of_type(traces["baseline"], "kv-event"))
            == _canonical(_rows_of_type(traces["treatment"], "kv-event")),
            f"{len(_rows_of_type(traces['baseline'], 'kv-event'))} events",
        )
        changed, total = _rotation_changes(traces["treatment"])
        fraction = 0.0 if total == 0 else changed / total
        _score(
            scored,
            "R3",
            name,
            total > 0 and fraction > ROTATION_TOKEN_FRACTION,
            f"{changed}/{total} tokens change under a one-layer rotation",
        )

    passed = sum(1 for row in scored if row["passed"])
    fatal_failures = [row for row in guards if not row["passed"]]
    summary = {
        "study": "sglang_layer_id_v1",
        "sglang_authored_against_source": SGLANG_AUTHORED_AGAINST,
        "sglang_observed_source": _git_head(args.sglang_source),
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "scored_passed": passed,
        "scored_total": EXPECTED_SCORED,
        "fatal_guards": len(guards),
        "fatal_failures": len(fatal_failures),
        "status": "PASS"
        if passed == EXPECTED_SCORED and not fatal_failures
        else "FAIL",
        "guards": guards,
        "scored": scored,
    }
    output = args.run_dir / "summary.json"
    output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: summary[key] for key in (
        "scored_passed",
        "scored_total",
        "fatal_guards",
        "fatal_failures",
        "status",
    )}, sort_keys=True))
    for row in fatal_failures:
        print("FATAL", row["guard"], row["detail"])
    for row in scored:
        if not row["passed"]:
            print("FAILED", row["family"], row["cell"], row["detail"])
    return 0


def check_only(args: argparse.Namespace) -> None:
    if not args.run_dir.is_absolute():
        raise SystemExit("run directory must be an explicit absolute path")
    if args.run_dir.resolve() == REPOSITORY_ROOT:
        raise SystemExit("run directory must be outside the repository")
    _validate_hashes(args.sglang_source)
    if not args.sglang_python.is_file() or not os.access(args.sglang_python, os.X_OK):
        raise SystemExit(f"SGLang Python is missing or not executable: {args.sglang_python}")
    if _python_version(args.sglang_python) < (3, 10):
        raise SystemExit("SGLang Python is too old")
    if not (args.model_path / "config.json").is_file():
        raise SystemExit(f"model snapshot is missing: {args.model_path}")
    if re.fullmatch(r"[0-9a-f]{40}", SGLANG_AUTHORED_AGAINST) is None:
        raise AssertionError("SGLang authored-against provenance is malformed")
    if tuple(CELLS) != ("short", "long", "preempt"):
        raise AssertionError("frozen cell family changed")
    if EXPECTED_SCORED != 9:
        raise AssertionError("behavioral evidence denominator changed")
    if MOE_LAYER_COUNT != 24:
        raise AssertionError("frozen Granite MoE module count changed")
    prompt_lengths = {
        name: len(CELLS[name]["prompt"](0)) for name in CELLS
    }
    if prompt_lengths != {"short": 8, "long": 96, "preempt": 8}:
        raise AssertionError("frozen prompt shapes changed")
    if platform.machine().lower() not in {"x86_64", "amd64"}:
        raise SystemExit("the frozen CPU engine ladder is x86-64 only")
    print(
        json.dumps(
            {
                "check_only": True,
                "artifacts_written": 0,
                "cells": list(CELLS),
                "scored_total": EXPECTED_SCORED,
                "sglang_authored_against_source": SGLANG_AUTHORED_AGAINST,
                "sglang_observed_source": _git_head(args.sglang_source),
            },
            sort_keys=True,
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--sglang-source", type=Path, required=True)
    parser.add_argument("--sglang-python", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--phase", choices=(*PHASES, "compare"))
    parser.add_argument("--layer-audit", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--child-cell", choices=tuple(CELLS), help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.check_only:
        check_only(args)
        return 0
    if args.child_cell is not None:
        _run_child(args)
        return 0
    if args.phase is None:
        raise SystemExit("--phase is required unless --check-only is given")
    if args.phase == "compare":
        return compare(args)
    _validate_hashes(args.sglang_source)
    for name in CELLS:
        _launch_cell(args, name)
    print(json.dumps({"phase": args.phase, "cells": list(CELLS)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
