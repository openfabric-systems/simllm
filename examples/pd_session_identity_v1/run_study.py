"""Validate the complete native vLLM request comparison boundary."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from copy import deepcopy
from dataclasses import fields, is_dataclass
from enum import Enum
from fractions import Fraction
from pathlib import Path

from examples.pd_session_kernel_cycle_v1 import run_study as baseline

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
FREEZE_COMMIT = "ae0a9d7ed52e1f13be1faebb8676f2234ea1c211"
EXCLUDED = ("prefill_internal_request_id", "decode_internal_request_id")


def exact_json_bytes(value):
    """Sort JSON members without normalizing strings or collapsing types."""

    def validate(item):
        if type(item) is dict:
            if any(type(key) is not str for key in item):
                raise TypeError("comparison JSON requires string object keys")
            for child in item.values():
                validate(child)
        elif type(item) is list:
            for child in item:
                validate(child)
        elif type(item) not in (str, int, float, bool, type(None)):
            raise TypeError("comparison requires JSON values")

    validate(value)
    return json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("ascii")


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def state_snapshot(value):
    """Retain typed result state, including records absent from to_json()."""
    name = type(value).__module__ + "." + type(value).__qualname__
    if is_dataclass(value):
        return {
            "type": name,
            "fields": {
                field.name: state_snapshot(getattr(value, field.name)) for field in fields(value)
            },
        }
    if isinstance(value, Enum):
        return {"type": name, "value": state_snapshot(value.value)}
    if isinstance(value, Fraction):
        return {"type": name, "numerator": value.numerator, "denominator": value.denominator}
    if type(value) is dict:
        return {
            "type": name,
            "items": [[state_snapshot(key), state_snapshot(child)] for key, child in value.items()],
        }
    if type(value) in (tuple, list):
        return {"type": name, "items": [state_snapshot(child) for child in value]}
    if type(value) in (str, int, float, bool, type(None)):
        return {"type": name, "value": value}
    raise TypeError("unsupported native result state: " + name)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(exact_json_bytes(value))


def difference_paths(first, second, path=()):
    """Return exact type-aware JSON paths, including missing members."""
    if type(first) is not type(second):
        return [path]
    if type(first) is dict:
        differences = []
        for key in sorted(first.keys() | second.keys()):
            if key not in first or key not in second:
                differences.append((*path, key))
            else:
                differences.extend(difference_paths(first[key], second[key], (*path, key)))
        return differences
    if type(first) is list:
        if len(first) != len(second):
            return [path]
        return [
            found
            for index, (a, b) in enumerate(zip(first, second, strict=True))
            for found in difference_paths(a, b, (*path, index))
        ]
    return [] if exact_json_bytes(first) == exact_json_bytes(second) else [path]


def scalar_paths(value, path=()):
    if type(value) is dict:
        return [
            (p, v) for key, child in value.items() for p, v in scalar_paths(child, (*path, key))
        ]
    if type(value) is list:
        return [
            (p, v)
            for index, child in enumerate(value)
            for p, v in scalar_paths(child, (*path, index))
        ]
    return [(path, value)]


def project_serialized(value):
    """Exercise the production projection on a counterfactual serialized input."""
    from simllm.adapters.vllm.pd_session import VllmPdRequestResult

    class SerializedControl:
        def to_json(self):
            return value

    return VllmPdRequestResult.to_comparison_json(SerializedControl())


def corruption_controls(raw):
    projected = exact_json_bytes(project_serialized(raw))
    rows = []
    for path, value in scalar_paths(raw):
        if path in [(name,) for name in EXCLUDED]:
            continue
        changed = deepcopy(raw)
        parent = changed
        for key in path[:-1]:
            parent = parent[key]
        replacement = (
            (not value)
            if type(value) is bool
            else (
                value + 1
                if type(value) in (int, float)
                else value + ":changed"
                if type(value) is str
                else "changed-null"
            )
        )
        parent[path[-1]] = replacement
        rows.append(
            {
                "path": list(path),
                "discriminated": exact_json_bytes(project_serialized(changed)) != projected,
            }
        )
    for label, value in (
        ("future-member", {**raw, "future_metadata": "retained"}),
        ("nested-opaque-name", {**raw, "future_metadata": {EXCLUDED[0]: "retained"}}),
    ):
        rows.append(
            {
                "path": [label],
                "discriminated": exact_json_bytes(project_serialized(value)["request"])
                == exact_json_bytes(
                    {key: child for key, child in value.items() if key not in EXCLUDED}
                ),
            }
        )
    changed_ids = {**raw, **{name: raw[name] + ":different" for name in EXCLUDED}}
    rows.append(
        {
            "path": ["only-opaque-root-fields"],
            "discriminated": exact_json_bytes(project_serialized(changed_ids)) == projected,
        }
    )
    for name in EXCLUDED:
        for bad in (None, "", " ", 1):
            try:
                project_serialized({**raw, name: bad})
            except ValueError:
                rejected = True
            else:
                rejected = False
            rows.append({"path": [name, "invalid", bad], "discriminated": rejected})
        absent = {key: value for key, value in raw.items() if key != name}
        try:
            project_serialized(absent)
        except ValueError:
            rejected = True
        else:
            rejected = False
        rows.append({"path": [name, "missing"], "discriminated": rejected})
    return rows


def native_arm(args):
    import torch
    import vllm
    from huggingface_hub import hf_hub_download

    from simllm.adapters.vllm.pd_session import VllmDisaggregatedSession
    from simllm.core import DeclaredKvHandoffPolicy
    from simllm.core.step import step_record_to_json

    frozen = json.loads((HERE / "expectations.json").read_bytes())
    package = Path(vllm.__file__).resolve().parent
    if package != args.vllm_source.resolve() or vllm.__version__ != frozen["frontend"]["version"]:
        raise ValueError("native vLLM source origin or version differs")
    source_rows = {}
    for name, expected in frozen["source_audit_sha256"].items():
        actual = sha((package / name.removeprefix("vllm/")).read_bytes())
        if actual != expected:
            raise ValueError("native source mismatch: " + name)
        source_rows[name] = actual
    cached = Path(
        hf_hub_download(
            baseline.MODEL_ID,
            "config.json",
            revision=baseline.MODEL_REVISION,
            local_files_only=True,
        )
    )
    if sha(cached.read_bytes()) != frozen["frontend"]["model_config_sha256"]:
        raise ValueError("native cached model configuration differs")
    prompt = baseline._prompt_tokens()
    cells = []
    cuda_initialized_before = torch.cuda.is_initialized()
    with VllmDisaggregatedSession(
        baseline._session_config(args.output_root / "engine-work")
    ) as session:
        for prompt_length in frozen["request_sweep"]["prompt_tokens"]:
            for handoff in frozen["request_sweep"]["handoff_ps"]:
                label = baseline._cell_key(prompt_length, handoff)
                result = session.run_request(
                    label,
                    prompt[:prompt_length],
                    decode_output_tokens=frozen["request_sweep"]["decode_output_tokens"],
                    handoff_policy=DeclaredKvHandoffPolicy(handoff),
                )
                before = deepcopy(result.to_json())
                before_bytes = exact_json_bytes(before)
                state_before = state_snapshot(result)
                projected = result.to_comparison_json()
                after = deepcopy(result.to_json())
                after_bytes = exact_json_bytes(after)
                state_after = state_snapshot(result)
                timeline = result.timeline
                cells.append(
                    {
                        "label": label,
                        "prompt_tokens": prompt_length,
                        "handoff_ps": handoff,
                        "kv_bytes": timeline.handoff.kv_bytes,
                        "prefill_service_ps": timeline.prefill_service_ps,
                        "decode_first_token_service_ps": timeline.decode_first_token_service_ps,
                        "ttft_ps": timeline.ttft_ps,
                        "tpot_ps": baseline._fraction_int(timeline.tpot_ps),
                        "decomposition_total_ps": timeline.decomposition_total_ps,
                        "request_result": before,
                        "request_result_after": after,
                        "request_result_sha256": sha(before_bytes),
                        "request_result_after_sha256": sha(after_bytes),
                        "state_before": state_before,
                        "state_after": state_after,
                        "comparison": projected,
                        "comparison_sha256": sha(exact_json_bytes(projected)),
                    }
                )
                write(
                    args.output_root / f"{label}.steps.json",
                    {
                        "prefill": [
                            step_record_to_json(record) for record in result.prefill_records
                        ],
                        "decode": [step_record_to_json(record) for record in result.decode_records],
                    },
                )
        engines = (*session.prefill_engines, *session.decode_engines)
        services = [
            step.step_latency_ps
            for engine in engines
            for record, step in zip(
                engine.executor.step_records, engine.executor.step_results, strict=True
            )
            if record.scheduled
        ]
        pricing = {
            engine.role.value: engine.executor.compute_provider.pricing_provenance()
            for engine in engines
        }
        backend_runs = sum(
            outcome.backend_runs
            for engine in engines
            for outcome in engine.step_sink.locality_outcomes
        )
        executor_classes = [
            type(engine.executor).__module__ + "." + type(engine.executor).__name__
            for engine in engines
        ]
    value = {
        "schema": "simllm-pd-session-identity-native-v1",
        "pid": os.getpid(),
        "vllm_version": vllm.__version__,
        "vllm_package": str(package),
        "native_sources": source_rows,
        "model_config_sha256": sha(cached.read_bytes()),
        "cells": cells,
        "step_service_ps": services,
        "pricing_provenance": pricing,
        "backend_runs": backend_runs,
        "executor_classes": executor_classes,
        "cuda_context_initialized": [cuda_initialized_before, torch.cuda.is_initialized()],
        "worker_mode": os.environ.get("SIMLLM_VLLM_WORKER_MODE"),
    }
    write(args.output_root / "native.json", value)
    return 0


def timeline_checks(row):
    """Derive causal timing directly from the retained request serialization."""
    raw = row["request_result"]
    handoff = raw["handoff"]
    tokens = raw["decode_token_completed_at_ps"]
    ordered = [
        raw["admitted_at_ps"],
        raw["prefill_eligible_at_ps"],
        raw["prefill_completed_at_ps"],
        *[
            handoff[key]
            for key in (
                "submitted_at_ps",
                "eligible_at_ps",
                "started_at_ps",
                "finished_at_ps",
                "completed_at_ps",
            )
        ],
        raw["decode_eligible_at_ps"],
        *tokens,
    ]
    if len(tokens) != 4:
        return False
    tpot = Fraction(tokens[-1] - tokens[0], len(tokens) - 1)
    decomposition = {
        "prefill_queue_ps": raw["prefill_eligible_at_ps"] - raw["admitted_at_ps"],
        "prefill_service_ps": raw["prefill_completed_at_ps"] - raw["prefill_eligible_at_ps"],
        "handoff_ps": handoff["completed_at_ps"] - handoff["submitted_at_ps"],
        "decode_admission_wait_ps": raw["decode_eligible_at_ps"] - handoff["completed_at_ps"],
        "decode_first_token_service_ps": tokens[0] - raw["decode_eligible_at_ps"],
    }
    decomposition["total_ps"] = sum(decomposition.values())
    return (
        ordered == sorted(ordered)
        and raw["request_id"] == row["label"]
        and raw["prefill_completed_at_ps"] == handoff["submitted_at_ps"]
        and handoff["completed_at_ps"] - handoff["submitted_at_ps"] == row["handoff_ps"]
        and handoff["kv_bytes"] == row["kv_bytes"]
        and raw["prefill_completed_at_ps"] - raw["prefill_eligible_at_ps"]
        == row["prefill_service_ps"]
        and tokens[0] - raw["decode_eligible_at_ps"] == row["decode_first_token_service_ps"]
        and tokens[0] - raw["admitted_at_ps"] == raw["ttft_ps"] == row["ttft_ps"]
        and raw["tpot_ps"] == {"numerator": tpot.numerator, "denominator": tpot.denominator}
        and tpot == row["tpot_ps"]
        and raw["decomposition"] == decomposition
        and raw["decomposition"]["total_ps"] == row["decomposition_total_ps"] == raw["ttft_ps"]
    )


def evaluate(arms, frozen):
    guards, oracles, relations = [], [], []

    def guard(name, passed):
        guards.append({"name": name, "passed": passed})

    guard("independent-native-processes", len({arm["pid"] for arm in arms}) == 2)
    expected = json.loads((ROOT / "examples/pd_session_v1/results.json").read_bytes())["cells"]
    grid = [
        (p, h)
        for p in frozen["request_sweep"]["prompt_tokens"]
        for h in frozen["request_sweep"]["handoff_ps"]
    ]
    controls = []
    for number, arm in enumerate(arms):
        guard(
            f"arm{number}:grid",
            [(row["prompt_tokens"], row["handoff_ps"]) for row in arm["cells"]] == grid,
        )
        guard(f"arm{number}:accepted-compact", baseline._compact(arm["cells"]) == expected)
        guard(f"arm{number}:skeleton", arm["worker_mode"] == "skeleton")
        guard(
            f"arm{number}:native-executors",
            arm["executor_classes"] == ["simllm.adapters.vllm.executor.SimExecutor"] * 2,
        )
        guard(f"arm{number}:no-packet-backend", arm["backend_runs"] == 0)
        guard(f"arm{number}:no-cuda-context", arm["cuda_context_initialized"] == [False, False])
        guard(
            f"arm{number}:record-absent-providers",
            arm["pricing_provenance"] == {"prefill": None, "decode": None},
        )
        bound = frozen["physical_bounds"]["per_step_service_ps"]
        guard(
            f"arm{number}:service-bounds",
            bool(arm["step_service_ps"])
            and all(
                bound["floor"] <= value <= bound["ceiling"] for value in arm["step_service_ps"]
            ),
        )
        for row in arm["cells"]:
            key = f"arm{number}:" + row["label"]
            raw = row["request_result"]
            guard(key + ":causal-timestamp-joins", timeline_checks(row))
            guard(key + ":off-pricing", "compute_pricing" not in raw)
            guard(
                key + ":raw-content-address",
                sha(exact_json_bytes(raw)) == row["request_result_sha256"],
            )
            guard(
                key + ":nonmutating-projection",
                row["request_result_sha256"] == row["request_result_after_sha256"],
            )
            guard(
                key + ":retained-after-content-address",
                sha(exact_json_bytes(row["request_result_after"]))
                == row["request_result_after_sha256"],
            )
            guard(
                key + ":full-state-nonmutation",
                exact_json_bytes(row["state_before"]) == exact_json_bytes(row["state_after"]),
            )
            expected_projection = {
                "schema": frozen["projection"]["schema"],
                "excluded_root_fields": list(EXCLUDED),
                "request": {k: v for k, v in raw.items() if k not in EXCLUDED},
            }
            guard(
                key + ":complete-projection",
                exact_json_bytes(row["comparison"]) == exact_json_bytes(expected_projection),
            )
            guard(
                key + ":projection-content-address",
                sha(exact_json_bytes(row["comparison"])) == row["comparison_sha256"],
            )
            handoff_bound = frozen["physical_bounds"]["handoff_bounds_ps"][
                str(row["prompt_tokens"])
            ]
            guard(
                key + ":handoff-bounds",
                handoff_bound["floor"] <= row["handoff_ps"] <= handoff_bound["ceiling"],
            )
            for name, actual, target in (
                ("cache-bytes", row["kv_bytes"], row["prompt_tokens"] * 49152),
                ("ttft-decomposition", row["ttft_ps"], row["decomposition_total_ps"]),
            ):
                oracles.append(
                    {
                        "name": key + ":" + name,
                        "actual": actual,
                        "expected": target,
                        "passed": actual == target,
                    }
                )
            cell_controls = corruption_controls(raw)
            guard(
                key + ":preserved-leaf-discrimination",
                all(control["discriminated"] for control in cell_controls),
            )
            controls.append({"cell": key, "controls": cell_controls})
        indexed = {(row["prompt_tokens"], row["handoff_ps"]): row for row in arm["cells"]}
        for prompt in frozen["request_sweep"]["prompt_tokens"]:
            first, second = (indexed[(prompt, h)] for h in frozen["request_sweep"]["handoff_ps"])
            delta = [second[name] - first[name] for name in ("ttft_ps", "tpot_ps")]
            relations.append(
                {
                    "family": "handoff",
                    "arm": number,
                    "prompt_tokens": prompt,
                    "actual": delta,
                    "expected": [100000000, 0],
                    "passed": delta == [100000000, 0],
                }
            )
        for handoff in frozen["request_sweep"]["handoff_ps"]:
            first, second = (
                indexed[(prompt, handoff)] for prompt in frozen["request_sweep"]["prompt_tokens"]
            )
            delta = [
                second[name] - first[name] for name in ("prefill_service_ps", "ttft_ps", "tpot_ps")
            ]
            relations.append(
                {
                    "family": "prompt",
                    "arm": number,
                    "handoff_ps": handoff,
                    "actual_deltas": delta,
                    "passed": delta[0] > 0 and delta[1] > 0 and delta[2] >= 0,
                }
            )
    comparisons = []
    for first, second in zip(arms[0]["cells"], arms[1]["cells"], strict=True):
        paths = difference_paths(first["request_result"], second["request_result"])
        equal = exact_json_bytes(first["comparison"]) == exact_json_bytes(second["comparison"])
        guard(
            first["label"] + ":raw-difference-boundary",
            set(paths) == {(name,) for name in EXCLUDED},
        )
        guard(first["label"] + ":projected-identity", equal)
        comparisons.append(
            {
                "cell": first["label"],
                "raw_difference_paths": [list(p) for p in paths],
                "projected_bytes_equal": equal,
                "comparison_sha256": first["comparison_sha256"],
            }
        )
    valid = all(row["passed"] for row in guards + oracles + relations)
    return {
        "verdict": "PASS" if valid else "VOID",
        "behavioral_score": {"families": 2, "instances": 8} if valid else None,
        "guards": guards,
        "oracles": oracles,
        "relations": relations,
        "corruption_controls": controls,
        "comparisons": comparisons,
        "findings": [row for row in guards + oracles + relations if not row["passed"]],
    }


def run(args):
    args.output_root.mkdir(parents=True, exist_ok=False)
    frozen = json.loads((HERE / "expectations.json").read_bytes())
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip()
    try:
        if subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT
        ):
            raise ValueError("study requires committed source")
        for name in ("expectations.json", "expectations.md"):
            expected = subprocess.check_output(
                ["git", "show", f"{FREEZE_COMMIT}:examples/pd_session_identity_v1/{name}"], cwd=ROOT
            )
            if (HERE / name).read_bytes() != expected:
                raise ValueError("frozen expectations changed")
        for name, expected in frozen["tracked_baseline_sha256"].items():
            if sha((ROOT / name).read_bytes()) != expected:
                raise ValueError("accepted artifact changed: " + name)
        if sha(args.model_config.read_bytes()) != frozen["frontend"]["model_config_sha256"]:
            raise ValueError("configured model input differs")
        arms = []
        for number in range(frozen["native_processes"]):
            output = args.output_root / f"native-{number}"
            output.mkdir()
            env = dict(
                os.environ,
                PYTHONPATH=str(ROOT),
                PYTHONDONTWRITEBYTECODE="1",
                HF_HUB_OFFLINE="1",
                TRANSFORMERS_OFFLINE="1",
                VLLM_ENABLE_V1_MULTIPROCESSING="0",
                SIMLLM_VLLM_WORKER_MODE="skeleton",
            )
            command = [
                str(args.vllm_python),
                "-m",
                "examples.pd_session_identity_v1.run_study",
                "--worker",
                "--output-root",
                str(output),
                "--vllm-source",
                str(args.vllm_source),
            ]
            with (
                (output / "stdout.log").open("wb") as stdout,
                (output / "stderr.log").open("wb") as stderr,
            ):
                process = subprocess.run(
                    command,
                    cwd=ROOT,
                    env=env,
                    stdout=stdout,
                    stderr=stderr,
                    timeout=1200,
                    check=False,
                )
            if process.returncode:
                raise RuntimeError(
                    f"native process {number} failed with exit {process.returncode}; raw logs retained"
                )
            arms.append(json.loads((output / "native.json").read_bytes()))
        result = evaluate(arms, frozen)
        result["native_sha256"] = [
            sha((args.output_root / f"native-{i}/native.json").read_bytes()) for i in range(2)
        ]
        result["accepted_cells"] = baseline._compact(arms[0]["cells"])
    except (
        OSError,
        ValueError,
        TypeError,
        RuntimeError,
        KeyError,
        subprocess.SubprocessError,
    ) as error:
        result = {
            "verdict": "VOID",
            "behavioral_score": None,
            "findings": [{"error": f"{type(error).__name__}: {error}"}],
        }
    result.update(
        schema="simllm-pd-session-identity-study-v1",
        task="CORE-58",
        freeze_commit=FREEZE_COMMIT,
        source_commit=commit,
        original_core53_verdict="VOID",
        hardware_measurements=0,
    )
    write(args.output_root / "summary.json", result)
    print(json.dumps({key: result[key] for key in ("verdict", "behavioral_score", "findings")}))
    return 0 if result["verdict"] == "PASS" else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--vllm-source", type=Path, required=True)
    parser.add_argument("--vllm-python", type=Path)
    parser.add_argument("--model-config", type=Path)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    if args.worker:
        return native_arm(args)
    if args.vllm_python is None or args.model_config is None:
        parser.error("--vllm-python and --model-config are required for the coordinator")
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
