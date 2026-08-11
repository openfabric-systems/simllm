"""Run the frozen VLLM-16 GPU-invisible skeleton smoke attempts."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

MODEL_RELATIVE_PATH = (
    Path("hub")
    / "models--ibm-granite--granite-3.0-1b-a400m-instruct"
    / "snapshots"
    / "ffec3c35bdfd97a06f0b4cd5fcc92cd9b1584445"
)
SOURCE_HASHES = {
    "platforms/__init__.py": "a2bd800acc39b3215ccb78808d43317b351f137072b03e7f0f0ab3d069d91521",
    "platforms/cpu.py": "067f92d391b1c131e12a7ba9631921e4b9dd57d3c55b1d8724e9963e2fdc9c7d",
    "config/device.py": "7b82eee02ceb5842337451a27a3d5729920c47e25e8f6bf3997f5146f9330a9c",
    "v1/worker/worker_base.py": "7da44338c2645ebf03d23394e452b31a8e3da1011fd1b42fcfcccfe99551b3fe",
}
REPOSITORY_HASHES = {
    "simllm/adapters/vllm/worker.py": "07e2d26213a1899aaf2604787cd85f47a67731d660b94fd473943831e7bccd2e",
    "examples/vllm_skeleton_v1/live_smoke.py": "a43d5e6987b0322bc0a6d05d3b7046de84980f7cfd3600eb4b94b8a7d56782cc",
}
BWRAP_SHA256 = "a87328fd969d4bc9fbc62e56b15a393b2b23c7b47aa092a3ac02955a68da19e4"
INVALID_UUID = "GPU-00000000-0000-0000-0000-000000000000"
MECHANISMS = ("invalid-uuid", "device-namespace", "cpu-platform")
EXPECTED_STEP_SCHEMA = "atlahs-closed-loop-step-v1"
EXPECTED_OUTPUT_TOKENS = 2
EXPECTED_SCORED = 3
FREEZE_COMMIT = "25e79be"


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_only(args: argparse.Namespace) -> None:
    model = args.cache_dir / MODEL_RELATIVE_PATH
    if not model.is_dir() or not (model / "config.json").is_file():
        raise SystemExit(f"pinned model snapshot is missing: {model}")
    if importlib.metadata.version("vllm") != "0.26.0":
        raise SystemExit("VLLM-16 requires vLLM 0.26.0")
    for relative, expected in SOURCE_HASHES.items():
        path = args.vllm_package_root / relative
        if not path.is_file() or file_sha256(path) != expected:
            raise SystemExit(f"pinned vLLM source changed: {relative}")
    for relative, expected in REPOSITORY_HASHES.items():
        path = REPOSITORY_ROOT / relative
        if not path.is_file() or file_sha256(path) != expected:
            raise SystemExit(f"pinned repository source changed: {relative}")
    if not args.bwrap.is_file() or not args.bwrap.stat().st_mode & 0o111:
        raise SystemExit(f"bubblewrap is missing or not executable: {args.bwrap}")
    if file_sha256(args.bwrap) != BWRAP_SHA256:
        raise SystemExit("bubblewrap executable identity changed")
    if MECHANISMS != ("invalid-uuid", "device-namespace", "cpu-platform"):
        raise AssertionError("isolation mechanism order changed")
    if re.fullmatch(r"GPU-0{8}-0{4}-0{4}-0{4}-0{12}", INVALID_UUID) is None:
        raise AssertionError("invalid UUID sentinel changed")
    if EXPECTED_OUTPUT_TOKENS != 2:
        raise AssertionError("smoke output count changed")
    if EXPECTED_STEP_SCHEMA != "atlahs-closed-loop-step-v1":
        raise AssertionError("smoke schema changed")
    if EXPECTED_SCORED != len(MECHANISMS):
        raise AssertionError("evidence denominator changed")
    print(
        f"check-only run-dir={args.run_dir}; validated frozen VLLM-16 inputs "
        "and produced no artifacts"
    )


def _device_probe() -> dict[str, Any]:
    import pynvml
    import torch

    nodes = sorted(str(path) for path in Path("/dev").glob("nvidia*"))
    nvml: dict[str, Any]
    initialized = False
    try:
        pynvml.nvmlInit()
        initialized = True
        count = int(pynvml.nvmlDeviceGetCount())
        names = [
            str(pynvml.nvmlDeviceGetName(pynvml.nvmlDeviceGetHandleByIndex(index)))
            for index in range(count)
        ]
        nvml = {"status": "available", "count": count, "names": names}
    except Exception as exc:  # noqa: BLE001 - visibility failures are evidence
        nvml = {
            "status": "unavailable",
            "exception_type": type(exc).__name__,
            "message": str(exc),
        }
    finally:
        if initialized:
            pynvml.nvmlShutdown()
    cuda_error = None
    try:
        allocated = int(torch.cuda.memory_allocated())
    except Exception as exc:  # noqa: BLE001 - allocation probe failure is evidence
        allocated = None
        cuda_error = f"{type(exc).__name__}: {exc}"
    return {
        "device_nodes": nodes,
        "device_node_count": len(nodes),
        "nvml": nvml,
        "torch_cuda_available": bool(torch.cuda.is_available()),
        "torch_cuda_device_count": int(torch.cuda.device_count()),
        "torch_cuda_allocated_bytes": allocated,
        "torch_cuda_probe_error": cuda_error,
    }


def _probe_invisible(probe: dict[str, Any]) -> bool:
    nvml = probe["nvml"]
    nvml_hidden = nvml["status"] == "unavailable" or nvml.get("count") == 0
    return (
        probe["device_node_count"] == 0
        and nvml_hidden
        and not probe["torch_cuda_available"]
        and probe["torch_cuda_device_count"] == 0
        and probe["torch_cuda_allocated_bytes"] == 0
    )


def _run_internal(args: argparse.Namespace) -> None:
    mechanism = args.internal
    if mechanism not in MECHANISMS:
        raise AssertionError("internal mechanism is not registered")
    attempt_dir = args.run_dir / mechanism
    result_path = attempt_dir / "attempt.json"
    stream_path = attempt_dir / "steps.jsonl"
    before = _device_probe()
    reached = ["pre-import-probe"]
    llm: Any | None = None
    platform_before = None
    platform_after = None
    diagnostics: list[str] = []
    smoke: dict[str, Any] = {
        "worker_reached": False,
        "runner_is_sim": False,
        "worker_device_unset": False,
        "output_count": 0,
        "sampled_token_ids": [],
        "fabricated_token_id": None,
        "record_count": 0,
        "record_schemas": [],
        "passed": False,
    }
    exception = None
    try:
        if mechanism == "cpu-platform":
            import vllm.platforms
            from vllm.platforms.cpu import CpuPlatform

            vllm.platforms._current_platform = CpuPlatform()
            reached.append("cpu-platform-forced")

        from vllm import LLM, SamplingParams
        from vllm.platforms import current_platform

        from simllm.adapters.vllm import SimModelRunner, latest_worker

        platform_before = type(current_platform).__name__
        reached.append(f"platform:{platform_before}")
        llm = LLM(
            model=str(args.cache_dir / MODEL_RELATIVE_PATH),
            worker_cls="simllm.adapters.vllm.SimWorker",
            enforce_eager=True,
            max_model_len=64,
            num_gpu_blocks_override=64,
            disable_log_stats=True,
            enable_chunked_prefill=False,
            async_scheduling=False,
        )
        reached.append("llm-constructed")
        outputs = llm.generate(
            ["The simulated worker"],
            SamplingParams(max_tokens=EXPECTED_OUTPUT_TOKENS, ignore_eos=True),
            use_tqdm=False,
        )
        reached.append("generation-completed")
        worker = latest_worker()
        choices = outputs[0].outputs if len(outputs) == 1 else ()
        sampled = list(choices[0].token_ids) if len(choices) == 1 else []
        records = [
            json.loads(line)
            for line in stream_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        schemas = sorted({record.get("schema") for record in records})
        smoke = {
            "worker_reached": worker is not None,
            "runner_is_sim": (
                worker is not None and isinstance(worker.model_runner, SimModelRunner)
            ),
            "worker_device_unset": worker is not None and worker.device is None,
            "output_count": len(outputs),
            "sampled_token_ids": sampled,
            "fabricated_token_id": worker.token_id if worker is not None else None,
            "record_count": len(records),
            "record_schemas": schemas,
            "passed": (
                worker is not None
                and isinstance(worker.model_runner, SimModelRunner)
                and worker.device is None
                and len(outputs) == 1
                and len(choices) == 1
                and sampled == [worker.token_id] * EXPECTED_OUTPUT_TOKENS
                and len(records) == EXPECTED_OUTPUT_TOKENS
                and schemas == [EXPECTED_STEP_SCHEMA]
            ),
        }
        platform_after = type(current_platform).__name__
    except Exception as exc:  # noqa: BLE001 - the first smoke boundary is evidence
        exception = {
            "exception_type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        try:
            from simllm.adapters.vllm import latest_worker

            worker = latest_worker()
            smoke["worker_reached"] = worker is not None
            if worker is not None:
                smoke["worker_device_unset"] = worker.device is None
        except Exception as diagnostic_exc:  # noqa: BLE001 - diagnostic only
            diagnostics.append(f"worker probe: {type(diagnostic_exc).__name__}: {diagnostic_exc}")
    finally:
        if llm is not None:
            llm.llm_engine.engine_core.shutdown()
    after = _device_probe()
    try:
        from vllm.platforms import current_platform

        platform_after = platform_after or type(current_platform).__name__
    except Exception as diagnostic_exc:  # noqa: BLE001 - diagnostic only
        diagnostics.append(f"platform probe: {type(diagnostic_exc).__name__}: {diagnostic_exc}")
    payload = {
        "mechanism": mechanism,
        "status": "completed" if exception is None else "blocked",
        "before": before,
        "after": after,
        "platform_before": platform_before,
        "platform_after": platform_after,
        "reached": reached,
        "diagnostics": diagnostics,
        "smoke": smoke,
        "exception": exception,
    }
    result_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _child_arguments(args: argparse.Namespace, mechanism: str) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--cache-dir",
        str(args.cache_dir),
        "--vllm-package-root",
        str(args.vllm_package_root),
        "--bwrap",
        str(args.bwrap),
        "--run-dir",
        str(args.run_dir),
        "--internal",
        mechanism,
    ]


def _attempt_environment(attempt_dir: Path) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
            "VLLM_USE_V2_MODEL_RUNNER": "0",
            "SIMLLM_VLLM_WORKER_MODE": "skeleton",
            "SIMLLM_VLLM_MODE": "virtual",
            "SIMLLM_VLLM_STEP_RECORDS": str(attempt_dir / "steps.jsonl"),
        }
    )
    return env


def _launch_attempt(args: argparse.Namespace, mechanism: str) -> dict[str, Any]:
    attempt_dir = args.run_dir / mechanism
    attempt_dir.mkdir(parents=True, exist_ok=False)
    child = _child_arguments(args, mechanism)
    env = _attempt_environment(attempt_dir)
    if mechanism == "invalid-uuid":
        env["CUDA_VISIBLE_DEVICES"] = INVALID_UUID
        command = child
    elif mechanism == "device-namespace":
        command = [
            str(args.bwrap),
            "--die-with-parent",
            "--unshare-all",
            "--ro-bind",
            "/",
            "/",
            "--bind",
            str(attempt_dir),
            str(attempt_dir),
            "--dev",
            "/dev",
            "--proc",
            "/proc",
            "--tmpfs",
            "/tmp",
            "--chdir",
            str(REPOSITORY_ROOT),
            "--unsetenv",
            "CUDA_VISIBLE_DEVICES",
            *child,
        ]
    else:
        env.pop("CUDA_VISIBLE_DEVICES", None)
        command = child
    log_path = attempt_dir / "attempt.log"
    returncode = None
    launcher_exception = None
    try:
        with log_path.open("x", encoding="utf-8") as log:
            completed = subprocess.run(
                command,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=180,
                check=False,
            )
        returncode = completed.returncode
    except Exception as exc:  # noqa: BLE001 - launcher failure is one outcome
        launcher_exception = {
            "exception_type": type(exc).__name__,
            "message": str(exc),
        }
    result_path = attempt_dir / "attempt.json"
    if result_path.is_file():
        payload = json.loads(result_path.read_text(encoding="utf-8"))
    else:
        payload = {
            "mechanism": mechanism,
            "status": "launcher-failed",
            "before": None,
            "after": None,
            "platform_before": None,
            "platform_after": None,
            "reached": [],
            "smoke": {"passed": False, "worker_reached": False},
            "exception": launcher_exception,
        }
        result_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    before = payload.get("before")
    after = payload.get("after")
    platform_names = (
        str(payload.get("platform_before")),
        str(payload.get("platform_after")),
    )
    log_gpu_free = not any(
        marker in log_text.lower()
        for marker in (
            "gtx 1660 ti",
            "device_config=cuda",
            "cuda platform",
        )
    )
    invisibility = {
        "pre_import_probe": before is not None and _probe_invisible(before),
        "post_smoke_probe": after is not None and _probe_invisible(after),
        "platform_not_cuda": all("cuda" not in name.lower() for name in platform_names),
        "log_gpu_free": log_gpu_free,
    }
    joint_pass = all(invisibility.values()) and bool(payload["smoke"]["passed"])
    return {
        **payload,
        "returncode": returncode,
        "launcher_exception": launcher_exception,
        "invisibility": invisibility,
        "invisibility_passed": all(invisibility.values()),
        "joint_passed": joint_pass,
        "log_sha256": file_sha256(log_path),
    }


def run_study(args: argparse.Namespace) -> dict[str, Any]:
    args.run_dir.mkdir(parents=True, exist_ok=False)
    baseline = _device_probe()
    attempts = [_launch_attempt(args, mechanism) for mechanism in MECHANISMS]
    if tuple(attempt["mechanism"] for attempt in attempts) != MECHANISMS:
        raise AssertionError("attempt order changed")
    passed = sum(attempt["joint_passed"] for attempt in attempts)
    closest = None
    for attempt in attempts:
        if attempt["invisibility_passed"]:
            closest = {
                "mechanism": attempt["mechanism"],
                "boundary": (
                    "smoke-passed"
                    if attempt["smoke"]["passed"]
                    else attempt["reached"][-1]
                    if attempt["reached"]
                    else "launcher"
                ),
                "exception": attempt["exception"],
            }
            break
    summary = {
        "freeze_commit": FREEZE_COMMIT,
        "baseline": baseline,
        "attempt_order": list(MECHANISMS),
        "attempts": attempts,
        "scored": {
            "executed": EXPECTED_SCORED,
            "passed": passed,
            "genuine_risk_numerator": EXPECTED_SCORED,
            "genuine_risk_denominator": EXPECTED_SCORED,
        },
        "vllm16_complete": passed > 0,
        "closest_gpu_invisible_boundary": closest,
    }
    (args.run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--vllm-package-root", type=Path, required=True)
    parser.add_argument("--bwrap", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--internal", choices=MECHANISMS, help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    check_only(args)
    if args.check_only:
        return
    if args.internal is not None:
        _run_internal(args)
        return
    summary = run_study(args)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
