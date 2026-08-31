"""Observe the exact vLLM scheduler boundary for the CORE-61 retry."""

from __future__ import annotations

import builtins
import functools
import json
import os
import sys
import time
from pathlib import Path
from typing import Any


def _active() -> dict[str, Any]:
    path = os.environ.get("CORE61_ACTIVE_FILE")
    if not path:
        return {}
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _scheduler_record(scheduler_output: Any) -> dict[str, Any]:
    cached = scheduler_output.scheduled_cached_reqs
    output_tokens = list(getattr(cached, "num_output_tokens", []))
    computed_tokens = list(getattr(cached, "num_computed_tokens", []))
    is_decode = (
        not scheduler_output.scheduled_new_reqs
        and bool(output_tokens)
        and all(value > 0 for value in output_tokens)
        and scheduler_output.total_num_scheduled_tokens
        == len(scheduler_output.num_scheduled_tokens)
    )
    return {
        "request_ids": sorted(scheduler_output.num_scheduled_tokens),
        "num_requests": len(scheduler_output.num_scheduled_tokens),
        "total_num_scheduled_tokens": scheduler_output.total_num_scheduled_tokens,
        "new_request_count": len(scheduler_output.scheduled_new_reqs),
        "cached_output_tokens_by_request": dict(zip(cached.req_ids, output_tokens)),
        "cached_num_computed_tokens_by_request": dict(
            zip(cached.req_ids, computed_tokens)
        ),
        "is_decode": is_decode,
    }


def _write_marker(active: dict[str, Any], scheduler: dict[str, Any]) -> None:
    root = os.environ.get("CORE61_MARKER_DIR")
    if not root:
        return
    path = Path(root)
    path.mkdir(parents=True, exist_ok=True)
    cell = str(active["cell"]).replace("/", "_")
    target = path / f"{cell}-rank0.json"
    temporary = target.with_suffix(".next")
    temporary.write_text(
        json.dumps(
            {
                "cell": active["cell"],
                "epoch_ns": time.time_ns(),
                "scheduler": scheduler,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(target)


def _patch_legacy_runner() -> None:
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    if getattr(GPUModelRunner.execute_model, "_core61_wrapped", False):
        return
    original = GPUModelRunner.execute_model
    marked_cells: set[str] = set()

    @functools.wraps(original)
    def wrapped(self: Any, scheduler_output: Any, *args: Any, **kwargs: Any):
        active = _active()
        scheduler = _scheduler_record(scheduler_output)
        cell = str(active.get("cell", "inactive"))
        expected_batch = int(active.get("batch_size") or 0)
        full_decode = (
            scheduler["is_decode"]
            and scheduler["num_requests"] == expected_batch
            and cell not in marked_cells
        )
        if full_decode and active.get("kind") == "kv_calibration":
            marked_cells.add(cell)
            _write_marker(active, scheduler)
        elif full_decode and active.get("kind") == "core61_exact_decode":
            target = int(active["target_kv_tokens"])
            computed = scheduler["cached_num_computed_tokens_by_request"]
            if len(computed) == expected_batch and all(
                int(value) == target for value in computed.values()
            ):
                marked_cells.add(cell)
                _write_marker(active, scheduler)
        return original(self, scheduler_output, *args, **kwargs)

    wrapped._core61_wrapped = True
    GPUModelRunner.execute_model = wrapped


def _install_import_hook() -> None:
    original_import = builtins.__import__

    @functools.wraps(original_import)
    def wrapped_import(
        name: str,
        globals: Any = None,
        locals: Any = None,
        fromlist: Any = (),
        level: int = 0,
    ):
        result = original_import(name, globals, locals, fromlist, level)
        module = sys.modules.get("vllm.v1.worker.gpu_model_runner")
        if module is not None and hasattr(module, "GPUModelRunner"):
            builtins.__import__ = original_import
            _patch_legacy_runner()
        return result

    builtins.__import__ = wrapped_import


if os.environ.get("CORE61_SCHEDULER_HOOK") == "1":
    _install_import_hook()
