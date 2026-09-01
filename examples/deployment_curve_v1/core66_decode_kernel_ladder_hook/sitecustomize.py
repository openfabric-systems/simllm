"""Mark the exact vLLM batch-32, KV-2,000 model-execution boundary."""

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
    path = os.environ.get("KLADDER_ACTIVE_FILE")
    if not path:
        return {}
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _write_marker(active: dict[str, Any], scheduler: dict[str, Any]) -> None:
    root = os.environ.get("KLADDER_MARKER_DIR")
    if not root:
        return
    path = Path(root)
    path.mkdir(parents=True, exist_ok=True)
    target = path / "exact-decode-rank0.json"
    temporary = target.with_suffix(".next")
    temporary.write_text(
        json.dumps(
            {
                "epoch_ns": time.time_ns(),
                "active": active,
                "scheduler": scheduler,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(target)


def _patch_runner() -> None:
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    if getattr(GPUModelRunner.execute_model, "_kladder_wrapped", False):
        return
    original = GPUModelRunner.execute_model
    marked = False

    @functools.wraps(original)
    def wrapped(self: Any, scheduler_output: Any, *args: Any, **kwargs: Any):
        nonlocal marked
        active = _active()
        cached = scheduler_output.scheduled_cached_reqs
        output_tokens = list(getattr(cached, "num_output_tokens", []))
        computed_tokens = list(getattr(cached, "num_computed_tokens", []))
        exact = (
            not marked
            and active.get("enabled") is True
            and not scheduler_output.scheduled_new_reqs
            and len(scheduler_output.num_scheduled_tokens) == 32
            and len(output_tokens) == 32
            and all(int(value) > 0 for value in output_tokens)
            and len(computed_tokens) == 32
            and all(int(value) == 2000 for value in computed_tokens)
        )
        if exact:
            marked = True
            _write_marker(
                active,
                {
                    "request_ids": sorted(scheduler_output.num_scheduled_tokens),
                    "num_requests": len(scheduler_output.num_scheduled_tokens),
                    "num_computed_tokens": computed_tokens,
                    "num_output_tokens": output_tokens,
                    "is_exact_decode": True,
                },
            )
        return original(self, scheduler_output, *args, **kwargs)

    wrapped._kladder_wrapped = True
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
            _patch_runner()
        return result

    builtins.__import__ = wrapped_import


if os.environ.get("KLADDER_SCHEDULER_HOOK") == "1":
    _install_import_hook()
