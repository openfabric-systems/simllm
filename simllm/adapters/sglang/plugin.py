"""SGLang plugin that installs :class:`SimTpModelWorker` without a fork.

SGLang discovers setuptools entry points in the ``sglang.srt.plugins`` group
and runs them inside ``run_scheduler_process`` before the ``Scheduler`` is
constructed, exactly so a plugin can override scheduler dependencies.
SimLLM declares :func:`register` as that entry point (see pyproject.toml),
and it applies a ``REPLACE`` hook on ``Scheduler.init_tp_model_worker``, the
same construction point where SGLang swaps in its MLX worker.

Two gates keep this inert unless explicitly requested:

- ``SIMLLM_SGLANG_ENABLE=1`` must be set, because SGLang loads *every*
  discovered plugin when its own ``SGLANG_PLUGINS`` allowlist is unset;
  without this gate, merely having simllm installed in an SGLang venv would
  hijack every real run.
- SGLang's ``SGLANG_PLUGINS=simllm`` allowlist works as an additional
  opt-in when other plugins are present.

:func:`install` is the in-process alternative for drivers that construct
the scheduler themselves (tests, notebooks): it applies the same
replacement directly.
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

__all__ = ["ENABLE_ENV", "install", "register"]

ENABLE_ENV = "SIMLLM_SGLANG_ENABLE"

#: dotted path of the method the plugin replaces, at the pinned commit
HOOK_TARGET = "sglang.srt.managers.scheduler.Scheduler.init_tp_model_worker"


def _init_sim_tp_model_worker(scheduler: Any) -> None:
    """Replacement for ``Scheduler.init_tp_model_worker``.

    Mirrors the original's ``worker_kwargs`` exactly (verified at the
    pinned commit) and sets the one attribute the original sets.
    """
    from simllm.adapters.sglang.worker import SimTpModelWorker

    scheduler.tp_worker = SimTpModelWorker(
        server_args=scheduler.server_args,
        gpu_id=scheduler.ps.gpu_id,
        ps=scheduler.ps,
        nccl_port=scheduler.nccl_port,
    )


def enabled(env: Any = None) -> bool:
    """Whether the activation gate is set."""
    env = os.environ if env is None else env
    return env.get(ENABLE_ENV, "") == "1"


def register() -> None:
    """Entry point for SGLang's ``sglang.srt.plugins`` group.

    A no-op unless ``SIMLLM_SGLANG_ENABLE=1``, so an installed simllm never
    changes the behavior of a real SGLang deployment by accident.
    """
    if not enabled():
        logger.info(
            "simllm SGLang plugin present but inactive (%s != 1)", ENABLE_ENV
        )
        return
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    HookRegistry.register(HOOK_TARGET, _init_sim_tp_model_worker, HookType.REPLACE)
    logger.info("simllm SGLang plugin active: %s replaced", HOOK_TARGET)


def install() -> None:
    """Apply the replacement directly, for in-process scheduler drivers.

    Unlike :func:`register` this ignores the environment gate: calling it
    is itself the explicit opt-in.
    """
    from sglang.srt.managers.scheduler import Scheduler

    Scheduler.init_tp_model_worker = _init_sim_tp_model_worker
    logger.info("simllm SGLang worker installed on Scheduler directly")
