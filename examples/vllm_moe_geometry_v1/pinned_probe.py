"""Re-derive the frozen parallel-side cells against a real pinned ParallelConfig.

This script runs under an interpreter that has vLLM v0.26.0 installed. It
builds the study's cells from vLLM's own ``ParallelConfig`` instead of a
stand-in namespace, so the field names, defaults and validation the geometry
reader depends on come from the pinned package rather than from this
repository's idea of them. It writes one JSON object on stdout and never
imports a model checkpoint.
"""

from __future__ import annotations

import json
import os
import sys


def _parallel_config(dp: int, pcp: int, tp: int, pp: int, rank: int, enable_ep: bool):
    from vllm.config import ParallelConfig

    world = dp * pcp * tp * pp
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = "0"
    os.environ["WORLD_SIZE"] = str(world)
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    return ParallelConfig(
        data_parallel_size=dp,
        prefill_context_parallel_size=pcp,
        tensor_parallel_size=tp,
        pipeline_parallel_size=pp,
        rank=rank,
        enable_expert_parallel=enable_ep,
        distributed_executor_backend="external_launcher",
    )


def main() -> None:
    from types import SimpleNamespace

    import vllm

    from simllm.adapters.vllm.executor import expert_parallel_geometry

    request = json.load(sys.stdin)
    result: dict = {
        "available": True,
        "vllm_version": vllm.__version__,
        "geometry": {},
        "layout": {},
    }

    for cell, inputs in request["geometry"].items():
        dp, pcp, tp, enable_ep, rank, _experts, _redundant, pp = inputs
        try:
            parallel = _parallel_config(dp, pcp, tp, pp, rank, bool(enable_ep))
        except Exception as exc:  # noqa: BLE001 - a refused shape is a finding
            result["geometry"][cell] = {"constructed": False, "error": str(exc)[:200]}
            continue
        geometry = expert_parallel_geometry(SimpleNamespace(parallel_config=parallel))
        result["geometry"][cell] = {
            "constructed": True,
            "flatten_tp_size": geometry.flatten_tp_size,
            "use_ep": geometry.use_ep,
            "ep_size": geometry.ep_size,
            "moe_tp_size": geometry.moe_tp_size,
            "ep_ranks": list(geometry.ep_ranks),
            "ep_rank": geometry.ep_rank,
            "num_redundant_experts": parallel.eplb_config.num_redundant_experts,
        }

    for cell, inputs in request["layout"].items():
        _external_dp, dp, pp, pcp, tp, rank = inputs
        try:
            parallel = _parallel_config(dp, pcp, tp, pp, rank, True)
        except Exception as exc:  # noqa: BLE001 - a refused shape is a finding
            result["layout"][cell] = {"constructed": False, "error": str(exc)[:200]}
            continue
        geometry = expert_parallel_geometry(SimpleNamespace(parallel_config=parallel))
        result["layout"][cell] = {
            "constructed": True,
            "ep_ranks": list(geometry.ep_ranks),
            "ep_rank": geometry.ep_rank,
        }

    json.dump(result, sys.stdout)


if __name__ == "__main__":
    main()
