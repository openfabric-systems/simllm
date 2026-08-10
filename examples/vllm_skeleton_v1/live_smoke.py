"""One in-process vLLM smoke for the flagged skeleton worker seam."""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

from vllm import LLM, SamplingParams

MODEL = Path(
    "/home/yifeng/packages/vllm-rnic-capture/hf-cache/hub/"
    "models--ibm-granite--granite-3.0-1b-a400m-instruct/snapshots/"
    "ffec3c35bdfd97a06f0b4cd5fcc92cd9b1584445"
)


def worker_reached() -> bool:
    module = sys.modules.get("simllm.adapters.vllm.worker")
    workers = getattr(module, "_LATEST_WORKERS", ()) if module is not None else ()
    return bool(workers)


def main() -> None:
    print(f"SMOKE_MODEL={MODEL}")
    print("SMOKE_WORKER_CLS=simllm.adapters.vllm.SimWorker")
    try:
        llm = LLM(
            model=str(MODEL),
            worker_cls="simllm.adapters.vllm.SimWorker",
            enforce_eager=True,
            max_model_len=64,
            num_gpu_blocks_override=64,
            disable_log_stats=True,
        )
        outputs = llm.generate(
            ["The simulated worker"],
            SamplingParams(max_tokens=2, ignore_eos=True),
        )
    except BaseException:
        print(f"SMOKE_SIMWORKER_REACHED={worker_reached()}")
        traceback.print_exc()
        raise

    print(f"SMOKE_SIMWORKER_REACHED={worker_reached()}")
    print(f"SMOKE_OUTPUT_COUNT={len(outputs)}")


if __name__ == "__main__":
    main()
