"""One in-process vLLM smoke for the flagged skeleton worker seam."""

from __future__ import annotations

import json
import os
import traceback
from pathlib import Path

from vllm import LLM, SamplingParams

from simllm.adapters.vllm import SimModelRunner, latest_worker

MODEL = Path(
    "/home/yifeng/packages/vllm-rnic-capture/hf-cache/hub/"
    "models--ibm-granite--granite-3.0-1b-a400m-instruct/snapshots/"
    "ffec3c35bdfd97a06f0b4cd5fcc92cd9b1584445"
)
EXPECTED_STEP_SCHEMA = "atlahs-closed-loop-step-v1"
EXPECTED_OUTPUT_TOKENS = 2


def worker_reached() -> bool:
    return latest_worker() is not None


def main() -> None:
    print(f"SMOKE_MODEL={MODEL}")
    print("SMOKE_WORKER_CLS=simllm.adapters.vllm.SimWorker")
    try:
        stream_value = os.environ.get("SIMLLM_VLLM_STEP_RECORDS")
        if not stream_value:
            raise RuntimeError("SIMLLM_VLLM_STEP_RECORDS must name a fresh JSONL path")
        stream_path = Path(stream_value)
        if stream_path.exists():
            raise RuntimeError(f"refusing stale smoke evidence at {stream_path}")

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
        worker = latest_worker()
        assert worker is not None, "SimWorker was not constructed"
        assert isinstance(worker.model_runner, SimModelRunner), (
            f"unexpected runner {type(worker.model_runner)!r}"
        )
        assert len(outputs) == 1, f"expected one request output, got {len(outputs)}"
        assert len(outputs[0].outputs) == 1, (
            f"expected one completion, got {len(outputs[0].outputs)}"
        )
        sampled_token_ids = tuple(outputs[0].outputs[0].token_ids)
        expected_token_ids = (worker.token_id,) * EXPECTED_OUTPUT_TOKENS
        assert sampled_token_ids == expected_token_ids, (
            f"sampled {sampled_token_ids}, expected {expected_token_ids}"
        )

        records = [
            json.loads(line)
            for line in stream_path.read_text().splitlines()
            if line.strip()
        ]
        assert len(records) == EXPECTED_OUTPUT_TOKENS, (
            f"expected {EXPECTED_OUTPUT_TOKENS} records, got {len(records)}"
        )
        schemas = {record.get("schema") for record in records}
        assert schemas == {EXPECTED_STEP_SCHEMA}, (
            f"unexpected step-record schemas {schemas}"
        )
    except BaseException:
        print(f"SMOKE_SIMWORKER_REACHED={worker_reached()}")
        traceback.print_exc()
        raise

    print(f"SMOKE_SIMWORKER_REACHED={worker_reached()}")
    print("SMOKE_SIMRUNNER_MIRROR=True")
    print(f"SMOKE_OUTPUT_COUNT={len(outputs)}")
    print(f"SMOKE_FABRICATED_TOKEN_ID={worker.token_id}")
    print(f"SMOKE_SAMPLED_TOKEN_IDS={','.join(map(str, sampled_token_ids))}")
    print(f"SMOKE_STEP_RECORD_COUNT={len(records)}")
    print(f"SMOKE_STEP_SCHEMA={EXPECTED_STEP_SCHEMA}")


if __name__ == "__main__":
    main()
