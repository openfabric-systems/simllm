# CORE-66 EP72 physical capture remainder

Run the pinned SGLang source at commit
`bfeae4e79a8dc4600e006f1a5fbc85321a01c1a3` with the pre-staged local
DeepSeek-V3 model. Do not download weights. The cell is DP72/EP72, batch 32
and KV length 2,000 per rank, MTP disabled, and one measured decode iteration.

## Required rank coverage

- one of ranks 0-39 with four logical experts in four slots
- one of ranks 40-71 with three logical experts plus one redundant slot

The preferred capture covers all 72 ranks, not only these representatives.

## Kernel-trace command

Run the following once on each of nine nodes with node-specific `NODE_RANK`.

```bash
export PYTHONPATH="$SIMLLM_SGLANG_SOURCE/python"
export SIMLLM_CORE66_RUN_ROOT="$SIMLLM_WAVE_RUNS/core66/ep72-b32-c2000"
nsys profile --force-overwrite=true --trace=cuda,nvtx,osrt,cublas --sample=none --gpu-metrics-device=all --trace-fork-before-exec=true --output "$SIMLLM_CORE66_RUN_ROOT/node-$NODE_RANK" \
  "$SIMLLM_SGLANG_PYTHON" -m sglang.benchmark.one_batch \
  --model-path "$SIMLLM_DEEPSEEK_V3_LOCAL_MODEL" \
  --tp-size 72 --dp-size 72 --ep-size 72 \
  --nnodes 9 --node-rank "$NODE_RANK" --dist-init-addr "$MASTER_ADDR:$MASTER_PORT" \
  --enable-dp-attention --enable-dp-lm-head --moe-a2a-backend deepep \
  --batch-size 32 --input-len 2000 --output-len 2 \
  --profile --profile-stage decode --profile-start-step 0 --profile-steps 1 \
  --profile-record-shapes --profile-activities CUDA_PROFILER \
  --result-filename "$SIMLLM_CORE66_RUN_ROOT/result-node-$NODE_RANK.jsonl"

```

## Required evidence

- all 72 rank identities and devices
- every CUDA launch name, order, duration, grid, block, stream, and correlation ID
- NVTX semantic phase and resolved attention, MoE, and LM-head backend
- per-layer routed expert IDs, assignment counts, and local physical slot IDs
- DeepEP dispatch and combine launches, peers, payload bytes, and durations
- actual per-kernel and per-step HBM read and write bytes
- fusion flags and the local resident weight byte inventory

Repeat the identical cell with rank-aware CUPTI or Nsight Compute application
replay and `dram__bytes_read.sum` plus `dram__bytes_write.sum`. The stock
`one_batch` profiler marker is not by itself an all-rank HBM-counter capture;
the counter pass must instrument every child rank and preserve rank identity.
