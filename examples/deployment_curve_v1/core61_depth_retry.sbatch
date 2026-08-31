#!/usr/bin/env bash
#SBATCH --account=merlin
#SBATCH --partition=gh-hourly
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

set -uo pipefail

: "${MODEL:?MODEL is required}"
: "${REVISION:?REVISION is required}"
: "${REDUCED_LAYERS:?REDUCED_LAYERS is required}"
: "${GPU_MEMORY_UTILIZATION:?GPU_MEMORY_UTILIZATION is required}"
: "${MAX_MODEL_LEN:?MAX_MODEL_LEN is required}"
: "${STARTUP_MAX_NUM_BATCHED_TOKENS:?STARTUP_MAX_NUM_BATCHED_TOKENS is required}"
: "${MAX_NUM_SEQS:?MAX_NUM_SEQS is required}"
: "${BATCH_SIZE:?BATCH_SIZE is required}"
: "${REMOTE_KV_TOKENS:?REMOTE_KV_TOKENS is required}"
: "${SIMLLM_CORE61_RETRY_STAGE_ROOT:?SIMLLM_CORE61_RETRY_STAGE_ROOT is required}"
: "${SIMLLM_CORE61_RETRY_RUN_ROOT:?SIMLLM_CORE61_RETRY_RUN_ROOT is required}"

module purge
module load gcc/12.3.0
module load cuda/12.9.1

venv_dir="$HOME/venvs/simllm-gh200lane-py311-vllm0271"
stage_dir="$SIMLLM_CORE61_RETRY_STAGE_ROOT"
run_dir="$SIMLLM_CORE61_RETRY_RUN_ROOT/decode-${SLURM_JOB_ID}"
hf_root="$HOME/simllm-data/gh200lane/hf-cache"

if [[ -e "$run_dir" ]]; then
  echo "refusing to overwrite $run_dir" >&2
  exit 1
fi
mkdir -p "$run_dir/markers"
exec > >(tee -a "$run_dir/job.log") 2>&1

export PATH="$venv_dir/bin:$PATH"
export HF_HOME="$hf_root"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_LOGGING_LEVEL=INFO
export CORE61_SCHEDULER_HOOK=1
export CORE61_ACTIVE_FILE="$run_dir/active-cell.json"
export CORE61_MARKER_DIR="$run_dir/markers"
export PYTHONPATH="$stage_dir/core61_depth_retry_hook${PYTHONPATH:+:$PYTHONPATH}"

capture="$stage_dir/core61_depth_retry_capture.py"
analyzer="$stage_dir/core61_depth_retry_analyze.py"

finalize() {
  local verdict=$1
  printf '%s\n' "$verdict" | tee "$run_dir/verdict.txt"
  date -Is | tee "$run_dir/finished_at.txt"
  find "$run_dir" -type f ! -name 'sha256.txt' -print0 \
    | sort -z | xargs -0 sha256sum > "$run_dir/sha256.txt"
}

common_args=(
  --model "$MODEL"
  --revision "$REVISION"
  --reduced-layers "$REDUCED_LAYERS"
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
  --max-model-len "$MAX_MODEL_LEN"
  --startup-max-num-batched-tokens "$STARTUP_MAX_NUM_BATCHED_TOKENS"
  --max-num-seqs "$MAX_NUM_SEQS"
  --batch-size "$BATCH_SIZE"
  --remote-kv-tokens "$REMOTE_KV_TOKENS"
  --active-file "$CORE61_ACTIVE_FILE"
  --marker-dir "$CORE61_MARKER_DIR"
  --alignment "$run_dir/alignment.json"
)

date -Is | tee "$run_dir/started_at.txt"
hostname | tee "$run_dir/hostname.txt"
uname -a | tee "$run_dir/uname.txt"
nvidia-smi -L | tee "$run_dir/nvidia_smi_list.txt"
nvidia-smi | tee "$run_dir/nvidia_smi.txt"
nsys --version | tee "$run_dir/nsys_version.txt"
"$venv_dir/bin/python" -m pip show vllm torch | tee "$run_dir/package_versions.txt"
"$venv_dir/bin/python" -m pip check \
  > "$run_dir/pip_check_stdout.txt" 2> "$run_dir/pip_check_stderr.txt"
pip_status=$?
printf '%s\n' "$pip_status" > "$run_dir/pip_check_status.txt"
if [[ "$pip_status" != 0 ]]; then
  finalize "capture_state=BLOCKED: compute-node pip dependency check failed"
  exit 0
fi

sha256sum "$capture" "$analyzer" \
  "$stage_dir/core61_depth_retry_hook/sitecustomize.py" \
  > "$run_dir/harness_sha256.txt"

"$venv_dir/bin/python" "$capture" "${common_args[@]}" --phase align
align_status=$?
printf '%s\n' "$align_status" > "$run_dir/alignment_status.txt"
if [[ "$align_status" != 0 ]]; then
  finalize "capture_state=BLOCKED: exact KV alignment failed"
  exit 0
fi

(
  while true; do
    nvidia-smi \
      --query-gpu=timestamp,index,clocks.current.sm,clocks.current.memory,power.draw,temperature.gpu,memory.used,memory.total \
      --format=csv,noheader,nounits
    sleep 0.2
  done
) > "$run_dir/clocks.csv" 2>&1 &
clock_pid=$!
stop_clock_log() {
  kill "$clock_pid" 2>/dev/null || true
  wait "$clock_pid" 2>/dev/null || true
}
trap stop_clock_log EXIT

nsys profile \
  --output="$run_dir/profile" \
  --force-overwrite=false \
  --cuda-graph-trace=node \
  --cuda-event-trace=false \
  --trace=cuda,nvtx,cublas,cudnn \
  --trace-fork-before-exec=true \
  --sample=none \
  --cpuctxsw=none \
  --wait=all \
  --export=sqlite \
  "$venv_dir/bin/python" "$capture" "${common_args[@]}" \
    --phase profile --output "$run_dir/profile.json"
profile_status=$?
printf '%s\n' "$profile_status" > "$run_dir/profile_status.txt"
stop_clock_log
trap - EXIT
if [[ "$profile_status" != 0 ]]; then
  finalize "capture_state=BLOCKED: Nsight Systems model capture failed"
  exit 0
fi

find "$HF_HOME" -type f \
  \( -name '*.safetensors' -o -name '*.bin' -o -name '*.pt' -o -name '*.pth' -o -name '*.gguf' \) \
  -print > "$run_dir/weight_files.txt"
if [[ -s "$run_dir/weight_files.txt" ]]; then
  finalize "capture_state=VOID: model weight file appeared in the isolated cache"
  exit 0
fi

"$venv_dir/bin/python" "$analyzer" "$run_dir" \
  --output "$run_dir/measurement.json" > "$run_dir/analysis.log" 2>&1
analysis_status=$?
printf '%s\n' "$analysis_status" > "$run_dir/analysis_status.txt"
if [[ "$analysis_status" != 0 ]]; then
  finalize "capture_state=BLOCKED: exact decode boundary analysis failed"
  exit 0
fi

finalize "capture_state=CAPTURED: exact depth-8 batch-32 KV-2000 decode service"
