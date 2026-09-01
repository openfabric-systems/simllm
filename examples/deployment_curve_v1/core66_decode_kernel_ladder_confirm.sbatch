#!/usr/bin/env bash
#SBATCH --account=merlin
#SBATCH --clusters=gmerlin7
#SBATCH --partition=gh-hourly
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --gres=gpu:1

set -uo pipefail

required=(
  KLADDER_HF_HOME
  KLADDER_MODEL_NAME
  KLADDER_MODEL_REVISION
  KLADDER_MODEL_SNAPSHOT
  KLADDER_OUTPUT_ROOT
  KLADDER_REPO_ROOT
  KLADDER_VENV_DIR
)
for variable in "${required[@]}"; do
  if [[ -z "${!variable:-}" ]]; then
    printf 'missing required environment variable: %s\n' "$variable" >&2
    exit 2
  fi
done

module purge
module load gcc/12.3.0
module load cuda/12.9.1

study_dir="$KLADDER_REPO_ROOT/examples/deployment_curve_v1"
run_dir="$KLADDER_OUTPUT_ROOT/core66-kladder-confirm-$SLURM_JOB_ID"
mkdir -p "$run_dir/markers"
exec > >(tee -a "$run_dir/job.log") 2>&1

export PATH="$KLADDER_VENV_DIR/bin:$PATH"
export PYTHONPATH="$study_dir/core66_decode_kernel_ladder_hook:$study_dir${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${KLADDER_HF_HOME:?missing required environment variable: KLADDER_HF_HOME}"
export HF_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_DISABLE_SHARED_EXPERTS_STREAM=1
export VLLM_LOGGING_LEVEL=INFO
export KLADDER_SCHEDULER_HOOK=1
export KLADDER_ACTIVE_FILE="$run_dir/active.json"
export KLADDER_MARKER_DIR="$run_dir/markers"

expectations="$study_dir/core66_decode_kernel_ladder_expectations.json"
alignment="$study_dir/core66_decode_kernel_ladder_alignment.json"
capture="$run_dir/capture.json"
profile="$run_dir/profile.sqlite"

find -L "$KLADDER_MODEL_SNAPSHOT" -maxdepth 1 -type f \
  \( -name '*.safetensors' -o -name '*.bin' -o -name '*.pt' \
  -o -name '*.pth' -o -name '*.gguf' \) -print | sort \
  > "$run_dir/weights_before.txt"

started=$(date +%s)
nsys profile \
  --output="$run_dir/profile" \
  --force-overwrite=false \
  --cuda-graph-trace=node \
  --cuda-event-trace=false \
  --trace=cuda,nvtx,cublas \
  --trace-fork-before-exec=true \
  --sample=none \
  --cpuctxsw=none \
  --wait=all \
  --export=sqlite \
  "$KLADDER_VENV_DIR/bin/python" \
    "$study_dir/core66_decode_kernel_ladder_capture.py" \
    --model "$KLADDER_MODEL_NAME" \
    --revision "$KLADDER_MODEL_REVISION" \
    --expectations "$expectations" \
    --alignment "$alignment" \
    --active-file "$run_dir/active.json" \
    --marker "$run_dir/markers/exact-decode-rank0.json" \
    --output "$capture"
process_status=$?
finished=$(date +%s)

find -L "$KLADDER_MODEL_SNAPSHOT" -maxdepth 1 -type f \
  \( -name '*.safetensors' -o -name '*.bin' -o -name '*.pt' \
  -o -name '*.pth' -o -name '*.gguf' \) -print | sort \
  > "$run_dir/weights_after.txt"

printf '%s\n' "$process_status" > "$run_dir/process_status.txt"
printf '%s\n' "$((finished - started))" > "$run_dir/elapsed_seconds.txt"
commit=$(git -C "$KLADDER_REPO_ROOT" rev-parse HEAD)

score_status=1
if [[ "$process_status" == 0 && -f "$profile" && -f "$capture" ]]; then
  "$KLADDER_VENV_DIR/bin/python" \
    "$study_dir/core66_decode_kernel_ladder_score.py" \
    --sqlite "$profile" \
    --capture "$capture" \
    --expectations "$expectations" \
    --weights-before "$run_dir/weights_before.txt" \
    --weights-after "$run_dir/weights_after.txt" \
    --process-status "$run_dir/process_status.txt" \
    --repo-root "$KLADDER_REPO_ROOT" \
    --commit "$commit" \
    --output "$run_dir/analysis.json"
  score_status=$?
fi
printf '%s\n' "$score_status" > "$run_dir/score_status.txt"

if [[ "$process_status" != 0 ]]; then
  printf '%s\n' 'BLOCKED: confirmatory capture process failed' > "$run_dir/verdict.txt"
elif [[ "$score_status" != 0 ]]; then
  printf '%s\n' 'BLOCKED: confirmatory scorer failed' > "$run_dir/verdict.txt"
else
  "$KLADDER_VENV_DIR/bin/python" -c \
    'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["status"])' \
    "$run_dir/analysis.json" > "$run_dir/verdict.txt"
fi

sha256sum \
  "$study_dir/core66_decode_kernel_ladder_expectations.json" \
  "$study_dir/core66_decode_kernel_ladder_alignment.json" \
  "$study_dir/core66_decode_kernel_ladder_capture.py" \
  "$study_dir/core66_decode_kernel_ladder_score.py" \
  "$study_dir/core66_decode_kernel_ladder_hook/sitecustomize.py" \
  > "$run_dir/harness.sha256"
find "$run_dir" -maxdepth 2 -type f ! -name 'files.sha256' -print0 \
  | sort -z | xargs -0 sha256sum > "$run_dir/files.sha256"
