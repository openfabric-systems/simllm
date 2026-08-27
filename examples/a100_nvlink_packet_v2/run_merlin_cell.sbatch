#!/bin/bash
#SBATCH --job-name=traf70-nvlink
#SBATCH --partition=a100-hourly
#SBATCH --account=merlin
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --time=00:12:00
#SBATCH --array=0-85%1
#SBATCH --signal=B:TERM@60
#SBATCH --output=%x-%A_%a.out
#SBATCH --error=%x-%A_%a.err

# One short, resumable TRAF-70 cell. The %1 array limit prevents overlapping
# four-GPU occupancy. A submitted batch job is independent of the SSH session;
# HUP, INT, and TERM still stop the child and leave a resumable STOPPED record.

set -u

STAGE="${TRAF70_STAGE_ROOT:?set TRAF70_STAGE_ROOT}"
DATA="${TRAF70_DATA_ROOT:?set TRAF70_DATA_ROOT}"
EXPECTED_HEAD="${TRAF70_EXPECTED_HEAD:?set TRAF70_EXPECTED_HEAD}"
NCCL_ROOT="${TRAF70_NCCL_ROOT:?set TRAF70_NCCL_ROOT}"
FREEZE="f0ab026e054873a56614af63ab3a7ae3219dc0b045423808cb41522910fa6da6"
CUDA_MODULE="${TRAF70_CUDA_MODULE:-cuda/12.9.1}"
PYTHON_MODULE="${TRAF70_PYTHON_MODULE:-Python/3.10.16}"
PYTHON="${TRAF70_PYTHON:-python3}"
CELL_INDEX="${SLURM_ARRAY_TASK_ID:?Slurm array task ID is required}"
STOP_ROOT="${DATA}/${FREEZE}/scheduler-stops"
CHILD_PID=""

mkdir -p "${STOP_ROOT}"

stop_cell() {
  signal_name="$1"
  {
    echo "job_id=${SLURM_JOB_ID}"
    echo "array_task_id=${CELL_INDEX}"
    echo "signal=${signal_name}"
    echo "date=$(date -Is)"
  } > "${STOP_ROOT}/${SLURM_JOB_ID}_${CELL_INDEX}.txt"
  if [ -n "${CHILD_PID}" ]; then
    kill -TERM "${CHILD_PID}" 2>/dev/null || true
    wait "${CHILD_PID}" 2>/dev/null || true
  fi
  exit 143
}
trap 'stop_cell HUP' HUP
trap 'stop_cell INT' INT
trap 'stop_cell TERM' TERM

fail() {
  echo "TRAF-70 FATAL: $*" >&2
  exit 2
}

[ -f "${STAGE}/examples/a100_nvlink_packet_v2/expectations.json" ] \
  || fail "staged expectations are missing"
[ "$(sha256sum "${STAGE}/examples/a100_nvlink_packet_v2/expectations.json" | awk '{print $1}')" = "${FREEZE}" ] \
  || fail "staged expectations digest does not match the freeze"
[ -f "${NCCL_ROOT}/include/nccl.h" ] || fail "NCCL headers are missing"
[ -f "${NCCL_ROOT}/lib/libnccl.so" ] || fail "NCCL shared library is missing"

module purge 2>/dev/null
module load "${CUDA_MODULE}" || fail "failed to load ${CUDA_MODULE}"
module load "${PYTHON_MODULE}" || fail "failed to load ${PYTHON_MODULE}"

SOURCE="${STAGE}/examples/a100_nvlink_packet_v2/nvlink_packet_lane.cu"
SHA_HEADER="${STAGE}/examples/a100_nvlink_packet_v2/sha256.h"
IMPLEMENTATION_SHA=$({ sha256sum "${SOURCE}"; sha256sum "${SHA_HEADER}"; } | sha256sum | awk '{print $1}')
BUILD_ROOT="${DATA}/${FREEZE}/build/${EXPECTED_HEAD}/${IMPLEMENTATION_SHA}"
BINARY="${BUILD_ROOT}/nvlink_packet_lane"
mkdir -p "${BUILD_ROOT}"

exec 9>"${BUILD_ROOT}/compile.lock"
flock -x 9
if [ ! -x "${BINARY}" ]; then
  TEMP_BINARY="${BUILD_ROOT}/nvlink_packet_lane.${SLURM_JOB_ID}.${CELL_INDEX}.part"
  nvcc -std=c++17 -O3 -arch=sm_80 -DSIMLLM_NVLINK_ENABLE_NCCL \
    -I"${NCCL_ROOT}/include" \
    "${SOURCE}" \
    -L"${NCCL_ROOT}/lib" -lnccl -lnvidia-ml \
    -Xlinker -rpath="${NCCL_ROOT}/lib" \
    -o "${TEMP_BINARY}" \
    || fail "CUDA/NCCL harness build failed"
  chmod 755 "${TEMP_BINARY}"
  mv "${TEMP_BINARY}" "${BINARY}"
  sha256sum "${SOURCE}" "${SHA_HEADER}" "${BINARY}" > "${BUILD_ROOT}/digests.txt"
fi
flock -u 9

# A small task-indexed pause avoids a burst of scheduler-side starts when a
# remainder is resubmitted. The allocation remains short and only one cell can
# occupy the four GPUs at a time.
sleep "$((CELL_INDEX % 3))"

"${PYTHON}" "${STAGE}/examples/a100_nvlink_packet_v2/run_study.py" \
  --mode hardware \
  --binary "${BINARY}" \
  --output-root "${DATA}" \
  --array-index "${CELL_INDEX}" \
  --expected-head "${EXPECTED_HEAD}" \
  --freeze-sha256 "${FREEZE}" &
CHILD_PID=$!
wait "${CHILD_PID}"
STATUS=$?
CHILD_PID=""
exit "${STATUS}"
