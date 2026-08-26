#!/bin/bash
#SBATCH --job-name=traf65-nvlink
#SBATCH --partition=a100-hourly
#SBATCH --account=merlin
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --time=00:06:00
#SBATCH --array=0-85%1
#SBATCH --signal=B:TERM@60
#SBATCH --output=%x-%A_%a.out
#SBATCH --error=%x-%A_%a.err

# One short, resumable TRAF-65 cell. The %1 array limit prevents overlapping
# four-GPU occupancy. A submitted batch job is independent of the SSH session;
# HUP, INT, and TERM still stop the child and leave a resumable STOPPED record.

set -u

STAGE="${TRAF65_STAGE_ROOT:?set TRAF65_STAGE_ROOT}"
DATA="${TRAF65_DATA_ROOT:?set TRAF65_DATA_ROOT}"
EXPECTED_HEAD="${TRAF65_EXPECTED_HEAD:?set TRAF65_EXPECTED_HEAD}"
NCCL_ROOT="${TRAF65_NCCL_ROOT:?set TRAF65_NCCL_ROOT}"
FREEZE="212a7a26f54e444c9b18f1e528bd0d00b5a28e4f9e005b0dc137f477ad642571"
CUDA_MODULE="${TRAF65_CUDA_MODULE:-cuda/12.9.1}"
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
  echo "TRAF-65 FATAL: $*" >&2
  exit 2
}

[ -f "${STAGE}/examples/a100_nvlink_packet_v1/expectations.json" ] \
  || fail "staged expectations are missing"
[ "$(sha256sum "${STAGE}/examples/a100_nvlink_packet_v1/expectations.json" | awk '{print $1}')" = "${FREEZE}" ] \
  || fail "staged expectations digest does not match the freeze"
[ -f "${NCCL_ROOT}/include/nccl.h" ] || fail "NCCL headers are missing"
[ -f "${NCCL_ROOT}/lib/libnccl.so" ] || fail "NCCL shared library is missing"

module purge 2>/dev/null
module load "${CUDA_MODULE}" || fail "failed to load ${CUDA_MODULE}"

SOURCE="${STAGE}/examples/a100_nvlink_packet_v1/nvlink_packet_lane.cu"
SOURCE_SHA=$(sha256sum "${SOURCE}" | awk '{print $1}')
BUILD_ROOT="${DATA}/${FREEZE}/build/${EXPECTED_HEAD}/${SOURCE_SHA}"
BINARY="${BUILD_ROOT}/nvlink_packet_lane"
mkdir -p "${BUILD_ROOT}"

exec 9>"${BUILD_ROOT}/compile.lock"
flock -x 9
if [ ! -x "${BINARY}" ]; then
  TEMP_BINARY="${BUILD_ROOT}/nvlink_packet_lane.${SLURM_JOB_ID}.${CELL_INDEX}.part"
  nvcc -std=c++17 -O3 -arch=sm_80 -DSIMLLM_NVLINK_ENABLE_NCCL \
    -I"${NCCL_ROOT}/include" \
    "${SOURCE}" \
    -L"${NCCL_ROOT}/lib" -lnccl \
    -Xlinker -rpath="${NCCL_ROOT}/lib" \
    -o "${TEMP_BINARY}" \
    || fail "CUDA/NCCL harness build failed"
  chmod 755 "${TEMP_BINARY}"
  mv "${TEMP_BINARY}" "${BINARY}"
  sha256sum "${SOURCE}" "${BINARY}" > "${BUILD_ROOT}/digests.txt"
fi
flock -u 9

# A small task-indexed pause avoids a burst of scheduler-side starts when a
# remainder is resubmitted. The allocation remains short and only one cell can
# occupy the four GPUs at a time.
sleep "$((CELL_INDEX % 3))"

python3 "${STAGE}/examples/a100_nvlink_packet_v1/run_study.py" \
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
