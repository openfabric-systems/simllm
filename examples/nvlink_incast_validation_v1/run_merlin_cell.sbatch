#!/bin/bash
#SBATCH --job-name=traf73-nvincast
#SBATCH --partition=a100-hourly
#SBATCH --account=merlin
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --time=00:12:00
#SBATCH --signal=B:TERM@60
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

# One short, paced and resumable TRAF-74 cell. Slurm owns the submitted job,
# so loss of the staging SSH session does not orphan a local child. Signals
# stop the producer, write one stop record and preserve prior attempts.

set -u

STAGE="${TRAF73_STAGE_ROOT:?set TRAF73_STAGE_ROOT}"
DATA="${TRAF73_DATA_ROOT:?set TRAF73_DATA_ROOT}"
EXPECTED_HEAD="${TRAF73_EXPECTED_HEAD:?set TRAF73_EXPECTED_HEAD}"
NCCL_ROOT="${TRAF73_NCCL_ROOT:?set TRAF73_NCCL_ROOT}"
FREEZE="9f50aadba0085a54e78c156d61837e4c7db19a498d8fef9c1aba7b32e0a163b4"
CUDA_MODULE="${TRAF73_CUDA_MODULE:-cuda/12.9.1}"
PYTHON_MODULE="${TRAF73_PYTHON_MODULE:-Python/3.10.16}"
PYTHON="${TRAF73_PYTHON:-python3}"
STOP_ROOT="${DATA}/${FREEZE}/scheduler-stops"
CHILD_PID=""

mkdir -p "${STOP_ROOT}"

stop_cell() {
  signal_name="$1"
  {
    echo "job_id=${SLURM_JOB_ID}"
    echo "signal=${signal_name}"
    echo "date=$(date -Is)"
  } > "${STOP_ROOT}/${SLURM_JOB_ID}.txt"
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
  echo "TRAF-74 FATAL: $*" >&2
  exit 2
}

[ -f "${STAGE}/examples/nvlink_incast_validation_v1/expectations.json" ] \
  || fail "staged expectations are missing"
[ "$(sha256sum "${STAGE}/examples/nvlink_incast_validation_v1/expectations.json" | awk '{print $1}')" = "${FREEZE}" ] \
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
  TEMP_BINARY="${BUILD_ROOT}/nvlink_packet_lane.${SLURM_JOB_ID}.part"
  nvcc -std=c++17 -O3 -arch=sm_80 -DSIMLLM_NVLINK_ENABLE_NCCL \
    -I"${NCCL_ROOT}/include" \
    "${SOURCE}" \
    -L"${NCCL_ROOT}/lib" -lnccl -lnvidia-ml \
    -Xlinker -rpath="${NCCL_ROOT}/lib" \
    -o "${TEMP_BINARY}" \
    || fail "CUDA/NCCL producer build failed"
  chmod 755 "${TEMP_BINARY}"
  mv "${TEMP_BINARY}" "${BINARY}"
  sha256sum "${SOURCE}" "${SHA_HEADER}" "${BINARY}" > "${BUILD_ROOT}/digests.txt"
fi
flock -u 9

# The cell is deliberately submitted once. This short pause avoids an
# immediate scheduler-side launch burst after allocation.
sleep 2

"${PYTHON}" "${STAGE}/examples/nvlink_incast_validation_v1/run_campaign.py" \
  --mode hardware \
  --binary "${BINARY}" \
  --output-root "${DATA}" \
  --expected-head "${EXPECTED_HEAD}" \
  --freeze-sha256 "${FREEZE}" &
CHILD_PID=$!
wait "${CHILD_PID}"
STATUS=$?
CHILD_PID=""
exit "${STATUS}"
