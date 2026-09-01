#!/bin/bash
#SBATCH --job-name=traf73-identify
#SBATCH --partition=a100-hourly
#SBATCH --account=merlin
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --time=00:30:00
#SBATCH --signal=B:TERM@60
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

# One resumable TRAF-73 hypothesis family. Submit H1, H2 and H3 separately,
# only after the preceding family has completed and passed its digest audit.

set -u

STAGE="${TRAF73_STAGE_ROOT:?set TRAF73_STAGE_ROOT}"
DATA="${TRAF73_DATA_ROOT:?set TRAF73_DATA_ROOT}"
EXPECTED_HEAD="${TRAF73_EXPECTED_HEAD:?set TRAF73_EXPECTED_HEAD}"
FAMILY="${TRAF73_FAMILY:?set TRAF73_FAMILY to h1, h2 or h3}"
FREEZE="a17b9e298d11a4a6ba92b382121c15a5a48f8100b6f343893260419b1d3382f6"
CUDA_MODULE="${TRAF73_CUDA_MODULE:-cuda/12.9.1}"
PYTHON_MODULE="${TRAF73_PYTHON_MODULE:-Python/3.10.16}"
PYTHON="${TRAF73_PYTHON:-python3}"
STOP_ROOT="${DATA}/${FREEZE}/scheduler-stops"
CHILD_PID=""

case "${FAMILY}" in
  h1|h2|h3) ;;
  *) echo "TRAF-73 FATAL: invalid family ${FAMILY}" >&2; exit 2 ;;
esac

mkdir -p "${STOP_ROOT}"

stop_cell() {
  signal_name="$1"
  {
    echo "job_id=${SLURM_JOB_ID}"
    echo "family=${FAMILY}"
    echo "signal=${signal_name}"
    echo "date=$(date -Is)"
  } > "${STOP_ROOT}/${SLURM_JOB_ID}-${FAMILY}.txt"
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
  echo "TRAF-73 FATAL: $*" >&2
  exit 2
}

[ -f "${STAGE}/examples/nvlink_credit_arbitration_v1/aligned_expectations.json" ] \
  || fail "staged aligned expectations are missing"
[ "$(sha256sum "${STAGE}/examples/nvlink_credit_arbitration_v1/aligned_expectations.json" | awk '{print $1}')" = "${FREEZE}" ] \
  || fail "staged aligned expectations do not match the freeze"

module purge 2>/dev/null
module load "${CUDA_MODULE}" || fail "failed to load ${CUDA_MODULE}"
module load "${PYTHON_MODULE}" || fail "failed to load ${PYTHON_MODULE}"

BASE_SOURCE="${STAGE}/examples/a100_nvlink_packet_v2/nvlink_packet_lane.cu"
EXTENSION="${STAGE}/examples/nvlink_credit_arbitration_v1/producer_traf73.patch"
SHA_HEADER="${STAGE}/examples/a100_nvlink_packet_v2/sha256.h"
IMPLEMENTATION_SHA=$({ sha256sum "${BASE_SOURCE}"; sha256sum "${EXTENSION}"; sha256sum "${SHA_HEADER}"; } | sha256sum | awk '{print $1}')
BUILD_ROOT="${DATA}/${FREEZE}/build/${EXPECTED_HEAD}/${IMPLEMENTATION_SHA}"
BINARY="${BUILD_ROOT}/nvlink_packet_lane"
DERIVED_ROOT="${BUILD_ROOT}/derived-source"
SOURCE="${DERIVED_ROOT}/examples/a100_nvlink_packet_v2/nvlink_packet_lane.cu"
mkdir -p "${BUILD_ROOT}"

exec 9>"${BUILD_ROOT}/compile.lock"
flock -x 9
if [ ! -x "${BINARY}" ]; then
  if [ ! -f "${SOURCE}" ]; then
    PART_ROOT="${BUILD_ROOT}/derived-source.${SLURM_JOB_ID}.part"
    mkdir -p "${PART_ROOT}/examples/a100_nvlink_packet_v2"
    cp "${BASE_SOURCE}" "${PART_ROOT}/examples/a100_nvlink_packet_v2/nvlink_packet_lane.cu"
    git -C "${PART_ROOT}" apply --whitespace=nowarn "${EXTENSION}" \
      || fail "TRAF-73 producer extension did not apply to the frozen lineage"
    mv "${PART_ROOT}" "${DERIVED_ROOT}"
  fi
  TEMP_BINARY="${BUILD_ROOT}/nvlink_packet_lane.${SLURM_JOB_ID}.part"
  nvcc -std=c++17 -O3 -arch=sm_80 \
    -I"${STAGE}/examples/a100_nvlink_packet_v2" \
    "${SOURCE}" -lnvidia-ml -o "${TEMP_BINARY}" \
    || fail "CUDA producer build failed"
  chmod 755 "${TEMP_BINARY}"
  mv "${TEMP_BINARY}" "${BINARY}"
  sha256sum "${BASE_SOURCE}" "${EXTENSION}" "${SOURCE}" "${SHA_HEADER}" \
    "${BINARY}" > "${BUILD_ROOT}/digests.txt"
fi
flock -u 9

sleep 2

"${PYTHON}" "${STAGE}/examples/nvlink_credit_arbitration_v1/run_hardware_campaign.py" \
  --family "${FAMILY}" \
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
