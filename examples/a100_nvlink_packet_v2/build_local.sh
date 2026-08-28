#!/bin/sh
# GPU-free compile check for the TRAF-70 producer harness.
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
DEVELOPMENT_ROOT=$(CDPATH= cd -- "${SCRIPT_DIR}/../../../.." && pwd)
TASK_ROOT=${TRAF70_LOCAL_ROOT:-${DEVELOPMENT_ROOT}/wave-runs/traf70}
BUILD_ROOT="${TASK_ROOT}/local-build"
BINARY="${BUILD_ROOT}/nvlink_packet_lane_mock"

mkdir -p "${BUILD_ROOT}"
"${CXX:-c++}" -x c++ -std=c++17 -O2 -Wall -Wextra -Wpedantic \
  -DSIMLLM_NVLINK_MOCK \
  "${SCRIPT_DIR}/nvlink_packet_lane.cu" \
  -o "${BINARY}"
sha256sum "${SCRIPT_DIR}/nvlink_packet_lane.cu" "${SCRIPT_DIR}/sha256.h" "${BINARY}"
python3 "${SCRIPT_DIR}/run_study.py" \
  --mode mock \
  --binary "${BINARY}" \
  --check-only

echo "${BINARY}"
