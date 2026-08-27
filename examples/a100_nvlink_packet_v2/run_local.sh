#!/bin/sh
# Complete the local mock arm under the required bulk root.
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
DEVELOPMENT_ROOT=$(CDPATH= cd -- "${SCRIPT_DIR}/../../../.." && pwd)
TASK_ROOT=${TRAF70_LOCAL_ROOT:-${DEVELOPMENT_ROOT}/wave-runs/traf70}
BINARY=$("${SCRIPT_DIR}/build_local.sh" | tail -n 1)

exec python3 "${SCRIPT_DIR}/run_study.py" \
  --mode mock \
  --binary "${BINARY}" \
  --output-root "${TASK_ROOT}" \
  --all-cells \
  --pace-seconds 0.05
