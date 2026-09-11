#!/bin/bash
#SBATCH --clusters=gmerlin7
#SBATCH --account=merlin
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:4
#SBATCH --time=00:55:00
#SBATCH --no-requeue
set -euo pipefail
STAGE="${NCCL_FIFO_STAGE:?set NCCL_FIFO_STAGE}"
DATA="${NCCL_FIFO_DATA:?set NCCL_FIFO_DATA}"
ARCH="${NCCL_FIFO_ARCH:?set NCCL_FIFO_ARCH}"
LIBRARY="${NCCL_FIFO_LIBRARY:?set NCCL_FIFO_LIBRARY}"
HEADERS="${NCCL_FIFO_PROFILER_HEADERS:?set NCCL_FIFO_PROFILER_HEADERS}"
OUT="${DATA}/${ARCH}/${SLURM_JOB_ID}"
mkdir -p "${OUT}/bin"
module purge
case "${ARCH}" in
 a100) module load cuda/12.8.1; TARGET=80 ;;
 gh200) module load cuda/12.9.1; TARGET=90 ;;
 *) exit 2 ;;
esac
export LD_LIBRARY_PATH="${LIBRARY}/lib:${LD_LIBRARY_PATH:-}"
cp "${STAGE}/"* "${OUT}/bin/"
printf '%s\n' "${NCCL_FIFO_FREEZE:?set NCCL_FIFO_FREEZE}" > "${OUT}/expectations_commit.txt"
date -Is > "${OUT}/started.txt"
hostname > "${OUT}/hostname.txt"
nvcc --version > "${OUT}/nvcc_version.txt"
nvidia-smi --query-gpu=index,uuid,name,driver_version,mig.mode.current --format=csv > "${OUT}/gpu_identity.csv"
nvidia-smi topo -m > "${OUT}/topology.txt" 2>&1
nvidia-smi nvlink -s > "${OUT}/nvlink_status.txt" 2>&1
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv > "${OUT}/processes_before.csv"
test "$(tail -n +2 "${OUT}/processes_before.csv" | wc -l)" -eq 0
sha256sum "${LIBRARY}/lib/libnccl.so.2" > "${OUT}/library.sha256"
python3 - "${OUT}" "${ARCH}" <<'PY'
from pathlib import Path
import json,sys
out=Path(sys.argv[1]);cfg=json.loads((out/'bin/hardware_expectations.json').read_text())
assert (out/'library.sha256').read_text().split()[0] == cfg['library_sha256'][sys.argv[2]]
PY
nvcc -O3 -std=c++17 -arch="sm_${TARGET}" -I"${LIBRARY}/include" \
 "${OUT}/bin/channel_probe.cu" -L"${LIBRARY}/lib" -lnccl -lcuda -lpthread \
 -o "${OUT}/bin/channel_probe" > "${OUT}/probe_build.log" 2>&1
nvcc -shared -std=c++17 -Xcompiler -fPIC -I"${LIBRARY}/include" -I"${HEADERS}" \
 "${OUT}/bin/channel_observer.cc" -o "${OUT}/bin/libnccl-channel-observer.so" > "${OUT}/observer_build.log" 2>&1
sha256sum "${OUT}/bin/"* > "${OUT}/staged.sha256"
ldd "${OUT}/bin/channel_probe" > "${OUT}/linked_libraries.txt"
# The allocation's default application clock is the only lock target. Failure
# is recorded; it never becomes an invented fixed-frequency qualification.
nvidia-smi --query-gpu=clocks.applications.graphics --format=csv,noheader,nounits > "${OUT}/requested_clock_mhz.txt"
LOCKED=0
CLOCK_VALUE=$(head -n 1 "${OUT}/requested_clock_mhz.txt")
if [[ "${CLOCK_VALUE}" =~ ^[0-9]+$ ]] && nvidia-smi -lgc "${CLOCK_VALUE},${CLOCK_VALUE}" > "${OUT}/clock_lock.log" 2>&1; then LOCKED=1; fi
nvidia-smi --query-gpu=index,clocks.sm,clocks.mem,power.draw,temperature.gpu --format=csv --loop=5 > "${OUT}/state_samples.csv" 2>&1 &
MONITOR=$!
cleanup() { kill "${MONITOR}" 2>/dev/null || true; if [ "${LOCKED}" = 1 ]; then nvidia-smi -rgc > "${OUT}/clock_restore.log" 2>&1 || true; fi; }
trap cleanup EXIT
python3 -u "${OUT}/bin/run_capture.py" --stage "${OUT}/bin" --output "${OUT}/raw" > "${OUT}/campaign.log" 2>&1
cleanup
trap - EXIT
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv > "${OUT}/processes_after.csv"
test "$(tail -n +2 "${OUT}/processes_after.csv" | wc -l)" -eq 0
date -Is > "${OUT}/completed.txt"
