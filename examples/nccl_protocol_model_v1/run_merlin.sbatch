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
STAGE="${NCCL_MODEL_STAGE:?set NCCL_MODEL_STAGE}"
DATA="${NCCL_MODEL_DATA:?set NCCL_MODEL_DATA}"
ARCH="${NCCL_MODEL_ARCH:?set NCCL_MODEL_ARCH}"
LIBRARY="${NCCL_MODEL_LIBRARY:?set NCCL_MODEL_LIBRARY}"
PRIOR="${NCCL_MODEL_PRIOR:?set NCCL_MODEL_PRIOR}"
OUT="${DATA}/${ARCH}/${SLURM_JOB_ID}"
mkdir -p "${OUT}/bin"
module purge
case "${ARCH}" in
 a100) module load cuda/12.2.2; TARGET=80; LIB_SHA=dba12e429fe11268b895d0531ba96a7f679f35227d5b1ec77c5febbcd02281bd ;;
 gh200) module load cuda/12.9.1; TARGET=90; LIB_SHA=1dbd9a78c092f7b20e597793ca21622644ba1d1baba8a82292605e808f276dd9 ;;
 *) exit 2 ;;
esac
export LD_LIBRARY_PATH="${LIBRARY}/lib:${LD_LIBRARY_PATH:-}"
cp "${STAGE}/"* "${OUT}/bin/"
printf '%s\n' "${NCCL_MODEL_COMMIT:?set NCCL_MODEL_COMMIT}" > "${OUT}/candidate_commit.txt"
date -Is > "${OUT}/started.txt"
hostname > "${OUT}/hostname.txt"
nvcc --version > "${OUT}/nvcc_version.txt"
nvidia-smi --query-gpu=index,uuid,name,driver_version,mig.mode.current --format=csv > "${OUT}/gpu_identity.csv"
nvidia-smi topo -m > "${OUT}/topology.txt" 2>&1
nvidia-smi nvlink -s > "${OUT}/nvlink_status.txt" 2>&1
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv > "${OUT}/processes_before.csv"
test "$(tail -n +2 "${OUT}/processes_before.csv" | wc -l)" -eq 0
sha256sum "${LIBRARY}/lib/libnccl.so.2" > "${OUT}/library.sha256"
test "$(cut -d ' ' -f1 "${OUT}/library.sha256")" = "${LIB_SHA}"
cp "${PRIOR}/build/all_reduce_perf" "${OUT}/bin/"
cp "${PRIOR}/legacy_dense.cu" "${OUT}/bin/legacy_fresh.cu"
python3 - "${OUT}/bin" <<'PY'
from pathlib import Path
import json
import sys
root=Path(sys.argv[1]);p=root/'legacy_fresh.cu';s=p.read_text()
old='for (size_t b = 262144; b <= 4194304; b += 16384) sizes.push_back(b);'
config=json.loads((root/'expectations.json').read_text())
grid=config['fresh_payloads_bytes']
new=f'for (size_t b = {grid[0]}; b <= {grid[-1]}; b += 65536) sizes.push_back(b);'
assert s.count(old)==1
p.write_text(s.replace(old,new))
PY
for program in legacy_fresh method_probe visibility_probe; do
 nvcc -O3 -std=c++17 -arch="sm_${TARGET}" -I"${LIBRARY}/include" \
 "${OUT}/bin/${program}.cu" -L"${LIBRARY}/lib" -lnccl -lcublas -lpthread \
 -o "${OUT}/bin/${program}" > "${OUT}/${program}_build.log" 2>&1
done
sha256sum "${OUT}/bin/"* > "${OUT}/staged.sha256"
ldd "${OUT}/bin/all_reduce_perf" > "${OUT}/linked_libraries.txt"
nvidia-smi --query-gpu=index,clocks.sm,clocks.mem,power.draw,temperature.gpu --format=csv --loop=5 > "${OUT}/state_samples.csv" 2>&1 &
MONITOR=$!
trap 'kill "${MONITOR}" 2>/dev/null || true' EXIT
python3 "${OUT}/bin/run_capture.py" --stage "${OUT}/bin" --output "${OUT}/raw" > "${OUT}/campaign.log" 2>&1
kill "${MONITOR}" 2>/dev/null || true
wait "${MONITOR}" 2>/dev/null || true
trap - EXIT
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv > "${OUT}/processes_after.csv"
test "$(tail -n +2 "${OUT}/processes_after.csv" | wc -l)" -eq 0
date -Is > "${OUT}/completed.txt"
