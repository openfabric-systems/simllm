#!/bin/bash
#SBATCH --clusters=gmerlin7
#SBATCH --account=merlin
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:4
#SBATCH --time=00:50:00
#SBATCH --no-requeue

# Supply partition, output path and the task-local environment at submission.
set -euo pipefail
STAGE="${NCCL_TRANSITION_STAGE:?set NCCL_TRANSITION_STAGE}"
DATA="${NCCL_TRANSITION_DATA:?set NCCL_TRANSITION_DATA}"
ARCH="${NCCL_TRANSITION_ARCH:?set NCCL_TRANSITION_ARCH}"
NCCL_ROOT="${NCCL_TRANSITION_LIBRARY:?set NCCL_TRANSITION_LIBRARY}"
FREEZE="${NCCL_TRANSITION_FREEZE:?set NCCL_TRANSITION_FREEZE}"
IMPLEMENTATION="${NCCL_TRANSITION_IMPLEMENTATION:?set NCCL_TRANSITION_IMPLEMENTATION}"
OUT="${DATA}/${ARCH}/${SLURM_JOB_ID}"
mkdir -p "${OUT}"

module purge
case "${ARCH}" in
  a100) module load cuda/12.2.2; TARGET=80 ;;
  gh200) module load cuda/12.9.1; TARGET=90 ;;
  *) exit 2 ;;
esac
export LD_LIBRARY_PATH="${NCCL_ROOT}/lib:${LD_LIBRARY_PATH:-}"

date -Is > "${OUT}/started.txt"
hostname > "${OUT}/hostname.txt"
env | sort | grep -E '^(CUDA_VISIBLE_DEVICES|SLURM_JOB_ID|SLURM_JOB_NODELIST|SLURM_CPUS_PER_TASK|NCCL_)=' \
  > "${OUT}/allocation_environment.txt" || true
python3 --version > "${OUT}/python_version.txt" 2>&1
nvcc --version > "${OUT}/nvcc_version.txt"
git -C "${STAGE}/nccl-tests" rev-parse HEAD > "${OUT}/nccl_tests_commit.txt"
test "$(cat "${OUT}/nccl_tests_commit.txt")" = b4d5beebca8a76cf01335f724d154b9b9d394d96
nvidia-smi --query-gpu=index,uuid,name,driver_version,mig.mode.current --format=csv \
  > "${OUT}/gpu_identity.csv"
nvidia-smi topo -m > "${OUT}/topology.txt" 2>&1
nvidia-smi nvlink -s > "${OUT}/nvlink_status.txt" 2>&1
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv \
  > "${OUT}/processes_before.csv"
if [ "$(tail -n +2 "${OUT}/processes_before.csv" | wc -l)" -ne 0 ]; then
  echo 'FATAL: foreign GPU process present'; exit 3
fi
nvidia-smi --query-gpu=index,clocks.sm,clocks.mem,power.draw,temperature.gpu --format=csv \
  > "${OUT}/state_before.csv"
for config in /etc/nccl.conf "${HOME}/.nccl.conf"; do
  if [ -f "${config}" ]; then
    cat "${config}" >> "${OUT}/nccl_configuration.txt"
  fi
done
cp "${STAGE}/study/"* "${OUT}/"
cp "${STAGE}/${ARCH}_lane_b_multi_card.cu" "${OUT}/inherited_lane_b.cu"
sha256sum "${NCCL_ROOT}/lib/libnccl.so.2" "${NCCL_ROOT}/include/nccl.h" \
  > "${OUT}/library.sha256"
grep -E '^#define NCCL_(MAJOR|MINOR|PATCH|VERSION_CODE)' "${NCCL_ROOT}/include/nccl.h" \
  > "${OUT}/nccl_header_version.txt"
python3 "${OUT}/make_legacy_dense.py" "${OUT}/inherited_lane_b.cu" \
  "${OUT}/legacy_dense.cu" --expectations "${OUT}/expectations.json"

nvcc -O3 -std=c++17 -arch="sm_${TARGET}" -I"${NCCL_ROOT}/include" \
  "${OUT}/legacy_dense.cu" -L"${NCCL_ROOT}/lib" -lnccl -lcublas -lpthread \
  -o "${OUT}/legacy_dense" > "${OUT}/legacy_build.log" 2>&1
make -C "${STAGE}/nccl-tests/src" -j8 BUILDDIR="${OUT}/build" \
  NCCL_HOME="${NCCL_ROOT}" MPI=0 NVCC_GENCODE="-gencode=arch=compute_${TARGET},code=sm_${TARGET}" \
  "${OUT}/build/all_reduce_perf" > "${OUT}/nccl_tests_build.log" 2>&1
sha256sum "${OUT}/legacy_dense" "${OUT}/build/all_reduce_perf" > "${OUT}/binary.sha256"
ldd "${OUT}/build/all_reduce_perf" > "${OUT}/linked_libraries.txt"

# Low-frequency state samples do not enable per-collective instrumentation.
nvidia-smi --query-gpu=index,clocks.sm,clocks.mem,power.draw,temperature.gpu --format=csv \
  --loop=5 > "${OUT}/state_samples.csv" 2>&1 &
MONITOR_PID=$!
trap 'kill "${MONITOR_PID}" 2>/dev/null || true' EXIT
nvcc -shared -std=c++17 -Xcompiler -fPIC -I"${NCCL_ROOT}/include" \
  -I"${STAGE}/nccl-tests/src" "${OUT}/selection_observer.cc" \
  -o "${OUT}/libnccl-selection-observer.so" > "${OUT}/observer_build.log" 2>&1
sha256sum "${OUT}/libnccl-selection-observer.so" >> "${OUT}/binary.sha256"
if [ "${NCCL_TRANSITION_OBSERVER_ONLY:-0}" != 1 ]; then
  python3 "${OUT}/run_campaign.py" --expectations "${OUT}/expectations.json" \
    --binary "${OUT}/build/all_reduce_perf" --legacy "${OUT}/legacy_dense" \
    --output "${OUT}/raw" --architecture "${ARCH}" --freeze "${FREEZE}" \
    --implementation "${IMPLEMENTATION}" > "${OUT}/campaign.log" 2>&1
fi
python3 "${OUT}/run_observer.py" --expectations "${OUT}/expectations.json" \
  --binary "${OUT}/build/all_reduce_perf" --legacy "${OUT}/legacy_dense" \
  --plugin "${OUT}/libnccl-selection-observer.so" --output "${OUT}/observer" \
  --architecture "${ARCH}" --implementation "${IMPLEMENTATION}" \
  > "${OUT}/observer.log" 2>&1
kill "${MONITOR_PID}" 2>/dev/null || true
wait "${MONITOR_PID}" 2>/dev/null || true
trap - EXIT
nvidia-smi --query-gpu=index,clocks.sm,clocks.mem,power.draw,temperature.gpu --format=csv \
  > "${OUT}/state_after.csv"
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv \
  > "${OUT}/processes_after.csv"
if [ "$(tail -n +2 "${OUT}/processes_after.csv" | wc -l)" -ne 0 ]; then
  echo 'FATAL: foreign GPU process present after capture'; exit 4
fi
date -Is > "${OUT}/completed.txt"
echo "Completed ${ARCH} job ${SLURM_JOB_ID}"
