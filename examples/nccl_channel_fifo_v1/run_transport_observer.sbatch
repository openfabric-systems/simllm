#!/bin/bash
#SBATCH --clusters=gmerlin7
#SBATCH --account=merlin
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:4
#SBATCH --time=00:15:00
#SBATCH --no-requeue
set -euo pipefail
PRIOR="${NCCL_FIFO_PRIOR:?set completed timing allocation}"
STAGE="${NCCL_FIFO_STAGE:?set NCCL_FIFO_STAGE}"
DATA="${NCCL_FIFO_DATA:?set NCCL_FIFO_DATA}"
ARCH="${NCCL_FIFO_ARCH:?set NCCL_FIFO_ARCH}"
LIBRARY="${NCCL_FIFO_LIBRARY:?set NCCL_FIFO_LIBRARY}"
OUT="${DATA}/${ARCH}/${SLURM_JOB_ID}"
test -f "${PRIOR}/completed.txt"
mkdir -p "${OUT}/bin"
module purge
case "${ARCH}" in
 a100) module load cuda/12.8.1 ;;
 gh200) module load cuda/12.9.1 ;;
 *) exit 2 ;;
esac
export LD_LIBRARY_PATH="${LIBRARY}/lib:${LD_LIBRARY_PATH:-}"
cp "${STAGE}/capture_transports.py" "${STAGE}/transport_observer_expectations.md" "${OUT}/bin/"
cp "${PRIOR}/bin/channel_probe" "${PRIOR}/bin/libnccl-channel-observer.so" "${OUT}/bin/"
printf '%s\n' '34e8170f' > "${OUT}/expectations_commit.txt"
printf '%s\n' "${PRIOR}" > "${OUT}/source_capture.txt"
date -Is > "${OUT}/started.txt"
hostname > "${OUT}/hostname.txt"
nvcc --version > "${OUT}/nvcc_version.txt"
nvidia-smi --query-gpu=index,uuid,name,driver_version,mig.mode.current --format=csv > "${OUT}/gpu_identity.csv"
nvidia-smi topo -m > "${OUT}/topology.txt" 2>&1
sha256sum "${LIBRARY}/lib/libnccl.so.2" > "${OUT}/library.sha256"
for item in gpu_identity.csv topology.txt library.sha256 nvcc_version.txt; do cmp "${OUT}/${item}" "${PRIOR}/${item}"; done
sha256sum "${OUT}/bin/"* > "${OUT}/staged.sha256"
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv > "${OUT}/processes_before.csv"
test "$(tail -n +2 "${OUT}/processes_before.csv" | wc -l)" -eq 0
"${NCCL_FIFO_PYTHON:-python3}" -u "${OUT}/bin/capture_transports.py" --stage "${OUT}/bin" --output "${OUT}/raw" --architecture "${ARCH}" > "${OUT}/campaign.log" 2>&1
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv > "${OUT}/processes_after.csv"
test "$(tail -n +2 "${OUT}/processes_after.csv" | wc -l)" -eq 0
date -Is > "${OUT}/completed.txt"
