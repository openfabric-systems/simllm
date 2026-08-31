#!/bin/bash
# Record rank placement, driver identity, topology, NUMA binding and GDR inputs.

set -u

OUT_DIR="${1:?pass the rank identity output directory}"
RANK="${SLURM_PROCID:-0}"
HOST="$(hostname)"
OUTPUT="${OUT_DIR}/rank_$(printf '%02d' "${RANK}")_${HOST}.txt"
mkdir -p "${OUT_DIR}"

exec > "${OUTPUT}" 2>&1
echo "evidence_class=hardware-capture"
echo "host=${HOST}"
echo "date=$(date -Is)"
echo "SLURM_PROCID=${SLURM_PROCID:-unset}"
echo "SLURM_LOCALID=${SLURM_LOCALID:-unset}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-unset}"
echo "=== driver and GPU identity ==="
nvidia-smi
nvidia-smi --query-gpu=index,uuid,pci.bus_id,driver_version,name --format=csv
echo "=== CUDA version ==="
nvcc --version
echo "=== NVIDIA topology ==="
nvidia-smi topo -m
echo "=== NUMA binding ==="
numactl --show 2>&1 || true
taskset -pc $$ 2>&1 || true
echo "=== CPU identity ==="
lscpu
echo "=== host interfaces ==="
ip -br link 2>&1 || true
echo "=== Cassini inventory ==="
lspci 2>&1 | grep -i cassini || true
