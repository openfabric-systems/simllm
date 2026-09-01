#!/bin/bash
# Collect one rank's allowlisted TRAF-81 environment and physical guards.

set -eu

out_root="${1:?output root required}"
phase="${2:?phase required}"
rank="${SLURM_PROCID:?SLURM_PROCID is required}"
local_rank="${SLURM_LOCALID:?SLURM_LOCALID is required}"
rank_root="${out_root}/rank_$(printf '%02d' "${rank}")_$(hostname)"
mkdir -p "${rank_root}"
exec >"${rank_root}/guards_${phase}.txt" 2>&1

visible="${CUDA_VISIBLE_DEVICES:?CUDA_VISIBLE_DEVICES is required}"
IFS=',' read -r -a visible_devices <<<"${visible}"
if [ "${#visible_devices[@]}" -eq 0 ]; then
  echo "guard_error=no visible GPU selector"
  exit 2
fi
device_index=$((local_rank % ${#visible_devices[@]}))
selector="${visible_devices[${device_index}]}"

echo "host=$(hostname)"
echo "date=$(date -Is)"
echo "slurm_procid=${SLURM_PROCID:-unset}"
echo "slurm_localid=${SLURM_LOCALID:-unset}"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
echo "assigned_gpu_selector=${selector}"
echo "=== target identity ==="
nvidia-smi -L
nvidia-smi --query-gpu=uuid,name,pci.bus_id --format=csv,noheader |
  sed 's/^/gpu=/'
assigned_gpu=$(nvidia-smi --id="${selector}" \
  --query-gpu=uuid,name,pci.bus_id --format=csv,noheader,nounits)
if [ "$(printf '%s\n' "${assigned_gpu}" | grep -c .)" -ne 1 ]; then
  echo "guard_error=assigned selector did not resolve to exactly one GPU"
  exit 2
fi
echo "assigned_gpu=${assigned_gpu}"
echo "=== topology ==="
nvidia-smi topo -m
echo "=== clocks and state ==="
nvidia-smi --query-gpu=uuid,clocks.sm,clocks.mem,power.draw,temperature.gpu \
  --format=csv,noheader
echo "=== compute processes ==="
nvidia-smi --id="${selector}" \
  --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
  --format=csv,noheader,nounits || true
echo "=== high-speed ports ==="
port_count=$(lspci 2>/dev/null | grep -ic cassini || true)
echo "cassini_port_count=${port_count}"
lspci 2>/dev/null | grep -iE 'cassini|mellanox|infiniband' || true
echo "=== infiniband class ==="
ib_count=$(find /sys/class/infiniband -mindepth 1 -maxdepth 1 2>/dev/null |
  wc -l)
echo "ib_device_count=${ib_count}"
find /sys/class/infiniband -mindepth 1 -maxdepth 1 -printf 'ib_device=%f\n' \
  2>/dev/null || true
