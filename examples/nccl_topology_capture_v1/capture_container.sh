#!/usr/bin/env bash
# Topology capture inside a rented provider container, i.e. the script that
# produced the tracked HGX B200 and HGX H200 fixtures. Writes into the
# directory given as the first argument; the optional second argument is the
# rank count of the NCCL dump and defaults to the visible device count.
#
# Everything here is read-only inventory: no timing, no allocation sweep. The
# caller runs it from the repository checkout, so paths stay relative.
set -u
OUT="${1:?usage: capture_container.sh <output dir> [ranks]}"
RANKS="${2:-0}"
mkdir -p "$OUT"

{
  echo "hostname: $(hostname)"
  echo "date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "kernel: $(uname -r)"
  echo "--- nvidia-smi -L ---"; nvidia-smi -L
  echo "--- nvidia-smi ---"; nvidia-smi
  echo "--- nvidia-smi topo -m ---"; nvidia-smi topo -m
  echo "--- nvidia-smi topo -p2p n ---"; nvidia-smi topo -p2p n
  echo "--- nvidia-smi nvlink -s ---"; nvidia-smi nvlink -s
  echo "--- nvidia-smi nvlink -c ---"; nvidia-smi nvlink -c
  echo "--- nvidia-smi nvlink -R (remote pci bus id per link) ---"; nvidia-smi nvlink -R
  echo "--- nvidia-smi nvlink -p (link pci info) ---"; nvidia-smi nvlink -p
  echo "--- nvidia-smi -q full ---"; nvidia-smi -q
  echo "--- pci devices class 0x0680 (nvswitch/bridge) ---"
  for d in /sys/bus/pci/devices/*; do
    c=$(cat "$d/class" 2>/dev/null)
    case "$c" in
      0x0680*) echo "$(basename "$d") vendor=$(cat "$d/vendor") device=$(cat "$d/device")" ;;
    esac
  done
  echo "--- pci devices class 0x0604 (pcie switches) ---"
  for d in /sys/bus/pci/devices/*; do
    c=$(cat "$d/class" 2>/dev/null)
    case "$c" in
      0x0604*) echo "$(basename "$d") vendor=$(cat "$d/vendor") device=$(cat "$d/device")" ;;
    esac
  done
  echo "--- nvidia-smi -q -d PCI ---"; nvidia-smi -q -d PCI
  echo "--- nvidia-smi -q (GPU board/module ids) ---"
  nvidia-smi -q | grep -E "Product Name|Serial Number|Minor Number|Board ID|Board Part Number|Chassis Serial Number|Module Id|Bus Id"
  echo "--- lspci nvidia/bridges/nic ---"
  lspci 2>/dev/null | grep -iE "nvidia|mellanox|bridge|infiniband|ethernet|network"
  echo "--- lspci -tv ---"; lspci -tv 2>/dev/null
  echo "--- infiniband devices ---"; ls /sys/class/infiniband 2>/dev/null
  echo "--- net devices ---"; ls /sys/class/net 2>/dev/null
  echo "--- numa ---"; lscpu | grep -iE "numa|model name|socket|core|thread"
  echo "--- nvidia pci devices with class and numa ---"
  for d in /sys/bus/pci/devices/*; do
    v=$(cat "$d/vendor" 2>/dev/null)
    if [ "$v" = "0x10de" ]; then
      echo "$(basename "$d") class=$(cat "$d/class") numa=$(cat "$d/numa_node")"
    fi
  done
  echo "--- nvswitch devices ---"; ls /dev/nvidia-nvswitch* 2>/dev/null
} > "$OUT/node_inventory.txt" 2>&1

python - <<'PYEOF' > "$OUT/nccl_env.txt" 2>&1
import torch

print("torch", torch.__version__, "cuda", torch.version.cuda, "nccl", torch.cuda.nccl.version())
print("devices", torch.cuda.device_count())
PYEOF

if [ "$RANKS" = "0" ]; then
  RANKS=$(python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo 1)
fi

cat > "$OUT/nccl_dump.py" <<'PYEOF'
import torch
import torch.distributed as dist


def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    x = torch.ones(1024, device="cuda") * (rank + 1)
    dist.all_reduce(x)
    torch.cuda.synchronize()
    if rank == 0:
        print("all_reduce sum per element:", float(x[0]))
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
PYEOF

NCCL_TOPO_DUMP_FILE="$OUT/nccl_topo.xml" \
NCCL_GRAPH_DUMP_FILE="$OUT/nccl_graph.xml" \
NCCL_DEBUG=INFO \
NCCL_DEBUG_SUBSYS=INIT,GRAPH,NET \
  python -m torch.distributed.run --standalone --nproc_per_node="$RANKS" \
  "$OUT/nccl_dump.py" > "$OUT/nccl_run.log" 2>&1
echo "exit: $?" >> "$OUT/nccl_run.log"

ls -la "$OUT" > "$OUT/listing.txt"
echo CAPTURE_DONE >> "$OUT/listing.txt"
