#!/usr/bin/env bash
# Topology capture inside a rented provider container (an image with torch and NCCL present).
# Usage: bash capture_container.sh <output directory>
set -u
OUT="$1"; mkdir -p "$OUT"; cd "$OUT" || exit 1
export OMP_NUM_THREADS=8
{
  echo "hostname: $(hostname)"; echo "date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"; echo "kernel: $(uname -r)"
  echo "--- nvidia-smi -L ---"; nvidia-smi -L
  echo "--- nvidia-smi ---"; nvidia-smi
  echo "--- nvidia-smi topo -m ---"; nvidia-smi topo -m
  echo "--- nvidia-smi topo -p2p n ---"; nvidia-smi topo -p2p n
  echo "--- nvidia-smi nvlink -s ---"; nvidia-smi nvlink -s
  echo "--- nvidia-smi nvlink -c ---"; nvidia-smi nvlink -c 2>&1 | head -80
  echo "--- nvidia-smi nvlink -R (remote pci bus id per link) ---"; nvidia-smi nvlink -R 2>&1 | head -200
  echo "--- nvidia-smi nvlink -p (link pci info) ---"; nvidia-smi nvlink -p 2>&1 | head -200
  echo "--- nvidia-smi -q full ---"; nvidia-smi -q 2>&1 | grep -iE "Product Name|Board ID|Module ID|Serial Number|GPU UUID|Bus Id|Minor Number|Board Part|VBIOS|Driver Version" | head -120
  echo "--- pci devices class 0x0680 (nvswitch/bridge) ---"; for d in /sys/bus/pci/devices/*; do c=$(cat $d/class 2>/dev/null); v=$(cat $d/vendor 2>/dev/null); case "$c" in 0x0680*) echo "$(basename $d) class=$c vendor=$v device=$(cat $d/device 2>/dev/null) numa=$(cat $d/numa_node 2>/dev/null)";; esac; done
  echo "--- pci devices class 0x0604 (pcie switches) ---"; for d in /sys/bus/pci/devices/*; do c=$(cat $d/class 2>/dev/null); case "$c" in 0x0604*) echo "$(basename $d) vendor=$(cat $d/vendor) device=$(cat $d/device)";; esac; done | head -40
  echo "--- nvidia-smi -q -d PCI ---"; nvidia-smi -q -d PCI 2>&1 | head -160
  echo "--- nvidia-smi -q (GPU board/module ids) ---"; nvidia-smi -q 2>&1 | grep -iE "Board ID|Module ID|Serial|Product Name|Bus Id|Minor" | head -60
  echo "--- lspci nvidia/bridges/nic ---"; lspci 2>/dev/null | grep -iE "nvidia|mellanox|infiniband|ethernet|network|bridge" | head -80
  echo "--- lspci -tv ---"; lspci -tv 2>/dev/null | head -200
  echo "--- net devices ---"; ls /sys/class/net; ls /sys/class/infiniband 2>/dev/null
  echo "--- numa ---"; lscpu | grep -iE "numa|model name|socket|core"
  echo "--- nvidia pci devices with class and numa ---"
  for d in /sys/bus/pci/devices/*; do v=$(cat $d/vendor 2>/dev/null); if [ "$v" = "0x10de" ]; then echo "$(basename $d) class=$(cat $d/class) numa=$(cat $d/numa_node 2>/dev/null)"; fi; done
  echo "--- nvswitch devices ---"; ls /dev/nvidia-nvswitch* 2>/dev/null; nvidia-smi -q 2>&1 | grep -i "nvswitch" | head
} > node_inventory.txt 2>&1
python3 - <<'EOF' > nccl_env.txt 2>&1
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "nccl", torch.cuda.nccl.version())
print("devices", torch.cuda.device_count())
EOF
cat > nccl_dump.py <<'EOF'
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
EOF
N=$(nvidia-smi -L | wc -l)
export NCCL_TOPO_DUMP_FILE="$OUT/nccl_topo.xml" NCCL_GRAPH_DUMP_FILE="$OUT/nccl_graph.xml"
export NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,GRAPH,NET
python3 -m torch.distributed.run --standalone --nproc_per_node="$N" "$OUT/nccl_dump.py" > nccl_run.log 2>&1
echo "exit: $?" >> nccl_run.log
ls -la "$OUT" > listing.txt; echo CAPTURE_DONE >> listing.txt
