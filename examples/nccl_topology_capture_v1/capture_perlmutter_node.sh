#!/usr/bin/env bash
# Topology capture on one Perlmutter GPU node. Writes into the directory given
# as the first argument (absolute path on $PSCRATCH).
set -u
OUT="$1"
mkdir -p "$OUT"
cd "$OUT" || exit 1
export OMP_NUM_THREADS=8
export SLURM_CPU_BIND=cores
{
  echo "hostname: $(hostname)"
  echo "date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "slurm job: ${SLURM_JOB_ID:-unset}"
  echo "kernel: $(uname -r)"
  echo "--- nvidia-smi -L ---"; nvidia-smi -L
  echo "--- nvidia-smi ---"; nvidia-smi
  echo "--- nvidia-smi topo -m ---"; nvidia-smi topo -m
  echo "--- nvidia-smi topo -p2p n ---"; nvidia-smi topo -p2p n
  echo "--- nvidia-smi nvlink -s ---"; nvidia-smi nvlink -s
  echo "--- nvidia-smi nvlink -c (capabilities) ---"; nvidia-smi nvlink -c 2>&1 | head -80
  echo "--- nvidia-smi -q (pci + nvlink sections) ---"; nvidia-smi -q -d PCI 2>&1 | head -120
  echo "--- lspci nvidia/mellanox/cray/nic ---"; lspci 2>/dev/null | grep -iE "nvidia|mellanox|cray|infiniband|ethernet|network"
  echo "--- lspci -tv (top) ---"; lspci -tv 2>/dev/null | head -150
  echo "--- ibdev / net devices ---"; ls /sys/class/infiniband 2>/dev/null; ls /sys/class/net; cat /sys/class/net/*/device/numa_node 2>/dev/null | head
  echo "--- cxi (slingshot) devices ---"; ls /sys/class/cxi 2>/dev/null; ls /dev/cxi* 2>/dev/null
  echo "--- numa ---"; lscpu | grep -iE "numa|model name|socket|core"
  echo "--- gpu numa ---"; for d in /sys/bus/pci/devices/*; do v=$(cat $d/vendor 2>/dev/null); if [ "$v" = "0x10de" ]; then echo "$(basename $d) class=$(cat $d/class) numa=$(cat $d/numa_node)"; fi; done
} > node_inventory.txt 2>&1

# NCCL topology dump via a 4-rank torch.distributed init (module pytorch).
module load pytorch 2>/dev/null || module load python 2>/dev/null
python - <<'EOF' > nccl_env.txt 2>&1
import torch, sys
print("torch", torch.__version__, "cuda", torch.version.cuda, "nccl", torch.cuda.nccl.version())
print("devices", torch.cuda.device_count())
EOF
cat > nccl_dump.py <<'EOF'
import os
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
export NCCL_TOPO_DUMP_FILE="$OUT/nccl_topo.xml"
export NCCL_GRAPH_DUMP_FILE="$OUT/nccl_graph.xml"
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,GRAPH,NET
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29517
python -m torch.distributed.run --standalone --nproc_per_node=4 "$OUT/nccl_dump.py" > nccl_run.log 2>&1
echo "exit: $?" >> nccl_run.log
ls -la "$OUT" > listing.txt
echo CAPTURE_DONE >> listing.txt
