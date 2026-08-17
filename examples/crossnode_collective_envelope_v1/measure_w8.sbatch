#!/bin/bash
#SBATCH --job-name=xnode-w8
#SBATCH --partition=a100-hourly
#SBATCH --account=merlin
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --gres=gpu:4
#SBATCH --time=00:45:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

# Width-8 cell of the cross-node collective envelope: two nodes, four GPUs
# each, so the ring has six NVLink hops and two fabric hops. This is the shape a
# real two-node tensor-parallel-8 deployment has, and it is the shape the
# composed SGLang deployment study prices with a transferred intra-node
# constant. The eight GPUs are exactly the gpu_hourly per-user cap.

export XNODE_CELL=w8
export XNODE_ARMS="default"
exec "${XNODE_STAGE_ROOT:?set XNODE_STAGE_ROOT}/run_lane.sh"
