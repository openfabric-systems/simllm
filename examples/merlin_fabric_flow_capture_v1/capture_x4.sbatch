#!/bin/bash
#SBATCH --job-name=w18b-x4-incast
#SBATCH --partition=a100-hourly
#SBATCH --account=merlin
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --gres=gpu:4
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

# x4-incast, BEST EFFORT: all four GPUs of one node source flows into one
# GPU of the other node, so the four flows share one host's stack and
# ports. Needs two drained nodes, the allocation shape that never
# scheduled in wave 16. Ranks 0-3 are the sources (first node), rank 4 the
# destination, ranks 5-7 idle (freeze 1).

export W18_CELL=x4-incast
export W18_WINDOW_S=180
export W18_OFFSETS=0,0,0,0
export W18_DEST_RANK=4
export W18_SOURCES=0,1,2,3
export W18_CUDA_MODULE=cuda/12.2.2
export W18_ARCH_FLAG=sm_80
exec "${W18_STAGE_ROOT:?set W18_STAGE_ROOT}/run_capture.sh"
