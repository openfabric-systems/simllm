#!/bin/bash
#SBATCH --job-name=w18b-i2-incast
#SBATCH --partition=a100-hourly
#SBATCH --account=merlin
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

# i2-incast: two source nodes converge on one destination node, both flows
# greedy from T0, 180-second window (freeze 1).

export W18_CELL=i2-incast
export W18_WINDOW_S=180
export W18_OFFSETS=0,0
export W18_CUDA_MODULE=cuda/12.2.2
export W18_ARCH_FLAG=sm_80
exec "${W18_STAGE_ROOT:?set W18_STAGE_ROOT}/run_capture.sh"
