#!/bin/bash
#SBATCH --job-name=w18b-s1-stream
#SBATCH --partition=a100-hourly
#SBATCH --account=merlin
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

# s1-stream: one greedy 8 MiB-chunk flow, 300-second window. The sustained
# solo reference series and the live jitter verification (freeze 1).

export W18_CELL=s1-stream
export W18_WINDOW_S=300
export W18_OFFSETS=0
export W18_CUDA_MODULE=cuda/12.2.2
export W18_ARCH_FLAG=sm_80
exec "${W18_STAGE_ROOT:?set W18_STAGE_ROOT}/run_capture.sh"
