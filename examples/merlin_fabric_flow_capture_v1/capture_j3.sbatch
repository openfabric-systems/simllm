#!/bin/bash
#SBATCH --job-name=w18b-j3-join
#SBATCH --partition=a100-hourly
#SBATCH --account=merlin
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

# j3-join: flows join one at a time on a fixed 60-second cadence against an
# established flow, the dcqcn_micro join-study shape. 240-second window,
# offsets 0, 60, 120 (freeze 1).

export W18_CELL=j3-join
export W18_WINDOW_S=240
export W18_OFFSETS=0,60,120
export W18_CUDA_MODULE=cuda/12.2.2
export W18_ARCH_FLAG=sm_80
exec "${W18_STAGE_ROOT:?set W18_STAGE_ROOT}/run_capture.sh"
