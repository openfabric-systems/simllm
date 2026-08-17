#!/bin/bash
#SBATCH --job-name=w18b-i4-incast
#SBATCH --partition=a100-hourly
#SBATCH --account=merlin
#SBATCH --nodes=5
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

# i4-incast, BEST EFFORT: four source nodes converge on one destination,
# which needs every A100 node on the cluster, including gpu104 under the
# psicourse01 FLEX reservation until 2026-08-19 08:00. If it never
# schedules its relations are unevaluated (freeze 1).

export W18_CELL=i4-incast
export W18_WINDOW_S=180
export W18_OFFSETS=0,0,0,0
export W18_CUDA_MODULE=cuda/12.2.2
export W18_ARCH_FLAG=sm_80
exec "${W18_STAGE_ROOT:?set W18_STAGE_ROOT}/run_capture.sh"
