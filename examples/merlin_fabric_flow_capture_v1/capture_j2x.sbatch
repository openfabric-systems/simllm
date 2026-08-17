#!/bin/bash
#SBATCH --job-name=w18b-j2x-join
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

# POST-SPECIFIED exploratory cell, added after freeze 1 and deliberately
# unscored.
#
# The frozen j3-join cell needs one free GPU on each of four nodes, and at
# submission time the scheduler estimated that allocation two days out,
# behind the psicourse01 reservation window. This cell runs the same
# join-study shape at three nodes: one established flow and one flow
# joining at 60 seconds, 180-second window. No relation in expectations.md
# is scored against it and none of its numbers may enter the scored
# denominator. Its series is published as post-specified evidence about
# the join transient, clearly labeled. The frozen j3-join job stays
# queued; if it lands, it is the scored cell and this one remains a
# reported side observation.

export W18_CELL=j2x-join
export W18_WINDOW_S=180
export W18_OFFSETS=0,60
export W18_CUDA_MODULE=cuda/12.2.2
export W18_ARCH_FLAG=sm_80
exec "${W18_STAGE_ROOT:?set W18_STAGE_ROOT}/run_capture.sh"
