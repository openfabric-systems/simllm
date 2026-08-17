#!/bin/bash
#SBATCH --job-name=xnode-w4x2
#SBATCH --partition=a100-hourly
#SBATCH --account=merlin
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:2
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

# POST-SPECIFIED exploratory cell, added after the freeze and deliberately
# unscored.
#
# The frozen width-8 cell needs four free GPUs on each of two nodes, which on
# this cluster means two whole nodes drained. Two of the five A100 nodes sit
# under a course reservation and the remaining three carry multi-hour jobs from
# other users, so the scheduler estimated the width-8 allocation a full day out.
#
# This cell asks for two GPUs on each of two nodes instead. That is width 4 on a
# ring with two NVLink hops and two fabric hops, i.e. it still has the mixed
# structure the frozen width-8 cell was chosen for, at a quarter of the GPU
# demand. It was added after the freeze, so no relation in expectations.md is
# scored against it and none of its numbers may enter the scored denominator.
# It is reported as post-specified evidence about the shape of a mixed ring.
#
# The frozen width-8 job stays queued. If it lands, it is the scored cell and
# this one remains a reported side observation.

export XNODE_CELL=w4x2
export XNODE_ARMS="default"
exec "${XNODE_STAGE_ROOT:?set XNODE_STAGE_ROOT}/run_lane.sh"
