#!/bin/bash
#SBATCH --job-name=xnode-w2
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

# Width-2 cell of the cross-node collective envelope: two nodes, one GPU each,
# so a send and receive pair is exactly one cross-node hop with nothing else in
# it. Runs both declared interface arms.

export XNODE_CELL=w2
export XNODE_ARMS="default fournic"
exec "${XNODE_STAGE_ROOT:?set XNODE_STAGE_ROOT}/run_lane.sh"
