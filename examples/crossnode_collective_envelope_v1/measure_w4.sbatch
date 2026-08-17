#!/bin/bash
#SBATCH --job-name=xnode-w4
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

# Optional best-effort width-4 cell: four nodes, one GPU each, which is the one
# width above 2 that this cluster can express with one rank per node. Its
# absence is declared in the freeze as unevaluated rather than failed, because
# two of the five A100 nodes sit under a course reservation.

export XNODE_CELL=w4
export XNODE_ARMS="default"
exec "${XNODE_STAGE_ROOT:?set XNODE_STAGE_ROOT}/run_lane.sh"
