#!/bin/bash
#SBATCH --job-name=traf81-r8
#SBATCH --partition=a100-hourly
#SBATCH --account=merlin
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --gres=gpu:4
#SBATCH --time=01:00:00

export TRAF81_CELL=rank-8
export TRAF81_EXPECTED_RANKS=8
export TRAF81_BATCH_SOURCE=measure_r8.sbatch
exec "${TRAF81_STAGE_ROOT:?set TRAF81_STAGE_ROOT}/run_merlin_cell.sh"
