#!/bin/bash
#SBATCH --job-name=traf81-r2
#SBATCH --partition=a100-hourly
#SBATCH --account=merlin
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:2
#SBATCH --time=01:00:00

export TRAF81_CELL=rank-2
export TRAF81_EXPECTED_RANKS=2
export TRAF81_BATCH_SOURCE=measure_r2.sbatch
exec "${TRAF81_STAGE_ROOT:?set TRAF81_STAGE_ROOT}/run_merlin_cell.sh"
