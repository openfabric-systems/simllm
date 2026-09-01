#!/bin/bash
#SBATCH --job-name=traf77-w2-four
#SBATCH --partition=a100-hourly
#SBATCH --account=merlin
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

# Width 2, four-port concentration. This job covers all four operations at all
# 22 payloads, for 88 frozen cells. NCCL uses the site default interface spread.

export TRAF77_WIDTH=2
export TRAF77_CONCENTRATION=four-port
export TRAF77_SUBMITTED_SCRIPT=capture_w2_four_port.sbatch
exec "${TRAF77_STAGE_ROOT:?set TRAF77_STAGE_ROOT}/run_capture.sh"
