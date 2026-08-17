#!/bin/bash
#SBATCH --job-name=w18b-mx-pair
#SBATCH --partition=a100-hourly
#SBATCH --account=merlin
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH hetjob
#SBATCH --partition=gh-hourly
#SBATCH --account=merlin
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00

# mx-pair: the heterogeneous A100 plus GH200 capture cell of freeze 1.
# Matrix probe plus one 60-second stream in each direction, bootstrap and
# data pinned to the shared hsn fabric subnet.

exec "${W18_STAGE_ROOT:?set W18_STAGE_ROOT}/run_capture_mx.sh"
