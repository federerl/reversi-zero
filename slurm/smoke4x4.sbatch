#!/bin/bash
#SBATCH --job-name=rz-smoke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --signal=B:USR1@120
#SBATCH --output=slurm/logs/%x-%j.out
#
# The 4x4 learning gate as a batch job. Twelve generations, about five minutes on
# an L40S. Its purpose is to prove the batch path, not to produce a result:
# the GPU is visible, the environment loads, a run directory appears under
# RZ_RUN_ROOT, and the log ends with a checkpoint path.
#
# Submit from the repository root:
#     sbatch slurm/smoke4x4.sbatch
#
# To test the wall-clock path (criterion S9), resubmit the same run id with a
# limit it cannot meet. The job should log "asking the run to stop", finish the
# generation it is on, and exit; a third submission then resumes and completes:
#     RUN_ID=slurm-smoke sbatch --time=00:03:00 --signal=B:USR1@60 slurm/smoke4x4.sbatch

source slurm/env.sh

rz_run uv run reversi train \
    -c configs/smoke4x4.yaml \
    --run-id "${RUN_ID:-slurm-smoke}" \
    --device cuda
