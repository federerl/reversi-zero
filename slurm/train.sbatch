#!/bin/bash
#SBATCH --job-name=rz-train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --signal=B:USR1@900
#SBATCH --requeue
#SBATCH --output=slurm/logs/%x-%j.out
#
# One 24-hour slice of a training run.
#
# The cluster's wall-clock limit is 24 hours on every partition, and a full 8x8
# run takes longer than that. So a run is a chain of identical jobs, each one
# picking up where the last stopped. Three things already in the code make that
# work without any logic here:
#
# * `--generations N` means "stop once the run has N generations in total", so
#   a job that finds the run already finished exits in seconds.
# * `--run-id` reuses the existing run directory, and `--resume auto` (the
#   default) loads the newest checkpoint that verifies.
# * Fifteen minutes before the limit, Slurm sends USR1; `rz_run` forwards it and
#   the loop finishes the current generation, writes the checkpoint, and exits.
#   The next job in the chain then resumes from a complete generation instead of
#   from whatever was on disk when the node killed the process.
#
# `--requeue` covers the other case: a node failure puts the job back in the
# queue, and it resumes the same way.
#
# Required environment (set by slurm/submit_chain.sh, or by hand):
#     CONFIG       profile YAML, e.g. configs/full8x8_e1_10x128.yaml
#     RUN_ID       the run directory name; the same value on every job of a chain
#     GENERATIONS  total generations the run should reach
# Optional:
#     WORKERS      self-play worker processes; default is the CPU count minus two

source slurm/env.sh

: "${CONFIG:?set CONFIG to the profile YAML}"
: "${RUN_ID:?set RUN_ID to the run directory name}"
: "${GENERATIONS:?set GENERATIONS to the total generation target}"

workers="${WORKERS:-$(( SLURM_CPUS_PER_TASK - 2 ))}"

echo "config $CONFIG, run $RUN_ID, target $GENERATIONS generations, $workers self-play workers"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

rz_run uv run reversi train \
    -c "$CONFIG" \
    --run-id "$RUN_ID" \
    --generations "$GENERATIONS" \
    --device cuda \
    --set "selfplay.n_workers=$workers"
