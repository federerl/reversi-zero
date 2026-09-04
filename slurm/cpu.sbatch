#!/bin/bash
#SBATCH --job-name=rz-cpu
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=64
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --output=slurm/logs/%x-%j.out
#
# Run any CPU-only command on the cpu partition, which has 984 cores across four
# nodes. Evaluation is what belongs here: round robins between agents are many
# independent games, each on one core, and the GPU would sit idle waiting for one
# position at a time. The difficulty calibration that took 5.9 hours on eight
# laptop workers fits in well under an hour on 64.
#
# The command is whatever follows the script name. Submit from the repository
# root, for example:
#
#     sbatch slurm/cpu.sbatch uv run reversi calibrate models/reversi-8x8-gen60.pt \
#         --out runs/calibration/report.json --workers 64
#
# Use $SLURM_CPUS_PER_TASK in the command if the tool takes a worker count, so a
# change to the header above is the only change needed.

source slurm/env.sh

if [ $# -eq 0 ]; then
    echo "cpu.sbatch: give the command to run after the script name" >&2
    exit 2
fi

echo "command: $*"
rz_run "$@"
