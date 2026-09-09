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
#         --out runs/calibration/report.json --workers @CPUS@
#
# Write @CPUS@ where the tool wants a worker count. It is replaced with the cores
# this job was actually given, so changing the header above is the only change
# needed.
#
# It is a placeholder rather than $SLURM_CPUS_PER_TASK because of *when* things
# expand. The command is typed on the login node, where that variable does not
# exist, so the login shell replaces it with nothing before sbatch ever sees the
# argument -- and the job runs with an empty --workers. Escaping it does not help
# either: these arguments are passed to a program, not evaluated as a script, so
# the program would receive the literal text. Inside the job is the only place
# the answer is known.

source slurm/env.sh

if [ $# -eq 0 ]; then
    echo "cpu.sbatch: give the command to run after the script name" >&2
    exit 2
fi

# Fill in @CPUS@ now that the allocation is known. Substituting inside each
# argument rather than re-splitting the whole command line keeps a path with a
# space in it intact.
args=()
for arg in "$@"; do
    args+=("${arg//@CPUS@/${SLURM_CPUS_PER_TASK:-1}}")
done

echo "command: ${args[*]}"
rz_run "${args[@]}"
