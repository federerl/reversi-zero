#!/bin/bash
# Submit a training run as a chain of 24-hour jobs.
#
#     slurm/submit_chain.sh CONFIG RUN_ID GENERATIONS [JOBS]
#
# Example: 120 generations of the E1 candidate, as four back-to-back jobs.
#
#     slurm/submit_chain.sh configs/full8x8_e1_10x128.yaml e1-10x128 120 4
#
# Each job depends on the previous one ending, for any reason (`afterany`), and
# every job runs the identical command. A job that starts after the run has
# already reached its target exits in seconds, so asking for one job too many
# costs nothing, while asking for one too few leaves the run short. Four jobs is
# about 96 hours; at 12 to 20 minutes per generation that covers 120 generations
# with margin.
#
# Check progress with `squeue -u $USER` and `tail -f slurm/logs/rz-<run id>-<job>.out`.
# Cancel the whole chain with `scancel --name=rz-<run id>`.
set -euo pipefail

if [ $# -lt 3 ]; then
    echo "usage: slurm/submit_chain.sh CONFIG RUN_ID GENERATIONS [JOBS]" >&2
    exit 2
fi

config="$1"
run_id="$2"
generations="$3"
jobs="${4:-4}"

if [ ! -f "$config" ]; then
    echo "no such config: $config" >&2
    exit 2
fi
if [ ! -f slurm/train.sbatch ]; then
    echo "run this from the repository root" >&2
    exit 2
fi

dependency=()
for i in $(seq 1 "$jobs"); do
    job_id=$(sbatch --parsable "${dependency[@]}" \
        --job-name="rz-$run_id" \
        --export=ALL,CONFIG="$config",RUN_ID="$run_id",GENERATIONS="$generations" \
        slurm/train.sbatch)
    echo "job $i of $jobs: $job_id"
    dependency=(--dependency="afterany:$job_id")
done

echo "chain submitted for $run_id: $config, $generations generations"
