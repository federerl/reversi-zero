# Shared setup for every job on the CSSE Slurm cluster. Sourced by the .sbatch
# files, never run on its own.
#
# What this does and why:
#
# * Puts `uv` on the PATH. It was installed once with the official installer into
#   ~/.local/bin, which a fresh non-interactive job shell does not have on its PATH.
# * Points run output at RZ_RUN_ROOT. The training code reads that variable
#   (`reversi.obs.runmeta.run_root`) and otherwise knows nothing about the cluster.
#   Home is shared NFS with plenty of room, so runs live under ~/reversi-runs.
# * Defines `rz_run`, which starts a command in the background and forwards
#   Slurm's wall-clock warning to it. Without this the warning stops at the
#   batch shell, the training process never hears it, and the job is killed
#   mid-checkpoint when the limit arrives.
#
# What this deliberately does not do: `uv sync`. Syncing needs the network and is
# done once on the login node by slurm/setup.sh. A job should start in seconds
# and should not depend on whether a compute node can reach the internet.

export PATH="$HOME/.local/bin:$PATH"
export RZ_RUN_ROOT="${RZ_RUN_ROOT:-$HOME/reversi-runs}"

# Each self-play worker sets torch to one thread itself. This stops the parent
# process's BLAS from grabbing every core the job was given.
export OMP_NUM_THREADS=1

# Jobs are submitted from the repository root; the relative paths below assume it.
if [ ! -f pyproject.toml ] || [ ! -d slurm ]; then
    echo "env.sh: submit jobs from the repository root (cd ~/reversi-zero first)" >&2
    exit 2
fi

mkdir -p "$RZ_RUN_ROOT" slurm/logs

echo "job ${SLURM_JOB_ID:-?} on ${SLURMD_NODENAME:-?}: ${SLURM_CPUS_PER_TASK:-?} cpus, gpus ${SLURM_GPUS_ON_NODE:-0} (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}), restart count ${SLURM_RESTART_COUNT:-0}"
echo "run root: $RZ_RUN_ROOT"

# Run a command so that it receives the scheduler's signals.
#
# `--signal=B:USR1@N` in the sbatch header makes Slurm send USR1 to the batch
# shell N seconds before the wall clock. The training loop treats USR1 as
# "finish this generation, then stop" and exits cleanly with everything on disk
# complete, so the next job in the chain resumes without losing work. The shell
# only forwards; `wait` is interrupted by the trapped signal and must be repeated
# until the child has actually gone.
rz_run() {
    "$@" &
    local pid=$!
    trap 'echo "[$(date +%T)] wall clock is near: asking the run to stop after this generation"; kill -USR1 "$pid" 2>/dev/null' USR1
    trap 'kill -TERM "$pid" 2>/dev/null' TERM INT
    local status=0
    while :; do
        if wait "$pid"; then
            status=0
        else
            status=$?
        fi
        kill -0 "$pid" 2>/dev/null || break
    done
    trap - USR1 TERM INT
    echo "[$(date +%T)] command exited with status $status"
    return "$status"
}
