# Running on the CSSE Slurm cluster

Slurm is a job scheduler. Nothing heavy runs on the login node; a short script says what
resources a job needs, `sbatch` hands it to Slurm, and Slurm runs it on a compute node when
one is free. These files are those scripts.

## The cluster, as measured

| Fact | Value |
|---|---|
| Login node | `slurm.csse.rose-hulman.edu` (SSH), or the web portal at `hpc.csse.rose-hulman.edu` |
| GPU partition `gpu` | **gus**: 6× NVIDIA L40S (48 GB each), 255 CPU threads, 1 TB RAM. **gebru**: 8× Quadro RTX 6000, 48 threads |
| CPU partition `cpu` | floyd, gauss, hopper, noether: 984 cores in total |
| Wall-clock limit | 24 hours on every partition |
| Per-user limits | none configured |
| GPU request | `--gres=gpu:nvidia_l40s:1`, which lands on gus |
| Software | no module system; system Python 3.11; `uv` installed per user |
| Storage | home is shared NFS with plenty of room; runs live in `~/reversi-runs` |

Training goes to gus, not gebru. Self-play is CPU-bound tree search with the GPU scoring
positions in batches, and gebru has six CPU threads per GPU, which would starve the workers.

## Files

| File | What it does |
|---|---|
| `setup.sh` | one-time, on the login node: install `uv`, build the environment, run the fast tests |
| `env.sh` | sourced by every job: PATH, `RZ_RUN_ROOT`, and `rz_run`, which forwards the wall-clock warning to the training process |
| `smoke4x4.sbatch` | the 4×4 learning gate as a batch job; proves the batch path works |
| `bench.sbatch` | self-play throughput for one network size; decides how big a network a 20-minute generation allows |
| `train.sbatch` | one 24-hour slice of a run; takes `CONFIG`, `RUN_ID`, `GENERATIONS` from the environment |
| `submit_chain.sh` | submits N copies of `train.sbatch` that run back to back |
| `cpu.sbatch` | runs any CPU-only command on the `cpu` partition; used for calibration and arena jobs |
| `fetch_run.sh` | copies a run's checkpoints, metrics and logs to a laptop; never the replay shards |

Every job is submitted from the repository root, and every log lands in `slurm/logs/`,
which is gitignored.

## First time

```bash
ssh <you>@slurm.csse.rose-hulman.edu
git clone https://github.com/federerl/reversi-zero.git && cd reversi-zero
bash slurm/setup.sh
sbatch slurm/smoke4x4.sbatch
squeue -u $USER                          # PD waiting, R running
tail -f slurm/logs/rz-smoke-<jobid>.out  # Ctrl-C stops tail, not the job
```

The smoke log should end with a checkpoint path. Then prove the wall-clock path once:

```bash
RUN_ID=slurm-smoke sbatch --time=00:03:00 --signal=B:USR1@60 slurm/smoke4x4.sbatch
```

The log should say "asking the run to stop after this generation" and exit cleanly, and a
third plain submission of the same run id resumes and finishes. `env.json` in the run
directory records `SLURM_RESTART_COUNT`, which is the evidence that resume across jobs works.

## A real run

```bash
N_BLOCKS=10 CHANNELS=128 sbatch slurm/bench.sbatch     # is a generation under 20 minutes?
slurm/submit_chain.sh configs/full8x8_e1_10x128.yaml e1-10x128 120 4
slurm/submit_chain.sh configs/full8x8.yaml control 120 4
```

Watch with `squeue -u $USER`. Read progress with `tail -f slurm/logs/rz-e1-10x128-<jobid>.out`,
or from the metric streams under `~/reversi-runs/<run id>/metrics/`. Stop a whole chain with
`scancel --name=rz-e1-10x128`.

## Getting results home

From the laptop, in Git Bash:

```bash
slurm/fetch_run.sh e1-10x128 <you>@slurm.csse.rose-hulman.edu
```

## Habits

* Never run training on the login node. It is shared by everyone.
* Request the CPU cores the job uses. Workers that exceed the request are throttled.
* One `--set` on the command line is fine; anything more belongs in a config file so the
  run's `config.yaml` tells the whole story.
