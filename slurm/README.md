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

Add generation numbers to bring specific checkpoints back as well:

```bash
slurm/fetch_run.sh e2-ownership <you>@slurm.csse.rose-hulman.edu 120
```

`latest.pt` is whatever the run stopped on and `best.pt` follows the in-loop
quick evaluation, so neither is a stable name for a particular network. A release
publishes a checkpoint by its generation, because that is what the rating report
and the web manifest call it.

## Sending a model the other way

```bash
slurm/push_model.sh models/reversi-8x8-gen60.pt <you>@slurm.csse.rose-hulman.edu
```

Files land in `~/reversi-models/`, deliberately outside the run directories:
they are not the output of any run there.

This is needed to rate a network trained somewhere else. Run 1 predates the
cluster, so its generation 60 -- the network the site currently serves -- exists
only on the laptop, and putting it on the same scale as a cluster run means one
of them has to travel. A 1.9 MB checkpoint going up is cheaper than a
120-generation run coming down.

The sidecar travels with the checkpoint. Without it there is no architecture
record and no checksum, and the entrant gets named after its file rather than
after its generation.

## Rating two runs on one scale

Ratings from two tournaments are not comparable, even when both are anchored at
random play. Depth-4 minimax is frozen and played in both of this project's
cross-generation tables; it rates +523 in one and +503 in the other. To compare
networks from different runs, they have to play in the same round robin.

`crossgen` takes extra entrants, so this is one command:

```bash
sbatch slurm/cpu.sbatch uv run reversi arena --suite crossgen     --run-id e2-ownership --max-checkpoints 6     --entrant "run1-gen60=$HOME/reversi-models/reversi-8x8-gen60.pt"     --games 100 --simulations 50 --workers "$SLURM_CPUS_PER_TASK"     --out docs/ratings/release-one-scale.json
```

Name an outside entrant something that does not start with `gen`. The web
manifest reads a `genNN` entrant as a playable model and parses the digits after
the prefix, so `run1-gen60` becomes a rated reference row while `gen60-run1`
would fail the manifest build.

The round robin gives each pairing 100 games. For a claim about one specific
pairing, follow it with a 1000-game match — the standard this project holds a
two-way strength statement to:

```bash
sbatch slurm/cpu.sbatch uv run reversi arena --suite custom     -e "gen120=$HOME/reversi-runs/e2-ownership/checkpoints/gen_00120.pt"     -e "run1-gen60=$HOME/reversi-models/reversi-8x8-gen60.pt"     -e random     --games 1000 --simulations 50 --workers "$SLURM_CPUS_PER_TASK"     --out docs/ratings/release-head-to-head-1000.json
```

`random` is in the field so the fit is anchored where every other table in this
repository is anchored. Without it the ratings would be anchored at whichever
entrant came first and would not be comparable to anything.

## Habits

* Never run training on the login node. It is shared by everyone.
* Request the CPU cores the job uses. Workers that exceed the request are throttled.
* One `--set` on the command line is fine; anything more belongs in a config file so the
  run's `config.yaml` tells the whole story.
