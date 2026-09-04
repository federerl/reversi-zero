#!/bin/bash
# Copy what matters from a run on the cluster to this machine.
#
#     slurm/fetch_run.sh RUN_ID [USER@HOST]
#
# Brings back the newest and best checkpoints with their sidecars, the metric
# streams, the arena results, the logs, and the provenance files. It does not
# bring back the replay shards: they are the only large thing in a run, and
# nothing on a laptop needs them. Everything lands under runs/<run id>/, which
# is gitignored.
#
# Uses scp, which Windows ships with OpenSSH; rsync is not assumed.
set -uo pipefail

if [ $# -lt 1 ]; then
    echo "usage: slurm/fetch_run.sh RUN_ID [USER@HOST]" >&2
    exit 2
fi

run_id="$1"
host="${2:-slurm.csse.rose-hulman.edu}"
remote="$host:reversi-runs/$run_id"
local_dir="runs/$run_id"

mkdir -p "$local_dir/checkpoints"

for item in config.yaml meta.json env.json git.json cmdline.txt; do
    scp -q "$remote/$item" "$local_dir/" || echo "  (no $item)"
done
for dir in metrics arena logs; do
    scp -qr "$remote/$dir" "$local_dir/" || echo "  (no $dir/)"
done
scp -q "$remote/checkpoints/latest.*" "$local_dir/checkpoints/" || echo "  (no latest checkpoint)"
scp -q "$remote/checkpoints/best.*" "$local_dir/checkpoints/" || echo "  (no best checkpoint)"

echo "fetched $run_id into $local_dir"
ls -la "$local_dir/checkpoints"
