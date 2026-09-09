#!/bin/bash
# Copy what matters from a run on the cluster to this machine.
#
#     slurm/fetch_run.sh RUN_ID [USER@HOST] [GENERATION...]
#
# Brings back the newest and best checkpoints with their sidecars, the metric
# streams, the arena results, the logs, and the provenance files. It does not
# bring back the replay shards: they are the only large thing in a run, and
# nothing on a laptop needs them. Everything lands under runs/<run id>/, which
# is gitignored.
#
# Naming generations fetches those checkpoints as well:
#
#     slurm/fetch_run.sh e2-ownership me@slurm.csse.rose-hulman.edu 120
#
# which is what a release needs. `latest.pt` is whatever the run stopped on and
# `best.pt` follows the in-loop quick evaluation, so neither is a stable name for
# a specific network -- and the checkpoint a release publishes has to be named by
# its generation, because that is what the rating report and the manifest call it.
#
# Uses scp, which Windows ships with OpenSSH; rsync is not assumed.
set -uo pipefail

if [ $# -lt 1 ]; then
    echo "usage: slurm/fetch_run.sh RUN_ID [USER@HOST] [GENERATION...]" >&2
    exit 2
fi

run_id="$1"
shift

# A bare number in the host position is a generation, not a host. Without this,
# `fetch_run.sh e2-ownership 120` tries to scp from a machine called 120 and
# fails several lines later with a name-resolution error that says nothing about
# the actual mistake.
if [ $# -gt 0 ] && ! printf '%s' "$1" | grep -Eq '^[0-9]+$'; then
    host="$1"
    shift
else
    host="slurm.csse.rose-hulman.edu"
fi

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

# Any generations named on the command line, zero-padded the way the trainer
# writes them.
for generation in "$@"; do
    name=$(printf 'gen_%05d' "$generation")
    scp -q "$remote/checkpoints/$name.*" "$local_dir/checkpoints/"         || echo "  (no checkpoint for generation $generation)"
done

echo "fetched $run_id into $local_dir"
ls -la "$local_dir/checkpoints"
