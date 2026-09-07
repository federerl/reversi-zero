#!/bin/bash
# Copy a local model file to the cluster, so it can enter a tournament there.
#
#     slurm/push_model.sh PATH [USER@HOST]
#
# The direction `fetch_run.sh` does not cover, and the release needs it. Run 1
# was trained on a laptop before the cluster existed, so its generation 60 --
# the network the site currently serves -- exists nowhere else. Rating the 1.0
# network against it means one of them has to travel, and a 1.9 MB checkpoint
# travelling to the cluster is cheaper than every checkpoint of a 120-generation
# run travelling the other way.
#
# The sidecar goes with it. A checkpoint without its sidecar has no architecture
# record and no checksum, and `checkpoint_label` falls back to naming the entrant
# after its file, which is how an entrant ends up called `reversi-8x8-gen60`
# instead of `gen60` in a rating table.
#
# Files land in ~/reversi-models/ on the cluster, which is outside the run
# directories on purpose: it is not the output of any run there.
set -uo pipefail

if [ $# -lt 1 ]; then
    echo "usage: slurm/push_model.sh PATH [USER@HOST]" >&2
    exit 2
fi

model="$1"
host="${2:-slurm.csse.rose-hulman.edu}"

if [ ! -f "$model" ]; then
    echo "push_model.sh: no such file: $model" >&2
    exit 2
fi

sidecar="${model%.*}.json"

ssh "$host" 'mkdir -p ~/reversi-models' || exit 1

scp -q "$model" "$host:reversi-models/" || {
    echo "push_model.sh: could not copy $model" >&2
    exit 1
}
if [ -f "$sidecar" ]; then
    scp -q "$sidecar" "$host:reversi-models/"
else
    echo "  (no sidecar beside $model; the entrant will be named after its file)"
fi

echo "pushed $(basename "$model") to $host:reversi-models/"
ssh "$host" 'ls -la ~/reversi-models'
