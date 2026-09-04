#!/bin/bash
# One-time setup on the login node. Run from the repository root:
#
#     bash slurm/setup.sh
#
# Installs uv if it is missing, builds the Python environment with the CUDA build
# of torch, and runs the fast tests so a broken environment is found here rather
# than inside a queued job. Needs the network, which the login node has.
set -euo pipefail

if [ ! -f pyproject.toml ]; then
    echo "run this from the repository root" >&2
    exit 2
fi

if ! command -v uv >/dev/null 2>&1 && [ ! -x "$HOME/.local/bin/uv" ]; then
    echo "installing uv"
    wget -qO- https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"

echo "syncing the environment (torch with CUDA, a few minutes the first time)"
uv sync --extra cu124 --extra obs --extra dev

echo "fast tests on the login node (CPU only, about 80 seconds)"
uv run pytest -m "not slow and not gpu" -q 2>&1 | tail -3

mkdir -p slurm/logs "$HOME/reversi-runs"
echo
echo "ready. next: sbatch slurm/smoke4x4.sbatch"
