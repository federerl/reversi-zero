#!/bin/bash
#SBATCH --job-name=rz-bench
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=slurm/logs/%x-%j.out
#
# Measure self-play throughput on the cluster for one network size.
#
# The question this answers: how long does one generation take here? The 8x8
# configs were sized on a laptop GPU with 20 cores; the cluster has an L40S and
# far more cores, and a bigger network costs more per position. The bench's
# `inference` phase at the configured batch size is the number to read, and a
# generation should stay under about twenty minutes.
#
# Submit from the repository root, once per network size under consideration:
#     sbatch slurm/bench.sbatch                                   # 6x64, the 0.0 network
#     N_BLOCKS=10 CHANNELS=128 sbatch slurm/bench.sbatch          # E1 candidate
#     N_BLOCKS=8  CHANNELS=96  sbatch slurm/bench.sbatch          # E1 fallback

source slurm/env.sh

n_blocks="${N_BLOCKS:-6}"
channels="${CHANNELS:-64}"
gpu="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr ' ' '-' | tr '[:upper:]' '[:lower:]')"
out="bench/results/cluster-${gpu}-${n_blocks}x${channels}.json"

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

rz_run uv run python bench/selfplay_bench.py \
    --device cuda \
    --n-blocks "$n_blocks" \
    --channels "$channels" \
    --simulations 200 \
    --games-batched 48 \
    --out "$out"

echo "wrote $out"
