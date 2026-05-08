#!/usr/bin/env bash
set -euo pipefail

# 项目根目录（自动检测）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "${SCRIPT_DIR}")"
RUNNER="${SCRIPT_DIR}/training/run_baseline.py"
DEVICE="${MYDIFFUSION_DEVICE:-cuda:0}"

TASKS=(
  "three_piece_assembly_d0_lowdim_abs"
)

SMOKE_SEED=42
FULL_SEEDS=(43 44)

cd "${ROOT_DIR}"

# Smoke test
for task in "${TASKS[@]}"; do
  python "${RUNNER}" --task "${task}" --seed "${SMOKE_SEED}" --exp-name smoke_seed42 --device "${DEVICE}"
done

# Full runs
for task in "${TASKS[@]}"; do
  for seed in "${FULL_SEEDS[@]}"; do
    python "${RUNNER}" --task "${task}" --seed "${seed}" --exp-name baseline_multiseed --device "${DEVICE}"
  done
done
