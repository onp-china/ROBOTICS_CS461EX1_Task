#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/Users/zzzgys/Desktop/robot/ROBOTICS_CS461EX1_Task"
RUNNER="${ROOT_DIR}/mydiffusion/scripts/run_baseline.py"
DEVICE="${MYDIFFUSION_DEVICE:-cuda:0}"

TASKS=(
  "mug_cleanup_d0_lowdim_abs"
  "coffee_d0_lowdim_abs"
  "three_piece_assembly_d0_lowdim_abs"
)

SMOKE_SEED=42
FULL_SEEDS=(43 44)

cd "${ROOT_DIR}"

for task in "${TASKS[@]}"; do
  python "${RUNNER}" --task "${task}" --seed "${SMOKE_SEED}" --exp-name smoke_seed42 --device "${DEVICE}"
done

for task in "${TASKS[@]}"; do
  for seed in "${FULL_SEEDS[@]}"; do
    python "${RUNNER}" --task "${task}" --seed "${seed}" --exp-name baseline_multiseed --device "${DEVICE}"
  done
done
