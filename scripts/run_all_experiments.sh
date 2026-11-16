#!/usr/bin/env bash
set -euo pipefail

# Sequential runner for all 7 Maze2D obstacle gradient guidance experiments.
# Uses a fixed random seed and batch_size=256 so every sample index (0..255)
# starts from the exact same initial noise and reverse-diffusion noise across all runs.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-/home/linux/miniconda3/envs/hardflow_backup/bin/python}"
DATASET="${DATASET:-maze2d-large-v1}"
CONFIG="${CONFIG:-config.maze2d}"
BATCH_SIZE="${BATCH_SIZE:-256}"
SEED="${SEED:-42}"

COMMON_ARGS=(
    --config "${CONFIG}"
    --dataset "${DATASET}"
    --batch_size "${BATCH_SIZE}"
    --seed "${SEED}"
)

echo "=================================================================="
echo " Starting Sequential Maze2D Experiments"
echo " Python:     ${PYTHON_BIN}"
echo " Dataset:    ${DATASET}"
echo " Batch Size: ${BATCH_SIZE}"
echo " Seed:       ${SEED}"
echo "=================================================================="

# 1. Original (unguided base model)
echo ""
echo "[1/7] Running: original (use_guidance=False)"
"${PYTHON_BIN}" scripts/plan_maze2d.py "${COMMON_ARGS[@]}" \
    --use_guidance False \
    --save_name original

# 2. x0 (DPS / clean-prediction with U-Net backprop, additive only)
echo ""
echo "[2/7] Running: x0 (guidance_method=x0, use_basin_escape=False)"
"${PYTHON_BIN}" scripts/plan_maze2d.py "${COMMON_ARGS[@]}" \
    --use_guidance True \
    --guidance_method x0 \
    --use_basin_escape False \
    --save_name x0

# 3. x0_fast (Clean-prediction without U-Net backprop, additive only)
echo ""
echo "[3/7] Running: x0_fast (guidance_method=x0_fast, use_basin_escape=False)"
"${PYTHON_BIN}" scripts/plan_maze2d.py "${COMMON_ARGS[@]}" \
    --use_guidance True \
    --guidance_method x0_fast \
    --use_basin_escape False \
    --save_name x0_fast

# 4. xt (Original Diffuser noisy-state guidance, additive only)
echo ""
echo "[4/7] Running: xt (guidance_method=xt, use_basin_escape=False)"
"${PYTHON_BIN}" scripts/plan_maze2d.py "${COMMON_ARGS[@]}" \
    --use_guidance True \
    --guidance_method xt \
    --use_basin_escape False \
    --save_name xt

# 5. x0_escape (x0 + Langevin basin escape)
echo ""
echo "[5/7] Running: x0_escape (guidance_method=x0, use_basin_escape=True)"
"${PYTHON_BIN}" scripts/plan_maze2d.py "${COMMON_ARGS[@]}" \
    --use_guidance True \
    --guidance_method x0 \
    --use_basin_escape True \
    --save_name x0_escape

# 6. fast_escape (x0_fast + Langevin basin escape)
echo ""
echo "[6/7] Running: fast_escape (guidance_method=x0_fast, use_basin_escape=True)"
"${PYTHON_BIN}" scripts/plan_maze2d.py "${COMMON_ARGS[@]}" \
    --use_guidance True \
    --guidance_method x0_fast \
    --use_basin_escape True \
    --save_name x0_fast_escape

# 7. xt_escape (xt + Langevin basin escape)
echo ""
echo "[7/7] Running: xt_escape (guidance_method=xt, use_basin_escape=True)"
"${PYTHON_BIN}" scripts/plan_maze2d.py "${COMMON_ARGS[@]}" \
    --use_guidance True \
    --guidance_method xt \
    --use_basin_escape True \
    --save_name xt_escape

echo ""
echo "=================================================================="
echo " All 7 experiments completed!"
echo " Results saved in: logs/${DATASET}/plans/{original,x0,x0_fast,xt,x0_escape,fast_escape,xt_escape}"
echo "=================================================================="

