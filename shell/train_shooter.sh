#!/usr/bin/env bash
set -euo pipefail

# Two-stage progressive shooter training.
# Stage I: motion tracking (adaptive sampling) → Stage II: perception-guided kicking.
#
# Usage:
#   bash shell/train_shooter.sh [run_name] [num_envs]
#
# Examples:
#   bash shell/train_shooter.sh my_experiment 4096
#   bash shell/train_shooter.sh test 64    # small-scale test

RUN_NAME="${1:-test}"
NUM_ENVS="${2:-4096}"
MOTION_DIR="src/assets/soccer/motions"
MAX_ITER_STAGE1="${3:-4000}"
MAX_ITER_STAGE2="${4:-6000}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
LOG_ROOT="${REPO_ROOT}/logs/rsl_rl/g1_soccer"

cd "${REPO_ROOT}"

echo "============================================"
echo " Stage I: Motion Tracking (adaptive sampling)"
echo "============================================"
echo "Run name:   ${RUN_NAME}_stage1"
echo "Num envs:   ${NUM_ENVS}"
echo "Max iters:  ${MAX_ITER_STAGE1}"
echo "Motion dir: ${MOTION_DIR}"
echo ""

python scripts/train.py Unitree-G1-Shooter-Stage1 \
    --motion-dir "${MOTION_DIR}" \
    --env.scene.num_envs "${NUM_ENVS}" \
    --agent.max_iterations "${MAX_ITER_STAGE1}" \
    --agent.run_name "${RUN_NAME}_stage1" \
    --agent.save_interval 100 \

echo ""
echo "Stage I complete. Resolving checkpoint..."

LOAD_DIR=$(find "${LOG_ROOT}" -maxdepth 1 -mindepth 1 -type d -name "*_${RUN_NAME}_stage1" | sort | tail -n 1)

if [ -z "${LOAD_DIR}" ]; then
    echo "ERROR: Could not find Stage I log directory in ${LOG_ROOT}"
    exit 1
fi

CKPT_FILE=$(find "${LOAD_DIR}" -name "model_*.pt" -type f | sort -V | tail -n 1)

if [ -z "${CKPT_FILE}" ]; then
    echo "ERROR: No checkpoint found in ${LOAD_DIR}"
    exit 1
fi

LOAD_RUN=$(basename "${LOAD_DIR}")
LOAD_CKPT=$(basename "${CKPT_FILE}")

echo "Load run:    ${LOAD_RUN}"
echo "Checkpoint:  ${LOAD_CKPT}"
echo ""

echo "============================================"
echo " Stage II: Perception-Guided Kicking"
echo "============================================"
echo "Run name:   ${RUN_NAME}_stage2"
echo "Num envs:   ${NUM_ENVS}"
echo "Max iters:  ${MAX_ITER_STAGE2}"
echo "Resuming from: ${LOAD_RUN}/${LOAD_CKPT}"
echo ""

python scripts/train.py Unitree-G1-Shooter-Stage2 \
    --motion-dir "${MOTION_DIR}" \
    --env.scene.num_envs "${NUM_ENVS}" \
    --agent.max_iterations "${MAX_ITER_STAGE2}" \
    --agent.run_name "${RUN_NAME}_stage2" \
    --agent.resume True \
    --agent.load_run "${LOAD_RUN}" \
    --agent.load_checkpoint "${LOAD_CKPT}" \
    --agent.save_interval 100 \

echo ""
echo "Training complete!"
