#!/usr/bin/env bash
# ============================================================
#  Stage 3: Generate synthetic calibration dataset achieving
#           balanced activated experts
#  Requires a running vLLM/NIM server at VLLM_BASE_URL.
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
SRC="${REPO_ROOT}/src/stage3/pipeline_calibration_per_domain.py"

# ── Configuration ──────────────────────────────────────────
MODEL_NAME="qwen3_30b_a3b"
DATASET_ID="nemo_dataset"
STAGE="0"
CALIB_SIZE=128

# Input: bracketed samples from Stage 1
SEED_DATA_PATH="${REPO_ROOT}/results/stage1_routing/${MODEL_NAME}_${DATASET_ID}/D${STAGE}_${CALIB_SIZE}/s6_apply_bracket/bracketed_balance.jsonl"

# Input: guidelines from Stage 2
INSTRUCTION_DIR="${REPO_ROOT}/results/stage2_guidelines/instruction"

# Output
OUTPUT_ROOT="${REPO_ROOT}/results/stage3_dataset/output_per_domain"

echo "=========================================="
echo " Stage 3: Generate Synthetic Dataset"
echo " Seed data  : ${SEED_DATA_PATH}"
echo " Guidelines : ${INSTRUCTION_DIR}"
echo " Output     : ${OUTPUT_ROOT}"
echo "=========================================="

if [ ! -f "${SEED_DATA_PATH}" ]; then
    echo "[Error] Seed data not found: ${SEED_DATA_PATH}"
    echo "        Please run Stage 1 first (scripts/run_stage1.sh)"
    exit 1
fi

if [ ! -d "${INSTRUCTION_DIR}" ]; then
    echo "[Error] Instruction directory not found: ${INSTRUCTION_DIR}"
    echo "        Please run Stage 2 first (scripts/run_stage2.sh)"
    exit 1
fi

mkdir -p "${OUTPUT_ROOT}"

python3 "${SRC}" \
    --seed_data_path "${SEED_DATA_PATH}" \
    --instruction_dir "${INSTRUCTION_DIR}" \
    --output_root "${OUTPUT_ROOT}"

echo ""
echo "=============================="
echo " Stage 3 Complete!"
echo " Dataset: ${OUTPUT_ROOT}"
echo "=============================="
