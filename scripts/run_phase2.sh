#!/usr/bin/env bash
# ============================================================
#  Stage 2: Extract text patterns causing frequent/scarce experts
#  Requires a running vLLM/NIM server at VLLM_BASE_URL.
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
SRC="${REPO_ROOT}/src/stage2/extract_pattern_agent_balanced.py"

# ── Configuration ──────────────────────────────────────────
MODEL_NAME="qwen3_30b_a3b"
DATASET_ID="nemo_dataset"
STAGE="0"
CALIB_SIZE=128

# Input: bracketed samples from Stage 1
BRACKETED_JSONL="${REPO_ROOT}/results/stage1_routing/${MODEL_NAME}_${DATASET_ID}/D${STAGE}_${CALIB_SIZE}/s6_apply_bracket/bracketed_balance.jsonl"

# Output: per-domain guideline markdown files
OUTPUT_DIR="${REPO_ROOT}/results/stage2_guidelines/instruction"

echo "=========================================="
echo " Stage 2: Extract Text Patterns"
echo " Input : ${BRACKETED_JSONL}"
echo " Output: ${OUTPUT_DIR}"
echo "=========================================="

if [ ! -f "${BRACKETED_JSONL}" ]; then
    echo "[Error] Input file not found: ${BRACKETED_JSONL}"
    echo "        Please run Stage 1 first (scripts/run_stage1.sh)"
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"

python3 "${SRC}" \
    --bracketed_jsonl "${BRACKETED_JSONL}" \
    --output_dir "${OUTPUT_DIR}"

echo ""
echo "=============================="
echo " Stage 2 Complete!"
echo " Guidelines: ${OUTPUT_DIR}"
echo "=============================="
