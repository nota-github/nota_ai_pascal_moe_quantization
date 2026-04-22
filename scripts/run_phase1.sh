#!/usr/bin/env bash
# ============================================================
#  Stage 1: Analyze activated experts statistics from calibration data
#  Runs all 4 sub-steps sequentially in a single script.
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
SRC_DIR="${REPO_ROOT}/src/phase1"

# ── Virtual environment ────────────────────────────────────
VENV_DIR="/home/work/nota-data/nemo_hackathon/venv/quant_expert_analysis/.venv"
source "${VENV_DIR}/bin/activate"

# ── Configuration ──────────────────────────────────────────
export CUDA_VISIBLE_DEVICES=4

STAGE="0"
NUM_SAMPLES=128
CALIB_SIZE=$NUM_SAMPLES

MODEL_BASE_PATH="/home/work/nota-data/ghlee/storage/base_model"
MODEL_NAME="qwen3_30b_a3b"
MODEL_PATH="${MODEL_BASE_PATH}/${MODEL_NAME}"

NUM_SAMPLES_PER_DOMAIN=$((CALIB_SIZE / 4))
MIN_LENGTH=1024
MAX_LENGTH=2048
DATASET_ID="nemo_dataset"       # nemo_dataset | custom

DATASET_DIR="/home/work/nota-data/nemo_hackathon/datasets/D${STAGE}_${CALIB_SIZE}"

SAVE_BASE_PATH="${REPO_ROOT}/results/phase1_routing"
SAVE_PATH="${SAVE_BASE_PATH}/${MODEL_NAME}_${DATASET_ID}/D${STAGE}_${CALIB_SIZE}"
LOG_DIR="${SAVE_PATH}/log"
mkdir -p "${LOG_DIR}"

SAVE_PATH_1="${SAVE_PATH}/s1_dataset_dir"
SAVE_PATH_2="${SAVE_PATH}/s2_expert_count"
SAVE_PATH_3="${SAVE_PATH}/s3_expert_dist"
SAVE_PATH_4="${SAVE_PATH}/s4_weight_outlier"
SAVE_PATH_5="${SAVE_PATH}/s5_sorted_token"
SAVE_PATH_6="${SAVE_PATH}/s6_apply_bracket"

# Expert thresholds (tuned in sub-step 2)
freq_thr=0.25
less_thr=0.25
rob_thr=0.25
sen_thr=0.25

# Token thresholds (tuned in sub-step 3)
# balance:      token_blue = freq>btx & less<bty ; token_red = freq<rtx & less>rty
balance_thr_blue_x=0.1
balance_thr_blue_y=0.77
balance_thr_red_x=0.15
balance_thr_red_y=0.6
# q_sensitivity: token_blue = sensitive<btx & robust>bty ; token_red = sensitive>rtx & robust<rty
q_sensitive_thr_blue_x=0.2
q_sensitive_thr_blue_y=0.2
q_sensitive_thr_red_x=0.35
q_sensitive_thr_red_y=0.25

# Optional: uncomment to limit samples for debug
# LIMIT=32

echo "=========================================="
echo " Stage 1: Analyze Expert Routing"
echo " MODEL     : ${MODEL_NAME}"
echo " DATASET   : ${DATASET_DIR}"
echo " OUTPUT    : ${SAVE_PATH}"
echo "=========================================="

# ── Sub-step 1 (optional): Dataset load ────────────────────
# Uncomment if you need to generate a custom calibration dataset.
# mkdir -p "${SAVE_PATH_1}"
# python3 "${SRC_DIR}/step1_dataset_load.py" \
#     --model-name "${MODEL_PATH}" \
#     --dataset-kind "${DATASET_ID}" \
#     --nemo-samples-per-domain "${NUM_SAMPLES_PER_DOMAIN}" \
#     --min-length "${MIN_LENGTH}" \
#     --max-length "${MAX_LENGTH}" \
#     --save-path "${SAVE_PATH_1}" \
#     2>&1 | tee -a "${LOG_DIR}/step1_dataset_load.log"
# DATASET_DIR="${SAVE_PATH_1}"

# ── Sub-step 2: Count expert activations ───────────────────
echo ""
echo "[Sub-step 2] Counting expert activations..."
mkdir -p "${SAVE_PATH_2}"
python3 "${SRC_DIR}/step2_count_expert.py" \
    --model-name "${MODEL_PATH}" \
    --dataset-dir "${DATASET_DIR}" \
    --save-path "${SAVE_PATH_2}" \
    --trust-remote-code \
    --sample-json-token mid_first \
    2>&1 | tee -a "${LOG_DIR}/step2_count_expert.log"
echo "[Sub-step 2] Done → ${SAVE_PATH_2}"

# ── Sub-step 3a: Expert distribution (no threshold) ────────
echo ""
echo "[Sub-step 3a] Expert distribution (no threshold)..."
mkdir -p "${SAVE_PATH_3}"
python3 "${SRC_DIR}/step3_count_expert_dist.py" \
    --jsonl "${SAVE_PATH_2}/token_routing.jsonl" \
    --model-name "${MODEL_PATH}" \
    --out-path "${SAVE_PATH_3}" \
    --trust-remote-code \
    2>&1 | tee -a "${LOG_DIR}/step3_count_expert_dist.log"

# ── Sub-step 4a: Weight outlier distribution (no threshold) ─
echo ""
echo "[Sub-step 4a] Weight outlier distribution (no threshold)..."
mkdir -p "${SAVE_PATH_4}"
python3 "${SRC_DIR}/step4_weight_outlier_dist.py" \
    --jsonl "${SAVE_PATH_2}/token_routing.jsonl" \
    --model-name "${MODEL_PATH}" \
    --out-path "${SAVE_PATH_4}" \
    --scatter-x linear \
    2>&1 | tee -a "${LOG_DIR}/step4_weight_outlier_dist.log"

# ── Sub-step 3b: Expert distribution (with threshold) ──────
echo ""
echo "[Sub-step 3b] Expert distribution (with threshold)..."
python3 "${SRC_DIR}/step3_count_expert_dist.py" \
    --jsonl "${SAVE_PATH_2}/token_routing.jsonl" \
    --model-name "${MODEL_PATH}" \
    --out-path "${SAVE_PATH_3}" \
    --freq-thr "${freq_thr}" \
    --less-thr "${less_thr}" \
    --trust-remote-code \
    2>&1 | tee -a "${LOG_DIR}/step3_count_expert_dist_thr.log"

# ── Sub-step 4b: Weight outlier distribution (with threshold)
echo ""
echo "[Sub-step 4b] Weight outlier distribution (with threshold)..."
python3 "${SRC_DIR}/step4_weight_outlier_dist.py" \
    --jsonl "${SAVE_PATH_2}/token_routing.jsonl" \
    --model-name "${MODEL_PATH}" \
    --out-path "${SAVE_PATH_4}" \
    --sen-thr "${sen_thr}" \
    --rob-thr "${rob_thr}" \
    --trust-remote-code \
    2>&1 | tee -a "${LOG_DIR}/step4_weight_outlier_dist_thr.log"

# ── Sub-step 5a: Token scatter plot (no token threshold) ───
echo ""
echo "[Sub-step 5a] Token sort/plot (no token threshold)..."
mkdir -p "${SAVE_PATH_5}"
python3 "${SRC_DIR}/step5_sort_token_plot.py" \
    --mode "balance" \
    --jsonl "${SAVE_PATH_2}/token_routing.jsonl" \
    --threshold-json "${SAVE_PATH_3}/expert_dist_threshold_experts.json" \
    --model-name "${MODEL_PATH}" \
    --out-path "${SAVE_PATH_5}" \
    2>&1 | tee -a "${LOG_DIR}/step5_sort_token_plot_balance.log"

python3 "${SRC_DIR}/step5_sort_token_plot.py" \
    --mode "q_sensitivity" \
    --jsonl "${SAVE_PATH_2}/token_routing.jsonl" \
    --threshold-json "${SAVE_PATH_4}/expert_weight_sensitivity_threshold_experts.json" \
    --model-name "${MODEL_PATH}" \
    --out-path "${SAVE_PATH_5}" \
    2>&1 | tee -a "${LOG_DIR}/step5_sort_token_plot_q_sensitivity.log"

# ── Sub-step 5b: Token scatter plot (with token threshold) ──
echo ""
echo "[Sub-step 5b] Token sort/plot (with token threshold)..."
python3 "${SRC_DIR}/step5_sort_token_plot.py" \
    --mode "balance" \
    --jsonl "${SAVE_PATH_2}/token_routing.jsonl" \
    --threshold-json "${SAVE_PATH_3}/expert_dist_threshold_experts.json" \
    --model-name "${MODEL_PATH}" \
    --out-path "${SAVE_PATH_5}" \
    --token-thr-blue "${balance_thr_blue_x}" "${balance_thr_blue_y}" \
    --token-thr-red "${balance_thr_red_x}" "${balance_thr_red_y}" \
    2>&1 | tee -a "${LOG_DIR}/step5_sort_token_plot_balance_thr.log"

python3 "${SRC_DIR}/step5_sort_token_plot.py" \
    --mode "q_sensitivity" \
    --jsonl "${SAVE_PATH_2}/token_routing.jsonl" \
    --threshold-json "${SAVE_PATH_4}/expert_weight_sensitivity_threshold_experts.json" \
    --model-name "${MODEL_PATH}" \
    --out-path "${SAVE_PATH_5}" \
    --token-thr-blue "${q_sensitive_thr_blue_x}" "${q_sensitive_thr_blue_y}" \
    --token-thr-red "${q_sensitive_thr_red_x}" "${q_sensitive_thr_red_y}" \
    2>&1 | tee -a "${LOG_DIR}/step5_sort_token_plot_q_sensitivity_thr.log"

# ── Sub-step 6: Apply <red>/<blue> brackets ─────────────────
echo ""
echo "[Sub-step 6] Applying <red>/<blue> brackets..."
mkdir -p "${SAVE_PATH_6}"
python3 "${SRC_DIR}/step6_apply_bracket.py" \
    --mode "balance" \
    --text-jsonl "${DATASET_DIR}/samples_text.jsonl" \
    --classified-json "${SAVE_PATH_5}/token_freq_less_scatter_balance_classified.json" \
    --model-name "${MODEL_PATH}" \
    --out-path "${SAVE_PATH_6}" \
    --one-json-out "${SAVE_PATH_6}/bracketed_balance_one.json" \
    ${LIMIT:+--limit ${LIMIT}} \
    --trust-remote-code \
    --bracket_after_input \
    2>&1 | tee -a "${LOG_DIR}/step6_apply_bracket_balance.log"

python3 "${SRC_DIR}/step6_apply_bracket.py" \
    --mode "q_sensitivity" \
    --text-jsonl "${DATASET_DIR}/samples_text.jsonl" \
    --classified-json "${SAVE_PATH_5}/token_freq_less_scatter_q_sensitivity_classified.json" \
    --model-name "${MODEL_PATH}" \
    --out-path "${SAVE_PATH_6}" \
    --one-json-out "${SAVE_PATH_6}/bracketed_q_sensitivity_one.json" \
    ${LIMIT:+--limit ${LIMIT}} \
    --trust-remote-code \
    --bracket_after_input \
    2>&1 | tee -a "${LOG_DIR}/step6_apply_bracket_q_sensitivity.log"

echo ""
echo "=============================="
echo " Stage 1 Complete!"
echo " Output : ${SAVE_PATH}"
echo " Key output: ${SAVE_PATH_6}/bracketed_balance.jsonl"
echo "=============================="
