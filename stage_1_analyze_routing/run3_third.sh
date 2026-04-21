#!/usr/bin/env bash
# Step 5 (no token thr) then THIRD RUN — step5 with --token-thr-* (same order as pipeline.sh).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
# shellcheck source=config.sh
source "${SCRIPT_DIR}/config.sh"

# == STEP 5: sorted token plot (between SECOND and THIRD in pipeline.sh)
# Requires step3 with --freq-thr/--less-thr so expert_dist_threshold_experts.json exists.

echo "!! THIRD RUN !!"

python3 step5_sort_token_plot.py \
    --mode "balance" \
    --jsonl ${SAVE_PATH_2}/token_routing.jsonl \
    --threshold-json ${SAVE_PATH_3}/expert_dist_threshold_experts.json \
    --model-name ${MODEL_PATH} \
    --out-path ${SAVE_PATH_5} \
    --token-thr-blue ${balance_thr_blue_x} ${balance_thr_blue_y} \
    --token-thr-red ${balance_thr_red_x} ${balance_thr_red_y} \
    2>&1 | tee -a ${LOG_BASE_DIR}/step5_sort_token_plot_balance_thr.log

python3 step5_sort_token_plot.py \
    --mode "q_sensitivity" \
    --jsonl ${SAVE_PATH_2}/token_routing.jsonl \
    --threshold-json ${SAVE_PATH_4}/expert_weight_sensitivity_threshold_experts.json \
    --model-name ${MODEL_PATH} \
    --out-path ${SAVE_PATH_5} \
    --token-thr-blue ${q_sensitive_thr_blue_x} ${q_sensitive_thr_blue_y} \
    --token-thr-red ${q_sensitive_thr_red_x} ${q_sensitive_thr_red_y} \
    2>&1 | tee -a ${LOG_BASE_DIR}/step5_sort_token_plot_q_sensitivity_thr.log
