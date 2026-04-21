#!/usr/bin/env bash
# SECOND RUN — expert thresholds + step3/4 with thresholds (same as pipeline.sh).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
# shellcheck source=config.sh
source "${SCRIPT_DIR}/config.sh"

echo "!! SECOND RUN !!"

python3 step3_count_expert_dist.py \
    --jsonl ${SAVE_PATH_2}/token_routing.jsonl \
    --model-name ${MODEL_PATH} \
    --out-path ${SAVE_PATH_3} \
    --freq-thr ${freq_thr} \
    --less-thr ${less_thr} \
    --trust-remote-code \
    2>&1 | tee -a ${LOG_BASE_DIR}/step3_count_expert_dist_thr.log

python3 step4_weight_outlier_dist.py \
    --jsonl ${SAVE_PATH_2}/token_routing.jsonl \
    --model-name ${MODEL_PATH} \
    --out-path ${SAVE_PATH_4} \
    --sen-thr ${sen_thr} \
    --rob-thr ${rob_thr} \
    --trust-remote-code \
    2>&1 | tee -a ${LOG_BASE_DIR}/step4_weight_outlier_dist_thr.log



mkdir -p ${SAVE_PATH_5}

python3 step5_sort_token_plot.py \
    --mode "balance" \
    --jsonl ${SAVE_PATH_2}/token_routing.jsonl \
    --threshold-json ${SAVE_PATH_3}/expert_dist_threshold_experts.json \
    --model-name ${MODEL_PATH} \
    --out-path ${SAVE_PATH_5} \
    2>&1 | tee -a ${LOG_BASE_DIR}/step5_sort_token_plot_balance.log

python3 step5_sort_token_plot.py \
    --mode "q_sensitivity" \
    --jsonl ${SAVE_PATH_2}/token_routing.jsonl \
    --threshold-json ${SAVE_PATH_4}/expert_weight_sensitivity_threshold_experts.json \
    --model-name ${MODEL_PATH} \
    --out-path ${SAVE_PATH_5} \
    2>&1 | tee -a ${LOG_BASE_DIR}/step5_sort_token_plot_q_sensitivity.log

