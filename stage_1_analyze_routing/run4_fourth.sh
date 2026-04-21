#!/usr/bin/env bash
# FOURTH RUN — step6 apply bracket (same as pipeline.sh).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
# shellcheck source=config.sh
source "${SCRIPT_DIR}/config.sh"

echo "!! FOURTH RUN !!"

mkdir -p ${SAVE_PATH_6}
# step5 가 --token-thr-* 로 만든 JSON은 token_freq_less_scatter_<mode>_classified.json (고정 이름).

python3 step6_apply_bracket.py \
    --mode "balance" \
    --text-jsonl ${DATASET_DIR}/samples_text.jsonl \
    --classified-json ${SAVE_PATH_5}/token_freq_less_scatter_balance_classified.json \
    --model-name ${MODEL_PATH} \
    --out-path ${SAVE_PATH_6} \
    --one-json-out ${SAVE_PATH_6}/bracketed_balance_one.json \
    ${LIMIT:+--limit ${LIMIT}} \
    --trust-remote-code \
    --bracket_after_input \
    2>&1 | tee -a ${LOG_BASE_DIR}/step6_apply_bracket_balance.log

python3 step6_apply_bracket.py \
    --mode "q_sensitivity" \
    --text-jsonl ${DATASET_DIR}/samples_text.jsonl \
    --classified-json ${SAVE_PATH_5}/token_freq_less_scatter_q_sensitivity_classified.json \
    --model-name ${MODEL_PATH} \
    --out-path ${SAVE_PATH_6} \
    --one-json-out ${SAVE_PATH_6}/bracketed_q_sensitivity_one.json \
    ${LIMIT:+--limit ${LIMIT}} \
    --trust-remote-code \
    --bracket_after_input \
    2>&1 | tee -a ${LOG_BASE_DIR}/step6_apply_bracket_q_sensitivity.log
