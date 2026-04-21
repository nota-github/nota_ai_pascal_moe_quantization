#!/usr/bin/env bash
# FIRST RUN — same steps as pipeline.sh (step1 remains commented like the original).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
# shellcheck source=config.sh
source "${SCRIPT_DIR}/config.sh"

echo "!! FIRST RUN !!"

# # == STEP 1: dataset load
# # when there is no dataset, run this
# DATASET_DIR="${SAVE_PATH_1}"
# mkdir -p ${SAVE_PATH_1}
# python3 step1_dataset_load.py \
#     --model-name ${MODEL_PATH} \
#     --dataset-kind ${DATASET_ID} \
#     --nemo-samples-per-domain ${NUM_SAMPLES_PER_DOMAIN} \
#     --min-length ${MIN_LENGTH} \
#     --max-length ${MAX_LENGTH} \
#     --save-path ${SAVE_PATH_1} \
#     2>&1 | tee -a ${LOG_BASE_DIR}/step1_dataset_load.log

mkdir -p ${SAVE_PATH_2}
python3 step2_count_expert.py \
    --model-name ${MODEL_PATH} \
    --dataset-dir ${DATASET_DIR} \
    --save-path ${SAVE_PATH_2} \
    --trust-remote-code \
    --sample-json-token mid_first \
    2>&1 | tee -a ${LOG_BASE_DIR}/step2_count_expert.log

mkdir -p ${SAVE_PATH_3}
python3 step3_count_expert_dist.py \
    --jsonl ${SAVE_PATH_2}/token_routing.jsonl \
    --model-name ${MODEL_PATH} \
    --out-path ${SAVE_PATH_3} \
    --trust-remote-code \
    2>&1 | tee -a ${LOG_BASE_DIR}/step3_count_expert_dist.log

mkdir -p ${SAVE_PATH_4}
python3 step4_weight_outlier_dist.py \
    --jsonl ${SAVE_PATH_2}/token_routing.jsonl \
    --model-name ${MODEL_PATH} \
    --out-path ${SAVE_PATH_4} \
    --scatter-x linear \
    2>&1 | tee -a ${LOG_BASE_DIR}/step4_weight_outlier_dist.log
