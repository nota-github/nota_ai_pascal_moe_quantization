# Sourced by run1_first.sh … run4_fourth.sh (same defaults as pipeline.sh).

export CUDA_VISIBLE_DEVICES=4

STAGE="0"
NUM_SAMPLES=128
CALIB_SIZE=$NUM_SAMPLES

MODEL_BASE_PATH="/home/work/nota-data/ghlee/storage/base_model"
MODEL_NAME="qwen3_30b_a3b"
MODEL_PATH="${MODEL_BASE_PATH}/${MODEL_NAME}"

#NUM_SAMPLES_PER_DOMAIN=128
NUM_SAMPLES_PER_DOMAIN=$((CALIB_SIZE / 4))
MIN_LENGTH=1024
MAX_LENGTH=2048

DATASET_ID="nemo_dataset" # nemo_dataset, custom

DATASET_DIR="/home/work/nota-data/ghlee/nemo_hack/quant_pipe/dataset_dir/D${STAGE}_${CALIB_SIZE}"

SAVE_BASE_PATH="/home/work/nota-data/ghlee/nemo_hack/expert_balance/save_dir/"
SAVE_PATH="${SAVE_BASE_PATH}/${MODEL_NAME}_${DATASET_ID}/D${STAGE}_${CALIB_SIZE}"

LOG_BASE_DIR="${SAVE_PATH}/log/"
mkdir -p ${LOG_BASE_DIR}

echo "=========================================="
echo "STAGE: ${STAGE}"
echo "MODEL_NAME: ${MODEL_NAME}"
echo "DATASET_DIR: ${DATASET_DIR}"
echo "SAVE_PATH: ${SAVE_PATH}"
echo "=========================================="

SAVE_PATH_1="${SAVE_PATH}/s1_dataset_dir"
SAVE_PATH_2="${SAVE_PATH}/s2_expert_count"
SAVE_PATH_3="${SAVE_PATH}/s3_expert_dist"
SAVE_PATH_4="${SAVE_PATH}/s4_weight_outlier"
SAVE_PATH_5="${SAVE_PATH}/s5_sorted_token"
SAVE_PATH_6="${SAVE_PATH}/s6_apply_bracket"

# Second run — expert thresholds (BALANCE / Q_SENSITIVITY)
freq_thr=0.25
less_thr=0.25
rob_thr=0.25
sen_thr=0.25

# Third run — token thresholds (balance / q_sensitivity)
# b: token_blue = freq>btx & less<bty ; token_red = freq<rtx & less>rty (x=freq, y=less)
balance_thr_blue_x=0.1
balance_thr_blue_y=0.77
balance_thr_red_x=0.15
balance_thr_red_y=0.6

# q: token_blue = sensitive<btx & robust>bty ; token_red = sensitive>rtx & robust<rty (x=sensitive, y=robust)
q_sensitive_thr_blue_x=0.2
q_sensitive_thr_blue_y=0.2
q_sensitive_thr_red_x=0.35
q_sensitive_thr_red_y=0.25

# Fourth run — optional debug limit (unset = no --limit; pipeline.sh had LIMIT commented)
# LIMIT=32
