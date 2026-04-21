#!/usr/bin/env bash
# ============================================================
#  Stage 0: Quantize initial model and evaluate on benchmark tasks
#  ModelOpt GPTQ → TensorRT-LLM engine → NeMo Evaluator
# ============================================================
set -e
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
SRC_DIR="${REPO_ROOT}/src/stage0"

# ── Configuration ──────────────────────────────────────────
STAGE="0"
CALIB_SIZE=128

QUANTIZE="GPTQ_W4A16"          # UNQUANTIZED | GPTQ_W4A16 | RTN_W4A16

FULL_GPU_DEVICES="0,1,2,3,4,5,6,7"
TARGET_GPU_DEVICES="7"

PORT=6050

export CUDA_VISIBLE_DEVICES=$TARGET_GPU_DEVICES
export OPENAI_API_KEY="${OPENAI_API_KEY:-local-dummy}"
export NUMEXPR_MAX_THREADS="${NUMEXPR_MAX_THREADS:-128}"

MODEL_NAME="qwen3_30b_a3b"
MODEL_BASE_PATH="/your_base_model_path"   # <-- set this
MODEL="${MODEL_BASE_PATH}/${MODEL_NAME}"

CALIB_DATASET_DIR="${REPO_ROOT}/model/calib_dataset/D${STAGE}_${CALIB_SIZE}"
mkdir -p "$CALIB_DATASET_DIR"

Q_MODEL_NAME="Q${STAGE}_${CALIB_SIZE}-${MODEL_NAME}-${QUANTIZE}"
QUANT_DIR="${REPO_ROOT}/model/quantized/${Q_MODEL_NAME}"
MAX_INPUT_LENGTH=5120
MAX_OUTPUT_LENGTH=6144
ENGINE_DIR="${REPO_ROOT}/model/trt_engine/${Q_MODEL_NAME}_${MAX_INPUT_LENGTH}_${MAX_OUTPUT_LENGTH}"
API_MODEL_NAME="$(basename "$ENGINE_DIR")"

EVAL_MODE="lm_eval"             # lm_eval | nel_core
TASKS="fewshot_cot"             # fewshot_cot | fewshot_cot_basic
ENDPOINT_TYPE="chat"
if [ "$EVAL_MODE" = "nel_core" ] && [ "$TASKS" = "fewshot_cot" ]; then
  ENDPOINT_TYPE="completions"
fi

RESULTS_DIR="${REPO_ROOT}/model/eval_results/${Q_MODEL_NAME}_${MAX_INPUT_LENGTH}_${MAX_OUTPUT_LENGTH}/${EVAL_MODE}_${TASKS}"
LOG_DIR="${REPO_ROOT}/model/eval_results/${Q_MODEL_NAME}_${MAX_INPUT_LENGTH}_${MAX_OUTPUT_LENGTH}/log"

mkdir -p "$QUANT_DIR" "$ENGINE_DIR" "$RESULTS_DIR" "$LOG_DIR"

# ── Environment check ──────────────────────────────────────
echo "=============================="
echo " Environment Check"
echo "=============================="
python -c "import torch; print(f'PyTorch: {torch.__version__} | CUDA: {torch.cuda.is_available()}')"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# ── Step 1a: Calibration dataset generation (STAGE=0 only) ─
if [ "$STAGE" -eq 0 ]; then
  echo "=============================="
  echo " Step 1a: Calibration Dataset Generation"
  echo "=============================="
  python "${SRC_DIR}/step1_gptq_quantize.py" \
      --model "$MODEL" \
      --dataset_dir "$CALIB_DATASET_DIR" \
      --calib_size "$CALIB_SIZE" \
      --quantize CALIB_DATASET_ONLY \
      2>&1 | tee -a "${LOG_DIR}/step1a_dataset_gen.log"

  status=${PIPESTATUS[0]}
  if [ $status -ne 0 ]; then
    echo "[Step 1a] Error. Check ${LOG_DIR}/step1a_dataset_gen.log"
    exit $status
  fi
  echo "[Step 1a] Done: ${CALIB_DATASET_DIR}"
  echo ""
fi

# ── Step 1b: Hessian generation ────────────────────────────
echo "=============================="
echo " Step 1b: Hessian Generation"
echo "=============================="
python "${SRC_DIR}/step1_gptq_quantize.py" \
    --model "$MODEL" \
    --dataset_dir "$CALIB_DATASET_DIR" \
    --calib_size "$CALIB_SIZE" \
    --quantize HESSIAN \
    2>&1 | tee -a "${LOG_DIR}/step1b_hessian.log"
echo "[Step 1b] Done"
echo ""

# ── Step 1c: GPTQ quantization (multi-GPU) ─────────────────
export CUDA_VISIBLE_DEVICES=$FULL_GPU_DEVICES
echo "=============================="
echo " Step 1c: GPTQ Quantization"
echo "=============================="
python "${SRC_DIR}/step1_gptq_quantize.py" \
    --model "$MODEL" \
    --save_dir "$QUANT_DIR" \
    --calib_size "$CALIB_SIZE" \
    --dataset_dir "$CALIB_DATASET_DIR" \
    --quantize "$QUANTIZE" \
    --parallel_gptq_batch 8 \
    --gptq_require_cached_hessian \
    2>&1 | tee -a "${LOG_DIR}/step1c_quantization.log"
echo "[Step 1c] Done: ${QUANT_DIR}"
echo ""

export CUDA_VISIBLE_DEVICES=$TARGET_GPU_DEVICES

# ── Step 2a: TRT-LLM engine build ──────────────────────────
echo "=============================="
echo " Step 2a: TRT-LLM Engine Build"
echo "=============================="
python "${SRC_DIR}/step2_trtllm_build_serve.py" \
    --mode build \
    --quantized_dir "$QUANT_DIR" \
    --engine_dir "$ENGINE_DIR" \
    --max_input_len "$MAX_INPUT_LENGTH" \
    --max_output_len "$MAX_OUTPUT_LENGTH" \
    2>&1 | tee "${LOG_DIR}/step2a_engine_build.log"
echo "[Step 2a] Done: ${ENGINE_DIR}"
echo ""

# ── Step 2b: Inference sanity check ────────────────────────
echo "=============================="
echo " Step 2b: Inference Sanity Check"
echo "=============================="
python "${SRC_DIR}/step2_trtllm_build_serve.py" \
    --mode test \
    --engine_dir "$ENGINE_DIR" \
    --tokenizer "$MODEL" \
    2>&1 | tee "${LOG_DIR}/step2b_sanity_check.log"
echo ""

# ── Step 2c: Start TRT-LLM server ──────────────────────────
echo "=============================="
echo " Step 2c: Start TRT-LLM Server"
echo "=============================="
SERVER_PID=""
cleanup_trtllm_server() {
    if [[ -z "${SERVER_PID:-}" ]]; then return 0; fi
    if kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "[Server] Stopping TRT-LLM (PID $SERVER_PID)..."
        pgid=$(ps -o pgid= -p "$SERVER_PID" 2>/dev/null | tr -d ' ')
        if [[ -n "$pgid" ]] && [[ "$pgid" =~ ^[0-9]+$ ]]; then
            kill -TERM "-$pgid" 2>/dev/null || true
        fi
        kill -TERM "$SERVER_PID" 2>/dev/null || true
        sleep 1
        kill -KILL "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
    if command -v fuser >/dev/null 2>&1; then
        fuser -k "${PORT}/tcp" 2>/dev/null || true
    fi
    echo "[Server] Stopped."
}
trap cleanup_trtllm_server EXIT

python "${SRC_DIR}/step2_trtllm_build_serve.py" \
    --mode serve \
    --model_name "$Q_MODEL_NAME" \
    --engine_dir "$ENGINE_DIR" \
    --tokenizer "$MODEL" \
    --port "$PORT" \
    --max_input_len "$MAX_INPUT_LENGTH" \
    --max_output_len "$MAX_OUTPUT_LENGTH" \
    >>"${LOG_DIR}/step2c_server.log" 2>&1 &
SERVER_PID=$!

echo "[Server] PID: ${SERVER_PID} | Waiting for readiness (max 5 min)..."
MAX_WAIT_SEC=300; WAIT_INTERVAL_SEC=30; elapsed=0
while true; do
    if curl -sf "http://localhost:${PORT}/v1/models" | python -m json.tool > /dev/null 2>&1; then
        echo "[Server] Server is up!"
        curl -sf "http://localhost:${PORT}/v1/models" | python -m json.tool
        echo ""
        break
    fi
    if (( elapsed >= MAX_WAIT_SEC )); then
        echo "[Error] Server not ready within ${MAX_WAIT_SEC}s. Check ${LOG_DIR}/step2c_server.log"
        exit 1
    fi
    echo "  ...waiting (${elapsed}/${MAX_WAIT_SEC}s)..."
    sleep $WAIT_INTERVAL_SEC
    ((elapsed+=WAIT_INTERVAL_SEC))
done

# ── Step 3: NeMo Evaluator benchmark ───────────────────────
echo "=============================="
echo " Step 3: NeMo Evaluator Benchmark"
echo "=============================="
python "${SRC_DIR}/step3_nemo_eval.py" \
    --mode "$EVAL_MODE" \
    --server_url "http://localhost:${PORT}/v1" \
    --model_name "$API_MODEL_NAME" \
    --output_dir "$RESULTS_DIR" \
    --tasks "$TASKS" \
    --endpoint_type "$ENDPOINT_TYPE" \
    --max_tokens "$MAX_OUTPUT_LENGTH" \
    --temperature 0.0 \
    --top_p 1.0 \
    --mmlu_pro_limit 400 \
    2>&1 | tee "${LOG_DIR}/step3_nemo_eval.log"

echo ""
echo "=============================="
echo " Stage 0 Complete!"
echo " Results : ${RESULTS_DIR}"
echo " Logs    : ${LOG_DIR}"
echo "=============================="
