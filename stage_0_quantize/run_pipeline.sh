#!/usr/bin/env bash
# ============================================================
#  ModelOpt GPTQ → TensorRT-LLM → NeMo Evaluator
#  Execute the entire pipeline script.
# ============================================================
set -e
set -o pipefail

STAGE="0"
CALIB_SIZE=128

QUANTIZE="GPTQ_W4A16" # UNQUANTIZED, GPTQ_W4A16, RTN_W4A16

FULL_GPU_DEVICES="0,1,2,3,4,5,6,7"
TARGET_GPU_DEVICES="7"

PORT=6050

export CUDA_VISIBLE_DEVICES=$TARGET_GPU_DEVICES

# Step 3 lm_eval: EleutherAI `openai-chat-completions` requires an Authorization key (local server is unverified).
export OPENAI_API_KEY="${OPENAI_API_KEY:-local-dummy}"
# If the request thread count exceeds the upper limit, "NUMEXPR_MAX_THREADS" warning will be issued.
export NUMEXPR_MAX_THREADS="${NUMEXPR_MAX_THREADS:-128}"


MODEL_NAME="qwen3_30b_a3b"
MODEL_BASE_PATH="/your_base_model_path"
MODEL=${MODEL_BASE_PATH}/${MODEL_NAME}

DATASET_DIR="./dataset_dir/D${STAGE}_${CALIB_SIZE}"
mkdir -p $DATASET_DIR


Q_MODEL_NAME="Q${STAGE}_${CALIB_SIZE}-${MODEL_NAME}-${QUANTIZE}"
QUANT_BASE="../../models/q_models"
QUANT_DIR="$QUANT_BASE/$Q_MODEL_NAME"
MAX_INPUT_LENGTH=5120
MAX_OUTPUT_LENGTH=6144
ENGINE_BASE="../../models/trt_engines"
ENGINE_DIR="$ENGINE_BASE/${Q_MODEL_NAME}_${MAX_INPUT_LENGTH}_${MAX_OUTPUT_LENGTH}"                    # TRT-LLM 엔진 저장
# TRT-LLM Server uses the engine folder basename as the model id for /v1/models → step3 --model_name must match.
API_MODEL_NAME="$(basename "$ENGINE_DIR")"
#EVAL_MODE="nel_core" # nel_core, lm_eval
EVAL_MODE="lm_eval"
# step3: fewshot_cot = GSM8K·GPQA Diamond·MMLU-Pro few-shot CoT (mode-specific registered names). Or comma-separated list.
TASKS="fewshot_cot" # fewshot_cot, fewshot_cot_basic
RESULTS_DIR="../../eval_results/${Q_MODEL_NAME}_${MAX_INPUT_LENGTH}_${MAX_OUTPUT_LENGTH}/${EVAL_MODE}_${TASKS}"                 # 벤치마크 결과
LOG_DIR="../../log/${Q_MODEL_NAME}_${MAX_INPUT_LENGTH}_${MAX_OUTPUT_LENGTH}_${EVAL_MODE}_${TASKS}"





# nel_core + fewshot_cot(ADLR) allows only completions → pass completions to step3.
ENDPOINT_TYPE="chat"
if [ "$EVAL_MODE" = "nel_core" ] && [ "$TASKS" = "fewshot_cot" ]; then
  ENDPOINT_TYPE="completions"
fi

mkdir -p $QUANT_DIR
mkdir -p $ENGINE_DIR
mkdir -p $RESULTS_DIR
mkdir -p $LOG_DIR


# ── 환경 체크 ──────────────────────────────────────────────
echo "=============================="
echo " Environment Check"
echo "=============================="
python -c "import torch; print(f'PyTorch: {torch.__version__} | CUDA: {torch.cuda.is_available()}')"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""


if [ "$STAGE" -eq 0 ]; then
  # ── Step 1: Dataset Gen (D0) ────────────────────────────────────
  echo "=============================="
  echo " Step 1: Dataset Gen (D0)"
  echo "=============================="
  python step1_gptq_quantize.py \
      --model "$MODEL" \
      --dataset_dir "$DATASET_DIR" \
      --calib_size "$CALIB_SIZE" \
      --quantize CALIB_DATASET_ONLY \
      2>&1 | tee -a $LOG_DIR/dataset_gen.log

  status=${PIPESTATUS[0]}
  if [ $status -ne 0 ]; then
    echo "[Step 1] Error occurred during dataset generation. Check $LOG_DIR/dataset_gen.log for details."
    exit $status
  fi

  echo "[Step 1] Done: $DATASET_DIR"
  echo ""
fi


# ── Step 1: Hessian Generation ────────────────────────────────────
echo "=============================="
echo " Step 1: Hessian Generation"
echo "=============================="
python step1_gptq_quantize.py \
    --model "$MODEL" \
    --dataset_dir "$DATASET_DIR" \
    --calib_size "$CALIB_SIZE" \
    --quantize HESSIAN \
    2>&1 | tee -a $LOG_DIR/hessian_generation.log

echo "[Step 1] Done: ${DATASET_DIR}/${MODEL_NAME}-hessian"
echo ""


export CUDA_VISIBLE_DEVICES=$FULL_GPU_DEVICES

# ── Step 1: GPTQ Quantization ────────────────────────────────────
echo "=============================="
echo " Step 1: GPTQ Quantization"
echo "=============================="
python step1_gptq_quantize.py \
    --model "$MODEL" \
    --save_dir "$QUANT_DIR" \
    --calib_size "$CALIB_SIZE" \
    --dataset_dir "$DATASET_DIR" \
    --quantize "$QUANTIZE" \
    --parallel_gptq_batch 8 \
    --gptq_require_cached_hessian \
    2>&1 | tee -a $LOG_DIR/quantization.log

echo "[Step 1] Done: $QUANT_DIR"
echo ""




export CUDA_VISIBLE_DEVICES=$TARGET_GPU_DEVICES


# ── Step 2a: TRT-LLM Engine Build ────────────────────────────
echo "=============================="
echo " Step 2a: TRT-LLM Engine Build"
echo "=============================="
python step2_trtllm_build_serve.py \
    --mode build \
    --quantized_dir "$QUANT_DIR" \
    --engine_dir "$ENGINE_DIR" \
    --max_input_len "$MAX_INPUT_LENGTH" \
    --max_output_len "$MAX_OUTPUT_LENGTH" \
    2>&1 | tee $LOG_DIR/trtllm_engine_build.log

echo "[Step 2a] Done: $ENGINE_DIR"
echo ""


# ── Step 2b: Inference Sanity Check ──────────────────────────────
echo "=============================="
echo " Step 2b: Inference Sanity Check"
echo "=============================="
python step2_trtllm_build_serve.py \
    --mode test \
    --engine_dir "$ENGINE_DIR" \
    --tokenizer "$MODEL" \
    2>&1 | tee $LOG_DIR/inference_sanity_check.log
echo ""

# ── Step 2c: Start TRT-LLM Server ──────────────────────────
# Note: Using `python | tee &` will set $! to the tee process, preventing proper termination of the inference server.
# To ensure $! points to the python process, redirect logs directly and do not use tee in the background.
echo "=============================="
echo " Step 2c: Start TRT-LLM Server"
echo "=============================="
SERVER_PID=""
cleanup_trtllm_server() {
    # TRT-LLM Server is the python process, not the tee process.
    # For legacy subprocess worker cleanup: PGID signal + port cleanup.
    if [[ -z "${SERVER_PID:-}" ]]; then
        return 0
    fi
    if kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "[Server] Stopping TRT-LLM (PID $SERVER_PID)..."
        pgid=$(ps -o pgid= -p "$SERVER_PID" 2>/dev/null | tr -d ' ')
        if [[ -n "$pgid" ]] && [[ "$pgid" =~ ^[0-9]+$ ]]; then
            kill -TERM "-$pgid" 2>/dev/null || true
        fi
        kill -TERM "$SERVER_PID" 2>/dev/null || true
        sleep 1
        kill -KILL "$SERVER_PID" 2>/dev/null || true
        if [[ -n "${pgid:-}" ]] && [[ "$pgid" =~ ^[0-9]+$ ]]; then
            kill -KILL "-$pgid" 2>/dev/null || true
        fi
        wait "$SERVER_PID" 2>/dev/null || true
    fi
    if command -v fuser >/dev/null 2>&1; then
        fuser -k "${PORT}/tcp" 2>/dev/null || true
    fi
    echo "[Server] Stopped."
}
trap cleanup_trtllm_server EXIT

python step2_trtllm_build_serve.py \
    --mode serve \
    --model_name "$Q_MODEL_NAME" \
    --engine_dir "$ENGINE_DIR" \
    --tokenizer "$MODEL" \
    --port "$PORT" \
    --max_input_len "$MAX_INPUT_LENGTH" \
    --max_output_len "$MAX_OUTPUT_LENGTH" \
    >>"$LOG_DIR/start_trtllm_server.log" 2>&1 &
SERVER_PID=$!

echo "[Server] PID: $SERVER_PID (log: $LOG_DIR/start_trtllm_server.log)"
echo "[Server] Waiting for TRT-LLM server to become ready (max 5min)..."
MAX_WAIT_SEC=300
WAIT_INTERVAL_SEC=30
elapsed=0

while true; do
    if curl -sf "http://localhost:$PORT/v1/models" | python -m json.tool > /dev/null 2>&1; then
        echo "[Server] Server is up!"
        curl -sf "http://localhost:$PORT/v1/models" | python -m json.tool
        echo ""
        break
    fi

    if (( elapsed >= MAX_WAIT_SEC )); then
        echo "[Error] Server did not become ready within $MAX_WAIT_SEC seconds. Check logs."
        exit 1
    fi

    echo "  ...waiting ($elapsed/$MAX_WAIT_SEC sec)..."
    sleep $WAIT_INTERVAL_SEC
    ((elapsed+=WAIT_INTERVAL_SEC))
done

# ── Step 3: NeMo Evaluator Benchmark ───────────────────────
echo "=============================="
echo " Step 3: NeMo Evaluator"
echo "=============================="
python step3_nemo_eval.py \
    --mode "$EVAL_MODE" \
    --server_url "http://localhost:$PORT/v1" \
    --model_name "$API_MODEL_NAME" \
    --output_dir "$RESULTS_DIR" \
    --tasks "$TASKS" \
    --endpoint_type "$ENDPOINT_TYPE" \
    --max_tokens "$MAX_OUTPUT_LENGTH" \
    --temperature 0.0 \
    --top_p 1.0 \
    --mmlu_pro_limit 400 \
    2>&1 | tee $LOG_DIR/nemo_eval.log

echo ""
echo "=============================="
echo " Pipeline Complete!"
echo " Results: $RESULTS_DIR"
echo "=============================="
# Server cleanup is performed in the EXIT trap (cleanup_trtllm_server).
