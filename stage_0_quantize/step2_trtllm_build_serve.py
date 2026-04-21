"""
Step 2: TensorRT-LLM 엔진 빌드 & 서버 실행
==========================================
- Step 1에서 생성한 GPTQ quantized checkpoint를 TRT-LLM 엔진으로 변환
- Triton Inference Server 또는 trtllm-serve로 서빙
- 설치: pip install tensorrt-llm  (또는 NGC 컨테이너 권장)
"""

import os
import shlex
import subprocess
import json
import argparse
from pathlib import Path
import time
import torch


# ──────────────────────────────────────────────
# 설정
# ──────────────────────────────────────────────
QUANTIZED_DIR    = "./quantized_model"      # Step 1 출력 경로
ENGINE_DIR       = "./trt_engine"           # TRT 엔진 저장 경로
SERVE_PORT       = 8000                     # 서버 포트
MAX_BATCH_SIZE   = 1
MAX_INPUT_LEN    = 2048
MAX_OUTPUT_LEN   = 512
# 멀티 GPU: trtllm-build는 --tp_size/--pp_size 미지원. quantized checkpoint의 config.json
# `mapping`(tp_size, pp_size, world_size)을 수정한 뒤 빌드.


def resolve_gemm_plugin(quantized_dir: str, override: str | None) -> str:
    """
    checkpoint config.json의 dtype과 맞춤. --gemm_plugin float16 인데 모델이 bfloat16이면
    lm_head Gemm 등에서 'could not find any supported formats' (dtype 불일치)가 날 수 있음.
    """
    if override:
        return override
    cfg_path = Path(quantized_dir) / "config.json"
    if not cfg_path.is_file():
        return "bfloat16"
    with open(cfg_path, encoding="utf-8") as f:
        dt = json.load(f).get("dtype", "bfloat16")
    s = str(dt).lower()
    if s == "bfloat16":
        return "bfloat16"
    if s in ("float16", "fp16", "half"):
        return "float16"
    if s in ("float32", "fp32"):
        return "float32"
    return "auto"


def _dir_has_hf_tokenizer(path: Path) -> bool:
    return path.is_dir() and (
        (path / "tokenizer.json").is_file()
        or (path / "tokenizer_config.json").is_file()
    )


def resolve_tokenizer_path(
    engine_dir: str,
    engine_config: dict,
    explicit: str | None,
) -> str:
    """
    TRT 엔진 config에는 tokenizer_dir가 없는 경우가 많다.
    engine_dir만으로는 tokenizer 파일이 없어 AutoTokenizer가 실패한다.
    """
    if explicit:
        return explicit
    for env_key in ("TRTLLM_TOKENIZER", "HF_TOKENIZER_ID", "HF_MODEL_ID"):
        v = os.environ.get(env_key)
        if v:
            return v
    td = engine_config.get("tokenizer_dir")
    if td:
        return str(td)
    pc = engine_config.get("pretrained_config") or {}
    td = pc.get("tokenizer_dir")
    if td:
        return str(td)

    eng = Path(engine_dir)
    if _dir_has_hf_tokenizer(eng):
        return str(eng)

    # Step 1 출력과 형제인 경우
    qm = eng.parent / "quantized_model"
    if _dir_has_hf_tokenizer(qm):
        return str(qm)

    raise ValueError(
        "엔진 디렉터리에 HuggingFace 토크나이저(tokenizer.json 등)가 없습니다.\n"
        "  --tokenizer <HF 모델 ID 또는 로컬 경로> 를 주거나,\n"
        "  환경변수 TRTLLM_TOKENIZER / HF_MODEL_ID 를 설정하세요.\n"
        "  예: --tokenizer Qwen/Qwen3-30B-A3B-Instruct-2507"
    )


def run_cmd(cmd: str, desc: str = "") -> subprocess.CompletedProcess:
    """쉘 명령 실행 및 출력 스트리밍."""
    if desc:
        print(f"\n>> {desc}")
    print(f"   $ {cmd}\n")
    result = subprocess.run(
        cmd, shell=True, check=True,
        text=True, capture_output=False,
    )
    return result


def inject_dummy_prequant_for_awq_checkpoint(quantized_dir: str) -> None:
    """
    ModelOpt TRT export는 config상 W4A16_AWQ + pre_quant_scale=true 인데
    safetensors에 prequant_scaling_factor가 빠진 경우가 있다.

    config를 W4A16_GPTQ로 바꾸면 TRT-LLM이 GPTQ용 가중치 레이아웃을 기대해
    qkv.weight 등에서 형상 불일치(예: (2560,2048) vs (2048,1280))가 난다.
    가중치는 AWQ(그룹 스케일) 포맷이므로 메타는 AWQ를 유지하고,
    각 *.weights_scaling_factor 옆에 (1, in_features) 더미 prequant만 채운다.
    in_features = scale 텐서 마지막 차원 * group_size (TRT groupwise 정의와 맞춤).
    """
    cfg_path = Path(quantized_dir) / "config.json"
    st_path = Path(quantized_dir) / "rank0.safetensors"
    if not cfg_path.is_file() or not st_path.is_file():
        return

    with open(cfg_path, encoding="utf-8") as f:
        cfg = json.load(f)
    q = cfg.get("quantization") or {}
    if q.get("quant_algo") != "W4A16_AWQ" or not q.get("pre_quant_scale", False):
        return

    try:
        from safetensors.torch import load_file, save_file
    except ImportError:
        print("[Checkpoint] safetensors 미설치: prequant 주입 생략 (pip install safetensors)")
        return

    def _cfg_to_torch_dtype(s: str | None):
        s = (s or "bfloat16").lower()
        if s == "bfloat16":
            return torch.bfloat16
        if s in ("float16", "fp16"):
            return torch.float16
        return torch.float32

    pq_dtype = _cfg_to_torch_dtype(cfg.get("dtype"))

    tensors = load_file(str(st_path))
    group_size = int(q.get("group_size") or 128)
    added = 0
    for name in list(tensors.keys()):
        if not name.endswith(".weights_scaling_factor"):
            continue
        prefix = name[: -len(".weights_scaling_factor")]
        pq_key = prefix + ".prequant_scaling_factor"
        wsf = tensors[name]
        in_features = int(wsf.shape[-1]) * group_size
        if pq_key in tensors:
            # 이미 bf16 등으로 맞으면 스킵 (실제 export와 충돌 방지)
            if tensors[pq_key].dtype == pq_dtype:
                continue
            # 예전에 float32 더미로 넣은 경우 → TRT 플러그인 dtype 불일치 수정
            if tensors[pq_key].dtype != torch.float32:
                continue
        tensors[pq_key] = torch.ones(1, in_features, dtype=pq_dtype)
        added += 1

    if added == 0:
        return

    save_file(tensors, str(st_path))
    print(
        f"[Checkpoint] AWQ 누락 보정: prequant_scaling_factor {added}개 설정 "
        f"(group_size={group_size}, dtype={pq_dtype}, 값=1.0)"
    )


# ──────────────────────────────────────────────
# 2-1. trtllm-build: 엔진 컴파일
# ──────────────────────────────────────────────
def build_trtllm_engine(quantized_dir: str = QUANTIZED_DIR,
                         engine_dir: str = ENGINE_DIR,
                         gemm_plugin: str | None = None,
                         max_input_len: int = MAX_INPUT_LEN,
                         max_output_len: int = MAX_OUTPUT_LEN):
    """
    trtllm-build CLI로 TensorRT-LLM 엔진 빌드.

    TensorRT-LLM 1.2.x CLI 기준:
      - GPTQ/INT4 설정은 checkpoint(config.json)에 포함되므로 별도 --use_weight_only 불필요
      - --use_fused_mlp 는 enable|disable 인자 필요
      - --kv_cache_type paged, --tokens_per_block 으로 paged KV 설정
      - --max_seq_len 은 프롬프트+생성 최대 길이(기존 max_input_len + max_output_len)
      - TP/PP는 trtllm-build 인자가 아니라 checkpoint의 config.json mapping을 사용 (멀티 GPU 시 checkpoint 수정)
    NUMEXPR_MAX_THREADS 경고가 나오면 빌드 전에 export NUMEXPR_MAX_THREADS=64 등으로 맞춤.
    """
    print(f"\n{'='*60}")
    print(f"  TensorRT-LLM Engine Build")
    print(f"  Input : {quantized_dir}")
    print(f"  Output: {engine_dir}")
    print(f"{'='*60}\n")

    Path(engine_dir).mkdir(parents=True, exist_ok=True)
    inject_dummy_prequant_for_awq_checkpoint(quantized_dir)

    gp = resolve_gemm_plugin(quantized_dir, gemm_plugin)
    print(f"  --gemm_plugin {gp} (checkpoint dtype와 일치시킴)\n")

    max_seq_len = max_input_len + max_output_len
    build_cmd = f"""trtllm-build \
        --checkpoint_dir {quantized_dir} \
        --output_dir {engine_dir} \
        --gemm_plugin {gp} \
        --kv_cache_type paged \
        --tokens_per_block 64 \
        --use_fused_mlp enable \
        --context_fmha enable \
        --max_batch_size {MAX_BATCH_SIZE} \
        --max_input_len {max_input_len} \
        --max_seq_len {max_seq_len} \
        --max_num_tokens {max_input_len * MAX_BATCH_SIZE} \
        --workers 1
    """

    run_cmd(build_cmd, "Building TensorRT-LLM engine...")
    print(f"\n[Done] Engine saved to: {engine_dir}")
    print(f"       Files: {list(Path(engine_dir).iterdir())}")


# ──────────────────────────────────────────────
# 2-2. Python API: 직접 추론 테스트
# ──────────────────────────────────────────────
def test_inference_python_api(
    engine_dir: str = ENGINE_DIR,
    tokenizer_path: str | None = None,
):
    """
    TensorRT-LLM Python API로 직접 추론 테스트.
    서버 없이 빠르게 엔진 동작 확인 가능.
    """
    print(f"\n{'='*60}")
    print("  TensorRT-LLM Python API Inference Test")
    print(f"{'='*60}\n")

    # tensorrt_llm 임포트 (엔진 빌드 환경에서만 사용 가능)
    import tensorrt_llm
    from tensorrt_llm.runtime import ModelRunnerCpp
    from transformers import AutoTokenizer

    # 엔진 설정 파일에서 tokenizer 경로 읽기
    config_path = Path(engine_dir) / "config.json"
    with open(config_path) as f:
        engine_config = json.load(f)

    model_name = engine_config.get("pretrained_config", {}).get("architecture", "LlamaForCausalLM")
    print(f"[Info] Loaded engine config: {model_name}")

    tok_path = resolve_tokenizer_path(engine_dir, engine_config, tokenizer_path)
    print(f"[Info] Tokenizer: {tok_path}")

    # Tokenizer 로드
    tokenizer = AutoTokenizer.from_pretrained(
        tok_path,
        trust_remote_code=True,
    )

    # Runner 초기화
    runner = ModelRunnerCpp.from_dir(
        engine_dir=engine_dir,
        rank=0,
        max_batch_size=MAX_BATCH_SIZE,
    )

    # 테스트 프롬프트
    prompts = [
        "Explain the concept of quantization in machine learning.",
        "What are the advantages of GPTQ over other quantization methods?",
        "Write a Python function to compute Fibonacci numbers.",
    ]

    print(f"\n[Inference] Running {len(prompts)} test prompts...\n")

    eos_id = tokenizer.eos_token_id
    for i, prompt in enumerate(prompts, 1):
        input_ids = tokenizer.encode(prompt, return_tensors="pt")
        # generate()가 각 배치 항목에 대해 .tolist() 호출 → torch.Tensor만 전달 (list 금지)
        outputs = runner.generate(
            batch_input_ids=[input_ids[0]],
            max_new_tokens=200,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            end_id=eos_id,
        )
        # return_dict=False → (batch, num_sequences, max_seq_len); 슬롯에는 생성 토큰만 + end_id 패딩
        gen_row = outputs[0, 0].tolist()
        while gen_row and eos_id is not None and gen_row[-1] == eos_id:
            gen_row.pop()
        response = tokenizer.decode(gen_row, skip_special_tokens=True)
        print(f"--- Prompt {i} ---")
        print(f"Input : {prompt}")
        print(f"Output: {response.strip()}\n")


# ──────────────────────────────────────────────
# 2-3. trtllm-serve: OpenAI-compatible REST API 서버
# ──────────────────────────────────────────────
def start_trtllm_serve(engine_dir: str = ENGINE_DIR,
                        model_name: str = "llama-3.1-8b-gptq",
                        port: int = SERVE_PORT,
                        tokenizer: str | None = None):
    """
    trtllm-serve로 OpenAI-compatible 서버 시작.
    NeMo Evaluator가 이 엔드포인트에 연결합니다.

    TensorRT-LLM 1.2+: `trtllm-serve serve MODEL --backend tensorrt` 형식이며,
    MODEL은 엔진 디렉터리 경로이다. /v1/models 에 노출되는 model id는
    존재하는 디렉터리 경로면 그 폴더 이름(basename)이다.

    TensorRT 백엔드는 엔진만으로 HF 토크나이저가 로드되지 않는 경우가 있어
    `--tokenizer`(베이스 모델 로컬 경로 또는 HF ID)를 반드시 넘겨야 한다.

    백그라운드 파이프라인에서 부모 PID만 kill 되는 문제를 피하기 위해
    `subprocess` 대신 `os.execvp`로 현재 프로세스를 `trtllm-serve`로 교체한다
    (같은 PID가 곧 서버 프로세스).

    서버 실행 후 테스트 (model 은 엔진 폴더 basename, 예: quantized_경로와 동일):
      curl http://localhost:<PORT>/v1/chat/completions \\
        -H "Content-Type: application/json" \\
        -d '{"model": "<engine_dir_basename>", "messages": [{"role": "user", "content": "Hello!"}]}'
    """
    print(f"\n{'='*60}")
    print(f"  Starting TensorRT-LLM Server")
    print(f"  Engine : {engine_dir}")
    print(f"  Port   : {port}")
    advertised = Path(engine_dir).name if Path(engine_dir).is_dir() else model_name
    print(f"  OpenAI model id (expected): {advertised}")
    if tokenizer:
        print(f"  Tokenizer: {tokenizer}")
    print(f"{'='*60}\n")

    if not tokenizer:
        raise SystemExit(
            "[serve] TensorRT 엔진 서빙 시 --tokenizer 가 필요합니다. "
            "예: 베이스 HF 모델 로컬 경로 또는 모델 ID (test 모드와 동일)."
        )

    # TRT-LLM 1.2+: 서브커맨드 `serve`, 엔진 경로는 positional MODEL, --backend tensorrt 필수
    argv = [
        "trtllm-serve",
        "serve",
        str(engine_dir),
        "--backend",
        "tensorrt",
        "--tokenizer",
        tokenizer,
        "--trust_remote_code",
        "--port",
        str(port),
        "--max_batch_size",
        str(MAX_BATCH_SIZE),
        "--max_beam_width",
        "1",
        "--free_gpu_memory_fraction",
        "0.9",
    ]
    print(">> Launching trtllm-serve (exec, no subprocess wrapper)")
    print(f"   $ {' '.join(shlex.quote(a) for a in argv)}\n")
    print(f"[Server] Starting... (Ctrl+C to stop)")
    print(f"[Server] OpenAI endpoint: http://localhost:{port}/v1")
    try:
        os.execvp("trtllm-serve", argv)
    except OSError as e:
        raise SystemExit(f"[serve] trtllm-serve 실행 실패: {e}") from e


# ──────────────────────────────────────────────
# 2-4. Triton 방식 서버 (대규모 배포용)
# ──────────────────────────────────────────────
def generate_triton_config(engine_dir: str = ENGINE_DIR,
                            triton_model_dir: str = "./triton_models"):
    """
    Triton Inference Server용 모델 저장소 구조 생성.
    NeMo Evaluator는 Triton 백엔드도 지원합니다.
    """
    print(f"\n[Triton] Generating model repository at: {triton_model_dir}")

    # Triton 모델 디렉토리 구조
    # triton_models/
    #   tensorrt_llm/
    #     1/                  (버전 디렉토리)
    #       *.engine
    #       config.json
    #     config.pbtxt         (Triton 설정)
    #   preprocessing/        (토크나이징 파이썬 모델)
    #   postprocessing/       (디코딩 파이썬 모델)

    triton_config = {
        "name": "tensorrt_llm",
        "backend": "tensorrtllm",
        "max_batch_size": MAX_BATCH_SIZE,
        "model_transaction_policy": {"decoupled": True},
        "input": [
            {"name": "input_ids", "data_type": "TYPE_INT32", "dims": [-1]},
            {"name": "input_lengths", "data_type": "TYPE_INT32", "dims": [1]},
            {"name": "request_output_len", "data_type": "TYPE_INT32", "dims": [1]},
        ],
        "output": [
            {"name": "output_ids", "data_type": "TYPE_INT32", "dims": [-1, -1]},
            {"name": "sequence_length", "data_type": "TYPE_INT32", "dims": [1]},
        ],
        "parameters": {
            "engine_dir": {"string_value": engine_dir},
            "max_tokens_in_paged_kv_cache": {"string_value": "2560"},
            "kv_cache_free_gpu_mem_fraction": {"string_value": "0.9"},
            "batching_strategy": {"string_value": "inflight_fused_batching"},
        }
    }

    Path(f"{triton_model_dir}/tensorrt_llm/1").mkdir(parents=True, exist_ok=True)
    with open(f"{triton_model_dir}/triton_config.json", "w") as f:
        json.dump(triton_config, f, indent=2)

    print(f"[Triton] Config saved. Launch with:")
    print(f"  tritonserver --model-repository={triton_model_dir} --grpc-port=8001 --http-port=8000")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TensorRT-LLM Engine Build & Serve")
    parser.add_argument("--model_name", default="temp")
    parser.add_argument("--mode", choices=["build", "test", "serve", "triton"],
                        default="build", help="실행 모드")
    parser.add_argument("--quantized_dir", default=QUANTIZED_DIR)
    parser.add_argument("--engine_dir",    default=ENGINE_DIR)
    parser.add_argument("--port",          default=SERVE_PORT, type=int)
    parser.add_argument(
        "--gemm_plugin",
        default=None,
        help="trtllm-build --gemm_plugin (기본: quantized_model/config.json 의 dtype)",
    )
    parser.add_argument("--max_input_len", default=MAX_INPUT_LEN, type=int)
    parser.add_argument("--max_output_len", default=MAX_OUTPUT_LEN, type=int)
    parser.add_argument(
        "--tokenizer",
        default=None,
        help="HF 토크나이저 경로 또는 모델 ID. test/serve(tensorrt): 엔진만으로는 토크나이저가 없을 수 있어 필수에 가깝게 필요",
    )
    args = parser.parse_args()

    if args.mode == "build":
        start_time = time.time()
        build_trtllm_engine(
            args.quantized_dir, args.engine_dir, gemm_plugin=args.gemm_plugin, max_input_len=args.max_input_len, max_output_len=args.max_output_len
        )
        end_time = time.time()
        print(f"Engine build time: {(end_time - start_time)/60:.2f} minutes")
    elif args.mode == "test":
        start_time = time.time()
        test_inference_python_api(args.engine_dir, tokenizer_path=args.tokenizer)
        end_time = time.time()
        print(f"Inference time: {(end_time - start_time)/60:.2f} minutes")
    elif args.mode == "serve":
        start_trtllm_serve(
            args.engine_dir,
            model_name=args.model_name,
            port=args.port,
            tokenizer=args.tokenizer,
        )
    elif args.mode == "triton":
        generate_triton_config(args.engine_dir)
