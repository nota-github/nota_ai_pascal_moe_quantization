"""
Step 3: NVIDIA NeMo Evaluator 벤치마크
======================================
- Step 2의 TensorRT-LLM(OpenAI 호환) 서버를 대상으로 평가 실행
- 모드:
  - nel_core: nemo_evaluator `evaluate()` + EvaluationConfig.type (Core 퀵스타트와 동일)
  - 기본 `--tasks fewshot_cot`: ADLR 태스크로 GSM8K·GPQA Diamond·MMLU-Pro few-shot CoT
  - `--tasks fewshot_cot_basic`: 하네스 **표준(비-ADLR)** — `gsm8k_cot`, `gpqa_diamond_cot_n_shot`(+5-shot), `mmlu_pro` 그룹
- 참고: https://github.com/NVIDIA-NeMo/Evaluator
- Core Quickstart: https://docs.nvidia.com/nemo/evaluator/latest/get-started/quickstart/core.html
- 매뉴얼: quant_pipe/NVIDIA_NEMO_EVALUATOR_MANUAL.md
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any
import time
import requests
import yaml


# ──────────────────────────────────────────────
# 설정
# ──────────────────────────────────────────────
TRTLLM_SERVER_URL = "http://localhost:8000/v1"  # Step 2 서버 베이스 (/v1 까지)
MODEL_NAME = "llama-3.1-8b-gptq"
RESULTS_DIR = "./eval_results"

# nel_core: evaluate() + simple_evals — Core 퀵스타트처럼 EvaluationConfig.type 만 전달
# (별도 lm-evaluation-harness 패키지 없이 동작). gsm8k → mgsm 은 simple_evals에 gsm8k 태스크가 없어서 대응.
NEL_CORE_TASK_ALIASES: dict[str, str] = {
    "mmlu": "mmlu",
    "gsm8k": "mgsm",
    "humaneval": "humaneval",
    "humanevalplus": "humanevalplus",
}

# nel_core + `core_evals.lm_evaluation_harness`: `EvaluationConfig.type` 은 framework.yml 의 `config.type` 과 일치해야 함.
# 짧은 별칭·lm_eval 스타일 이름 → NeMo 등록 type (few-shot CoT 스위트는 ADLR 항목과 맞춤).
NEL_CORE_LM_EVAL_STYLE_TO_CONFIG_TYPE: dict[str, str] = {
    "gsm8k_cot": "adlr_gsm8k_cot_8_shot",
    "adlr_gsm8k_fewshot_cot": "adlr_gsm8k_cot_8_shot",
    "gpqa_diamond_cot_zeroshot": "adlr_gpqa_diamond_cot_5_shot",
    "gpqa_diamond_cot_n_shot": "adlr_gpqa_diamond_cot_5_shot",
    "mmlu_pro": "adlr_mmlu_pro_5_shot_base",
}

# NeMo Core `EvaluationConfig.type` 과 lm-eval 하네스 `get_task_dict()` 등록명이 다른 경우만 보정.
NEL_CORE_TYPE_TO_LM_EVAL_TASK: dict[str, str] = {
    "adlr_gsm8k_cot_8_shot": "adlr_gsm8k_fewshot_cot",
}

# `--tasks fewshot_cot` 일 때 모드별로 펼칠 목록 (이름 몰라도 동일 스위트).
TASK_PRESET_FEWSHOT_COT = "fewshot_cot"
NEL_CORE_FEWSHOT_COT_TYPES = (
    "adlr_gsm8k_cot_8_shot",
    "adlr_gpqa_diamond_cot_5_shot",
    "adlr_mmlu_pro_5_shot_base",
)
LM_EVAL_FEWSHOT_COT_TASKS = (
    "adlr_gsm8k_fewshot_cot",
    "adlr_gpqa_diamond_cot_5_shot",
    "adlr_mmlu_pro_5_shot_base",
)

# `--tasks fewshot_cot_basic` — lm-eval 기본 태스크( ADLR 아님 ). GPQA는 YAML에 num_fewshot 없어 CLI로 5 지정.
TASK_PRESET_FEWSHOT_COT_BASIC = "fewshot_cot_basic"
NEL_CORE_FEWSHOT_COT_BASIC_TYPES = (
    "lm-evaluation-harness.gsm8k_cot",
    "lm-evaluation-harness.gpqa_diamond_cot_n_shot",
    #"lm-evaluation-harness.mmlu_pro",
)
LM_EVAL_FEWSHOT_COT_BASIC_TASKS = (
    "gsm8k_cot",
    "gpqa_diamond_cot_n_shot",
    #"mmlu_pro",
)

# MMLU-Pro 계열: `--mmlu_pro_limit` 지정 시 0..N-1 구간에서 균등 간격 인덱스 `--samples` (기본 N=12000).
MMLU_PRO_INDEX_POOL_SIZE = 12000
MMLU_PRO_LM_EVAL_TASKS: frozenset[str] = frozenset(
    {"adlr_mmlu_pro_5_shot_base", "mmlu_pro"},
)

# lm_eval CLI: num_fewshot_cli=None → `--num_fewshot` 생략(YAML 기본 샷).
LM_EVAL_TASK_SPECS: dict[str, dict[str, Any]] = {
    "adlr_gsm8k_fewshot_cot": {"num_fewshot_cli": None, "limit": None},
    "adlr_gpqa_diamond_cot_5_shot": {"num_fewshot_cli": None, "limit": None},
    "adlr_mmlu_pro_5_shot_base": {"num_fewshot_cli": None, "limit": None},
    "gsm8k_cot": {"num_fewshot_cli": None, "limit": None},
    "gpqa_diamond_cot_n_shot": {"num_fewshot_cli": 5, "limit": None},
    "mmlu_pro": {"num_fewshot_cli": None, "limit": None},
}

# 참고용(이전 nemo_skills/lm-eval 한도와 유사한 스모크 테스트 값)
BENCHMARK_HINTS = {
    "mmlu": {"num_shots": 5, "limit": 200},
    "gsm8k": {"num_shots": 8, "limit": 100},
    "gsm8k_cot": {"num_shots": 8, "limit": 100},
    "hellaswag": {"num_shots": 10, "limit": 200},
    "truthfulqa": {"num_shots": 0, "limit": 100},
    "humaneval": {"num_shots": 0, "limit": 50},
    "winogrande": {"num_shots": 5, "limit": 200},
}


# ──────────────────────────────────────────────
# URL / 환경
# ──────────────────────────────────────────────
def to_chat_completions_url(server_url: str) -> str:
    """
    NeMo Evaluator target.api_endpoint.url 은 OpenAI 스펙상
    .../v1/chat/completions 전체 경로를 기대합니다.
    """
    u = server_url.rstrip("/")
    if u.endswith("/chat/completions"):
        return u
    if u.endswith("/v1"):
        return f"{u}/chat/completions"
    return f"{u}/v1/chat/completions"


def to_completions_url(server_url: str) -> str:
    """OpenAI 호환 **텍스트 completions** 엔드포인트 (`.../v1/completions`). ADLR 등 logprob/legacy 평가에 필요할 수 있음."""
    u = server_url.rstrip("/")
    if u.endswith("/completions"):
        return u
    if u.endswith("/v1"):
        return f"{u}/completions"
    return f"{u}/v1/completions"


def ensure_openai_api_key_env() -> None:
    """로컬 TRT-LLM은 키 검증을 안 해도 되는 경우가 많음.

    - nel_core: ApiEndpoint 용 더미.
    - lm_eval `openai-chat-completions`: subprocess 가 Authorization 헤더를 만들 때 필수(ValueError 방지).
    """
    if not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = "local-dummy"


def parse_task_arg(raw: str, mode: str) -> list[str]:
    """`fewshot_cot` / `fewshot_cot_basic` 프리셋 또는 콤마 구분 목록."""
    key = raw.strip().lower()
    if key == TASK_PRESET_FEWSHOT_COT:
        return list(NEL_CORE_FEWSHOT_COT_TYPES if mode == "nel_core" else LM_EVAL_FEWSHOT_COT_TASKS)
    if key == TASK_PRESET_FEWSHOT_COT_BASIC:
        return list(NEL_CORE_FEWSHOT_COT_BASIC_TYPES if mode == "nel_core" else LM_EVAL_FEWSHOT_COT_BASIC_TASKS)
    return [t.strip() for t in raw.split(",") if t.strip()]


def resolve_lm_eval_harness_task_name(raw: str) -> str:
    """짧은 별칭·닉네임 → `lm_eval.tasks.get_task_dict()`에 넣을 등록 태스크명.

    `NEL_CORE_LM_EVAL_STYLE_TO_CONFIG_TYPE`와 동일한 키로 NeMo용 type으로 올린 뒤,
    하네스에만 존재하는 이름(예: GSM8K)으로 `NEL_CORE_TYPE_TO_LM_EVAL_TASK` 보정.

    - `resolve_nel_core_tasks`와 달리 `gsm8k`→`mgsm` 같은 simple_evals 별칭은 적용하지 않음.
    - `framework.task` 형식(점 하나)은 그대로 반환.
    """
    n = raw.strip()
    if not n:
        raise ValueError("빈 태스크 문자열입니다.")
    if n.count(".") > 1:
        raise ValueError(
            "태스크는 'framework.task' 형식일 때 점(.)이 한 개만 허용됩니다: "
            f"{raw!r}"
        )
    if n.count(".") == 1:
        return n
    key = n.lower()
    out = NEL_CORE_LM_EVAL_STYLE_TO_CONFIG_TYPE.get(key, n)
    return NEL_CORE_TYPE_TO_LM_EVAL_TASK.get(out, out)


def resolve_nel_core_tasks(names: list[str]) -> list[str]:
    """nel_core 전용: Core는 `type` 한 덩어리로 simple_evals / lm-eval 하네스 등을 고릅니다.

    - `mmlu_pro` : 설치된 하네스 중 해당 type이 하나일 때만 유효. 둘 이상이면 NeMo가
      `framework.task` 형식을 요구합니다 (예: simple_evals.mmlu_pro).
    - `simple_evals.mmlu_pro` / `lm-evaluation-harness.mmlu` : 정확히 점 하나(두 구간)만 허용.
    - lm_eval CLI와 동일한 별칭(`gsm8k_cot` 등)은 `NEL_CORE_LM_EVAL_STYLE_TO_CONFIG_TYPE` 으로
      `lm_evaluation_harness` 의 등록 `type` 으로 치환합니다.
    """
    resolved: list[str] = []
    for raw in names:
        n = raw.strip()
        if not n:
            continue
        if n.count(".") > 1:
            print(f"[nel_core] type은 'framework.task' 한 개의 점만 허용됩니다. '{n}' 건너뜀.")
            continue
        if n.count(".") == 1:
            resolved.append(n)
            continue
        key = n.lower()
        if key == "hellaswag":
            print(
                "[nel_core] hellaswag 는 simple_evals에 없습니다. "
                "건너뜀. (--mode lm_eval 또는 mmlu_pro 등 다른 type 사용)"
            )
            continue
        if key in NEL_CORE_TASK_ALIASES:
            out = NEL_CORE_TASK_ALIASES[key]
        else:
            out = n
        mapped = NEL_CORE_LM_EVAL_STYLE_TO_CONFIG_TYPE.get(out.lower())
        if mapped and mapped != out:
            print(f"[nel_core] '{out}' → EvaluationConfig.type '{mapped}' (lm_eval 별칭 → NeMo 하네스 등록명)")
            out = mapped
        resolved.append(out)
    return resolved


# ──────────────────────────────────────────────
# 헬스체크
# ──────────────────────────────────────────────
def check_server_health(server_url: str = TRTLLM_SERVER_URL) -> bool:
    """TRT-LLM OpenAI 서버 준비 여부 확인."""
    print(f"[Health Check] Connecting to {server_url}...")
    try:
        resp = requests.get(f"{server_url.rstrip('/')}/models", timeout=10)
        if resp.status_code == 200:
            models = resp.json().get("data", [])
            print(f"  Available models: {[m.get('id') for m in models]}")
            return True
    except requests.exceptions.ConnectionError:
        print(f"  [Error] Server not reachable at {server_url}")
        print("  → Step 2를 먼저 실행하세요: python step2_trtllm_build_serve.py --mode serve")
    return False


def collect_eval_artifacts(output_dir: str) -> dict[str, Any]:
    """결과 디렉터리에서 results.yml 등 탐색."""
    root = Path(output_dir)
    info: dict[str, Any] = {"output_dir": str(root.resolve())}
    if not root.exists():
        return info

    candidates = list(root.rglob("results.yml")) + list(root.rglob("results.yaml"))
    for p in candidates:
        info["results_file"] = str(p)
        try:
            with open(p, encoding="utf-8") as f:
                info["results_yaml"] = yaml.safe_load(f)
        except Exception as e:
            info["results_yaml_error"] = str(e)
        break

    return info


def _safe_task_subdir(task_id: str) -> str:
    """output 하위 폴더 이름으로 쓸 수 있게 태스크 id 정규화."""
    return task_id.replace(os.sep, "_").replace(":", "_").replace(".", "_")


# ──────────────────────────────────────────────
# NeMo Evaluator Core (Python API)
# ──────────────────────────────────────────────
def run_nemo_evaluator_core(
    tasks: list[str],
    server_url: str,
    model_name: str,
    output_dir: str,
    limit_samples: int | None,
    parallelism: int,
    request_timeout: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    api_key_name: str,
    max_retries: int | None,
    endpoint_type: str = "chat",
) -> dict[str, Any]:
    """
    nemo_evaluator.core.evaluate.evaluate() 로 태스크별 순차 실행.
    EvaluationConfig.type (예: mmlu_pro, simple_evals.mmlu_pro). 짧은 이름은 resolve_nel_core_tasks 참고.
    """
    ensure_openai_api_key_env()
    nel_tasks = resolve_nel_core_tasks(tasks)
    if not nel_tasks:
        print("[Error] No valid tasks to run.")
        return {}

    try:
        from nemo_evaluator.api.api_dataclasses import (
            ApiEndpoint,
            ConfigParams,
            EndpointType,
            EvaluationConfig,
            EvaluationTarget,
        )
        from nemo_evaluator.core.evaluate import evaluate
    except ImportError as e:
        print(f"[Error] nemo-evaluator import 실패 (pip install nemo-evaluator): {e}")
        return {"_error": True, "import_error": str(e)}

    base_out = Path(output_dir)
    base_out.mkdir(parents=True, exist_ok=True)

    ep = EndpointType.CHAT if endpoint_type.strip().lower() == "chat" else EndpointType.COMPLETIONS
    api_url = to_chat_completions_url(server_url) if ep == EndpointType.CHAT else to_completions_url(server_url)

    target_config = EvaluationTarget(
        api_endpoint=ApiEndpoint(
            url=api_url,
            model_id=model_name,
            api_key_name=api_key_name,
            type=ep,
        )
    )

    per_task: dict[str, Any] = {}
    by_task_artifacts: list[dict[str, Any]] = []

    for task_type in nel_tasks:
        sub = base_out / _safe_task_subdir(task_type)
        sub.mkdir(parents=True, exist_ok=True)

        params_kwargs: dict[str, Any] = {
            "parallelism": parallelism,
            "temperature": temperature,
            "top_p": top_p,
            "max_new_tokens": max_new_tokens,
            "request_timeout": request_timeout,
        }
        if limit_samples is not None:
            params_kwargs["limit_samples"] = limit_samples
        if max_retries is not None:
            params_kwargs["max_retries"] = max_retries

        eval_config = EvaluationConfig(
            type=task_type,
            output_dir=str(sub),
            params=ConfigParams(**params_kwargs),
        )

        print(f"{'='*60}")
        print(f"[NeMo Core] evaluate()  type={task_type}")
        print(f"output_dir={sub}")
        print(f"{'='*60}")

        result = evaluate(eval_cfg=eval_config, target_cfg=target_config)

        dumped: Any
        try:
            dumped = result.model_dump(mode="json")
        except Exception:
            dumped = {"repr": repr(result)}

        per_task[task_type] = {"evaluation_result": dumped, "output_dir": str(sub)}

        art = collect_eval_artifacts(str(sub))
        by_task_artifacts.append({"task": task_type, **art})

    return {
        "nemo_evaluator_core": True,
        "tasks": nel_tasks,
        "per_task": per_task,
        "artifacts": {"by_task": by_task_artifacts, "output_dir": str(base_out.resolve())},
    }


# ──────────────────────────────────────────────
# lm-evaluation-harness (직접, 대안)
# ──────────────────────────────────────────────
def _lm_eval_task_spec(task: str) -> dict[str, Any]:
    return LM_EVAL_TASK_SPECS.get(task, {"num_fewshot_cli": None, "limit": None})


def mmlu_pro_evenly_spaced_indices(n_total: int, k: int) -> list[int]:
    """[0, n_total-1] 구간을 k개로 균등 분할한 정수 인덱스(끝점 포함). lm_eval `--samples` 용."""
    if k < 1:
        raise ValueError("k must be >= 1")
    if n_total < 1:
        raise ValueError("n_total must be >= 1")
    if k > n_total:
        raise ValueError(f"k ({k}) cannot exceed n_total ({n_total})")
    if k == 1:
        return [0]
    return [int(round(i * (n_total - 1) / (k - 1))) for i in range(k)]


def _resolve_lm_eval_results_json(output_base: str) -> Path | None:
    """lm-eval `EvaluationTracker`는 `--output_path`를 파일이 아니라 저장용 베이스 디렉터리로 쓰고
    `<output_base>/<model_sanitized>/results_<timestamp>.json` 에 집계 결과를 쓴다.
    `.json` 접미로 끝나는 경로를 넘기면 디렉터리가 `*.json` 이름으로 생성될 수 있음(IsADirectoryError).
    """
    root = Path(output_base)
    if not root.exists():
        return None
    if root.is_file():
        return root
    matches = sorted(
        root.rglob("results_*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return matches[0] if matches else None


def _pick_lm_eval_score(task_metrics: dict[str, Any]) -> tuple[str, Any]:
    """lm_eval `results[task]` 항목에서 요약용 점수 키 하나 고름."""
    skip = {"alias", "pretty_name", "original", "effective"}
    for k, v in task_metrics.items():
        if k in skip or not isinstance(v, (int, float)):
            continue
        if "exact_match" in k or k.startswith("acc") or "pass@" in k:
            return k, v
    for k, v in task_metrics.items():
        if k not in skip and isinstance(v, (int, float)):
            return k, v
    return "n/a", "N/A"


def run_lm_eval_harness(
    tasks: list[str],
    server_url: str,
    model_name: str,
    output_dir: str,
    max_tokens: int = 2048,
    *,
    smoke_limits: bool = False,
    gen_kwargs: str | None = None,
    mmlu_pro_limit: int | None = None,
    mmlu_pro_pool_size: int = MMLU_PRO_INDEX_POOL_SIZE,
) -> dict[str, Any]:
    """EleutherAI lm-eval — OpenAI 호환 서버. 태스크별 `num_fewshot`이 다르면 subprocess 분리 후 results 병합.

    `openai-chat-completions`는 프롬프트를 chat `messages`로 보내려면 `--apply_chat_template`가 필요함(lm-eval assert).

    lm-eval `TemplateAPI`는 `base_url`에 POST 하므로 OpenAI 스펙상 **…/v1/chat/completions 전체 URL** 이어야 함
    (`http://…/v1` 만 넘기면 404).

    최대 생성 길이는 `--max_tokens` 인자만 사용(코드에서 태스크별로 올리지 않음). 기본값은 argparse와 동일(2048).

    결과 JSON은 lm-eval이 `output_base` 아래에 `results_*.json` 으로 저장하므로, 병합 시 `_resolve_lm_eval_results_json` 으로 탐색한다.

    전체 데이터셋 평가: `smoke_limits=False`(기본) — `BENCHMARK_HINTS`의 `--limit`을 붙이지 않음.
    CoT 길이: 태스크 YAML `max_gen_toks`와 맞추려면 `--lm_eval_gen_kwargs` 예: `max_gen_toks=2048` (HF Qwen 토론 참고).

    `mmlu_pro_limit`가 지정되면 `adlr_mmlu_pro_5_shot_base` / `mmlu_pro` 태스크에 대해
    `mmlu_pro_pool_size`(기본 12000)개를 균등 간격으로 나눈 인덱스로 `--samples`를 넣고 `--limit`은 쓰지 않음.
    """
    ensure_openai_api_key_env()
    chat_api_url = to_chat_completions_url(server_url)
    short = [t.strip() for t in tasks if t.strip()]
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    merged: dict[str, Any] = {
        "lm_eval_merged": True,
        "results": {},
        "groups": {},
        "configs": {},
        "versions": {},
        "n-shot": {},
        "higher_is_better": {},
        "n-samples": {},
        "per_task_output": [],
    }

    for task in short:
        task_start_time = time.time()
        spec = _lm_eval_task_spec(task)
        mtoks = int(max_tokens)
        # 디렉터리로 전달 (접미사 .json 는 lm-eval이 하위 디렉터리로 만들어 파일 open 시 IsADirectoryError 남)
        out_base = f"{output_dir}/lm_eval_{task.replace(',', '_')}"
        parts = [
            "lm_eval --model openai-chat-completions",
            f'--model_args "base_url={chat_api_url},model={model_name},max_tokens={mtoks}"',
            f"--tasks {task}",
        ]
        nfs = spec.get("num_fewshot_cli")
        if nfs is not None:
            parts.append(f"--num_fewshot {int(nfs)}")
        use_mmlu_even_samples = (
            mmlu_pro_limit is not None
            and task in MMLU_PRO_LM_EVAL_TASKS
        )
        if use_mmlu_even_samples:
            idx = mmlu_pro_evenly_spaced_indices(int(mmlu_pro_pool_size), int(mmlu_pro_limit))
            samples_path = Path(output_dir) / f"lm_eval_{task.replace(',', '_')}_samples.json"
            samples_path.write_text(
                json.dumps({task: idx}, separators=(",", ":")),
                encoding="utf-8",
            )
            parts.append(f"--samples {shlex.quote(str(samples_path.resolve()))}")
        else:
            lim = spec.get("limit")
            if lim is not None:
                parts.append(f"--limit {int(lim)}")
            elif smoke_limits:
                # 스모크: BENCHMARK_HINTS 한도. 기본(smoke_limits=False)은 전체 샘플.
                hint_lim = BENCHMARK_HINTS.get(task.lower(), {}).get("limit")
                if hint_lim is not None:
                    parts.append(f"--limit {int(hint_lim)}")
        if gen_kwargs and gen_kwargs.strip():
            parts.extend(["--gen_kwargs", shlex.quote(gen_kwargs.strip())])
        parts.extend(
            [
                "--apply_chat_template",
                "--output_path",
                out_base,
                "--log_samples",
                "--batch_size",
                "1",
            ]
        )
        cmd = " ".join(parts)

        print(
            f"\n[lm-eval] Running: {task} (max_tokens={mtoks}, num_fewshot_cli={nfs!r}, "
            f"smoke_limits={smoke_limits}, gen_kwargs={gen_kwargs!r}, "
            f"mmlu_pro_limit={mmlu_pro_limit!r}, mmlu_pro_pool_size={mmlu_pro_pool_size})"
        )
        subprocess.run(cmd, shell=True, check=True)
        results_json = _resolve_lm_eval_results_json(out_base)
        merged["per_task_output"].append(str(results_json) if results_json else out_base)
        if results_json is None or not results_json.is_file():
            continue
        with open(results_json, encoding="utf-8") as f:
            blob = json.load(f)
        for key in ("results", "groups", "configs", "versions", "n-shot", "higher_is_better", "n-samples"):
            chunk = blob.get(key)
            if isinstance(chunk, dict):
                merged[key].update(chunk)

        task_end_time = time.time()
        task_duration = task_end_time - task_start_time
        print(f"[lm-eval] Task {task} completed in {(task_duration/60):.2f} minutes")

    return merged


# ──────────────────────────────────────────────
# 리포트
# ──────────────────────────────────────────────
def generate_report(
    all_results: dict[str, Any],
    model_name: str = MODEL_NAME,
    output_dir: str = RESULTS_DIR,
) -> dict[str, Any]:
    """요약 리포트 JSON 저장."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if all_results.get("nemo_evaluator_core"):
        evaluator_label = "nemo-evaluator (Core API)"
    else:
        evaluator_label = "lm-eval / other"

    report: dict[str, Any] = {
        "model": model_name,
        "timestamp": timestamp,
        "quantization": "GPTQ_W4A16",
        "backend": "TensorRT-LLM",
        "evaluator": evaluator_label,
        "results": all_results,
        "summary": {},
    }

    summary: dict[str, Any] = {}
    if all_results.get("nemo_evaluator_core"):
        art = all_results.get("artifacts") or {}
        summary["nemo_evaluator_core"] = {
            "output_dir": art.get("output_dir"),
            "by_task": [
                {"task": x.get("task"), "results_file": x.get("results_file")}
                for x in (art.get("by_task") or [])
            ],
            "note": "태스크별 output_dir 아래 results.yml 참고. per_task.evaluation_result에 요약.",
        }
    else:
        task_block: dict[str, Any] = {}
        if isinstance(all_results.get("results"), dict):
            task_block = all_results["results"]
        else:
            task_block = {k: v for k, v in all_results.items() if isinstance(v, dict)}

        for task, results in task_block.items():
            if not isinstance(results, dict):
                continue
            metric_key, score = _pick_lm_eval_score(results)
            summary[task] = {
                "metric": metric_key,
                "score": score,
                "score_pct": f"{float(score) * 100:.2f}%"
                if isinstance(score, (int, float))
                else score,
            }
        if all_results.get("groups"):
            summary["_groups"] = list(all_results["groups"].keys())

    report["summary"] = summary

    report_path = str(Path(output_dir) / f"benchmark_report_{timestamp}.json")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"  Benchmark Report — {model_name}")
    print(f"  Backend: TensorRT-LLM | Evaluator: {evaluator_label}")
    print(f"{'='*60}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"{'='*60}")
    print(f"\n  Full report saved: {report_path}")

    return report


# ──────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="NeMo Evaluator — TensorRT-LLM OpenAI 서버 벤치마크 (Core API / lm-eval)",
    )
    parser.add_argument("--server_url", default=TRTLLM_SERVER_URL)
    parser.add_argument("--model_name", default=MODEL_NAME)
    parser.add_argument("--output_dir", default=RESULTS_DIR)
    parser.add_argument(
        "--tasks",
        default=TASK_PRESET_FEWSHOT_COT,
        help=(
            f"콤마 구분 평가 목록, 또는 프리셋: `{TASK_PRESET_FEWSHOT_COT}` (ADLR few-shot CoT), "
            f"`{TASK_PRESET_FEWSHOT_COT_BASIC}` (표준 하네스: gsm8k_cot + gpqa_diamond_cot_n_shot@5 + mmlu_pro). "
            "nel_core: EvaluationConfig.type 또는 lm-evaluation-harness.<task>. lm_eval: 등록 태스크·그룹명."
        ),
    )
    parser.add_argument(
        "--mode",
        default="nel_core",
        choices=["nel_core", "lm_eval"],
        help="nel_core: nemo_evaluator evaluate() API, lm_eval: lm-eval CLI",
    )
    parser.add_argument("--skip_health_check", action="store_true")
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=2048,
        help="최대 생성 길이(외부 인자만 사용). lm_eval: model_args max_tokens. nel_core: ConfigParams.max_new_tokens.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="nel_core: ConfigParams.temperature (벤치마크 기본은 보통 0).",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=1.0,
        help="nel_core: ConfigParams.top_p",
    )
    parser.add_argument(
        "--limit_samples",
        type=int,
        default=None,
        help="nel_core: ConfigParams.limit_samples (스모크). None이면 생략(전체 샘플).",
    )
    parser.add_argument(
        "--parallelism",
        type=int,
        default=1,
        help="nel_core: ConfigParams.parallelism (서버 부하에 맞게 조정).",
    )
    parser.add_argument(
        "--request_timeout",
        type=int,
        default=3600,
        help="nel_core: ConfigParams.request_timeout(초).",
    )
    parser.add_argument(
        "--api_key_name",
        default="OPENAI_API_KEY",
        help="nel_core: ApiEndpoint.api_key_name (환경 변수 이름). 로컬 TRT-LLM은 더미 값으로 설정.",
    )
    parser.add_argument(
        "--max_retries",
        type=int,
        default=None,
        help="nel_core: ConfigParams.max_retries (기본 None은 하네스 기본값 따름).",
    )
    parser.add_argument(
        "--endpoint_type",
        choices=["chat", "completions"],
        default="chat",
        help=(
            "nel_core 전용: ApiEndpoint.type 및 URL (/v1/chat/completions vs /v1/completions). "
            "`fewshot_cot` 의 ADLR 벤치는 completions 만 지원 → `completions` 필요."
        ),
    )
    parser.add_argument(
        "--lm_eval_smoke_limits",
        action="store_true",
        help=(
            "lm_eval 전용: BENCHMARK_HINTS 로 --limit 스모크(예: gsm8k_cot 100문항). "
            "기본은 끔 → 태스크 전체 샘플. 공정 비교·리포트 재현 시 스모크 끄기."
        ),
    )
    parser.add_argument(
        "--lm_eval_gen_kwargs",
        type=str,
        default=None,
        help=(
            'lm_eval 전용: `--gen_kwargs` 문자열(태스크 generation_kwargs 보강). '
            '예: "max_gen_toks=2048,temperature=0.0" — CoT가 잘리면 max_gen_toks 조정.'
        ),
    )
    parser.add_argument(
        "--mmlu_pro_limit",
        type=int,
        default=None,
        metavar="K",
        help=(
            "lm_eval 전용: MMLU-Pro 계열 태스크(adlr_mmlu_pro_5_shot_base, mmlu_pro)에 대해 "
            "`--limit` 대신 `--samples`로 평가할 문항 수 K. 인덱스는 0..N-1(N=--mmlu_pro_total, 기본 12000)을 "
            "균등 간격으로 나눈 K개. 미지정 시 태스크 스펙의 `--limit`(있으면) 또는 전체."
        ),
    )
    parser.add_argument(
        "--mmlu_pro_total",
        type=int,
        default=MMLU_PRO_INDEX_POOL_SIZE,
        metavar="N",
        help=(
            "`--mmlu_pro_limit` 사용 시 인덱스 풀 크기(기본 12000). 데이터셋 전체 문항 수와 맞출 것."
        ),
    )
    args = parser.parse_args()

    task_list = parse_task_arg(args.tasks, args.mode)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    _api_url = (
        to_completions_url(args.server_url)
        if args.mode == "nel_core" and args.endpoint_type == "completions"
        else to_chat_completions_url(args.server_url)
    )

    print(f"\n{'='*60}")
    print("  NeMo Evaluator — Step 3")
    print(f"  Mode   : {args.mode}")
    print(f"  Tasks  : {task_list}")
    print(f"  Server : {args.server_url}")
    if args.mode == "nel_core":
        print(f"  nel_core API: type={args.endpoint_type}  url={_api_url}")
    else:
        print(f"  lm_eval base (chat): {_api_url}")
    print(f"{'='*60}\n")

    if not args.skip_health_check:
        if not check_server_health(args.server_url):
            print("[Abort] Server not available. Use --skip_health_check to bypass.")
            return

    all_results: dict[str, Any] = {}

    if args.mode == "nel_core":
        all_results = run_nemo_evaluator_core(
            tasks=task_list,
            server_url=args.server_url,
            model_name=args.model_name,
            output_dir=args.output_dir,
            limit_samples=args.limit_samples,
            parallelism=args.parallelism,
            request_timeout=args.request_timeout,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            api_key_name=args.api_key_name,
            max_retries=args.max_retries,
            endpoint_type=args.endpoint_type,
        )
    elif args.mode == "lm_eval":
        mlim = args.mmlu_pro_limit
        if mlim is not None and mlim < 1:
            raise SystemExit("--mmlu_pro_limit must be >= 1 when set")
        mtot = args.mmlu_pro_total
        if mtot is not None and mtot < 1:
            raise SystemExit("--mmlu_pro_total must be >= 1")
        all_results = run_lm_eval_harness(
            tasks=task_list,
            server_url=args.server_url,
            model_name=args.model_name,
            output_dir=args.output_dir,
            max_tokens=args.max_tokens,
            smoke_limits=args.lm_eval_smoke_limits,
            gen_kwargs=args.lm_eval_gen_kwargs,
            mmlu_pro_limit=mlim,
            mmlu_pro_pool_size=int(mtot),
        )

    if all_results:
        generate_report(all_results, args.model_name, args.output_dir)


if __name__ == "__main__":
    main()
