"""
Step 1: NVIDIA ModelOpt로 LLM 처리 후 TensorRT-LLM checkpoint 저장
====================================================================
- GPTQ_W4A16: INT4 블록 W4A16 + ModelOpt `gptq_lite` 후 TensorRT-LLM export. Hessian은 `--hessian_state_path`(기본: dataset_dir 하위)에 있으면 로드·없으면 캘리브로 계산. `--gptq_require_cached_hessian`이면 파일 필수. `--parallel_gptq_batch`>1 이면 Phase-4만 CUDA 스트림 청크 병렬(런타임 패치).
- HESSIAN: `gptq_lite`의 Hessian만 캘리브 forward로 누적해 디스크에 저장(Phase-4·export 없음). 이후 같은 경로로 `GPTQ_W4A16`만 단독 실행 가능.
- RTN_W4A16: 동일 dtype/포맷(INT4 블록 W4A16)이나 calibration 없이 가중치 min–max만 적용
- UNQUANTIZED: 양자화 없이 Hugging Face 가중치를 TensorRT-LLM checkpoint로만 변환
- CALIB_DATASET_ONLY: 모델 로드 없이 Nemotron 캘리브 번들만 생성
  (`--dataset_dir`에 `save_to_disk` + samples_text.jsonl + calib_chunks.pt + 메타 JSON; 하위 calib_* 폴더 없음.
  디렉터리에 번들이 이미 있으면 덮어쓰지 않고 FileExistsError)
"""

from typing import Any, TypedDict


import copy
import json
import torch
import argparse
from datetime import datetime, timezone
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
from datasets import Dataset

from nemotron_post_training_calib import (
    build_nemotron_post_training_v1_dataset_config,
    prepare_nemotron_post_training_v1_calibration_dataset,
)

# nvidia-modelopt 설치: pip install nvidia-modelopt[torch]
import modelopt.torch.quantization as mtq
from modelopt.torch.export import export_tensorrt_llm_checkpoint
from modelopt.torch.export.model_utils import get_model_type

from modelopt_parallel_gptq import (  # type: ignore[import-not-found]
    install_gptq_hessian_only_patch,
    install_parallel_gptq_patch_if_needed,
    restore_parallel_gptq_patch,
)


# ──────────────────────────────────────────────
# 설정
# ──────────────────────────────────────────────
DEFAULT_MODEL = "Qwen/Qwen3-30B-A3B"
DEFAULT_SAVE_DIR = "./quantized_model"
# Nemotron-Post-Training-Dataset-v1: chat / math / code / stem, 행 메타 reasoning == "on" (`nemotron_post_training_calib.py`)
CALIB_DATASET = "nvidia/Nemotron-Post-Training-Dataset-v1"
CALIB_SIZE = 512  # 총 캘리브레이션 샘플 수 (도메인당 128, 4도메인)
CALIB_MIN_SEQ_LEN = 1024
CALIB_MAX_SEQ_LEN = 2048
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# `dataset_dir` 루트에 저장 (하위 calib_* 폴더 없음)
CALIB_DATASET_METADATA_FILENAME = "calibration_dataset_metadata.json"


class CalibCacheMeta(TypedDict):
    """캘리브레이션 캐시 파일 내부 메타데이터 (로드 시 현재 인자와 일치 검증)."""

    dataset: str
    model_name: str
    num_samples: int
    min_seq_len: int
    max_seq_len: int
    has_text: bool
    has_domain: bool
    storage_format: str


def _model_name_for_dataset_storage(model_name: str) -> str:
    """캘리브 번들 경로·메타에 쓸 짧은 모델 식별자. 로컬 절대/상대 경로·HF `org/name` 모두 마지막 `/`·`\\` 이후만 사용."""
    s = (model_name or "").strip()
    if not s:
        return s
    base = Path(s.rstrip("/\\")).name
    return base if base else s


def calibration_chunks_pt_path(bundle_dir: Path) -> Path:
    """번들 디렉터리 안의 GPTQ용 tensor 청크 .pt 경로."""
    return bundle_dir / "calib_chunks.pt"


def default_gptq_hessian_state_pt(dataset_dir: str, model_name: str) -> str:
    """
    gptq_lite Hessian 캐시 파일 기본 경로.

    `{dataset_dir}/{모델베이스이름}-hessian/hessian_state.pt`
    (모델 ID는 `_model_name_for_dataset_storage`와 동일 규칙)
    """
    base = _model_name_for_dataset_storage(model_name)
    sub = Path(dataset_dir) / f"{base}-hessian"
    return str(sub / "hessian_state.pt")


def _hf_disk_bundle_looks_valid(bundle_dir: Path) -> bool:
    """data-*.arrow, dataset_info.json, state.json 이 있으면 HF 디스크 캐시로 간주."""
    if not bundle_dir.is_dir():
        return False
    if not (bundle_dir / "dataset_info.json").is_file():
        return False
    if not (bundle_dir / "state.json").is_file():
        return False
    return any(bundle_dir.glob("data-*.arrow"))


def _calibration_bundle_present(bundle_dir: Path) -> bool:
    """캘리브 번들이 이미 있으면 True (HF save_to_disk 또는 calib_chunks.pt)."""
    return calibration_chunks_pt_path(bundle_dir).is_file() or _hf_disk_bundle_looks_valid(
        bundle_dir
    )


def _build_calibration_chunks(
    tokenizer: AutoTokenizer,
    num_samples: int,
    min_seq_len: int,
    max_seq_len: int,
    *,
    bundle_dir: Path | None = None,
) -> list[dict[str, Any]]:
    """Nemotron Post-Training v1 (chat·math·code·stem reasoning-on) 스트리밍 수집 후 배치 리스트 생성."""
    print(
        f"[Calibration] {CALIB_DATASET}: total_samples={num_samples}, "
        f"min_tokens={min_seq_len}, max_tokens={max_seq_len}"
    )
    dataset_config = build_nemotron_post_training_v1_dataset_config()
    text_jsonl_path = None
    if bundle_dir is not None:
        bundle_dir.mkdir(parents=True, exist_ok=True)
        text_jsonl_path = bundle_dir / "samples_text.jsonl"

    ds = prepare_nemotron_post_training_v1_calibration_dataset(
        tokenizer,
        None,
        min_seq_len,
        max_seq_len,
        num_samples,
        dataset_config=dataset_config,
        always_truncate_over_max=False,
        log_tag="[gptq_calib]",
        text_jsonl_path=text_jsonl_path,
        log_per_domain_token_stats=False,
    )
    if bundle_dir is not None:
        ds.save_to_disk(str(bundle_dir))
        print(f"[Calibration] Saved HF Dataset to disk: {bundle_dir.resolve()}")

    chunks = _dataset_to_calibration_chunks(ds)
    print(f"[Calibration] {len(chunks)} calibration batches prepared.")
    return chunks


def _dataset_to_calibration_chunks(ds: Dataset) -> list[dict[str, Any]]:
    chunks = []
    for row in tqdm(ds, desc="Calibration batches", total=len(ds)):
        input_ids = torch.tensor([row["input_ids"]], dtype=torch.long)
        attn = torch.tensor([row["attention_mask"]], dtype=torch.long)
        chunk: dict[str, Any] = {"input_ids": input_ids, "attention_mask": attn}
        if "text" in row and row["text"] is not None:
            chunk["text"] = row["text"]
        if "domain" in row and row["domain"] is not None:
            chunk["domain"] = row["domain"]
        chunks.append(chunk)
    return chunks


def _write_calibration_dataset_metadata(
    dataset_dir: Path,
    *,
    tokenizer_model: str,
    num_samples: int,
    min_seq_len: int,
    max_seq_len: int,
) -> None:
    """HF 디스크 캘리브 번들 옆에 두는 사람이 읽기 쉬운 메타데이터 JSON."""
    payload: dict[str, Any] = {
        "source_dataset": CALIB_DATASET,
        "dataset_dir": str(dataset_dir.resolve()),
        "tokenizer_model": tokenizer_model,
        "tokenizer_model_basename": _model_name_for_dataset_storage(tokenizer_model),
        "num_samples": num_samples,
        "min_seq_len": min_seq_len,
        "max_seq_len": max_seq_len,
        "storage_format": "nemotron_bundle_v1",
        "files": {
            "hf_dataset": "save_to_disk 산출물 (data-*.arrow, dataset_info.json, state.json)",
            "samples_text": "samples_text.jsonl",
            "calib_chunks": "calib_chunks.pt",
        },
        "written_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    meta_path = dataset_dir / CALIB_DATASET_METADATA_FILENAME
    meta_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"[Calibration] Wrote metadata: {meta_path.resolve()}")


def _meta_matches(
    meta: CalibCacheMeta,
    model_name: str,
    num_samples: int,
    min_seq_len: int,
    max_seq_len: int,
) -> bool:
    return (
        meta.get("dataset") == CALIB_DATASET
        and meta.get("model_name") == model_name
        and int(meta.get("num_samples", -1)) == num_samples
        and int(meta.get("min_seq_len", -1)) == min_seq_len
        and int(meta.get("max_seq_len", -1)) == max_seq_len
        and meta.get("has_text") is True
        and meta.get("has_domain") is True
        and meta.get("storage_format") == "nemotron_bundle_v1"
    )


def _chunks_pt_cache_valid(chunks: list[dict[str, Any]]) -> bool:
    if not chunks or not isinstance(chunks[0], dict):
        return False
    first = chunks[0]
    return "text" in first and "domain" in first


def load_or_build_calibration_chunks(
    tokenizer: AutoTokenizer,
    dataset_dir: str,
    model_name: str,
    num_samples: int,
    min_seq_len: int,
    max_seq_len: int,
) -> list[dict[str, Any]]:
    """
    `dataset_dir`에 직접 HF Dataset(save_to_disk) + samples_text.jsonl + calib_chunks.pt 를 두며,
    메타 일치 시 로드, 없거나 불일치면 생성 후 저장. 하위 `calib_*` 폴더는 만들지 않는다.
    """
    bundle_dir = Path(dataset_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)
    storage_name = _model_name_for_dataset_storage(model_name)
    chunks_pt = calibration_chunks_pt_path(bundle_dir)

    if chunks_pt.is_file():
        try:
            payload = torch.load(
                chunks_pt, map_location="cpu", weights_only=False
            )
        except TypeError:
            payload = torch.load(chunks_pt, map_location="cpu")
        if isinstance(payload, dict) and "meta" in payload and "chunks" in payload:
            meta = payload["meta"]
            if isinstance(meta, dict) and _meta_matches(
                meta, storage_name, num_samples, min_seq_len, max_seq_len
            ):
                chunks = payload["chunks"]
                if not _chunks_pt_cache_valid(chunks):
                    print(
                        f"[Calibration] Cache chunks missing text/domain; rebuilding: {chunks_pt}"
                    )
                elif not _hf_disk_bundle_looks_valid(bundle_dir):
                    print(
                        f"[Calibration] HF disk bundle missing or incomplete; rebuilding: {bundle_dir}"
                    )
                else:
                    print(
                        f"[Calibration] Loaded cached calibration data: {chunks_pt} "
                        f"(dataset_dir={bundle_dir.resolve()}, meta OK: model={storage_name}, n={num_samples}, "
                        f"min={min_seq_len}, max={max_seq_len})"
                    )
                    _write_calibration_dataset_metadata(
                        bundle_dir,
                        tokenizer_model=model_name,
                        num_samples=num_samples,
                        min_seq_len=min_seq_len,
                        max_seq_len=max_seq_len,
                    )
                    return chunks
        print(
            f"[Calibration] Cache file present but invalid or meta mismatch, rebuilding: {chunks_pt}"
        )

    chunks = _build_calibration_chunks(
        tokenizer,
        num_samples,
        min_seq_len,
        max_seq_len,
        bundle_dir=bundle_dir,
    )
    meta: CalibCacheMeta = {
        "dataset": CALIB_DATASET,
        "model_name": storage_name,
        "num_samples": num_samples,
        "min_seq_len": min_seq_len,
        "max_seq_len": max_seq_len,
        "has_text": True,
        "has_domain": True,
        "storage_format": "nemotron_bundle_v1",
    }
    torch.save({"meta": meta, "chunks": chunks}, chunks_pt)
    print(f"[Calibration] Saved calibration chunks tensor cache to: {chunks_pt}")
    _write_calibration_dataset_metadata(
        bundle_dir,
        tokenizer_model=model_name,
        num_samples=num_samples,
        min_seq_len=min_seq_len,
        max_seq_len=max_seq_len,
    )
    return chunks


def get_calibration_dataloader(
    tokenizer: AutoTokenizer,
    num_samples: int = CALIB_SIZE,
    min_seq_len: int = CALIB_MIN_SEQ_LEN,
    max_seq_len: int = CALIB_MAX_SEQ_LEN,
):
    """(하위 호환) 캐시 없이 매번 Nemotron 캘리브레이션 배치만 생성."""
    return _build_calibration_chunks(
        tokenizer, num_samples, min_seq_len, max_seq_len
    )


def resolve_decoder_type(model: torch.nn.Module, decoder_type: str | None) -> str:
    """
    TensorRT-LLM export용 decoder_type.

    - MoE: `llama`는 Mixtral식 전문가 이름(w1,w2,w3)을 기대하고,
      Qwen3 MoE 등은 `gate_proj`/`up_proj`/`down_proj`이므로 반드시 `qwen`이어야 함.
    - `decoder_type`이 None이거나 'auto'이면 모델 클래스명으로 추론(get_model_type).
    """
    if decoder_type and decoder_type.lower() != "auto":
        return decoder_type
    inferred = get_model_type(model)
    if inferred is None:
        raise ValueError(
            "decoder_type을 지정하거나, ModelOpt가 인식하는 모델이어야 합니다. "
            "예: Llama 계열 → llama, Qwen(모E 포함) → qwen, Mixtral → llama"
        )
    return inferred


def build_calibration_dataset_only(
    model_name: str,
    dataset_dir: str,
    calib_size: int = CALIB_SIZE,
    min_seq_len: int = CALIB_MIN_SEQ_LEN,
    max_seq_len: int = CALIB_MAX_SEQ_LEN,
) -> Path:
    """
    전체 가중치 로드 없이 토크나이저만 사용해 Nemotron 캘리브 번들을 생성한다.

    `dataset_dir`에 직접 HF `save_to_disk` 산출물(data-*.arrow, dataset_info.json, state.json),
    `samples_text.jsonl`, GPTQ용 `calib_chunks.pt`, 메타 JSON 이 들어간다. 청크마다 `text`, `domain` 포함.

    이미 동일 형태의 번들이 있으면 갱신하지 않고 FileExistsError.
    """
    bundle_dir = Path(dataset_dir)
    if _calibration_bundle_present(bundle_dir):
        raise FileExistsError(
            f"캘리브레이션 데이터셋 디렉터리에 이미 데이터가 있습니다: {bundle_dir.resolve()}. "
            "다른 경로를 쓰거나 기존 내용을 제거한 뒤 다시 실행하세요."
        )
    chunks_pt = calibration_chunks_pt_path(bundle_dir)
    print(f"\n{'=' * 60}")
    print("  Calibration dataset only (no model load, no TRT-LLM export)")
    print(f"  Tokenizer / cache model id: {model_name}")
    print(f"  Dataset dir: {dataset_dir}")
    print(
        f"  Calib      : samples={calib_size}, min_len={min_seq_len}, max_len={max_seq_len}"
    )
    print(f"  Output root: {bundle_dir.resolve()}/")
    print(f"  Chunks .pt : {chunks_pt.name}")
    print(f"{'=' * 60}\n")

    print("[1/1] Loading tokenizer and building calibration cache...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_or_build_calibration_chunks(
        tokenizer=tokenizer,
        dataset_dir=dataset_dir,
        model_name=model_name,
        num_samples=calib_size,
        min_seq_len=min_seq_len,
        max_seq_len=max_seq_len,
    )
    _write_quant_metadata(
        dataset_dir,
        model_name,
        "CALIB_DATASET_ONLY",
        calib_size=calib_size,
        calib_min_seq_len=min_seq_len,
        calib_max_seq_len=max_seq_len,
    )
    print(f"\n[Done] Calibration bundle ready: {bundle_dir.resolve()}")
    return bundle_dir


def load_model_for_export(model_name: str, decoder_type: str | None):
    """토크나이저·모델 로드 및 TensorRT-LLM decoder_type 결정."""
    print("[1/4] Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    export_decoder_type = resolve_decoder_type(model, decoder_type)
    print(f"  Model loaded on: {DEVICE}")
    print(f"  TensorRT-LLM decoder_type: {export_decoder_type}")
    return tokenizer, model, export_decoder_type


def _write_quant_metadata(
    save_dir: str,
    model_name: str,
    quantization: str,
    calib_size: int | None,
    calib_min_seq_len: int | None = None,
    calib_max_seq_len: int | None = None,
    extra: dict | None = None,
):
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    config: dict[str, Any] = {
        "model_name": model_name,
        "quantization": quantization,
        "save_dir": save_dir,
    }
    if calib_size is not None:
        config["calib_dataset"] = CALIB_DATASET
        config["calib_size"] = calib_size
        config["calib_min_seq_len"] = (
            calib_min_seq_len if calib_min_seq_len is not None else CALIB_MIN_SEQ_LEN
        )
        config["calib_max_seq_len"] = (
            calib_max_seq_len if calib_max_seq_len is not None else CALIB_MAX_SEQ_LEN
        )
    if quantization in ("GPTQ_W4A16", "RTN_W4A16"):
        config["group_size"] = 128
    if extra:
        config.update(extra)
    with open(f"{save_dir}/quant_config.json", "w") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


def _export_to_tensorrt_llm(model: torch.nn.Module, export_decoder_type: str, save_dir: str):
    print("[4/4] Exporting TensorRT-LLM checkpoint...")
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    export_tensorrt_llm_checkpoint(
        model=model,
        decoder_type=export_decoder_type,
        dtype=torch.bfloat16,
        export_dir=save_dir,
        inference_tensor_parallel=1,
        inference_pipeline_parallel=1,
    )


def export_unquantized_to_tensorrt_llm(
    model_name: str,
    save_dir: str,
    decoder_type: str | None = "auto",
):
    """
    양자화 없이 Hugging Face 모델을 TensorRT-LLM checkpoint로 저장.
    """
    print(f"\n{'=' * 60}")
    print("  HF → TensorRT-LLM checkpoint (UNQUANTIZED)")
    print(f"  Model   : {model_name}")
    print(f"  Save to : {save_dir}")
    print(f"{'=' * 60}\n")

    _, model, export_decoder_type = load_model_for_export(model_name, decoder_type)

    print("[2/4] Skipping calibration (not required for UNQUANTIZED).")
    print("[3/4] Skipping GPTQ quantization.")

    _export_to_tensorrt_llm(model, export_decoder_type, save_dir)
    _write_quant_metadata(save_dir, model_name, "UNQUANTIZED", calib_size=None)

    print(f"\n[Done] TensorRT-LLM checkpoint saved to: {save_dir}")
    print(f"       Files: {list(Path(save_dir).iterdir())}")
    return save_dir


def _int4_w4a16_quant_config() -> dict[str, Any]:
    """INT4 블록 weight-only(W4A16) 공통 설정. 전역 기본 dict 변조를 피하기 위해 deepcopy."""
    quant_config = copy.deepcopy(mtq.INT4_BLOCKWISE_WEIGHT_ONLY_CFG)
    quant_config["quant_cfg"]["*lm_head*"] = {"enable": False}
    quant_config["quant_cfg"]["*embed_tokens*"] = {"enable": False}
    quant_config["quant_cfg"]["*gate.weight"] = {"enable": False}
    return quant_config


def quantize_with_rtn_w4a16(
    model_name: str,
    save_dir: str,
    decoder_type: str | None = "auto",
):
    """
    Calibration 없이 weight-only INT4 블록 양자화(ModelOpt `algorithm`: max, `forward_loop`=None).

    가중치 스케일은 블록별 min–max로 잡고 반올림하는 RTN 스타일 PTQ에 해당합니다.
    """
    print(f"\n{'=' * 60}")
    print("  ModelOpt W4A16 weight-only (RTN / min–max, no calibration forward)")
    print(f"  Model   : {model_name}")
    print(f"  Save to : {save_dir}")
    print(f"{'=' * 60}\n")

    _, model, export_decoder_type = load_model_for_export(model_name, decoder_type)

    print("[2/4] Skipping calibration data (not used for RTN_W4A16).")
    print("[3/4] Applying INT4 block weight-only quantization (max, weight-only)...")

    quant_config = _int4_w4a16_quant_config()
    mtq.quantize(model, quant_config, forward_loop=None)
    print("  RTN (min–max weight-only) quantization complete.")

    _export_to_tensorrt_llm(model, export_decoder_type, save_dir)
    _write_quant_metadata(save_dir, model_name, "RTN_W4A16", calib_size=None)

    print(f"\n[Done] Quantized checkpoint saved to: {save_dir}")
    print(f"       Files: {list(Path(save_dir).iterdir())}")
    return save_dir


def build_gptq_hessian_cache(
    model_name: str,
    dataset_dir: str,
    hessian_state_path: str,
    calib_size: int = CALIB_SIZE,
    min_seq_len: int = CALIB_MIN_SEQ_LEN,
    max_seq_len: int = CALIB_MAX_SEQ_LEN,
    decoder_type: str | None = "auto",
    *,
    gptq_percdamp: float = 0.01,
    gptq_block_size: int = 128,
):
    """
    ModelOpt `gptq_lite` Hessian만 계산(또는 캐시 로드) 후 디스크에 저장하고 종료.

    Phase-4(가중치 GPTQ 업데이트)와 TensorRT-LLM export는 수행하지 않는다.
    이후 `GPTQ_W4A16`으로 동일 `hessian_state_path`를 주면 Hessian 로드 후 Phase-4만 진행된다.
    """
    bundle_dir = Path(dataset_dir)
    chunks_pt = calibration_chunks_pt_path(bundle_dir)
    hpath = Path(hessian_state_path)

    print(f"\n{'=' * 60}")
    print("  GPTQ-lite Hessian cache only (no Phase-4, no TRT-LLM export)")
    print(f"  Model   : {model_name}")
    print(f"  Dataset dir (calib cache): {dataset_dir}")
    print(
        f"  Calib   : samples={calib_size}, min_len={min_seq_len}, max_len={max_seq_len}"
    )
    print(f"  Calib files (HF+jsonl+pt): {bundle_dir.resolve()}/ → {chunks_pt.name}")
    print(f"  Hessian out : {hpath.resolve()} (상위 디렉터리까지 생성)")
    print(
        f"  gptq_lite (Phase 1–3): percdamp={gptq_percdamp}, block_size={gptq_block_size}"
    )
    print(f"{'=' * 60}\n")

    tokenizer, model, _export_decoder_type = load_model_for_export(model_name, decoder_type)

    print("[2/3] Preparing calibration data...")
    calib_dataloader = load_or_build_calibration_chunks(
        tokenizer=tokenizer,
        dataset_dir=dataset_dir,
        model_name=model_name,
        num_samples=calib_size,
        min_seq_len=min_seq_len,
        max_seq_len=max_seq_len,
    )

    print("[3/3] Running gptq_lite through Hessian (patched: Phase-4 skipped)...")
    quant_config = _int4_w4a16_quant_config()
    quant_config["algorithm"] = {
        "method": "gptq_lite",
        "percdamp": gptq_percdamp,
        "block_size": gptq_block_size,
        "hessian_state_path": str(hpath),
    }

    def calibrate_loop(model_inner):
        for batch in calib_dataloader:
            input_ids = batch["input_ids"].to(DEVICE)
            attn_mask = batch["attention_mask"].to(DEVICE)
            with torch.no_grad():
                model_inner(input_ids=input_ids, attention_mask=attn_mask)

    install_gptq_hessian_only_patch()
    try:
        mtq.quantize(model, quant_config, forward_loop=calibrate_loop)
    finally:
        restore_parallel_gptq_patch()

    meta_path = hpath.parent / "hessian_stage_metadata.json"
    hpath.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(
        json.dumps(
            {
                "stage": "HESSIAN",
                "model_name": model_name,
                "hessian_state_path": str(hpath.resolve()),
                "calibration_dataset_dir": str(bundle_dir.resolve()),
                "calib_dataset": CALIB_DATASET,
                "calib_size": calib_size,
                "calib_min_seq_len": min_seq_len,
                "calib_max_seq_len": max_seq_len,
                "gptq_percdamp": gptq_percdamp,
                "gptq_block_size": gptq_block_size,
                "written_at_utc": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"  Wrote metadata: {meta_path.resolve()}")
    print(f"\n[Done] Hessian stage complete. Next: --quantize GPTQ_W4A16 (same --hessian_state_path / dataset default).")
    return str(hpath.resolve())


def quantize_with_gptq(
    model_name: str,
    save_dir: str,
    dataset_dir: str,
    calib_size: int = CALIB_SIZE,
    min_seq_len: int = CALIB_MIN_SEQ_LEN,
    max_seq_len: int = CALIB_MAX_SEQ_LEN,
    decoder_type: str | None = "auto",
    *,
    gptq_percdamp: float = 0.01,
    gptq_block_size: int = 128,
    hessian_state_path: str | None = None,
    parallel_gptq_batch: int = 1,
    require_cached_hessian: bool = False,
):
    """
    ModelOpt `gptq_lite` 양자화 수행 (INT4 블록 W4A16).

    내부적으로 `max`로 1차 스케일 잡은 뒤, 캘리브 forward로 Hessian을 모아
    블록 단위 가중치를 갱신합니다. (ModelOpt 0.43+ `GPTQLiteConfig`, 실험적)

    캘리브레이션 데이터는 `dataset_dir`에 직접 저장·로드한다(하위 calib_* 폴더 없음).
    `hessian_state_path`가 None이 아니면 해당 .pt에 Hessian을 두고, 파일이 있으면 로드·없으면 계산 후 저장.

    `parallel_gptq_batch`: 1이면 ModelOpt 기본 동작. 1보다 크면 `gptq_lite` Phase-4에서
    매 청크마다 여러 Linear를 서로 다른 CUDA 스트림으로 겹쳐 실행(피크 VRAM 상승 가능).

    `require_cached_hessian`: True이면 `hessian_state_path`가 가리키는 Hessian .pt가
    반드시 존재해야 하며(캘리브 forward·Hessian 재계산 생략), 없으면 예외를 던진다.

    TensorRT-LLM checkpoint는 `save_dir`에 저장한다.
    """
    if require_cached_hessian:
        if not hessian_state_path:
            raise ValueError(
                "require_cached_hessian=True 인데 hessian_state_path 가 비어 있습니다."
            )
        hp = Path(hessian_state_path)
        if not hp.is_file():
            raise FileNotFoundError(
                f"require_cached_hessian: Hessian 캐시 파일이 없습니다: {hp.resolve()}"
            )

    bundle_dir = Path(dataset_dir)
    chunks_pt = calibration_chunks_pt_path(bundle_dir)
    print(f"\n{'=' * 60}")
    print("  ModelOpt GPTQ-lite (gptq_lite) W4A16")
    print(f"  Model   : {model_name}")
    print(f"  Save to : {save_dir}")
    print(f"  Dataset dir (calib cache): {dataset_dir}")
    print(
        f"  Calib   : samples={calib_size}, min_len={min_seq_len}, max_len={max_seq_len}"
    )
    print(f"  Calib files (HF+jsonl+pt): {bundle_dir.resolve()}/ → {chunks_pt.name}")
    if require_cached_hessian:
        print(f"  Hessian cache : 필수 (로드만, 파일 없으면 오류) → {Path(hessian_state_path).resolve()}")
    elif hessian_state_path:
        hpath = Path(hessian_state_path)
        print(
            f"  Hessian cache : {hpath.resolve()} "
            "(존재 시 로드, 없으면 계산 후 상위 디렉터리까지 생성하며 저장)"
        )
    else:
        print("  Hessian cache : (off — 매 실행마다 Hessian만 계산, 디스크에 저장 안 함)")
    print(
        f"  gptq_lite     : percdamp={gptq_percdamp}, block_size={gptq_block_size} "
        f"(group_size=128과 정렬)"
    )
    print(
        f"  parallel_gptq_batch: {parallel_gptq_batch} "
        "(1=ModelOpt 기본 직렬 Phase-4, >1=CUDA 스트림으로 청크당 동시 처리)"
    )
    print(f"{'=' * 60}\n")

    tokenizer, model, export_decoder_type = load_model_for_export(model_name, decoder_type)

    # 2) Calibration 데이터 준비 (dataset_dir 캐시 또는 생성 후 저장)
    print("[2/4] Preparing calibration data...")
    calib_dataloader = load_or_build_calibration_chunks(
        tokenizer=tokenizer,
        dataset_dir=dataset_dir,
        model_name=model_name,
        num_samples=calib_size,
        min_seq_len=min_seq_len,
        max_seq_len=max_seq_len,
    )

    # 3) gptq_lite (Hessian 기반) + INT4 블록 W4A16
    print("[3/4] Applying gptq_lite quantization (INT4 block W4A16)...")

    quant_config = _int4_w4a16_quant_config()
    quant_config["algorithm"] = {
        "method": "gptq_lite",
        "percdamp": gptq_percdamp,
        "block_size": gptq_block_size,
        "hessian_state_path": hessian_state_path,
    }

    def calibrate_loop(model_inner):
        """캘리브 배치로 forward → gptq_lite가 활성화된 linear에 Hessian 누적."""
        for batch in calib_dataloader:
            input_ids = batch["input_ids"].to(DEVICE)
            attn_mask = batch["attention_mask"].to(DEVICE)
            with torch.no_grad():
                model_inner(input_ids=input_ids, attention_mask=attn_mask)

    install_parallel_gptq_patch_if_needed(parallel_gptq_batch)
    try:
        mtq.quantize(model, quant_config, forward_loop=calibrate_loop)
    finally:
        restore_parallel_gptq_patch()
    print("  gptq_lite quantization complete.")

    _export_to_tensorrt_llm(model, export_decoder_type, save_dir)
    _write_quant_metadata(
        save_dir,
        model_name,
        "GPTQ_W4A16",
        calib_size=calib_size,
        calib_min_seq_len=min_seq_len,
        calib_max_seq_len=max_seq_len,
        extra={
            "calibration_dataset_dir": dataset_dir,
            "mtq_algorithm": "gptq_lite",
            "gptq_percdamp": gptq_percdamp,
            "gptq_block_size": gptq_block_size,
            "hessian_state_path": hessian_state_path,
            "parallel_gptq_batch": parallel_gptq_batch,
            "require_cached_hessian": require_cached_hessian,
        },
    )

    print(f"\n[Done] Quantized checkpoint saved to: {save_dir}")
    print(f"       Files: {list(Path(save_dir).iterdir())}")
    return save_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ModelOpt GPTQ Quantization / TRT-LLM export")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="HuggingFace 모델 ID")
    parser.add_argument(
        "--quantize",
        default="UNQUANTIZED",
        choices=["UNQUANTIZED", "GPTQ_W4A16", "HESSIAN", "RTN_W4A16", "CALIB_DATASET_ONLY"],
        help=(
            "UNQUANTIZED: HF→TensorRT-LLM만. "
            "GPTQ_W4A16: INT4 블록 W4A16 + ModelOpt gptq_lite 후 export (Hessian 캐시 사용 가능). "
            "HESSIAN: gptq_lite Hessian만 계산·저장(Phase-4·export 없음). 이후 GPTQ_W4A16 단독 실행용. "
            "RTN_W4A16: 동일 포맷, calibration 없이 weight min–max만. "
            "CALIB_DATASET_ONLY: 모델 로드 없이 Nemotron 캘리브 번들(HF 디스크+samples_text.jsonl+calib_chunks.pt)만 생성. 이미 있으면 오류."
        ),
    )
    parser.add_argument(
        "--save_dir",
        default=DEFAULT_SAVE_DIR,
        help="TensorRT-LLM checkpoint (및 quant_config.json) 저장 경로",
    )
    parser.add_argument(
        "--dataset_dir",
        default=None,
        help=(
            "캘리브레이션 데이터셋 디렉터리 (GPTQ_W4A16, HESSIAN, CALIB_DATASET_ONLY 필수). "
            "하위 폴더 없이 HF 디스크·samples_text.jsonl·calib_chunks.pt·calibration_dataset_metadata.json 을 이 경로에 둔다."
        ),
    )
    parser.add_argument(
        "--calib_size",
        default=CALIB_SIZE,
        type=int,
        help="Calibration 샘플 수 (GPTQ_W4A16, HESSIAN, CALIB_DATASET_ONLY)",
    )
    parser.add_argument(
        "--calib_min_seq_len",
        default=CALIB_MIN_SEQ_LEN,
        type=int,
        help="캘리브 토큰 길이 하한 (GPTQ_W4A16, HESSIAN, CALIB_DATASET_ONLY, 캐시 파일명·메타에 반영)",
    )
    parser.add_argument(
        "--calib_max_seq_len",
        default=CALIB_MAX_SEQ_LEN,
        type=int,
        help="캘리브 토큰 길이 상한 (GPTQ_W4A16, HESSIAN, CALIB_DATASET_ONLY, 캐시 파일명·메타에 반영)",
    )
    parser.add_argument(
        "--decoder_type",
        default="auto",
        help='TensorRT-LLM export용 아키텍처 (기본: auto = 모델에서 추론). Qwen3 MoE는 "qwen", Llama/Mixtral MoE는 "llama"',
    )
    parser.add_argument(
        "--gptq_percdamp",
        type=float,
        default=0.01,
        help="gptq_lite: Hessian 대각 댐핑 비율 (GPTQ 스타일, 기본 0.01)",
    )
    parser.add_argument(
        "--gptq_block_size",
        type=int,
        default=128,
        help="gptq_lite: 가중치 블록 업데이트 크기 (INT4 group_size 128과 정렬 권장)",
    )
    parser.add_argument(
        "--hessian_state_path",
        default=None,
        metavar="PATH",
        help=(
            "gptq_lite / HESSIAN: Hessian 상태 .pt 경로. "
            "지정하지 않으면 --dataset_dir 하위 "
            "`{모델베이스이름}-hessian/hessian_state.pt` 기본. "
            "GPTQ_W4A16: 있으면 로드·없으면 계산 후 저장(--hessian_no_cache 제외). "
            "HESSIAN: 이 경로에 저장(상위 디렉터리 생성)."
        ),
    )
    parser.add_argument(
        "--hessian_no_cache",
        action="store_true",
        help=(
            "gptq_lite: Hessian을 디스크에 두지 않음(매 실행 전부 계산). "
            "--hessian_state_path 와 동시에 쓰면 오류."
        ),
    )
    parser.add_argument(
        "--parallel_gptq_batch",
        type=int,
        default=1,
        metavar="N",
        help=(
            "gptq_lite Phase-4: 청크 크기. 1=ModelOpt와 동일 직렬. "
            ">1이면 청크 안에서 처리하되, 같은 GPU만 있으면 wall time은 거의 직렬(각 Linear가 Cholesky 등으로 GPU를 크게 씀). "
            "청크에 서로 다른 CUDA 디바이스에 올라간 모듈이 섞일 때만 스레드로 디바이스별 진짜 병렬(예: device_map 다중 GPU)."
        ),
    )
    parser.add_argument(
        "--gptq_require_cached_hessian",
        action="store_true",
        help=(
            "GPTQ_W4A16 전용: Hessian .pt가 이미 있어야 함(경로는 --hessian_state_path 또는 dataset_dir 기본). "
            "없으면 오류. 캘리브 forward로 Hessian을 다시 쌓지 않음."
        ),
    )
    args = parser.parse_args()

    if args.hessian_no_cache and args.hessian_state_path:
        parser.error("--hessian_no_cache 와 --hessian_state_path 는 함께 쓸 수 없습니다.")
    if args.parallel_gptq_batch < 1:
        parser.error("--parallel_gptq_batch 는 1 이상이어야 합니다.")
    if args.gptq_require_cached_hessian and args.quantize != "GPTQ_W4A16":
        parser.error("--gptq_require_cached_hessian 은 --quantize GPTQ_W4A16 과만 함께 쓸 수 있습니다.")
    if args.gptq_require_cached_hessian and args.hessian_no_cache:
        parser.error("--gptq_require_cached_hessian 와 --hessian_no_cache 는 함께 쓸 수 없습니다.")

    if args.quantize in ("GPTQ_W4A16", "HESSIAN", "CALIB_DATASET_ONLY"):
        if not args.dataset_dir:
            parser.error(
                "--dataset_dir is required for GPTQ_W4A16, HESSIAN, and CALIB_DATASET_ONLY"
            )

    resolved_hessian_path: str | None = None
    if args.quantize in ("GPTQ_W4A16", "HESSIAN"):
        if args.hessian_no_cache:
            if args.quantize == "HESSIAN":
                parser.error("HESSIAN 단계는 Hessian을 디스크에 저장해야 하므로 --hessian_no_cache 는 사용할 수 없습니다.")
            resolved_hessian_path = None
        elif args.hessian_state_path:
            resolved_hessian_path = args.hessian_state_path
        else:
            resolved_hessian_path = default_gptq_hessian_state_pt(
                args.dataset_dir, args.model
            )

    start_time = time.time()

    if args.quantize == "GPTQ_W4A16":
        quantize_with_gptq(
            model_name=args.model,
            save_dir=args.save_dir,
            dataset_dir=args.dataset_dir,
            calib_size=args.calib_size,
            min_seq_len=args.calib_min_seq_len,
            max_seq_len=args.calib_max_seq_len,
            decoder_type=args.decoder_type,
            gptq_percdamp=args.gptq_percdamp,
            gptq_block_size=args.gptq_block_size,
            hessian_state_path=resolved_hessian_path,
            parallel_gptq_batch=args.parallel_gptq_batch,
            require_cached_hessian=args.gptq_require_cached_hessian,
        )
    elif args.quantize == "HESSIAN":
        assert resolved_hessian_path is not None
        build_gptq_hessian_cache(
            model_name=args.model,
            dataset_dir=args.dataset_dir,
            hessian_state_path=resolved_hessian_path,
            calib_size=args.calib_size,
            min_seq_len=args.calib_min_seq_len,
            max_seq_len=args.calib_max_seq_len,
            decoder_type=args.decoder_type,
            gptq_percdamp=args.gptq_percdamp,
            gptq_block_size=args.gptq_block_size,
        )
    elif args.quantize == "CALIB_DATASET_ONLY":
        build_calibration_dataset_only(
            model_name=args.model,
            dataset_dir=args.dataset_dir,
            calib_size=args.calib_size,
            min_seq_len=args.calib_min_seq_len,
            max_seq_len=args.calib_max_seq_len,
        )
    elif args.quantize == "RTN_W4A16":
        quantize_with_rtn_w4a16(
            model_name=args.model,
            save_dir=args.save_dir,
            decoder_type=args.decoder_type,
        )
    elif args.quantize == "UNQUANTIZED":
        export_unquantized_to_tensorrt_llm(
            model_name=args.model,
            save_dir=args.save_dir,
            decoder_type=args.decoder_type,
        )
    else:
        raise ValueError(f"Unsupported --quantize: {args.quantize}")

    end_time = time.time()
    print(f"Elapsed time: {(end_time - start_time) / 60:.2f} minutes")
