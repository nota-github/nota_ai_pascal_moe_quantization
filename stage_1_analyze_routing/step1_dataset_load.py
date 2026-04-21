"""
캘리브레이션용 데이터셋 생성 (단일 스크립트, 외부 레포 import 없음).

- dataset-kind=custom (기본): Nemotron SFT Math-v3 + Competitive Programming-v2 스트리밍.
- dataset-kind=nemo_dataset: nvidia/Nemotron-Post-Training-Dataset-v1 의 chat / math / code / stem
  분할을 도메인당 동일 샘플 수(기본 25개씩, 총 100)로 수집. 행 메타 `reasoning` 플래그가 \"on\" 인 샘플만 사용
  (chat 은 on/off 혼재, math/code/stem 은 데이터상 on).

길이 필터·선택적 smart_truncate 후 HF Dataset(`input_ids`, `attention_mask`) 저장,
동일 디렉터리에 `samples_text.jsonl`(각 줄 `sample_id`, `text`)을 둔다.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from datasets import Dataset, IterableDataset, interleave_datasets, load_dataset

# --- Nemotron competitive code: HF JSON 스트리밍(load_dataset)은 행마다 스키마가 달라 CastError 가 나므로
#     레포의 JSONL을 직접 읽는다 (hf_hub list_files 기준 exercism / text_to_sql 도 단일 jsonl).
_NEMOTRON_COMPETITIVE_PROG_REPO = "nvidia/Nemotron-SFT-Competitive-Programming-v2"
_NEMOTRON_POST_TRAINING_V1_REPO = "nvidia/Nemotron-Post-Training-Dataset-v1"
_NEMOTRON_COMPETITIVE_CODE_JSONL_BY_SPLIT = {
    "competitive_coding_cpp": (
        "data/competitive_programming_cpp_00.jsonl",
        "data/competitive_programming_cpp_01.jsonl",
    ),
    "competitive_coding_python": (
        "data/competitive_programming_python_00.jsonl",
        "data/competitive_programming_python_01.jsonl",
    ),
    "exercism": ("data/exercism.jsonl",),
    "text_to_sql": ("data/text_to_sql.jsonl",),
}


def _normalize_nemotron_top_level_dataset_field(row):
    out = dict(row)
    d = out.get("dataset")
    if isinstance(d, list):
        out["dataset"] = ",".join(str(x) for x in d)
    elif d is not None and not isinstance(d, str):
        out["dataset"] = str(d)
    return out


def _iter_nemotron_competitive_code_jsonl(repo_id, rel_paths, revision):
    from huggingface_hub import hf_hub_download

    for rel in rel_paths:
        path = hf_hub_download(repo_id, rel, repo_type="dataset", revision=revision)
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield _normalize_nemotron_top_level_dataset_field(json.loads(line))


def _load_nemotron_competitive_code_split_streaming(repo_id, split_name, revision=None):
    rel_paths = _NEMOTRON_COMPETITIVE_CODE_JSONL_BY_SPLIT.get(split_name)
    if rel_paths is None:
        raise ValueError(f"unknown Nemotron competitive code split: {split_name!r}")
    return IterableDataset.from_generator(
        _iter_nemotron_competitive_code_jsonl,
        gen_kwargs={
            "repo_id": repo_id,
            "rel_paths": rel_paths,
            "revision": revision,
        },
    )


def _messages_to_plaintext_fallback(messages):
    parts = []
    for m in messages:
        if isinstance(m, dict):
            role = m.get("role") or ""
            body = m.get("content")
            if body is None:
                body = ""
            parts.append(f"{role}:\n{body}".strip())
        else:
            parts.append(str(m))
    return "\n\n".join(parts)


def apply_chat_template_or_fallback(
    tokenizer,
    messages,
    *,
    tokenize=False,
    thinking_mode=None,
    add_generation_prompt=False,
):
    if not hasattr(tokenizer, "apply_chat_template"):
        return _messages_to_plaintext_fallback(messages)
    sig = inspect.signature(tokenizer.apply_chat_template)
    kwargs = {"tokenize": tokenize}
    if "add_generation_prompt" in sig.parameters:
        kwargs["add_generation_prompt"] = add_generation_prompt
    if thinking_mode is not None and "enable_thinking" in sig.parameters:
        kwargs["enable_thinking"] = thinking_mode
    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except Exception:
        return _messages_to_plaintext_fallback(messages)


def _prepare_math_code_mix_dataset(
    tokenizer,
    thinking_mode,
    MIN_SEQUENCE_LENGTH,
    MAX_SEQUENCE_LENGTH,
    NUM_CALIBRATION_SAMPLES,
    *,
    dataset_config: dict[str, Any],
    always_truncate_over_max=False,
    log_tag="[math_code]",
    text_jsonl_path: str | os.PathLike[str] | None = None,
    log_per_domain_token_stats: bool = False,
):
    """math + code 소스만 가정한 스트리밍 수집·토큰화 (원본 load_pipeline_data.prepare_custom_reasoning_dataset 동등).

    `text_jsonl_path`가 주어지면 셔플 후 `input_ids`를 디코드한 문자열을 JSONL(`sample_id`, `text`)로 저장한다.
    `log_per_domain_token_stats`: True이면 `_source` 도메인별 `len(input_ids)` 합·샘플 수·평균을 stdout에 출력.
    """
    DATASET_CONFIG = dataset_config

    _raw_pe = os.environ.get("CUSTOM_REASONING_COLLECTION_PROGRESS_EVERY", "10").strip()
    try:
        collection_progress_every = int(_raw_pe)
    except ValueError:
        collection_progress_every = 10

    def _normalize_chat_messages_row(example):
        out = dict(example)
        msgs = out.get("messages")
        if msgs is None:
            return out
        fixed = []
        for m in msgs:
            if not isinstance(m, dict):
                fixed.append(m)
                continue
            entry = dict(m)
            role = entry.get("role") or ""
            entry["role"] = role
            if entry.get("content") is None:
                entry["content"] = ""
            rc = entry.pop("reasoning_content", None)
            if rc and not entry.get("reasoning"):
                entry["reasoning"] = rc
            r = entry.get("reasoning")
            entry["reasoning"] = "" if r is None else str(r)
            tid = entry.get("tool_call_id")
            entry["tool_call_id"] = "" if tid is None else str(tid)
            fixed.append(entry)
        out["messages"] = fixed
        return out

    def _unify_messages_schema_row(example):
        """샤드마다 messages struct 필드가 달라 interleave 시 CastError 가 나므로, 동일 키만 남긴다."""
        msgs = example.get("messages")
        if msgs is None:
            return {"messages": []}
        fixed = []
        for m in msgs:
            if not isinstance(m, dict):
                continue
            fixed.append(
                {
                    "role": str(m.get("role") or ""),
                    "content": str(m.get("content") or ""),
                    "reasoning": str(m.get("reasoning") or ""),
                    "tool_call_id": str(m.get("tool_call_id") or ""),
                }
            )
        return {"messages": fixed}

    def _drop_row_keys_factory(keys):
        def _fn(example):
            out = dict(example)
            for k in keys:
                out.pop(k, None)
            return out

        return _fn

    def _maybe_drop_keys(stream, cfg):
        keys = cfg.get("drop_row_keys")
        if keys:
            return stream.map(_drop_row_keys_factory(keys))
        return stream

    def _strip_pil_images_row(example):
        try:
            from PIL.Image import Image as PILImage
        except ImportError:
            return dict(example)
        out = dict(example)
        for k in list(out):
            if isinstance(out[k], PILImage):
                out.pop(k, None)
        return out

    def _strip_pil_stream(stream):
        return stream.map(_strip_pil_images_row)

    def _row_top_level_reasoning_on(example) -> bool:
        """HF Nemotron-Post-Training-Dataset-v1 등: 행 최상위 `reasoning` 이 \"on\" 일 때만 통과."""
        v = example.get("reasoning")
        if v is None:
            return False
        return str(v).strip().lower() == "on"

    def _apply_reasoning_on_filter(stream, cfg):
        if not cfg.get("filter_reasoning_on"):
            return stream
        return stream.filter(_row_top_level_reasoning_on)

    def _load_streaming_for_cfg(cfg):
        repo = cfg["path"]
        config_name = cfg.get("config")
        if "splits" in cfg:
            substreams = []
            for sp in cfg["splits"]:
                if repo == _NEMOTRON_COMPETITIVE_PROG_REPO and sp in _NEMOTRON_COMPETITIVE_CODE_JSONL_BY_SPLIT:
                    sub = _load_nemotron_competitive_code_split_streaming(repo, sp)
                elif config_name is not None:
                    sub = load_dataset(repo, config_name, split=sp, streaming=True)
                else:
                    sub = load_dataset(repo, split=sp, streaming=True)
                sub = sub.map(_normalize_chat_messages_row)
                sub = _apply_reasoning_on_filter(sub, cfg)
                sub = sub.map(_unify_messages_schema_row)
                sub = _maybe_drop_keys(sub, cfg)
                substreams.append(sub)
            w = [1.0 / len(substreams)] * len(substreams)
            merged = interleave_datasets(substreams, probabilities=w, seed=42)
            return _strip_pil_stream(merged)
        split = cfg.get("split", "train")
        if config_name is not None:
            sub = load_dataset(repo, config_name, split=split, streaming=True)
        else:
            sub = load_dataset(repo, split=split, streaming=True)
        sub = sub.map(_normalize_chat_messages_row)
        sub = _apply_reasoning_on_filter(sub, cfg)
        sub = sub.map(_unify_messages_schema_row)
        return _strip_pil_stream(_maybe_drop_keys(sub, cfg))

    ALLOW_TRUNCATION = True
    TRUNCATION_FRACTION = 0.2
    TRUNCATION_FRACTION_STEP = 0.2

    def _allocate_per_source_counts(total_n, config):
        keys = list(config.keys())
        weights = [config[k]["weight"] for k in keys]
        s = sum(weights)
        if abs(s - 1.0) > 1e-5:
            raise ValueError(f"{log_tag} weights must sum to 1.0, got {s}")
        exact = [total_n * w for w in weights]
        floors = [int(e) for e in exact]
        remainder = total_n - sum(floors)
        fracs = sorted([(exact[i] - floors[i], i) for i in range(len(keys))], reverse=True)
        counts = floors[:]
        for r in range(remainder):
            counts[fracs[r][1]] += 1
        return dict(zip(keys, counts))

    def preprocess(example):
        if "messages" in example:
            text = apply_chat_template_or_fallback(
                tokenizer,
                example["messages"],
                tokenize=False,
                thinking_mode=thinking_mode,
            )
        elif "question" in example and "reference_answer" in example:
            blocks = []
            for r in example.get("responses") or []:
                if not isinstance(r, dict):
                    continue
                body = r.get("response")
                if not body:
                    continue
                model = r.get("response_model")
                if model:
                    blocks.append(f"[{model}]\n{body}")
                else:
                    blocks.append(str(body))
            cot = "\n\n---\n\n".join(blocks)
            text = (
                f"Question:\n{example['question']}\n\n"
                f"Reasoning:\n{cot}\n\n"
                f"Answer:\n{example['reference_answer']}"
            )
        elif "Question" in example and "Answer" in example:
            q = example["Question"]
            if example.get("Picture"):
                q = f"{q}\n\n(Note: this item includes an image in the dataset; only the text question is shown here.)"
            ans_type = example.get("Answer_type")
            type_line = f"\nAnswer type: {ans_type}" if ans_type is not None else ""
            text = f"Question:\n{q}{type_line}\n\nAnswer:\n{example['Answer']}"
        elif "question" in example and "answer" in example:
            cot = example.get("reasoning", "")
            text = f"Question:\n{example['question']}\n\nReasoning:\n{cot}\n\nAnswer:\n{example['answer']}"
        elif "problem" in example and "solution" in example:
            text = f"Problem:\n{example['problem']}\n\nSolution:\n{example['solution']}"
        else:
            text = str(example)
        return {"text": text}

    def _print_per_source_prompt_previews(max_chars=2400, max_tries=16):
        print(
            f"\n{log_tag} Per-source prompting preview "
            "(first successful row per source after preprocess)\n"
        )
        for src_name, cfg in DATASET_CONFIG.items():
            repo = cfg["path"]
            banner = f"========== source={src_name!r}  path={repo!r} =========="
            print(banner)
            stream = _load_streaming_for_cfg(cfg)
            text = None
            last_err = None
            it = iter(stream)
            for _ in range(max_tries):
                try:
                    ex = next(it)
                except StopIteration:
                    last_err = "stream exhausted"
                    break
                except Exception as e:
                    last_err = repr(e)
                    break
                try:
                    t = preprocess(ex)["text"]
                except Exception as e:
                    last_err = repr(e)
                    continue
                if isinstance(t, str) and t.strip():
                    text = t
                    break
            if not text:
                print(f"(no preview: {last_err or 'empty text'})")
                print()
                continue
            full_len = len(text)
            if full_len <= max_chars:
                print(text)
            else:
                print(text[:max_chars])
                print(f"\n... [truncated, {full_len} chars total]\n")
            print()

    _print_per_source_prompt_previews()

    quotas = _allocate_per_source_counts(NUM_CALIBRATION_SAMPLES, DATASET_CONFIG)
    if any(DATASET_CONFIG[k].get("filter_reasoning_on") for k in DATASET_CONFIG):
        print(
            f"{log_tag} Streaming filter active: top-level column reasoning == 'on' "
            "(applied before message schema unify; chat split drops reasoning=off rows)."
        )
    print(f"{log_tag} Per-source target counts (largest remainder on weights):")
    print(
        f"NUM_CALIBRATION_SAMPLES: {NUM_CALIBRATION_SAMPLES}, "
        f"MIN_SEQUENCE_LENGTH: {MIN_SEQUENCE_LENGTH}, MAX_SEQUENCE_LENGTH: {MAX_SEQUENCE_LENGTH}"
    )
    for k in DATASET_CONFIG:
        print(f"  {k}: {quotas[k]} (weight={DATASET_CONFIG[k]['weight']})")
    if collection_progress_every > 0:
        print(
            f"{log_tag} collect: periodic progress every {collection_progress_every} "
            "accepted samples (CUSTOM_REASONING_COLLECTION_PROGRESS_EVERY)"
        )
    else:
        print(
            f"{log_tag} collect: progress = start + done per source only "
            "(CUSTOM_REASONING_COLLECTION_PROGRESS_EVERY=0)"
        )

    def smart_truncate(input_ids, max_length):
        if len(input_ids) <= max_length:
            return input_ids
        head_len = int(max_length * 0.3)
        tail_len = max_length - head_len
        return input_ids[:head_len] + input_ids[-tail_len:]

    num_truncated = 0
    processed_samples = []
    truncation_cap_fraction = TRUNCATION_FRACTION

    for src_name, cfg in DATASET_CONFIG.items():
        need = quotas[src_name]
        got = 0
        print(
            f"{log_tag} collect: start source={src_name!r} "
            f"path={cfg['path']!r} target_accepted={need}"
        )
        stream = _load_streaming_for_cfg(cfg)
        it = iter(stream)
        max_stream_draws = max(need * 1000, 100_000)
        draws = 0
        while got < need:
            draws += 1
            if draws > max_stream_draws:
                if (
                    not always_truncate_over_max
                    and ALLOW_TRUNCATION
                    and truncation_cap_fraction < 1.0 - 1e-9
                ):
                    truncation_cap_fraction = min(
                        1.0, truncation_cap_fraction + TRUNCATION_FRACTION_STEP
                    )
                    draws = 0
                    print(
                        f"{log_tag} source={src_name!r}: no progress in {max_stream_draws} draws "
                        f"(got {got}/{need}); raising truncation cap to "
                        f"{truncation_cap_fraction:.2f} of NUM_CALIBRATION_SAMPLES "
                        f"(max truncations allowed ≈ {int(NUM_CALIBRATION_SAMPLES * truncation_cap_fraction)})."
                    )
                    continue
                print(
                    f"{log_tag} WARNING: source={src_name!r} stopped after {max_stream_draws} "
                    f"stream rows (collected {got}/{need})"
                    + (
                        f"; truncation cap already at {truncation_cap_fraction:.2f}. Raise max_stream_draws or relax "
                        f"MIN_SEQUENCE_LENGTH / MAX_SEQUENCE_LENGTH."
                        if not always_truncate_over_max
                        else ". Raise max_stream_draws or relax MIN_SEQUENCE_LENGTH."
                    )
                )
                break
            try:
                raw = next(it)
            except StopIteration:
                print(
                    f"{log_tag} WARNING: stream exhausted for source={src_name!r} "
                    f"at {got}/{need} samples."
                )
                break
            ex = dict(raw)
            sample = preprocess(ex)

            tokens = tokenizer(
                sample["text"],
                padding=False,
                truncation=False,
                add_special_tokens=False,
            )

            input_ids = tokens["input_ids"]
            length = len(input_ids)

            if length < MIN_SEQUENCE_LENGTH:
                continue

            if length > MAX_SEQUENCE_LENGTH:
                if always_truncate_over_max:
                    input_ids = smart_truncate(input_ids, MAX_SEQUENCE_LENGTH)
                    num_truncated += 1
                else:
                    if not ALLOW_TRUNCATION:
                        continue

                    if num_truncated > NUM_CALIBRATION_SAMPLES * truncation_cap_fraction:
                        continue

                    input_ids = smart_truncate(input_ids, MAX_SEQUENCE_LENGTH)
                    num_truncated += 1

            processed_samples.append(
                {
                    "_source": src_name,
                    "input_ids": input_ids,
                    "attention_mask": [1] * len(input_ids),
                }
            )
            got += 1
            if collection_progress_every > 0 and got % collection_progress_every == 0:
                print(
                    f"{log_tag} collect: source={src_name!r} "
                    f"accepted {got}/{need} (merged stream draw #{draws}; "
                    f"~{draws / got:.1f} draws/accept)"
                )

        print(
            f"{log_tag} collect: done source={src_name!r} "
            f"accepted {got}/{need}, total_merged_stream_draws={draws}"
            + (f", avg_draws_per_accept={draws / got:.1f}" if got else "")
        )

    if always_truncate_over_max:
        print(
            f"{log_tag} Long sequences always truncated to MAX_SEQUENCE_LENGTH={MAX_SEQUENCE_LENGTH}; "
            f"truncated rows: {num_truncated}."
        )
    else:
        print(
            f"{log_tag} Truncation cap fraction ended at {truncation_cap_fraction:.2f} "
            f"(initial {TRUNCATION_FRACTION}, step {TRUNCATION_FRACTION_STEP}); "
            f"truncated rows used: {num_truncated}."
        )

    rng = random.Random(42)
    rng.shuffle(processed_samples)

    source_counts = Counter(row["_source"] for row in processed_samples)
    accepted_source_order = [row["_source"] for row in processed_samples]

    n_acc = len(accepted_source_order)
    preview_n = min(80, n_acc)
    preview = accepted_source_order[:preview_n]
    runs = 0
    if n_acc > 0:
        runs = 1
        for i in range(1, n_acc):
            if accepted_source_order[i] != accepted_source_order[i - 1]:
                runs += 1
    avg_run = n_acc / runs if runs else 0.0

    print(f"{log_tag} Collected samples per source (after length/trunc filters):")
    for k in DATASET_CONFIG:
        print(f"  {k}: {source_counts[k]} (target was {quotas[k]})")
    if source_counts:
        unk = sum(c for s, c in source_counts.items() if s not in DATASET_CONFIG)
        if unk:
            print(f"  (unknown): {unk}")

    print(
        f"{log_tag} Order check after shuffle — first {preview_n} sources: {preview}"
    )
    print(
        f"{log_tag} Order check — run segments (source switches+1): {runs} "
        f"for {n_acc} samples; avg run length ≈ {avg_run:.2f} "
        f"(post-shuffle; ~random mix)"
    )
    expected_w = {k: DATASET_CONFIG[k]["weight"] for k in DATASET_CONFIG}
    if n_acc > 0:
        print(f"{log_tag} Empirical vs config weight (reference):")
        for k in DATASET_CONFIG:
            emp = source_counts[k] / n_acc
            print(f"  {k}: empirical={emp:.3f}, config_weight={expected_w[k]:.3f}")

    if log_per_domain_token_stats:
        tok_sum_by_source: dict[str, int] = defaultdict(int)
        for row in processed_samples:
            tok_sum_by_source[row["_source"]] += len(row["input_ids"])
        # 요청 출력 순서: chat, code, stem, math → 나머지 소스는 설정 키 순으로 뒤에 붙임
        preferred_order = ("chat", "code", "stem", "math")
        printed: set[str] = set()
        print(f"{log_tag} Token counts per domain (sum of len(input_ids) after shuffle; tokenizer padding=False):")
        grand = 0
        for dom in preferred_order:
            if dom not in DATASET_CONFIG:
                continue
            printed.add(dom)
            n_s = source_counts[dom]
            t_s = tok_sum_by_source[dom]
            grand += t_s
            mean_t = (t_s / n_s) if n_s else 0.0
            print(f"  {dom}: total_tokens={t_s}, samples={n_s}, mean_tokens_per_sample={mean_t:.1f}")
        for dom in DATASET_CONFIG:
            if dom in printed:
                continue
            n_s = source_counts[dom]
            t_s = tok_sum_by_source[dom]
            grand += t_s
            mean_t = (t_s / n_s) if n_s else 0.0
            print(f"  {dom}: total_tokens={t_s}, samples={n_s}, mean_tokens_per_sample={mean_t:.1f}")
        print(f"  (all domains): total_tokens={grand}, samples={n_acc}")

    if text_jsonl_path is not None:
        jp = Path(text_jsonl_path)
        jp.parent.mkdir(parents=True, exist_ok=True)
        with jp.open("w", encoding="utf-8") as jf:
            for i, row in enumerate(processed_samples):
                # 저장된 input_ids 와 동일한 바이트열을 복원 (트렁케이션 반영)
                line_text = tokenizer.decode(row["input_ids"], skip_special_tokens=False)
                jf.write(
                    json.dumps({"sample_id": i, "text": line_text}, ensure_ascii=False) + "\n"
                )
        print(
            f"{log_tag} Wrote text sidecar JSONL ({len(processed_samples)} lines): {jp.resolve()}"
        )

    for row in processed_samples:
        del row["_source"]

    return Dataset.from_list(processed_samples)


# --- 이 스크립트 전용: math / code 가중치·저장 경로 ---

_MATH_CODE_DATASET_CONFIG_TEMPLATE: dict[str, Any] = {
    "math": {
        "path": "nvidia/Nemotron-SFT-Math-v3",
        "weight": 0.5,
    },
    "code": {
        "path": "nvidia/Nemotron-SFT-Competitive-Programming-v2",
        "weight": 0.5,
        "splits": [
            "exercism",
            "text_to_sql",
            "competitive_coding_cpp",
            "competitive_coding_python",
        ],
    },
}


def build_math_code_dataset_config(math_weight: float, code_weight: float) -> dict[str, Any]:
    s = math_weight + code_weight
    if abs(s - 1.0) > 1e-5:
        raise ValueError(f"math_weight + code_weight must sum to 1.0, got {s}")
    cfg = {}
    for k, v in _MATH_CODE_DATASET_CONFIG_TEMPLATE.items():
        cfg[k] = {**v}
    cfg["math"]["weight"] = float(math_weight)
    cfg["code"]["weight"] = float(code_weight)
    return cfg


def build_nemotron_post_training_v1_dataset_config() -> dict[str, Any]:
    """Nemotron-Post-Training-Dataset-v1: chat / math / code / stem 각 동일 비중(쿼터는 num_samples·weight로 분배).

    `filter_reasoning_on`: 데이터셋 행의 최상위 `reasoning` 필드가 \"on\" 인 행만 스트림에 남긴다
    (chat 은 off 행 제외; math/code/stem 은 원본이 on 이라 사실상 전부 통과).
    """
    w = 0.25
    path = _NEMOTRON_POST_TRAINING_V1_REPO
    fr = {"filter_reasoning_on": True}
    return {
        "chat": {"path": path, "split": "chat", "weight": w, **fr},
        "math": {"path": path, "split": "math", "weight": w, **fr},
        "code": {"path": path, "split": "code", "weight": w, **fr},
        "stem": {"path": path, "split": "stem", "weight": w, **fr},
    }


def build_and_save_math_code_dataset(
    model_name: str,
    tokenizer,
    num_samples: int,
    min_length: int,
    max_length: int,
    save_path: str | os.PathLike[str],
    *,
    math_weight: float = 0.5,
    code_weight: float = 0.5,
    thinking_mode=None,
    always_truncate_over_max: bool = False,
    log_tag: str = "[math_code]",
    save_text_jsonl: bool = True,
    text_jsonl_name: str = "samples_text.jsonl",
) -> str:
    out = Path(save_path)
    out.mkdir(parents=True, exist_ok=True)
    text_sidecar = (out / text_jsonl_name) if save_text_jsonl else None
    dataset_config = build_math_code_dataset_config(math_weight, code_weight)
    ds = _prepare_math_code_mix_dataset(
        tokenizer,
        thinking_mode,
        min_length,
        max_length,
        num_samples,
        dataset_config=dataset_config,
        always_truncate_over_max=always_truncate_over_max,
        log_tag=log_tag,
        text_jsonl_path=text_sidecar,
        log_per_domain_token_stats=False,
    )
    ds.save_to_disk(str(out))
    return str(out.resolve())


def build_and_save_nemotron_post_training_v1_dataset(
    model_name: str,
    tokenizer,
    samples_per_domain: int,
    min_length: int,
    max_length: int,
    save_path: str | os.PathLike[str],
    *,
    thinking_mode=None,
    always_truncate_over_max: bool = False,
    log_tag: str = "[nemo_pt_v1]",
    save_text_jsonl: bool = True,
    text_jsonl_name: str = "samples_text.jsonl",
) -> str:
    if samples_per_domain < 1:
        raise ValueError("samples_per_domain must be >= 1")
    num_samples = samples_per_domain * 4
    out = Path(save_path)
    out.mkdir(parents=True, exist_ok=True)
    text_sidecar = (out / text_jsonl_name) if save_text_jsonl else None
    dataset_config = build_nemotron_post_training_v1_dataset_config()
    ds = _prepare_math_code_mix_dataset(
        tokenizer,
        thinking_mode,
        min_length,
        max_length,
        num_samples,
        dataset_config=dataset_config,
        always_truncate_over_max=always_truncate_over_max,
        log_tag=log_tag,
        text_jsonl_path=text_sidecar,
        log_per_domain_token_stats=True,
    )
    ds.save_to_disk(str(out))
    return str(out.resolve())


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="캘리브 데이터셋 생성 및 save_to_disk (custom: Nemotron SFT math+code / nemo_dataset: Post-Training v1)",
    )
    p.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="토크나이저 로드용 HF 모델 경로 또는 모델 ID",
    )
    p.add_argument(
        "--dataset-kind",
        type=str,
        choices=("custom", "nemo_dataset"),
        default="custom",
        help="custom=Nemotron SFT Math+Code, nemo_dataset=Nemotron-Post-Training-Dataset-v1 (chat/math/code/stem)",
    )
    p.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="수집할 총 샘플 수 (dataset-kind=custom 일 때 필수)",
    )
    p.add_argument(
        "--nemo-samples-per-domain",
        type=int,
        default=25,
        help="dataset-kind=nemo_dataset 일 때 chat·math·code·stem 각 도메인별 목표 샘플 수 (총 4배)",
    )
    p.add_argument("--min-length", type=int, required=True, help="토큰 길이 하한(미만 제외)")
    p.add_argument(
        "--max-length",
        type=int,
        required=True,
        help="토큰 길이 상한(초과 시 smart_truncate 정책은 내부 _prepare_math_code_mix_dataset 과 동일)",
    )
    p.add_argument(
        "--save-path",
        type=str,
        required=True,
        help="save_to_disk 대상 디렉터리 (없으면 생성)",
    )
    p.add_argument(
        "--math-weight",
        type=float,
        default=0.5,
        help="dataset-kind=custom: math 도메인 비율 (code 와 합이 1.0)",
    )
    p.add_argument(
        "--code-weight",
        type=float,
        default=0.5,
        help="dataset-kind=custom: code 도메인 비율 (math 와 합이 1.0)",
    )
    p.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="AutoTokenizer.from_pretrained 에 trust_remote_code=True 전달",
    )
    p.add_argument(
        "--always-truncate-over-max",
        action="store_true",
        help="MAX 초과 시 항상 smart_truncate",
    )
    p.add_argument(
        "--no-text-jsonl",
        action="store_true",
        help="sample_id/text JSONL(samples_text.jsonl) 저장 생략",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    from transformers import AutoTokenizer

    if args.dataset_kind == "custom" and args.num_samples is None:
        raise SystemExit("--num-samples is required when --dataset-kind=custom")
    if args.dataset_kind == "nemo_dataset" and args.nemo_samples_per_domain < 1:
        raise SystemExit("--nemo-samples-per-domain must be >= 1 for nemo_dataset")

    tok = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=args.trust_remote_code,
    )
    if args.dataset_kind == "custom":
        out = build_and_save_math_code_dataset(
            args.model_name,
            tok,
            args.num_samples,
            args.min_length,
            args.max_length,
            args.save_path,
            math_weight=args.math_weight,
            code_weight=args.code_weight,
            thinking_mode=None,
            always_truncate_over_max=args.always_truncate_over_max,
            save_text_jsonl=not args.no_text_jsonl,
        )
    else:
        out = build_and_save_nemotron_post_training_v1_dataset(
            args.model_name,
            tok,
            args.nemo_samples_per_domain,
            args.min_length,
            args.max_length,
            args.save_path,
            thinking_mode=None,
            always_truncate_over_max=args.always_truncate_over_max,
            save_text_jsonl=not args.no_text_jsonl,
        )
    print(f"Saved dataset to: {out}")


if __name__ == "__main__":
    main()
