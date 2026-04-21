"""quant_pipe 전용: Nemotron Post-Training Dataset v1 스트리밍 캘리브 수집 (형제 폴더 import 없음).

동작은 expert_balance/step1_dataset_load.py 의 nemo_dataset 경로와 동일하게 맞춘다
(chat/math/code/stem, 행 메타 reasoning == "on", 길이 필터·smart_truncate 정책 동일).
"""
from __future__ import annotations

import inspect
import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from datasets import Dataset, load_dataset

NEMOTRON_POST_TRAINING_V1_REPO = "nvidia/Nemotron-Post-Training-Dataset-v1"


def build_nemotron_post_training_v1_dataset_config() -> dict[str, Any]:
    """Nemotron-Post-Training-Dataset-v1: chat / math / code / stem 동일 비중, reasoning==on 만."""
    w = 0.25
    path = NEMOTRON_POST_TRAINING_V1_REPO
    fr = {"filter_reasoning_on": True}
    return {
        "chat": {"path": path, "split": "chat", "weight": w, **fr},
        "math": {"path": path, "split": "math", "weight": w, **fr},
        "code": {"path": path, "split": "code", "weight": w, **fr},
        "stem": {"path": path, "split": "stem", "weight": w, **fr},
    }


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


def prepare_nemotron_post_training_v1_calibration_dataset(
    tokenizer,
    thinking_mode,
    MIN_SEQUENCE_LENGTH,
    MAX_SEQUENCE_LENGTH,
    NUM_CALIBRATION_SAMPLES,
    *,
    dataset_config: dict[str, Any],
    always_truncate_over_max=False,
    log_tag="[nemotron_pt_v1]",
    text_jsonl_path: str | os.PathLike[str] | None = None,
    log_per_domain_token_stats: bool = False,
) -> Dataset:
    """`dataset_config`에 따른 스트리밍 수집·토큰화(길이 필터·선택적 smart_truncate).

    `text_jsonl_path`가 주어지면 셔플 후 각 줄을 JSONL로 저장한다 (`sample_id`, `text`, `domain`: chat|math|code|stem).
    `log_per_domain_token_stats`: True이면 `_source` 도메인별 토큰 통계를 stdout에 출력.
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
        """Post-Training v1 단일 split 스트리밍만 (quant_pipe 전용; multi-split / competitive JSONL 없음)."""
        repo = cfg["path"]
        config_name = cfg.get("config")
        if "splits" in cfg:
            raise ValueError(
                f"{log_tag} unsupported cfg: 'splits' interleave — use expert_balance/step1_dataset_load.py"
            )
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

            line_text = tokenizer.decode(input_ids, skip_special_tokens=False)
            processed_samples.append(
                {
                    "_source": src_name,
                    "input_ids": input_ids,
                    "attention_mask": [1] * len(input_ids),
                    "text": line_text,
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
                line_text = row["text"]
                dom = row["_source"]
                jf.write(
                    json.dumps(
                        {"sample_id": i, "text": line_text, "domain": dom},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        print(
            f"{log_tag} Wrote text sidecar JSONL ({len(processed_samples)} lines): {jp.resolve()}"
        )

    for row in processed_samples:
        row["domain"] = row.pop("_source")

    return Dataset.from_list(processed_samples)
