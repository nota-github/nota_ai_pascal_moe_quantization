"""
step5 가 저장한 token 영역 JSON(token_blue_region / token_red_region)을 사용해
samples_text.jsonl 원문에 <blue>...</blue>, <red>...</red> 태그를 삽입한다.

각 출력 줄: sample_id, mode, text, n_tokens_total, n_tokens_blue, n_tokens_red
(개수는 overlap 정책 적용 후 라벨 기준)
원본 samples_text.jsonl 에 domain 등 메타데이터 키가 있으면 출력에도 그대로 유지한다.

--bracket_after_input: 원문에 "<think>\\n" 이 있으면 그 문자열 직후 문자
오프셋 이후에 시작하는 토큰(offset[0] >= boundary)에만 태그를 붙인다(생성 구간만).
마커가 없으면 해당 샘플에는 태그를 붙이지 않는다.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from transformers import AutoTokenizer


def load_blue_red_by_sample(classified_path: Path, mode: str) -> tuple[dict[int, set[int]], dict[int, set[int]]]:
    with open(classified_path, encoding="utf-8") as f:
        d = json.load(f)
    if d.get("mode") != mode:
        raise SystemExit(
            f"classified JSON mode 불일치: 파일={d.get('mode')!r}, --mode={mode!r}",
        )
    blue_list = d.get("token_blue_region") or d.get("freq_token") or []
    red_list = d.get("token_red_region") or d.get("less_token") or []
    blue_by: dict[int, set[int]] = defaultdict(set)
    red_by: dict[int, set[int]] = defaultdict(set)
    for e in blue_list:
        blue_by[int(e["sample_id"])].add(int(e["position_id"]))
    for e in red_list:
        red_by[int(e["sample_id"])].add(int(e["position_id"]))
    return dict(blue_by), dict(red_by)


def per_position_labels(
    seq_len: int,
    blue_pos: set[int],
    red_pos: set[int],
    *,
    overlap: str,
) -> list[str | None]:
    out: list[str | None] = [None] * seq_len
    for p in range(seq_len):
        ib = p in blue_pos
        ir = p in red_pos
        if ib and ir:
            if overlap == "red_first":
                out[p] = "red"
            else:
                out[p] = "blue"
        elif ib:
            out[p] = "blue"
        elif ir:
            out[p] = "red"
    return out


def merged_spans(
    labels: list[str | None],
    offsets: list[tuple[int, int]],
) -> list[tuple[int, int, str]]:
    spans: list[tuple[int, int, str]] = []
    i = 0
    n = len(labels)
    while i < n:
        if labels[i] is None:
            i += 1
            continue
        lab = labels[i]
        j = i + 1
        while j < n and labels[j] == lab:
            j += 1
        start_ch = offsets[i][0]
        end_ch = offsets[j - 1][1]
        spans.append((start_ch, end_ch, lab))
        i = j
    return spans


BRACKET_AFTER_MARKER = "<think>\n"


def clear_labels_before_char(
    labels: list[str | None],
    offsets: list[tuple[int, int]],
    after_char: int,
) -> None:
    """after_char 이전에 시작하는 토큰은 라벨을 제거한다."""
    for p, lab in enumerate(labels):
        if lab is None:
            continue
        if offsets[p][0] < after_char:
            labels[p] = None


def count_labels(labels: list[str | None]) -> tuple[int, int]:
    """최종 라벨 기준 blue / red 토큰 수."""
    n_blue = sum(1 for x in labels if x == "blue")
    n_red = sum(1 for x in labels if x == "red")
    return n_blue, n_red


def apply_tags(original: str, spans: list[tuple[int, int, str]]) -> str:
    s = original
    for start, end, lab in sorted(spans, key=lambda x: x[0], reverse=True):
        inner = original[start:end]
        tagged = f"<{lab}>{inner}</{lab}>"
        s = s[:start] + tagged + s[end:]
    return s


def main() -> None:
    p = argparse.ArgumentParser(description="원문에 blue/red XML 스타일 태그 삽입")
    p.add_argument("--text-jsonl", type=str, required=True, help="step1 samples_text.jsonl")
    p.add_argument(
        "--classified-json",
        type=str,
        required=True,
        help="step5 가 --token-thr-* 와 함께 저장한 token_freq_less_scatter_*.json",
    )
    p.add_argument("--mode", type=str, choices=("balance", "q_sensitivity"), required=True)
    p.add_argument("--model-name", type=str, required=True)
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument(
        "--out-path",
        type=str,
        required=True,
        help="출력 .jsonl 파일 경로 또는 디렉터리(파일명 bracketed_<mode>.jsonl)",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="출력할 샘플 최대 개수(text-jsonl 순서대로 상위 N개만 처리)",
    )
    p.add_argument(
        "--overlap",
        type=str,
        choices=("blue_first", "red_first"),
        default="blue_first",
        help="같은 position_id 가 blue·red 둘 다일 때 우선 라벨",
    )
    p.add_argument("--tag-blue", type=str, default="blue", help="XML 태그 로컬 이름 (기본 blue)")
    p.add_argument("--tag-red", type=str, default="red", help="XML 태그 로컬 이름 (기본 red)")
    p.add_argument(
        "--one-json-out",
        type=str,
        default=None,
        help="첫 번째로 처리한 샘플 1개만 들여쓰기 JSON으로 추가 저장 (미지정 시 생략)",
    )
    p.add_argument(
        "--bracket_after_input",
        action="store_true",
        help=(
            '"<think>\\n" 직후부터 시작하는 토큰에만 '
            "<blue>/<red> 태그 적용. 마커가 없는 줄은 태그 없음."
        ),
    )
    args = p.parse_args()

    classified_path = Path(args.classified_json).resolve()
    text_path = Path(args.text_jsonl).resolve()
    if not classified_path.is_file():
        raise SystemExit(f"파일 없음: {classified_path}")
    if not text_path.is_file():
        raise SystemExit(f"파일 없음: {text_path}")

    blue_by, red_by = load_blue_red_by_sample(classified_path, args.mode)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=args.trust_remote_code,
    )

    out_arg = Path(args.out_path).expanduser().resolve()
    if out_arg.is_dir():
        out_path = out_arg / f"bracketed_{args.mode}.jsonl"
    else:
        out_path = out_arg
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tag_map = {"blue": args.tag_blue, "red": args.tag_red}

    one_json_path = Path(args.one_json_out).expanduser().resolve() if args.one_json_out else None
    if one_json_path is not None:
        one_json_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    one_saved = False
    with open(text_path, encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            if args.limit is not None and written >= args.limit:
                break
            row = json.loads(line)
            sid = int(row["sample_id"])
            text = row["text"]

            enc = tokenizer(
                text,
                return_offsets_mapping=True,
                add_special_tokens=False,
            )
            offsets = enc["offset_mapping"]
            seq_len = len(offsets)

            blue_p = blue_by.get(sid, set())
            red_p = red_by.get(sid, set())

            labels = per_position_labels(seq_len, blue_p, red_p, overlap=args.overlap)
            if args.bracket_after_input:
                m = text.find(BRACKET_AFTER_MARKER)
                if m == -1:
                    for p in range(seq_len):
                        labels[p] = None
                else:
                    after_char = m + len(BRACKET_AFTER_MARKER)
                    clear_labels_before_char(labels, offsets, after_char)
            n_blue, n_red = count_labels(labels)
            raw_spans = merged_spans(labels, offsets)
            spans = [(a, b, tag_map[lab]) for a, b, lab in raw_spans]
            tagged_text = apply_tags(text, spans)

            # 원본 줄의 domain 등 메타데이터 유지; 아래 키만 이 스크립트 결과로 덮어씀
            _computed = {
                "sample_id",
                "mode",
                "text",
                "n_tokens_total",
                "n_tokens_blue",
                "n_tokens_red",
            }
            payload = {k: v for k, v in row.items() if k not in _computed}
            payload.update(
                {
                    "sample_id": sid,
                    "mode": args.mode,
                    "n_tokens_total": seq_len,
                    "n_tokens_blue": n_blue,
                    "n_tokens_red": n_red,
                    "text": tagged_text,
                },
            )
            fout.write(json.dumps(payload, ensure_ascii=False) + "\n")

            if one_json_path is not None and not one_saved:
                with open(one_json_path, "w", encoding="utf-8") as fj:
                    json.dump(payload, fj, indent=2, ensure_ascii=False)
                    fj.write("\n")
                one_saved = True

            written += 1

    print(f"Wrote {written} lines -> {out_path}")
    if one_json_path is not None:
        if one_saved:
            print(f"Wrote 1 sample (first) -> {one_json_path}")
        else:
            print(f"No sample written; skipped --one-json-out {one_json_path}")


if __name__ == "__main__":
    main()
