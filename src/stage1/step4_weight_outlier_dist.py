"""
MoE expert별 weight quantization sensitivity (GPTQ INT4 blockwise, group_size=128) 추정.

- ModelOpt INT4_BLOCKWISE_WEIGHT_ONLY_CFG(step1_gptq_quantize.py)와 동일하게
  Linear weight [out, in] 의 마지막 차원(in)을 group_size(기본 128)로 나눈 각 블록에서
  max(|w|)/median(|w|) 를 구하고, 블록 전체에 대해 max → 한 텐서 점수.
- expert 내 gate/up/down(또는 Mixtral w1/w3/w2) 중 max → 보수적 단일 점수.

step3_count_expert_dist.py 와 동일한 막대 플롯(full / optional active) 및
--sen-thr / --rob-thr 시 상·하위 expert 집합 JSON 저장.
  --thr-basis active 는 JSON expert 선정을 active 막대와 동일하게(활성 슬롯만 민감도 오름차순); --jsonl 필수.
  - 정렬: 민감도 오름차순(왼쪽=낮음, 오른쪽=높음; step3 count 막대의 좌→우와 동일한 방향감)
  - 임계 색: step3 와 동일 — 왼쪽=파랑(강건/rob), 오른쪽=빨강(민감/sen)
  - 상위 sen_thr 비율 → q_sensitive_expert (가장 민감한 쪽, 오른쪽 끝)
  - 하위 rob_thr 비율 → q_robust_group (가장 강건한 쪽, 왼쪽 끝)

Inference 없이 HF 가중치만 로드하여 분석 (--device cpu 권장).

--sen-thr / --rob-thr 가 **없고** `--jsonl` 이 있으면, full grid 각 슬롯에 대해
q_sensitivity(y) vs activation count(x) 2D 산점도를 추가 저장한다 (step2 집계와 비교).

임계(sen/rob)가 **없을 때** `{stem}_distribution_summary.json` 에
민감도 막대(full, 및 jsonl 시 active)의 정렬 축 2.5% 샘플 곡선과,
jsonl 이 있으면 산점도용 주변분포·2D 히스토그램·상관을 함께 저장한다 (LLM/도구용).

매 실행마다 `{stem}_sensitivity_by_expert.json` 에 expert 슬롯별 sensitivity 를 저장한다.
`--jsonl` 을 주면 각 entry 에 activation_count 및 routing 메타를 함께 넣는다.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM

from step2_count_expert import find_sparse_moe_blocks
from step3_count_expert_dist import (
    add_percentile_vlines,
    distribution_section_rank_curve,
    display_model_name,
    parse_optional_thr,
    rank_count_for_threshold,
    scan_jsonl,
    write_json,
)

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica", "sans-serif"],
    }
)

# Match quant_pipe/step1_gptq_quantize.py + ModelOpt INT4_BLOCKWISE_WEIGHT_ONLY_CFG
DEFAULT_GROUP_SIZE = 128

# Threshold bar colors (same as step3): left=blue, right=red — robust=left, sensitive=right
_THR_BAR_GRAY = "#d5d5d5"
_THR_BAR_FREQ = "#90caf9"
_THR_BAR_LESS = "#ffcdd2"
_LINE_FREQ = "#0d47a1"


def sort_by_score_asc(
    keys: list[tuple[int, int]],
    scores: list[float],
) -> tuple[list[tuple[int, int]], list[float]]:
    """score 오름차순(작은 값=왼쪽). 동률이면 (layer, expert_id) 오름차순. NaN 은 맨 오른쪽."""

    def key_fn(t: tuple[tuple[int, int], float]) -> tuple[float, int, int]:
        (L, e), s = t
        if isinstance(s, float) and math.isnan(s):
            s_key = float("inf")
        else:
            s_key = float(s)
        return (s_key, L, e)

    pairs = list(zip(keys, scores, strict=True))
    pairs.sort(key=key_fn)
    if not pairs:
        return [], []
    k, s = zip(*pairs, strict=True)
    return list(k), list(s)


def bar_colors_sensitivity_asc(
    n: int,
    sen_thr: float | None,
    rob_thr: float | None,
) -> list[str]:
    """
    오름차순 막대(왼쪽=낮은 민감도, 오른쪽=높은 민감도)에 대해 step3 와 동일 — 왼쪽 파랑, 오른쪽 빨강:
    - rob_thr → 왼쪽 n_rob 막대 = 가장 강건(q_robust) 쪽 (파랑, step3 freq 쪽과 동일)
    - sen_thr → 오른쪽 n_sen 막대 = 가장 민감(q_sensitive) 쪽 (빨강, step3 less 쪽과 동일)
    """
    if n <= 0:
        return []
    n_sen = rank_count_for_threshold(sen_thr, n)
    n_rob = rank_count_for_threshold(rob_thr, n)
    use = (sen_thr is not None) or (rob_thr is not None)
    if not use:
        return ["steelblue"] * n
    colors: list[str] = []
    for i in range(n):
        in_sensitive = sen_thr is not None and i >= n - n_sen
        in_robust = rob_thr is not None and i < n_rob
        if in_sensitive and in_robust:
            colors.append(_THR_BAR_GRAY)
        elif in_sensitive:
            colors.append(_THR_BAR_LESS)
        elif in_robust:
            colors.append(_THR_BAR_FREQ)
        else:
            colors.append(_THR_BAR_GRAY)
    return colors


def resolve_out_dir_stem(
    out_path: str | None,
    *,
    default_dir: Path,
    default_stem: str,
) -> tuple[Path, str]:
    if not out_path:
        return default_dir, default_stem
    p = Path(out_path).expanduser().resolve()
    if p.is_dir():
        return p, default_stem
    if p.is_file():
        raise SystemExit(f"--out-path must not be an existing file: {p}")
    out_dir = p.parent
    stem = p.name
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir, stem


def expert_fc_weight_tensors(expert: torch.nn.Module) -> list[torch.Tensor]:
    """
    Blockwise GPTQ 가정: weight 2D, 마지막 차원 = input features.
    Qwen2/3 MoE: gate_proj, up_proj, down_proj — Mixtral: w1, w3, w2 (SwiGLU).
    """
    if hasattr(expert, "gate_proj") and hasattr(expert, "up_proj") and hasattr(expert, "down_proj"):
        return [
            expert.gate_proj.weight.detach(),
            expert.up_proj.weight.detach(),
            expert.down_proj.weight.detach(),
        ]
    if hasattr(expert, "w1") and hasattr(expert, "w2") and hasattr(expert, "w3"):
        return [
            expert.w1.weight.detach(),
            expert.w3.weight.detach(),
            expert.w2.weight.detach(),
        ]
    raise ValueError(
        f"Unsupported expert MLP (expected gate/up/down or w1/w2/w3): {type(expert).__name__}"
    )


def blockwise_max_median_ratio(w: torch.Tensor, group_size: int) -> float:
    """
    ModelOpt export 의 get_scaling_factor_from_weight 와 동일하게
    weight 를 [n, k//g, g] 로 보고, 각 블록에서 max(|.|)/median(|.|) 후 전체 max.
    k 가 group_size 로 나누어떨어지지 않으면 앞쪽 usable 열만 사용.
    """
    if w.dim() != 2:
        raise ValueError(f"Expected 2D weight, got shape {tuple(w.shape)}")
    n, k = w.shape
    if k < group_size:
        return float("nan")
    usable = (k // group_size) * group_size
    if usable == 0:
        return float("nan")
    wf = w[:, :usable].float().abs()
    g = wf.reshape(n, usable // group_size, group_size)
    med = torch.median(g, dim=-1).values
    mx = g.max(dim=-1).values
    ratio = mx / med.clamp(min=1e-20)
    return float(ratio.max().cpu())


def expert_sensitivity_score(
    expert: torch.nn.Module,
    *,
    group_size: int,
) -> float:
    """한 expert: 세 FC 중 blockwise ratio max (보수적)."""
    scores: list[float] = []
    for w in expert_fc_weight_tensors(expert):
        scores.append(blockwise_max_median_ratio(w, group_size))
    valid = [s for s in scores if not math.isnan(s)]
    if not valid:
        return float("nan")
    return max(valid)


def build_rank_xticks_scores(
    n: int,
    scores: list[float],
) -> tuple[list[float], list[str]]:
    if n <= 1:
        return [], []
    entries: list[tuple[float, str]] = []
    for p in range(5, 100, 5):
        x_f = (p / 100.0) * (n - 1)
        idx = int(round(x_f))
        idx = max(0, min(n - 1, idx))
        s = scores[idx]
        entries.append((x_f, f"{p}%\nidx={idx}\ns={s:.4g}"))
    by_x: dict[float, list[str]] = {}
    for x_f, lab in entries:
        xr = round(float(x_f), 4)
        by_x.setdefault(xr, []).append(lab)
    positions = sorted(by_x.keys())
    labels = ["\n".join(by_x[x]) for x in positions]
    return positions, labels


def plot_sensitivity_bars(
    keys: list[tuple[int, int]],
    scores: list[float],
    *,
    title: str,
    ylabel: str,
    out_path: Path,
    xlabel: str,
    sen_thr: float | None = None,
    rob_thr: float | None = None,
) -> None:
    # 오름차순: 왼쪽=낮은 민감도(step3 의 count 플롯에서 오른쪽=저빈도 와 같은 축 방향)
    keys, scores = sort_by_score_asc(keys, scores)

    y = np.asarray(scores, dtype=np.float64)
    n = len(keys)
    x = np.arange(n)

    use_xticks = n > 1
    fig_h = 6.8 if use_xticks else 5.0
    fig, ax = plt.subplots(figsize=(14, fig_h))
    bar_colors = bar_colors_sensitivity_asc(n, sen_thr, rob_thr)
    ax.bar(x, y, width=1.0, color=bar_colors, edgecolor="none", linewidth=0)
    add_percentile_vlines(ax, n)

    n_sen = rank_count_for_threshold(sen_thr, n)
    n_rob = rank_count_for_threshold(rob_thr, n)
    # 민감 상위 sen_thr: 오른쪽 n_sen 막대(빨강 구간) — 경계선은 그 구간 왼쪽 (step3 less 경계와 동일 색)
    if sen_thr is not None and n > 0 and 0 < n_sen < n:
        x_sensitive = float(n - n_sen) - 0.5
        ax.axvline(
            x_sensitive,
            color="#c62828",
            linestyle="-",
            linewidth=1.4,
            alpha=0.9,
            zorder=5,
        )
        _, ymax = ax.get_ylim()
        ax.text(
            x_sensitive,
            ymax,
            f"← top {sen_thr:.0%} sensitive",
            ha="left",
            va="bottom",
            fontsize=7,
            color="#c62828",
            clip_on=False,
        )
    # 강건 하위 rob_thr: 왼쪽 n_rob 막대(파랑 구간) — 경계선은 그 구간 오른쪽 (step3 freq 경계와 동일 색)
    if rob_thr is not None and n > 0 and 0 < n_rob < n:
        x_robust = float(n_rob) - 0.5
        ax.axvline(
            x_robust,
            color=_LINE_FREQ,
            linestyle="-",
            linewidth=1.4,
            alpha=0.95,
            zorder=5,
        )
        _, ymax = ax.get_ylim()
        ax.text(
            x_robust,
            ymax,
            f"bottom {rob_thr:.0%} robust →",
            ha="right",
            va="bottom",
            fontsize=7,
            color=_LINE_FREQ,
            clip_on=False,
        )

    if use_xticks:
        xt_pos, xt_lab = build_rank_xticks_scores(n, scores)
        if xt_pos:
            ax.set_xticks(xt_pos)
            ax.set_xticklabels(
                xt_lab,
                fontsize=5.5,
                rotation=90,
                ha="center",
                va="top",
            )
            ax.tick_params(axis="x", which="major", length=3, width=0.6)
            ax.minorticks_off()
        else:
            ax.set_xticks([])
            ax.tick_params(axis="x", which="both", length=0)
    else:
        ax.set_xticks([])
        ax.tick_params(axis="x", which="both", length=0)

    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))

    fig.tight_layout()
    if use_xticks:
        fig.subplots_adjust(bottom=0.34)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_sensitivity_vs_activation_scatter(
    scores: list[float],
    activation_counts: list[int],
    *,
    title: str,
    out_path: Path,
    x_mode: str,
) -> None:
    """
    x: activation count (step2 token_routing 집계), y: quantization sensitivity.
    x_mode: 'log1p' | 'linear'
    """
    x = np.asarray(activation_counts, dtype=np.float64)
    y = np.asarray(scores, dtype=np.float64)
    mask = np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size == 0:
        print("[warn] sensitivity vs activation scatter skipped: no finite y values")
        return

    fig, ax = plt.subplots(figsize=(9.5, 6.8))
    if x_mode == "log1p":
        xs = np.log1p(np.maximum(x, 0.0))
        ax.set_xlabel("log(1 + activation count)")
    elif x_mode == "linear":
        xs = x
        ax.set_xlabel("activation count")
    else:
        raise ValueError(f"unknown x_mode: {x_mode}")

    ax.scatter(
        xs,
        y,
        s=16,
        alpha=0.38,
        c="steelblue",
        edgecolors="none",
        rasterized=True,
    )
    ax.set_ylabel("quantization sensitivity (max|w|/median|w|, blockwise)")
    ax.set_title(title)
    ax.grid(True, alpha=0.28)
    ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def scatter_activation_sensitivity_summary_dict(
    scores: list[float],
    activation_counts: list[int],
    *,
    x_mode: str,
    grid_bins: int = 16,
) -> dict[str, object]:
    """
    산점도와 동일한 점들에 대해 주변분포(2.5% 간격), 2D 히스토그램, Pearson/Spearman 상관을 요약한다.
    """
    x = np.asarray(activation_counts, dtype=np.float64)
    y = np.asarray(scores, dtype=np.float64)
    mask = np.isfinite(y)
    x = x[mask]
    y = y[mask]
    out: dict[str, object] = {
        "schema": "expert_balance.distribution_summary.scatter.v1",
        "x_mode": x_mode,
        "n_points": int(x.size),
    }
    if x.size == 0:
        out["note"] = "no finite sensitivity values"
        return out

    if x_mode == "log1p":
        xs = np.log1p(np.maximum(x, 0.0))
        out["x_axis_for_analysis"] = "log1p(activation_count)"
    elif x_mode == "linear":
        xs = x
        out["x_axis_for_analysis"] = "activation_count"
    else:
        raise ValueError(f"unknown x_mode: {x_mode}")

    qs = np.arange(0, 100.25, 2.5)

    def pct_dict(arr: np.ndarray) -> dict[str, float]:
        return {f"p{p:g}": float(np.percentile(arr, p)) for p in qs}

    out["marginal_x"] = pct_dict(xs)
    out["marginal_y"] = pct_dict(y)
    pearson: float | None
    spearman: float | None
    if xs.size > 1:
        c = np.corrcoef(xs, y)[0, 1]
        pearson = float(c) if np.isfinite(c) else None
        rx = np.argsort(np.argsort(xs))
        ry = np.argsort(np.argsort(y))
        sp = np.corrcoef(rx.astype(np.float64), ry.astype(np.float64))[0, 1]
        spearman = float(sp) if np.isfinite(sp) else None
    else:
        pearson = None
        spearman = None
    out["correlation"] = {"pearson": pearson, "spearman": spearman}

    H, xe, ye = np.histogram2d(xs, y, bins=grid_bins)
    out["histogram2d"] = {
        "n_bins_each_axis": grid_bins,
        "x_bin_edges": xe.tolist(),
        "y_bin_edges": ye.tolist(),
        "counts": H.tolist(),
    }
    return out


def _json_float_or_null(x: float) -> float | None:
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return None
    return float(x)


def save_sensitivity_by_expert_json(
    path: Path,
    *,
    model_name: str,
    group_size: int,
    moe_layer_indices: list[int],
    keys: list[tuple[int, int]],
    scores: list[float],
    activation_counts: list[int] | None = None,
    routing_tokens: int | None = None,
    routing_jsonl: str | None = None,
) -> None:
    """expert 슬롯별 sensitivity (및 선택적 activation_count) 를 JSON 으로 저장."""
    entries: list[dict] = []
    for i, ((L, e), s) in enumerate(zip(keys, scores, strict=True)):
        row: dict = {
            "layer": int(L),
            "expert_id": int(e),
            "sensitivity": _json_float_or_null(float(s)),
        }
        if activation_counts is not None:
            row["activation_count"] = int(activation_counts[i])
        entries.append(row)

    payload: dict = {
        "model_name": model_name,
        "group_size": group_size,
        "metric": "per_fc_max_over_blocks_of_max_abs_over_median_abs_along_input_groups",
        "moe_layer_indices": moe_layer_indices,
        "n_slots": len(keys),
        "entries": entries,
    }
    if routing_tokens is not None:
        payload["routing_total_tokens"] = routing_tokens
    if routing_jsonl is not None:
        payload["routing_jsonl"] = routing_jsonl

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def save_quant_threshold_json(
    path: Path,
    *,
    sen_thr: float | None,
    rob_thr: float | None,
    sorted_keys: list[tuple[int, int]],
    group_size: int,
    ranking_basis: str,
) -> None:
    """sorted_keys 는 민감도 오름차순(왼쪽이 가장 낮음, 오른쪽이 가장 높음)."""
    n = len(sorted_keys)
    n_sen = rank_count_for_threshold(sen_thr, n)
    n_rob = rank_count_for_threshold(rob_thr, n)
    sens_list = sorted_keys[-n_sen:] if n_sen else []
    robust_list = sorted_keys[:n_rob] if n_rob else []
    payload = {
        "group_size": group_size,
        "metric": "per_fc_max_over_blocks_of_max_abs_over_median_abs_along_input_groups",
        "ranking_basis": ranking_basis,
        "sen_thr": sen_thr,
        "rob_thr": rob_thr,
        "thr_y": sen_thr,
        "thr_x": rob_thr,
        "n_ranked": n,
        "n_sen": n_sen,
        "n_rob": n_rob,
        "n_y": n_sen,
        "n_x": n_rob,
        "q_sensitive_expert": [{"layer": L, "expert_id": e} for L, e in sens_list],
        "q_robust_group": [{"layer": L, "expert_id": e} for L, e in robust_list],
        "experts_y": [{"layer": L, "expert_id": e} for L, e in sens_list],
        "experts_x": [{"layer": L, "expert_id": e} for L, e in robust_list],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def collect_sensitivity_grid(
    model: torch.nn.Module,
    moe_layers: list[tuple[int, torch.nn.Module]],
    *,
    group_size: int,
) -> tuple[list[tuple[int, int]], list[float]]:
    keys: list[tuple[int, int]] = []
    vals: list[float] = []
    for layer_idx, moe in tqdm(moe_layers, desc="MoE layers"):
        experts = moe.experts
        n_exp = len(experts)
        for eid in range(n_exp):
            s = expert_sensitivity_score(experts[eid], group_size=group_size)
            keys.append((layer_idx, eid))
            vals.append(s)
    return keys, vals


def main() -> None:
    p = argparse.ArgumentParser(
        description="MoE expert weight blockwise outlier sensitivity → sorted bar plots (+ optional q groups)"
    )
    p.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="HuggingFace 모델 경로 (MoE)",
    )
    p.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="from_pretrained / AutoConfig",
    )
    p.add_argument(
        "--device",
        type=str,
        default="cpu",
        help='모델 로드 device_map (기본 "cpu", VRAM 절약)',
    )
    p.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="가중치 로드 dtype (점수 계산은 FC weight 를 float 로 변환)",
    )
    p.add_argument(
        "--group-size",
        type=int,
        default=DEFAULT_GROUP_SIZE,
        help="ModelOpt INT4 blockwise 와 동일한 그룹 크기 (기본 128)",
    )
    p.add_argument(
        "--out-path",
        type=str,
        default=None,
        help="출력 디렉터리 또는 파일 stem (step3 와 유사). 생략 시 CWD + expert_weight_sensitivity",
    )
    p.add_argument(
        "--sen-thr",
        type=str,
        default=None,
        dest="sen_thr",
        help="0~1. 민감도 상위 비율(오른쪽 끝) → q_sensitive_expert",
    )
    p.add_argument(
        "--rob-thr",
        type=str,
        default=None,
        dest="rob_thr",
        help="0~1. 민감도 하위 비율(왼쪽 끝) → q_robust_group",
    )
    p.add_argument(
        "--thr-basis",
        type=str,
        choices=["full", "active"],
        default="full",
        help="임계 expert 집합(JSON)의 rank 기준: full(전체 슬롯) 또는 active(--jsonl 로 count>0 만, active 막대와 동일). active 시 --jsonl 필수",
    )
    p.add_argument(
        "--jsonl",
        type=str,
        default=None,
        help="step2 token_routing.jsonl 경로. 지정 시 count>0 인 (layer,expert) 만 active 플롯; "
        "임계 미지정 시 sensitivity vs activation 2D 산점도에도 사용",
    )
    p.add_argument(
        "--scatter-x",
        type=str,
        default="log1p",
        choices=["log1p", "linear"],
        help="산점도 x축: log1+count (기본) 또는 raw count",
    )
    p.add_argument(
        "--dump-scores-json",
        type=str,
        default=None,
        help="기본 *_sensitivity_by_expert.json 외에 동일 내용을 추가 저장할 경로 (선택)",
    )
    args = p.parse_args()

    sen_thr = parse_optional_thr("--sen-thr", args.sen_thr)
    rob_thr = parse_optional_thr("--rob-thr", args.rob_thr)
    use_threshold = sen_thr is not None or rob_thr is not None
    if use_threshold and args.thr_basis == "active" and not args.jsonl:
        raise SystemExit("--thr-basis active requires --jsonl (routing 집계로 활성 슬롯을 정의)")

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map[args.dtype]

    out_dir, file_stem = resolve_out_dir_stem(
        args.out_path,
        default_dir=Path.cwd().resolve(),
        default_stem="expert_weight_sensitivity",
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_stem = f"{file_stem}_threshold" if use_threshold else file_stem

    print("[model] loading (weights only analysis)...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch_dtype,
        device_map=args.device,
        trust_remote_code=args.trust_remote_code,
    )
    model.eval()

    moe_layers = find_sparse_moe_blocks(model)
    layer_indices = [L for L, _ in moe_layers]
    print(f"[MoE] {len(moe_layers)} sparse layer(s): {layer_indices}")

    group_size = int(args.group_size)
    if group_size < 1:
        raise SystemExit("--group-size must be >= 1")

    keys, scores = collect_sensitivity_grid(model, moe_layers, group_size=group_size)

    cfg_ne: int | None = None
    try:
        c = AutoConfig.from_pretrained(args.model_name, trust_remote_code=args.trust_remote_code)
        ne = getattr(c, "num_experts", None)
        if ne is not None:
            cfg_ne = int(ne)
    except Exception:
        pass

    if cfg_ne is not None and len(moe_layers) > 0:
        n0 = len(moe_layers[0][1].experts)
        if cfg_ne != n0:
            print(f"[warn] config num_experts={cfg_ne} vs first MoE experts={n0}")

    model_disp = display_model_name(args.model_name)
    total_slots = len(keys)
    title_full = (
        f"{model_disp} | GPTQ-style block max|w|/median|w| (last-dim groups={group_size}) | "
        f"experts(layer×per_expert)={total_slots} (MoE layers={len(layer_indices)})"
    )

    score_by_key = dict(zip(keys, scores, strict=True))
    activation_counts: list[int] | None = None
    routing_tokens: int | None = None
    routing_jsonl_str: str | None = None
    active_keys: list[tuple[int, int]] = []
    active_scores: list[float] = []

    if args.jsonl:
        jsonl_path = Path(args.jsonl).resolve()
        if not jsonl_path.is_file():
            raise SystemExit(f"--jsonl file not found: {jsonl_path}")
        ctr, total_tokens, _max_eid_seen = scan_jsonl(jsonl_path)
        routing_tokens = total_tokens
        routing_jsonl_str = str(jsonl_path)
        activation_counts = [int(ctr.get(k, 0)) for k in keys]
        active_keys = [k for k in keys if ctr.get(k, 0) > 0]
        active_scores = [score_by_key[k] for k in active_keys]

    thr_kw: dict = {}
    if use_threshold:
        if args.thr_basis == "active":
            if not active_keys:
                raise SystemExit(
                    "--thr-basis active requires at least one activated expert slot (count>0) in --jsonl"
                )
            sorted_k, _ = sort_by_score_asc(active_keys, active_scores)
            ranking_basis = "active_slots_sensitivity_asc"
        else:
            sorted_k, _ = sort_by_score_asc(keys, scores)
            ranking_basis = "full_grid_sensitivity_asc"
        json_path = out_dir / f"{plot_stem}_experts.json"
        save_quant_threshold_json(
            json_path,
            sen_thr=sen_thr,
            rob_thr=rob_thr,
            sorted_keys=sorted_k,
            group_size=group_size,
            ranking_basis=ranking_basis,
        )
        thr_kw["sen_thr"] = sen_thr
        thr_kw["rob_thr"] = rob_thr

    plot_sensitivity_bars(
        keys,
        scores,
        title=title_full,
        ylabel="sensitivity (max over FCs of max over blocks: max|w|/median|w|)",
        out_path=out_dir / f"{plot_stem}_full.png",
        xlabel="sorted rank (xticks: percentile — idx & score)",
        **thr_kw,
    )
    print(f"Wrote {out_dir / (plot_stem + '_full.png')}")

    if args.jsonl:
        assert routing_tokens is not None
        if not use_threshold:
            scatter_path = out_dir / f"{plot_stem}_scatter_sensitivity_activation.png"
            plot_sensitivity_vs_activation_scatter(
                scores,
                activation_counts,
                title=(
                    f"{model_disp} | tokens={routing_tokens} | full grid N={len(keys)} | "
                    "y=sensitivity, x=activation count (same slots)"
                ),
                out_path=scatter_path,
                x_mode=args.scatter_x,
            )
            print(f"Wrote {scatter_path}")

        title_active = (
            f"{model_disp} | tokens={routing_tokens} | active expert slots = {len(active_keys)} / {total_slots}"
        )
        plot_sensitivity_bars(
            active_keys,
            active_scores,
            title=title_active,
            ylabel="sensitivity (same metric)",
            out_path=out_dir / f"{plot_stem}_active.png",
            xlabel="sorted rank (active-only — idx & score)",
            **thr_kw,
        )
        print(f"Wrote {out_dir / (plot_stem + '_active.png')}")

    if use_threshold:
        print(f"Wrote {out_dir / (plot_stem + '_experts.json')}")

    if not use_threshold:
        _, scores_full_sorted = sort_by_score_asc(keys, scores)
        dist_payload: dict[str, object] = {
            "schema": "expert_balance.distribution_summary.step4.v1",
            "script": "step4_weight_outlier_dist.py",
            "model_display": model_disp,
            "model_name": args.model_name,
            "group_size": group_size,
            "moe_layer_indices": layer_indices,
            "routing_jsonl": routing_jsonl_str,
            "routing_total_tokens": routing_tokens,
            "full_grid": distribution_section_rank_curve(
                panel="full_grid",
                sort_description="sensitivity ascending (left=low/robust, right=high/sensitive)",
                x_axis="sorted_rank_percentile (0=left ... 100=right, matches bar plot)",
                y_axis="quantization_sensitivity",
                y_plot_order=scores_full_sorted,
                value_key="sensitivity_at_rank",
            ),
        }
        if args.jsonl and active_scores:
            _, active_s_sorted = sort_by_score_asc(active_keys, active_scores)
            dist_payload["active_only"] = distribution_section_rank_curve(
                panel="active_only",
                sort_description="sensitivity ascending over count>0 slots only",
                x_axis="sorted_rank_percentile (0=left ... 100=right, matches bar plot)",
                y_axis="quantization_sensitivity",
                y_plot_order=active_s_sorted,
                value_key="sensitivity_at_rank",
            )
        if args.jsonl and activation_counts is not None:
            dist_payload["scatter_activation_vs_sensitivity"] = scatter_activation_sensitivity_summary_dict(
                scores,
                activation_counts,
                x_mode=args.scatter_x,
            )
        dist_path = out_dir / f"{plot_stem}_distribution_summary.json"
        write_json(dist_path, dist_payload)
        print(f"Wrote {dist_path}")

    scores_json_path = out_dir / f"{plot_stem}_sensitivity_by_expert.json"
    save_sensitivity_by_expert_json(
        scores_json_path,
        model_name=args.model_name,
        group_size=group_size,
        moe_layer_indices=layer_indices,
        keys=keys,
        scores=scores,
        activation_counts=activation_counts,
        routing_tokens=routing_tokens,
        routing_jsonl=routing_jsonl_str,
    )
    print(f"Wrote {scores_json_path}")

    if args.dump_scores_json:
        extra_path = Path(args.dump_scores_json).expanduser().resolve()
        save_sensitivity_by_expert_json(
            extra_path,
            model_name=args.model_name,
            group_size=group_size,
            moe_layer_indices=layer_indices,
            keys=keys,
            scores=scores,
            activation_counts=activation_counts,
            routing_tokens=routing_tokens,
            routing_jsonl=routing_jsonl_str,
        )
        print(f"Wrote {extra_path}")


if __name__ == "__main__":
    main()
