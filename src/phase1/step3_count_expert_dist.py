"""
step2 가 기록한 token_routing.jsonl 을 읽어 (layer, expert_id) 별 활성화 횟수를 집계하고,
count 막대 그래프를 2종(full / active-only) 저장한다.

막대는 count 기준 내림차순(왼쪽=고빈도, 오른쪽=저빈도).
full / active 공통: 5% 간격 세로 점선 + 동일 규칙의 percentile xtick(idx, count).
full 만 추가: 첫 count=0 경계(주황 실선, 상단 문구, c=0 xtick).

--freq-thr / --less-thr (0~1) 지정 시: 기본은 full 그리드 정렬 기준으로 상·하위 비율에 해당하는 expert 집합을 JSON 으로 저장한다.
--thr-basis active 로 지정하면 그림의 active 막대와 같이 count>0 슬롯만 정렬해 집합을 만든다.
각 플롯에 해당 rank 경계 세로선을 추가한다(미지정 시 그림만, 집합·경계선 없음).

출력은 --out-path stem(확장자 없음)에 _full.png / _active.png.
--freq-thr / --less-thr 중 하나라도 주면 stem 에 ``_threshold`` 가 붙고,
(옵션) ``{stem}_threshold_experts.json`` 및 플롯 제목에 threshold 안내가 추가된다.
막대 색: step4 와 동일하게 양끝(상·하위 rank 구간)은 색, 가운데 구간은 회색.

freq/less 임계가 **없을 때**: ``{stem}_distribution_summary.json`` 에 full/active 각각에 대해
정렬 축(막대 순서) 기준 2.5% 간격으로 y(activation count)를 샘플링한 곡선 + 요약 통계를 저장한다 (LLM/도구용).
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ASCII-only labels in saved figures (no CJK fonts in plot output)
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica", "sans-serif"],
    }
)


def resolve_num_experts_per_layer(
    model_name: str | None,
    *,
    trust_remote_code: bool,
    fallback_max_id_plus_one: int,
) -> int:
    if not model_name:
        return fallback_max_id_plus_one
    try:
        from transformers import AutoConfig

        cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        n = getattr(cfg, "num_experts", None)
        if n is not None and int(n) > 0:
            return int(n)
    except Exception:
        pass
    return fallback_max_id_plus_one


def scan_jsonl(
    jsonl_path: Path,
) -> tuple[Counter[tuple[int, int]], int, int]:
    """
    Returns:
        counter (layer, expert_id) -> activation count
        total_tokens (jsonl lines)
        max_expert_id_seen (global max eid, for fallback num_experts)
    """
    ctr: Counter[tuple[int, int]] = Counter()
    total_tokens = 0
    max_eid = 0
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total_tokens += 1
            row = json.loads(line)
            for layer_str, eids in row.get("activated_expert_ids", {}).items():
                L = int(layer_str)
                for eid in eids:
                    eid = int(eid)
                    max_eid = max(max_eid, eid)
                    ctr[(L, eid)] += 1
    return ctr, total_tokens, max_eid


def ordered_layers_from_counter(ctr: Counter[tuple[int, int]]) -> list[int]:
    return sorted({L for (L, _) in ctr.keys()})


def build_full_grid(
    layers: list[int],
    num_experts: int,
) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for L in layers:
        for e in range(num_experts):
            out.append((L, e))
    return out


def add_percentile_vlines(ax: plt.Axes, n_x: int, *, color: str = "0.5", alpha: float = 0.65) -> None:
    """x 축을 0..n_x-1 로 둘 때 5%, 10%, ..., 95% 위치에 세로 점선."""
    if n_x <= 1:
        return
    for p in range(5, 100, 5):
        x = (p / 100.0) * (n_x - 1)
        ax.axvline(x, color=color, linestyle="--", linewidth=0.9, alpha=alpha, zorder=0)


def build_rank_xticks(
    n: int,
    counts: list[int],
    first_zero: int | None,
    *,
    include_c0_boundary_tick: bool,
) -> tuple[list[float], list[str]]:
    """
    5%~95% 세로선 x좌표마다 xtick: 정렬 인덱스(0-based), 해당 막대 count.
    include_c0_boundary_tick 이 True 이고 첫 count=0 가 있으면 그 경계 tick 추가(full 전용).
    """
    if n <= 1:
        return [], []

    entries: list[tuple[float, str]] = []
    for p in range(5, 100, 5):
        x_f = (p / 100.0) * (n - 1)
        idx = int(round(x_f))
        idx = max(0, min(n - 1, idx))
        c = counts[idx]
        entries.append((x_f, f"{p}%\nidx={idx}\nc={c:,}"))

    if include_c0_boundary_tick and first_zero is not None and first_zero > 0:
        x_line = float(first_zero) - 0.5
        entries.append((x_line, f"c=0\nidx={first_zero}\nc=0"))

    by_x: dict[float, list[str]] = {}
    for x_f, lab in entries:
        xr = round(float(x_f), 4)
        by_x.setdefault(xr, []).append(lab)
    positions = sorted(by_x.keys())
    labels = ["\n".join(by_x[x]) for x in positions]
    return positions, labels


def sort_by_count_desc(
    keys: list[tuple[int, int]],
    counts: list[int],
) -> tuple[list[tuple[int, int]], list[int]]:
    """count 내림차순, 동률이면 (layer, expert_id) 오름차순."""
    pairs = list(zip(keys, counts, strict=True))
    pairs.sort(key=lambda t: (-t[1], t[0][0], t[0][1]))
    if not pairs:
        return [], []
    k, c = zip(*pairs, strict=True)
    return list(k), list(c)


def rank_count_for_threshold(thr: float | None, n: int) -> int:
    """상·하위 thr 비율에 해당하는 막대 개수(정수, 0..n). thr None 이면 0."""
    if thr is None or n <= 0:
        return 0
    if thr <= 0:
        return 0
    k = int(math.ceil(thr * n))
    return min(n, max(0, k))


def rank_index_at_sorted_rank_percentile(n: int, percentile: float) -> int:
    """막대 플롯 x축과 동일: percentile in [0,100] → 인덱스 0..n-1 (round)."""
    if n <= 0:
        raise ValueError("n must be positive")
    if n == 1:
        return 0
    p = max(0.0, min(100.0, float(percentile)))
    return max(0, min(n - 1, int(round((p / 100.0) * (n - 1)))))


def sorted_rank_percentile_curve(
    y_plot_order: Sequence[float | int],
    *,
    percentile_step: float = 2.5,
    value_key: str = "y_at_rank",
) -> list[dict[str, float | int]]:
    """
    y_plot_order: 플롯과 동일한 순서(왼쪽→오른쪽 막대)의 y 값.
    0, step, ..., 100 percentile 위치에서 rank_index 와 y 를 기록한다.
    """
    n = len(y_plot_order)
    if n == 0:
        return []
    rows: list[dict[str, float | int]] = []
    p = 0.0
    while p <= 100.0 + 1e-9:
        idx = rank_index_at_sorted_rank_percentile(n, p)
        raw = y_plot_order[idx]
        if isinstance(raw, (int, np.integer)):
            v_json: float | int | None = int(raw)
        else:
            v = float(raw)
            v_json = None if (math.isnan(v) or math.isinf(v)) else v
        rows.append(
            {
                "percentile_along_sorted_rank": round(float(p), 4),
                "rank_index": int(idx),
                value_key: v_json,
            }
        )
        p += percentile_step
    return rows


def stats_1d_numeric(y: np.ndarray) -> dict[str, float | int]:
    y = np.asarray(y, dtype=np.float64)
    m = np.isfinite(y)
    y = y[m]
    if y.size == 0:
        return {"n_finite": 0}
    return {
        "n_finite": int(y.size),
        "min": float(y.min()),
        "max": float(y.max()),
        "mean": float(y.mean()),
    }


def distribution_section_rank_curve(
    *,
    panel: str,
    sort_description: str,
    x_axis: str,
    y_axis: str,
    y_plot_order: Sequence[float | int],
    percentile_step: float = 2.5,
    value_key: str = "y_at_rank",
) -> dict[str, object]:
    """단일 패널(full / active 등)용 요약 dict."""
    arr = np.asarray(y_plot_order, dtype=np.float64)
    return {
        "panel": panel,
        "n_bars": len(y_plot_order),
        "sort": sort_description,
        "x_axis": x_axis,
        "y_axis": y_axis,
        "percentile_step": percentile_step,
        "y_stats": stats_1d_numeric(arr),
        "percentile_samples": sorted_rank_percentile_curve(
            list(y_plot_order),
            percentile_step=percentile_step,
            value_key=value_key,
        ),
    }


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


# Threshold bar colors (light); freq boundary line uses LINE_FREQ (dark blue)
_THR_BAR_GRAY = "#d5d5d5"
_THR_BAR_FREQ = "#90caf9"
_THR_BAR_LESS = "#ffcdd2"
_LINE_FREQ = "#0d47a1"


def bar_colors_threshold_regions(
    n: int,
    freq_thr: float | None,
    less_thr: float | None,
) -> list[str]:
    """Per-bar color: light blue (freq), light red (less), light gray (rest or freq∩less overlap)."""
    if n <= 0:
        return []
    n_freq = rank_count_for_threshold(freq_thr, n)
    n_less = rank_count_for_threshold(less_thr, n)
    use = (freq_thr is not None) or (less_thr is not None)
    if not use:
        return ["steelblue"] * n
    colors: list[str] = []
    for i in range(n):
        in_freq = freq_thr is not None and i < n_freq
        in_less = less_thr is not None and i >= n - n_less
        if in_freq and in_less:
            colors.append(_THR_BAR_GRAY)
        elif in_freq:
            colors.append(_THR_BAR_FREQ)
        elif in_less:
            colors.append(_THR_BAR_LESS)
        else:
            colors.append(_THR_BAR_GRAY)
    return colors


def plot_bar_counts(
    keys: list[tuple[int, int]],
    counts: list[int],
    *,
    title: str,
    ylabel: str,
    out_path: Path,
    xlabel: str,
    mark_first_zero_boundary: bool = False,
    freq_thr: float | None = None,
    less_thr: float | None = None,
) -> None:
    keys, counts = sort_by_count_desc(keys, counts)

    y = np.asarray(counts, dtype=np.float64)
    n = len(keys)
    x = np.arange(n)
    first_zero_idx: int | None = (
        next((i for i, c in enumerate(counts) if c == 0), None) if n else None
    )

    use_xticks = n > 1
    fig_h = 6.8 if use_xticks else 5.0
    fig, ax = plt.subplots(figsize=(14, fig_h))
    bar_colors = bar_colors_threshold_regions(n, freq_thr, less_thr)
    ax.bar(x, y, width=1.0, color=bar_colors, edgecolor="none", linewidth=0)
    add_percentile_vlines(ax, n)

    # full 전용: 첫 count==0 경계선·상단 문구 (c=0 xtick 은 아래 build_rank_xticks)
    if mark_first_zero_boundary and n > 0:
        if first_zero_idx is not None and first_zero_idx > 0:
            x_line = first_zero_idx - 0.5
            ax.axvline(
                x_line,
                color="#cc5500",
                linestyle="-",
                linewidth=1.6,
                alpha=0.95,
                zorder=6,
            )
            ymin, ymax = ax.get_ylim()
            ax.set_ylim(ymin, ymax * 1.1)
            y_top = ax.get_ylim()[1]
            ax.text(
                x_line,
                y_top,
                "← activated | count=0 →",
                ha="center",
                va="bottom",
                fontsize=8,
                color="#cc5500",
                clip_on=False,
            )

    n_freq = rank_count_for_threshold(freq_thr, n)
    n_less = rank_count_for_threshold(less_thr, n)
    if freq_thr is not None and n > 0 and 0 < n_freq < n:
        x_freq = float(n_freq) - 0.5
        ax.axvline(
            x_freq,
            color=_LINE_FREQ,
            linestyle="-",
            linewidth=1.4,
            alpha=0.95,
            zorder=5,
        )
        _, ymax = ax.get_ylim()
        ax.text(
            x_freq,
            ymax,
            f" top {freq_thr:.0%}→",
            ha="right",
            va="bottom",
            fontsize=7,
            color=_LINE_FREQ,
            clip_on=False,
        )
    if less_thr is not None and n > 0 and 0 < n_less < n:
        x_less = float(n - n_less) - 0.5
        ax.axvline(
            x_less,
            color="#c62828",
            linestyle="-",
            linewidth=1.4,
            alpha=0.9,
            zorder=5,
        )
        _, ymax = ax.get_ylim()
        ax.text(
            x_less,
            ymax,
            f"←bottom {less_thr:.0%} ",
            ha="left",
            va="bottom",
            fontsize=7,
            color="#c62828",
            clip_on=False,
        )

    if use_xticks:
        xt_pos, xt_lab = build_rank_xticks(
            n,
            counts,
            first_zero_idx,
            include_c0_boundary_tick=mark_first_zero_boundary,
        )
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


def display_model_name(model_name: str | None) -> str:
    if not model_name:
        return "unknown"
    s = model_name.rstrip("/")
    return Path(s).name if s else "unknown"


def resolve_out_dir_and_file_stem(
    out_path: str | None,
    jsonl_path: Path,
) -> tuple[Path, str]:
    """
    - If --out-path is omitted: jsonl parent + stem ``expert_dist``.
    - If --out-path exists and is a directory: write inside it as ``expert_dist_full.png``, etc.
    - Otherwise (file stem): write ``{parent}/{name}_full.png`` (legacy; parent is created).
    """
    default_stem = "expert_dist"
    if not out_path:
        return jsonl_path.parent, default_stem
    p = Path(out_path).expanduser().resolve()
    if p.is_dir():
        return p, default_stem
    if p.is_file():
        raise SystemExit(f"--out-path must not be an existing file: {p}")
    out_dir = p.parent
    stem = p.name
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir, stem


def format_threshold_title_suffix(
    freq_thr: float | None,
    less_thr: float | None,
) -> str:
    """threshold 모드일 때 figure 제목 끝에 붙이는 문구(미사용 시 빈 문자열)."""
    if freq_thr is None and less_thr is None:
        return ""
    parts: list[str] = ["threshold"]
    if freq_thr is not None:
        parts.append(f"freq={freq_thr:.0%}")
    if less_thr is not None:
        parts.append(f"less={less_thr:.0%}")
    return " | " + " ".join(parts)


def parse_optional_thr(name: str, raw: str | None) -> float | None:
    if raw is None or raw == "":
        return None
    v = float(raw)
    if not 0.0 <= v <= 1.0:
        raise SystemExit(f"{name} must be in [0, 1], got {v}")
    return v


def save_threshold_experts_json(
    path: Path,
    *,
    freq_thr: float | None,
    less_thr: float | None,
    sorted_keys: list[tuple[int, int]],
    ranking_basis: str,
) -> None:
    n = len(sorted_keys)
    n_freq = rank_count_for_threshold(freq_thr, n)
    n_less = rank_count_for_threshold(less_thr, n)
    freq_list = sorted_keys[:n_freq]
    less_list = sorted_keys[n - n_less :] if n_less else []
    payload = {
        "freq_thr": freq_thr,
        "less_thr": less_thr,
        "thr_y": freq_thr,
        "thr_x": less_thr,
        "ranking_basis": ranking_basis,
        "n_ranked": n,
        "n_freq": n_freq,
        "n_less": n_less,
        "n_y": n_freq,
        "n_x": n_less,
        "freq_expert": [{"layer": L, "expert_id": e} for L, e in freq_list],
        "less_expert": [{"layer": L, "expert_id": e} for L, e in less_list],
        "experts_y": [{"layer": L, "expert_id": e} for L, e in freq_list],
        "experts_x": [{"layer": L, "expert_id": e} for L, e in less_list],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def main() -> None:
    p = argparse.ArgumentParser(description="expert 활성화 분포 JSONL → count 막대 플롯 (full / active)")
    p.add_argument(
        "--jsonl",
        type=str,
        required=True,
        help="step2 의 token_routing.jsonl 경로",
    )
    p.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="제목 표시 및 num_experts 자동 조회(AutoConfig)용 HF 경로",
    )
    p.add_argument(
        "--num-experts",
        type=int,
        default=None,
        help="레이어당 expert 수(미지정 시 --model-name config 또는 데이터 max_id+1)",
    )
    p.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="AutoConfig 로드 시 trust_remote_code",
    )
    p.add_argument(
        "--out-path",
        type=str,
        default=None,
        help=(
            "Output directory (must exist or be created before run) → expert_dist_full.png inside it; "
            "or a path whose last component is a filename stem → parent/(stem)_full.png (legacy)."
        ),
    )
    p.add_argument(
        "--freq-thr",
        type=str,
        default=None,
        help="0~1. count 내림차순 rank 기준 상위 비율에 해당하는 expert → freq_expert 집합·경계선(미지정 시 비활성)",
    )
    p.add_argument(
        "--less-thr",
        type=str,
        default=None,
        help="0~1. 동일 정렬에서 하위 비율 → less_expert 집합·경계선(미지정 시 비활성)",
    )
    p.add_argument(
        "--thr-basis",
        type=str,
        choices=["full", "active"],
        default="full",
        help="임계 expert 집합(JSON)의 rank 기준: full(전체 슬롯, count 내림차순) 또는 active(count>0 만, active 막대와 동일)",
    )
    args = p.parse_args()

    jsonl_path = Path(args.jsonl).resolve()
    if not jsonl_path.is_file():
        raise SystemExit(f"파일 없음: {jsonl_path}")

    ctr, total_tokens, max_eid_seen = scan_jsonl(jsonl_path)
    fallback_ne = max_eid_seen + 1
    num_experts = args.num_experts
    if num_experts is None:
        num_experts = resolve_num_experts_per_layer(
            args.model_name,
            trust_remote_code=args.trust_remote_code,
            fallback_max_id_plus_one=fallback_ne,
        )
    if num_experts < 1:
        raise SystemExit("num_experts 유효하지 않음")

    layers = ordered_layers_from_counter(ctr)
    if not layers:
        raise SystemExit("집계된 (layer, expert) 가 없습니다. jsonl 형식을 확인하세요.")

    full_grid = build_full_grid(layers, num_experts)
    full_counts = [ctr.get(k, 0) for k in full_grid]
    total_activation_events = int(sum(full_counts))
    if total_activation_events == 0:
        raise SystemExit("활성화 합계가 0 입니다.")

    total_expert_slots = len(full_grid)
    active_keys = [k for k in full_grid if ctr.get(k, 0) > 0]
    active_counts = [ctr[k] for k in active_keys]
    n_active = len(active_keys)

    model_disp = display_model_name(args.model_name)
    out_dir, file_stem = resolve_out_dir_and_file_stem(args.out_path, jsonl_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    freq_thr = parse_optional_thr("--freq-thr", args.freq_thr)
    less_thr = parse_optional_thr("--less-thr", args.less_thr)
    use_threshold = freq_thr is not None or less_thr is not None
    # Plot/json basename: include "_threshold" when freq/less thresholds are set
    plot_stem = f"{file_stem}_threshold" if use_threshold else file_stem
    if use_threshold:
        if args.thr_basis == "active":
            sorted_for_json, _ = sort_by_count_desc(active_keys, active_counts)
            json_ranking_basis = "active_count_desc"
        else:
            sorted_for_json, _ = sort_by_count_desc(full_grid, full_counts)
            json_ranking_basis = "full_grid_count_desc"
        json_path = out_dir / f"{plot_stem}_experts.json"
        save_threshold_experts_json(
            json_path,
            freq_thr=freq_thr,
            less_thr=less_thr,
            sorted_keys=sorted_for_json,
            ranking_basis=json_ranking_basis,
        )

    thr_kw = {}
    if use_threshold:
        thr_kw["freq_thr"] = freq_thr
        thr_kw["less_thr"] = less_thr

    thr_suffix = format_threshold_title_suffix(freq_thr, less_thr)
    # --- Full: 모든 슬롯(0 포함)
    title_full = (
        f"{model_disp} | tokens={total_tokens} | experts(layer×per_expert)={total_expert_slots} "
        f"(MoE layers={len(layers)}, per_layer={num_experts})"
        f"{thr_suffix}"
    )
    plot_bar_counts(
        full_grid,
        full_counts,
        title=title_full,
        ylabel="activation count",
        out_path=out_dir / f"{plot_stem}_full.png",
        xlabel="sorted rank (xticks: percentile; full adds c=0 boundary — idx & count)",
        mark_first_zero_boundary=True,
        **thr_kw,
    )

    # --- Active: count>0 만 (full 과 동일한 percentile 점선·xtick, count=0 전용 표시 없음)
    title_active = (
        f"{model_disp} | tokens={total_tokens} | active / total experts = {n_active} / {total_expert_slots}"
        f"{thr_suffix}"
    )
    plot_bar_counts(
        active_keys,
        active_counts,
        title=title_active,
        ylabel="activation count",
        out_path=out_dir / f"{plot_stem}_active.png",
        xlabel="sorted rank (xticks: percentile — idx & count)",
        mark_first_zero_boundary=False,
        **thr_kw,
    )

    print(f"Wrote {out_dir / (plot_stem + '_full.png')}")
    print(f"Wrote {out_dir / (plot_stem + '_active.png')}")
    if use_threshold:
        print(f"Wrote {out_dir / (plot_stem + '_experts.json')}")
    if not use_threshold:
        _, full_counts_sorted = sort_by_count_desc(full_grid, full_counts)
        _, active_counts_sorted = sort_by_count_desc(active_keys, active_counts)
        summary_path = out_dir / f"{plot_stem}_distribution_summary.json"
        write_json(
            summary_path,
            {
                "schema": "expert_balance.distribution_summary.rank_curve.v1",
                "script": "step3_count_expert_dist.py",
                "model_display": model_disp,
                "jsonl": str(jsonl_path),
                "total_tokens": total_tokens,
                "total_activation_events": total_activation_events,
                "total_expert_slots": total_expert_slots,
                "n_active_experts": n_active,
                "full_grid": distribution_section_rank_curve(
                    panel="full_grid",
                    sort_description="count descending (left=high count, right=low/zero count)",
                    x_axis="sorted_rank_percentile (0=left ... 100=right, matches bar plot)",
                    y_axis="activation_count",
                    y_plot_order=full_counts_sorted,
                ),
                "active_only": distribution_section_rank_curve(
                    panel="active_only",
                    sort_description="count descending over experts with count>0 only",
                    x_axis="sorted_rank_percentile (0=left ... 100=right, matches bar plot)",
                    y_axis="activation_count",
                    y_plot_order=active_counts_sorted,
                ),
            },
        )
        print(f"Wrote {summary_path}")
    print(f"total_tokens={total_tokens}, total_activation_events={total_activation_events}")
    print(f"total_expert_slots={total_expert_slots}, active_experts={n_active}")


if __name__ == "__main__":
    main()
