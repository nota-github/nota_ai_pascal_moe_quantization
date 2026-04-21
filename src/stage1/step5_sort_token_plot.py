"""
threshold JSON 의 y/x 전문가 집합(활성화 분포 또는 양자화 민감도)을 사용해,
token_routing.jsonl 의 각 토큰에 대해 (해당 집합에 속하는 활성화 비율)을 계산하고 2D scatter 를 저장한다.

토큰 영역 임계(--token-thr-blue/red 등)를 **지정하지 않은** 경우,
동일 stem 의 ``*_distribution_summary.json`` 에 각 축 주변분포(2.5%), [0,1]² 그리드 카운트,
상관을 저장해 산점도를 텍스트로 근사한다 (LLM/도구용).

- balance: x축 = experts_x(less), y축 = experts_y(freq)
- q_sensitivity: x축 = experts_y(sensitive), y축 = experts_x(robust)

--mode balance:
  experts_y = freq_expert, experts_x = less_expert (step3, count rank)
--mode q_sensitivity:
  experts_y = q_sensitive_expert, experts_x = q_robust_group (step4, sensitivity rank)
  산점도: x축 = q_sensitive 비율, y축 = q_robust 비율

threshold JSON 은 step3/step4 가 함께 쓰는 통일 키(thr_y, thr_x, experts_y, experts_x)를 포함한다.
구버전 JSON 은 mode 에 맞는 기존 키(freq_expert/less_expert 또는 q_* )로 읽는다.

옵션: 토큰 영역 분할 (--token-thr-blue / --token-thr-red, 구명은 --freq-token-thr 등과 동일)
  balance·q_sensitivity 동일 부등식 (플롯 축 x,y 기준):
    token_blue: sensitive(x) < btx & robust(y) > bty  — balance 에서는 x=less, y=freq
    token_red:  sensitive(x) > rtx & robust(y) < rty
저장·강조 색 산점도(파랑/빨강).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from step3_count_expert_dist import (
    build_full_grid,
    ordered_layers_from_counter,
    parse_optional_thr,
    rank_count_for_threshold,
    resolve_num_experts_per_layer,
    scan_jsonl,
    sort_by_count_desc,
    write_json,
)

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica", "sans-serif"],
    }
)


def _expert_set_from_json_list(d: dict, key: str) -> set[tuple[int, int]]:
    return {(int(x["layer"]), int(x["expert_id"])) for x in d.get(key, [])}


def load_expert_sets_from_threshold_json(path: Path, mode: str) -> tuple[set[tuple[int, int]], set[tuple[int, int]], dict]:
    """
    experts_y / experts_x 가 있으면 mode 무관하게 사용.
    없으면 mode 에 따라 freq_expert/less_expert 또는 q_sensitive_expert/q_robust_group.
    """
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    ey, ex = d.get("experts_y"), d.get("experts_x")
    if ey is not None and ex is not None:
        return _expert_set_from_json_list(d, "experts_y"), _expert_set_from_json_list(d, "experts_x"), d
    if mode == "balance":
        return (
            _expert_set_from_json_list(d, "freq_expert"),
            _expert_set_from_json_list(d, "less_expert"),
            d,
        )
    if mode == "q_sensitivity":
        return (
            _expert_set_from_json_list(d, "q_sensitive_expert"),
            _expert_set_from_json_list(d, "q_robust_group"),
            d,
        )
    raise SystemExit(f"알 수 없는 mode: {mode}")


def expert_thr_y_x_from_meta(meta: dict, mode: str) -> tuple[float | None, float | None]:
    ty, tx = meta.get("thr_y"), meta.get("thr_x")
    if ty is not None or tx is not None:
        return ty, tx
    if mode == "balance":
        return meta.get("freq_thr"), meta.get("less_thr")
    return meta.get("sen_thr"), meta.get("rob_thr")


def build_freq_less_sets_from_counts(
    ctr,
    *,
    layers: list[int],
    num_experts: int,
    freq_thr: float | None,
    less_thr: float | None,
) -> tuple[set[tuple[int, int]], set[tuple[int, int]], int]:
    """step3 save_threshold_experts_json 과 동일한 규칙."""
    full_grid = build_full_grid(layers, num_experts)
    full_counts = [ctr.get(k, 0) for k in full_grid]
    sorted_k, _ = sort_by_count_desc(full_grid, full_counts)
    n = len(sorted_k)
    n_freq = rank_count_for_threshold(freq_thr, n)
    n_less = rank_count_for_threshold(less_thr, n)
    freq_set = set(sorted_k[:n_freq])
    less_set = set(sorted_k[n - n_less :]) if n_less else set()
    return freq_set, less_set, n


def token_y_x_ratios_and_meta(
    jsonl_path: Path,
    y_set: set[tuple[int, int]],
    x_set: set[tuple[int, int]],
    *,
    mode: str,
    max_lines: int | None,
) -> tuple[np.ndarray, np.ndarray, list[dict], int]:
    """
    각 jsonl 라인(토큰)마다 전체 (layer,eid) 활성화 슬롯 대비
    experts_y / experts_x 매칭 비율 + sample_id, position_id, token_id.
    balance: x = x_set, y = y_set.
    q_sensitivity: x = sensitive(y_set), y = robust(x_set).
    """
    xs: list[float] = []
    ys: list[float] = []
    metas: list[dict] = []
    n_lines = 0
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n_lines += 1
            if max_lines is not None and n_lines > max_lines:
                break
            row = json.loads(line)
            total = 0
            n_y = 0
            n_x = 0
            for layer_str, eids in row.get("activated_expert_ids", {}).items():
                L = int(layer_str)
                for eid in eids:
                    eid = int(eid)
                    total += 1
                    key = (L, eid)
                    if key in y_set:
                        n_y += 1
                    if key in x_set:
                        n_x += 1
            if total <= 0:
                continue
            if mode == "q_sensitivity":
                xs.append(n_y / total)
                ys.append(n_x / total)
            else:
                xs.append(n_x / total)
                ys.append(n_y / total)
            metas.append(
                {
                    "sample_id": int(row["sample_id"]),
                    "position_id": int(row["position_id"]),
                    "token_id": int(row["token_id"]),
                }
            )
    return (
        np.asarray(xs, dtype=np.float64),
        np.asarray(ys, dtype=np.float64),
        metas,
        n_lines,
    )


def classify_token_regions(
    x: np.ndarray,
    y: np.ndarray,
    *,
    btx: float,
    bty: float,
    rtx: float,
    rty: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    token_blue: x < btx & y > bty  (q: sensitive < btx, robust > bty)
    token_red:  x > rtx & y < rty  (q: sensitive > rtx, robust < rty)
    balance 에서는 x=less, y=freq.
    """
    mask_blue = (x < btx) & (y > bty)
    mask_red = (x > rtx) & (y < rty)
    return mask_blue, mask_red


def plot_scatter(
    x: np.ndarray,
    y: np.ndarray,
    *,
    out_path: Path,
    title: str,
    mode: str,
    thr_y: float | None,
    thr_x: float | None,
    mask_blue: np.ndarray | None = None,
    mask_red: np.ndarray | None = None,
) -> None:
    if mode == "balance":
        xlab = "less_expert activation ratio (per token)"
        ylab = "freq_expert activation ratio (per token)"
        leg_y, leg_x = "freq_token", "less_token"
    else:
        xlab = "q_sensitive_expert activation ratio (per token)"
        ylab = "q_robust_group activation ratio (per token)"
        # 파랑 = 민감↓·강건↑ 모서리 → q_robust_token, 빨강 = 민감↑·강건↓ → q_sensitive_token
        leg_y, leg_x = "q_robust_token", "q_sensitive_token"

    fig, ax = plt.subplots(figsize=(9.5, 8.0))
    n = len(x)
    alpha_bg = 0.35 if n < 5000 else (0.12 if n < 50000 else 0.05)
    alpha_hi = 0.45 if n < 5000 else (0.2 if n < 50000 else 0.12)
    s = 8 if n < 20000 else 4
    # 대용량(n>=20k)도 balance와 동일 분기라서, q_sensitivity는 두 구간 모두 약 2배로 키움
    if mode == "q_sensitivity":
        s = max(6, int(round(s * 5.0)))

    use_regions = mask_blue is not None and mask_red is not None
    if use_regions:
        # 전체: 연한 회색
        ax.scatter(
            x,
            y,
            s=s,
            c="#c8c8c8",
            alpha=alpha_bg,
            edgecolors="none",
            linewidths=0,
            label="other",
            zorder=1,
        )
        # token_blue 영역: 연한 파랑
        if np.any(mask_blue):
            ax.scatter(
                x[mask_blue],
                y[mask_blue],
                s=s * 1.15,
                c="#8ec5ff",
                alpha=alpha_hi,
                edgecolors="none",
                linewidths=0,
                label=leg_y,
                zorder=3,
            )
        # token_red 영역: 연한 빨강
        if np.any(mask_red):
            ax.scatter(
                x[mask_red],
                y[mask_red],
                s=s * 1.15,
                c="#ffb3b3",
                alpha=alpha_hi,
                edgecolors="none",
                linewidths=0,
                label=leg_x,
                zorder=2,
            )
        ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    else:
        ax.scatter(x, y, s=s, c="#1f77b4", alpha=alpha_bg, edgecolors="none", linewidths=0)

    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.grid(True, alpha=0.3)
    thr_bits = []
    if thr_y is not None:
        thr_bits.append(f"thr_y={thr_y:.4g}")
    if thr_x is not None:
        thr_bits.append(f"thr_x={thr_x:.4g}")
    sub = " | ".join(thr_bits) if thr_bits else "threshold from JSON"
    ax.set_title(f"{title}\n{sub} | n_tokens={n:,}")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def token_scatter_distribution_summary_dict(
    x: np.ndarray,
    y: np.ndarray,
    *,
    mode: str,
    model_label: str,
    thr_y: float | None,
    thr_x: float | None,
    grid_bins: int = 24,
) -> dict[str, object]:
    """
    토큰별 (x,y) 비율 산점도 요약: 축은 [0,1] 구간의 활성화 비율.
    balance: x=less 집합 비율, y=freq 집합 비율.
    q_sensitivity: x=q_sensitive 비율, y=q_robust 비율 (plot_scatter 와 동일).
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    n = int(x.size)
    qs = np.arange(0, 100.25, 2.5)

    def pct_dict(arr: np.ndarray) -> dict[str, float]:
        return {f"p{p:g}": float(np.percentile(arr, p)) for p in qs}

    if mode == "balance":
        x_name = "less_expert_activation_ratio"
        y_name = "freq_expert_activation_ratio"
    else:
        x_name = "q_sensitive_expert_activation_ratio"
        y_name = "q_robust_group_activation_ratio"

    out: dict[str, object] = {
        "schema": "expert_balance.distribution_summary.token_scatter.v1",
        "script": "step5_sort_token_plot.py",
        "mode": mode,
        "model_display": model_label,
        "expert_thr_y": thr_y,
        "expert_thr_x": thr_x,
        "n_tokens": n,
        "axes": {
            "x": x_name,
            "y": y_name,
            "domain": "[0,1] per-token routing fractions",
        },
    }
    if n == 0:
        out["note"] = "empty"
        return out

    xc = np.clip(x, 0.0, 1.0)
    yc = np.clip(y, 0.0, 1.0)
    out["marginal_x"] = pct_dict(xc)
    out["marginal_y"] = pct_dict(yc)
    if n > 1:
        c = np.corrcoef(xc, yc)[0, 1]
        pearson = float(c) if np.isfinite(c) else None
        rx = np.argsort(np.argsort(xc))
        ry = np.argsort(np.argsort(yc))
        sp = np.corrcoef(rx.astype(np.float64), ry.astype(np.float64))[0, 1]
        spearman = float(sp) if np.isfinite(sp) else None
    else:
        pearson = None
        spearman = None
    out["correlation"] = {"pearson": pearson, "spearman": spearman}

    H, xe, ye = np.histogram2d(xc, yc, bins=grid_bins, range=[[0.0, 1.0], [0.0, 1.0]])
    out["histogram2d_unit_square"] = {
        "n_bins_each_axis": grid_bins,
        "x_bin_edges": xe.tolist(),
        "y_bin_edges": ye.tolist(),
        "counts": H.tolist(),
    }
    # 질량이 어느 사분면에 있는지 대략 파악 (임계 없이)
    out["quadrant_mass_by_mean_position"] = {
        "mean_x_lt_0p5": float(np.mean(xc < 0.5)),
        "mean_x_ge_0p5": float(np.mean(xc >= 0.5)),
        "mean_y_lt_0p5": float(np.mean(yc < 0.5)),
        "mean_y_ge_0p5": float(np.mean(yc >= 0.5)),
    }
    return out


def save_token_sets_json(
    path: Path,
    *,
    mode: str,
    thr_y: float | None,
    thr_x: float | None,
    btx: float,
    bty: float,
    rtx: float,
    rty: float,
    metas: list[dict],
    x: np.ndarray,
    y: np.ndarray,
    mask_blue: np.ndarray,
    mask_red: np.ndarray,
) -> None:
    def entries_for_mask(mask: np.ndarray, *, balance_style: bool) -> list[dict]:
        out: list[dict] = []
        for i in np.flatnonzero(mask):
            i = int(i)
            m = metas[i]
            row = {
                **m,
                "ratio_x": float(x[i]),
                "ratio_y": float(y[i]),
            }
            if balance_style:
                row["less_expert_ratio"] = float(x[i])
                row["freq_expert_ratio"] = float(y[i])
            else:
                row["q_sensitive_expert_ratio"] = float(x[i])
                row["q_robust_group_ratio"] = float(y[i])
            out.append(row)
        return out

    balance_style = mode == "balance"
    tokens_blue = entries_for_mask(mask_blue, balance_style=balance_style)
    tokens_red = entries_for_mask(mask_red, balance_style=balance_style)

    region_blue_rule = "x < token_thr_blue.x and y > token_thr_blue.y"
    region_red_rule = "x > token_thr_red.x and y < token_thr_red.y"

    payload: dict = {
        "mode": mode,
        "thr_y": thr_y,
        "thr_x": thr_x,
        "token_thr_blue": {"x": btx, "y": bty},
        "token_thr_red": {"x": rtx, "y": rty},
        "token_region_blue_rule": region_blue_rule,
        "token_region_red_rule": region_red_rule,
        "n_tokens_total": int(len(x)),
        "n_token_blue_region": int(np.sum(mask_blue)),
        "n_token_red_region": int(np.sum(mask_red)),
        "token_blue_region": tokens_blue,
        "token_red_region": tokens_red,
    }
    if mode == "balance":
        payload["freq_token_thr"] = {"x": btx, "y": bty}
        payload["less_token_thr"] = {"x": rtx, "y": rty}
        payload["freq_token_rule"] = payload["token_region_blue_rule"]
        payload["less_token_rule"] = payload["token_region_red_rule"]
        payload["n_freq_token"] = payload["n_token_blue_region"]
        payload["n_less_token"] = payload["n_token_red_region"]
        payload["freq_token"] = tokens_blue
        payload["less_token"] = tokens_red
    else:
        # 파랑 영역 = 강건 우세(민감↓·강건↑), 빨강 = 민감 우세
        payload["q_robust_token_thr"] = {"x": btx, "y": bty}
        payload["q_sensitive_token_thr"] = {"x": rtx, "y": rty}
        payload["q_robust_token_rule"] = payload["token_region_blue_rule"]
        payload["q_sensitive_token_rule"] = payload["token_region_red_rule"]
        payload["n_q_robust_token"] = payload["n_token_blue_region"]
        payload["n_q_sensitive_token"] = payload["n_token_red_region"]
        payload["q_robust_token"] = tokens_blue
        payload["q_sensitive_token"] = tokens_red

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def main() -> None:
    p = argparse.ArgumentParser(
        description="token별 experts_y / experts_x 활성 비율 2D scatter (threshold JSON 기준)",
    )
    p.add_argument("--jsonl", type=str, required=True, help="step2 token_routing.jsonl")
    p.add_argument(
        "--threshold-json",
        type=str,
        default=None,
        help="step3/step4 *_threshold_experts.json (thr_y/thr_x, experts_y/experts_x 권장)",
    )
    p.add_argument(
        "--freq-thr",
        type=str,
        default=None,
        help="threshold-json 없을 때: thr_y (count 상위 비율) — --thr-y 와 동일",
    )
    p.add_argument(
        "--less-thr",
        type=str,
        default=None,
        help="threshold-json 없을 때: thr_x (count 하위 비율) — --thr-x 와 동일",
    )
    p.add_argument(
        "--thr-y",
        type=str,
        default=None,
        help="--freq-thr 과 동일 (통일 이름)",
    )
    p.add_argument(
        "--thr-x",
        type=str,
        default=None,
        help="--less-thr 과 동일 (통일 이름)",
    )
    p.add_argument("--model-name", type=str, default=None, help="제목용 (optional)")
    p.add_argument(
        "--mode",
        type=str,
        choices=("balance", "q_sensitivity"),
        default="balance",
        help="balance: freq/less 명명 · q_sensitivity: q_sensitive / q_robust 명명",
    )
    p.add_argument("--num-experts", type=int, default=None)
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument(
        "--out-path",
        type=str,
        default=None,
        help="PNG 경로 또는 디렉터리. 토큰 영역 미지정: token_freq_less_scatter_<mode>.png; "
        "지정 시: token_freq_less_scatter_<mode>_classified.png (+ 동명 .json)",
    )
    p.add_argument("--max-tokens", type=int, default=None, help="디버그용 상위 N토큰만")
    p.add_argument(
        "--token-thr-blue",
        type=float,
        nargs=2,
        metavar=("X", "Y"),
        default=None,
        help="파랑 토큰 영역 (balance: x<X & y>Y 등, 규칙은 출력 JSON 참고)",
    )
    p.add_argument(
        "--token-thr-red",
        type=float,
        nargs=2,
        metavar=("X", "Y"),
        default=None,
        help="빨강 토큰 영역",
    )
    p.add_argument(
        "--freq-token-thr",
        type=float,
        nargs=2,
        metavar=("X", "Y"),
        default=None,
        help="--token-thr-blue 와 동일 (구 이름)",
    )
    p.add_argument(
        "--less-token-thr",
        type=float,
        nargs=2,
        metavar=("X", "Y"),
        default=None,
        help="--token-thr-red 와 동일 (구 이름)",
    )
    p.add_argument(
        "--token-thr-y",
        type=float,
        nargs=2,
        metavar=("X", "Y"),
        default=None,
        help="--token-thr-blue 와 동일 (구 이름)",
    )
    p.add_argument(
        "--token-thr-x",
        type=float,
        nargs=2,
        metavar=("X", "Y"),
        default=None,
        help="--token-thr-red 와 동일 (구 이름)",
    )
    args = p.parse_args()
    mode = args.mode

    blue_arg = args.freq_token_thr or args.token_thr_y or args.token_thr_blue
    red_arg = args.less_token_thr or args.token_thr_x or args.token_thr_red

    jsonl_path = Path(args.jsonl).resolve()
    if not jsonl_path.is_file():
        raise SystemExit(f"파일 없음: {jsonl_path}")

    has_blue = blue_arg is not None
    has_red = red_arg is not None
    if has_blue ^ has_red:
        raise SystemExit(
            "토큰 영역은 --token-thr-blue 와 --token-thr-red 를 둘 다 주거나 둘 다 생략해야 합니다. "
            "(구 이름: --freq-token-thr/--token-thr-y, --less-token-thr/--token-thr-x)",
        )
    use_token_thr = has_blue and has_red
    if use_token_thr:
        btx, bty = float(blue_arg[0]), float(blue_arg[1])
        rtx, rty = float(red_arg[0]), float(red_arg[1])

    thr_y: float | None
    thr_x: float | None
    meta: dict = {}

    if args.threshold_json:
        tpath = Path(args.threshold_json).resolve()
        if not tpath.is_file():
            raise SystemExit(f"threshold json 없음: {tpath}")
        y_set, x_set, meta = load_expert_sets_from_threshold_json(tpath, mode)
        thr_y, thr_x = expert_thr_y_x_from_meta(meta, mode)
        title_model = Path(args.model_name).name if args.model_name else "model"
    else:
        thr_y = parse_optional_thr("--freq-thr", args.freq_thr)
        if thr_y is None:
            thr_y = parse_optional_thr("--thr-y", args.thr_y)
        thr_x = parse_optional_thr("--less-thr", args.less_thr)
        if thr_x is None:
            thr_x = parse_optional_thr("--thr-x", args.thr_x)
        if thr_y is None and thr_x is None:
            raise SystemExit(
                "Provide --threshold-json from step3/step4, or at least one of "
                "--freq-thr/--thr-y and --less-thr/--thr-x with the same --jsonl as step3 for consistent ranking.",
            )
        ctr, _, max_eid_seen = scan_jsonl(jsonl_path)
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
            raise SystemExit("집계된 (layer, expert) 가 없습니다.")
        y_set, x_set, _n_ranked = build_freq_less_sets_from_counts(
            ctr,
            layers=layers,
            num_experts=num_experts,
            freq_thr=thr_y,
            less_thr=thr_x,
        )
        title_model = Path(args.model_name).name if args.model_name else "model"

    x, y, metas, _n_read = token_y_x_ratios_and_meta(
        jsonl_path,
        y_set,
        x_set,
        mode=mode,
        max_lines=args.max_tokens,
    )
    if len(x) == 0:
        raise SystemExit("산출된 토큰이 없습니다 (activated_expert_ids 비어 있음?).")

    default_name = f"token_freq_less_scatter_{mode}.png"
    # 토큰 영역(--token-thr-*) 사용 시 임계값 숫자는 JSON 본문에만 두고, 파일명은 고정(_classified)으로 둔다.
    classified_name = f"token_freq_less_scatter_{mode}_classified.png"

    if args.out_path:
        op = Path(args.out_path).expanduser().resolve()
        if op.is_dir():
            if use_token_thr:
                png_path = op / classified_name
            else:
                png_path = op / default_name
        else:
            base = op
            if str(base).lower().endswith(".png"):
                if use_token_thr:
                    stem = base.stem
                    png_path = base.with_name(f"{stem}_{mode}_classified.png")
                else:
                    png_path = base.with_name(f"{base.stem}_{mode}.png")
            else:
                base = base.with_suffix(".png")
                if use_token_thr:
                    png_path = base.with_name(f"{base.stem}_{mode}_classified.png")
                else:
                    png_path = base.with_name(f"{base.stem}_{mode}.png")
            png_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        if use_token_thr:
            png_path = jsonl_path.parent / classified_name
        else:
            png_path = jsonl_path.parent / default_name

    mask_blue: np.ndarray | None = None
    mask_red: np.ndarray | None = None
    if use_token_thr:
        mask_blue, mask_red = classify_token_regions(
            x, y, btx=btx, bty=bty, rtx=rtx, rty=rty
        )
        json_path = png_path.with_suffix(".json")
        save_token_sets_json(
            json_path,
            mode=mode,
            thr_y=thr_y,
            thr_x=thr_x,
            btx=btx,
            bty=bty,
            rtx=rtx,
            rty=rty,
            metas=metas,
            x=x,
            y=y,
            mask_blue=mask_blue,
            mask_red=mask_red,
        )
        print(f"Wrote {json_path}")

    title = f"{title_model} | per-token routing"
    plot_scatter(
        x,
        y,
        out_path=png_path,
        title=title,
        mode=mode,
        thr_y=thr_y,
        thr_x=thr_x,
        mask_blue=mask_blue,
        mask_red=mask_red,
    )
    print(f"Wrote {png_path}")
    if not use_token_thr:
        summ_path = png_path.with_name(png_path.stem + "_distribution_summary.json")
        write_json(
            summ_path,
            token_scatter_distribution_summary_dict(
                x,
                y,
                mode=mode,
                model_label=title_model,
                thr_y=thr_y,
                thr_x=thr_x,
            ),
        )
        print(f"Wrote {summ_path}")
    print(f"tokens plotted={len(x):,}")


if __name__ == "__main__":
    main()
