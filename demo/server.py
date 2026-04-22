from __future__ import annotations

import asyncio
import json
from pathlib import Path

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="MoE Quantization Pipeline Demo")

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS   = REPO_ROOT / "results"
STATIC    = Path(__file__).resolve().parent / "static"

app.mount("/results-files", StaticFiles(directory=str(RESULTS)), name="results-files")
app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")


@app.get("/", response_class=HTMLResponse)
async def index():
    return (STATIC / "index.html").read_text(encoding="utf-8")


# ── Phase 1 ──────────────────────────────────────────────────────────────────

@app.get("/api/results/phase1")
async def phase1_results():
    base   = RESULTS / "phase1_routing/qwen3_30b_a3b_nemo_dataset/D0_128"
    prefix = "phase1_routing/qwen3_30b_a3b_nemo_dataset/D0_128"
    data: dict = {"plots": {}, "stats": {}, "bracket_sample": None}

    summaries = {
        "expert_dist":    base / "s3_expert_dist/expert_dist_distribution_summary.json",
        "weight_outlier": base / "s4_weight_outlier/expert_weight_sensitivity_distribution_summary.json",
        "balance_token":  base / "s5_sorted_token/token_freq_less_scatter_balance_distribution_summary.json",
        "q_sens_token":   base / "s5_sorted_token/token_freq_less_scatter_q_sensitivity_distribution_summary.json",
    }
    for key, path in summaries.items():
        if path.exists():
            try:
                data["stats"][key] = json.loads(path.read_text())
            except Exception:
                pass

    sample_path = base / "s6_apply_bracket/bracketed_balance_one.json"
    if sample_path.exists():
        try:
            data["bracket_sample"] = json.loads(sample_path.read_text())
        except Exception:
            pass

    data["plots"] = {
        "Expert Activation Distribution":  f"/results-files/{prefix}/s3_expert_dist/expert_dist_full.png",
        "Expert Activation (Threshold)":   f"/results-files/{prefix}/s3_expert_dist/expert_dist_threshold_full.png",
        "Weight Sensitivity (Full)":       f"/results-files/{prefix}/s4_weight_outlier/expert_weight_sensitivity_full.png",
        "Sensitivity vs Activation":       f"/results-files/{prefix}/s4_weight_outlier/expert_weight_sensitivity_scatter_sensitivity_activation.png",
        "Token Classification (Balance)":  f"/results-files/{prefix}/s5_sorted_token/token_freq_less_scatter_balance_classified.png",
        "Token Classification (Q-Sens)":   f"/results-files/{prefix}/s5_sorted_token/token_freq_less_scatter_q_sensitivity_classified.png",
    }
    return JSONResponse(data)


# ── Phase 2 ──────────────────────────────────────────────────────────────────

@app.get("/api/results/phase2")
async def phase2_results():
    inst_dir = RESULTS / "phase2_guidelines/instruction"
    domains: dict = {}
    for domain in ("chat", "code", "math", "stem"):
        domains[domain] = {}
        for gtype in ("balance", "q_sensitivity"):
            p = inst_dir / f"{gtype}_guideline_{domain}_nemotron-3-super.md"
            domains[domain][gtype] = p.read_text(encoding="utf-8") if p.exists() else ""
    return JSONResponse({"domains": domains})


# ── Phase 3 ──────────────────────────────────────────────────────────────────

@app.get("/api/results/phase3")
async def phase3_results():
    base = RESULTS / "phase3_dataset/output_per_domain"
    data: dict = {"balance": {}, "q_sensitivity": {}, "totals": {}}

    for gtype in ("balance", "q_sensitivity"):
        for domain in ("chat", "code", "math", "stem"):
            preview = base / gtype / domain / f"preview_{domain}.json"
            if preview.exists():
                try:
                    content = json.loads(preview.read_text())
                    data[gtype][domain] = content if isinstance(content, list) else [content]
                except Exception:
                    pass
        merged = base / f"calibration_{gtype}_all.json"
        if merged.exists():
            try:
                rows = json.loads(merged.read_text())
                data["totals"][gtype] = len(rows) if isinstance(rows, list) else "?"
            except Exception:
                data["totals"][gtype] = "?"

    return JSONResponse(data)


# ── Live run WebSocket ────────────────────────────────────────────────────────

@app.websocket("/ws/run/{phase}")
async def run_phase_ws(phase: int, websocket: WebSocket):
    await websocket.accept()
    script = REPO_ROOT / "scripts" / f"run_phase{phase}.sh"
    proc = None
    try:
        proc = await asyncio.create_subprocess_exec(
            "bash", str(script),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=str(REPO_ROOT),
        )
        assert proc.stdout is not None
        async for raw in proc.stdout:
            line = raw.decode("utf-8", errors="replace").rstrip()
            await websocket.send_json({"line": line})
        await proc.wait()
        await websocket.send_json({"done": True, "returncode": proc.returncode})
    except WebSocketDisconnect:
        if proc and proc.returncode is None:
            proc.kill()
    except Exception as exc:
        try:
            await websocket.send_json({"error": str(exc)})
        except Exception:
            pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass
