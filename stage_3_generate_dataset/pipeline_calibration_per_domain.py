"""
NeMo Data Designer - Per-Domain Calibration Pipeline v2

Generates 8 datasets:
  balance      × {chat, code, math, stem}  → 4 datasets
  q_sensitivity × {chat, code, math, stem} → 4 datasets

Key improvements over v1:
  - Separate single-guideline prompts per combo (simpler → higher valid rate)
  - system_style SamplerColumn: 8 diverse personas per domain → varied prefixes
  - 8 combos run in parallel via ThreadPoolExecutor
  - max_parallel_requests=16 per combo for higher throughput

Output layout:
  output_per_domain/
    balance/chat/   balance/code/   balance/math/   balance/stem/
    q_sensitivity/chat/  ...
    calibration_balance_all.parquet
    calibration_q_sensitivity_all.parquet
"""

import json
import os
import re
import requests
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

# ─── Configuration ────────────────────────────────────────────────────────────
SCRIPT_DIR      = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT       = os.path.dirname(SCRIPT_DIR)
VLLM_BASE_URL   = "http://localhost:8000/v1"
SEED_DATA_PATH  = os.path.join(REPO_ROOT, "stage_1_analyze_routing", "output",
                               "qwen3_30b_a3b_nemo_dataset", "D0_128", "s6_apply_bracket", "bracketed_balance.jsonl")
INSTRUCTION_DIR = os.path.join(REPO_ROOT, "stage_2_pattern_extract", "instruction")
OUTPUT_ROOT     = os.path.join(SCRIPT_DIR, "output_per_domain")
SLACK_WEBHOOK   = os.environ.get("SLACK_WEBHOOK_URL", "")  # set via env var or leave empty to disable

DOMAINS                  = ["chat", "code", "math", "stem"]
GUIDELINE_TYPES          = ["balance", "q_sensitivity"]
NUM_RECORDS_PER_COMBO    = 256   # oversample; ~50% empty-row rate expected
TARGET_RECORDS_PER_COMBO = 128
MAX_PARALLEL_REQUESTS    = 4    # 4 combos × 4 = 16 total concurrent (avoids vLLM timeout)
MAX_WORKERS              = 4    # run 4 combos at a time

# ─── Diverse system-prompt personas per domain ────────────────────────────────
SYSTEM_STYLES = {
    "chat": [
        "You are a technical consultant specializing in AI infrastructure and model deployment.",
        "You are an ML researcher with expertise in transformer architectures and quantization techniques.",
        "You are a systems engineer focusing on distributed training and inference optimization.",
        "You are an expert in MLOps, covering CI/CD pipelines for machine learning workflows.",
        "You are a networking specialist with knowledge of high-performance computing interconnects.",
        "You are a computer vision researcher with expertise in neural architecture search.",
        "You are a data scientist specializing in large-scale data pipelines and feature engineering.",
        "You are an AI security researcher with expertise in model robustness and adversarial evaluation.",
    ],
    "code": [
        "Act as an expert software developer specializing in performance-critical systems code.",
        "Act as a senior backend engineer with expertise in high-throughput API design.",
        "Act as a database architect with deep knowledge of query optimization and indexing strategies.",
        "Act as a compiler engineer with expertise in LLVM passes and code generation.",
        "Act as a DevOps engineer specializing in container orchestration and infrastructure-as-code.",
        "Act as a security engineer with expertise in cryptographic protocol implementation.",
        "Act as a graphics programmer with expertise in GPU shader and compute pipeline optimization.",
        "Act as a distributed systems engineer with expertise in consensus algorithms and fault tolerance.",
    ],
    "math": [
        "You are a mathematics professor specializing in abstract algebra and number theory.",
        "You are a computational mathematician with expertise in numerical analysis and approximation theory.",
        "You are a geometry researcher specializing in differential and Riemannian geometry.",
        "You are a probability theorist with deep expertise in stochastic processes and martingales.",
        "You are a mathematical logician with expertise in model theory and proof systems.",
        "You are a combinatorics researcher specializing in graph theory and extremal combinatorics.",
        "You are a mathematical physicist with expertise in variational methods and PDE analysis.",
        "You are an optimization researcher specializing in convex analysis and duality theory.",
    ],
    "stem": [
        "You are a quantum physicist specializing in quantum information theory and entanglement.",
        "You are a materials scientist with expertise in solid-state physics and crystal structure analysis.",
        "You are a chemical engineer specializing in reaction kinetics and thermodynamic process design.",
        "You are an electrical engineer with deep expertise in signal processing and control theory.",
        "You are a computational biologist specializing in protein structure prediction and genomics.",
        "You are an astrophysicist with expertise in stellar evolution and cosmological simulations.",
        "You are a neuroscientist specializing in computational models of neural circuits.",
        "You are a mechanical engineer with expertise in fluid dynamics and finite element analysis.",
    ],
}

# ─── Few-shot seed selection strategy per domain ──────────────────────────────
DOMAIN_SEED_CONFIG = {
    "chat": {"filter_kw": "helpful and harmless",
             "sort_asc": True, "n_shots": 3},
    "code": {"filter_kw": ["expert software developer", "SQL assistant"],
             "sort_asc": True, "n_shots": 3},
    "math": {"filter_kw": "integrate natural language reasoning",
             "sort_asc": False, "n_shots": 3},   # highest blue_ratio → most formal
    "stem": {"filter_kw": "integrate natural language reasoning",
             "sort_asc": True, "n_shots": 3},
}

# ─── Domain task descriptions ─────────────────────────────────────────────────
DOMAIN_TASK_DESC = {
    "chat": (
        "a technical chat conversation involving computing, networking, AI, or ML topics. "
        "Use domain-specific terminology (e.g., MoE, LLM, SDR, quantization). "
        "Avoid casual greetings, weather comments, or everyday conversational filler."
    ),
    "code": (
        "a programming conversation that uses uncommon operators or language features "
        "(bitwise, ternary, async/await), specific library or framework functions, and "
        "long descriptive identifier names. Avoid generic terms like 'variable', 'function', or 'class'."
    ),
    "math": (
        "a mathematics problem-solving conversation. Include multi-step algebraic, geometric, "
        "trigonometric, or calculus reasoning. Emphasize structural/symmetry arguments, not just "
        "procedural arithmetic. Use varied sub-domains (algebra, geometry, calculus) in a single sample."
    ),
    "stem": (
        "a STEM (Science, Technology, Engineering) factual Q&A. State facts concisely and directly "
        "using precise technical terminology. Avoid hedging words (may, might, possibly), "
        "redundant details, or excessive background explanation."
    ),
}


# ─── Helpers ──────────────────────────────────────────────────────────────────

def slack_notify(message: str) -> None:
    try:
        requests.post(SLACK_WEBHOOK, json={"text": message}, timeout=10)
    except Exception as e:
        print(f"[Slack] Failed: {e}")


def log(step: str, status: str, detail: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"\n{'='*60}\n[{ts}][{step}] {status}\n{detail}\n{'='*60}\n", flush=True)


def strip_color_tags(text: str) -> str:
    text = re.sub(r'<red>(.*?)</red>', r'\1', text, flags=re.DOTALL)
    text = re.sub(r'<blue>(.*?)</blue>', r'\1', text, flags=re.DOTALL)
    return text


def load_guideline(domain: str, guideline_type: str) -> str:
    filename = (
        f"balance_guideline_{domain}_nemotron-3-super.md"
        if guideline_type == "balance"
        else f"q_sensitivity_guideline_{domain}_nemotron-3-super.md"
    )
    with open(os.path.join(INSTRUCTION_DIR, filename), encoding="utf-8") as f:
        return f.read().strip()


def load_seed_records() -> list:
    with open(SEED_DATA_PATH, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def count_tokens_batch(texts: list, model_name: str, base_url: str) -> list:
    """Count tokens via vLLM /tokenize endpoint (no /v1 prefix)."""
    tokenize_url = base_url.split("/v1")[0] + "/tokenize"
    counts = []
    for text in texts:
        try:
            resp = requests.post(
                tokenize_url,
                json={"model": model_name, "prompt": text},
                timeout=30,
            )
            counts.append(resp.json().get("count", -1))
        except Exception:
            counts.append(-1)
    return counts


def select_few_shot(domain: str, seed_records: list) -> list:
    cfg = DOMAIN_SEED_CONFIG[domain]
    kws = cfg["filter_kw"]
    if isinstance(kws, str):
        kws = [kws]

    def matches(r):
        plain = strip_color_tags(r["text"])[:400]
        return any(kw in plain for kw in kws)

    subset = [r for r in seed_records if matches(r)]
    subset.sort(
        key=lambda r: r["n_tokens_blue"] / max(r["n_tokens_total"], 1),
        reverse=not cfg["sort_asc"],
    )
    chosen = subset[: cfg["n_shots"]]
    examples = []
    for r in chosen:
        clean = strip_color_tags(r["text"])
        # Double-escape braces so nested LaTeX like \frac{a}{b} doesn't crash
        # data-designer's Formatter().parse() validation step
        clean = clean.replace("{", "{{").replace("}", "}}")
        examples.append(clean[:900])
    return examples, list(chosen)   # (examples, raw_records) for logging


def build_prompt(
    domain: str,
    guideline_type: str,
    guideline_content: str,
    few_shot_examples: list,
) -> str:
    few_shot_block = "\n\n".join(
        f"--- Reference Example {i+1} ---\n{ex}\n[... end of example ...]"
        for i, ex in enumerate(few_shot_examples)
    )
    task_desc = DOMAIN_TASK_DESC[domain]
    gl_label = (
        "BALANCE  (Expert Activation Balance)"
        if guideline_type == "balance"
        else "Q-SENSITIVITY  (Quantization Sensitivity)"
    )

    return f"""\
You are a data generation assistant creating synthetic conversation samples \
for LLM quantization calibration. Produce a NEW, ORIGINAL conversation \
following the style of the reference examples and the guideline below.

════════════════════════════════════════════
REFERENCE EXAMPLES  (style guide — do NOT copy content)
════════════════════════════════════════════
{{% raw %}}
{few_shot_block}
{{% endraw %}}

════════════════════════════════════════════
GUIDELINE — {gl_label}
Domain: {domain.upper()}
════════════════════════════════════════════
{guideline_content}

════════════════════════════════════════════
TASK
════════════════════════════════════════════
Domain   : {{{{ domain }}}}
Persona  : {{{{ system_style }}}}

Generate {task_desc}

Apply the guideline above when writing every section.

Use the following format:

### System:
[Start this section with the Persona line above, then add any additional \
role-appropriate instructions]

### User:
[user request or question]

### Assistant:
[assistant response following the guideline]

IMPORTANT: Output ONLY the conversation using the ### markers above. \
No other text before or after.
"""


# ─── Per-combo pipeline runner ────────────────────────────────────────────────

def run_combo(
    guideline_type: str,
    domain: str,
    vllm_model_name: str,
    seed_records: list,
    data_designer,
    ModelConfig,
    ChatCompletionInferenceParams,
    DataDesignerConfigBuilder,
    SamplerColumnConfig,
    SamplerType,
    CategorySamplerParams,
    LLMTextColumnConfig,
    ExpressionColumnConfig,
) -> dict:
    tag = f"{guideline_type}/{domain}"
    output_dir = os.path.join(OUTPUT_ROOT, guideline_type, domain)
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'#'*60}\n  COMBO: {tag.upper()}\n{'#'*60}", flush=True)

    # Load guideline + build few-shot
    guideline_content     = load_guideline(domain, guideline_type)
    few_shots, raw_chosen = select_few_shot(domain, seed_records)
    prompt                = build_prompt(domain, guideline_type, guideline_content, few_shots)

    blue_ratios = [round(r["n_tokens_blue"] / r["n_tokens_total"], 4) for r in raw_chosen]
    log(f"{tag}/setup", "✅",
        f"Guideline: {len(guideline_content)} chars | "
        f"Few-shot: {len(few_shots)} examples (blue_ratios: {blue_ratios})")

    # Model config
    gen_model = ModelConfig(
        alias=f"gen-{guideline_type}-{domain}",
        model=vllm_model_name,
        provider="vllm-local",
        inference_parameters=ChatCompletionInferenceParams(
            temperature=0.85,
            top_p=0.95,
            max_tokens=2048,
            max_parallel_requests=MAX_PARALLEL_REQUESTS,
        ),
    )

    # Columns: domain (fixed) + system_style (diverse) + seed_augmentation + quality_score
    styles = SYSTEM_STYLES[domain]
    domain_col = SamplerColumnConfig(
        name="domain",
        sampler_type=SamplerType.CATEGORY,
        params=CategorySamplerParams(values=[domain], weights=[1.0]),
    )
    style_col = SamplerColumnConfig(
        name="system_style",
        sampler_type=SamplerType.CATEGORY,
        params=CategorySamplerParams(
            values=styles,
            weights=[1.0 / len(styles)] * len(styles),
        ),
    )
    text_col = LLMTextColumnConfig(
        name="seed_augmentation",
        prompt=prompt,
        model_alias=f"gen-{guideline_type}-{domain}",
    )
    score_col = ExpressionColumnConfig(name="quality_score", expr="5.0")

    config = DataDesignerConfigBuilder(model_configs=[gen_model])
    config.add_column(domain_col)
    config.add_column(style_col)
    config.add_column(text_col)
    config.add_column(score_col)

    data_designer.validate(config)

    # Preview
    slack_notify(f"🔄 [{tag}] Preview 실행 중 (5개)...")
    try:
        preview = data_designer.preview(config_builder=config, num_records=5)
        preview_df = preview.dataset
        preview_df.to_json(
            f"{output_dir}/preview_{domain}.json",
            orient="records", force_ascii=False, indent=2,
        )
        log(f"{tag}/preview", "✅ SUCCESS", f"{len(preview_df)} records")
        slack_notify(f"✅ [{tag}] Preview 완료 ({len(preview_df)}/5)")
    except Exception as e:
        log(f"{tag}/preview", "❌ FAILED", f"{e}\n{traceback.format_exc()}")
        slack_notify(f"❌ [{tag}] Preview 실패\n{str(e)[:300]}")
        raise

    # Full generation
    slack_notify(f"🔄 [{tag}] 본 생성 시작 ({NUM_RECORDS_PER_COMBO}개 → 목표 {TARGET_RECORDS_PER_COMBO}개)...")
    try:
        gen_results = data_designer.create(
            config_builder=config,
            num_records=NUM_RECORDS_PER_COMBO,
            dataset_name=f"calibration_{guideline_type}_{domain}",
        )
        df_raw = gen_results.load_dataset()
        df_raw.to_parquet(f"{output_dir}/{domain}_raw.parquet", index=False)
        df_raw.to_json(f"{output_dir}/{domain}_raw.json",
                       orient="records", force_ascii=False, indent=2)

        # Filter empty rows + count tokens
        df_clean = df_raw[df_raw["seed_augmentation"].str.strip().astype(bool)].reset_index(drop=True)
        print(f"  [{tag}] Counting tokens for {len(df_clean)} clean records...", flush=True)
        df_clean = df_clean.copy()
        df_clean["n_tokens"] = count_tokens_batch(
            df_clean["seed_augmentation"].tolist(), vllm_model_name, VLLM_BASE_URL
        )
        df_final = df_clean.iloc[:TARGET_RECORDS_PER_COMBO].copy()

        df_clean.to_parquet(f"{output_dir}/{domain}_clean.parquet", index=False)
        df_clean.to_json(f"{output_dir}/{domain}_clean.json",
                         orient="records", force_ascii=False, indent=2)
        df_final.to_parquet(f"{output_dir}/{domain}_final.parquet", index=False)
        df_final.to_json(f"{output_dir}/{domain}_final.json",
                         orient="records", force_ascii=False, indent=2)

        lengths  = df_final["seed_augmentation"].str.len()
        n_tokens = df_final["n_tokens"]
        detail = (
            f"raw={len(df_raw)} | valid={len(df_clean)} | final={len(df_final)}/{TARGET_RECORDS_PER_COMBO}\n"
            f"chars : min={lengths.min()} / mean={lengths.mean():.0f} / max={lengths.max()}\n"
            f"tokens: min={n_tokens.min()} / mean={n_tokens.mean():.0f} / max={n_tokens.max()}"
        )
        log(f"{tag}/generate", "✅ SUCCESS", detail)
        slack_notify(
            f"✅ [{tag}] 생성 완료!\n"
            f"raw={len(df_raw)} | valid={len(df_clean)} | final={len(df_final)}/{TARGET_RECORDS_PER_COMBO}\n"
            f"avg_tokens={n_tokens.mean():.0f} | avg_chars={lengths.mean():.0f}"
        )
        return {
            "guideline_type": guideline_type,
            "domain": domain,
            "n_raw": len(df_raw),
            "n_valid": len(df_clean),
            "n_final": len(df_final),
            "df_final": df_final,
        }

    except Exception as e:
        log(f"{tag}/generate", "❌ FAILED", f"{e}\n{traceback.format_exc()}")
        slack_notify(f"❌ [{tag}] 생성 실패\n{str(e)[:500]}")
        raise


# ─── Main ─────────────────────────────────────────────────────────────────────

print("\n[Step 1] Verifying vLLM connection...")
try:
    resp = requests.get(f"{VLLM_BASE_URL}/models", timeout=10)
    model_ids = [m["id"] for m in resp.json().get("data", [])]
    VLLM_MODEL_NAME = model_ids[0]
    os.environ["NVIDIA_BASE_URL"] = VLLM_BASE_URL
    os.environ["NVIDIA_API_KEY"] = "not-used"
    log("Step 1", "✅ SUCCESS", f"Model: {VLLM_MODEL_NAME}")
    slack_notify(
        f"🚀 [Per-Domain Calibration Pipeline v2] 시작\n"
        f"모델: {VLLM_MODEL_NAME}\n"
        f"조합: {GUIDELINE_TYPES} × {DOMAINS} = {len(GUIDELINE_TYPES)*len(DOMAINS)}개\n"
        f"목표: {TARGET_RECORDS_PER_COMBO}개 × 8 = {TARGET_RECORDS_PER_COMBO*8}개 총"
    )
except Exception as e:
    log("Step 1", "❌ FAILED", str(e))
    raise

print("\n[Step 2] Loading seed data...")
seed_records = load_seed_records()
log("Step 2", "✅ SUCCESS", f"Loaded {len(seed_records)} seed records")

print("\n[Step 3] Importing data-designer...")
from data_designer.interface import DataDesigner
from data_designer.config import (
    CategorySamplerParams,
    ChatCompletionInferenceParams,
    DataDesignerConfigBuilder,
    ExpressionColumnConfig,
    LLMTextColumnConfig,
    ModelConfig,
    ModelProvider,
    SamplerColumnConfig,
    SamplerType,
)
log("Step 3", "✅ SUCCESS", "data-designer imported")

os.makedirs(OUTPUT_ROOT, exist_ok=True)
for gl in GUIDELINE_TYPES:
    for d in DOMAINS:
        os.makedirs(os.path.join(OUTPUT_ROOT, gl, d), exist_ok=True)

import pandas as pd

def _combo_already_done(gl: str, d: str) -> bool:
    """Return True if this combo's final parquet already exists (skip re-run)."""
    path = os.path.join(OUTPUT_ROOT, gl, d, f"{d}_final.parquet")
    return os.path.exists(path)

COMBINATIONS = [
    (gl, d)
    for gl in GUIDELINE_TYPES
    for d in DOMAINS
    if not _combo_already_done(gl, d)
]
_skipped = [(gl, d) for gl in GUIDELINE_TYPES for d in DOMAINS if _combo_already_done(gl, d)]
if _skipped:
    print(f"  [skip] Already completed: {_skipped}")
print(f"  [run]  Remaining combos: {COMBINATIONS}")


def _run_combo_thread(guideline_type: str, domain: str) -> dict:
    """Thread-safe: each combo gets its own DataDesigner + ModelProvider."""
    provider = ModelProvider(
        name="vllm-local",
        endpoint=VLLM_BASE_URL,
        provider_type="openai",
        api_key="not-used",
    )
    designer = DataDesigner(model_providers=[provider])
    return run_combo(
        guideline_type=guideline_type,
        domain=domain,
        vllm_model_name=VLLM_MODEL_NAME,
        seed_records=seed_records,
        data_designer=designer,
        ModelConfig=ModelConfig,
        ChatCompletionInferenceParams=ChatCompletionInferenceParams,
        DataDesignerConfigBuilder=DataDesignerConfigBuilder,
        SamplerColumnConfig=SamplerColumnConfig,
        SamplerType=SamplerType,
        CategorySamplerParams=CategorySamplerParams,
        LLMTextColumnConfig=LLMTextColumnConfig,
        ExpressionColumnConfig=ExpressionColumnConfig,
    )


print(f"\n[Step 4] Running {len(COMBINATIONS)} combos in parallel "
      f"(max_parallel_requests={MAX_PARALLEL_REQUESTS} per combo)...")

all_results: dict = {}
errors: list = []

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    future_map = {
        executor.submit(_run_combo_thread, gl, d): (gl, d)
        for gl, d in COMBINATIONS
    }
    for future in as_completed(future_map):
        gl, d = future_map[future]
        tag = f"{gl}/{d}"
        try:
            result = future.result()
            all_results[(gl, d)] = result
            print(f"  ✅ [{tag}] done: {result['n_final']}/{TARGET_RECORDS_PER_COMBO} records",
                  flush=True)
        except Exception as e:
            errors.append((tag, str(e)))
            print(f"  ❌ [{tag}] FAILED: {e}", flush=True)
            slack_notify(f"❌ [{tag}] 실패\n{str(e)[:300]}")

# ─── Combine per guideline type ───────────────────────────────────────────────
print("\n[Final] Combining datasets per guideline type...")

for gl_type in GUIDELINE_TYPES:
    combo_results = [all_results[(gl_type, d)] for d in DOMAINS if (gl_type, d) in all_results]
    if not combo_results:
        print(f"  ⚠️  No results for guideline_type={gl_type}, skipping combine.")
        continue
    df_combined = pd.concat([r["df_final"] for r in combo_results], ignore_index=True)
    out_path = f"{OUTPUT_ROOT}/calibration_{gl_type}_all"
    df_combined.to_parquet(f"{out_path}.parquet", index=False)
    df_combined.to_json(f"{out_path}.json", orient="records", force_ascii=False, indent=2)
    print(f"  [{gl_type}] combined: {len(df_combined)} records → {out_path}.parquet")

# ─── Summary ──────────────────────────────────────────────────────────────────
summary_lines = [f"{'GL-TYPE':12s} {'DOMAIN':6s}  RAW  VALID  FINAL"]
total_final = 0
for gl in GUIDELINE_TYPES:
    for d in DOMAINS:
        if (gl, d) in all_results:
            r = all_results[(gl, d)]
            summary_lines.append(
                f"  {gl:12s} {d:6s}  {r['n_raw']:3d}   {r['n_valid']:3d}   {r['n_final']:3d}"
            )
            total_final += r["n_final"]
        else:
            summary_lines.append(f"  {gl:12s} {d:6s}  FAILED")
summary_lines.append(f"\n  TOTAL: {total_final} records across {len(all_results)} combos")
if errors:
    summary_lines.append(f"\n  ERRORS ({len(errors)}): {[t for t, _ in errors]}")
summary = "\n".join(summary_lines)

log("Final", "✅ PIPELINE COMPLETE" if not errors else "⚠️ PARTIAL", summary)
slack_notify(
    f"{'✅' if not errors else '⚠️'} [Per-Domain Pipeline v2] 완료!\n{summary}"
)

print("\n✅ Per-domain calibration pipeline v2 completed!")
