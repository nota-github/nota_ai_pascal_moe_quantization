# MoE Quantization Calibration Pipeline

A four-stage pipeline for quantizing Mixture-of-Experts (MoE) LLMs with expert-routing-aware calibration data. The pipeline analyzes how tokens are routed to experts, extracts activation patterns, and synthesizes calibration datasets that improve quantization quality by balancing expert activation.

## Overview

```
Stage 0: Quantize initial model and evaluate on benchmark tasks
    ↓  (produces: calib dataset, quantized checkpoint, TRT engine, benchmark results)
Stage 1: Analyze activated experts statistics from calibration data
    ↓  (produces: token routing maps, expert activation distributions, bracketed samples)
Stage 2: Extract text patterns causing frequent/scarce experts
    ↓  (produces: per-domain guidelines for sparse/dense expert activation)
Stage 3: Generate synthetic calibration dataset achieving balanced activated experts
    ↓  (produces: synthetic calibration samples optimized for expert balance)
```

> **Iterative refinement**: Repeat Stages 1–3 until the compound dataset (original + synthesized) achieves balanced expert activation across all layers.

| Stage | Name | venv | Key Tool |
|-------|------|------|----------|
| 0 | Quantize initial model and evaluate on benchmark tasks | `stage0` | NVIDIA ModelOpt + TensorRT-LLM |
| 1 | Analyze activated experts statistics from calibration data | `stage1` | PyTorch + Transformers |
| 2 | Extract text patterns causing frequent/scarce experts | `stage2` | OpenAI-compatible API (NIM/vLLM) |
| 3 | Generate synthetic calibration dataset achieving balanced activated experts | `stage3` | NVIDIA Data Designer + vLLM |

---

## Demo

You can run the hosted PASCAL-MoE demo [here](https://capability-wichita-flight-demonstrated.trycloudflare.com/).

You can review the entire execution process of the PASCAL-MoE pipeline here. While live execution is supported, the current video demonstrates the process by loading pre-recorded execution logs and results.

<video src="https://media.githubusercontent.com/media/nota-github/nota_ai_pascal_moe_quantization/main/pascal_moe_run_video.mov" controls width="100%"></video>

---

## Repository Structure

```
.
├── README.md
├── .gitignore
├── scripts/                         # One bash script per stage — single entry point
│   ├── run_stage0.sh
│   ├── run_stage1.sh
│   ├── run_stage2.sh
│   └── run_stage3.sh
├── src/                             # Python source files called by the scripts
│   ├── stage0/
│   │   ├── step1_gptq_quantize.py           # Dataset gen + Hessian + GPTQ quantization
│   │   ├── step2_trtllm_build_serve.py      # TRT-LLM engine build & server
│   │   ├── step3_nemo_eval.py               # NeMo Evaluator benchmarks
│   │   ├── nemotron_post_training_calib.py  # Calibration dataset builder
│   │   └── modelopt_parallel_gptq.py        # Parallel GPTQ patch utilities
│   ├── stage1/
│   │   ├── step1_dataset_load.py            # (Optional) custom dataset loading
│   │   ├── step2_count_expert.py            # Token → expert routing
│   │   ├── step3_count_expert_dist.py       # Expert activation distribution
│   │   ├── step4_weight_outlier_dist.py     # Expert weight sensitivity
│   │   ├── step5_sort_token_plot.py         # Token classification scatter plots
│   │   └── step6_apply_bracket.py           # Apply <red>/<blue> token tags
│   ├── stage2/
│   │   └── extract_pattern_agent_balanced.py  # LLM-based pattern extraction
│   └── stage3/
│       └── pipeline_calibration_per_domain.py  # Parallel synthetic data generation
├── requirements/                    # Per-stage pip requirements
│   ├── stage0.txt
│   ├── stage1.txt
│   ├── stage2.txt
│   └── stage3.txt
├── results/                         # (generated, gitignored)
│   ├── stage1_routing/              # Stage 1: expert analysis output
│   ├── stage2_guidelines/           # Stage 2: per-domain instruction files
│   └── stage3_dataset/              # Stage 3: synthetic calibration datasets
└── model/                           # (generated, gitignored)
    ├── calib_dataset/               # Stage 0: calibration dataset
    ├── quantized/                   # Stage 0: quantized TRT-LLM checkpoint
    ├── trt_engine/                  # Stage 0: TensorRT-LLM engine
    └── eval_results/                # Stage 0: benchmark results & logs
```

---

## Prerequisites

- **GPU**: NVIDIA H100 (recommended) or equivalent with CUDA 12.x
- **Python**: 3.12
- **OS**: Linux
- **Model**: [Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B) (or any compatible MoE model)
- **Stages 2 & 3**: A running vLLM or NVIDIA NIM server (e.g., `nemotron-3-super` at `http://localhost:8000/v1`)

---

## Quick Start

```bash
# Clone and enter the repo
git clone https://github.com/nota-github/nota_ai_pascal_moe_quantization.git
cd nota_ai_pascal_moe_quantization

# Set your model path in each script before running:
#   scripts/run_stage0.sh  →  MODEL_BASE_PATH
#   scripts/run_stage1.sh  →  MODEL_BASE_PATH

bash scripts/run_stage0.sh   # Quantize + evaluate
bash scripts/run_stage1.sh   # Analyze expert routing
bash scripts/run_stage2.sh   # Extract activation patterns
bash scripts/run_stage3.sh   # Generate synthetic dataset
```

---

## Stage 0: Quantize initial model and evaluate on benchmark tasks

**Script**: `scripts/run_stage0.sh`  
**Source**: `src/stage0/`  
**Requirements**: `requirements/stage0.txt`

### What it does

1. Generates a calibration dataset from NVIDIA Nemotron Post-Training Dataset v1
2. Computes GPTQ Hessians for accurate weight quantization
3. Applies GPTQ W4A16 quantization using NVIDIA ModelOpt
4. Builds a TensorRT-LLM engine from the quantized checkpoint
5. Runs NeMo Evaluator benchmarks (GSM8K, GPQA Diamond, MMLU-Pro)

### Environment Setup

Stage 0 requires NVIDIA-specific packages. Use `uv` for installation:

```bash
uv venv --python 3.12 --seed
source .venv/bin/activate
uv pip install tensorrt-llm --extra-index-url https://pypi.nvidia.com
uv pip install nvidia-modelopt[torch] --extra-index-url https://pypi.nvidia.com
uv pip install nvidia-lm-eval --extra-index-url https://pypi.nvidia.com
uv pip install math_verify
pip install -r requirements/stage0.txt --extra-index-url https://pypi.nvidia.com
```

### Configuration

Edit the configuration section at the top of `scripts/run_stage0.sh`:

```bash
MODEL_NAME="qwen3_30b_a3b"
MODEL_BASE_PATH="/your_base_model_path"   # <-- set this

QUANTIZE="GPTQ_W4A16"                    # UNQUANTIZED | GPTQ_W4A16 | RTN_W4A16
FULL_GPU_DEVICES="0,1,2,3,4,5,6,7"      # GPUs for Hessian + GPTQ
TARGET_GPU_DEVICES="7"                   # GPU for engine build, serve, eval
```

### Run

```bash
source .venv/bin/activate
bash scripts/run_stage0.sh
```

### Output

| Path | Content |
|------|---------|
| `model/calib_dataset/D0_128/` | Calibration dataset (Arrow, JSONL, Hessians) |
| `model/quantized/<Q_MODEL_NAME>/` | Quantized TRT-LLM checkpoint |
| `model/trt_engine/<ENGINE_NAME>/` | TensorRT-LLM engine |
| `model/eval_results/` | Benchmark results (JSON) and execution logs |

---

## Stage 1: Analyze activated experts statistics from calibration data

**Script**: `scripts/run_stage1.sh`  
**Source**: `src/stage1/`  
**Requirements**: `requirements/stage1.txt`

### What it does

Runs a sequential 6-step analysis pipeline measuring how calibration tokens activate MoE experts:

| Sub-step | Script | Description |
|----------|--------|-------------|
| Step 2 | `step2_count_expert.py` | Forward-pass calibration samples; record per-token expert selection |
| Step 3a/b | `step3_count_expert_dist.py` | Compute per-expert activation frequency (with/without threshold) |
| Step 4a/b | `step4_weight_outlier_dist.py` | Measure weight outlier sensitivity per expert (with/without threshold) |
| Step 5a/b | `step5_sort_token_plot.py` | Sort tokens by balance/Q-sensitivity and plot scatter (with/without token threshold) |
| Step 6 | `step6_apply_bracket.py` | Tag tokens with `<red>`/`<blue>` based on sparse/dense expert activation |

### Environment Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements/stage1.txt
```

### Configuration

Edit the configuration section at the top of `scripts/run_stage1.sh`:

```bash
MODEL_BASE_PATH="/your_base_model_path"   # <-- set this
MODEL_NAME="qwen3_30b_a3b"

# Expert thresholds (sub-steps 3b, 4b)
freq_thr=0.25;  less_thr=0.25
rob_thr=0.25;   sen_thr=0.25

# Token thresholds (sub-step 5b)
balance_thr_blue_x=0.1;  balance_thr_blue_y=0.77
balance_thr_red_x=0.15;  balance_thr_red_y=0.6
```

### Run

```bash
source .venv/bin/activate
bash scripts/run_stage1.sh
```

### Output

| Path | Content |
|------|---------|
| `results/stage1_routing/<MODEL>_<DATASET>/D0_128/s2_expert_count/token_routing.jsonl` | Per-token expert routing data |
| `results/stage1_routing/<MODEL>_<DATASET>/D0_128/s3_expert_dist/` | Expert activation distribution plots and JSON |
| `results/stage1_routing/<MODEL>_<DATASET>/D0_128/s4_weight_outlier/` | Weight sensitivity plots and JSON |
| `results/stage1_routing/<MODEL>_<DATASET>/D0_128/s5_sorted_token/` | Token classification scatter plots and JSON |
| `results/stage1_routing/<MODEL>_<DATASET>/D0_128/s6_apply_bracket/bracketed_balance.jsonl` | Annotated samples with `<red>`/`<blue>` tags → **input to Stage 2** |

---

## Stage 2: Extract text patterns causing frequent/scarce experts

**Script**: `scripts/run_stage2.sh`  
**Source**: `src/stage2/`  
**Requirements**: `requirements/stage2.txt`

### What it does

Feeds bracketed samples (from Stage 1) to a reasoning LLM (Nemotron-3-Super) to extract activation patterns. For each domain (`chat`, `code`, `stem`, `math`), it analyzes which token patterns activate sparsely-used experts (`<red>`) vs. frequently-used experts (`<blue>`), and produces per-domain guidelines for Stage 3.

### Prerequisites

A running NIM or vLLM server:

```bash
vllm serve <model-path> --port 8000
```

### Environment Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements/stage2.txt
```

### Run

```bash
source .venv/bin/activate
bash scripts/run_stage2.sh
```

Input and output paths are resolved automatically from the repo structure. Override them at the top of `scripts/run_stage2.sh` if needed.

### Output

| Path | Content |
|------|---------|
| `results/stage2_guidelines/instruction/balance_guideline_{domain}_nemotron-3-super.md` | Per-domain balance guidelines → **input to Stage 3** |

---

## Stage 3: Generate synthetic calibration dataset achieving balanced activated experts

**Script**: `scripts/run_stage3.sh`  
**Source**: `src/stage3/`  
**Requirements**: `requirements/stage3.txt`

### What it does

Generates 8 synthetic calibration datasets (2 guideline types × 4 domains) using NVIDIA Data Designer, with all 8 combinations running in parallel:

| Guideline Type | Domains | Target Records |
|----------------|---------|----------------|
| `balance` | chat, code, math, stem | 128 per domain |
| `q_sensitivity` | chat, code, math, stem | 128 per domain |

### Prerequisites

- Running vLLM or NIM server (same as Stage 2)
- Stage 1 output: `bracketed_balance.jsonl`
- Stage 2 output: `instruction/` directory with guideline files

### Environment Setup

```bash
# Reuse the Stage 2 venv, or create a new one:
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements/stage3.txt
```

### Run

```bash
source .venv/bin/activate
bash scripts/run_stage3.sh
```

### Output

| Path | Content |
|------|---------|
| `results/stage3_dataset/output_per_domain/balance/{domain}/` | Per-domain balance datasets |
| `results/stage3_dataset/output_per_domain/q_sensitivity/{domain}/` | Per-domain Q-sensitivity datasets |
| `results/stage3_dataset/output_per_domain/calibration_balance_all.parquet` | Merged balance dataset (all domains) |
| `results/stage3_dataset/output_per_domain/calibration_q_sensitivity_all.parquet` | Merged Q-sensitivity dataset (all domains) |

---

## Pipeline Data Flow

```
model/calib_dataset/D0_128/              ← Stage 0 output
  ├── data-*.arrow
  ├── samples_text.jsonl
  └── calib_chunks.pt
        │
        ▼
results/stage1_routing/.../s6_apply_bracket/
  └── bracketed_balance.jsonl            ← Stage 1 output / Stage 2 & 3 input
        │
        ├──────────────────────────────────────┐
        ▼                                      ▼
results/stage2_guidelines/instruction/   results/stage3_dataset/ (seed)
  └── balance_guideline_{domain}_*.md    └── bracketed_balance.jsonl
        │
        ▼
results/stage3_dataset/output_per_domain/
  └── calibration_{balance,q_sensitivity}_all.parquet   ← Final output
```

---

## Notes

- **Model paths**: Set `MODEL_BASE_PATH` in `scripts/run_stage0.sh` and `scripts/run_stage1.sh` to your local model directory. All other paths are resolved automatically relative to the repository root.
- **results/ and model/**: Both directories are gitignored. They are created automatically when the scripts run.
- **GPU memory**: Stage 0 (GPTQ) benefits from multi-GPU. Stage 1 runs on a single GPU. Stages 2 & 3 are CPU-bound (API calls to a separately-served model).
- **Slack notifications** (Stage 3): Set `SLACK_WEBHOOK_URL` environment variable to enable progress notifications.
- **Calibration dataset**: Uses [nvidia/Nemotron-Post-Training-Dataset-v1](https://huggingface.co/datasets/nvidia/Nemotron-Post-Training-Dataset-v1) by default. Modify `src/stage0/step1_gptq_quantize.py` or `src/stage1/step1_dataset_load.py` to use a custom dataset.
