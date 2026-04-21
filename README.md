# MoE Quantization Calibration Pipeline

A four-stage pipeline for quantizing Mixture-of-Experts (MoE) LLMs with expert-routing-aware calibration data. The pipeline analyzes how tokens are routed to experts, extracts activation patterns, and synthesizes calibration datasets that improve quantization quality by balancing expert activation.

## Overview

```
Stage 0: Quantize initial model and evaluate on benchmark tasks
    ↓  (produces: calibration dataset, quantized checkpoint, benchmark results)
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
| 0 | Quantize initial model and evaluate on benchmark tasks | `quant_expert_analysis` | NVIDIA ModelOpt + TensorRT-LLM |
| 1 | Analyze activated experts statistics from calibration data | `quant_expert_analysis` | PyTorch + Transformers |
| 2 | Extract text patterns causing frequent/scarce experts | `nemo_data_designer` | OpenAI-compatible API (NIM/vLLM) |
| 3 | Generate synthetic calibration dataset achieving balanced activated experts | `nemo_data_designer` | NVIDIA Data Designer + vLLM |

---

## Prerequisites

- **GPU**: NVIDIA H100 (recommended) or equivalent with CUDA 12.x
- **Python**: 3.12
- **OS**: Linux
- **Model**: [Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B) (or any compatible MoE model)
- **Serving**: A running vLLM or NVIDIA NIM server for Stages 2 & 3 (e.g., `nemotron-3-super`)

---

## Stage 0: Quantize initial model and evaluate on benchmark tasks

**Directory**: `stage_0_quantize/`  
**Virtual environment**: `quant_expert_analysis`  
**Run script**: `stage_0_quantize/run_pipeline.sh`

### What it does

1. Generates a calibration dataset from NVIDIA Nemotron Post-Training Dataset v1
2. Computes GPTQ Hessians for accurate weight quantization
3. Applies GPTQ W4A16 quantization using NVIDIA ModelOpt
4. Builds a TensorRT-LLM engine from the quantized checkpoint
5. Runs NeMo Evaluator benchmarks (GSM8K, GPQA Diamond, MMLU-Pro)

### Environment Setup

Stage 0 requires NVIDIA-specific packages from the NVIDIA PyPI index. Use `uv` for fast installation:

```bash
# Install uv if not already installed
pip install uv

# Create virtual environment
uv venv --python 3.12 --seed
source .venv/bin/activate

# Install NVIDIA packages (order matters)
uv pip install tensorrt-llm --extra-index-url https://pypi.nvidia.com
uv pip install nvidia-modelopt[torch] --extra-index-url https://pypi.nvidia.com
uv pip install nvidia-lm-eval --extra-index-url https://pypi.nvidia.com
uv pip install math_verify

# Install remaining dependencies
pip install -r stage_0_quantize/requirements.txt --extra-index-url https://pypi.nvidia.com
```

### Configuration

Edit the top section of `stage_0_quantize/run_pipeline.sh` to match your environment:

```bash
STAGE="0"               # 0 = also generate calibration dataset
CALIB_SIZE=128          # number of calibration samples
QUANTIZE="GPTQ_W4A16"  # UNQUANTIZED | GPTQ_W4A16 | RTN_W4A16

FULL_GPU_DEVICES="0,1,2,3,4,5,6,7"  # GPUs for Hessian + GPTQ
TARGET_GPU_DEVICES="7"               # GPU for engine build, serve, eval

MODEL_NAME="qwen3_30b_a3b"
MODEL_BASE_PATH="/your_base_model_path"   # <-- set this
```

### Run

```bash
cd stage_0_quantize
source .venv/bin/activate
bash run_pipeline.sh
```

### Output

| Path | Content |
|------|---------|
| `stage_0_quantize/dataset_dir/D0_128/` | Calibration dataset (Arrow, JSONL, Hessians) |
| `models/q_models/<Q_MODEL_NAME>/` | Quantized TRT-LLM checkpoint |
| `models/trt_engines/<ENGINE_NAME>/` | TensorRT-LLM engine |
| `eval_results/` | Benchmark results (JSON) |
| `log/` | Step-by-step execution logs |

---

## Stage 1: Analyze activated experts statistics from calibration data

**Directory**: `stage_1_analyze_routing/`  
**Virtual environment**: `quant_expert_analysis` (same as Stage 0)  
**Run script**: `stage_1_analyze_routing/run_pipeline.sh`

### What it does

Runs a 4-step analysis pipeline on how tokens from the calibration dataset activate MoE experts:

| Sub-step | Script | Description |
|----------|--------|-------------|
| Step 2 | `step2_count_expert.py` | Forward-passes calibration samples through the model; records which expert each token selects |
| Step 3 | `step3_count_expert_dist.py` | Computes per-expert activation frequency distribution |
| Step 4 | `step4_weight_outlier_dist.py` | Measures weight outlier sensitivity per expert |
| Step 5 | `step5_sort_token_plot.py` | Sorts tokens by balance/Q-sensitivity scores and plots scatter |
| Step 6 | `step6_apply_bracket.py` | Tags tokens with `<red>`/`<blue>` markers based on sparse/dense expert activation |

### Environment Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r stage_1_analyze_routing/requirements.txt
```

### Configuration

Edit `stage_1_analyze_routing/config.sh`:

```bash
STAGE="0"
NUM_SAMPLES=128
CALIB_SIZE=$NUM_SAMPLES

MODEL_BASE_PATH="/your_base_model_path"   # <-- set this
MODEL_NAME="qwen3_30b_a3b"

# Automatically points to Stage 0 output inside the repo
DATASET_DIR="${REPO_ROOT}/stage_0_quantize/dataset_dir/D${STAGE}_${CALIB_SIZE}"

# Outputs are saved inside stage_1_analyze_routing/output/
SAVE_BASE_PATH="${SCRIPT_DIR}/output"

# Expert thresholds (tuned per run 2)
freq_thr=0.25
less_thr=0.25
rob_thr=0.25
sen_thr=0.25

# Token thresholds (tuned per run 3)
balance_thr_blue_x=0.1
balance_thr_blue_y=0.77
balance_thr_red_x=0.15
balance_thr_red_y=0.6
```

### Run

The pipeline is split into 4 sequential runs. Run them all at once:

```bash
cd stage_1_analyze_routing
source .venv/bin/activate
bash run_pipeline.sh all
```

Or run them individually to inspect intermediate results and tune thresholds:

```bash
bash run_pipeline.sh 1   # step2 + step3 + step4 (no thresholds)
bash run_pipeline.sh 2   # step3 + step4 with expert thresholds + step5
bash run_pipeline.sh 3   # step5 with token thresholds
bash run_pipeline.sh 4   # step6: apply <red>/<blue> brackets
```

### Output

| Path | Content |
|------|---------|
| `stage_1_analyze_routing/output/<MODEL>_<DATASET>/D0_128/s2_expert_count/token_routing.jsonl` | Per-token expert routing data |
| `stage_1_analyze_routing/output/<MODEL>_<DATASET>/D0_128/s3_expert_dist/` | Expert activation distribution plots and JSON |
| `stage_1_analyze_routing/output/<MODEL>_<DATASET>/D0_128/s4_weight_outlier/` | Weight sensitivity plots and JSON |
| `stage_1_analyze_routing/output/<MODEL>_<DATASET>/D0_128/s5_sorted_token/` | Token classification scatter plots and JSON |
| `stage_1_analyze_routing/output/<MODEL>_<DATASET>/D0_128/s6_apply_bracket/bracketed_balance.jsonl` | Annotated samples with `<red>`/`<blue>` tags (input to Stage 2) |

---

## Stage 2: Extract text patterns causing frequent/scarce experts

**Directory**: `stage_2_pattern_extract/`  
**Virtual environment**: `nemo_data_designer`  
**Run script**: `stage_2_pattern_extract/extract_pattern_agent_balanced.py`

### What it does

Feeds bracketed samples (from Stage 1) to a reasoning LLM (Nemotron-3-Super) to extract activation patterns. For each domain (`chat`, `code`, `stem`, `math`), it analyzes:
- Which token patterns activate sparsely-used experts (`<red>` tokens)
- Which token patterns activate frequently-used experts (`<blue>` tokens)

The output is a set of per-domain guidelines used in Stage 3 to generate targeted synthetic data.

### Prerequisites

A running NIM or vLLM server with `nemotron-3-super` or compatible model:

```bash
# Example: start vLLM with Nemotron-3-Super
vllm serve <model-path> --port 8000
```

### Environment Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r stage_2_pattern_extract/requirements.txt
```

### Configuration

All paths are derived automatically from the script location. Only the server URL and model name need to be adjusted if needed:

```python
# On-premise NIM (default)
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="no-key",
)
CHAT_MODEL = "nemotron-3-super"

# Input: automatically resolved to Stage 1 output inside the repo
# dataset_annotated = <repo_root>/stage_1_analyze_routing/output/qwen3_30b_a3b_nemo_dataset/D0_128/s6_apply_bracket/bracketed_balance.jsonl

# Output: saved inside stage_2_pattern_extract/instruction/
# OUTPUT_DIR = <stage_2_pattern_extract>/instruction/
```

### Run

```bash
cd stage_2_pattern_extract
source .venv/bin/activate
python extract_pattern_agent_balanced.py
```

### Output

Per-domain guideline markdown files saved to `stage_2_pattern_extract/instruction/`:

```
stage_2_pattern_extract/instruction/
  balance_guideline_chat_nemotron-3-super.md
  balance_guideline_code_nemotron-3-super.md
  balance_guideline_stem_nemotron-3-super.md
  balance_guideline_math_nemotron-3-super.md
```

---

## Stage 3: Generate synthetic calibration dataset achieving balanced activated experts

**Directory**: `stage_3_generate_dataset/`  
**Virtual environment**: `nemo_data_designer` (same as Stage 2)  
**Run script**: `stage_3_generate_dataset/pipeline_calibration_per_domain.py`

### What it does

Generates 8 synthetic calibration datasets (2 guideline types × 4 domains) using NVIDIA Data Designer:

| Guideline Type | Domains | Target Records |
|---------------|---------|---------------|
| `balance` | chat, code, math, stem | 128 per domain |
| `q_sensitivity` | chat, code, math, stem | 128 per domain |

All 8 domain/guideline combinations run in parallel using `ThreadPoolExecutor`.

### Prerequisites

- Running vLLM or NIM server (same as Stage 2)
- Guideline files from Stage 2 in `instruction/` directory
- Seed data (`bracketed_balance.jsonl`) from Stage 1

### Environment Setup

```bash
# Reuse the same venv as Stage 2
source stage_2_pattern_extract/.venv/bin/activate
# or create a new one:
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r stage_3_generate_dataset/requirements.txt
```

### Configuration

All paths are derived automatically from the script location. Only the server URL and generation parameters need to be adjusted:

```python
VLLM_BASE_URL   = "http://localhost:8000/v1"  # <-- vLLM server URL

# Input: automatically resolved inside the repo
# SEED_DATA_PATH  = <repo_root>/stage_1_analyze_routing/output/qwen3_30b_a3b_nemo_dataset/D0_128/s6_apply_bracket/bracketed_balance.jsonl
# INSTRUCTION_DIR = <repo_root>/stage_2_pattern_extract/instruction/

# Output: saved inside stage_3_generate_dataset/output_per_domain/
# OUTPUT_ROOT = <stage_3_generate_dataset>/output_per_domain/

NUM_RECORDS_PER_COMBO    = 256   # oversample count (expect ~50% valid rate)
TARGET_RECORDS_PER_COMBO = 128   # final target records per combo
MAX_PARALLEL_REQUESTS    = 16    # parallel requests per combo
MAX_WORKERS              = 8     # parallel combos
```

Optionally configure a Slack webhook for progress notifications via environment variable:

```bash
export SLACK_WEBHOOK_URL="https://hooks.slack.com/services/..."  # or leave unset to disable
```

### Run

```bash
cd stage_3_generate_dataset
source .venv/bin/activate
python pipeline_calibration_per_domain.py
```

### Output

```
stage_3_generate_dataset/output_per_domain/
  balance/
    chat/   {chat_raw,clean,final}.{parquet,json}
    code/   ...
    math/   ...
    stem/   ...
  q_sensitivity/
    chat/   ...
    code/   ...
    math/   ...
    stem/   ...
  calibration_balance_all.parquet
  calibration_q_sensitivity_all.parquet
```

The final merged parquet files (`calibration_balance_all.parquet` and `calibration_q_sensitivity_all.parquet`) are the synthetic calibration datasets ready for use in subsequent quantization runs.

---

## Pipeline Data Flow

```
Stage 0 output
  └── stage_0_quantize/dataset_dir/D0_128/
        ├── data-*.arrow          (HuggingFace dataset)
        ├── samples_text.jsonl    (raw text samples)
        └── calib_chunks.pt       (tokenized calibration chunks)
            │
            ▼
Stage 1 input → Stage 1 output
  └── stage_1_analyze_routing/output/qwen3_30b_a3b_nemo_dataset/D0_128/s6_apply_bracket/
        └── bracketed_balance.jsonl    (samples with <red>/<blue> token tags)
            │
            ├──────────────────────────────────────┐
            ▼                                      ▼
    Stage 2 input                          Stage 3 seed input
      └── 30 samples per domain             └── bracketed_balance.jsonl
          │
          ▼
    Stage 2 output
      └── stage_2_pattern_extract/instruction/
            └── balance_guideline_{domain}_nemotron-3-super.md
                │
                ▼
        Stage 3 input → Stage 3 output
          └── stage_3_generate_dataset/output_per_domain/
                └── calibration_{balance,q_sensitivity}_all.parquet
```

---

## Repository Structure

```
.
├── README.md
├── stage_0_quantize/
│   ├── run_pipeline.sh                  # Main pipeline script
│   ├── step1_gptq_quantize.py           # Dataset gen + Hessian + GPTQ quantization
│   ├── step2_trtllm_build_serve.py      # TRT-LLM engine build & server
│   ├── step3_nemo_eval.py               # NeMo Evaluator benchmarks
│   ├── nemotron_post_training_calib.py  # Calibration dataset builder
│   ├── modelopt_parallel_gptq.py        # Parallel GPTQ patch utilities
│   └── requirements.txt
├── stage_1_analyze_routing/
│   ├── run_pipeline.sh                  # Dispatcher (runs run1..run4)
│   ├── config.sh                        # Shared configuration
│   ├── run1_first.sh                    # Run 1: expert count + dist + weight
│   ├── run2_second.sh                   # Run 2: expert thresholds + token sort
│   ├── run3_third.sh                    # Run 3: token thresholds
│   ├── run4_fourth.sh                   # Run 4: apply brackets
│   ├── step1_dataset_load.py            # (Optional) dataset loading
│   ├── step2_count_expert.py            # Token → expert routing
│   ├── step3_count_expert_dist.py       # Expert activation distribution
│   ├── step4_weight_outlier_dist.py     # Expert weight sensitivity
│   ├── step5_sort_token_plot.py         # Token classification plots
│   ├── step6_apply_bracket.py           # Apply <red>/<blue> token tags
│   ├── requirements.txt
│   └── output/                          # (generated) expert analysis results
├── stage_2_pattern_extract/
│   ├── extract_pattern_agent_balanced.py  # LLM-based pattern extraction
│   ├── requirements.txt
│   └── instruction/                     # (generated) per-domain guidelines
├── stage_3_generate_dataset/
│   ├── pipeline_calibration_per_domain.py  # Parallel synthetic data generation
│   ├── requirements.txt
│   └── output_per_domain/               # (generated) synthetic calibration datasets
├── models/                              # (generated) quantized checkpoints & TRT engines
├── eval_results/                        # (generated) benchmark results
└── log/                                 # (generated) execution logs
```

---

## Notes

- **Calibration dataset**: The pipeline uses [nvidia/Nemotron-Post-Training-Dataset-v1](https://huggingface.co/datasets/nvidia/Nemotron-Post-Training-Dataset-v1) by default. Modify `step1_gptq_quantize.py` or `step1_dataset_load.py` to use a custom dataset.
- **Model paths**: All input/output paths are resolved relative to the repository root. Only `MODEL_BASE_PATH` in `stage_1_analyze_routing/config.sh` and `stage_0_quantize/run_pipeline.sh` must be set to your local model directory.
- **GPU memory**: Stage 0 (GPTQ) benefits from multi-GPU setup. Stage 1 (routing analysis) runs on a single GPU. Stages 2 & 3 are CPU-bound (API calls to a separately-served model).
- **Stage 2 & 3 server**: Both stages require a running vLLM or NVIDIA NIM endpoint serving a reasoning-capable model (e.g., Nemotron-3-Super). The default endpoint is `http://localhost:8000/v1`.
