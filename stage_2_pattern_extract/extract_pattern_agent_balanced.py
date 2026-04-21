import os
import json
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
from collections import defaultdict
from tqdm import tqdm

load_dotenv()

# 온프레미스 NIM (권장)
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="no-key",  # 로컬에서는 인증 불필요
)

# Cloud API 사용 시 위 코드를 아래로 교체
# client = OpenAI(
#     base_url="https://integrate.api.nvidia.com/v1",
#     api_key=os.environ["NVIDIA_API_KEY"],
# )

CHAT_MODEL = "nemotron-3-super"

USE_STREAM = True  # False: 일반 응답, True: 스트리밍

#dataset_annotated = "bracketed_balance.jsonl"
dataset_annotated = "/home/work/nota-data/nemo_hackathon/expert_analysis/qwen3_30b_a3b_nemo_dataset/D0_128/s6_apply_bracket/bracketed_balance.jsonl"

DOMAINS = ['chat', 'code', 'stem', 'math']

DOMAIN_CONTEXT = {
    'chat': "The samples are from conversational/dialogue data.",
    'code': "The samples are from programming/code-related data.",
    'stem': "The samples are from science, technology, engineering, and mathematics (STEM) academic data.",
    'math': "The samples are from mathematical problem-solving data.",
}

PROMPT_HEADER = """\
We aim to quantize a Mixture-of-Experts (MoE) LLM.
When tokens from calibration samples are balanced across all experts, the statistics for each expert become more stable, thereby reducing quantization errors.
By feeding a large-scale pre-prepared corpus into the target model, we have distinguished between "sparsely activated experts" and "frequently activated experts."
Our goal is to generate additional synthetically crafted samples that specifically increase the activation of these sparsely activated experts.

In the provided samples, tokens enclosed in <red> and </red> are those that relatively frequently activate the set of sparsely activated experts.
Conversely, tokens between <blue> and </blue> are those that activate the frequently activated experts.
Analyze the structural or contextual patterns of the tokens within <red></red> versus <blue></blue>, considering the given context for each token.
Based on this analysis, provide a set of guidelines for creating samples that maximize the activation of sparsely activated experts while minimizing the activation of frequently activated experts.

{domain_context}

The following are the samples:
"""

PROMPT_FOOTER = """\
\nAnalyze the patterns from the provided samples and present a list of guidelines in the format: "1. Guideline sentence, 2. Guideline sentence, ..., N. Guideline sentence."

Precautions:
(1) When analyzing patterns, consolidate any redundant or overlapping information into a single point.
(2) Each guideline must align with common sense and logical reasoning.
(3) Each guideline must be specific enough to ensure there is no ambiguity when generating actual sentences.\
"""

# Group samples by domain
samples_by_domain = defaultdict(list)
with open(dataset_annotated, "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line.strip())
        domain = data.get('domain')
        if domain in DOMAINS:
            samples_by_domain[domain].append(data['text'])

# Build and print prompt per domain
prompts = {}
for domain in DOMAINS:
    samples = samples_by_domain[domain][:30]
    prompt = PROMPT_HEADER.format(domain_context=DOMAIN_CONTEXT[domain])
    for text in samples:
        prompt += f"\n\n[sample]: {text}"
    prompt += PROMPT_FOOTER
    prompts[domain] = prompt

OUTPUT_DIR = "/home/work/nota-data/gmkim/Nemotron-data-designer/instruction"

def run_stream(prompt, enable_thinking=True):
    MAX_TOKENS = 100000
    full_response = []
    in_answer = False

    stream = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=1.0,
        max_tokens=MAX_TOKENS,
        extra_body={
            "reasoning_budget": 8192,
            "chat_template_kwargs": {"enable_thinking": enable_thinking},
        },
        stream=True,
    )

    reasoning_bar = tqdm(total=8192, desc="  Reasoning", unit="tok", leave=False, colour="blue")
    content_bar   = tqdm(total=MAX_TOKENS, desc="  Response ", unit="tok", leave=True, colour="green")

    for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        reasoning = getattr(delta, "reasoning", None) or getattr(delta, "reasoning_content", None)
        content = delta.content
        if reasoning:
            reasoning_bar.update(len(reasoning.split()))
        elif content:
            if not in_answer:
                reasoning_bar.close()
                in_answer = True
            content_bar.update(len(content.split()))
            full_response.append(content)

    content_bar.close()
    print()
    return "".join(full_response)


for domain in DOMAINS:
    print(f"\n{'='*80}")
    print(f"DOMAIN: {domain.upper()}")
    print('='*80)

    result = run_stream(prompts[domain], enable_thinking=True)

    if not result.strip():
        print(f"  [WARN] content empty with thinking=True, retrying with thinking=False ...")
        result = run_stream(prompts[domain], enable_thinking=False)

    out_path = os.path.join(OUTPUT_DIR, f"balance_guideline_{domain}_{CHAT_MODEL}.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(result)
    print(f"[Saved] {out_path}")