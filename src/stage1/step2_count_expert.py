"""
Forward MoE on the step1 calibration dataset and write per-token activated expert ids as JSONL.

- Same forward as ModelOpt GPTQ calibration in quant_pipe/step1_gptq_quantize.py:
  ``model(input_ids=input_ids, attention_mask=attention_mask)`` (torch.no_grad)
- Dataset: HF Dataset from step1_dataset_load.py save_to_disk (input_ids, attention_mask lists).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from datasets import load_from_disk
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def resolve_dataset_dir(
    args: argparse.Namespace,
) -> Path:
    if getattr(args, "dataset_dir", None):
        return Path(args.dataset_dir).resolve()
    return Path(args.save_path).resolve()


def resolve_output_dir(
    args: argparse.Namespace,
    dataset_dir: Path,
) -> Path:
    """Directory for routing JSONL: --routing-subdir, else --save-path if set, else dataset dir."""
    if getattr(args, "routing_subdir", None):
        return Path(args.routing_subdir)
    if getattr(args, "save_path", None):
        return Path(args.save_path).resolve()
    return dataset_dir


def find_sparse_moe_blocks(model: torch.nn.Module) -> list[tuple[int, torch.nn.Module]]:
    """Find Sparse MoE blocks (gate + top_k) per layer (Qwen2/3 MoE, Mixtral, etc.)."""
    inner = getattr(model, "model", None)
    if inner is None:
        raise ValueError("model.model is missing.")
    layers = getattr(inner, "layers", None)
    if layers is None:
        raise ValueError("model.model.layers not found.")

    out: list[tuple[int, torch.nn.Module]] = []
    for i, layer in enumerate(layers):
        candidates = []
        if getattr(layer, "mlp", None) is not None:
            candidates.append(layer.mlp)
        if getattr(layer, "block_sparse_moe", None) is not None:
            candidates.append(layer.block_sparse_moe)
        for mod in candidates:
            if hasattr(mod, "gate") and hasattr(mod, "top_k"):
                out.append((i, mod))
                break
    if not out:
        raise ValueError("No Sparse MoE block found (no gate/top_k).")
    return out


def expert_ids_from_router_logits(
    moe_block: torch.nn.Module,
    router_logits: torch.Tensor,
) -> torch.Tensor:
    """
    Same expert selection as model forward (top-k indices).
    Softmax then topk; norm_topk_prob only affects weights, not top-k ids.
    """
    routing_weights = F.softmax(router_logits.float(), dim=1, dtype=torch.float)
    _, selected_experts = torch.topk(
        routing_weights,
        moe_block.top_k,
        dim=-1,
    )
    return selected_experts


class MoERoutingCapture:
    """Collect per-token expert ids from each MoE block router_logits (forward output)."""

    def __init__(self, moe_layers: list[tuple[int, torch.nn.Module]]):
        self.moe_layers = moe_layers
        self._handles: list[Any] = []
        # After forward: layer_idx -> (batch*seq, top_k) CPU tensor
        self._last: dict[int, torch.Tensor] = {}

    def _make_hook(self, layer_idx: int, moe_block: torch.nn.Module):
        def hook(module, _inputs, output):
            if not isinstance(output, tuple) or len(output) < 2:
                return
            router_logits = output[1]
            sel = expert_ids_from_router_logits(module, router_logits)
            self._last[layer_idx] = sel.detach().cpu()

        return hook

    def register(self):
        for layer_idx, moe_block in self.moe_layers:
            h = moe_block.register_forward_hook(self._make_hook(layer_idx, moe_block))
            self._handles.append(h)

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def clear(self):
        self._last.clear()

    def get_token_experts(self, seq_len: int) -> dict[str, list[list[int]]]:
        """Assume batch=1: layer_str -> list over positions of expert id lists."""
        out: dict[str, list[list[int]]] = {}
        for layer_idx, _ in self.moe_layers:
            t = self._last.get(layer_idx)
            if t is None:
                raise RuntimeError(f"layer {layer_idx}: routing not captured.")
            if t.shape[0] != seq_len:
                raise RuntimeError(
                    f"layer {layer_idx}: expected {seq_len} tokens, got {t.shape[0]}"
                )
            key = str(layer_idx)
            out[key] = t.tolist()
        return out


def gptq_style_forward(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> None:
    """Same call pattern as step1_gptq_quantize.calibrate_loop."""
    input_ids = input_ids.to(DEVICE)
    attention_mask = attention_mask.to(DEVICE)
    with torch.no_grad():
        model(input_ids=input_ids, attention_mask=attention_mask)


def token_str_for_id(tokenizer, tid: int) -> str:
    try:
        s = tokenizer.decode([tid], skip_special_tokens=False)
    except Exception:
        s = ""
    if not s.strip():
        try:
            tok = tokenizer.convert_ids_to_tokens(tid)
            s = str(tok) if tok is not None else ""
        except Exception:
            s = ""
    if not s:
        s = f"<id={tid}>"
    return s


def build_record(
    sample_id: int,
    position_id: int,
    token_id: int,
    token_str: str,
    layer_to_experts: dict[str, list[list[int]]],
) -> dict[str, Any]:
    return {
        "sample_id": sample_id,
        "position_id": position_id,
        "token_id": token_id,
        "token_str": token_str,
        "activated_expert_ids": layer_to_experts,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="MoE routing: forward step1 dataset, write per-token expert ids JSONL.",
    )
    p.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="HF model path or id (same as step1 --model-name).",
    )
    p.add_argument(
        "--save-path",
        type=str,
        default=None,
        help="Step1 dataset dir if --dataset-dir unset; otherwise default directory for routing outputs.",
    )
    p.add_argument(
        "--dataset-dir",
        type=str,
        default=None,
        help="Path passed to load_from_disk (if set, --save-path not used).",
    )
    p.add_argument(
        "--routing-subdir",
        type=str,
        default=None,
        help="Override output directory (instead of --save-path or dataset dir).",
    )
    p.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Process only the first N samples (debug).",
    )
    p.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True to AutoTokenizer / AutoModelForCausalLM.",
    )
    p.add_argument(
        "--sample-json-token",
        type=str,
        default="mid_first",
        choices=["mid_first", "mid_dataset"],
        help="sample.json: mid_first=middle token of first sample; mid_dataset=global stream midpoint.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.dataset_dir is None and args.save_path is None:
        raise SystemExit(
            "Provide either --dataset-dir or --save-path (step1 dataset directory)."
        )
    dataset_dir = resolve_dataset_dir(args)
    if not dataset_dir.is_dir():
        raise SystemExit(f"Dataset path does not exist or is not a directory: {dataset_dir}")

    out_dir = resolve_output_dir(args, dataset_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "token_routing.jsonl"
    sample_path = out_dir / "sample.json"

    print(f"[dataset] {dataset_dir}")
    print(f"[out]     {out_dir}")

    ds = load_from_disk(str(dataset_dir))
    n = len(ds)
    if args.max_samples is not None:
        n = min(n, args.max_samples)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("[model] loading...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=args.trust_remote_code,
    )
    model.eval()

    moe_layers = find_sparse_moe_blocks(model)
    print(f"[MoE] {len(moe_layers)} sparse layer(s): {[x[0] for x in moe_layers]}")

    capture = MoERoutingCapture(moe_layers)
    capture.register()

    # Single record for sample.json
    sample_record: dict[str, Any] | None = None
    mid_dataset_pos: int | None = None

    if args.sample_json_token == "mid_dataset":
        total_tokens_est = 0
        for i in range(n):
            total_tokens_est += len(ds[i]["input_ids"])
        mid_dataset_pos = max(0, total_tokens_est // 2)
        print(f"[sample.json] mid_dataset global token index ≈ {mid_dataset_pos} (total_tokens≈{total_tokens_est})")

    global_tok_idx = 0

    with open(jsonl_path, "w", encoding="utf-8") as fj:
        for sample_id in tqdm(range(n), desc="routing"):
            row = ds[sample_id]
            ids = row["input_ids"]
            mask = row["attention_mask"]
            seq_len = len(ids)
            if seq_len != len(mask):
                raise ValueError(
                    f"sample {sample_id}: input_ids and attention_mask length mismatch"
                )

            input_ids = torch.tensor([ids], dtype=torch.long)
            attention_mask = torch.tensor([mask], dtype=torch.long)

            capture.clear()
            gptq_style_forward(model, input_ids, attention_mask)
            layer_experts = capture.get_token_experts(seq_len)

            for pos in range(seq_len):
                tid = int(ids[pos])
                pos_layer = {lk: lv[pos] for lk, lv in layer_experts.items()}
                rec = build_record(
                    sample_id,
                    pos,
                    tid,
                    token_str_for_id(tokenizer, tid),
                    pos_layer,
                )
                fj.write(json.dumps(rec, ensure_ascii=False) + "\n")

                if sample_record is None and args.sample_json_token == "mid_first":
                    if sample_id == 0 and pos == seq_len // 2:
                        sample_record = rec
                if args.sample_json_token == "mid_dataset" and mid_dataset_pos is not None:
                    if global_tok_idx == mid_dataset_pos:
                        sample_record = rec
                global_tok_idx += 1

            if (
                sample_record is None
                and args.sample_json_token == "mid_first"
                and sample_id == 0
                and seq_len > 0
            ):
                mid = seq_len // 2
                tid = int(ids[mid])
                sample_record = build_record(
                    0,
                    mid,
                    tid,
                    token_str_for_id(tokenizer, tid),
                    {lk: lv[mid] for lk, lv in layer_experts.items()},
                )

    capture.remove()

    if sample_record is None:
        # Fallback: first jsonl line
        with open(jsonl_path, encoding="utf-8") as fj:
            first_line = fj.readline()
        if first_line.strip():
            sample_record = json.loads(first_line)

    with open(sample_path, "w", encoding="utf-8") as f:
        json.dump(sample_record, f, indent=2, ensure_ascii=False)

    print(f"Wrote {jsonl_path}")
    print(f"Wrote {sample_path}")


if __name__ == "__main__":
    main()
