# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Phase 4 of `gptq_lite` is derived from:
#   modelopt.torch.quantization.model_calib.gptq_lite
# Parallel CUDA-stream scheduling is not part of upstream ModelOpt.

"""Runtime patches for ModelOpt ``gptq_lite`` (see ``mode.BaseCalibrateModeDescriptor``).

- **Parallel Phase-4** when ``parallel_gptq_batch > 1``: overlap ``blockwise_weight_update`` across CUDA streams.
- **Hessian-only**: skip Phase-4 so ``mtq.quantize`` ends after Hessian compute/load + optional disk save.

Always call ``restore_parallel_gptq_patch()`` in a ``finally`` block after ``mtq.quantize``.
"""

from __future__ import annotations

import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
import torch.nn as nn
from tqdm import tqdm

from modelopt.torch.opt.searcher import ForwardLoop
from modelopt.torch.quantization import model_calib as mc
from modelopt.torch.quantization.mode import GPTQLiteModeDescriptor
from modelopt.torch.utils import print_rank_0
from modelopt.torch.utils.perf import get_used_gpu_mem_fraction

# Populated on first install (parallel batch >1 or hessian-only)
_orig_gptq_lite = None
_patch_installed = False
# Read at Phase 4 (set in ``install_parallel_gptq_patch_if_needed`` before ``mtq.quantize``).
_parallel_module_batch: int = 1
# When True, ``_gptq_lite_impl`` returns after Hessian load/compute + save (no weight update).
_hessian_only: bool = False
_warned_single_gpu_parallel: bool = False


def _phase4_weight_updates(
    quantized_modules: list,
    hessian_state: dict,
    block_size: int,
    percdamp: float,
    parallel_batch: int,
) -> None:
    """Phase 4: apply ``blockwise_weight_update`` to each quantized linear."""
    if parallel_batch <= 1:
        for _name, module in tqdm(quantized_modules, desc="Quantizing layers"):
            state = hessian_state[module.name]
            hessian = state["hessian"].to(module.weight.device)
            mc.blockwise_weight_update(module, hessian, block_size, percdamp)
            del hessian_state[module.name]
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return

    global _warned_single_gpu_parallel
    desc = f"Quantizing layers (parallel batch={parallel_batch}; multi-GPU per chunk if applicable)"
    idx = 0
    pbar = tqdm(total=len(quantized_modules), desc=desc)
    while idx < len(quantized_modules):
        chunk = quantized_modules[idx : idx + parallel_batch]
        idx += len(chunk)

        cuda_jobs: list[tuple[str, nn.Module]] = []
        cpu_jobs: list[tuple[str, nn.Module]] = []
        for name, module in chunk:
            if module.weight.device.type == "cuda":
                cuda_jobs.append((name, module))
            else:
                cpu_jobs.append((name, module))

        for _name, module in cpu_jobs:
            state = hessian_state[module.name]
            hessian = state["hessian"].to(module.weight.device)
            mc.blockwise_weight_update(module, hessian, block_size, percdamp)
            del hessian_state[module.name]

        if cuda_jobs:
            devices = {m.weight.device for _, m in cuda_jobs}
            if len(devices) == 1:
                if parallel_batch > 1 and not _warned_single_gpu_parallel:
                    print_rank_0(
                        "[parallel_gptq_batch] 청크 안의 Linear가 모두 같은 CUDA 디바이스입니다. "
                        "각 `blockwise_weight_update`는 Cholesky 등으로 GPU를 크게 쓰기 때문에 "
                        "스트림만으로는 거의 직렬( wall time ≈ N×1초 )에 가깝습니다. "
                        "진짜 병렬(시간이 겹침)은 서로 다른 GPU에 올라간 가중치일 때만 기대할 수 있습니다 "
                        "(예: `device_map`으로 여러 GPU에 모듈이 흩어진 경우)."
                    )
                    _warned_single_gpu_parallel = True
                (only_dev,) = tuple(devices)
                with torch.cuda.device(only_dev):
                    for _name, module in cuda_jobs:
                        state = hessian_state[module.name]
                        hessian = state["hessian"].to(module.weight.device)
                        mc.blockwise_weight_update(module, hessian, block_size, percdamp)
                torch.cuda.synchronize(only_dev)
            else:
                by_idx: dict[int, list[tuple[str, nn.Module]]] = defaultdict(list)
                for name, module in cuda_jobs:
                    di = module.weight.device.index
                    if di is None:
                        by_idx[-1].append((name, module))
                    else:
                        by_idx[di].append((name, module))

                def _worker(device_index: int, jobs: list[tuple[str, nn.Module]]) -> None:
                    dev = torch.device("cuda", device_index)
                    with torch.cuda.device(dev):
                        for _n, mod in jobs:
                            st = hessian_state[mod.name]
                            h = st["hessian"].to(mod.weight.device)
                            mc.blockwise_weight_update(mod, h, block_size, percdamp)

                if -1 in by_idx:
                    for _name, module in by_idx.pop(-1):
                        state = hessian_state[module.name]
                        hessian = state["hessian"].to(module.weight.device)
                        mc.blockwise_weight_update(module, hessian, block_size, percdamp)

                if by_idx:
                    with ThreadPoolExecutor(max_workers=len(by_idx)) as pool:
                        futures = [
                            pool.submit(_worker, d_i, job_list)
                            for d_i, job_list in by_idx.items()
                        ]
                        for fut in as_completed(futures):
                            fut.result()
                    for d_i in by_idx:
                        torch.cuda.synchronize(torch.device("cuda", d_i))

        for _name, module in cuda_jobs:
            del hessian_state[module.name]

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        pbar.update(len(chunk))
    pbar.close()


def _gptq_lite_impl(
    model: nn.Module,
    forward_loop: ForwardLoop | None = None,
    percdamp: float = 0.01,
    block_size: int = 128,
    hessian_state_path: str | None = None,
) -> None:
    """Same as upstream ``gptq_lite`` except Phase 4 uses ``_phase4_weight_updates``."""
    hessian_state = {}

    def initialize_hessian_state(tensor_mapping):
        for name, (shape, device) in tensor_mapping.items():
            target_device = "cpu" if get_used_gpu_mem_fraction(device) > 0.65 else device
            hessian_state[name] = {
                "hessian": torch.zeros(shape, dtype=torch.float32, device=target_device),
                "n_samples": 0,
            }

    def load_hessian_state(path, tensor_mapping):
        print_rank_0(f"Loading hessian state from {path}")
        loaded_state = torch.load(path, map_location="cpu")

        for name, (shape, device) in tensor_mapping.items():
            if name not in loaded_state:
                raise KeyError(f"Layer '{name}' not found in loaded hessian state")

            target_device = "cpu" if get_used_gpu_mem_fraction(device) > 0.65 else device
            hessian_state[name] = {
                "hessian": loaded_state[name]["hessian"].to(target_device),
                "n_samples": loaded_state[name]["n_samples"],
            }

        print_rank_0(f"Successfully loaded hessian state with {len(hessian_state)} layers")

    def save_hessian_state(path):
        print_rank_0(f"Saving hessian state to {path}")
        try:
            cpu_state = {
                name: {"hessian": state["hessian"].cpu(), "n_samples": state["n_samples"]}
                for name, state in hessian_state.items()
            }

            os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
            torch.save(cpu_state, path)
            print_rank_0(f"Successfully saved hessian state to {path}")
        except Exception as e:
            print_rank_0(f"Error saving hessian state: {e}")
            print_rank_0("Continuing execution...")

    def hessian_hook(module, input, output):
        state = hessian_state[module.name]
        hessian, n_samples = mc.update_hessian(input[0], state["hessian"], state["n_samples"])
        hessian_state[module.name] = {"hessian": hessian, "n_samples": n_samples}

    mc.max_calibrate(model)

    tensor_mapping = {}
    for name, module in model.named_modules():
        if mc.is_quantized_linear(module) and module.weight_quantizer.is_enabled:
            in_features = module.weight.shape[-1]
            tensor_mapping[name] = ((in_features, in_features), module.weight.device)
            module.name = name

    hessian_exists = hessian_state_path is not None and os.path.exists(hessian_state_path)
    save_hessians = hessian_state_path is not None and not hessian_exists

    if hessian_exists:
        print_rank_0(f"Loading hessian state from {hessian_state_path}")
        load_hessian_state(hessian_state_path, tensor_mapping)
    else:
        if forward_loop is None:
            raise ValueError("forward_loop must be provided when computing Hessians")

        initialize_hessian_state(tensor_mapping)

        handles = []
        for name, module in model.named_modules():
            if mc.is_quantized_linear(module) and module.weight_quantizer.is_enabled:
                handles.append(module.register_forward_hook(hessian_hook))

        print_rank_0("Computing Hessian matrices...")
        forward_loop(model)

        for handle in handles:
            handle.remove()

    if save_hessians:
        try:
            save_hessian_state(hessian_state_path)
        except Exception as e:
            print_rank_0(f"Error saving hessian state: {e}")
            print_rank_0("Continuing execution...")

    if _hessian_only:
        print_rank_0("Hessian-only mode: skipping GPTQ-lite Phase-4 (per-Linear weight update).")
        return

    print_rank_0("Updating weights using GPTQ-lite algorithm...")

    quantized_modules = [
        (name, module)
        for name, module in model.named_modules()
        if mc.is_quantized_linear(module) and module.weight_quantizer.is_enabled
    ]

    batch = _parallel_module_batch
    _phase4_weight_updates(quantized_modules, hessian_state, block_size, percdamp, batch)

    print_rank_0("GPTQ-lite quantization completed successfully")


def install_parallel_gptq_patch_if_needed(parallel_gptq_batch: int) -> None:
    """If ``parallel_gptq_batch > 1``, replace ``gptq_lite`` with Phase-4 parallel implementation."""
    global _orig_gptq_lite, _patch_installed, _parallel_module_batch, _hessian_only

    _hessian_only = False
    _parallel_module_batch = max(1, int(parallel_gptq_batch))

    if _parallel_module_batch <= 1:
        restore_parallel_gptq_patch()
        return

    if _orig_gptq_lite is None:
        _orig_gptq_lite = mc.gptq_lite  # upstream reference (only captured once)

    if mc.gptq_lite is not _gptq_lite_impl:
        mc.gptq_lite = _gptq_lite_impl
        GPTQLiteModeDescriptor._calib_func = _gptq_lite_impl
    _patch_installed = True
    print_rank_0(
        f"[parallel_gptq] Patched gptq_lite Phase-4: parallel_gptq_batch={_parallel_module_batch} "
        "(CUDA streams per chunk; CPU modules in chunk stay sequential)."
    )


def install_gptq_hessian_only_patch() -> None:
    """Replace ``gptq_lite`` with implementation that stops after Hessian (no Phase-4 weight update)."""
    global _orig_gptq_lite, _patch_installed, _parallel_module_batch, _hessian_only

    _hessian_only = True
    _parallel_module_batch = 1

    if _orig_gptq_lite is None:
        _orig_gptq_lite = mc.gptq_lite

    mc.gptq_lite = _gptq_lite_impl
    GPTQLiteModeDescriptor._calib_func = _gptq_lite_impl
    _patch_installed = True
    print_rank_0(
        "[hessian] Patched gptq_lite: Hessian-only (Phase-4 weight update skipped; no TRT export in caller)."
    )


def restore_parallel_gptq_patch() -> None:
    """Undo any ``gptq_lite`` runtime patch (parallel and/or Hessian-only)."""
    global _patch_installed, _hessian_only
    _hessian_only = False
    if not _patch_installed or _orig_gptq_lite is None:
        _patch_installed = False
        return
    mc.gptq_lite = _orig_gptq_lite
    GPTQLiteModeDescriptor._calib_func = _orig_gptq_lite
    _patch_installed = False
