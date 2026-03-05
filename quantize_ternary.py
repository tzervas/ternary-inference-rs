#!/usr/bin/env python3
"""
Ternary quantization for Qwen2.5-Coder-32B-Instruct.

Quantizes linear projection weights to ternary {-1, 0, +1} with per-group
scales using optimal MSE scaling (better than naive AbsMean).

Key improvements over the existing tzervas/qwen2.5-coder-32b-bitnet-1.58b:
1. embed_tokens and lm_head kept in bf16 (not quantized to ternary)
2. Iterative optimal MSE scale instead of single-pass AbsMean
3. Clean shard-by-shard processing (low peak RAM)

Output format: base-3 packed uint8 (4 trits/byte), compatible with
ternary-inference-rs Qwen2 loader.

Usage:
    .venv/bin/python quantize_ternary.py
    .venv/bin/python quantize_ternary.py --group-size 128
    .venv/bin/python quantize_ternary.py --model Qwen/Qwen2.5-Coder-32B-Instruct --output models/qwen2-32b-ternary
"""

import argparse
import gc
import json
import logging
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file, safe_open
from huggingface_hub import snapshot_download

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def quantize_ternary_group(w: torch.Tensor, group_size: int = 64):
    """
    Quantize a 2D weight tensor to ternary {-1, 0, +1} with per-group scales.

    Uses iterative optimal MSE scale: for each group, alternates between
    computing ternary assignments and finding the scale that minimizes
    ||w - scale * q||^2.

    Returns:
        q: int8 tensor of shape [out, in] with values {-1, 0, +1}
        scales: float32 tensor of shape [out, num_groups]
    """
    out_features, in_features = w.shape
    if in_features % group_size != 0:
        # Pad to group_size boundary
        pad = group_size - (in_features % group_size)
        w = torch.nn.functional.pad(w, (0, pad))
        in_features = w.shape[1]

    num_groups = in_features // group_size
    w_groups = w.reshape(out_features, num_groups, group_size).float()

    # Initial scale: AbsMean
    scales = w_groups.abs().mean(dim=-1)  # [out, num_groups]

    # Iterative refinement (3 rounds)
    for _ in range(3):
        safe_scales = scales.unsqueeze(-1).clamp(min=1e-10)
        q = (w_groups / safe_scales).round().clamp(-1, 1)

        # Optimal scale: scale = (w · q) / (q · q)
        wq = (w_groups * q).sum(dim=-1)
        qq = (q * q).sum(dim=-1).clamp(min=1)
        scales = (wq / qq).clamp(min=0)

    # Final quantization
    safe_scales = scales.unsqueeze(-1).clamp(min=1e-10)
    q = (w_groups / safe_scales).round().clamp(-1, 1).to(torch.int8)
    q = q.reshape(out_features, in_features)

    return q, scales


def pack_ternary_base3(q: torch.Tensor) -> torch.Tensor:
    """
    Pack ternary int8 tensor into base-3 uint8 format.

    Encoding: byte = (v0+1) + (v1+1)*3 + (v2+1)*9 + (v3+1)*27
    """
    out_features, in_features = q.shape
    assert in_features % 4 == 0, f"in_features={in_features} not divisible by 4"

    q_shifted = (q.numpy().astype(np.int16) + 1)  # {-1,0,+1} -> {0,1,2}
    q_4 = q_shifted.reshape(out_features, in_features // 4, 4)

    packed = (q_4[:, :, 0] +
              q_4[:, :, 1] * 3 +
              q_4[:, :, 2] * 9 +
              q_4[:, :, 3] * 27).astype(np.uint8)

    return torch.from_numpy(packed)


def compute_quantization_stats(w_orig, q, scales, group_size):
    """Compute MSE and SNR for a quantized weight."""
    out_f, in_f = w_orig.shape
    num_groups = in_f // group_size
    dequant = (q.float().reshape(out_f, num_groups, group_size) *
               scales.unsqueeze(-1)).reshape(out_f, in_f)
    mse = ((w_orig.float() - dequant) ** 2).mean().item()
    signal = (w_orig.float() ** 2).mean().item()
    snr_db = 10 * math.log10(signal / max(mse, 1e-15))

    # Value distribution
    total = q.numel()
    n_neg = (q == -1).sum().item()
    n_zero = (q == 0).sum().item()
    n_pos = (q == 1).sum().item()

    return {
        "mse": mse,
        "snr_db": snr_db,
        "pct_neg": 100 * n_neg / total,
        "pct_zero": 100 * n_zero / total,
        "pct_pos": 100 * n_pos / total,
    }


def main():
    parser = argparse.ArgumentParser(description="Ternary quantization for Qwen2.5-Coder")
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-32B-Instruct", help="HF model ID")
    parser.add_argument("--output", default="models/qwen2-32b-ternary", help="Output directory")
    parser.add_argument("--group-size", type=int, default=64, help="Quantization group size")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    # Resolve model path (download if needed)
    log.info(f"Resolving model: {args.model}")
    model_path = Path(snapshot_download(args.model, local_files_only=True))
    log.info(f"Model path: {model_path}")

    # Load config and index
    with open(model_path / "config.json") as f:
        config = json.load(f)

    with open(model_path / "model.safetensors.index.json") as f:
        index = json.load(f)

    weight_map = index["weight_map"]
    group_size = args.group_size

    # Build shard -> tensors map
    shard_tensors = {}
    for name, shard in weight_map.items():
        shard_tensors.setdefault(shard, []).append(name)

    # Patterns for what to quantize vs keep
    QUANTIZE_SUFFIXES = [
        "q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight",
        "gate_proj.weight", "up_proj.weight", "down_proj.weight",
    ]
    KEEP_BF16 = ["embed_tokens.weight", "lm_head.weight"]

    # Output tracking
    out_shard_idx = 0
    out_tensors = {}
    out_size = 0
    max_shard_bytes = 4 * 1024**3  # 4 GB per shard
    all_file_map = {}
    stats_summary = []

    total_quantized = 0
    total_bf16 = 0

    def flush():
        nonlocal out_shard_idx, out_tensors, out_size
        if not out_tensors:
            return
        out_shard_idx += 1
        fname = f"model-{out_shard_idx:05d}.safetensors"
        save_file(out_tensors, str(output_path / fname))
        for name in out_tensors:
            all_file_map[name] = fname
        mb = out_size / 1024**2
        log.info(f"  Wrote {fname}: {len(out_tensors)} tensors, {mb:.0f} MB")
        out_tensors = {}
        out_size = 0

    def add(name, tensor):
        nonlocal out_size
        sz = tensor.numel() * tensor.element_size()
        if out_size + sz > max_shard_bytes and out_tensors:
            flush()
        out_tensors[name] = tensor
        out_size += sz

    t0 = time.time()

    for shard_file in sorted(shard_tensors.keys()):
        shard_path = model_path / shard_file
        log.info(f"Processing {shard_file} ({len(shard_tensors[shard_file])} tensors)")

        sf = safe_open(str(shard_path), framework="pt")

        for tensor_name in sorted(shard_tensors[shard_file]):
            tensor = sf.get_tensor(tensor_name)
            is_quantize = any(tensor_name.endswith(s) for s in QUANTIZE_SUFFIXES)
            is_keep_bf16 = any(p in tensor_name for p in KEEP_BF16)

            if is_keep_bf16:
                # Keep embeddings and lm_head in bf16
                t_bf16 = tensor.to(torch.bfloat16)
                add(tensor_name, t_bf16)
                total_bf16 += tensor.numel()
                log.info(f"  bf16: {tensor_name} {list(tensor.shape)} ({tensor.numel()/1e6:.1f}M params)")

            elif is_quantize:
                # Ternary quantization
                w = tensor.float()
                out_f, in_f = w.shape

                # Ensure in_features divisible by group_size and 4
                lcm = max(group_size, 4)
                if in_f % lcm != 0:
                    pad = lcm - (in_f % lcm)
                    w = torch.nn.functional.pad(w, (0, pad))
                    log.info(f"    Padded {tensor_name} in_features {in_f} -> {w.shape[1]}")

                q, scales = quantize_ternary_group(w, group_size)
                packed = pack_ternary_base3(q)

                stats = compute_quantization_stats(w, q, scales, group_size)
                stats_summary.append({"name": tensor_name, **stats})

                add(tensor_name, packed)
                add(f"{tensor_name}.scales", scales)
                total_quantized += tensor.numel()

                log.info(f"  quant: {tensor_name} {list(tensor.shape)} -> packed {list(packed.shape)} "
                         f"SNR={stats['snr_db']:.1f}dB "
                         f"({stats['pct_neg']:.0f}%- {stats['pct_zero']:.0f}%0 {stats['pct_pos']:.0f}%+)")

                del q, scales, packed, w
            else:
                # Biases, layernorm weights, etc. - keep as-is
                add(tensor_name, tensor)
                log.info(f"  keep:  {tensor_name} {list(tensor.shape)} {tensor.dtype}")

            del tensor

        del sf
        gc.collect()

    flush()

    elapsed = time.time() - t0

    # Rename shards to include total count
    total_shards = out_shard_idx
    for i in range(1, total_shards + 1):
        old = output_path / f"model-{i:05d}.safetensors"
        new = output_path / f"model-{i:05d}-of-{total_shards:05d}.safetensors"
        old.rename(new)
        old_name = f"model-{i:05d}.safetensors"
        new_name = f"model-{i:05d}-of-{total_shards:05d}.safetensors"
        for k in list(all_file_map.keys()):
            if all_file_map[k] == old_name:
                all_file_map[k] = new_name

    # Write index
    index_out = {
        "metadata": {"total_size": 0},  # Will be computed by loader
        "weight_map": all_file_map,
    }
    with open(output_path / "model.safetensors.index.json", "w") as f:
        json.dump(index_out, f, indent=2)

    # Write config
    config["quantization_config"] = {
        "quant_method": "bitnet_158",
        "bits": 1.58,
        "group_size": group_size,
        "packing": "base3_columns",
        "embed_bf16": True,
        "lm_head_bf16": True,
    }
    with open(output_path / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Copy tokenizer files
    for tok_file in ["tokenizer.json", "tokenizer_config.json", "vocab.json",
                      "merges.txt", "special_tokens_map.json", "generation_config.json"]:
        src = model_path / tok_file
        if src.exists():
            import shutil
            shutil.copy2(src, output_path / tok_file)

    # Summary
    log.info(f"\n{'='*60}")
    log.info(f"Quantization complete in {elapsed:.0f}s")
    log.info(f"  Quantized: {total_quantized/1e9:.2f}B params -> ternary")
    log.info(f"  Kept bf16: {total_bf16/1e6:.0f}M params (embed + lm_head)")
    log.info(f"  Shards: {total_shards}")
    log.info(f"  Output: {output_path}")

    # SNR summary
    if stats_summary:
        snrs = [s["snr_db"] for s in stats_summary]
        log.info(f"  SNR: min={min(snrs):.1f}dB, mean={sum(snrs)/len(snrs):.1f}dB, max={max(snrs):.1f}dB")

    # Save stats
    with open(output_path / "quantization_stats.json", "w") as f:
        json.dump(stats_summary, f, indent=2)

    log.info(f"\nTo test: update the Rust loader to handle bf16 embed/lm_head, then run:")
    log.info(f"  RUST_LOG=info cargo run --example generate --release --features cuda -- --gpu --model-path {output_path}")


if __name__ == "__main__":
    main()
