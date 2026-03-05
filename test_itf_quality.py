#!/usr/bin/env python3
"""
Quick validation: Compare ITF (PT2-LLM) vs naive RTN ternary quantization
on a single weight matrix from the model.
"""

import math
import sys
import time

import torch
from safetensors import safe_open
from huggingface_hub import snapshot_download
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from quantize_pt2_qep import iterative_ternary_fitting, compute_stats

def quantize_rtn(W, group_size=64):
    """Naive RTN: scale = AbsMean per group, round to {-1,0,+1}."""
    out_f, in_f = W.shape
    num_groups = in_f // group_size
    W_groups = W.reshape(out_f, num_groups, group_size)
    scales = W_groups.abs().mean(dim=-1)
    safe_scales = scales.unsqueeze(-1).clamp(min=1e-10)
    T = (W_groups / safe_scales).round().clamp(-1, 1).to(torch.int8)
    T = T.reshape(out_f, in_f)
    # Per-row alpha from scales (average across groups)
    alpha = scales.mean(dim=1)
    mu = torch.zeros(out_f)
    return T, alpha, mu

def quantize_rtn_optimal(W, group_size=64, n_iters=3):
    """RTN with iterative optimal MSE scale."""
    out_f, in_f = W.shape
    num_groups = in_f // group_size
    W_groups = W.reshape(out_f, num_groups, group_size).float()
    scales = W_groups.abs().mean(dim=-1)
    for _ in range(n_iters):
        safe_scales = scales.unsqueeze(-1).clamp(min=1e-10)
        q = (W_groups / safe_scales).round().clamp(-1, 1)
        wq = (W_groups * q).sum(dim=-1)
        qq = (q * q).sum(dim=-1).clamp(min=1)
        scales = (wq / qq).clamp(min=0)
    safe_scales = scales.unsqueeze(-1).clamp(min=1e-10)
    T = (W_groups / safe_scales).round().clamp(-1, 1).to(torch.int8)
    T = T.reshape(out_f, in_f)
    alpha = scales.mean(dim=1)
    mu = torch.zeros(out_f)
    return T, alpha, mu

def main():
    model_path = Path(snapshot_download("Qwen/Qwen2.5-Coder-32B-Instruct", local_files_only=True))

    # Load a few weight matrices for comparison
    test_weights = [
        ("model.layers.0.self_attn.q_proj.weight", "model-00001-of-00019.safetensors"),
        ("model.layers.0.mlp.gate_proj.weight", "model-00001-of-00019.safetensors"),
        ("model.layers.31.self_attn.q_proj.weight", "model-00010-of-00019.safetensors"),
        ("model.layers.63.self_attn.q_proj.weight", "model-00019-of-00019.safetensors"),
    ]

    # Try to find the right shard for each weight
    import json
    with open(model_path / "model.safetensors.index.json") as f:
        index = json.load(f)
    weight_map = index["weight_map"]

    print(f"{'Weight':<55} {'RTN(AbsM)':<12} {'RTN(OptMSE)':<12} {'ITF':<12} {'ITF gain':<12}")
    print("-" * 103)

    for weight_name, _ in test_weights:
        shard_file = weight_map.get(weight_name)
        if shard_file is None:
            print(f"  {weight_name}: NOT FOUND in index")
            continue

        sf = safe_open(str(model_path / shard_file), framework="pt")
        W = sf.get_tensor(weight_name).float()
        print(f"Loading {weight_name} ({list(W.shape)})...", end=" ", flush=True)

        # Method 1: RTN AbsMean
        t0 = time.time()
        T_rtn, alpha_rtn, mu_rtn = quantize_rtn(W)
        t_rtn = time.time() - t0
        stats_rtn = compute_stats(W, T_rtn, alpha_rtn, mu_rtn)

        # Method 2: RTN Optimal MSE
        t0 = time.time()
        T_opt, alpha_opt, mu_opt = quantize_rtn_optimal(W)
        t_opt = time.time() - t0
        stats_opt = compute_stats(W, T_opt, alpha_opt, mu_opt)

        # Method 3: PT2-LLM ITF
        t0 = time.time()
        T_itf, alpha_itf, mu_itf = iterative_ternary_fitting(W, n_iters=10)
        t_itf = time.time() - t0
        stats_itf = compute_stats(W, T_itf, alpha_itf, mu_itf)

        gain = stats_itf["snr_db"] - stats_rtn["snr_db"]
        short_name = weight_name.replace("model.layers.", "L").replace(".self_attn.", ".attn.").replace(".weight", "")
        print(f"\r{short_name:<55} {stats_rtn['snr_db']:>8.2f} dB  {stats_opt['snr_db']:>8.2f} dB  {stats_itf['snr_db']:>8.2f} dB  {gain:>+8.2f} dB")

        # Also print distribution
        print(f"{'':>55} "
              f"({stats_rtn['pct_neg']:.0f}/{stats_rtn['pct_zero']:.0f}/{stats_rtn['pct_pos']:.0f}%)  "
              f"({stats_opt['pct_neg']:.0f}/{stats_opt['pct_zero']:.0f}/{stats_opt['pct_pos']:.0f}%)  "
              f"({stats_itf['pct_neg']:.0f}/{stats_itf['pct_zero']:.0f}/{stats_itf['pct_pos']:.0f}%)")
        print(f"{'':>55} "
              f"t={t_rtn:.1f}s         t={t_opt:.1f}s         t={t_itf:.1f}s")
        print()

        del W, sf
        import gc
        gc.collect()

if __name__ == "__main__":
    main()
