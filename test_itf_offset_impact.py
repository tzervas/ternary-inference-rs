#!/usr/bin/env python3
"""Check the impact of asymmetric offset (mu) on quantization quality."""

import math
import sys
import torch
from safetensors import safe_open
from huggingface_hub import snapshot_download
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).parent))
from quantize_pt2_qep import iterative_ternary_fitting, compute_stats

model_path = Path(snapshot_download("Qwen/Qwen2.5-Coder-32B-Instruct", local_files_only=True))
with open(model_path / "model.safetensors.index.json") as f:
    weight_map = json.load(f)["weight_map"]

name = "model.layers.0.self_attn.q_proj.weight"
sf = safe_open(str(model_path / weight_map[name]), framework="pt")
W = sf.get_tensor(name).float()
print(f"Weight: {name} {list(W.shape)}")

# ITF with offset
T, alpha, mu = iterative_ternary_fitting(W, n_iters=10)
stats_with_mu = compute_stats(W, T, alpha, mu)
print(f"\nWith offset (mu):")
print(f"  SNR = {stats_with_mu['snr_db']:.2f} dB")
print(f"  mu range: [{mu.min():.6f}, {mu.max():.6f}], mean={mu.mean():.6f}, std={mu.std():.6f}")
print(f"  alpha range: [{alpha.min():.6f}, {alpha.max():.6f}], mean={alpha.mean():.6f}")

# Without offset (force mu=0)
stats_no_mu = compute_stats(W, T, alpha, torch.zeros_like(mu))
print(f"\nWithout offset (mu=0):")
print(f"  SNR = {stats_no_mu['snr_db']:.2f} dB")
print(f"  Offset contribution: {stats_with_mu['snr_db'] - stats_no_mu['snr_db']:.2f} dB")

# What % of rows have significant mu?
significant = (mu.abs() > 0.1 * alpha).sum().item()
print(f"\n  Rows with |mu| > 10% of alpha: {significant}/{W.shape[0]} ({100*significant/W.shape[0]:.0f}%)")

# Row-wise mean analysis
row_means = W.mean(dim=1)
print(f"\n  Row mean range: [{row_means.min():.6f}, {row_means.max():.6f}]")
print(f"  Row mean std: {row_means.std():.6f}")
print(f"  (Offset captures this non-zero mean per row)")
