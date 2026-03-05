#!/usr/bin/env python3
"""
Test GPTQ error compensation on a real weight matrix.
Compare ITF-only vs ITF+GPTQ to measure the GPTQ contribution.
"""

import math
import sys
import time
import torch
from safetensors import safe_open
from huggingface_hub import snapshot_download
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).parent))
from quantize_pt2_qep import (
    iterative_ternary_fitting,
    gptq_quantize_weight,
    compute_stats,
)

model_path = Path(snapshot_download("Qwen/Qwen2.5-Coder-32B-Instruct", local_files_only=True))
with open(model_path / "model.safetensors.index.json") as f:
    weight_map = json.load(f)["weight_map"]

# Test on a few projections
test_names = [
    "model.layers.0.self_attn.q_proj.weight",
    "model.layers.0.mlp.down_proj.weight",
]

for name in test_names:
    sf = safe_open(str(model_path / weight_map[name]), framework="pt")
    W = sf.get_tensor(name).float()
    print(f"\n{'='*70}")
    print(f"Weight: {name} {list(W.shape)}")
    print(f"{'='*70}")

    # Method 1: ITF only
    t0 = time.time()
    T_itf, alpha_itf, mu_itf = iterative_ternary_fitting(W, n_iters=10)
    t_itf = time.time() - t0
    stats_itf = compute_stats(W, T_itf, alpha_itf, mu_itf)
    print(f"ITF only:     SNR = {stats_itf['snr_db']:.2f} dB  ({t_itf:.1f}s)")

    # Method 2: GPTQ (needs a Hessian)
    # Since we don't have calibration data here, use identity + some noise as Hessian
    # This simulates having uniform activation distribution
    in_f = W.shape[1]
    # Use diagonal Hessian (identity) -- simplest case
    H = torch.eye(in_f, dtype=torch.float32)
    # Add small off-diagonal to make it more realistic
    H += 0.01 * torch.randn(in_f, in_f)
    H = H @ H.T  # Make positive definite

    t0 = time.time()
    T_gptq, alpha_gptq, mu_gptq = gptq_quantize_weight(W, H, block_size=128)
    t_gptq = time.time() - t0
    stats_gptq = compute_stats(W, T_gptq, alpha_gptq, mu_gptq)
    print(f"ITF+GPTQ:     SNR = {stats_gptq['snr_db']:.2f} dB  ({t_gptq:.1f}s)")
    gain = stats_gptq["snr_db"] - stats_itf["snr_db"]
    print(f"GPTQ gain:    {gain:+.2f} dB")

    # Note: with identity Hessian, GPTQ reduces to basic error compensation
    # Real gains come with actual activation Hessian from calibration
    print(f"\n  Note: Using identity Hessian (no calibration). Real gains require")
    print(f"  actual activation covariance from calibration data.")

    del W, sf, H
    import gc; gc.collect()
