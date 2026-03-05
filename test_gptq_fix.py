#!/usr/bin/env python3
"""
Quick validation: does the fixed GPTQ beat ITF on activation-weighted error?

Tests on synthetic data with known activation distribution, then on real
Pythia-160M weights if available.
"""

import math
import sys
import time
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from quantize_universal import (
    iterative_ternary_fitting,
    activation_aware_alignment,
    gptq_ternary,
    compute_snr,
    compute_weighted_error,
)


def test_synthetic():
    """Test on synthetic weight matrix with known activation covariance."""
    print("=" * 60)
    print("Test 1: Synthetic data (256x512)")
    print("=" * 60)

    torch.manual_seed(42)
    out_f, in_f = 256, 512

    # Realistic weight distribution (roughly normal, low-rank structure)
    W = torch.randn(out_f, in_f) * 0.02

    # Non-trivial Hessian: some features are much more important than others
    # This is where GPTQ should shine - it prioritizes high-activation columns
    importance = torch.exp(torch.linspace(0, 3, in_f))  # exponential importance
    X = torch.randn(1024, in_f) * importance.unsqueeze(0)
    H = (X.T @ X) / X.shape[0]

    # Method 1: ITF only
    T_itf, alpha_itf, mu_itf = iterative_ternary_fitting(W, n_iters=10)
    snr_itf = compute_snr(W, T_itf, alpha_itf, mu_itf)
    werr_itf = compute_weighted_error(W, T_itf, alpha_itf, mu_itf, H)

    # Method 2: ITF + AGA
    alpha_aga, mu_aga = activation_aware_alignment(W, T_itf, H, itf_alpha=alpha_itf, itf_mu=mu_itf)
    snr_aga = compute_snr(W, T_itf, alpha_aga, mu_aga)
    werr_aga = compute_weighted_error(W, T_itf, alpha_aga, mu_aga, H)

    # Method 3: GPTQ
    t0 = time.time()
    T_gptq, alpha_gptq, mu_gptq = gptq_ternary(W, H, block_size=128)
    t_gptq = time.time() - t0

    # AGA on GPTQ result
    alpha_gptq_aga, mu_gptq_aga = activation_aware_alignment(
        W, T_gptq, H, itf_alpha=alpha_gptq, itf_mu=mu_gptq
    )
    snr_gptq = compute_snr(W, T_gptq, alpha_gptq_aga, mu_gptq_aga)
    werr_gptq = compute_weighted_error(W, T_gptq, alpha_gptq_aga, mu_gptq_aga, H)

    print(f"{'Method':<15} {'SNR (dB)':>10} {'Weighted Err':>15} {'Rel Err':>10}")
    print("-" * 52)
    print(f"{'ITF':<15} {snr_itf:>10.2f} {werr_itf:>15.4e} {'1.000':>10}")
    print(f"{'ITF+AGA':<15} {snr_aga:>10.2f} {werr_aga:>15.4e} {werr_aga/werr_itf:>10.3f}")
    print(f"{'GPTQ+AGA':<15} {snr_gptq:>10.2f} {werr_gptq:>15.4e} {werr_gptq/werr_itf:>10.3f}")
    print(f"\nGPTQ time: {t_gptq:.2f}s")

    # Also compute actual output error: ||W@X - W_hat@X||^2
    X_test = torch.randn(32, in_f) * importance.unsqueeze(0)
    Y_true = W @ X_test.T  # [out_f, 32]

    Y_itf = (alpha_itf.unsqueeze(1) * T_itf.float() + mu_itf.unsqueeze(1)) @ X_test.T
    Y_aga = (alpha_aga.unsqueeze(1) * T_itf.float() + mu_aga.unsqueeze(1)) @ X_test.T
    Y_gptq = (alpha_gptq_aga.unsqueeze(1) * T_gptq.float() + mu_gptq_aga.unsqueeze(1)) @ X_test.T

    out_err_itf = ((Y_true - Y_itf) ** 2).mean().item()
    out_err_aga = ((Y_true - Y_aga) ** 2).mean().item()
    out_err_gptq = ((Y_true - Y_gptq) ** 2).mean().item()

    print(f"\nActual output MSE (||W@X - W_hat@X||²):")
    print(f"  ITF:      {out_err_itf:.6e}  (1.000)")
    print(f"  ITF+AGA:  {out_err_aga:.6e}  ({out_err_aga/out_err_itf:.3f})")
    print(f"  GPTQ+AGA: {out_err_gptq:.6e}  ({out_err_gptq/out_err_itf:.3f})")

    gptq_wins = werr_gptq < werr_itf
    print(f"\nGPTQ {'WINS' if gptq_wins else 'LOSES'} on weighted error "
          f"({werr_gptq/werr_itf:.3f}x)")
    return gptq_wins


def test_real_pythia():
    """Test on real Pythia-160M weights if available."""
    try:
        from transformers import AutoModelForCausalLM
    except ImportError:
        print("\nSkipping real model test (transformers not installed)")
        return

    print("\n" + "=" * 60)
    print("Test 2: Real Pythia-160M weights")
    print("=" * 60)

    try:
        model = AutoModelForCausalLM.from_pretrained(
            "EleutherAI/pythia-160m", torch_dtype=torch.float32
        )
    except Exception as e:
        print(f"Skipping: {e}")
        return

    # Test a few different projections
    test_layers = [
        "gpt_neox.layers.0.attention.query_key_value",
        "gpt_neox.layers.0.mlp.dense_h_to_4h",
        "gpt_neox.layers.5.attention.query_key_value",
        "gpt_neox.layers.11.mlp.dense_4h_to_h",
    ]

    for layer_name in test_layers:
        module = dict(model.named_modules()).get(layer_name)
        if module is None:
            continue

        W = module.weight.data.float()
        out_f, in_f = W.shape
        print(f"\n{layer_name}: {list(W.shape)}")

        # Generate calibration-like Hessian (identity + noise since we don't
        # have real activations in this quick test)
        # For a proper test, we'd run calibration through the model
        torch.manual_seed(123)
        X = torch.randn(256, in_f) * 0.1
        H = (X.T @ X) / X.shape[0]

        # ITF
        T_itf, alpha_itf, mu_itf = iterative_ternary_fitting(W, n_iters=10)
        snr_itf = compute_snr(W, T_itf, alpha_itf, mu_itf)
        werr_itf = compute_weighted_error(W, T_itf, alpha_itf, mu_itf, H)

        # GPTQ
        t0 = time.time()
        T_gptq, alpha_gptq, mu_gptq = gptq_ternary(W, H, block_size=128)
        t_gptq = time.time() - t0

        alpha_gptq, mu_gptq = activation_aware_alignment(
            W, T_gptq, H, itf_alpha=alpha_gptq, itf_mu=mu_gptq
        )
        snr_gptq = compute_snr(W, T_gptq, alpha_gptq, mu_gptq)
        werr_gptq = compute_weighted_error(W, T_gptq, alpha_gptq, mu_gptq, H)

        ratio = werr_gptq / max(werr_itf, 1e-30)
        win = "WIN" if ratio < 1.0 else "LOSE"
        print(f"  ITF:  SNR={snr_itf:.2f}dB  weighted_err={werr_itf:.4e}")
        print(f"  GPTQ: SNR={snr_gptq:.2f}dB  weighted_err={werr_gptq:.4e}  "
              f"ratio={ratio:.3f} [{win}] ({t_gptq:.1f}s)")

    del model


if __name__ == "__main__":
    win = test_synthetic()
    test_real_pythia()

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    if win:
        print("GPTQ fix validated: activation-weighted error is lower than ITF.")
        print("Ready to run full pipeline: python quantize_universal.py --model EleutherAI/pythia-160m")
    else:
        print("GPTQ still not beating ITF on weighted error. Further investigation needed.")
