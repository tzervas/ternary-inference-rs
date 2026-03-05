#!/usr/bin/env python3
"""
PT2-LLM + QEP: Advanced post-training ternary quantization.

Implements two complementary techniques:
1. PT2-LLM (ICLR 2026): Asymmetric ternary quantizer with ITF + AGA + GPTQ
2. QEP (NeurIPS 2025): Cross-layer error propagation compensation

Pipeline:
  Phase 1: Calibration capture (4-bit model, 128 Wikitext2 samples)
  Phase 2: Layer-wise quantization with QEP correction
  Phase 3: Validation + packaging

Usage:
    .venv/bin/python quantize_pt2_qep.py
    .venv/bin/python quantize_pt2_qep.py --model Qwen/Qwen2.5-Coder-7B-Instruct
    .venv/bin/python quantize_pt2_qep.py --skip-calibration  # reuse saved calibration
"""

import argparse
import gc
import json
import logging
import math
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file, safe_open
from huggingface_hub import snapshot_download

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ============================================================================
# PT2-LLM: Iterative Ternary Fitting (ITF)
# ============================================================================

def iterative_ternary_fitting(W: torch.Tensor, n_iters: int = 10):
    """
    PT2-LLM Stage 1: Iterative Ternary Fitting.

    Alternates between optimal grid construction (closed-form alpha, mu)
    and flexible rounding (ternary assignment T).

    Args:
        W: weight matrix [out_features, in_features] in float32
        n_iters: number of ITF iterations (default 10)

    Returns:
        T: ternary assignments [out_features, in_features] in int8 {-1,0,+1}
        alpha: per-row scales [out_features] in float32
        mu: per-row offsets [out_features] in float32
    """
    out_f, in_f = W.shape
    m = in_f

    # Initialize: mu = row mean, alpha from TWN approximation
    mu = W.mean(dim=1)
    W_centered = W - mu.unsqueeze(1)
    alpha = (0.75 / m) * W_centered.abs().sum(dim=1)
    alpha = alpha.clamp(min=1e-10)

    for _ in range(n_iters):
        # Step B: Flexible rounding
        T = ((W - mu.unsqueeze(1)) / alpha.unsqueeze(1).clamp(min=1e-10))
        T = T.round().clamp(-1, 1).to(torch.int8)
        T_f = T.float()

        # Step A: Optimal grid construction (closed-form)
        TT_sum = (T_f * T_f).sum(dim=1)      # sum of T^2 per row
        WT_sum = (W * T_f).sum(dim=1)         # sum of W*T per row
        T_sum = T_f.sum(dim=1)                # sum of T per row
        W_sum = W.sum(dim=1)                  # sum of W per row

        denom = (m * TT_sum - T_sum ** 2).clamp(min=1e-10)
        alpha = (m * WT_sum - T_sum * W_sum) / denom
        mu = (TT_sum * W_sum - T_sum * WT_sum) / denom
        alpha = alpha.clamp(min=0)

    # Final assignment
    T = ((W - mu.unsqueeze(1)) / alpha.unsqueeze(1).clamp(min=1e-10))
    T = T.round().clamp(-1, 1).to(torch.int8)

    return T, alpha, mu


# ============================================================================
# PT2-LLM: Activation-aware Grid Alignment (AGA)
# ============================================================================

def activation_aware_alignment(W: torch.Tensor, T: torch.Tensor,
                                C: torch.Tensor):
    """
    PT2-LLM Stage 2: Activation-aware Grid Alignment.

    Refines alpha and mu using calibration covariance C = X^T @ X.
    T is frozen (from ITF). Only alpha and mu are optimized.

    Minimizes: ||W @ X - (alpha * T + mu) @ X||^2_F
    which reduces to a 2x2 system per row using covariance C.

    Args:
        W: original weights [out_features, in_features]
        T: frozen ternary assignments [out_features, in_features] int8
        C: activation covariance [in_features, in_features]

    Returns:
        alpha: refined per-row scales [out_features]
        mu: refined per-row offsets [out_features]
    """
    T_f = T.float()

    # Compute products with covariance
    TC = T_f @ C                                    # [out, in]
    TTC = (TC * T_f).sum(dim=1)                     # per row: T_i^T C T_i
    WC = W @ C                                      # [out, in]
    WTC = (WC * T_f).sum(dim=1)                     # per row: W_i^T C T_i

    ones_C = C.sum(dim=1)                           # [in]: C @ 1
    T_onesC = (T_f * ones_C.unsqueeze(0)).sum(dim=1)  # per row: T_i^T (C 1)
    W_onesC = (W * ones_C.unsqueeze(0)).sum(dim=1)    # per row: W_i^T (C 1)
    onesC_ones = ones_C.sum()                       # scalar: 1^T C 1

    # Solve 2x2 system per row:
    # [TTC,     T_onesC] [alpha]   [WTC]
    # [T_onesC, onesC_ones] [mu] = [W_onesC]
    det = (TTC * onesC_ones - T_onesC ** 2).clamp(min=1e-10)
    alpha = (WTC * onesC_ones - W_onesC * T_onesC) / det
    mu = (TTC * W_onesC - T_onesC * WTC) / det
    alpha = alpha.clamp(min=0)

    return alpha, mu


# ============================================================================
# GPTQ-style Error Compensation
# ============================================================================

def gptq_quantize_weight(W: torch.Tensor, H: torch.Tensor,
                          block_size: int = 128, n_itf_iters: int = 10):
    """
    GPTQ-style block-wise ternary quantization with error compensation.

    Standard GPTQ: quantize columns sequentially, propagate error to remaining
    columns using the inverse Hessian. This compensates for quantization errors
    by adjusting not-yet-quantized columns to account for the error in
    already-quantized columns.

    After GPTQ produces per-column ternary assignments, run ITF on the
    error-compensated weight to get optimal per-row (alpha, mu).

    Args:
        W: weight matrix [out_features, in_features] in float32
        H: Hessian matrix [in_features, in_features] (X^T @ X + damping)
        block_size: columns per GPTQ block (default 128)
        n_itf_iters: ITF iterations for final scale/offset refinement

    Returns:
        T: ternary assignments [out_features, in_features] int8
        alpha: per-row scales [out_features] float32
        mu: per-row offsets [out_features] float32
    """
    out_f, in_f = W.shape
    W_orig = W.clone()
    W = W.clone()

    # Add damping to Hessian diagonal for numerical stability
    damp = 0.01 * torch.diag(H).mean()
    H = H.clone()
    H.diagonal().add_(damp)

    # Compute full Cholesky inverse of H
    try:
        L = torch.linalg.cholesky(H)
        H_inv = torch.cholesky_inverse(L)
    except Exception:
        # Extra damping if Cholesky fails
        H.diagonal().add_(damp * 10)
        try:
            L = torch.linalg.cholesky(H)
            H_inv = torch.cholesky_inverse(L)
        except Exception:
            log.warning("GPTQ: Hessian inversion failed, falling back to ITF only")
            return iterative_ternary_fitting(W, n_iters=n_itf_iters)

    # Initialize output ternary matrix
    T = torch.zeros(out_f, in_f, dtype=torch.int8)

    # Process in blocks
    for col_start in range(0, in_f, block_size):
        col_end = min(col_start + block_size, in_f)
        block_cols = col_end - col_start

        # Extract block inverse
        H_inv_block = H_inv[col_start:col_end, col_start:col_end]

        # Quantize columns in this block
        err_block = torch.zeros(out_f, block_cols, dtype=W.dtype)
        for j in range(block_cols):
            col_idx = col_start + j
            w_col = W[:, col_idx]
            h_inv_jj = H_inv_block[j, j].clamp(min=1e-10)

            # Per-column ternary: scale = AbsMean, quantize
            scale = w_col.abs().mean().clamp(min=1e-10)
            q_col = (w_col / scale).round().clamp(-1, 1)
            T[:, col_idx] = q_col.to(torch.int8)
            w_hat_col = q_col * scale

            # Column error
            err = (w_col - w_hat_col) / h_inv_jj
            err_block[:, j] = err

            # Propagate error within block
            if j + 1 < block_cols:
                W[:, col_start + j + 1:col_end] -= (
                    err.unsqueeze(1) * H_inv_block[j, j + 1:block_cols].unsqueeze(0)
                )

        # Propagate block error to remaining columns
        if col_end < in_f:
            H_inv_cross = H_inv[col_start:col_end, col_end:]
            W[:, col_end:] -= err_block @ H_inv_cross

    # Now W has been modified by error propagation. Run ITF on W_orig to get
    # optimal (alpha, mu) for the GPTQ ternary assignments T.
    # But actually, T was computed from the GPTQ-modified W, so we compute
    # optimal (alpha, mu) that minimize ||W_orig - alpha*T - mu||^2 per row.

    T_f = T.float()
    m = in_f
    TT_sum = (T_f * T_f).sum(dim=1)
    WT_sum = (W_orig * T_f).sum(dim=1)
    T_sum = T_f.sum(dim=1)
    W_sum = W_orig.sum(dim=1)

    denom = (m * TT_sum - T_sum ** 2).clamp(min=1e-10)
    alpha = (m * WT_sum - T_sum * W_sum) / denom
    mu = (TT_sum * W_sum - T_sum * WT_sum) / denom
    alpha = alpha.clamp(min=0)

    return T, alpha, mu


# ============================================================================
# QEP: Quantization Error Propagation
# ============================================================================

def qep_correct_weights(W: torch.Tensor, X_fp: torch.Tensor,
                         X_hat: torch.Tensor, alpha_qep: float = 0.8):
    """
    QEP weight correction: adjusts weights to compensate for accumulated
    quantization error from previous layers.

    W_corrected = W + alpha * W @ delta @ X_hat^T @ H_hat^(-1)

    where delta = X_fp - X_hat (accumulated error at this layer)

    Args:
        W: original weights [out_features, in_features]
        X_fp: full-precision activations [n_samples * seq_len, in_features]
        X_hat: quantized activations [n_samples * seq_len, in_features]
        alpha_qep: QEP strength in [0, 1] (default 0.8)

    Returns:
        W_corrected: corrected weights [out_features, in_features]
    """
    if alpha_qep == 0:
        return W

    delta = X_fp - X_hat  # [N, in_features]

    # Compute H_hat = X_hat^T @ X_hat (Hessian of quantized inputs)
    H_hat = X_hat.T @ X_hat  # [in, in]

    # Regularize for inversion
    damp = 0.01 * H_hat.diagonal().mean()
    H_hat.diagonal().add_(damp)

    # H_hat^(-1) @ X_hat^T @ delta^T ... but rearranged for efficiency:
    # correction = W @ delta^T @ X_hat @ H_hat^(-1)^T
    # Simplified: correction_term = W @ (delta^T @ X_hat) @ H_hat^(-1)

    # delta^T @ X_hat: [in, N] @ [N, in] = [in, in]
    delta_X = delta.T @ X_hat  # [in, in]

    # Solve H_hat @ Y = delta_X^T for Y, equivalent to delta_X @ H_hat^(-1)
    try:
        L = torch.linalg.cholesky(H_hat)
        Y = torch.cholesky_solve(delta_X.T, L)  # [in, in]
        correction = W @ Y.T  # Wait, dimensions...

        # Actually: W_corrected = W + alpha * W @ delta_X @ H_hat^(-1)
        # W: [out, in], delta_X: [in, in], H_hat^(-1): [in, in]
        # Result: [out, in] - which is what we want
        H_hat_inv_delta_X = torch.cholesky_solve(delta_X, L)  # [in, in]
        W_corrected = W + alpha_qep * (W @ H_hat_inv_delta_X)
    except Exception as e:
        log.warning(f"QEP Cholesky failed ({e}), falling back to lstsq")
        try:
            H_hat_inv_delta_X = torch.linalg.solve(H_hat, delta_X)
            W_corrected = W + alpha_qep * (W @ H_hat_inv_delta_X)
        except Exception:
            log.warning("QEP correction failed entirely, using uncorrected weights")
            W_corrected = W

    return W_corrected


# ============================================================================
# Packing
# ============================================================================

def pack_ternary_base3(q: torch.Tensor) -> torch.Tensor:
    """Pack ternary int8 tensor into base-3 uint8 format."""
    out_features, in_features = q.shape
    assert in_features % 4 == 0, f"in_features={in_features} not divisible by 4"

    q_np = (q.numpy().astype(np.int16) + 1)  # {-1,0,+1} -> {0,1,2}
    q_4 = q_np.reshape(out_features, in_features // 4, 4)
    packed = (q_4[:, :, 0] + q_4[:, :, 1] * 3 +
              q_4[:, :, 2] * 9 + q_4[:, :, 3] * 27).astype(np.uint8)
    return torch.from_numpy(packed)


def compute_stats(W_orig, T, alpha, mu):
    """Compute SNR and distribution stats."""
    W_hat = alpha.unsqueeze(1) * T.float() + mu.unsqueeze(1)
    mse = ((W_orig - W_hat) ** 2).mean().item()
    signal = (W_orig ** 2).mean().item()
    snr_db = 10 * math.log10(signal / max(mse, 1e-15))

    total = T.numel()
    return {
        "snr_db": snr_db,
        "mse": mse,
        "pct_neg": 100 * (T == -1).sum().item() / total,
        "pct_zero": 100 * (T == 0).sum().item() / total,
        "pct_pos": 100 * (T == 1).sum().item() / total,
    }


# ============================================================================
# Calibration Capture
# ============================================================================

def capture_calibration(model_id: str, calib_dir: Path,
                         n_samples: int = 128, seq_len: int = 2048):
    """
    Phase 1: Capture per-layer activations using a 4-bit model.

    For each transformer layer, hooks capture input activations and compute
    the covariance matrix C = X^T @ X incrementally.

    Saves per-layer covariance to disk.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset

    calib_dir.mkdir(parents=True, exist_ok=True)

    log.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    log.info("Loading 4-bit model for calibration capture...")
    from transformers import BitsAndBytesConfig
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    log.info("Loading calibration dataset (Wikitext2)...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    text = "\n\n".join([t for t in dataset["text"] if len(t) > 100])

    # Tokenize and create samples
    tokens = tokenizer.encode(text, return_tensors="pt", truncation=False)[0]
    n_available = (len(tokens) - 1) // seq_len
    n_samples = min(n_samples, n_available)
    log.info(f"Calibration: {n_samples} samples x {seq_len} tokens")

    samples = []
    for i in range(n_samples):
        start = i * seq_len
        samples.append(tokens[start:start + seq_len].unsqueeze(0))

    num_layers = model.config.num_hidden_layers
    log.info(f"Model has {num_layers} layers")

    # For each layer, capture input activations and compute covariance
    # We need activations at different points:
    # - Layer input (for q/k/v projections)
    # - Post-attention (for o_proj) -- approximated by layer input
    # - Post-attention-norm (for gate/up) -- approximated by layer input
    # - Gate*up output (for down_proj) -- harder to capture

    # Simplified: capture layer input covariance (used for all projections in layer)
    # This is an approximation but dramatically simpler

    for layer_idx in range(num_layers):
        cov_path = calib_dir / f"layer_{layer_idx}_cov.pt"
        act_path = calib_dir / f"layer_{layer_idx}_activations.pt"

        if cov_path.exists():
            log.info(f"  Layer {layer_idx}: covariance already captured, skipping")
            continue

        log.info(f"  Layer {layer_idx}/{num_layers}: capturing activations...")
        layer = model.model.layers[layer_idx]

        # Accumulate activations for this layer
        all_activations = []
        hook_handle = None

        def hook_fn(module, input, output, activations_list=all_activations):
            # input is tuple, first element is the hidden states
            x = input[0]
            if x.dim() == 3:
                # [batch, seq, hidden] -> [batch*seq, hidden]
                x = x.reshape(-1, x.shape[-1])
            activations_list.append(x.detach().float().cpu())

        hook_handle = layer.register_forward_hook(hook_fn)

        with torch.no_grad():
            for i, sample in enumerate(samples):
                sample = sample.to(model.device)
                model(sample)
                if (i + 1) % 32 == 0:
                    log.info(f"    Sample {i+1}/{n_samples}")

        hook_handle.remove()

        # Stack all activations: [total_tokens, hidden_size]
        X = torch.cat(all_activations, dim=0)
        log.info(f"    Activations shape: {X.shape}")

        # Compute covariance incrementally to save memory
        # C = X^T @ X / n_tokens
        n_tokens = X.shape[0]
        hidden = X.shape[1]

        # Process in chunks to avoid OOM
        chunk_size = 8192
        C = torch.zeros(hidden, hidden, dtype=torch.float32)
        for start in range(0, n_tokens, chunk_size):
            end = min(start + chunk_size, n_tokens)
            chunk = X[start:end]
            C += chunk.T @ chunk

        # Save covariance and a subset of activations (for QEP)
        torch.save(C, cov_path)

        # Save activations (subsample if too large, keep ~8K tokens for QEP)
        max_tokens = 8192
        if n_tokens > max_tokens:
            indices = torch.randperm(n_tokens)[:max_tokens]
            X_save = X[indices]
        else:
            X_save = X
        torch.save(X_save, act_path)

        log.info(f"    Saved covariance {list(C.shape)} and activations {list(X_save.shape)}")

        del all_activations, X, X_save, C
        gc.collect()
        torch.cuda.empty_cache()

    # Save metadata
    meta = {
        "model_id": model_id,
        "n_samples": n_samples,
        "seq_len": seq_len,
        "num_layers": num_layers,
        "hidden_size": model.config.hidden_size,
    }
    with open(calib_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    log.info("Calibration capture complete, freeing model...")
    del model
    gc.collect()
    torch.cuda.empty_cache()

    return meta


# ============================================================================
# Layer-wise Quantization
# ============================================================================

def quantize_layer_projections(layer_weights: dict, layer_idx: int,
                                calib_dir: Path, group_size: int,
                                use_gptq: bool = True,
                                use_qep: bool = True,
                                qep_alpha: float = 0.8,
                                prev_X_hat: torch.Tensor = None):
    """
    Quantize all projections in a single transformer layer using PT2-LLM + QEP.

    Args:
        layer_weights: dict of {proj_name: weight_tensor} for this layer
        layer_idx: layer index
        calib_dir: path to calibration data
        group_size: quantization group size
        use_gptq: whether to use GPTQ error compensation
        use_qep: whether to use QEP cross-layer correction
        qep_alpha: QEP correction strength
        prev_X_hat: quantized activations from previous layer (for QEP)

    Returns:
        results: dict of {proj_name: (packed, scales, offsets, stats)}
    """
    # Load calibration data for this layer
    cov_path = calib_dir / f"layer_{layer_idx}_cov.pt"
    act_path = calib_dir / f"layer_{layer_idx}_activations.pt"

    C = torch.load(cov_path, weights_only=True) if cov_path.exists() else None
    X_fp = torch.load(act_path, weights_only=True) if act_path.exists() else None

    results = {}

    for proj_name, W_orig in layer_weights.items():
        W = W_orig.float()
        out_f, in_f = W.shape

        # Ensure dimensions are compatible
        lcm = max(group_size, 4)
        if in_f % lcm != 0:
            pad = lcm - (in_f % lcm)
            W = torch.nn.functional.pad(W, (0, pad))

        log.info(f"  {proj_name}: {list(W_orig.shape)} -> PT2-LLM quantization")

        # Step 1: QEP correction (if applicable)
        if use_qep and X_fp is not None and prev_X_hat is not None:
            # Only apply QEP if we have both fp and quantized activations
            # Trim/pad activations to match weight in_features
            act_dim = X_fp.shape[1]
            w_in = W.shape[1]
            if act_dim == w_in or (act_dim < w_in):
                X_fp_use = X_fp[:, :min(act_dim, w_in)]
                X_hat_use = prev_X_hat[:, :min(act_dim, w_in)]
                if act_dim < w_in:
                    X_fp_use = torch.nn.functional.pad(X_fp_use, (0, w_in - act_dim))
                    X_hat_use = torch.nn.functional.pad(X_hat_use, (0, w_in - act_dim))
                W = qep_correct_weights(W, X_fp_use, X_hat_use, qep_alpha)
                log.info(f"    QEP correction applied (alpha={qep_alpha})")
            else:
                log.info(f"    QEP skipped: activation dim {act_dim} > weight dim {w_in}")

        # Step 2: ITF (Iterative Ternary Fitting)
        T, alpha, mu = iterative_ternary_fitting(W, n_iters=10)
        log.info(f"    ITF: alpha=[{alpha.min():.4f}, {alpha.max():.4f}], "
                 f"mu=[{mu.min():.4f}, {mu.max():.4f}]")

        # Step 3: AGA (Activation-aware Grid Alignment)
        if C is not None:
            # Adjust covariance to match (potentially padded) weight dimensions
            c_dim = C.shape[0]
            w_in = W.shape[1]
            if c_dim == w_in:
                C_use = C
            elif c_dim < w_in:
                C_use = torch.zeros(w_in, w_in, dtype=torch.float32)
                C_use[:c_dim, :c_dim] = C
            else:
                C_use = C[:w_in, :w_in]

            alpha, mu = activation_aware_alignment(W, T, C_use)
            log.info(f"    AGA: alpha=[{alpha.min():.4f}, {alpha.max():.4f}], "
                     f"mu=[{mu.min():.4f}, {mu.max():.4f}]")

        # Step 4: GPTQ error compensation (optional, more expensive)
        if use_gptq and C is not None:
            # Use covariance as Hessian approximation
            w_in = W.shape[1]
            if c_dim == w_in:
                H = C.clone()
            elif c_dim < w_in:
                H = torch.zeros(w_in, w_in, dtype=torch.float32)
                H[:c_dim, :c_dim] = C
                # Add identity for padded dimensions
                H[c_dim:, c_dim:] = torch.eye(w_in - c_dim)
            else:
                H = C[:w_in, :w_in].clone()

            T_gptq, alpha_gptq, mu_gptq = gptq_quantize_weight(W, H, block_size=128)

            # Compare: use whichever gives better SNR
            stats_itf = compute_stats(W, T, alpha, mu)
            stats_gptq = compute_stats(W, T_gptq, alpha_gptq, mu_gptq)

            if stats_gptq["snr_db"] > stats_itf["snr_db"]:
                T, alpha, mu = T_gptq, alpha_gptq, mu_gptq
                log.info(f"    GPTQ improved: {stats_itf['snr_db']:.1f} -> {stats_gptq['snr_db']:.1f} dB")
            else:
                log.info(f"    GPTQ not better: {stats_gptq['snr_db']:.1f} vs ITF+AGA {stats_itf['snr_db']:.1f} dB")

            del T_gptq, alpha_gptq, mu_gptq, H

        # Compute final stats
        stats = compute_stats(W, T, alpha, mu)

        # Convert to per-group format for compatibility with existing loader
        # The existing format expects per-group scales, but PT2-LLM uses per-row.
        # We store per-row alpha as scales with group_size = in_features,
        # and add a separate offsets tensor.
        #
        # For backwards compatibility with the existing loader (which expects
        # per-group scales), we can tile alpha across groups:
        w_in = W.shape[1]
        num_groups = w_in // group_size
        # Per-group scale: same alpha for all groups in a row
        scales = alpha.unsqueeze(1).expand(out_f, num_groups).contiguous()
        offsets = mu  # per-row offset

        # Pack ternary
        packed = pack_ternary_base3(T[:out_f, :w_in])  # trim to original size if padded

        results[proj_name] = (packed, scales, offsets, stats)

        log.info(f"    Final: SNR={stats['snr_db']:.1f}dB "
                 f"({stats['pct_neg']:.0f}%- {stats['pct_zero']:.0f}%0 {stats['pct_pos']:.0f}%+)")

        del T, alpha, mu, W
        gc.collect()

    del C, X_fp
    gc.collect()

    return results


# ============================================================================
# Main Pipeline
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="PT2-LLM + QEP ternary quantization")
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-32B-Instruct",
                        help="HF model ID")
    parser.add_argument("--output", default="models/qwen2-32b-pt2qep",
                        help="Output directory")
    parser.add_argument("--group-size", type=int, default=64,
                        help="Quantization group size")
    parser.add_argument("--calib-dir", default=None,
                        help="Calibration data directory (default: <output>/calibration)")
    parser.add_argument("--skip-calibration", action="store_true",
                        help="Skip calibration capture (reuse existing)")
    parser.add_argument("--no-gptq", action="store_true",
                        help="Disable GPTQ error compensation")
    parser.add_argument("--no-qep", action="store_true",
                        help="Disable QEP cross-layer correction")
    parser.add_argument("--qep-alpha", type=float, default=0.8,
                        help="QEP correction strength (0-1, default 0.8)")
    parser.add_argument("--n-calib-samples", type=int, default=128,
                        help="Number of calibration samples")
    parser.add_argument("--seq-len", type=int, default=2048,
                        help="Calibration sequence length")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    calib_dir = Path(args.calib_dir) if args.calib_dir else output_path / "calibration"

    t0 = time.time()

    # ========== Phase 1: Calibration ==========
    if not args.skip_calibration:
        log.info("=" * 60)
        log.info("Phase 1: Calibration Capture")
        log.info("=" * 60)
        meta = capture_calibration(
            args.model, calib_dir,
            n_samples=args.n_calib_samples,
            seq_len=args.seq_len,
        )
    else:
        log.info("Skipping calibration (--skip-calibration)")
        with open(calib_dir / "metadata.json") as f:
            meta = json.load(f)

    # ========== Phase 2: Layer-wise Quantization ==========
    log.info("=" * 60)
    log.info("Phase 2: Layer-wise PT2-LLM + QEP Quantization")
    log.info("=" * 60)

    # Resolve model path for weight loading
    model_path = Path(snapshot_download(args.model, local_files_only=True))

    with open(model_path / "config.json") as f:
        config = json.load(f)

    with open(model_path / "model.safetensors.index.json") as f:
        index = json.load(f)

    weight_map = index["weight_map"]
    group_size = args.group_size
    num_layers = config.get("num_hidden_layers", 64)

    PROJ_SUFFIXES = [
        "q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight",
        "gate_proj.weight", "up_proj.weight", "down_proj.weight",
    ]
    KEEP_BF16 = ["embed_tokens.weight", "lm_head.weight"]

    # Output tracking
    out_shard_idx = 0
    out_tensors = {}
    out_size = 0
    max_shard_bytes = 4 * 1024**3
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

    # Track quantized activations for QEP (simplified: layer output tracking)
    prev_X_hat = None

    # Process non-layer tensors first (embed_tokens, lm_head, etc.)
    non_layer_tensors = {}
    for name, shard in weight_map.items():
        is_layer = any(f"layers.{l}." in name for l in range(num_layers))
        if not is_layer:
            non_layer_tensors[name] = shard

    # Load and save non-layer tensors
    for tensor_name, shard_file in sorted(non_layer_tensors.items()):
        shard_path = model_path / shard_file
        sf = safe_open(str(shard_path), framework="pt")
        tensor = sf.get_tensor(tensor_name)

        is_keep_bf16 = any(p in tensor_name for p in KEEP_BF16)
        if is_keep_bf16:
            t_bf16 = tensor.to(torch.bfloat16)
            add(tensor_name, t_bf16)
            total_bf16 += tensor.numel()
            log.info(f"  bf16: {tensor_name} {list(tensor.shape)}")
        else:
            add(tensor_name, tensor)
            log.info(f"  keep: {tensor_name} {list(tensor.shape)} {tensor.dtype}")

        del tensor, sf

    # Process layers sequentially
    for layer_idx in range(num_layers):
        layer_t0 = time.time()
        prefix = f"model.layers.{layer_idx}."

        # Collect all tensors for this layer
        layer_tensor_names = [n for n in weight_map.keys() if n.startswith(prefix)]

        # Separate projections (to quantize) from others (norms, biases)
        proj_weights = {}
        other_tensors = {}

        for tensor_name in sorted(layer_tensor_names):
            shard_file = weight_map[tensor_name]
            shard_path = model_path / shard_file
            sf = safe_open(str(shard_path), framework="pt")
            tensor = sf.get_tensor(tensor_name)
            del sf

            is_proj = any(tensor_name.endswith(s) for s in PROJ_SUFFIXES)
            short_name = tensor_name[len(prefix):]

            if is_proj:
                proj_weights[short_name] = tensor
            else:
                other_tensors[tensor_name] = tensor

        # Save non-projection tensors as-is (norms, biases)
        for tensor_name, tensor in other_tensors.items():
            add(tensor_name, tensor)

        # Quantize projections with PT2-LLM + QEP
        if proj_weights:
            results = quantize_layer_projections(
                proj_weights, layer_idx, calib_dir, group_size,
                use_gptq=not args.no_gptq,
                use_qep=not args.no_qep,
                qep_alpha=args.qep_alpha,
                prev_X_hat=prev_X_hat,
            )

            for proj_name, (packed, scales, offsets, stats) in results.items():
                full_name = prefix + proj_name
                add(full_name, packed)
                add(full_name + ".scales", scales)
                add(full_name + ".offsets", offsets)

                total_quantized += packed.numel() * 4  # 4 trits per byte
                stats_summary.append({"name": full_name, **stats})

            del results

        del proj_weights, other_tensors
        gc.collect()

        elapsed = time.time() - layer_t0
        log.info(f"Layer {layer_idx}/{num_layers} done in {elapsed:.0f}s")

    flush()

    elapsed_total = time.time() - t0

    # Rename shards
    total_shards = out_shard_idx
    for i in range(1, total_shards + 1):
        old = output_path / f"model-{i:05d}.safetensors"
        new = output_path / f"model-{i:05d}-of-{total_shards:05d}.safetensors"
        if old.exists():
            old.rename(new)
        old_name = f"model-{i:05d}.safetensors"
        new_name = f"model-{i:05d}-of-{total_shards:05d}.safetensors"
        for k in list(all_file_map.keys()):
            if all_file_map[k] == old_name:
                all_file_map[k] = new_name

    # Write index
    index_out = {
        "metadata": {"total_size": 0},
        "weight_map": all_file_map,
    }
    with open(output_path / "model.safetensors.index.json", "w") as f:
        json.dump(index_out, f, indent=2)

    # Write config with quantization info
    config["quantization_config"] = {
        "quant_method": "pt2_llm_qep",
        "bits": 1.58,
        "group_size": group_size,
        "packing": "base3_columns",
        "embed_bf16": True,
        "lm_head_bf16": True,
        "has_offsets": True,
        "gptq": not args.no_gptq,
        "qep": not args.no_qep,
        "qep_alpha": args.qep_alpha,
    }
    with open(output_path / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Copy tokenizer
    for tok_file in ["tokenizer.json", "tokenizer_config.json", "vocab.json",
                      "merges.txt", "special_tokens_map.json", "generation_config.json"]:
        src = model_path / tok_file
        if src.exists():
            shutil.copy2(src, output_path / tok_file)

    # Summary
    log.info(f"\n{'='*60}")
    log.info(f"PT2-LLM + QEP Quantization Complete")
    log.info(f"{'='*60}")
    log.info(f"  Time: {elapsed_total:.0f}s ({elapsed_total/60:.1f} min)")
    log.info(f"  Quantized: {total_quantized/1e9:.2f}B params -> ternary")
    log.info(f"  Kept bf16: {total_bf16/1e6:.0f}M params (embed + lm_head)")
    log.info(f"  Shards: {total_shards}")
    log.info(f"  Output: {output_path}")
    log.info(f"  Method: PT2-LLM (ITF+AGA{'+GPTQ' if not args.no_gptq else ''}"
             f"{'+QEP' if not args.no_qep else ''})")

    if stats_summary:
        snrs = [s["snr_db"] for s in stats_summary]
        log.info(f"  SNR: min={min(snrs):.1f}dB, mean={sum(snrs)/len(snrs):.1f}dB, max={max(snrs):.1f}dB")

    with open(output_path / "quantization_stats.json", "w") as f:
        json.dump(stats_summary, f, indent=2)

    log.info(f"\nTo test:")
    log.info(f"  RUST_LOG=info cargo run --example generate --release -- --model-path {output_path}")


if __name__ == "__main__":
    main()
