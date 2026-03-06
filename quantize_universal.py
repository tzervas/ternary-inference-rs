#!/usr/bin/env python3
"""
Universal PT2-LLM ternary quantization with perplexity evaluation.

Architecture-agnostic: works with any HF causal LM (Pythia, LLaMA, Qwen, etc.)

Pipeline:
  1. Load model (fp16 or 4-bit depending on size)
  2. Evaluate baseline perplexity (Wikitext2)
  3. Capture calibration data (per-layer Hessians)
  4. Quantize all linear layers with PT2-LLM (ITF + AGA + GPTQ)
  5. Evaluate quantized perplexity
  6. Save quantized model

Usage:
    .venv/bin/python quantize_universal.py --model EleutherAI/pythia-160m
    .venv/bin/python quantize_universal.py --model EleutherAI/pythia-1b
    .venv/bin/python quantize_universal.py --model EleutherAI/pythia-1b --no-gptq  # ITF only
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

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from datasets import load_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ============================================================================
# Hadamard Rotation (QuIP#-style preprocessing)
# ============================================================================

def hadamard_matrix(n):
    """Generate normalized Hadamard-like orthogonal matrix of size n."""
    # Use randomized Hadamard: H = D @ Had @ P where D is random signs,
    # Had is Walsh-Hadamard, P is random permutation.
    # For non-power-of-2, we use a random orthogonal matrix.
    if n & (n - 1) == 0 and n > 0:  # power of 2
        # Walsh-Hadamard via recursive construction
        H = torch.tensor([[1.0]])
        while H.shape[0] < n:
            H = torch.cat([
                torch.cat([H, H], dim=1),
                torch.cat([H, -H], dim=1),
            ], dim=0)
        return H / math.sqrt(n)
    else:
        # Random orthogonal via QR decomposition
        Q, _ = torch.linalg.qr(torch.randn(n, n))
        return Q


def apply_hadamard_rotation(W, H_hessian, seed=42):
    """
    Apply Hadamard rotation to weight columns and Hessian.
    W' = W @ R^T,  H' = R @ H @ R^T
    For large in_f, uses block-diagonal application to avoid storing full R.
    Returns: W_rotated, H_rotated, R
    """
    in_f = W.shape[1]
    device = W.device
    torch.manual_seed(seed)

    # Random sign flips (generated before Hadamard for reproducibility)
    signs = (torch.randint(0, 2, (in_f,), device=device) * 2 - 1).float()

    if in_f > 2048:
        # Block-diagonal: apply blocks directly without materializing full R
        block_size = 256
        W_rot = W.clone()
        H_rot = H_hessian.clone()
        # Apply sign flips first (diagonal, trivially parallel)
        W_rot = W_rot * signs.unsqueeze(0)
        H_rot = H_rot * signs.unsqueeze(0) * signs.unsqueeze(1)

        for i in range(0, in_f, block_size):
            end = min(i + block_size, in_f)
            bs = end - i
            Hb = hadamard_matrix(bs).to(device)
            # W_rot[:, i:end] = W_rot[:, i:end] @ Hb.T
            W_rot[:, i:end] = W_rot[:, i:end] @ Hb.T
            # H_rot[i:end, :] = Hb @ H_rot[i:end, :]
            H_rot[i:end, :] = Hb @ H_rot[i:end, :]
            # H_rot[:, i:end] = H_rot[:, i:end] @ Hb.T
            H_rot[:, i:end] = H_rot[:, i:end] @ Hb.T

        # Build full R for inverse rotation (needed for W_hat @ R)
        R = torch.zeros(in_f, in_f, device=device)
        for i in range(0, in_f, block_size):
            end = min(i + block_size, in_f)
            bs = end - i
            R[i:end, i:end] = hadamard_matrix(bs).to(device)
        R = R * signs.unsqueeze(0)
    else:
        R = hadamard_matrix(in_f).to(device)
        R = R * signs.unsqueeze(0)
        W_rot = W @ R.T
        H_rot = R @ H_hessian @ R.T

    return W_rot, H_rot, R


# ============================================================================
# PTQTP: Multi Trit-Plane Decomposition (2 or 3 planes)
# ============================================================================

def ptqtp_tri_trit(W: torch.Tensor, group_size: int = 128, max_iters: int = 50,
                   eps: float = 1e-4, H: torch.Tensor = None):
    """
    3-plane trit decomposition: W ≈ α₁·T₁ + α₂·T₂ + α₃·T₃

    27 possible values per element (~4.75 bits/weight + scale overhead).
    With G=128: ~5.5 effective bits. Should achieve <1.02x PPL ratio.

    Returns: T1, T2, T3 (int8), alpha1, alpha2, alpha3 (per-group), group_size
    """
    out_f, in_f = W.shape

    pad = (group_size - in_f % group_size) % group_size
    if pad > 0:
        W_padded = torch.nn.functional.pad(W, (0, pad))
    else:
        W_padded = W
    n_groups = W_padded.shape[1] // group_size
    padded_in = W_padded.shape[1]

    W_g = W_padded.reshape(out_f, n_groups, group_size)

    # Hessian weights
    h_weights = None
    if H is not None:
        h_diag = H.diagonal()
        if h_diag.shape[0] < padded_in:
            h_diag = torch.nn.functional.pad(h_diag, (0, padded_in - h_diag.shape[0]), value=1.0)
        elif h_diag.shape[0] > padded_in:
            h_diag = h_diag[:padded_in]
        h_weights = h_diag.reshape(1, n_groups, group_size).clamp(min=1e-10)
        h_weights = h_weights / (h_weights.mean(dim=2, keepdim=True) + 1e-10)

    # Initialize T1 from sign
    T1 = torch.sign(W_g).to(torch.int8)
    T1[T1 == 0] = 1

    row_mag = W_g.abs().mean(dim=2)
    alpha1 = row_mag.clone()
    alpha2 = (row_mag * 0.3).clone()
    alpha3 = (row_mag * 0.1).clone()

    # T2 from residual after T1
    res1 = W_g - alpha1.unsqueeze(2) * T1.float()
    T2 = torch.sign(res1).to(torch.int8)
    T2[T2 == 0] = 1

    # T3 from residual after T1+T2
    res2 = W_g - alpha1.unsqueeze(2) * T1.float() - alpha2.unsqueeze(2) * T2.float()
    T3 = torch.sign(res2).to(torch.int8)
    T3[T3 == 0] = 1

    lam = torch.full((out_f, n_groups), 0.01, device=W.device)
    lam_max = 1.0
    kappa_thresh = 1e6

    # Pre-compute all 27 combos as tensors for vectorized GPU evaluation
    combo_vals = []
    for t1 in [-1, 0, 1]:
        for t2 in [-1, 0, 1]:
            for t3 in [-1, 0, 1]:
                combo_vals.append((t1, t2, t3))
    # Shape: [27, 3] — all combos as a single tensor on device
    combo_tensor = torch.tensor(combo_vals, dtype=torch.float32, device=W.device)  # [27, 3]

    for iteration in range(max_iters):
        alpha1_old = alpha1.clone()

        T1_f = T1.float()
        T2_f = T2.float()
        T3_f = T3.float()

        # 3x3 weighted least squares per (row, group)
        if h_weights is not None:
            hT1 = h_weights * T1_f
            hT2 = h_weights * T2_f
            hT3 = h_weights * T3_f
            hW = h_weights * W_g
        else:
            hT1 = T1_f
            hT2 = T2_f
            hT3 = T3_f
            hW = W_g

        a11 = (hT1 * T1_f).sum(2)
        a12 = (hT1 * T2_f).sum(2)
        a13 = (hT1 * T3_f).sum(2)
        a22 = (hT2 * T2_f).sum(2)
        a23 = (hT2 * T3_f).sum(2)
        a33 = (hT3 * T3_f).sum(2)
        b1 = (hT1 * W_g).sum(2)  # using W_g not hW
        b2 = (hT2 * W_g).sum(2)
        b3 = (hT3 * W_g).sum(2)

        if h_weights is not None:
            b1 = (hW * T1_f).sum(2)
            b2 = (hW * T2_f).sum(2)
            b3 = (hW * T3_f).sum(2)

        # Solve 3x3 system with regularization: (A + λI)α = b
        # Use batched solve via torch
        A = torch.stack([
            torch.stack([a11 + lam, a12, a13], dim=-1),
            torch.stack([a12, a22 + lam, a23], dim=-1),
            torch.stack([a13, a23, a33 + lam], dim=-1),
        ], dim=-2)  # [out_f, n_groups, 3, 3]
        b = torch.stack([b1, b2, b3], dim=-1).unsqueeze(-1)  # [out_f, n_groups, 3, 1]

        try:
            alpha_vec = torch.linalg.solve(A, b).squeeze(-1)  # [out_f, n_groups, 3]
        except Exception:
            # Fallback: add more regularization
            A.diagonal(dim1=-2, dim2=-1).add_(0.1)
            alpha_vec = torch.linalg.solve(A, b).squeeze(-1)

        alpha1 = alpha_vec[:, :, 0].clamp(min=0)
        alpha2 = alpha_vec[:, :, 1].clamp(min=0)
        alpha3 = alpha_vec[:, :, 2].clamp(min=0)

        # Exhaustive 27-way search — fully vectorized, single kernel per chunk
        # Process in row chunks to bound VRAM: err tensor is [chunk, n_groups, group_size, 27]
        combo_int = torch.tensor(combo_vals, dtype=torch.int8, device=W.device)  # [27, 3]
        ct = combo_tensor.unsqueeze(0)  # [1, 27, 3]

        # Chunk size: target ~8GB peak for the err tensor (RTX 3090 Ti has 24GB)
        # err: chunk_rows * n_groups * group_size * 27 * 4 bytes
        max_bytes = 8 * 1024**3
        elems_per_row = n_groups * group_size * 27
        chunk_rows = max(1, max_bytes // (elems_per_row * 4))
        chunk_rows = min(chunk_rows, out_f)

        for r0 in range(0, out_f, chunk_rows):
            r1 = min(r0 + chunk_rows, out_f)
            # Per-group scales for this chunk: [chunk, n_groups, 1, 3]
            s_chunk = torch.stack([
                alpha1[r0:r1].unsqueeze(2),
                alpha2[r0:r1].unsqueeze(2),
                alpha3[r0:r1].unsqueeze(2),
            ], dim=-1)
            # 27 possible values per (row, group): [chunk, n_groups, 27]
            w_hat_27 = (s_chunk * ct.unsqueeze(0)).sum(dim=-1)
            # Squared error for all combos: [chunk, n_groups, group_size, 27]
            err_all = (W_g[r0:r1].unsqueeze(-1) - w_hat_27.unsqueeze(2)) ** 2
            if h_weights is not None:
                err_all = err_all * h_weights.unsqueeze(-1)  # [1, n_groups, group_size, 1] broadcasts
            # Best combo per element
            best_idx = err_all.argmin(dim=-1).reshape(-1)
            T1[r0:r1] = combo_int[best_idx, 0].reshape(r1 - r0, n_groups, group_size)
            T2[r0:r1] = combo_int[best_idx, 1].reshape(r1 - r0, n_groups, group_size)
            T3[r0:r1] = combo_int[best_idx, 2].reshape(r1 - r0, n_groups, group_size)
            del err_all, w_hat_27, s_chunk, best_idx

        alpha_diff = (alpha1 - alpha1_old).abs().max().item()
        if alpha_diff < eps:
            break

    T1_out = T1.reshape(out_f, -1)[:, :in_f]
    T2_out = T2.reshape(out_f, -1)[:, :in_f]
    T3_out = T3.reshape(out_f, -1)[:, :in_f]

    return T1_out, T2_out, T3_out, alpha1, alpha2, alpha3, group_size


def dequantize_tri_trit(T1, T2, T3, alpha1, alpha2, alpha3, group_size, out_f, in_f):
    """Dequantize 3-plane trit decomposition."""
    n_groups = alpha1.shape[1]
    a1 = alpha1.unsqueeze(2).expand(-1, -1, group_size).reshape(out_f, -1)[:, :in_f]
    a2 = alpha2.unsqueeze(2).expand(-1, -1, group_size).reshape(out_f, -1)[:, :in_f]
    a3 = alpha3.unsqueeze(2).expand(-1, -1, group_size).reshape(out_f, -1)[:, :in_f]
    return a1 * T1.float() + a2 * T2.float() + a3 * T3.float()


def ptqtp_dual_trit(W: torch.Tensor, group_size: int = 128, max_iters: int = 50,
                    eps: float = 1e-4, H: torch.Tensor = None):
    """
    PTQTP-style dual trit-plane decomposition (improved).

    W ≈ diag(alpha1) * T1 + diag(alpha2) * T2

    Key improvements over initial version:
    - Adaptive regularization (condition-number guided, as in PTQTP paper)
    - Activation-aware element search when Hessian H is provided
    - max_iters=50 (paper default) with ε=1e-4 tolerance

    Returns: T1, T2 (int8), alpha1, alpha2 (per-group scales), group_size
    """
    out_f, in_f = W.shape

    # Pad to multiple of group_size
    pad = (group_size - in_f % group_size) % group_size
    if pad > 0:
        W_padded = torch.nn.functional.pad(W, (0, pad))
    else:
        W_padded = W

    n_groups = W_padded.shape[1] // group_size
    padded_in = W_padded.shape[1]

    # Reshape to (out_f, n_groups, group_size)
    W_g = W_padded.reshape(out_f, n_groups, group_size)

    # Compute per-element Hessian weights for activation-aware search
    h_weights = None
    if H is not None:
        h_diag = H.diagonal()
        # Pad Hessian diagonal to match
        if h_diag.shape[0] < padded_in:
            h_diag = torch.nn.functional.pad(h_diag, (0, padded_in - h_diag.shape[0]), value=1.0)
        elif h_diag.shape[0] > padded_in:
            h_diag = h_diag[:padded_in]
        # Reshape to (1, n_groups, group_size) for broadcasting
        h_weights = h_diag.reshape(1, n_groups, group_size).clamp(min=1e-10)
        # Normalize per-group so weights sum to group_size (relative importance)
        h_weights = h_weights / (h_weights.mean(dim=2, keepdim=True) + 1e-10)

    # Initialize T1 from sign, T2 from sign of residual
    T1 = torch.sign(W_g).to(torch.int8)
    T1[T1 == 0] = 1

    # Initialize scales with per-row magnitude
    row_mag = W_g.abs().mean(dim=2)  # (out_f, n_groups)
    alpha1 = row_mag.clone()
    alpha2 = (row_mag * 0.3).clone()  # T2 typically has smaller scale

    # Initial residual for T2
    residual = W_g - alpha1.unsqueeze(2) * T1.float()
    T2 = torch.sign(residual).to(torch.int8)
    T2[T2 == 0] = 1

    # Adaptive regularization per (row, group)
    lam = torch.full((out_f, n_groups), 0.01, device=W.device)
    lam_max = 1.0
    kappa_thresh = 1e6

    for iteration in range(max_iters):
        alpha1_old = alpha1.clone()

        # Step 1: Ridge regression for alpha1, alpha2 (per-group, per-row)
        T1_f = T1.float()
        T2_f = T2.float()

        # 2x2 system per (row, group)
        if h_weights is not None:
            # Weighted regression: use Hessian diagonal as weights
            a11 = (h_weights * T1_f * T1_f).sum(dim=2)
            a12 = (h_weights * T1_f * T2_f).sum(dim=2)
            a22 = (h_weights * T2_f * T2_f).sum(dim=2)
            b1 = (h_weights * T1_f * W_g).sum(dim=2)
            b2 = (h_weights * T2_f * W_g).sum(dim=2)
        else:
            a11 = (T1_f * T1_f).sum(dim=2)
            a12 = (T1_f * T2_f).sum(dim=2)
            a22 = (T2_f * T2_f).sum(dim=2)
            b1 = (T1_f * W_g).sum(dim=2)
            b2 = (T2_f * W_g).sum(dim=2)

        # Adaptive regularization (condition-number guided)
        det_unreg = a11 * a22 - a12 * a12
        trace = a11 + a22
        # Approximate condition number: trace^2 / (4 * det)
        kappa_approx = (trace * trace) / (4 * det_unreg.abs().clamp(min=1e-15))
        # Increase lambda where condition number is high
        lam = torch.where(kappa_approx > kappa_thresh, (lam * 2).clamp(max=lam_max), lam)
        lam = torch.where(kappa_approx < kappa_thresh * 0.1, (lam * 0.5).clamp(min=1e-8), lam)

        det = (a11 + lam) * (a22 + lam) - a12 * a12
        det = det.clamp(min=1e-10)

        alpha1 = ((a22 + lam) * b1 - a12 * b2) / det
        alpha2 = ((a11 + lam) * b2 - a12 * b1) / det

        alpha1 = alpha1.clamp(min=0)
        alpha2 = alpha2.clamp(min=0)

        # Step 2: Exhaustive 9-way search — vectorized with row chunking
        # All 9 combos as tensors on device
        dual_combos = torch.tensor(
            [(-1,-1),(-1,0),(-1,1),(0,-1),(0,0),(0,1),(1,-1),(1,0),(1,1)],
            dtype=torch.float32, device=W.device
        )  # [9, 2]
        dual_combos_int = torch.tensor(
            [(-1,-1),(-1,0),(-1,1),(0,-1),(0,0),(0,1),(1,-1),(1,0),(1,1)],
            dtype=torch.int8, device=W.device
        )  # [9, 2]
        dc = dual_combos.unsqueeze(0)  # [1, 9, 2]

        # Chunk to bound VRAM (~8GB peak for err tensor)
        max_bytes = 8 * 1024**3
        elems_per_row = n_groups * group_size * 9
        chunk_rows = max(1, max_bytes // (elems_per_row * 4))
        chunk_rows = min(chunk_rows, out_f)

        for r0 in range(0, out_f, chunk_rows):
            r1 = min(r0 + chunk_rows, out_f)
            s_chunk = torch.stack([
                alpha1[r0:r1].unsqueeze(2),
                alpha2[r0:r1].unsqueeze(2),
            ], dim=-1)  # [chunk, n_groups, 1, 2]
            w_hat_9 = (s_chunk * dc.unsqueeze(0)).sum(dim=-1)  # [chunk, n_groups, 9]
            err_9 = (W_g[r0:r1].unsqueeze(-1) - w_hat_9.unsqueeze(2)) ** 2
            if h_weights is not None:
                err_9 = err_9 * h_weights.unsqueeze(-1)  # [1, n_groups, group_size, 1] broadcasts
            best_idx = err_9.argmin(dim=-1).reshape(-1)
            T1[r0:r1] = dual_combos_int[best_idx, 0].reshape(r1 - r0, n_groups, group_size)
            T2[r0:r1] = dual_combos_int[best_idx, 1].reshape(r1 - r0, n_groups, group_size)
            del err_9, w_hat_9, s_chunk, best_idx

        # Check convergence
        alpha_diff = (alpha1 - alpha1_old).abs().max().item()
        if alpha_diff < eps:
            break

    # Reshape back to (out_f, in_f)
    T1_out = T1.reshape(out_f, -1)[:, :in_f]
    T2_out = T2.reshape(out_f, -1)[:, :in_f]

    return T1_out, T2_out, alpha1, alpha2, group_size


def gptq_dual_trit(W: torch.Tensor, H: torch.Tensor, group_size: int = 128,
                    block_size: int = 128):
    """
    GPTQ-style column-sequential quantization for dual trit-planes.

    Key advantage over element-wise PTQTP: error propagation via full H_inv
    captures cross-column correlations. When column j is quantized with error δ,
    future columns are adjusted: W[:, j+1:] -= δ * H_inv[j, j+1:] / H_inv[j,j].

    Pipeline:
    1. Quick PTQTP pass for initial per-group scales (α1, α2)
    2. Column-sequential 9-way search with H_inv error propagation
    3. Re-optimize scales per-group after all columns processed

    Returns: T1, T2 (int8), alpha1, alpha2 (per-group), group_size
    """
    out_f, in_f = W.shape
    W_orig = W.clone()

    # Pad to multiple of group_size
    pad = (group_size - in_f % group_size) % group_size
    if pad > 0:
        W_padded = torch.nn.functional.pad(W, (0, pad))
        H_padded = torch.zeros(in_f + pad, in_f + pad)
        H_padded[:in_f, :in_f] = H
        H_padded[in_f:, in_f:] = torch.eye(pad) * H.diagonal().mean()
    else:
        W_padded = W
        H_padded = H
    padded_in = W_padded.shape[1]
    n_groups = padded_in // group_size

    # Step 1: Initialize scales from quick PTQTP (20 iterations)
    _, _, alpha1, alpha2, gs = ptqtp_dual_trit(
        W_padded, group_size=group_size, H=H_padded, max_iters=20
    )

    # Step 2: Compute Hessian inverse
    damp = 0.01 * H_padded.diagonal().mean()
    H_work = H_padded.clone()
    H_work.diagonal().add_(damp)

    try:
        L = torch.linalg.cholesky(H_work)
        H_inv = torch.cholesky_inverse(L)
    except Exception:
        H_work.diagonal().add_(damp * 10)
        try:
            L = torch.linalg.cholesky(H_work)
            H_inv = torch.cholesky_inverse(L)
        except Exception:
            log.warning("gptq_dual_trit: Cholesky failed, returning PTQTP result")
            T1, T2, a1, a2, gs = ptqtp_dual_trit(
                W_padded, group_size=group_size, H=H_padded
            )
            return T1[:, :in_f], T2[:, :in_f], a1, a2, group_size

    # Step 3: Column-sequential GPTQ with 9-way search
    T1 = torch.zeros(out_f, padded_in, dtype=torch.int8)
    T2 = torch.zeros(out_f, padded_in, dtype=torch.int8)
    W_work = W_padded.clone()

    for col_start in range(0, padded_in, block_size):
        col_end = min(col_start + block_size, padded_in)
        n_block = col_end - col_start
        H_inv_block = H_inv[col_start:col_end, col_start:col_end]
        err_block = torch.zeros(out_f, n_block)

        for j_local in range(n_block):
            j = col_start + j_local
            g = j // group_size

            w_col = W_work[:, j]
            h_jj = H_inv_block[j_local, j_local].clamp(min=1e-10)

            a1 = alpha1[:, g]
            a2 = alpha2[:, g]

            # Vectorized 9-way search — single kernel
            _dc = torch.tensor(
                [(-1,-1),(-1,0),(-1,1),(0,-1),(0,0),(0,1),(1,-1),(1,0),(1,1)],
                dtype=torch.float32, device=w_col.device
            )  # [9, 2]
            w_hat_all = a1.unsqueeze(-1) * _dc[:, 0] + a2.unsqueeze(-1) * _dc[:, 1]  # [out_f, 9]
            err_all = (w_col.unsqueeze(-1) - w_hat_all) ** 2  # [out_f, 9]
            best_ci = err_all.argmin(dim=-1)  # [out_f]
            _dc_i = _dc.to(torch.int8)
            best_t1 = _dc_i[best_ci, 0]
            best_t2 = _dc_i[best_ci, 1]

            T1[:, j] = best_t1
            T2[:, j] = best_t2

            # Error propagation (GPTQ core)
            w_hat_col = a1 * best_t1.float() + a2 * best_t2.float()
            err = (w_col - w_hat_col) / h_jj
            err_block[:, j_local] = err

            # Intra-block propagation
            if j_local + 1 < n_block:
                W_work[:, j + 1:col_end] -= (
                    err.unsqueeze(1) * H_inv_block[j_local, j_local + 1:].unsqueeze(0)
                )

        # Inter-block propagation
        if col_end < padded_in:
            H_inv_cross = H_inv[col_start:col_end, col_end:]
            W_work[:, col_end:] -= err_block @ H_inv_cross

    # Step 4: Re-optimize scales per-group using original W
    h_diag = H_padded.diagonal().clamp(min=1e-10)
    W_g = W_padded.reshape(out_f, n_groups, group_size)
    T1_g = T1.reshape(out_f, n_groups, group_size).float()
    T2_g = T2.reshape(out_f, n_groups, group_size).float()
    h_g = h_diag.reshape(1, n_groups, group_size)

    a11 = (h_g * T1_g * T1_g).sum(dim=2)
    a12 = (h_g * T1_g * T2_g).sum(dim=2)
    a22 = (h_g * T2_g * T2_g).sum(dim=2)
    b1 = (h_g * T1_g * W_g).sum(dim=2)
    b2 = (h_g * T2_g * W_g).sum(dim=2)

    lam = 0.001
    det = (a11 + lam) * (a22 + lam) - a12 * a12
    det = det.clamp(min=1e-10)
    alpha1 = (((a22 + lam) * b1 - a12 * b2) / det).clamp(min=0)
    alpha2 = (((a11 + lam) * b2 - a12 * b1) / det).clamp(min=0)

    return T1[:, :in_f], T2[:, :in_f], alpha1, alpha2, group_size


def dequantize_ptqtp(T1, T2, alpha1, alpha2, group_size, out_f, in_f):
    """Dequantize PTQTP dual trit-plane to full precision."""
    # Expand alpha from (out_f, n_groups) to (out_f, in_f)
    n_groups = alpha1.shape[1]
    a1 = alpha1.unsqueeze(2).expand(-1, -1, group_size).reshape(out_f, -1)[:, :in_f]
    a2 = alpha2.unsqueeze(2).expand(-1, -1, group_size).reshape(out_f, -1)[:, :in_f]
    return a1 * T1.float() + a2 * T2.float()


# ============================================================================
# Block Reconstruction: Output-Aware Scale + Rounding Optimization
# ============================================================================

def block_reconstruct_ptqtp(W, T1, T2, alpha1, alpha2, group_size, H,
                             n_steps=200, lr=5e-4, flip_fraction=0.1):
    """
    Block reconstruction for PTQTP: Hessian-weighted scale optimization +
    multi-flip greedy refinement using full Hessian.

    Phase 1: Re-optimize scales using Hessian-weighted closed-form solution
    Phase 2: Greedy flip pass — top K flips per row per iteration (K=5)
             with scale re-optimization every 10 iterations

    Loss: trace((W - W_hat) @ H @ (W - W_hat)^T)
    """
    out_f, in_f = W.shape
    n_groups = alpha1.shape[1]
    device = W.device

    T1_f = T1.float().to(device)
    T2_f = T2.float().to(device)
    W = W.to(device)
    H = H.to(device)

    if H.shape[0] > in_f:
        H = H[:in_f, :in_f]
    elif H.shape[0] < in_f:
        H_pad = torch.eye(in_f, device=device)
        H_pad[:H.shape[0], :H.shape[0]] = H
        H = H_pad

    h_diag = H.diagonal()  # [in_f]

    # --- Phase 1: Hessian-weighted scale re-optimization (closed form) ---
    padded_in = n_groups * group_size
    if padded_in > in_f:
        W_pad = torch.nn.functional.pad(W, (0, padded_in - in_f))
        T1_pad = torch.nn.functional.pad(T1_f, (0, padded_in - in_f))
        T2_pad = torch.nn.functional.pad(T2_f, (0, padded_in - in_f))
        h_pad = torch.nn.functional.pad(h_diag, (0, padded_in - in_f), value=1e-8)
    else:
        W_pad, T1_pad, T2_pad, h_pad = W, T1_f, T2_f, h_diag

    W_g = W_pad.reshape(out_f, n_groups, group_size)
    T1_g = T1_pad.reshape(out_f, n_groups, group_size)
    T2_g = T2_pad.reshape(out_f, n_groups, group_size)
    h_g = h_pad.reshape(1, n_groups, group_size)

    def _solve_scales(T1_g, T2_g):
        a11 = (h_g * T1_g * T1_g).sum(dim=2)
        a12 = (h_g * T1_g * T2_g).sum(dim=2)
        a22 = (h_g * T2_g * T2_g).sum(dim=2)
        b1 = (h_g * T1_g * W_g).sum(dim=2)
        b2 = (h_g * T2_g * W_g).sum(dim=2)
        lam = 0.001
        det = (a11 + lam) * (a22 + lam) - a12 * a12
        det = det.clamp(min=1e-10)
        a1 = (((a22 + lam) * b1 - a12 * b2) / det).clamp(min=0)
        a2 = (((a11 + lam) * b2 - a12 * b1) / det).clamp(min=0)
        return a1, a2

    alpha1, alpha2 = _solve_scales(T1_g, T2_g)

    # --- Phase 2: Greedy single-flip per row (conservative) ---
    T1_work = T1.clone()
    T2_work = T2.clone()

    a1_exp = alpha1.unsqueeze(2).expand(-1, -1, group_size).reshape(out_f, -1)[:, :in_f]
    a2_exp = alpha2.unsqueeze(2).expand(-1, -1, group_size).reshape(out_f, -1)[:, :in_f]

    n_flips = 0

    for flip_iter in range(10):
        W_hat = a1_exp * T1_work.float() + a2_exp * T2_work.float()
        E = W - W_hat
        EH = E @ H  # [out_f, in_f]

        old_val = W_hat  # [out_f, in_f]

        # Vectorized 9-way flip search
        dc_f = torch.tensor(
            [(-1,-1),(-1,0),(-1,1),(0,-1),(0,0),(0,1),(1,-1),(1,0),(1,1)],
            dtype=torch.float32, device=device
        )  # [9, 2]
        dc_i = dc_f.to(torch.int8)

        # Per-group column indices for scale lookup
        group_idx = torch.arange(in_f, device=device) // group_size  # [in_f]
        # Scales expanded: [out_f, in_f]
        # All 9 new values: [out_f, in_f, 9]
        # a1_exp: [out_f, in_f], dc_f[:, 0]: [9] -> [1, 1, 9]
        new_vals = a1_exp.unsqueeze(-1) * dc_f[:, 0] + a2_exp.unsqueeze(-1) * dc_f[:, 1]  # [out_f, in_f, 9]
        d_all = old_val.unsqueeze(-1) - new_vals  # [out_f, in_f, 9]
        # delta_all: [out_f, in_f, 9]
        delta_all = 2 * d_all * EH.unsqueeze(-1) + d_all * d_all * h_diag.unsqueeze(0).unsqueeze(-1)
        # Best combo per element
        best_combo_idx = delta_all.argmin(dim=-1)  # [out_f, in_f]
        best_delta = delta_all.gather(-1, best_combo_idx.unsqueeze(-1)).squeeze(-1)  # [out_f, in_f]
        flat_idx = best_combo_idx.reshape(-1)
        best_t1 = dc_i[flat_idx, 0].reshape(out_f, in_f)
        best_t2 = dc_i[flat_idx, 1].reshape(out_f, in_f)
        del delta_all, d_all, new_vals

        # Per-row: apply SINGLE best flip
        best_col_per_row = best_delta.argmin(dim=1)
        best_delta_per_row = best_delta.gather(1, best_col_per_row.unsqueeze(1)).squeeze(1)

        worth_flipping = best_delta_per_row < -1e-12
        if not worth_flipping.any():
            break

        row_idx = torch.arange(out_f, device=device)[worth_flipping]
        col_idx = best_col_per_row[worth_flipping]
        n_flips += len(row_idx)

    # Final scale re-optimization
    T1_gf = torch.nn.functional.pad(T1_work.float(), (0, max(0, padded_in - in_f))).reshape(out_f, n_groups, group_size)
    T2_gf = torch.nn.functional.pad(T2_work.float(), (0, max(0, padded_in - in_f))).reshape(out_f, n_groups, group_size)
    alpha1, alpha2 = _solve_scales(T1_gf, T2_gf)

    return T1_work, T2_work, alpha1, alpha2, n_flips


# ============================================================================
# PT2-LLM Core: ITF + AGA
# ============================================================================

def iterative_ternary_fitting(W: torch.Tensor, n_iters: int = 10):
    """ITF: Iterative Ternary Fitting with asymmetric (alpha, mu) quantizer."""
    out_f, in_f = W.shape
    m = float(in_f)

    mu = W.mean(dim=1)
    W_centered = W - mu.unsqueeze(1)
    alpha = (0.75 / m) * W_centered.abs().sum(dim=1)
    alpha = alpha.clamp(min=1e-10)

    for _ in range(n_iters):
        T = ((W - mu.unsqueeze(1)) / alpha.unsqueeze(1).clamp(min=1e-10))
        T = T.round().clamp(-1, 1)
        T_f = T

        TT_sum = (T_f * T_f).sum(dim=1)
        WT_sum = (W * T_f).sum(dim=1)
        T_sum = T_f.sum(dim=1)
        W_sum = W.sum(dim=1)

        denom = (m * TT_sum - T_sum ** 2).clamp(min=1e-10)
        alpha = (m * WT_sum - T_sum * W_sum) / denom
        mu = (TT_sum * W_sum - T_sum * WT_sum) / denom
        alpha = alpha.clamp(min=0)

    T = ((W - mu.unsqueeze(1)) / alpha.unsqueeze(1).clamp(min=1e-10))
    T = T.round().clamp(-1, 1).to(torch.int8)
    return T, alpha, mu


def activation_aware_alignment(W, T, H, itf_alpha=None, itf_mu=None):
    """
    AGA: refine alpha, mu using Hessian H = X^T @ X.
    Falls back to ITF values per-row when AGA produces worse SNR.
    """
    T_f = T.float()
    TH = T_f @ H
    THT = (TH * T_f).sum(dim=1)
    WH = W @ H
    WHT = (WH * T_f).sum(dim=1)

    ones_H = H.sum(dim=1)
    T_onesH = (T_f * ones_H.unsqueeze(0)).sum(dim=1)
    W_onesH = (W * ones_H.unsqueeze(0)).sum(dim=1)
    onesH_ones = ones_H.sum()

    det = THT * onesH_ones - T_onesH ** 2

    # Only use AGA for rows where determinant is well-conditioned
    well_conditioned = det.abs() > 1e-6
    alpha = torch.zeros_like(det)
    mu = torch.zeros_like(det)

    if well_conditioned.any():
        safe_det = det.clone()
        safe_det[~well_conditioned] = 1.0  # avoid div by zero
        alpha_aga = (WHT * onesH_ones - W_onesH * T_onesH) / safe_det
        mu_aga = (THT * W_onesH - T_onesH * WHT) / safe_det
        alpha_aga = alpha_aga.clamp(min=0)

        alpha[well_conditioned] = alpha_aga[well_conditioned]
        mu[well_conditioned] = mu_aga[well_conditioned]

    # For ill-conditioned rows, use ITF values
    if itf_alpha is not None:
        alpha[~well_conditioned] = itf_alpha[~well_conditioned]
        mu[~well_conditioned] = itf_mu[~well_conditioned]
    else:
        # Compute simple per-row scale as fallback
        TT_sum = (T_f * T_f).sum(dim=1).clamp(min=1)
        WT_sum = (W * T_f).sum(dim=1)
        alpha[~well_conditioned] = (WT_sum / TT_sum)[~well_conditioned].clamp(min=0)

    # Per-row quality check: if AGA is worse than ITF, revert that row
    if itf_alpha is not None:
        W_hat_aga = alpha.unsqueeze(1) * T_f + mu.unsqueeze(1)
        W_hat_itf = itf_alpha.unsqueeze(1) * T_f + itf_mu.unsqueeze(1)
        err_aga = ((W - W_hat_aga) ** 2).sum(dim=1)
        err_itf = ((W - W_hat_itf) ** 2).sum(dim=1)
        worse = err_aga > err_itf
        if worse.any():
            alpha[worse] = itf_alpha[worse]
            mu[worse] = itf_mu[worse]

    return alpha, mu


def _ssr_select_block(W_remaining, block_size):
    """
    SSR: Structural Similarity-based Reordering.
    From remaining columns, select the top-k most similar to the mean reference.
    Returns indices into the remaining columns.
    """
    n_remaining = W_remaining.shape[1]
    if n_remaining <= block_size:
        return list(range(n_remaining))

    # Mean reference vector
    w_mean = W_remaining.mean(dim=1)  # [out_f]
    w_mean_norm = w_mean.norm()
    if w_mean_norm < 1e-10:
        return list(range(block_size))

    # Cosine similarity between each column and the mean
    col_norms = W_remaining.norm(dim=0)  # [n_remaining]
    similarities = (W_remaining.T @ w_mean) / (col_norms * w_mean_norm).clamp(min=1e-10)

    # Select top-k most similar columns
    _, indices = similarities.abs().topk(block_size)
    return indices.sort().values.tolist()


def gptq_ternary(W, H, block_size=128, use_ssr=True):
    """
    GPTQ-style error compensation for ternary quantization with SSR.

    Pipeline: ITF → GPTQ with SSR column reordering → re-optimize (alpha, mu)

    SSR (Structural Similarity Reordering): before each GPTQ block, select
    the top-k columns most similar to the mean of remaining columns. This
    clusters outliers together, preventing them from distorting normal columns.
    """
    out_f, in_f = W.shape
    W_orig = W.clone()

    # Prepare Hessian inverse
    damp = 0.01 * H.diagonal().mean()
    H_work = H.clone()
    H_work.diagonal().add_(damp)

    try:
        L = torch.linalg.cholesky(H_work)
        H_inv = torch.cholesky_inverse(L)
    except Exception:
        H_work.diagonal().add_(damp * 10)
        try:
            L = torch.linalg.cholesky(H_work)
            H_inv = torch.cholesky_inverse(L)
        except Exception:
            log.warning("GPTQ: Cholesky failed, falling back to ITF")
            return iterative_ternary_fitting(W, n_iters=10)

    # SSR: Build column processing order
    if use_ssr:
        # Build the full permutation by iteratively selecting blocks
        perm = []
        remaining = list(range(in_f))
        W_work = W.clone()
        while len(remaining) > 0:
            k = min(block_size, len(remaining))
            W_rem = W_work[:, remaining]
            block_local_idx = _ssr_select_block(W_rem, k)
            block_global_idx = [remaining[i] for i in block_local_idx]
            perm.extend(block_global_idx)
            for idx in sorted(block_local_idx, reverse=True):
                remaining.pop(idx)

        perm = torch.tensor(perm, dtype=torch.long)
        inv_perm = torch.empty_like(perm)
        inv_perm[perm] = torch.arange(in_f)

        # Permute W and H
        W = W_orig[:, perm].clone()
        H_inv = H_inv[perm][:, perm]
    else:
        W = W_orig.clone()
        perm = None

    T = torch.zeros(out_f, in_f, dtype=torch.int8)

    # Initial per-row (alpha, mu) from ITF
    _, alpha_cur, mu_cur = iterative_ternary_fitting(W, n_iters=10)

    # Process columns in blocks
    for col_start in range(0, in_f, block_size):
        col_end = min(col_start + block_size, in_f)
        block_cols = col_end - col_start
        H_inv_block = H_inv[col_start:col_end, col_start:col_end]

        err_block = torch.zeros(out_f, block_cols, dtype=W.dtype)

        for j in range(block_cols):
            col_idx = col_start + j
            w_col = W[:, col_idx]
            h_jj = H_inv_block[j, j].clamp(min=1e-10)

            t_col = ((w_col - mu_cur) / alpha_cur.clamp(min=1e-10)).round().clamp(-1, 1)
            T[:, col_idx] = t_col.to(torch.int8)

            w_hat_col = alpha_cur * t_col + mu_cur
            err = (w_col - w_hat_col) / h_jj
            err_block[:, j] = err

            if j + 1 < block_cols:
                W[:, col_start + j + 1:col_end] -= (
                    err.unsqueeze(1) * H_inv_block[j, j + 1:].unsqueeze(0)
                )

        # Inter-block propagation
        if col_end < in_f:
            H_inv_cross = H_inv[col_start:col_end, col_end:]
            W[:, col_end:] -= err_block @ H_inv_cross

            # Re-run ITF on remaining modified columns
            W_remaining = W[:, col_end:]
            if W_remaining.shape[1] >= 4:
                _, alpha_rem, mu_rem = iterative_ternary_fitting(
                    W_remaining, n_iters=5
                )
                alpha_cur = alpha_rem
                mu_cur = mu_rem

    # Un-permute T back to original column order
    if perm is not None:
        T = T[:, inv_perm]

    # Re-compute optimal (alpha, mu) for the final T against original W
    T_f = T.float()
    m = float(in_f)
    TT_sum = (T_f * T_f).sum(dim=1)
    WT_sum = (W_orig * T_f).sum(dim=1)
    T_sum = T_f.sum(dim=1)
    W_sum = W_orig.sum(dim=1)
    denom = (m * TT_sum - T_sum ** 2).clamp(min=1e-10)
    alpha = (m * WT_sum - T_sum * W_sum) / denom
    mu = (TT_sum * W_sum - T_sum * WT_sum) / denom
    alpha = alpha.clamp(min=0)

    return T, alpha, mu


def greedy_bitflip(W, T, alpha, mu, H, n_iters=3):
    """
    Greedy bit-flip optimization using FULL Hessian (not diagonal approximation).

    For each element (i,j), the change in activation-weighted error when flipping
    T[i,j] from t_old to t_new (with fixed alpha, mu) is:
        delta = 2 * d * (EH)[i,j] + d^2 * H[j,j]
    where d = alpha[i] * (t_old - t_new) and EH = E @ H.

    This preserves GPTQ's cross-column optimization because it uses the full H.
    """
    T_f = T.float().clone()
    out_f, in_f = W.shape
    h_diag = H.diagonal()  # [in_f]

    for iteration in range(n_iters):
        W_hat = alpha.unsqueeze(1) * T_f + mu.unsqueeze(1)
        E = W - W_hat  # [out_f, in_f]
        EH = E @ H  # [out_f, in_f] - full Hessian product

        best_val = T_f.clone()
        best_delta = torch.zeros_like(T_f)

        for new_val in [-1.0, 0.0, 1.0]:
            # d[i,j] = alpha[i] * (T_f[i,j] - new_val)
            d = alpha.unsqueeze(1) * (T_f - new_val)
            # delta[i,j] = 2 * d * EH[i,j] + d^2 * H[j,j]
            delta = 2 * d * EH + d * d * h_diag.unsqueeze(0)
            # Negative delta means improvement
            better = delta < best_delta
            # Skip elements that are already at this value
            same = (T_f == new_val)
            better = better & ~same
            best_val = torch.where(better, torch.tensor(new_val), best_val)
            best_delta = torch.where(better, delta, best_delta)

        changed = (best_val != T_f)
        n_flips = changed.sum().item()
        if n_flips == 0:
            break

        T_f = best_val

        # Re-optimize alpha, mu
        m = float(in_f)
        TT_sum = (T_f * T_f).sum(dim=1)
        WT_sum = (W * T_f).sum(dim=1)
        T_sum = T_f.sum(dim=1)
        W_sum = W.sum(dim=1)
        denom = (m * TT_sum - T_sum ** 2).clamp(min=1e-10)
        alpha = (m * WT_sum - T_sum * W_sum) / denom
        mu = (TT_sum * W_sum - T_sum * WT_sum) / denom
        alpha = alpha.clamp(min=0)

    T = T_f.to(torch.int8)
    return T, alpha, mu


# ============================================================================
# Perplexity Evaluation
# ============================================================================

@torch.no_grad()
def evaluate_perplexity(model, tokenizer, device, max_samples=40, seq_len=2048):
    """Evaluate perplexity on Wikitext2 test set."""
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join([t for t in dataset["text"] if len(t) > 50])
    tokens = tokenizer.encode(text, return_tensors="pt", truncation=False)[0]

    n_samples = min(max_samples, (len(tokens) - 1) // seq_len)
    nlls = []

    for i in range(n_samples):
        start = i * seq_len
        input_ids = tokens[start:start + seq_len].unsqueeze(0).to(device)
        target_ids = tokens[start + 1:start + seq_len + 1].unsqueeze(0).to(device)

        outputs = model(input_ids)
        logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
        logits = logits[:, :seq_len, :]

        loss = nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            target_ids.reshape(-1),
            reduction="mean",
        )
        nlls.append(loss.item())

        if (i + 1) % 10 == 0:
            ppl_so_far = math.exp(sum(nlls) / len(nlls))
            log.info(f"  PPL progress: {i+1}/{n_samples} samples, running PPL = {ppl_so_far:.2f}")

    avg_nll = sum(nlls) / len(nlls)
    ppl = math.exp(avg_nll)
    return ppl


# ============================================================================
# Calibration Data
# ============================================================================

def get_calibration_samples(tokenizer, n_samples=128, seq_len=2048):
    """Load calibration samples from Wikitext2 train set."""
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    text = "\n\n".join([t for t in dataset["text"] if len(t) > 100])
    tokens = tokenizer.encode(text, return_tensors="pt", truncation=False)[0]
    n_available = (len(tokens) - 1) // seq_len
    n_samples = min(n_samples, n_available)

    samples = []
    for i in range(n_samples):
        start = i * seq_len
        samples.append(tokens[start:start + seq_len].unsqueeze(0))
    return samples


def capture_layer_hessian(model, layer_name, module, samples, device, n_samples=32):
    """
    Capture Hessian for a single layer by running calibration through
    the (partially quantized) model. Accumulates X^T X on GPU for speed.
    """
    in_features = module.in_features
    # Accumulate on GPU — outer product matmul is much faster on GPU
    hessian_device = device if torch.cuda.is_available() else 'cpu'
    H = torch.zeros(in_features, in_features, dtype=torch.float32, device=hessian_device)
    count = [0]

    def hook_fn(mod, input, output):
        x = input[0].detach().float().to(hessian_device)
        if x.dim() == 3:
            x = x.reshape(-1, x.shape[-1])
        H.add_(x.T @ x)
        count[0] += x.shape[0]

    hook = module.register_forward_hook(hook_fn)

    model.eval()
    with torch.no_grad():
        for i, sample in enumerate(samples[:n_samples]):
            sample = sample.to(device)
            model(sample)

    hook.remove()
    if count[0] > 0:
        H.div_(count[0])
    return H


# ============================================================================
# Sequential Layer-by-Layer Quantization
# ============================================================================

def _is_first_or_last_layer(name, model):
    """Check if a linear layer belongs to the first or last transformer layer."""
    # Common naming patterns: layers.0, layers.N-1, h.0, h.N-1
    import re
    match = re.search(r'layers?\.(\d+)', name)
    if match:
        layer_idx = int(match.group(1))
        # Find max layer index
        max_idx = 0
        for n, _ in model.named_modules():
            m = re.search(r'layers?\.(\d+)', n)
            if m:
                max_idx = max(max_idx, int(m.group(1)))
        return layer_idx == 0 or layer_idx == max_idx
    return False


def quantize_sequential(model, tokenizer, device, use_gptq=True,
                        n_calib=128, seq_len=2048, n_hessian_samples=32,
                        skip_first_last=False, use_hadamard=False,
                        use_bitflip=False, bitflip_iters=3,
                        use_ptqtp=False, ptqtp_group_size=128,
                        use_tri_trit=False):
    """
    Quantize layers SEQUENTIALLY: after quantizing layer i, the Hessian
    for layer i+1 is captured through the already-quantized layers.
    This naturally adapts to accumulated quantization error.
    """
    samples = get_calibration_samples(tokenizer, n_calib, seq_len)

    # Find all linear layers in order, skip embedding/lm_head
    skip_patterns = ["embed", "lm_head", "head"]
    linear_layers = []
    skipped = []
    def _is_linear(m):
        """Check if module is a linear layer (including BnB 4-bit)."""
        if isinstance(m, nn.Linear):
            return True
        try:
            import bitsandbytes as bnb
            return isinstance(m, (bnb.nn.Linear4bit, bnb.nn.Linear8bitLt))
        except ImportError:
            return False

    for name, module in model.named_modules():
        if _is_linear(module):
            if any(p in name.lower() for p in skip_patterns):
                skipped.append(name)
            elif skip_first_last and _is_first_or_last_layer(name, model):
                skipped.append(name)
            else:
                linear_layers.append((name, module))

    if skipped:
        log.info(f"Skipping {len(skipped)} sensitive layers: {skipped}")
    log.info(f"Found {len(linear_layers)} linear layers to quantize sequentially")

    def _replace_weight(model, name, module, W_hat):
        """Replace module weight, handling BnB 4-bit by converting to nn.Linear."""
        try:
            import bitsandbytes as bnb
            is_4bit = isinstance(module, (bnb.nn.Linear4bit, bnb.nn.Linear8bitLt))
        except ImportError:
            is_4bit = False

        if is_4bit:
            # Replace BnB module with standard nn.Linear
            out_f, in_f = W_hat.shape
            bias = module.bias is not None
            dev = next(module.parameters()).device
            # Match the dtype used by the model's non-quantized layers
            compute_dtype = torch.bfloat16  # Most HF models use bf16
            new_linear = nn.Linear(in_f, out_f, bias=bias, device='cpu',
                                   dtype=compute_dtype)
            new_linear.weight.data = W_hat.to(compute_dtype).to(dev)
            new_linear = new_linear.to(dev)
            if bias:
                new_linear.bias.data = module.bias.data.to(compute_dtype).to(dev)
            # Replace in model
            parts = name.rsplit('.', 1)
            if len(parts) == 2:
                parent = dict(model.named_modules())[parts[0]]
                setattr(parent, parts[1], new_linear)
            else:
                setattr(model, name, new_linear)
            return new_linear
        else:
            module.weight.data = W_hat.to(module.weight.dtype).to(module.weight.device)
            return module

    stats_all = []
    for layer_idx, (name, module) in enumerate(linear_layers):
        # Capture Hessian through the partially-quantized model
        H = capture_layer_hessian(model, name, module, samples, device,
                                  n_samples=n_hessian_samples)

        # Extract weight - handle BnB 4-bit quantized weights
        try:
            import bitsandbytes as bnb
            if hasattr(module.weight, 'quant_state') and module.weight.quant_state is not None:
                W = bnb.functional.dequantize_4bit(
                    module.weight.data, module.weight.quant_state
                ).float().cpu()
            else:
                W = module.weight.data.float().cpu()
        except (ImportError, AttributeError):
            W = module.weight.data.float().cpu()
        out_f, in_f = W.shape

        # Pad to multiple of 4 for packing
        if in_f % 4 != 0:
            pad = 4 - (in_f % 4)
            W = torch.nn.functional.pad(W, (0, pad))

        # Pad Hessian if needed
        h_dim = H.shape[0]
        w_in = W.shape[1]
        if h_dim < w_in:
            H_padded = torch.zeros(w_in, w_in)
            H_padded[:h_dim, :h_dim] = H
            H_padded[h_dim:, h_dim:] = torch.eye(w_in - h_dim)
            H_use = H_padded
        elif h_dim > w_in:
            H_use = H[:w_in, :w_in]
        else:
            H_use = H

        # Move to GPU for faster quantization if available
        quant_device = device if torch.cuda.is_available() else 'cpu'
        W_gpu = W.to(quant_device)
        H_gpu = H_use.to(quant_device)

        if use_tri_trit:
            # 3-plane trit decomposition (27 values, ~4.75 bits/weight)
            R = None
            W_q = W_gpu
            H_q = H_gpu
            if use_hadamard:
                W_q, H_q, R = apply_hadamard_rotation(W_gpu, H_gpu, seed=42 + layer_idx)

            T1, T2, T3, a1, a2, a3, gs = ptqtp_tri_trit(
                W_q, group_size=ptqtp_group_size, H=H_q
            )

            W_hat = dequantize_tri_trit(T1, T2, T3, a1, a2, a3, gs, out_f, in_f)
            W_hat = W_hat[:out_f, :in_f]

            if R is not None:
                W_hat = W_hat @ R
                method = "TriTrit+Had"
            else:
                method = "TriTrit"

            W_hat = W_hat.cpu()
            mse = ((W[:out_f, :in_f] - W_hat) ** 2).mean().item()
            signal = (W[:out_f, :in_f] ** 2).mean().item()
            best_snr = 10 * math.log10(signal / max(mse, 1e-15))
            snr_itf = best_snr
            snr_aga = best_snr
            snr_gptq = None

            _replace_weight(model, name, module, W_hat)
        elif use_ptqtp:
            # PTQTP: Dual trit-plane decomposition
            R = None
            W_q = W_gpu
            H_q = H_gpu
            if use_hadamard:
                W_q, H_q, R = apply_hadamard_rotation(W_gpu, H_gpu, seed=42 + layer_idx)

            T1, T2, a1, a2, gs = ptqtp_dual_trit(
                W_q, group_size=ptqtp_group_size, H=H_q
            )

            # Block reconstruction: greedy flip refinement
            T1, T2, a1, a2, n_flips = block_reconstruct_ptqtp(
                W_q[:out_f, :in_f], T1, T2, a1, a2, gs, H_q,
                n_steps=200, lr=1e-3
            )

            W_hat = dequantize_ptqtp(T1, T2, a1, a2, gs, out_f, in_f)
            W_hat = W_hat[:out_f, :in_f]

            if R is not None:
                W_hat = W_hat @ R
                method = f"PTQTP+BR+Had({n_flips}f)"
            else:
                method = f"PTQTP+BR({n_flips}f)"

            W_hat = W_hat.cpu()
            mse = ((W[:out_f, :in_f] - W_hat) ** 2).mean().item()
            signal = (W[:out_f, :in_f] ** 2).mean().item()
            best_snr = 10 * math.log10(signal / max(mse, 1e-15))
            snr_itf = best_snr
            snr_aga = best_snr
            snr_gptq = None

            _replace_weight(model, name, module, W_hat)
        else:
            # PT2-LLM pipeline: ITF + AGA + GPTQ
            # Optional: Hadamard rotation for incoherence processing
            R = None
            W_q = W_gpu
            H_q = H_gpu
            if use_hadamard:
                W_q, H_q, R = apply_hadamard_rotation(W_gpu, H_gpu, seed=42 + layer_idx)

            # Step 1: ITF
            T, alpha, mu = iterative_ternary_fitting(W_q, n_iters=10)
            snr_itf = compute_snr(W_q, T, alpha, mu)

            # Step 2: AGA
            alpha, mu = activation_aware_alignment(W_q, T, H_q, itf_alpha=alpha, itf_mu=mu)
            snr_aga = compute_snr(W_q, T, alpha, mu)

            # Step 3: GPTQ
            best_T, best_alpha, best_mu = T, alpha, mu
            best_snr = snr_aga

            if use_gptq:
                T_g, alpha_g, mu_g = gptq_ternary(W_q, H_q)
                alpha_g, mu_g = activation_aware_alignment(W_q, T_g, H_q,
                                                           itf_alpha=alpha_g, itf_mu=mu_g)
                snr_gptq = compute_snr(W_q, T_g, alpha_g, mu_g)

                err_itf = compute_weighted_error(W_q, T, alpha, mu, H_q)
                err_gptq = compute_weighted_error(W_q, T_g, alpha_g, mu_g, H_q)

                if err_gptq < err_itf:
                    best_T, best_alpha, best_mu = T_g, alpha_g, mu_g
                    best_snr = snr_gptq
                    method = "GPTQ+AGA"
                else:
                    method = "ITF+AGA"
            else:
                method = "ITF+AGA"
                snr_gptq = None

            # Step 4: Greedy bitflip refinement (full Hessian)
            if use_bitflip:
                err_before = compute_weighted_error(W_q, best_T, best_alpha, best_mu, H_q)
                T_bf, alpha_bf, mu_bf = greedy_bitflip(
                    W_q, best_T, best_alpha, best_mu, H_q, n_iters=bitflip_iters
                )
                err_after = compute_weighted_error(W_q, T_bf, alpha_bf, mu_bf, H_q)
                if err_after < err_before:
                    best_T, best_alpha, best_mu = T_bf, alpha_bf, mu_bf
                    best_snr = compute_snr(W_q, best_T, best_alpha, best_mu)
                    method += "+BF"

            if use_hadamard:
                method += "+Had"

            # Replace the linear layer IN-PLACE
            W_hat = best_alpha.unsqueeze(1) * best_T.float() + best_mu.unsqueeze(1)
            if R is not None:
                W_hat = W_hat @ R
            W_hat = W_hat[:out_f, :in_f].cpu()
            _replace_weight(model, name, module, W_hat)

        stats = {
            "name": name,
            "shape": [out_f, in_f],
            "snr_itf": snr_itf,
            "snr_aga": snr_aga,
            "snr_gptq": snr_gptq if (not use_ptqtp and use_gptq) else None,
            "snr_final": best_snr,
            "method": method,
        }
        stats_all.append(stats)

        log.info(f"  [{layer_idx+1}/{len(linear_layers)}] {name}: {[out_f, in_f]} "
                 f"SNR={best_snr:.2f}dB ({method})"
                 f" [ITF={snr_itf:.1f}, AGA={snr_aga:.1f}"
                 f"{f', GPTQ={snr_gptq:.1f}' if snr_gptq is not None else ''}]")

        del W, H, H_use, W_gpu, H_gpu
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    return stats_all


def compute_snr(W, T, alpha, mu):
    W_hat = alpha.unsqueeze(1) * T.float() + mu.unsqueeze(1)
    mse = ((W - W_hat) ** 2).mean().item()
    signal = (W ** 2).mean().item()
    return 10 * math.log10(signal / max(mse, 1e-15))


def compute_weighted_error(W, T, alpha, mu, H):
    """
    Compute activation-weighted output error: trace((W-W_hat) @ H @ (W-W_hat)^T).
    This is what GPTQ actually optimizes, not per-weight MSE.
    """
    W_hat = alpha.unsqueeze(1) * T.float() + mu.unsqueeze(1)
    E = W - W_hat  # [out_f, in_f]
    # EH = E @ H  -> [out_f, in_f]
    # trace(EH @ E^T) = sum of element-wise (EH * E)
    EH = E @ H
    return (EH * E).sum().item()


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Universal PT2-LLM ternary quantization")
    parser.add_argument("--model", default="EleutherAI/pythia-160m", help="HF model ID")
    parser.add_argument("--output", default=None, help="Output directory (default: models/<model>-ternary)")
    parser.add_argument("--no-gptq", action="store_true", help="Disable GPTQ")
    parser.add_argument("--n-calib", type=int, default=128, help="Calibration samples")
    parser.add_argument("--seq-len", type=int, default=2048, help="Sequence length")
    parser.add_argument("--eval-samples", type=int, default=40, help="PPL eval samples")
    parser.add_argument("--hessian-samples", type=int, default=32,
                        help="Calibration samples per layer for Hessian (default: 32)")
    parser.add_argument("--skip-first-last", action="store_true",
                        help="Skip first and last transformer layers")
    parser.add_argument("--hadamard", action="store_true",
                        help="Use Hadamard rotation (QuIP#-style, for quality testing)")
    parser.add_argument("--bitflip", action="store_true",
                        help="Enable greedy bitflip post-GPTQ refinement (full Hessian)")
    parser.add_argument("--bitflip-iters", type=int, default=3,
                        help="Number of bitflip iterations (default: 3)")
    parser.add_argument("--ptqtp", action="store_true",
                        help="Use PTQTP dual trit-plane decomposition (~3.16 bits)")
    parser.add_argument("--ptqtp-group-size", type=int, default=128,
                        help="PTQTP group size for scales (default: 128)")
    parser.add_argument("--tri-trit", action="store_true",
                        help="Use 3-plane trit decomposition (27 values, ~4.75 bits, highest quality)")
    parser.add_argument("--use-4bit", action="store_true",
                        help="Load model in 4-bit (for large models)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    model_name = args.model.split("/")[-1]
    output_path = Path(args.output or f"models/{model_name}-ternary")
    output_path.mkdir(parents=True, exist_ok=True)

    device = args.device

    # ---- Load model ----
    log.info(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.use_4bit:
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model, quantization_config=bnb_config,
            device_map="auto", trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.float16,
            device_map=device, trust_remote_code=True,
        )
    model.eval()

    # ---- Baseline PPL ----
    log.info("Evaluating baseline perplexity...")
    t0 = time.time()
    baseline_ppl = evaluate_perplexity(model, tokenizer, device,
                                        max_samples=args.eval_samples,
                                        seq_len=args.seq_len)
    t_ppl = time.time() - t0
    log.info(f"Baseline PPL: {baseline_ppl:.2f} ({t_ppl:.0f}s)")

    # ---- Sequential Quantization (calibration + quantize interleaved) ----
    log.info("Sequential layer-by-layer quantization...")
    t0 = time.time()
    stats = quantize_sequential(model, tokenizer, device,
                                 use_gptq=not args.no_gptq,
                                 n_calib=args.n_calib, seq_len=args.seq_len,
                                 n_hessian_samples=args.hessian_samples,
                                 skip_first_last=args.skip_first_last,
                                 use_hadamard=args.hadamard,
                                 use_bitflip=args.bitflip,
                                 bitflip_iters=args.bitflip_iters,
                                 use_ptqtp=args.ptqtp,
                                 ptqtp_group_size=args.ptqtp_group_size,
                                 use_tri_trit=args.tri_trit)
    t_quant = time.time() - t0
    log.info(f"Quantization done ({t_quant:.0f}s)")

    # ---- Post-quant PPL ----
    log.info("Evaluating quantized perplexity...")
    t0 = time.time()
    quant_ppl = evaluate_perplexity(model, tokenizer, device,
                                     max_samples=args.eval_samples,
                                     seq_len=args.seq_len)
    t_ppl2 = time.time() - t0
    log.info(f"Quantized PPL: {quant_ppl:.2f} ({t_ppl2:.0f}s)")

    # ---- Summary ----
    snrs = [s["snr_final"] for s in stats]
    ppl_ratio = quant_ppl / baseline_ppl

    log.info(f"\n{'='*60}")
    log.info(f"Results: {args.model}")
    log.info(f"{'='*60}")
    log.info(f"  Baseline PPL:  {baseline_ppl:.2f}")
    log.info(f"  Quantized PPL: {quant_ppl:.2f}")
    log.info(f"  PPL ratio:     {ppl_ratio:.2f}x")
    log.info(f"  SNR: min={min(snrs):.1f}, mean={sum(snrs)/len(snrs):.1f}, max={max(snrs):.1f} dB")
    log.info(f"  Time: quant={t_quant:.0f}s (sequential, includes per-layer calibration)")
    log.info(f"  Layers quantized: {len(stats)}")

    # ---- Save results ----
    results = {
        "model": args.model,
        "baseline_ppl": baseline_ppl,
        "quantized_ppl": quant_ppl,
        "ppl_ratio": ppl_ratio,
        "snr_min": min(snrs),
        "snr_mean": sum(snrs) / len(snrs),
        "snr_max": max(snrs),
        "n_layers": len(stats),
        "use_gptq": not args.no_gptq,
        "sequential": True,
        "n_calib": args.n_calib,
        "seq_len": args.seq_len,
        "layer_stats": stats,
    }
    with open(output_path / "quantization_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Save model (the modified weights)
    log.info(f"Saving quantized model to {output_path}...")
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    log.info("Done!")


if __name__ == "__main__":
    main()
