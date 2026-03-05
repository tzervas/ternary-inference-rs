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


def gptq_ternary(W, H, block_size=128):
    """GPTQ-style error compensation for ternary quantization."""
    out_f, in_f = W.shape
    W = W.clone()
    W_orig = W.clone()

    damp = 0.01 * H.diagonal().mean()
    H = H.clone()
    H.diagonal().add_(damp)

    try:
        L = torch.linalg.cholesky(H)
        H_inv = torch.cholesky_inverse(L)
    except Exception:
        H.diagonal().add_(damp * 10)
        try:
            L = torch.linalg.cholesky(H)
            H_inv = torch.cholesky_inverse(L)
        except Exception:
            return iterative_ternary_fitting(W, n_iters=10)

    T = torch.zeros(out_f, in_f, dtype=torch.int8)

    for col_start in range(0, in_f, block_size):
        col_end = min(col_start + block_size, in_f)
        block_cols = col_end - col_start
        H_inv_block = H_inv[col_start:col_end, col_start:col_end]

        err_block = torch.zeros(out_f, block_cols, dtype=W.dtype)
        for j in range(block_cols):
            col_idx = col_start + j
            w_col = W[:, col_idx]
            h_jj = H_inv_block[j, j].clamp(min=1e-10)

            scale = w_col.abs().mean().clamp(min=1e-10)
            q_col = (w_col / scale).round().clamp(-1, 1)
            T[:, col_idx] = q_col.to(torch.int8)
            err = (w_col - q_col * scale) / h_jj
            err_block[:, j] = err

            if j + 1 < block_cols:
                W[:, col_start + j + 1:col_end] -= (
                    err.unsqueeze(1) * H_inv_block[j, j + 1:].unsqueeze(0)
                )

        if col_end < in_f:
            H_inv_cross = H_inv[col_start:col_end, col_end:]
            W[:, col_end:] -= err_block @ H_inv_cross

    # Compute optimal (alpha, mu) for the GPTQ T against original W
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
# Calibration + Hessian Capture
# ============================================================================

def capture_hessians(model, tokenizer, device, n_samples=128, seq_len=2048):
    """
    Capture per-linear-layer Hessians (X^T @ X) from calibration data.
    Returns dict: {module_name: Hessian_tensor}.
    """
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    text = "\n\n".join([t for t in dataset["text"] if len(t) > 100])
    tokens = tokenizer.encode(text, return_tensors="pt", truncation=False)[0]
    n_available = (len(tokens) - 1) // seq_len
    n_samples = min(n_samples, n_available)

    samples = []
    for i in range(n_samples):
        start = i * seq_len
        samples.append(tokens[start:start + seq_len].unsqueeze(0))

    # Find all linear layers
    linear_layers = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            linear_layers[name] = module

    log.info(f"Found {len(linear_layers)} linear layers to quantize")

    # Register hooks to accumulate H = X^T @ X
    hessians = {}
    n_tokens_per_layer = {}
    hooks = []

    def make_hook(layer_name, in_features):
        H = torch.zeros(in_features, in_features, dtype=torch.float32, device='cpu')
        count = [0]

        def hook_fn(module, input, output):
            x = input[0].detach().float()
            if x.dim() == 3:
                x = x.reshape(-1, x.shape[-1])
            # Accumulate on CPU to save VRAM
            x_cpu = x.cpu()
            H.add_(x_cpu.T @ x_cpu)
            count[0] += x_cpu.shape[0]

        hessians[layer_name] = H
        n_tokens_per_layer[layer_name] = count
        return hook_fn

    for name, module in linear_layers.items():
        hook = module.register_forward_hook(make_hook(name, module.in_features))
        hooks.append(hook)

    log.info(f"Running {n_samples} calibration samples...")
    model.eval()
    with torch.no_grad():
        for i, sample in enumerate(samples):
            sample = sample.to(device)
            model(sample)
            if (i + 1) % 32 == 0:
                log.info(f"  Calibration: {i+1}/{n_samples}")

    for h in hooks:
        h.remove()

    log.info("Calibration complete")
    return hessians, linear_layers


# ============================================================================
# Quantize + Replace
# ============================================================================

def quantize_and_replace(model, hessians, linear_layers, use_gptq=True):
    """
    Quantize all linear layers in-place using PT2-LLM.
    Replaces nn.Linear with a ternary-quantized version.
    """
    stats_all = []

    for name, module in linear_layers.items():
        W = module.weight.data.float().cpu()
        out_f, in_f = W.shape

        # Pad to multiple of 4 for packing
        if in_f % 4 != 0:
            pad = 4 - (in_f % 4)
            W = torch.nn.functional.pad(W, (0, pad))

        H = hessians.get(name)

        # Step 1: ITF
        T, alpha, mu = iterative_ternary_fitting(W, n_iters=10)
        snr_itf = compute_snr(W, T, alpha, mu)

        # Step 2: AGA (if Hessian available)
        if H is not None:
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

            alpha, mu = activation_aware_alignment(W, T, H_use, itf_alpha=alpha, itf_mu=mu)
            snr_aga = compute_snr(W, T, alpha, mu)
        else:
            snr_aga = snr_itf
            H_use = None

        # Step 3: GPTQ (if enabled and Hessian available)
        best_T, best_alpha, best_mu = T, alpha, mu
        best_snr = snr_aga

        if use_gptq and H_use is not None:
            T_g, alpha_g, mu_g = gptq_ternary(W, H_use)
            # AGA on GPTQ result (with GPTQ values as fallback)
            alpha_g, mu_g = activation_aware_alignment(W, T_g, H_use, itf_alpha=alpha_g, itf_mu=mu_g)
            snr_gptq = compute_snr(W, T_g, alpha_g, mu_g)

            if snr_gptq > best_snr:
                best_T, best_alpha, best_mu = T_g, alpha_g, mu_g
                best_snr = snr_gptq
                method = "GPTQ+AGA"
            else:
                method = "ITF+AGA"
        else:
            method = "ITF+AGA" if H_use is not None else "ITF"
            snr_gptq = None

        # Replace the linear layer with quantized version
        W_hat = best_alpha.unsqueeze(1) * best_T.float() + best_mu.unsqueeze(1)
        # Trim back to original size
        W_hat = W_hat[:out_f, :in_f]

        module.weight.data = W_hat.to(module.weight.dtype).to(module.weight.device)

        stats = {
            "name": name,
            "shape": [out_f, in_f],
            "snr_itf": snr_itf,
            "snr_aga": snr_aga,
            "snr_gptq": snr_gptq,
            "snr_final": best_snr,
            "method": method,
        }
        stats_all.append(stats)

        log.info(f"  {name}: {list(W.shape)} SNR={best_snr:.2f}dB ({method})"
                 f" [ITF={snr_itf:.1f}, AGA={snr_aga:.1f}"
                 f"{f', GPTQ={snr_gptq:.1f}' if snr_gptq is not None else ''}]")

        del W, T, best_T, H
        gc.collect()

    return stats_all


def compute_snr(W, T, alpha, mu):
    W_hat = alpha.unsqueeze(1) * T.float() + mu.unsqueeze(1)
    mse = ((W - W_hat) ** 2).mean().item()
    signal = (W ** 2).mean().item()
    return 10 * math.log10(signal / max(mse, 1e-15))


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
            bnb_4bit_compute_dtype=torch.float16,
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

    # ---- Calibration ----
    log.info("Capturing calibration Hessians...")
    t0 = time.time()
    hessians, linear_layers = capture_hessians(
        model, tokenizer, device,
        n_samples=args.n_calib, seq_len=args.seq_len,
    )
    t_calib = time.time() - t0
    log.info(f"Calibration done ({t_calib:.0f}s)")

    # ---- Quantize ----
    log.info("Quantizing all linear layers...")
    t0 = time.time()
    stats = quantize_and_replace(model, hessians, linear_layers,
                                  use_gptq=not args.no_gptq)
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
    log.info(f"  Time: calib={t_calib:.0f}s, quant={t_quant:.0f}s")
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
