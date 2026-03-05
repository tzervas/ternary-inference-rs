#!/usr/bin/env python3
"""
Ternary quantization with QEP (Quantization Error Propagation).

QEP is the KEY technique that makes post-training ternary work by compensating
for error accumulation across layers. Without QEP, ~7dB SNR per layer compounds
to total signal destruction. With QEP, weights are pre-corrected to account for
the fact that they receive noisy (quantized) inputs from previous layers.

Pipeline:
  1. Load fp16 model, evaluate baseline PPL
  2. For each layer sequentially:
     a. Capture fp activations X_fp and quantized activations X_hat
     b. Compute error correction: W* = W + alpha * W @ (X_fp - X_hat)^T @ X_hat @ H^{-1}
     c. Apply ITF to corrected weights W*
     d. Replace layer with quantized version
     e. Forward pass to compute X_hat for next layer
  3. Evaluate quantized PPL

This is fundamentally different from independent per-layer quantization because
each layer's quantization accounts for the errors from all previous layers.

Usage:
    .venv/bin/python quantize_with_qep.py --model EleutherAI/pythia-160m
"""

import argparse
import gc
import json
import logging
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def iterative_ternary_fitting(W, n_iters=10):
    """ITF with asymmetric (alpha, mu) per row."""
    out_f, in_f = W.shape
    m = float(in_f)
    mu = W.mean(dim=1)
    W_c = W - mu.unsqueeze(1)
    alpha = (0.75 / m) * W_c.abs().sum(dim=1)
    alpha = alpha.clamp(min=1e-10)

    for _ in range(n_iters):
        T = ((W - mu.unsqueeze(1)) / alpha.unsqueeze(1).clamp(min=1e-10))
        T = T.round().clamp(-1, 1)
        TT = (T * T).sum(dim=1)
        WT = (W * T).sum(dim=1)
        Ts = T.sum(dim=1)
        Ws = W.sum(dim=1)
        denom = (m * TT - Ts ** 2).clamp(min=1e-10)
        alpha = (m * WT - Ts * Ws) / denom
        mu = (TT * Ws - Ts * WT) / denom
        alpha = alpha.clamp(min=0)

    T = ((W - mu.unsqueeze(1)) / alpha.unsqueeze(1).clamp(min=1e-10))
    T = T.round().clamp(-1, 1)
    return T, alpha, mu


def dequantize(T, alpha, mu):
    """Reconstruct W_hat = alpha * T + mu."""
    return alpha.unsqueeze(1) * T.float() + mu.unsqueeze(1)


def compute_snr(W, T, alpha, mu):
    W_hat = dequantize(T, alpha, mu)
    mse = ((W - W_hat) ** 2).mean().item()
    signal = (W ** 2).mean().item()
    return 10 * math.log10(signal / max(mse, 1e-15))


@torch.no_grad()
def evaluate_perplexity(model, tokenizer, device, max_samples=40, seq_len=2048):
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
            target_ids.reshape(-1), reduction="mean")
        nlls.append(loss.item())
        if (i + 1) % 10 == 0:
            log.info(f"  PPL: {i+1}/{n_samples}, running = {math.exp(sum(nlls)/len(nlls)):.2f}")
    return math.exp(sum(nlls) / len(nlls))


def get_calibration_tokens(tokenizer, n_samples=128, seq_len=2048):
    """Get calibration token sequences from Wikitext2 train."""
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


def get_transformer_layers(model):
    """Get the list of transformer layers, handling different architectures."""
    # Try common patterns
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        return model.model.layers  # LLaMA, Qwen, Mistral
    if hasattr(model, 'gpt_neox') and hasattr(model.gpt_neox, 'layers'):
        return model.gpt_neox.layers  # Pythia, GPT-NeoX
    if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        return model.transformer.h  # GPT-2
    raise ValueError(f"Cannot find transformer layers in model: {type(model)}")


def get_embed_and_norm(model):
    """Get embedding layer and first norm, handling different architectures."""
    if hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
        return model.model.embed_tokens, model.model.norm  # LLaMA/Qwen
    if hasattr(model, 'gpt_neox'):
        return model.gpt_neox.embed_in, model.gpt_neox.final_layer_norm  # Pythia
    if hasattr(model, 'transformer'):
        return model.transformer.wte, model.transformer.ln_f  # GPT-2
    raise ValueError(f"Cannot find embeddings in model: {type(model)}")


def get_lm_head(model):
    """Get the language model head."""
    if hasattr(model, 'lm_head'):
        return model.lm_head
    if hasattr(model, 'embed_out'):
        return model.embed_out
    raise ValueError(f"Cannot find lm_head in model: {type(model)}")


def get_linear_layers_in_module(module):
    """Get all nn.Linear layers in a module (non-recursive first level only needed)."""
    linears = {}
    for name, child in module.named_modules():
        if isinstance(child, nn.Linear):
            linears[name] = child
    return linears


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="EleutherAI/pythia-160m")
    parser.add_argument("--output", default=None)
    parser.add_argument("--n-calib", type=int, default=64)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--eval-samples", type=int, default=20)
    parser.add_argument("--qep-alpha", type=float, default=0.5,
                        help="QEP correction strength (0=off, 1=full)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    model_name = args.model.split("/")[-1]
    output_path = Path(args.output or f"models/{model_name}-ternary-qep")
    output_path.mkdir(parents=True, exist_ok=True)
    device = args.device

    # Load model
    log.info(f"Loading: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float16,
        device_map=device, trust_remote_code=True,
    )
    model.eval()

    # Baseline PPL
    log.info("Baseline PPL evaluation...")
    baseline_ppl = evaluate_perplexity(model, tokenizer, device,
                                        max_samples=args.eval_samples,
                                        seq_len=args.seq_len)
    log.info(f"Baseline PPL: {baseline_ppl:.2f}")

    # Get calibration data
    log.info("Preparing calibration data...")
    calib_samples = get_calibration_tokens(tokenizer, args.n_calib, args.seq_len)

    # Get model components
    layers = get_transformer_layers(model)
    embed, final_norm = get_embed_and_norm(model)
    num_layers = len(layers)
    log.info(f"Model has {num_layers} transformer layers")

    # =========================================================================
    # QEP: Sequential layer-by-layer quantization with error tracking
    # =========================================================================
    #
    # Key idea: run calibration data through the model twice simultaneously:
    # 1. Full-precision path: gives X_fp[l] (what layer l SHOULD receive)
    # 2. Quantized path: gives X_hat[l] (what layer l ACTUALLY receives)
    #
    # For each layer l:
    # - delta[l] = X_fp[l] - X_hat[l] captures accumulated error
    # - Correct weights: W* = W + alpha * correction_term
    # - Quantize W* with ITF
    # - Forward quantized layer to get X_hat[l+1]

    log.info("=" * 60)
    log.info("QEP: Sequential layer-wise quantization with error compensation")
    log.info("=" * 60)

    # Step 1: Compute fp activations for all layers
    # We'll use a subset of calibration samples to save memory
    n_qep_samples = min(16, len(calib_samples))
    log.info(f"Using {n_qep_samples} samples for QEP activation tracking")

    # Capture all layer inputs by running full-precision model
    fp_inputs_per_layer = []  # List of [N, hidden] tensors (on CPU)

    def capture_layer_inputs():
        """Run calibration through model, capture input to each transformer layer."""
        layer_inputs = [[] for _ in range(num_layers)]
        hooks = []

        for idx, layer in enumerate(layers):
            def make_hook(layer_idx):
                def hook_fn(module, input, output):
                    x = input[0].detach().float().cpu()
                    if x.dim() == 3:
                        x = x.reshape(-1, x.shape[-1])
                    layer_inputs[layer_idx].append(x)
                return hook_fn
            hooks.append(layer.register_forward_hook(make_hook(idx)))

        with torch.no_grad():
            for i in range(n_qep_samples):
                model(calib_samples[i].to(device))

        for h in hooks:
            h.remove()

        return [torch.cat(li, dim=0) if li else None for li in layer_inputs]

    log.info("Capturing full-precision activations...")
    fp_layer_inputs = capture_layer_inputs()
    log.info(f"Captured {len(fp_layer_inputs)} layers, "
             f"shape={fp_layer_inputs[0].shape if fp_layer_inputs[0] is not None else 'None'}")

    # Step 2: Sequentially quantize layers with QEP correction
    stats_all = []
    qep_alpha = args.qep_alpha

    # Track quantized activations (start with fp, diverges as we quantize)
    # After quantizing layer l, we re-run calibration through layers 0..l
    # to get X_hat for layer l+1

    for layer_idx in range(num_layers):
        layer = layers[layer_idx]
        X_fp = fp_layer_inputs[layer_idx]  # [N, hidden]

        if X_fp is None:
            log.warning(f"Layer {layer_idx}: no activations captured, skipping QEP")
            continue

        # Get quantized activations (run through already-quantized layers)
        # For the first layer, X_hat = X_fp (no previous quantization)
        if layer_idx == 0:
            X_hat = X_fp.clone()
        else:
            # Re-run calibration through the model to get current (partially quantized) activations
            log.info(f"  Re-capturing quantized activations for layer {layer_idx}...")
            current_inputs = []
            hook_handle = None

            def capture_hook(module, input, output, storage=current_inputs):
                x = input[0].detach().float().cpu()
                if x.dim() == 3:
                    x = x.reshape(-1, x.shape[-1])
                storage.append(x)

            hook_handle = layer.register_forward_hook(capture_hook)
            with torch.no_grad():
                for i in range(n_qep_samples):
                    model(calib_samples[i].to(device))
            hook_handle.remove()
            X_hat = torch.cat(current_inputs, dim=0)

        # Compute error
        delta = X_fp - X_hat  # [N, hidden]
        delta_norm = delta.norm().item()
        x_fp_norm = X_fp.norm().item()
        error_ratio = delta_norm / max(x_fp_norm, 1e-10)
        log.info(f"Layer {layer_idx}: error ratio = {error_ratio:.4f} "
                 f"(delta_norm={delta_norm:.2f}, x_norm={x_fp_norm:.2f})")

        # Get all linear layers in this transformer layer
        linears = get_linear_layers_in_module(layer)

        for lin_name, lin_module in linears.items():
            W = lin_module.weight.data.float().cpu()
            out_f, in_f = W.shape

            # QEP weight correction
            if qep_alpha > 0 and error_ratio > 0.001:
                # Trim/pad activations to match weight dimension
                act_dim = X_fp.shape[1]
                if act_dim != in_f:
                    # Skip QEP for this specific projection (mismatched dimensions)
                    W_corrected = W
                else:
                    # QEP correction: W* = W + alpha * W @ delta^T @ X_hat @ H_hat^{-1}
                    # Simplified: correct using correlation between error and quantized input
                    H_hat = X_hat.T @ X_hat  # [in, in]
                    damp = 0.01 * H_hat.diagonal().mean()
                    H_hat.diagonal().add_(damp)

                    # delta^T @ X_hat: correlation of error with quantized input
                    delta_Xhat = delta.T @ X_hat  # [in, in]

                    try:
                        # Solve: H_hat @ Y = delta_Xhat for Y
                        L = torch.linalg.cholesky(H_hat)
                        Y = torch.cholesky_solve(delta_Xhat, L)  # [in, in]
                        correction = W @ Y  # [out, in]
                        W_corrected = W + qep_alpha * correction
                    except Exception as e:
                        log.warning(f"  QEP solve failed for {lin_name}: {e}")
                        W_corrected = W
            else:
                W_corrected = W

            # Pad to multiple of 4
            if in_f % 4 != 0:
                pad = 4 - (in_f % 4)
                W_corrected = torch.nn.functional.pad(W_corrected, (0, pad))

            # ITF quantization
            T, alpha, mu = iterative_ternary_fitting(W_corrected, n_iters=10)
            snr = compute_snr(W_corrected[:out_f, :in_f], T[:out_f, :in_f], alpha, mu)

            # Replace weight with dequantized version
            W_hat = dequantize(T, alpha, mu)[:out_f, :in_f]
            lin_module.weight.data = W_hat.to(lin_module.weight.dtype).to(lin_module.weight.device)

            stats_all.append({"layer": layer_idx, "name": lin_name, "snr": snr,
                              "error_ratio": error_ratio})
            log.info(f"  {lin_name}: SNR={snr:.2f}dB")

            del W, W_corrected, T, alpha, mu, W_hat
            gc.collect()

        del X_fp, X_hat, delta
        gc.collect()
        torch.cuda.empty_cache()

    # Quantized PPL
    log.info("\nEvaluating quantized PPL...")
    quant_ppl = evaluate_perplexity(model, tokenizer, device,
                                     max_samples=args.eval_samples,
                                     seq_len=args.seq_len)

    # Results
    snrs = [s["snr"] for s in stats_all]
    ppl_ratio = quant_ppl / baseline_ppl

    log.info(f"\n{'='*60}")
    log.info(f"Results: {args.model} (QEP alpha={qep_alpha})")
    log.info(f"{'='*60}")
    log.info(f"  Baseline PPL:  {baseline_ppl:.2f}")
    log.info(f"  Quantized PPL: {quant_ppl:.2f}")
    log.info(f"  PPL ratio:     {ppl_ratio:.2f}x")
    log.info(f"  SNR: min={min(snrs):.1f}, mean={sum(snrs)/len(snrs):.1f}, max={max(snrs):.1f} dB")
    log.info(f"  Layers quantized: {len(stats_all)}")

    # Save results
    results = {
        "model": args.model,
        "baseline_ppl": baseline_ppl,
        "quantized_ppl": quant_ppl,
        "ppl_ratio": ppl_ratio,
        "qep_alpha": qep_alpha,
        "snr_min": min(snrs),
        "snr_mean": sum(snrs) / len(snrs),
        "snr_max": max(snrs),
        "layer_stats": stats_all,
    }
    with open(output_path / "quantization_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Save model
    log.info(f"Saving to {output_path}...")
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    log.info("Done!")


if __name__ == "__main__":
    main()
