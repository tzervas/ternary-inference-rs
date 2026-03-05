# QEP: Quantization Error Propagation

**Status**: Critical complementary technique for preventing error compounding
**Paper**: [arXiv:2504.09629](https://arxiv.org/abs/2504.09629) (NeurIPS 2025)

## Overview

QEP is a framework that explicitly propagates quantization errors across layers
and compensates for accumulated errors. It wraps around any per-layer PTQ method
(including ternary quantization) and prevents the catastrophic error compounding
that makes naive layer-wise PTQ fail at extreme low-bit regimes.

## The Core Problem QEP Solves

Standard layer-wise PTQ optimizes each layer independently using **full-precision
inputs** X_l. But the next layer receives **quantized outputs**, not full-precision
ones. This mismatch causes errors to accumulate exponentially:

```
Layer 0: small error e_0
Layer 1: receives e_0, adds e_1 -> total ~e_0 + e_1
Layer 2: receives e_0 + e_1, adds e_2 -> growing error
...
Layer 63: catastrophic accumulated error
```

This is EXACTLY why our naive RTN ternary quantization produces garbled output.

## Algorithm

### Standard PTQ (what fails)

For each layer l, minimize:
```
||W_l @ X_l - Q(W_l) @ X_l||^2
```
Uses full-precision inputs X_l for both reference and quantized computation.

### QEP (what fixes it)

For each layer l, minimize:
```
||W_l @ X_l - Q(W_l*) @ X_hat_l||^2
```
Where:
- X_hat_l = quantized inputs (output of previous quantized layers)
- W_l* = corrected weights that compensate for accumulated error

**Correction formula (Proposition 5.1):**
```
W_l* = W_l + alpha_l * W_l @ delta_l @ X_hat_l^T @ H_hat_l^(-1)
```
Where:
- delta_l = X_l - X_hat_l (accumulated quantization error at layer l)
- H_hat_l = X_hat_l @ X_hat_l^T (Hessian of quantized inputs)
- alpha_l in [0, 1] controls propagation strength (tunable per layer)

### Procedure

1. Run calibration data through the full-precision model, save all X_l
2. For each layer l (sequentially):
   a. Compute delta_l = X_l - X_hat_l (error so far)
   b. Compute corrected weights W_l* using the formula above
   c. Quantize W_l* using any PTQ method (GPTQ, RTN, PT2-LLM, etc.)
   d. Run X_hat_l through quantized layer l to get X_hat_{l+1}
3. The correction term adjusts weights BEFORE quantization to compensate
   for the fact that they'll receive noisy inputs

## Key Results

| Model | Method | Bits | PPL (Wikitext2) |
|-------|--------|------|-----------------|
| LLaMA2-7B | RTN | INT2g32 | 90.686 |
| LLaMA2-7B | RTN + QEP | INT2g32 | **12.248** |
| LLaMA2-7B | GPTQ | INT2g32 | 14.907 |
| LLaMA2-7B | GPTQ + QEP | INT2g32 | **10.456** |

QEP reduces INT2 RTN perplexity by **7.4x** (from 90.7 to 12.2).
QEP improves GPTQ INT2 by 30% (from 14.9 to 10.5).

## Computational Overhead

- **Memory**: Negligible (Hessian already needed for GPTQ-style methods)
- **Runtime**: QEP+RTN is actually FASTER than GPTQ (10.9 min vs 14.9 min for 7B)
  because RTN is simpler than GPTQ and QEP overhead is minimal
- **Selective alpha**: Setting alpha_l = 0 for MLP blocks reduces time further

## Compatibility with Ternary Quantization

QEP is method-agnostic. It wraps around any per-layer quantizer:
- QEP + RTN ternary: simplest, might be sufficient
- QEP + PT2-LLM ternary: best quality, more complex
- QEP + GPTQ ternary: middle ground

## Why This Matters for Us

Our experiments showed that naive RTN ternary gives ~7 dB SNR per layer, which
compounds to total signal destruction across 64 layers. QEP directly addresses
this by correcting weights to compensate for incoming errors.

The correction term W_l* = W_l + correction means we quantize slightly different
weights than the originals — weights that are pre-compensated for the errors
they'll receive from previous quantized layers.

## Implementation Requirements

1. Need full-precision activations X_l for all calibration samples at each layer
   - Can compute incrementally: run calibration through layer 0, save output, etc.
   - Memory: 128 samples * 2048 * 5120 * 4 bytes = 5.2 GB per layer
   - BUT: only need X_l and X_hat_l simultaneously for one layer at a time

2. Need quantized activations X_hat_l for all calibration samples
   - Computed by running calibration through already-quantized layers 0..l-1
   - Same memory as above

3. Correction computation is just matrix operations on the weight matrix
   - O(out * in) per layer, same as GPTQ

## Strategy: QEP + PT2-LLM

The optimal approach combines both:
1. QEP provides cross-layer error compensation
2. PT2-LLM provides per-layer ternary quality (asymmetric, iterative)

Pipeline:
```
For each layer l:
  1. Compute delta_l = X_l - X_hat_l
  2. Compute W_l* (corrected weights) via QEP formula
  3. Run PT2-LLM on W_l* (ITF + AGA + GPTQ blocks)
  4. Pack to ternary format
  5. Run calibration through quantized layer l to get X_hat_{l+1}
```

This is the most promising path to viable post-training ternary quantization
on consumer hardware.
