# PT2-LLM: Post-Training Ternarization for Large Language Models

**Status**: Primary implementation target
**Paper**: [arXiv:2510.03267](https://arxiv.org/abs/2510.03267) (ICLR 2026)
**Code**: [github.com/XIANGLONGYAN/PT2-LLM](https://github.com/XIANGLONGYAN/PT2-LLM) (placeholder, code not yet released)

## Overview

PT2-LLM is a post-training ternarization framework that achieves competitive performance
with 2-bit PTQ methods while using only 1.58 bits per weight. No retraining or fine-tuning
required -- only 128 calibration samples.

## Key Results

- LLaMA-7B: 13.48 GB -> 1.88 GB (7.17x compression), quantized in 32 minutes
- Scales to 70B parameters
- Competitive with GPTQ 2-bit and other SOTA 2-bit PTQ methods
- Hardware used in paper: single NVIDIA A800-80GB

## Algorithm: Asymmetric Ternary Quantizer

### Core Idea

Unlike symmetric ternary quantization (just a scale alpha), PT2-LLM uses an
**asymmetric** quantizer with both scale (alpha) and offset (mu) per row:

```
W_hat = alpha * T + mu
```

where T is in {-1, 0, +1}. This captures non-zero-mean weight distributions.

### Stage 1: Iterative Ternary Fitting (ITF)

Alternates between two steps (~10 iterations):

**Step A - Optimal Grid Construction (closed-form):**
Given ternary assignments T, solve for optimal alpha* and mu*:

```
alpha* = [m * (W . T) @ 1 - (T @ 1) . (W @ 1)] / [m * (T . T) @ 1 - (T @ 1)^2]
mu*    = [(T . T) @ 1 . (W @ 1) - (T @ 1) . (W . T) @ 1] / [m * (T . T) @ 1 - (T @ 1)^2]
```

where m = number of columns, . = element-wise multiply, @ 1 = row sum.

**Step B - Flexible Rounding:**
Given alpha* and mu*, assign each weight to nearest ternary value:

```
T_ij = argmin_{t in {-1,0,+1}} |W_ij - (alpha_i * t + mu_i)|
     = round((W_ij - mu_i) / alpha_i) clamped to {-1, 0, +1}
```

**Initialization:**
- mu = mean of each weight row (captures asymmetry)
- W_centered = W - mu
- alpha = TWN approximation: 0.75/m * sum(|W_centered|) per row

### Stage 2: Activation-aware Grid Alignment (AGA)

After ITF converges, refines alpha and mu using calibration activations.

Given calibration input X in R^(B x L x m), compute covariance:
```
C = sum_b sum_i X_bi @ X_bi^T    (m x m matrix)
```

Then solve for alpha* and mu* that minimize output error:
```
minimize ||W @ X - W_hat @ X||^2_F
```

This has closed-form solutions using C. Critically, the ternary matrix T is
**frozen** after ITF -- only alpha and mu are refined. This prevents overfitting
on the small calibration set.

### GPTQ Integration

PT2-LLM uses GPTQ-style block processing:

1. **Block size**: 128 columns
2. **Structural Similarity Reordering (SSR)**: Before each block, cluster remaining
   columns by cosine similarity. Process similar columns together so that GPTQ
   error compensation is more effective.
3. **Error propagation**: Standard GPTQ column-wise error compensation within blocks,
   with inter-block error propagation.

After GPTQ updates each block, re-run ITF on the updated weights to find new
optimal alpha, mu, T for the affected rows.

## Calibration Requirements

- **128 samples** from Wikitext2
- **Sequence length**: 2048
- Robust to calibration size (64-256 samples give similar results)

## Comparison with Naive AbsMean

| Aspect | AbsMean | PT2-LLM |
|--------|---------|---------|
| Scale | alpha = mean(\|W\|) | Iteratively optimized |
| Offset | None (symmetric) | Per-row mu (asymmetric) |
| Rounding | Fixed threshold | Flexible per-element |
| Calibration | None | 128 samples (AGA stage) |
| Error compensation | None | GPTQ with SSR |
| Column ordering | Sequential | Cosine similarity clustering |

## Memory Requirements for Our Hardware

For Qwen2.5-Coder-32B (per layer, worst case = down_proj):

| Component | Size |
|-----------|------|
| Covariance C (27648 x 27648 x f32) | 3.0 GB |
| Weight tensor (5120 x 27648 x f32) | 537 MB |
| Ternary matrix T (5120 x 27648 x i8) | 135 MB |
| GPTQ Hinv block (128 x 128 x f32) | 64 KB |
| Working memory | ~1 GB |
| **Total peak per layer** | **~5 GB** |

This fits comfortably in 46 GB RAM. Processing is sequential (one layer at a time).

## Implementation Notes

- The paper uses block size 128 for GPTQ, which is standard
- SSR uses cosine similarity between column vectors -- O(m^2) but only computed once per layer
- ITF convergence in ~10 iterations is fast (just matrix operations on the weight)
- AGA is a single optimization step after ITF (closed-form)
- The Hessian for GPTQ can be computed incrementally from calibration activations
