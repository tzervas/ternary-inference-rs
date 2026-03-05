# PT2-LLM Implementation Plan for Consumer Hardware

## Target

Implement PT2-LLM post-training ternarization for Qwen2.5-Coder-32B-Instruct
on RTX 3090 Ti (24 GB VRAM) + 46 GB RAM.

## Architecture

```
quantize_pt2_ternary.py
  |
  +-- Phase 1: Calibration capture (~30 min)
  |     - Load model in 4-bit (bitsandbytes)
  |     - 128 Wikitext2 samples, seq_len=2048
  |     - Capture per-layer activations sequentially
  |     - Store covariance C = X^T @ X per projection
  |
  +-- Phase 2: Layer-wise quantization (~2-4 hours)
  |     For each of 64 layers:
  |       - Load bf16 weights from safetensors shard
  |       - For each projection (q/k/v/o/gate/up/down):
  |         1. SSR: cluster columns by cosine similarity
  |         2. ITF: 10 iterations of grid optimization
  |         3. AGA: activation-aware refinement using C
  |         4. GPTQ: block-wise error compensation
  |       - Pack to base-3 format
  |       - Save to output shard
  |
  +-- Phase 3: Finalize
        - Keep embed_tokens + lm_head in bf16
        - Copy tokenizer, config
        - Write safetensors index
```

## Memory Budget (per layer, worst case)

| Component | Size | Location |
|-----------|------|----------|
| Calibration model (4-bit) | ~20 GB | GPU (Phase 1 only) |
| Covariance C (27648^2) | 3.0 GB | CPU RAM |
| Weight tensor (f32) | 537 MB | CPU RAM |
| Ternary matrix T (i8) | 135 MB | CPU RAM |
| GPTQ working memory | ~1 GB | CPU RAM |
| Output packed tensor | ~35 MB | CPU RAM |
| **Peak (Phase 1)** | **~20 GB VRAM** | |
| **Peak (Phase 2)** | **~5 GB RAM** | |

Phase 1 and Phase 2 don't overlap -- the 4-bit model is freed before Phase 2 begins.

## Key Implementation Details

### Calibration Capture (Phase 1)

```python
# Hook each layer to capture input activations
for layer_idx in range(num_layers):
    # Register hook on layer
    # Run all 128 calibration samples
    # Compute covariance C = X^T @ X incrementally (never store full X)
    # Save C to disk: f"calibration/layer_{layer_idx}_{proj_name}_cov.pt"
```

Memory optimization: compute C incrementally as sum of outer products.
Never store the full activation matrix (which would be 262144 x 5120 = 5 GB).

### Covariance for Different Projections

The layer input is used for q/k/v projections, but:
- o_proj input = attention output (different from layer input)
- gate/up_proj input = post-attention-norm output (same as layer input if no residual)
- down_proj input = gate * up output (different)

For a proper implementation, we need separate covariances for each input source.
As a simplification, we can use the layer input covariance for all projections
within a layer (approximate but much simpler).

### ITF Implementation

```python
def iterative_ternary_fitting(W, n_iters=10):
    out_f, in_f = W.shape

    # Initialize
    mu = W.mean(dim=1)  # per-row offset
    W_centered = W - mu.unsqueeze(1)
    alpha = 0.75 / in_f * W_centered.abs().sum(dim=1)  # TWN init

    for _ in range(n_iters):
        # Flexible rounding
        T = ((W - mu.unsqueeze(1)) / alpha.unsqueeze(1).clamp(min=1e-10))
        T = T.round().clamp(-1, 1).to(torch.int8)
        T_float = T.float()

        # Optimal grid (closed-form)
        m = in_f
        TT_sum = (T_float * T_float).sum(dim=1)  # sum of T^2 per row
        WT_sum = (W * T_float).sum(dim=1)         # sum of W*T per row
        T_sum = T_float.sum(dim=1)                # sum of T per row
        W_sum = W.sum(dim=1)                      # sum of W per row

        denom = (m * TT_sum - T_sum ** 2).clamp(min=1e-10)
        alpha = (m * WT_sum - T_sum * W_sum) / denom
        mu = (TT_sum * W_sum - T_sum * WT_sum) / denom
        alpha = alpha.clamp(min=0)

    # Final assignment
    T = ((W - mu.unsqueeze(1)) / alpha.unsqueeze(1).clamp(min=1e-10))
    T = T.round().clamp(-1, 1).to(torch.int8)

    return T, alpha, mu
```

### AGA Implementation

```python
def activation_aware_alignment(W, T, C, alpha_init, mu_init):
    """
    C: covariance matrix [in_features, in_features]
    T: frozen ternary assignments [out_features, in_features]
    """
    T_float = T.float()

    # Solve for alpha, mu that minimize ||W @ X - (alpha * T + mu) @ X||^2
    # Using covariance C = X^T @ X

    TC = T_float @ C         # [out, in]
    TTC = (TC * T_float).sum(dim=1)  # sum_j T_ij * (TC)_ij per row
    WC = W @ C               # [out, in]
    WTC = (WC * T_float).sum(dim=1)  # sum_j W_ij * (TC)_ij per row

    ones_C = C.sum(dim=1)    # [in]
    T_onesC = (T_float * ones_C.unsqueeze(0)).sum(dim=1)  # per row
    W_onesC = (W * ones_C.unsqueeze(0)).sum(dim=1)        # per row
    onesC_ones = ones_C.sum()  # scalar

    # Solve 2x2 system per row for (alpha, mu)
    # [TTC, T_onesC] [alpha]   [WTC]
    # [T_onesC, onesC_ones] [mu] = [W_onesC]

    det = (TTC * onesC_ones - T_onesC ** 2).clamp(min=1e-10)
    alpha = (WTC * onesC_ones - W_onesC * T_onesC) / det
    mu = (TTC * W_onesC - T_onesC * WTC) / det
    alpha = alpha.clamp(min=0)

    return alpha, mu
```

### Packing with Asymmetric Quantizer

The output format needs to encode both alpha (scale) and mu (offset):
- Option A: Store alpha and mu as separate tensors (simple, compatible with our loader
  if we add offset support)
- Option B: Fold mu into a bias term: W_hat = alpha * T + mu = alpha * T + mu * ones^T.
  The mu term is equivalent to adding mu to each output element, which is a bias.
  So: output = (alpha * T) @ x + mu * sum(x). This changes the computation.
- **Recommended: Option A** -- store alpha as scales and mu as a new "offset" tensor.

## Output Format

```
*.weight          uint8   Packed ternary (base-3, same as current)
*.weight.scales   float32 Per-row alpha [out_features]
*.weight.offsets  float32 Per-row mu [out_features]  (NEW)
```

This requires a small Rust loader change to apply the offset during dequantization.

## Testing Strategy

1. Quantize a small model first (e.g., Qwen2.5-Coder-3B if available)
2. Compare perplexity on Wikitext2 against baseline
3. Scale to 32B if results are promising
4. Test with our Rust inference engine for end-to-end validation
