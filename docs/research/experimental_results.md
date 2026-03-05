# Experimental Results: PT2-LLM Implementation

## Summary

Per-layer ternary quantization with ITF+AGA achieves ~7 dB SNR consistently,
but this compounds to catastrophic quality loss across layers without proper
GPTQ error compensation.

## Results Table

| Model | Params | Hidden | Layers | Baseline PPL | Quant PPL | PPL Ratio | Method |
|-------|--------|--------|--------|-------------|-----------|-----------|--------|
| Pythia-160M | 160M | 768 | 12 | 28.83 | 393M | 13.6M x | ITF+AGA |
| Pythia-1B | 1.04B | 2048 | 16 | 14.16 | 14,598 | 1031x | ITF+AGA |

## Per-Layer SNR Analysis

Consistent ~7 dB SNR across all layers and model sizes:
- ITF: 7.1-7.3 dB (best achievable for ternary without error compensation)
- AGA: No improvement over ITF (weights are already near-zero mean)
- GPTQ: DEGRADED to 3-6 dB (implementation needs fixing)

## Key Findings

### 1. ITF = Optimal MSE
ITF with per-row (alpha, mu) gives essentially the same SNR as simple optimal
MSE scaling. The per-row offset (mu) adds only ~0.02 dB because Pythia weights
are already near-zero-mean.

### 2. AGA Does Not Help
AGA refines alpha/mu using calibration covariance, but since ITF already finds
the global optimum, AGA can only improve if the activation-weighted error differs
from the unweighted error. In practice, the improvement is negligible.

### 3. GPTQ Implementation Is Critical
Our GPTQ implementation makes quality WORSE because:
- Per-column ternary quantization within GPTQ loses the per-row structure
- The Hessian-based error propagation is fighting against ITF's optimal assignment
- Proper GPTQ for ternary needs to maintain per-row (alpha, mu) during column processing

### 4. QEP Overcorrects
QEP weight correction with alpha=0.5 causes error to explode (error ratio > 200x
by layer 2). The correction term is too large relative to the original weights
when the accumulated error is already significant.

### 5. Error Compounding Is Exponential
- Layer 0: error ratio = 0 (perfect)
- Layer 1: error ratio = 0.45 (45% of signal is noise)
- Layer 2: error ratio = 302 (errors have completely dominated)
- This confirms the QEP paper's finding that cross-layer error compensation
  is essential, not optional.

## What's Needed

1. **Fix GPTQ for ternary**: Need GPTQ that quantizes columns while maintaining
   per-row scale/offset consistency. The PT2-LLM paper's approach quantizes
   blocks of columns and re-runs ITF after each block.

2. **Proper QEP integration**: QEP needs careful alpha tuning and possibly
   layer-wise adaptation. alpha=0.5 is too aggressive when errors compound fast.

3. **Better baseline**: Test with a model that has published PT2-LLM results
   (e.g., LLaMA-7B) to validate against paper claims before debugging.

## Reference: PT2-LLM Paper Claims

| Model | Method | PPL (Wiki2) |
|-------|--------|-------------|
| LLaMA-7B | FP16 baseline | 5.68 |
| LLaMA-7B | PT2-LLM ternary | ~6.5-7.0 (estimated) |
| LLaMA-7B | GPTQ 2-bit | ~7.0 |

The paper shows competitive results with 2-bit GPTQ, suggesting the GPTQ
integration is the key differentiator, not ITF/AGA alone.
