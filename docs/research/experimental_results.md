# Experimental Results: PT2-LLM Implementation

## Summary

Sequential layer-by-layer GPTQ with activation-weighted error selection,
Hadamard rotation, and SSR column reordering achieves 41x PPL ratio on
Pythia-1B. GPTQ provides 3.7x improvement over ITF-only, confirming that
cross-column error propagation is essential for ternary PTQ.

## Results Table

### Pythia-160M (768 hidden, 12 layers)

| Method | Quant PPL | Ratio | Notes |
|--------|-----------|-------|-------|
| Baseline (FP16) | 26.98 | 1.0x | |
| ITF-only (parallel, all layers) | 418M | 15.5Mx | LM head quantization destroys output |
| ITF-only (seq, skip embed/head) | 18,803 | 697x | Skipping LM head is critical |
| GPTQ (seq, skip first/last) | 5,828 | 216x | First/last layers most sensitive |
| GPTQ + Hadamard (skip first/last) | 3,530 | 131x | Hadamard rotation helps 39% |

### Pythia-1B (2048 hidden, 16 layers)

| Method | Quant PPL | Ratio | Notes |
|--------|-----------|-------|-------|
| Baseline (FP16) | 13.21 | 1.0x | |
| ITF-only (seq, skip first/last) | 2,986 | 226x | Per-weight SNR ~7.1 dB |
| GPTQ (seq, skip first/last) | 810 | 61x | **3.7x improvement over ITF** |
| GPTQ + SSR | 581 | 44x | Column reordering helps 28% |
| **GPTQ + Hadamard** | **546** | **41x** | Best result, all layers GPTQ wins |

### Pythia-2.8B (2560 hidden, 32 layers)

| Method | Quant PPL | Ratio | Notes |
|--------|-----------|-------|-------|
| Baseline (FP16) | 10.23 | 1.0x | |
| **GPTQ + Hadamard + SSR** | **329** | **32x** | 120/120 layers GPTQ wins, SNR 5.8-6.9 dB |

### Pythia-6.9B (4096 hidden, 32 layers)

| Method | Quant PPL | Ratio | Notes |
|--------|-----------|-------|-------|
| Baseline (FP16) | 9.31 | 1.0x | |
| **GPTQ + Hadamard + SSR** | **665** | **71x** | Worse than 2.8B despite better per-weight SNR |

**Key insight**: Scaling trend reverses at 6.9B. Per-weight SNR is slightly better
(6.4 dB mean vs 6.3 dB for 2.8B), but total error accumulation from 6B quantized
params (vs 2.4B) overwhelms the redundancy benefit. Same 32 layers but 2.5x more
params per layer.

### Phi-2 (2560 hidden, 32 layers, PhiForCausalLM)

| Method | Quant PPL | Ratio | Notes |
|--------|-----------|-------|-------|
| Baseline (FP16) | 9.81 | 1.0x | |
| **GPTQ + Hadamard + SSR** | **98.8** | **10x** | 3x better than Pythia-2.8B (similar size!) |

**Key insight**: Architecture matters far more than model size! Phi-2 (2.7B) achieves
10x ratio vs Pythia-2.8B's 32x, with essentially the same dimensions (2560 hidden,
10240 intermediate, 32 layers). Phi's architecture is significantly more
quantization-friendly.

## Key Findings (Updated)

### 1. GPTQ Is Working Correctly
Fixed GPTQ implementation with:
- **Per-row (alpha, mu) from ITF** for column quantization (not per-column scaling)
- **Activation-weighted error** (`trace((W-W_hat)@H@(W-W_hat)^T)`) for method selection
- **ITF re-run after each block** to adapt (alpha, mu) to error-compensated weights
- GPTQ reduces per-weight SNR (~6.2 dB) but improves activation-weighted error by ~46%
- End-to-end PPL confirms: GPTQ 3.7x better than ITF-only

### 2. Sequential Hessian Capture Is Essential
Capturing Hessians through the partially-quantized model (sequential pipeline)
naturally adapts to accumulated quantization error. Each layer's Hessian reflects
the actual noisy activations from prior quantized layers.

### 3. LM Head Must Not Be Quantized
Quantizing the LM head (embed_out) causes catastrophic PPL degradation (400M+ PPL).
The final projection to logits is the most sensitive layer.

### 4. Hadamard Rotation Helps Significantly
QuIP#-style Hadamard rotation spreads outlier columns across all dimensions,
making the weight distribution more uniform before quantization. This gives
~30-40% improvement in PPL ratio.

### 5. SSR Column Reordering Provides Moderate Improvement
PT2-LLM's Structural Similarity Reordering groups similar columns before
GPTQ block processing. Worth ~28% improvement on Pythia-1B.

### 6. Architecture Matters More Than Scale
- Pythia-160M: 131x, Pythia-1B: 41x, Pythia-2.8B: 32x, Pythia-6.9B: 71x
- **Phi-2 (2.7B): 10x** — same dimensions as Pythia-2.8B but 3x better ratio!
- Architecture is the dominant factor, not just parameter count
- GPT-NeoX (Pythia) may have inherent quantization sensitivity
- LLaMA, Phi, and other architectures likely much more quantization-friendly

## PT2-LLM Paper Reference (ICLR 2026)

| Model | PT2-LLM PPL | Baseline | Ratio |
|-------|-------------|----------|-------|
| LLaMA-7B | 11.39 | 5.68 | 2.0x |
| LLaMA-13B | 9.11 | ~5.25 | 1.7x |
| LLaMA-65B | 6.62 | ~5.2 | 1.3x |
| LLaMA-3-8B | 32.19 | ~6.23 | 5.2x |

Note: No published code yet (repo is placeholder). Our implementation is
based on the paper description of ITF + AGA + GPTQ + SSR.

## Remaining Gap Analysis

Our best (32x on Pythia-2.8B) vs PT2-LLM (2x on LLaMA-7B):
1. **Architecture gap**: LLaMA is likely more quantization-friendly than Pythia
2. **LLaMA-3-8B gets 5.2x** in PT2-LLM paper, much worse than LLaMA-7B's 2x
3. **Pythia-6.9B regression** (71x) shows total error accumulation is critical
4. Our implementation likely differs from PT2-LLM in calibration/error management
5. Need to test on LLaMA-7B directly for fair comparison

## Next Steps

1. Test on LLaMA-7B to compare directly against paper claims
2. Test on different architectures (Mistral, Qwen, Gemma) for generality
3. Consider PTQTP (dual trit-plane, ~3 bits) for better quality
4. Investigate weight distribution transformation (DBellQuant approach)
5. Publish successful quantizations to HuggingFace under tzervas/
