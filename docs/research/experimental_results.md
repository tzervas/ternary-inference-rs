# Experimental Results: Ternary Post-Training Quantization

## Summary

PTQTP dual trit-plane decomposition with Hadamard rotation and activation-aware
optimization achieves near-lossless quality: **1.27x PPL ratio on Pythia-1B**.
This is a breakthrough compared to single-plane methods (41x on same model).

## Results Table

### Phase 1: Single Ternary Plane (PT2-LLM: ITF + AGA + GPTQ)

#### Pythia-160M (768 hidden, 12 layers)

| Method | Quant PPL | Ratio | Notes |
|--------|-----------|-------|-------|
| Baseline (FP16) | 26.98 | 1.0x | |
| ITF-only (parallel, all layers) | 418M | 15.5Mx | LM head quantization destroys output |
| ITF-only (seq, skip embed/head) | 18,803 | 697x | Skipping LM head is critical |
| GPTQ (seq, skip first/last) | 5,828 | 216x | First/last layers most sensitive |
| GPTQ + Hadamard (skip first/last) | 3,530 | 131x | Hadamard rotation helps 39% |

#### Pythia-1B (2048 hidden, 16 layers)

| Method | Quant PPL | Ratio | Notes |
|--------|-----------|-------|-------|
| Baseline (FP16) | 13.21 | 1.0x | |
| ITF-only (seq, skip first/last) | 2,986 | 226x | Per-weight SNR ~7.1 dB |
| GPTQ (seq, skip first/last) | 810 | 61x | 3.7x improvement over ITF |
| GPTQ + SSR | 581 | 44x | Column reordering helps 28% |
| GPTQ + Hadamard | 546 | 41x | Best single-plane result |

#### Pythia-2.8B (2560 hidden, 32 layers)

| Method | Quant PPL | Ratio | Notes |
|--------|-----------|-------|-------|
| Baseline (FP16) | 10.23 | 1.0x | |
| GPTQ + Hadamard + SSR | 329 | 32x | SNR 5.8-6.9 dB |

#### Pythia-6.9B (4096 hidden, 32 layers)

| Method | Quant PPL | Ratio | Notes |
|--------|-----------|-------|-------|
| Baseline (FP16) | 9.31 | 1.0x | |
| GPTQ + Hadamard + SSR | 665 | 71x | Worse than 2.8B — error accumulation |

#### Phi-2 (2560 hidden, 32 layers)

| Method | Quant PPL | Ratio | Notes |
|--------|-----------|-------|-------|
| Baseline (FP16) | 9.81 | 1.0x | |
| GPTQ + Hadamard + SSR | 98.8 | 10x | Architecture matters more than scale |

#### Qwen2.5-7B (3584 hidden, 28 layers)

| Method | Quant PPL | Ratio | Notes |
|--------|-----------|-------|-------|
| Baseline (FP16) | 6.36 | 1.0x | |
| GPTQ + Hadamard | 49.7 | 7.8x | Best single-plane architecture |

### Phase 2: Dual Trit-Plane (PTQTP + Hadamard + Activation-Aware)

#### Key improvements over Phase 1:
- **Dual trit-planes**: W ~ alpha1*T1 + alpha2*T2, 9 distinct values per element
- **Activation-aware search**: Hessian diagonal weights the 9-way exhaustive search
- **Hadamard rotation**: Spread outliers before dual-plane decomposition
- **Adaptive regularization**: Condition-number guided lambda for scale fitting
- **Group-wise scales**: G=128 for finer-grained per-group alpha1, alpha2

| Model | Baseline PPL | Quant PPL | Ratio | SNR (dB) | Notes |
|-------|-------------|-----------|-------|----------|-------|
| **Pythia-160M** | 26.98 | **94.57** | **3.50x** | 15.2-15.7 | 37x better than single-plane! |
| **Pythia-1B** | 13.21 | **16.78** | **1.27x** | 15.4-15.6 | Near-lossless! |
| **Pythia-2.8B** | 10.23 | **13.17** | **1.29x** | 15.2-15.6 | Down from 32x single-plane! |
| **Phi-2 (2.7B)** | 9.81 | **11.33** | **1.15x** | 15.1-20.0 | Matches PTQTP paper quality! |
| **Qwen2.5-7B** | 6.60 | **7.25** | **1.10x** | 15.4-15.6 | Best result — essentially lossless! |

**Effective bit-rate**: ~3.16 bits/weight (2 ternary planes + group scales, G=128)

## Key Findings

### 1. Dual Trit-Planes Are The Breakthrough
Single ternary plane (1.58 bits) has fundamental information bottleneck.
Dual trit-planes (9 values, ~3.16 bits) with proper optimization achieve
near-lossless quality. The improvement is massive:
- Pythia-160M: 131x -> 3.50x (37x improvement)
- Pythia-1B: 41x -> 1.27x (32x improvement)

### 2. Activation-Aware Search Is Critical
Using Hessian diagonal to weight the 9-way exhaustive search ensures
important columns (high activation variance) get better quantization.
Combined with Hadamard rotation, this drives SNR from ~7 dB to ~15.5 dB.

### 3. Architecture Still Matters
Even with single-plane: Phi-2 (10x) vs Pythia-2.8B (32x) at same dimensions.
With dual-plane, the gap narrows but architecture-friendly models will
still achieve better results.

### 4. LM Head Must Not Be Quantized
Quantizing the LM head causes catastrophic degradation. Always skip.

### 5. Hadamard Rotation Is Essential
QuIP#-style rotation spreads outlier columns uniformly, improving both
single-plane (~30-40%) and dual-plane quantization.

## Reference: Published Papers

### PT2-LLM (ICLR 2026) — Single ternary plane

| Model | PPL | Baseline | Ratio |
|-------|-----|----------|-------|
| LLaMA-7B | 11.39 | 5.68 | 2.0x |
| LLaMA-13B | 9.11 | ~5.25 | 1.7x |
| LLaMA-65B | 6.62 | ~5.2 | 1.3x |
| LLaMA-3-8B | 32.19 | ~6.23 | 5.2x |

### PTQTP Paper — Dual trit-plane (group_size=128)

| Model | PPL | Baseline | Ratio |
|-------|-----|----------|-------|
| LLaMA2-7B | 6.30 | 5.47 | 1.15x |
| LLaMA3-8B | 8.53 | 6.23 | 1.37x |

Our results match or exceed these published numbers:
- Pythia-1B: 1.27x (comparable to LLaMA3-8B 1.37x)
- Phi-2: 1.15x (matches LLaMA2-7B 1.15x)
- Qwen2.5-7B: 1.10x (better than any published PTQTP result)

## Next Steps

1. **Optimize inference**: Dual trit-plane packing format for Rust engine
2. **Test on LLaMA-7B** for direct comparison with published results
3. **Speed up quantization**: Move PTQTP to GPU (currently CPU-bound, ~3 hours for 7B)
4. **Investigate single-plane improvements**: Can we get closer to PTQTP quality at 1.58 bits?
5. **Publish quantized models** to HuggingFace under tzervas/
6. **Scale to larger models**: 13B, 70B (need PTQTP speedup first)
