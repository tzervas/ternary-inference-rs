# Ternary Quantization Research

Research notes for making post-training ternary quantization viable on consumer hardware.

## Problem Statement

Post-training quantization of bf16 models to ternary {-1, 0, +1} produces unusable output.
Naive AbsMean quantization achieves ~5.8 dB SNR, optimal MSE achieves ~7.3 dB SNR.
Both are far too low for 64-layer transformers where errors compound catastrophically.

Natively-trained ternary models (e.g., microsoft/bitnet-b1.58-2B-4T) work well because
weights are constrained during training. The challenge is making post-training quantization
work for arbitrary pretrained models.

## Research Documents

| File | Topic |
|------|-------|
| [pt2_llm.md](pt2_llm.md) | PT2-LLM: Post-Training Ternarization (ICLR 2026) -- **primary target** |
| [bitdistill.md](bitdistill.md) | BitNet Distillation (Microsoft, Oct 2025) |
| [tequila.md](tequila.md) | Tequila: Deadzone-free Ternary QAT (ICLR 2026) |
| [pt_bitnet.md](pt_bitnet.md) | PT-BitNet: Scaling PTQ to 70B |
| [qep.md](qep.md) | QEP: Quantization Error Propagation (NeurIPS 2025) -- **critical complement** |
| [quip_sharp.md](quip_sharp.md) | QuIP#: Hadamard Incoherence (2-4 bit, foundational) |
| [ptqtp.md](ptqtp.md) | PTQTP: Dual Trit-Plane Decomposition (~3.17 bits, no calibration) |
| [ternaryllm.md](ternaryllm.md) | TernaryLLM: QAT with DLT + OFF (infeasible, but informative) |
| [baseline_experiments.md](baseline_experiments.md) | Our RTN quantization experiments and results |
| [implementation_plan.md](implementation_plan.md) | Plan for PT2-LLM implementation on consumer hardware |

## Key Findings

1. **PT2-LLM + QEP** is the most promising approach: pure post-training, no retraining,
   competitive with 2-bit PTQ at 1.58-bit. QEP prevents cross-layer error compounding.
2. **QEP** alone reduces INT2 RTN perplexity by 7.4x (90.7 -> 12.2). Combines with any PTQ.
3. **BitDistill** achieves best quality but requires 8x MI300X GPUs -- infeasible.
4. **Tequila/TernaryLLM** are QAT -- not applicable for existing models on consumer HW.
5. **PTQTP** is a fallback at ~3.17 bits (dual trit-planes), no calibration needed.
6. **QuIP#** provides foundational techniques (Hadamard rotation) applicable at any bit width.

## Hardware Constraints

- GPU: RTX 3090 Ti (24 GB VRAM, sm_86)
- RAM: 46 GB
- Target model: Qwen2.5-Coder-32B-Instruct (64 layers, 5120 hidden, 27648 intermediate)
- Strategy: Layer-by-layer processing, 4-bit model for calibration capture
