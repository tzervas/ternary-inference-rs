# BitNet Distillation (BitDistill)

**Status**: Not feasible on consumer hardware (requires 8x MI300X)
**Paper**: [arXiv:2510.13998](https://arxiv.org/abs/2510.13998) (Microsoft, Oct 2025)
**HF Paper Page**: [huggingface.co/papers/2510.13998](https://huggingface.co/papers/2510.13998)

## Overview

BitDistill is a 3-stage pipeline that fine-tunes full-precision LLMs into 1.58-bit
(ternary) models using knowledge distillation. Achieves near-FP16 accuracy with
~10x memory savings and ~2.65x CPU speedup.

## Three-Stage Pipeline

### Stage 1: SubLN Architectural Refinement
- Insert Sub-Layer Normalization (SubLN) modules at specific positions:
  - Before the output projection of Multi-Head Self-Attention (MHSA)
  - Before the output projection of Feed-Forward Network (FFN)
- This matches the BitNet b1.58 architecture (which also uses sub-norms)
- Stabilizes activation variance when weights are quantized to ternary

### Stage 2: Continued Pre-training
- 10 billion tokens from the FALCON corpus
- Pushes weight distributions toward BitNet-like distributions
- Described as "virtually negligible" vs full 1.58-bit pretraining (~4T tokens)

### Stage 3: Dual-Signal Knowledge Distillation
- **Logits distillation**: KL divergence between teacher/student output distributions
  - Temperature: 5.0
- **Attention distillation**: Multi-head relation matching (MiniLM formulation)
  - Transfers Q/K/V relations without requiring same number of heads
  - Applied at a single selected layer (not all layers)
- Loss coefficients: lambda=10, gamma=1e5 (classification); lambda=1, gamma=1e3 (summarization)

## Hardware Requirements

- **Training**: 8x AMD MI300X GPUs
- **Batch size**: 32
- **Sequence length**: 512 tokens
- **NOT feasible on consumer hardware** (would need at minimum multi-GPU setup with
  gradient checkpointing, and even then questionable for 32B models)

## Models Tested

- Primary: Qwen3 (0.6B, 1.7B, 4B)
- Validation: Gemma-3, Qwen2.5

## Relevance to Our Work

While BitDistill itself isn't feasible on our hardware, the SubLN insight is useful:
- Our BitNet-2B-4T loader already handles sub-norms (attn_sub_norm, ffn_sub_norm)
- If we ever do QAT, inserting SubLN would be the first architectural step
- The distillation loss formulation (logits + attention relations) is well-documented
  and could inform future work if more compute becomes available

## Key Takeaway

Best quality approach but gated to datacenter hardware. The 10B token continued
pre-training alone makes this impractical on a single GPU.
