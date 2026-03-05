# TernaryLLM: Dual Learnable Ternarization + Outlier-Friendly Feature KD

**Status**: QAT method — not directly applicable to our PTQ pipeline
**Paper**: [arXiv:2506.xxxxx](https://arxiv.org/) (2025)

## Overview

TernaryLLM is a quantization-aware training (QAT) method that achieves
state-of-the-art ternary model quality through two innovations:
1. Dual Learnable Ternarization (DLT) for weight quantization
2. Outlier-Friendly Feature Knowledge Distillation (OFF) for training

## Dual Learnable Ternarization (DLT)

Standard ternary quantization uses a single threshold to decide {-1, 0, +1}.
DLT uses two separate learnable thresholds:

```
T(w) = +1  if w > threshold_pos
T(w) =  0  if threshold_neg <= w <= threshold_pos
T(w) = -1  if w < threshold_neg
```

The thresholds are asymmetric and learned via straight-through estimation (STE)
during training. This allows the model to learn the optimal dead-zone width
per layer/group.

## Outlier-Friendly Feature Knowledge Distillation (OFF)

Standard KD minimizes KL divergence between teacher and student logits.
OFF instead distills intermediate features (hidden states) with special
handling for outlier channels:

1. Identify outlier channels (those with unusually large activations)
2. Apply separate loss weighting to outlier vs normal channels
3. This prevents the ternary student from wasting capacity trying to
   reproduce outlier magnitudes it can't represent

## Results

- State-of-the-art ternary quality on LLaMA-7B/13B/30B
- Requires full QAT training (100K+ steps with teacher model)
- Needs teacher model in memory alongside student

## Relevance to Our Goals

**Not directly applicable** — TernaryLLM requires QAT, which means:
- Full training loop with gradient computation
- Teacher model (fp16/bf16) must fit in memory alongside student
- For 32B model: need ~64 GB VRAM minimum (teacher + student + optimizer)
- Far beyond our RTX 3090 Ti (24 GB) capability

**Indirect value:**
- The DLT concept (asymmetric thresholds) is already captured by PT2-LLM's
  per-row offset mu, which achieves similar asymmetry without training
- The outlier insight suggests we might benefit from identifying and
  specially handling outlier channels during PTQ as well

## Key Takeaway

TernaryLLM confirms that asymmetric quantization (different positive/negative
thresholds) is critical for ternary quality. PT2-LLM's per-row (alpha, mu)
parameterization captures this insight in a PTQ-compatible way.

The OFF technique suggests that activation outliers matter — which aligns
with PT2-LLM's Activation-aware Grid Alignment (AGA) that uses activation
covariance to weight the quantization objective.
