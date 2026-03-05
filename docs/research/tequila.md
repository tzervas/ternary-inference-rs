# Tequila: Deadzone-free Ternary Quantization

**Status**: QAT method (training from scratch), not applicable for post-training
**Paper**: [arXiv:2509.23809](https://arxiv.org/abs/2509.23809) (ICLR 2026)
**OpenReview**: [openreview.net/forum?id=9CZzD5LWdy](https://openreview.net/forum?id=9CZzD5LWdy)

## Overview

Tequila addresses the "deadzone trapping" problem in ternary QAT, where a large
fraction of weights get stuck at 0 during training because they receive only noisy,
uninformative gradients at the ternary deadzone boundary.

## Core Problem: Deadzone Trapping

In standard ternary QAT:
- Weights near the boundary between 0 and +/-1 get "trapped"
- These trapped weights receive noisy gradients that prevent stable escape
- Results in excessive sparsity (too many zeros) and reduced model capacity

## Solution: Adaptive Dynamic Biases

Tequila repurposes trapped weights as adaptive dynamic biases:
- Trapped weights are reactivated to enhance model expressiveness
- Provides direct gradient signals for efficient escape from the deadzone
- Nearly zero inference overhead

## Results

- ARC benchmark: >4% accuracy gain over SOTA baseline
- Within <1% of full-precision performance with 3.0x inference speedup
- TequilaLLM-3B outperforms Spectra-3.9B by 0.9% using only 10% training tokens

## Relevance to Our Work

**Not directly applicable** -- Tequila requires QAT (quantization-aware training)
from scratch, which is computationally expensive and not a post-training method.

However, the deadzone trapping insight is relevant:
- Our RTN experiments show 45% zeros with optimal MSE scale (vs 31% with AbsMean)
- The optimal MSE scale pushes more weights to zero because it minimizes MSE
  globally, but this may not be optimal for model quality
- PT2-LLM's asymmetric quantizer with per-row offset (mu) may partially address
  this by shifting the deadzone boundaries

## Key Takeaway

If future work involves training a ternary model from scratch (rather than
post-training quantization), Tequila's deadzone-free approach should be considered.
