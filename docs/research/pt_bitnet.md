# PT-BitNet: Scaling up 1-Bit LLM with Post-Training Quantization

**Status**: Relevant but less detail available (paywalled journal paper)
**Paper**: [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S089360802500735X) (July 2025)
**Preprint**: [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4987078)

## Overview

PT-BitNet is a post-training quantization method that scales ternary quantization
to 70B parameters. Proposes a two-stage algorithm without requiring end-to-end
training or fine-tuning.

## Methodology

Two-stage algorithm:
1. **Distribution transformation**: Searches for optimal scales and thresholds to
   transform weight values into a ternarization-friendly distribution
2. **Element-wise optimization**: Optimizes each weight element to minimize
   block output error

## Key Results

- Scales to 70B parameters (largest previous BitNet was 3.9B)
- 70B PT-BitNet achieves 61% average downstream accuracy
  (vs 51.2% for BitNet b1.58 trained from scratch at that scale)
- Does not require end-to-end training or fine-tuning

## Comparison with PT2-LLM

| Aspect | PT-BitNet | PT2-LLM |
|--------|-----------|---------|
| Max model size | 70B | 70B (tested on 65B) |
| Approach | Scale/threshold search + element optimization | ITF + AGA + GPTQ |
| Calibration | Not detailed | 128 Wikitext2 samples |
| Code available | No | Placeholder repo |
| Publication | Journal (July 2025) | ICLR 2026 |
| Detail level | Less (paywalled) | Full paper available |

## Relevance

PT-BitNet validates that post-training ternary quantization CAN work at scale.
However, PT2-LLM appears to be the more refined and better-documented approach.
If PT-BitNet code becomes available, it would be worth comparing.

## Key Takeaway

Proof that post-training ternary is viable at 70B scale. The two-stage approach
(distribution transformation + element optimization) is conceptually similar to
PT2-LLM's ITF + AGA but with less detail available.
