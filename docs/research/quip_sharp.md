# QuIP#: Even Better LLM Quantization with Hadamard Incoherence and Lattice Codebooks

**Status**: Foundational technique, applicable at any bit width
**Paper**: [arXiv:2402.04396](https://arxiv.org/abs/2402.04396) (ICML 2024)
**Code**: [github.com/Cornell-RelaxML/quip-sharp](https://github.com/Cornell-RelaxML/quip-sharp)

## Overview

QuIP# is a weight-only post-training quantization method that achieves state-of-the-art
results in extreme compression (2-4 bits per weight). While not directly ternary,
its core technique (Hadamard incoherence) is applicable at any bit width.

## Key Techniques

### Hadamard Incoherence Processing
- Apply random Hadamard rotation to weights before quantization
- Makes weight distribution more uniform (reduces outliers)
- Better theoretical properties than raw weight quantization
- The rotation is inverted at inference time (applied to activations instead)

### E8 Lattice Codebooks
- Uses vector quantization with the E8 lattice (optimal 8D sphere packing)
- Hardware-efficient lookup-based dequantization
- Not directly applicable to ternary (ternary is scalar, not vector)

## Results

- First PTQ method where 3-bit models scale better than 4-bit models
- Near-lossless at 4-bit, competitive at 2-bit
- Focuses on 2-4 bit range, not 1.58-bit ternary

## Relevance to Ternary Quantization

The Hadamard rotation idea could be combined with PT2-LLM:

1. Apply Hadamard rotation H to weight matrix: W' = W @ H
2. Run PT2-LLM ternarization on W' (which has better distribution properties)
3. At inference: dequantize to W'_hat, then compute output = (W'_hat @ H^T) @ x
   = W'_hat @ (H^T @ x), so we just need to rotate the input activations

This would add O(d^2) cost per projection (the Hadamard rotation) but could
significantly improve quantization quality.

**Open question**: Does Hadamard rotation actually help at 1.58-bit? The benefit
is largest when quantization levels are few but > 2 (at exactly ternary, the
distribution may not matter as much since there are only 3 levels).

## Key Takeaway

Hadamard rotation is a powerful, free (no training) technique that could be
combined with any ternary quantization method. Worth experimenting with as
an optional pre-processing step before PT2-LLM's ITF.
