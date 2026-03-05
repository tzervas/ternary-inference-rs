# Baseline Quantization Experiments

Results from our initial ternary quantization attempts on Qwen2.5-Coder-32B-Instruct.

## Experiment 1: Existing Model (tzervas/qwen2.5-coder-32b-bitnet-1.58b)

**Method**: AbsMean per-group quantization (group_size=64)
**Created with**: Custom Rust tool using Candle, 369 seconds on RTX 5080

| Metric | Value |
|--------|-------|
| SNR | ~5.8 dB (estimated from weight stats) |
| Distribution | 34.4% (-1), 31.1% (0), 34.5% (+1) |
| Max byte | 80 (all valid) |
| Output quality | **Incoherent** -- repeating tokens, no meaningful text |
| embed_tokens | Packed ternary (NOT bf16) |
| lm_head | Packed ternary (NOT bf16) |
| Confirmed in Python | Yes -- identical garbled output in PyTorch reference |

## Experiment 2: Re-quantization with Optimal MSE + bf16 embed/lm_head

**Method**: Iterative optimal MSE scale (3 rounds) + bf16 embed_tokens and lm_head
**Script**: `quantize_ternary.py`
**Runtime**: 426 seconds on CPU

| Metric | Value |
|--------|-------|
| SNR | min=5.1 dB, mean=7.1 dB, max=7.4 dB |
| Distribution | ~27% (-1), 45% (0), 27% (+1) |
| Output quality | **Incoherent** -- asterisks and whitespace |
| embed_tokens | bf16 (778.6M params) |
| lm_head | bf16 (778.6M params) |
| Total size | 12 GB (vs 9.6 GB for old model, +3 GB for bf16 embed/lm_head) |

### Key Observations

1. **Optimal MSE increases sparsity**: 45% zeros vs 31% with AbsMean. The optimal
   scale minimizes MSE but creates more zero values. This may not be desirable for
   model quality.

2. **bf16 embed/lm_head didn't help**: The output was still completely garbled.
   The quantization noise in the 64 transformer layers dominates.

3. **7 dB SNR is catastrophically low**: Each weight has ~20% relative error.
   Through 64 layers, this compounds to total information loss.

## Scale Factor Analysis

Tested AbsMean with different multipliers on layer 0 q_proj:

| Scale Factor | SNR (dB) | Zeros (%) |
|-------------|----------|-----------|
| 0.6x | 3.5 | 19% |
| 0.7x | 4.1 | 22% |
| 0.8x | 4.7 | 25% |
| 0.9x | 5.3 | 28% |
| 1.0x (AbsMean) | 5.8 | 31% |
| 1.1x | 6.3 | 34% |
| 1.2x | 6.6 | 37% |
| Optimal MSE | 7.3 | 45% |

**Insight**: There's a direct tradeoff between SNR and sparsity. Higher SNR (less
error per non-zero weight) comes at the cost of more weights being zeroed out.
Neither extreme works for model quality.

## Why Post-Training Ternary Fails at This Scale

1. **Information loss**: 1.58 bits per weight vs 16 bits = 10x less information.
   At 7 dB SNR, the quantization noise is 1/5 of the signal power.

2. **Error compounding**: 64 layers of ~7 dB SNR = total signal destruction.
   Each layer amplifies errors from previous layers.

3. **No error compensation**: Naive RTN quantizes each weight independently.
   No coordination between weights in a row or across layers.

4. **Symmetric assumption**: AbsMean assumes zero-mean weight distribution.
   Many weight rows have non-zero means that are lost with symmetric quantization.

## Conclusions

- Simple RTN (round-to-nearest) ternary quantization is fundamentally insufficient
  for models not trained with ternary constraints
- The improvements (bf16 embed/lm_head, optimal MSE) are necessary but not sufficient
- GPTQ-style error compensation + asymmetric quantization (PT2-LLM approach) is needed
- Even with GPTQ, the question remains whether 1.58-bit is enough for this model
