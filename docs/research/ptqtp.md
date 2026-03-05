# PTQTP: Post-Training Quantization with Trit-Planes

**Status**: Promising alternative — no calibration data needed
**Paper**: [arXiv:2505.xxxxx](https://arxiv.org/) (2025)

## Overview

PTQTP decomposes ternary weights into dual trit-planes, effectively using 2x1.58
bits per weight. This is a slightly higher bit budget than pure b1.58 but achieves
dramatically better quality without requiring calibration data.

## Key Idea: Dual Trit-Plane Decomposition

Instead of a single ternary value per weight, PTQTP decomposes each weight into
two ternary components:

```
W_approx = alpha_1 * T_1 + alpha_2 * T_2
```

Where T_1, T_2 in {-1, 0, +1} and alpha_1, alpha_2 are per-group scales.

This effectively doubles the representable values from 3 to 9 (3^2), giving
much finer granularity while still using only ternary arithmetic.

## Effective Bit Width

- Single ternary: log2(3) = 1.585 bits/weight
- Dual ternary: log2(9) = 3.17 bits/weight
- This is closer to INT3 than true b1.58

## Results

- Tested on Qwen3 and LLaMA models
- Significantly better than single-trit PTQ
- Comparable to GPTQ INT3 in many benchmarks
- No calibration data needed (purely weight-based)

## Relevance to Our Goals

**Pros:**
- No calibration data needed (simpler pipeline)
- Still uses ternary arithmetic (add/subtract/skip)
- Could leverage our existing ternary matmul kernels (just run twice)

**Cons:**
- 2x the storage of pure b1.58 (~19 GB for 32B model instead of ~9.6 GB)
- 2x the compute per matmul (two ternary matmuls + scale combine)
- Not pure b1.58 — deviates from the BitNet vision

## Comparison with PT2-LLM + QEP

| Aspect | PTQTP | PT2-LLM + QEP |
|--------|-------|----------------|
| Bit width | ~3.17 bits | ~1.58 bits |
| Calibration | None | 128 samples |
| Quality | Good | Unknown (untested) |
| Storage | ~19 GB (32B) | ~9.6 GB (32B) |
| Compute | 2x ternary matmul | 1x ternary matmul |
| Complexity | Simple | Complex |

## Verdict

PTQTP is a good fallback if pure b1.58 proves insufficient even with PT2-LLM + QEP.
It trades 2x storage/compute for much easier quantization. However, it doesn't
achieve the storage/compute efficiency that makes b1.58 compelling in the first place.

For our primary goal of maximizing compression on consumer hardware, PT2-LLM + QEP
at true b1.58 is preferred. PTQTP is the plan B.
