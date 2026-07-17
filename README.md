# ternary-inference-rs

<!-- FLEET-BADGES:BEGIN -->
[![CI](https://github.com/tzervas/ternary-inference-rs/actions/workflows/fleet-ci.yml/badge.svg?branch=main)](https://github.com/tzervas/ternary-inference-rs/actions/workflows/fleet-ci.yml?query=branch%3Amain)
[![Security](https://github.com/tzervas/ternary-inference-rs/actions/workflows/fleet-security.yml/badge.svg?branch=main)](https://github.com/tzervas/ternary-inference-rs/actions/workflows/fleet-security.yml?query=branch%3Amain)
<!-- FLEET-BADGES:END -->

GPU-accelerated inference engine for BitNet 1.58-bit ternary models in Rust.

Runs natively-trained 1.58-bit models like [microsoft/bitnet-b1.58-2B-4T](https://huggingface.co/microsoft/bitnet-b1.58-2B-4T) on consumer GPUs with minimal VRAM.

## Features

- **Two ternary weight formats**: Microsoft's 2-bit interleaved packing and base-3 per-group packing
- **GPU-accelerated inference**: LUT-based dequantization + cuBLAS f16 matmul via CUDA
- **CPU fallback**: Direct packed-weight matmul (add/subtract/skip, no dequantization)
- **Qwen2 + BitNet architectures**: GQA attention, RoPE, SwiGLU/ReLU^2 MLP, sub-layer norms
- **Streaming text generation**: Greedy, temperature, and top-p sampling

## Supported Models

| Model | Size | Architecture | Status |
|-------|------|-------------|--------|
| [microsoft/bitnet-b1.58-2B-4T](https://huggingface.co/microsoft/bitnet-b1.58-2B-4T) | 2B params, ~600 MB | BitNet (natively trained) | Working |
| [tzervas/qwen2.5-coder-32b-bitnet-1.58b](https://huggingface.co/tzervas/qwen2.5-coder-32b-bitnet-1.58b) | 32B params, ~9.6 GB | Qwen2 (post-training quantized) | Loads, low quality |

## Quick Start

### Prerequisites

- Rust 1.82+
- CUDA toolkit (for GPU acceleration)
- Model downloaded via `huggingface-cli`

### Download a model

```bash
pip install huggingface-hub
huggingface-cli download microsoft/bitnet-b1.58-2B-4T
```

### Build and run

```bash
# GPU inference (recommended)
CUDA_COMPUTE_CAP=86 cargo run --example generate --release --features cuda -- --gpu

# CPU inference (slower)
cargo run --example generate --release

# Custom prompt
cargo run --example generate --release --features cuda -- \
  --gpu --max-tokens 100 "Write a hello world in Rust:"
```

### CLI Options

```
--gpu              Use CUDA GPU (requires --features cuda)
--model <name>     Model preset: "bitnet-2b" (default), "qwen2-32b"
--temp <f32>       Sampling temperature (0.0 = greedy, default)
--max-tokens <n>   Maximum tokens to generate (default: 256)
<prompt>           Text prompt
```

## Architecture

```
src/
  lib.rs           # Public API
  config.rs        # Model configs (Qwen2, BitNet presets)
  loader.rs        # SafeTensors weight loading (packed ternary + float)
  unpack.rs        # Ternary unpacking: base-3 (column) and 2-bit (interleaved row)
  model.rs         # Transformer forward pass (RMSNorm, RoPE, GQA, SwiGLU/ReLU^2)
  kv_cache.rs      # KV cache for autoregressive generation
  generate.rs      # Text generation with sampling
  kernels/
    mod.rs         # Kernel dispatch (CPU/GPU, format-aware)
    cpu.rs         # CPU ternary matmul (direct packed, no dequant)
    gpu.rs         # GPU LUT dequant + cuBLAS f16 matmul
```

## How It Works

### Ternary Weight Packing

BitNet models quantize weights to {-1, 0, +1} and pack them efficiently:

**Base-3 (Qwen2 models)**: 4 values per byte along columns.
`byte = v0 + v1*3 + v2*9 + v3*27` (max valid = 80)

**2-bit interleaved (BitNet-2B-4T)**: 4 values per byte along rows, interleaved.
`byte = v0 | (v1 << 2) | (v2 << 4) | (v3 << 6)` (max valid = 170)

### GPU Inference Pipeline

1. Pre-computed [256,4] lookup table maps each packed byte to 4 ternary values
2. `index_select` performs unpacking entirely on GPU (no CPU-GPU transfer per token)
3. Per-group/per-tensor scales applied via broadcast multiply
4. cuBLAS f16 GEMM handles the actual matrix multiplication
5. Results cast back to f32 for the transformer computation

### Performance

On RTX 3090 Ti (24 GB VRAM, sm_86):

| Model | Tokens/sec | VRAM Usage |
|-------|-----------|------------|
| BitNet-2B-4T | ~3 tok/s | ~2 GB |
| Qwen2-32B | ~0.2 tok/s | ~14 GB |

## CUDA Support

Set `CUDA_COMPUTE_CAP` for your GPU architecture:

| GPU | Compute Cap |
|-----|------------|
| RTX 3090/3090 Ti | 86 |
| RTX 4090 | 89 |
| RTX 5080/5090 | 120 |
| A100 | 80 |
| H100 | 90 |

```bash
CUDA_COMPUTE_CAP=86 cargo build --release --features cuda
```

## License

MIT
