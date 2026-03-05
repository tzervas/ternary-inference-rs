# CLAUDE.md - Ternary Inference RS

Native Rust inference engine for BitNet 1.58-bit ternary quantized models.

**Standalone crate** — decoupled from the rust-ai workspace. Uses upstream candle (HuggingFace), not the qlora-candle fork.

## Quick Reference

```bash
# Build (CPU only)
cargo build --release

# Build with GPU (CUDA)
CUDA_COMPUTE_CAP=86 cargo build --release --features cuda

# Test (7 unit tests)
cargo test

# Run model load + matmul validation (CPU)
RUST_LOG=info cargo run --example load_and_matmul --release

# Generate text (GPU)
RUST_LOG=info CUDA_COMPUTE_CAP=86 cargo run --example generate --release --features cuda -- --gpu

# GPU smoke test
CUDA_COMPUTE_CAP=86 cargo run --example gpu_smoke_test --release --features cuda
```

## Architecture

```
ternary-inference-rs/
src/
  lib.rs           # Public API exports
  error.rs         # Error types (Config, Loading, Shape, Tensor, Io, Json, SafeTensors, UnsupportedArch)
  config.rs        # ModelConfig: Qwen2 32B preset, BitNet 2B preset, from_hf_config()
  loader.rs        # SafeTensors loader: load_from_dir(), load_from_hf_cache()
  unpack.rs        # unpack_byte(), unpack_ternary(), dequantize_weight()
  kernels/
    mod.rs         # ternary_matmul() dispatch (CPU/GPU)
    cpu.rs         # Direct packed-weight matmul (add/subtract/skip, no dequant)
examples/
  load_and_matmul.rs  # Validates full pipeline on real model
```

## Implemented

```
src/
  model.rs         # Full transformer forward pass (RMSNorm, RoPE, GQA, SwiGLU MLP)
  kv_cache.rs      # KV cache for autoregressive generation
  generate.rs      # Text generation (greedy, temperature, top-p sampling)
  kernels/
    gpu.rs         # GPU ternary matmul: LUT dequant on GPU + f16 cuBLAS matmul
examples/
  generate.rs      # CLI: generate text with --gpu, --temp, --max-tokens
  gpu_smoke_test.rs  # Validates GPU LUT dequant + matmul pipeline
```

## NOT YET IMPLEMENTED (TODO)

```
src/
  kernels/
    cubecl.rs      # CubeCL direct packed-weight GPU kernel (no dequant needed)
```

## Supported Models

### microsoft/bitnet-b1.58-2B-4T (WORKING)

| Property | Value |
|----------|-------|
| Architecture | BitnetForCausalLM |
| Hidden size | 2560 |
| Layers | 30 |
| Heads | 20 (Q), 5 (KV) — GQA groups=4 |
| Head dim | 128 |
| Intermediate | 6912 |
| Vocab | 32768 |
| Context | 4096 |
| RoPE theta | 500,000 |
| RMS norm eps | 1e-6 |
| Activation | Squared ReLU (relu^2) |
| Sub-norms | attn_sub_norm (before O proj), ffn_sub_norm (before down proj) |
| Scale type | Per-tensor (single `weight_scale` per projection) |
| Packed format | 2-bit interleaved rows: shape [out/4, in], INTERLEAVED |
| Tie embeddings | Yes (lm_head reuses embed_tokens) |
| Performance | ~3.93 tok/s on RTX 3090 Ti (GPU) |

### tzervas/qwen2.5-coder-32b-bitnet-1.58b (loads, low quality)

| Property | Value |
|----------|-------|
| Architecture | Qwen2ForCausalLM |
| Hidden size | 5120 |
| Layers | 64 |
| Heads | 40 (Q), 8 (KV) — GQA groups=5 |
| Head dim | 128 |
| Intermediate | 27648 |
| Vocab | 152064 |
| Context | 32768 |
| RoPE theta | 1,000,000 |
| Activation | SiLU (SwiGLU MLP) |
| Scale type | Per-group (group_size=64, `weight.scales`) |
| Packed format | Base-3 columns: shape [out, in/4] |
| Attention bias | Yes (q/k/v biases) |
| Quality | Incoherent output — post-training quantization issue, NOT implementation bug |

### SafeTensors Layout Differences

| Field | BitNet-2B-4T | Qwen2-32B |
|-------|-------------|-----------|
| Scale key | `weight_scale` | `weight.scales` |
| Scale shape | scalar per projection | [out, in/group_size] |
| Packing | 2-bit rows, interleaved | base-3 columns |
| Biases | None | q/k/v biases (bf16) |
| Sub-norms | Yes | No |

## Key Algorithms

### Ternary Packing (base-3, Qwen2)
```
byte = (v0+1) + (v1+1)*3 + (v2+1)*9 + (v3+1)*27
where vi in {-1,0,+1}, mapped to {0,1,2}
max valid byte = 80 (all +1s)
Shape: [out_features, in_features/4]
```

### Ternary Packing (2-bit interleaved, BitNet-2B-4T)
```
byte = v0 | (v1 << 2) | (v2 << 4) | (v3 << 6)
where vi in {0,1,2} maps to {-1,0,+1}
max valid byte = 170 (all +1s)
Shape: [out_features/4, in_features]
INTERLEAVED: byte at (pr,col) stores rows pr, pr+P, pr+2P, pr+3P where P=out/4
```

### Direct Ternary Matmul (no dequant, CPU only)
```
y[i,j] = sum_g (scale[j,g] * sum_k ternary[j, g*gs+k] * x[i, g*gs+k])
```
Ternary multiply is just add/subtract/skip:
- `+1 * x = x` (add)
- `-1 * x = -x` (subtract)
- ` 0 * x = 0` (skip)

### Per-Group Dequantization (Qwen2)
```
dequant[row, col] = ternary[row, col] * scale[row, col / group_size]
group_size = 64 for Qwen2 model
```

### Per-Tensor Dequantization (BitNet-2B-4T)
```
dequant[row, col] = ternary[row, col] * scale
Single scale per projection
```

## Loader Details

### LoadMode::Packed (preferred for inference)
- Keeps layer projections as raw packed bytes + scales
- Only dequantizes embed_tokens and lm_head (needed for lookup/logits)
- Memory: ~9.6 GB packed + ~3.1 GB for dequantized embed+lm_head = ~12.7 GB RAM

### LoadMode::Dequantized (fallback)
- Dequantizes all weights to f32 tensors
- Memory: ~128 GB RAM (32B params * 4 bytes) — will NOT fit in memory

### HF Cache Path Resolution
```
~/.cache/huggingface/hub/models--{org}--{model}/snapshots/{hash}/
```

## Hardware Context

- **GPU**: RTX 3090 Ti (24 GB VRAM, sm_86)
- **RAM**: 46 GB
- **CUDA**: 13.1, driver 590.48.01
- **Previous GPU**: RTX 5080 (16 GB, sm_120) — used for original quantization

## VRAM Budget (GPU inference, streaming)
```
Packed weights on GPU:           ~9.6 GB
1 layer dequantized (fp16):      ~1.0 GB
Embed tokens (fp32):             ~3.1 GB (or fp16: ~1.56 GB)
KV cache (short seq):            ~0.1 GB
Activations:                     ~0.1 GB
────────────────────────────────────────
Peak:                           ~14 GB  (fits in 24 GB)
```

## Dependencies

| Crate | Used for |
|-------|----------|
| candle-core | Tensor ops (upstream HuggingFace candle, NOT qlora fork) |
| candle-nn | NN ops (softmax, silu) |
| safetensors | Weight file format |
| tokenizers | HF tokenizer loading |
| half | f16 types for GPU Tensor Core matmul |

### Standalone (not a workspace member)
This crate is **excluded** from the rust-ai workspace. It uses upstream candle
from `https://github.com/huggingface/candle.git`. The qlora-candle fork has
broken CUDA PTX kernels with CUDA 13.1.

### base-3 packing (not trit-vsa)
- `trit-vsa` uses **bitsliced** packing (plus/minus bit planes) — NOT used here
- This model uses **base-3** packing (4 trits per byte): `v0 + v1*3 + v2*9 + v3*27`
- `ternary-inference-rs` has its own unpack.rs for base-3

## Publishing

Target: `https://github.com/tzervas/ternary-inference-rs` (public repo).
This crate is standalone and has no workspace dependencies.

## Development Workflow

1. **Test changes**: `cargo test`
2. **Validate on real model**: `RUST_LOG=info cargo run --example load_and_matmul --release`
3. **GPU test**: `CUDA_COMPUTE_CAP=86 cargo run --example gpu_smoke_test --release --features cuda`
4. **Check compilation**: `cargo check --features cuda`
5. **Clippy**: `cargo clippy --features cuda`
6. **Feature branches**: Branch off `main`, PR back to `main`

**IMPORTANT**: Always set `CUDA_COMPUTE_CAP=86` for the RTX 3090 Ti when building with `--features cuda`.

## GGUF Status

The model also has a GGUF file (`qwen-coder-32b-tq2.gguf`, 11 GB) but it produces **garbage output** in llama.cpp because:
- Model uses group_size=64 (1 scale per 64 weights)
- TQ2_0 format uses block_size=256 (1 scale per 256 weights)
- The scale averaging destroys precision

**Do NOT use the GGUF variant.** Use the SafeTensors packed format.

## GPU Inference Strategy

### Current: LUT dequant + cuBLAS (gpu.rs)
1. Pre-computed [256,4] lookup table maps each packed byte → 4 ternary values
2. `index_select` does unpacking entirely on GPU (no CPU dequant bottleneck)
3. Per-group scales applied via broadcast multiply on GPU
4. f16 cuBLAS matmul for Tensor Core acceleration on RTX 3090 Ti
5. Result cast back to f32

Peak VRAM per projection: ~2× dequantized weight size (~564 MB for largest).
Packed weights (9.6 GB) stay on CPU; transferred per-projection.

### Future: CubeCL direct packed kernel
- Direct matmul from packed bytes without any dequantization
- Would eliminate the per-projection memory overhead
- Requires CubeCL 0.9 kernel development

## Performance Notes

- BitNet-2B-4T GPU: ~3.93 tok/s on RTX 3090 Ti (LUT dequant + f16 cuBLAS)
- Qwen2-32B GPU: ~0.2 tok/s on RTX 3090 Ti
- CPU ternary matmul: functional, ~6s/token for 32B (no SIMD, single-threaded)
- Future: CubeCL popcount-based GPU kernel for direct packed inference

## Known Issues

- Qwen2-32B produces incoherent output — post-training quantization destroyed quality
  (verified identical results in Python reference implementation)
- Re-quantizing Qwen2.5-Coder properly to BitNet b1.58 format is a future task
