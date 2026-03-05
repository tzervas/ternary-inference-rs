//! GPU-accelerated inference engine for BitNet 1.58-bit ternary models.
//!
//! This crate provides native Rust inference for models quantized with
//! BitNet b1.58 ternary quantization ({-1, 0, +1} weights). It supports:
//!
//! - **Packed ternary weight loading** from SafeTensors (base-3, 4 values/byte)
//! - **Per-group scale dequantization** (arbitrary group sizes including 64)
//! - **GPU-accelerated ternary matmul** via CubeCL (popcount-based)
//! - **Qwen2, LLaMA, and Mistral** architectures
//! - **GGUF loading** with proper TQ2_0 handling
//!
//! # Architecture
//!
//! ```text
//! ternary-inference-rs/
//! ├── src/
//! │   ├── lib.rs          # Public API
//! │   ├── error.rs        # Error types
//! │   ├── config.rs       # Model configuration (Qwen2, LLaMA, etc.)
//! │   ├── loader.rs       # Weight loading (SafeTensors, GGUF)
//! │   ├── unpack.rs       # Ternary unpacking and dequantization
//! │   ├── model.rs        # Transformer model (attention, MLP, norms)
//! │   ├── kv_cache.rs     # KV cache for autoregressive generation
//! │   ├── generate.rs     # Text generation (greedy, sampling, beam)
//! │   └── kernels/
//! │       ├── mod.rs       # Kernel dispatch (CPU/GPU)
//! │       ├── cpu.rs       # CPU ternary matmul (popcount)
//! │       └── cubecl.rs    # CubeCL GPU ternary matmul kernel
//! ```
//!
//! # Quick Start
//!
//! ```rust,ignore
//! use ternary_inference_rs::{TernaryModel, ModelConfig};
//!
//! // Load a BitNet-quantized model from HuggingFace
//! let config = ModelConfig::qwen2_32b();
//! let model = TernaryModel::from_hub("tzervas/qwen2.5-coder-32b-bitnet-1.58b", &config)?;
//!
//! // Generate text
//! let output = model.generate("Write a function to check primes:", 100)?;
//! println!("{}", output);
//! ```

pub mod config;
pub mod error;
pub mod generate;
pub mod kernels;
pub mod kv_cache;
pub mod loader;
pub mod model;
pub mod unpack;

pub use config::ModelConfig;
pub use error::{Error, Result};
pub use generate::{generate, GenerationConfig};
pub use model::TernaryModel;
pub use unpack::{dequantize_weight, unpack_ternary};
