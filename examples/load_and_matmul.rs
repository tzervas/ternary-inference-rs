//! Load the first layer from the BitNet model and run a ternary matmul.
//!
//! Validates the full pipeline: SafeTensors → packed ternary → matmul.
//!
//! Usage:
//!   cargo run -p ternary-inference-rs --example load_and_matmul --release

use candle_core::{Device, Tensor};
use ternary_inference_rs::config::ModelConfig;
use ternary_inference_rs::kernels::ternary_matmul;
use ternary_inference_rs::loader::{load_from_hf_cache, LoadMode};

fn main() -> anyhow::Result<()> {
    env_logger::init();

    let config = ModelConfig::qwen2_32b();
    let device = Device::Cpu;

    println!("Loading model: {:?}", config.hf_repo_id);
    println!("Architecture: {:?}", config.architecture);
    println!("Layers: {}, Hidden: {}", config.num_layers, config.hidden_size);
    println!("Group size: {}", config.group_size);
    println!("Load mode: Packed (direct ternary matmul, no dequant)");
    println!();

    let weights = load_from_hf_cache(
        "tzervas/qwen2.5-coder-32b-bitnet-1.58b",
        &config,
        &device,
        LoadMode::Packed,
    )?;

    println!("\nLoaded successfully!");
    println!("  embed_tokens shape: {:?}", weights.embed_tokens.shape());
    println!("  final_norm shape: {:?}", weights.final_norm.shape());
    println!("  lm_head shape: {:?}", weights.lm_head.shape());
    println!("  layers: {}", weights.layers.len());

    // Check layer 0
    let layer0 = &weights.layers[0];
    if let Some(ref pq) = layer0.packed_q {
        println!("\n  Layer 0 q_proj (packed):");
        println!("    packed bytes: {}", pq.packed.len());
        println!("    scales: {}", pq.scales.len());
        println!("    out_features: {}, in_features: {}", pq.out_features, pq.in_features);

        // Run a ternary matmul with a small random input
        println!("\n  Running ternary matmul on layer 0 q_proj...");
        let input = Tensor::randn(0.0f32, 1.0, (1, pq.in_features), &device)?;
        let output = ternary_matmul(
            &input,
            &pq.packed,
            &pq.scales,
            pq.out_features,
            pq.in_features,
            config.group_size,
            &device,
        )?;
        println!("    Input shape:  {:?}", input.shape());
        println!("    Output shape: {:?}", output.shape());

        // Print some output stats
        let out_vec: Vec<f32> = output.flatten_all()?.to_vec1()?;
        let mean = out_vec.iter().sum::<f32>() / out_vec.len() as f32;
        let max = out_vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let min = out_vec.iter().cloned().fold(f32::INFINITY, f32::min);
        println!("    Output stats: mean={mean:.4}, min={min:.4}, max={max:.4}");
        println!("\n  SUCCESS: Ternary matmul works on real model weights!");
    } else if let Some(ref q) = layer0.q_proj {
        println!("\n  Layer 0 q_proj loaded as float tensor: {:?}", q.shape());
    } else {
        println!("\n  WARNING: Layer 0 q_proj not loaded!");
    }

    // Check biases (Qwen2 specific)
    if layer0.q_bias.is_some() {
        println!("  Layer 0 has q_bias (Qwen2)");
    }
    if layer0.k_bias.is_some() {
        println!("  Layer 0 has k_bias (Qwen2)");
    }

    Ok(())
}
