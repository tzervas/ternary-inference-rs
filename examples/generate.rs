//! Generate text using BitNet 1.58-bit ternary models.
//!
//! Supports:
//!   - microsoft/bitnet-b1.58-2B-4T (recommended, natively trained ternary)
//!   - tzervas/qwen2.5-coder-32b-bitnet-1.58b (post-training quantized, experimental)
//!
//! Usage:
//!   RUST_LOG=info cargo run --example generate --release --features cuda -- --gpu
//!   RUST_LOG=info cargo run --example generate --release --features cuda -- --gpu --model qwen2-32b
//!
//! Options:
//!   --gpu              Use CUDA GPU
//!   --model <name>     Model preset: "bitnet-2b" (default), "qwen2-32b"
//!   --temp <f32>       Sampling temperature (0.0 = greedy)
//!   --max-tokens <n>   Maximum tokens to generate
//!   <prompt>           Text prompt (positional)

use candle_core::Device;
use ternary_inference_rs::config::ModelConfig;
use ternary_inference_rs::generate::{generate, load_tokenizer_from_hf_cache, GenerationConfig};
use ternary_inference_rs::loader::{load_from_hf_cache, LoadMode};
use ternary_inference_rs::model::TernaryModel;

fn main() -> anyhow::Result<()> {
    env_logger::init();

    let args: Vec<String> = std::env::args().skip(1).collect();

    let mut temperature = 0.0f32;
    let mut max_tokens = 256usize;
    let mut use_gpu = false;
    let mut model_name = String::from("bitnet-2b");
    let mut prompt = String::from("def fibonacci(n):");
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--temp" | "-t" => {
                i += 1;
                temperature = args[i].parse()?;
            }
            "--max-tokens" | "-n" => {
                i += 1;
                max_tokens = args[i].parse()?;
            }
            "--gpu" => use_gpu = true,
            "--model" | "-m" => {
                i += 1;
                model_name = args[i].clone();
            }
            _ => {
                prompt = args[i..].join(" ");
                break;
            }
        }
        i += 1;
    }

    let (model_id, config, eos_token_id) = match model_name.as_str() {
        "bitnet-2b" | "bitnet" => (
            "microsoft/bitnet-b1.58-2B-4T",
            ModelConfig::bitnet_2b(),
            128001u32,
        ),
        "qwen2-32b" | "qwen2" => (
            "tzervas/qwen2.5-coder-32b-bitnet-1.58b",
            ModelConfig::qwen2_32b(),
            151643u32,
        ),
        other => anyhow::bail!("Unknown model: {other}. Use 'bitnet-2b' or 'qwen2-32b'"),
    };

    let device = if use_gpu {
        Device::new_cuda(0)?
    } else {
        Device::Cpu
    };

    println!("Model: {model_id}");
    println!("Device: {device:?}");
    println!("Prompt: {prompt}");
    println!("Temperature: {temperature}");
    println!("Max tokens: {max_tokens}\n");

    println!("Loading tokenizer...");
    let tokenizer = load_tokenizer_from_hf_cache(model_id)?;
    println!("Tokenizer loaded ({} vocab)", tokenizer.get_vocab_size(false));

    println!("Loading model weights (packed mode)...");
    let weights = load_from_hf_cache(model_id, &config, &device, LoadMode::Packed)?;
    println!("Weights loaded: {} layers", weights.layers.len());

    println!("Initializing model...");
    let mut model = TernaryModel::new(config, weights, &device)?;
    println!("Model ready.\n");

    let gen_config = GenerationConfig {
        max_tokens,
        temperature,
        eos_token_id,
        stream: true,
        ..Default::default()
    };

    println!("--- Generation ---");
    eprint!("{prompt}");
    let start = std::time::Instant::now();
    let output = generate(&mut model, &tokenizer, &prompt, &gen_config)?;
    let elapsed = start.elapsed();

    let tokens_generated = tokenizer
        .encode(output.as_str(), false)
        .map(|e| e.get_ids().len())
        .unwrap_or(0);
    let tokens_per_sec = tokens_generated as f64 / elapsed.as_secs_f64();

    println!("\n--- Stats ---");
    println!(
        "Generated {tokens_generated} tokens in {:.1}s ({tokens_per_sec:.2} tok/s)",
        elapsed.as_secs_f64()
    );

    Ok(())
}
