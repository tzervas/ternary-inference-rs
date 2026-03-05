//! Text generation with greedy and temperature sampling.

use candle_core::{IndexOp, Tensor, D};

use crate::error::{Error, Result};
use crate::model::TernaryModel;

/// Generation parameters.
pub struct GenerationConfig {
    /// Maximum number of tokens to generate.
    pub max_tokens: usize,
    /// Sampling temperature (0.0 = greedy, >0 = stochastic).
    pub temperature: f32,
    /// Top-p (nucleus) sampling threshold. 1.0 = disabled.
    pub top_p: f32,
    /// Token ID for end-of-sequence. Generation stops when emitted.
    pub eos_token_id: u32,
    /// Whether to print tokens as they are generated.
    pub stream: bool,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_tokens: 256,
            temperature: 0.0,
            top_p: 1.0,
            eos_token_id: 151643, // Qwen2 <|endoftext|>
            stream: true,
        }
    }
}

/// Generate text from a prompt.
///
/// Returns the generated text (excluding the prompt).
pub fn generate(
    model: &mut TernaryModel,
    tokenizer: &tokenizers::Tokenizer,
    prompt: &str,
    config: &GenerationConfig,
) -> Result<String> {
    model.reset_cache();

    let device = model.device().clone();

    // Encode prompt
    let encoding = tokenizer
        .encode(prompt, true)
        .map_err(|e| Error::Config(format!("tokenizer error: {e}")))?;
    let prompt_ids: Vec<u32> = encoding.get_ids().to_vec();
    let prompt_len = prompt_ids.len();
    log::info!("Prompt tokens: {prompt_len}");

    // Prefill: process all prompt tokens at once
    let input = Tensor::new(prompt_ids.as_slice(), &device)
        .map_err(Error::Tensor)?
        .unsqueeze(0)
        .map_err(Error::Tensor)?;
    let logits = model.forward(&input)?;

    // Sample first generated token from last position's logits
    let last_logits = logits
        .i((0, prompt_len - 1))
        .map_err(Error::Tensor)?;
    let mut next_token = sample_token(&last_logits, config)?;

    let mut generated_ids = vec![next_token];

    if config.stream {
        if let Some(text) = decode_token(tokenizer, next_token) {
            eprint!("{text}");
        }
    }

    // Autoregressive generation: one token at a time
    for _ in 1..config.max_tokens {
        if next_token == config.eos_token_id {
            break;
        }

        let input = Tensor::new(&[next_token], &device)
            .map_err(Error::Tensor)?
            .unsqueeze(0)
            .map_err(Error::Tensor)?;
        let logits = model.forward(&input)?;

        let token_logits = logits.i((0, 0)).map_err(Error::Tensor)?;
        next_token = sample_token(&token_logits, config)?;
        generated_ids.push(next_token);

        if config.stream {
            if let Some(text) = decode_token(tokenizer, next_token) {
                eprint!("{text}");
            }
        }
    }

    if config.stream {
        eprintln!();
    }

    // Decode all generated tokens
    let output = tokenizer
        .decode(&generated_ids, true)
        .map_err(|e| Error::Config(format!("tokenizer decode error: {e}")))?;

    Ok(output)
}

/// Load tokenizer from model directory or HF cache.
pub fn load_tokenizer(model_dir: &std::path::Path) -> Result<tokenizers::Tokenizer> {
    let tokenizer_path = model_dir.join("tokenizer.json");
    if !tokenizer_path.exists() {
        return Err(Error::Loading(format!(
            "tokenizer.json not found at {}",
            tokenizer_path.display()
        )));
    }
    tokenizers::Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| Error::Loading(format!("failed to load tokenizer: {e}")))
}

/// Load tokenizer from HF cache for a given model ID.
pub fn load_tokenizer_from_hf_cache(model_id: &str) -> Result<tokenizers::Tokenizer> {
    let cache_dir = dirs_next::home_dir()
        .ok_or_else(|| Error::Loading("cannot find home directory".into()))?
        .join(".cache/huggingface/hub")
        .join(format!("models--{}", model_id.replace('/', "--")));

    let snapshots_dir = cache_dir.join("snapshots");
    if !snapshots_dir.exists() {
        return Err(Error::Loading(format!(
            "Model not cached. Run: huggingface-cli download {model_id}"
        )));
    }

    let snapshot = std::fs::read_dir(&snapshots_dir)
        .map_err(Error::Io)?
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().map(|ft| ft.is_dir()).unwrap_or(false))
        .map(|e| e.path())
        .next()
        .ok_or_else(|| Error::Loading("no snapshots found in HF cache".into()))?;

    load_tokenizer(&snapshot)
}

// ─── Sampling ────────────────────────────────────────────────────────────────

fn sample_token(logits: &Tensor, config: &GenerationConfig) -> Result<u32> {
    if config.temperature <= 0.0 {
        // Greedy: argmax
        let token = logits
            .argmax(D::Minus1)
            .map_err(Error::Tensor)?
            .to_scalar::<u32>()
            .map_err(Error::Tensor)?;
        Ok(token)
    } else {
        // Temperature sampling
        let scaled = (logits / config.temperature as f64).map_err(Error::Tensor)?;
        let probs = candle_nn::ops::softmax(&scaled, D::Minus1).map_err(Error::Tensor)?;
        let probs_vec: Vec<f32> = probs.to_vec1().map_err(Error::Tensor)?;

        // Top-p filtering
        let token = if config.top_p < 1.0 {
            sample_top_p(&probs_vec, config.top_p)
        } else {
            sample_categorical(&probs_vec)
        };

        Ok(token)
    }
}

fn sample_categorical(probs: &[f32]) -> u32 {
    let r: f32 = rand::random();
    let mut cumsum = 0.0;
    for (i, &p) in probs.iter().enumerate() {
        cumsum += p;
        if cumsum >= r {
            return i as u32;
        }
    }
    (probs.len() - 1) as u32
}

fn sample_top_p(probs: &[f32], top_p: f32) -> u32 {
    let mut indexed: Vec<(usize, f32)> = probs.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut cumsum = 0.0;
    let mut cutoff_idx = indexed.len();
    for (i, &(_, p)) in indexed.iter().enumerate() {
        cumsum += p;
        if cumsum >= top_p {
            cutoff_idx = i + 1;
            break;
        }
    }

    // Renormalize
    let kept = &indexed[..cutoff_idx];
    let total: f32 = kept.iter().map(|(_, p)| p).sum();
    let r: f32 = rand::random::<f32>() * total;

    let mut cumsum = 0.0;
    for &(idx, p) in kept {
        cumsum += p;
        if cumsum >= r {
            return idx as u32;
        }
    }

    kept.last().map(|&(idx, _)| idx as u32).unwrap_or(0)
}

fn decode_token(tokenizer: &tokenizers::Tokenizer, token_id: u32) -> Option<String> {
    tokenizer.decode(&[token_id], false).ok()
}
