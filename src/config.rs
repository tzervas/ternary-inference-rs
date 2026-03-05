//! Model configuration for supported architectures.

use serde::{Deserialize, Serialize};

/// Supported transformer architectures.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Architecture {
    /// Qwen2 / Qwen2.5 family (GQA, RoPE, SwiGLU, RMSNorm).
    Qwen2,
    /// LLaMA / Mistral family (GQA, RoPE, SwiGLU, RMSNorm).
    Llama,
    /// Microsoft BitNet native architecture.
    BitNet,
}

/// Hidden activation function.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HiddenAct {
    /// SiLU (SwiGLU MLP) — used by Qwen2, LLaMA
    Silu,
    /// Squared ReLU — used by BitNet
    Relu2,
}

/// Configuration for a ternary-quantized transformer model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Architecture variant.
    pub architecture: Architecture,
    /// Model hidden / embedding dimension.
    pub hidden_size: usize,
    /// Number of transformer layers.
    pub num_layers: usize,
    /// Number of query attention heads.
    pub num_heads: usize,
    /// Number of key/value heads (for GQA).
    pub num_kv_heads: usize,
    /// MLP intermediate dimension.
    pub intermediate_size: usize,
    /// Vocabulary size.
    pub vocab_size: usize,
    /// Maximum sequence length.
    pub max_position_embeddings: usize,
    /// RMS norm epsilon.
    pub rms_norm_eps: f64,
    /// RoPE base frequency.
    pub rope_theta: f32,
    /// Ternary quantization group size (weights per scale factor). 0 = per-tensor scale.
    pub group_size: usize,
    /// HuggingFace repo ID for weight download.
    pub hf_repo_id: Option<String>,
    /// Activation function for MLP.
    pub hidden_act: HiddenAct,
    /// Whether embeddings and lm_head share weights.
    pub tie_word_embeddings: bool,
    /// Whether attention uses Q/K/V biases.
    pub attention_bias: bool,
    /// Whether BitNet-style sub-norms (attn_sub_norm, ffn_sub_norm) are used.
    pub has_sub_norms: bool,
}

impl ModelConfig {
    /// Per-head dimension.
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_heads
    }

    /// GQA group factor (Q heads per KV head).
    pub fn gqa_groups(&self) -> usize {
        self.num_heads / self.num_kv_heads
    }

    /// Qwen2.5-Coder-32B-Instruct (BitNet quantized).
    pub fn qwen2_32b() -> Self {
        Self {
            architecture: Architecture::Qwen2,
            hidden_size: 5120,
            num_layers: 64,
            num_heads: 40,
            num_kv_heads: 8,
            intermediate_size: 27648,
            vocab_size: 152064,
            max_position_embeddings: 32768,
            rms_norm_eps: 1e-6,
            rope_theta: 1_000_000.0,
            group_size: 64,
            hf_repo_id: Some("tzervas/qwen2.5-coder-32b-bitnet-1.58b".to_string()),
            hidden_act: HiddenAct::Silu,
            tie_word_embeddings: false,
            attention_bias: true,
            has_sub_norms: false,
        }
    }

    /// Microsoft BitNet b1.58 2B-4T.
    pub fn bitnet_2b() -> Self {
        Self {
            architecture: Architecture::BitNet,
            hidden_size: 2560,
            num_layers: 30,
            num_heads: 20,
            num_kv_heads: 5,
            intermediate_size: 6912,
            vocab_size: 128256,
            max_position_embeddings: 4096,
            rms_norm_eps: 1e-5,
            rope_theta: 500_000.0,
            group_size: 0, // per-tensor scale
            hf_repo_id: Some("microsoft/bitnet-b1.58-2B-4T".to_string()),
            hidden_act: HiddenAct::Relu2,
            tie_word_embeddings: true,
            attention_bias: false,
            has_sub_norms: true,
        }
    }

    /// Load config from a HuggingFace-style config.json.
    pub fn from_hf_config(json: &serde_json::Value) -> crate::Result<Self> {
        let arch_str = json["architectures"][0]
            .as_str()
            .unwrap_or("unknown");

        let architecture = match arch_str {
            "Qwen2ForCausalLM" => Architecture::Qwen2,
            "LlamaForCausalLM" | "MistralForCausalLM" => Architecture::Llama,
            "BitnetForCausalLM" => Architecture::BitNet,
            other => return Err(crate::Error::UnsupportedArch(other.to_string())),
        };

        let hidden_act = match json["hidden_act"].as_str().unwrap_or("silu") {
            "relu2" => HiddenAct::Relu2,
            _ => HiddenAct::Silu,
        };

        let is_bitnet = architecture == Architecture::BitNet;

        Ok(Self {
            architecture,
            hidden_size: json["hidden_size"].as_u64().unwrap_or(0) as usize,
            num_layers: json["num_hidden_layers"].as_u64().unwrap_or(0) as usize,
            num_heads: json["num_attention_heads"].as_u64().unwrap_or(0) as usize,
            num_kv_heads: json["num_key_value_heads"].as_u64().unwrap_or(0) as usize,
            intermediate_size: json["intermediate_size"].as_u64().unwrap_or(0) as usize,
            vocab_size: json["vocab_size"].as_u64().unwrap_or(0) as usize,
            max_position_embeddings: json["max_position_embeddings"]
                .as_u64()
                .unwrap_or(32768) as usize,
            rms_norm_eps: json["rms_norm_eps"].as_f64().unwrap_or(1e-6),
            rope_theta: json["rope_theta"].as_f64().unwrap_or(10000.0) as f32,
            group_size: if is_bitnet { 0 } else { 64 },
            hf_repo_id: None,
            hidden_act,
            tie_word_embeddings: json["tie_word_embeddings"].as_bool().unwrap_or(false),
            attention_bias: json["attention_bias"].as_bool().unwrap_or(!is_bitnet),
            has_sub_norms: is_bitnet,
        })
    }
}
