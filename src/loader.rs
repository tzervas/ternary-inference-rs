//! Weight loading from SafeTensors (packed ternary format).
//!
//! Loads the custom packed ternary SafeTensors format where:
//! - `*.weight` tensors are uint8 packed (4 ternary values per byte)
//! - `*.weight.scales` tensors are float32 per-group scale factors
//! - Norm weights and biases are standard float32

use std::collections::HashMap;
use std::path::Path;

use candle_core::{DType, Device, Tensor};

use crate::config::{Architecture, ModelConfig};
use crate::error::{Error, Result};
use crate::unpack::{PackingFormat, dequantize_weight, dequantize_weight_2bit};

/// Loaded model weights, ready for inference.
pub struct LoadedWeights {
    /// Token embedding: [vocab_size, hidden_size]
    pub embed_tokens: Tensor,
    /// Per-layer weights
    pub layers: Vec<LayerWeights>,
    /// Final RMS norm weight: [hidden_size]
    pub final_norm: Tensor,
    /// LM head weight: [vocab_size, hidden_size]
    pub lm_head: Tensor,
}

/// Packed ternary weights for a single projection (kept on CPU).
///
/// For direct ternary matmul (no dequantization needed at inference time).
pub struct PackedProjection {
    /// Raw packed bytes
    pub packed: Vec<u8>,
    /// Scale factors: per-group [out_features * num_groups] or per-tensor [1]
    pub scales: Vec<f32>,
    /// Per-row offsets (PT2-LLM asymmetric quantization): [out_features]
    /// When present: W_hat = alpha * T + mu (instead of just alpha * T)
    pub offsets: Option<Vec<f32>>,
    /// Output dimension
    pub out_features: usize,
    /// Input dimension
    pub in_features: usize,
    /// Packing format
    pub format: PackingFormat,
}

/// Weights for a single transformer layer.
///
/// Projection weights can be either:
/// - Packed ternary (`packed_*`) for direct ternary matmul (preferred)
/// - Dequantized float (`*_proj`) for standard matmul (fallback)
pub struct LayerWeights {
    // ── Packed ternary projections (preferred path) ──
    /// Q projection packed ternary
    pub packed_q: Option<PackedProjection>,
    /// K projection packed ternary
    pub packed_k: Option<PackedProjection>,
    /// V projection packed ternary
    pub packed_v: Option<PackedProjection>,
    /// O projection packed ternary
    pub packed_o: Option<PackedProjection>,
    /// Gate projection packed ternary
    pub packed_gate: Option<PackedProjection>,
    /// Up projection packed ternary
    pub packed_up: Option<PackedProjection>,
    /// Down projection packed ternary
    pub packed_down: Option<PackedProjection>,

    // ── Dequantized float projections (fallback) ──
    /// Q projection: [num_heads * head_dim, hidden_size]
    pub q_proj: Option<Tensor>,
    /// K projection: [num_kv_heads * head_dim, hidden_size]
    pub k_proj: Option<Tensor>,
    /// V projection: [num_kv_heads * head_dim, hidden_size]
    pub v_proj: Option<Tensor>,
    /// Output projection: [hidden_size, num_heads * head_dim]
    pub o_proj: Option<Tensor>,
    /// Gate projection (SwiGLU): [intermediate_size, hidden_size]
    pub gate_proj: Option<Tensor>,
    /// Up projection (SwiGLU): [intermediate_size, hidden_size]
    pub up_proj: Option<Tensor>,
    /// Down projection: [hidden_size, intermediate_size]
    pub down_proj: Option<Tensor>,

    // ── Always float ──
    /// Input layer norm: [hidden_size]
    pub input_layernorm: Tensor,
    /// Post-attention layer norm: [hidden_size]
    pub post_attention_layernorm: Tensor,
    /// Q bias (Qwen2 only)
    pub q_bias: Option<Tensor>,
    /// K bias (Qwen2 only)
    pub k_bias: Option<Tensor>,
    /// V bias (Qwen2 only)
    pub v_bias: Option<Tensor>,
    /// BitNet attention sub-norm (applied after attention, before residual)
    pub attn_sub_norm: Option<Tensor>,
    /// BitNet FFN sub-norm (applied after MLP gate+up, before down projection)
    pub ffn_sub_norm: Option<Tensor>,
}

/// How to store projection weights after loading.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoadMode {
    /// Keep packed ternary bytes + scales (for direct ternary matmul).
    Packed,
    /// Dequantize to float tensors (for standard matmul).
    Dequantized,
}

/// Load model weights from a directory containing SafeTensors shards.
///
/// Handles the custom packed ternary format:
/// - uint8 weight tensors are paired with `.scales` tensors
/// - float32 norms/biases are loaded directly
///
/// `mode` controls whether projections are kept as packed ternary
/// (for direct ternary matmul) or dequantized to float.
pub fn load_from_dir(
    dir: &Path,
    config: &ModelConfig,
    device: &Device,
    mode: LoadMode,
) -> Result<LoadedWeights> {
    // Find safetensors files
    let mut shard_paths: Vec<_> = std::fs::read_dir(dir)
        .map_err(Error::Io)?
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.path()
                .extension()
                .is_some_and(|ext| ext == "safetensors")
        })
        .map(|e| e.path())
        .collect();
    shard_paths.sort();

    if shard_paths.is_empty() {
        return Err(Error::Loading(format!(
            "No .safetensors files found in {}",
            dir.display()
        )));
    }

    // Load all tensors from all shards into flat maps.
    // We keep shard data alive since SafeTensors borrows from it.
    let mut packed_tensors: HashMap<String, (Vec<u8>, Vec<usize>)> = HashMap::new();
    let mut float_tensors: HashMap<String, Tensor> = HashMap::new();

    for shard_path in &shard_paths {
        log::info!("Loading shard: {}", shard_path.display());
        let shard_data = std::fs::read(shard_path).map_err(Error::Io)?;
        let st = safetensors::tensor::SafeTensors::deserialize(&shard_data)
            .map_err(|e| Error::SafeTensors(e.to_string()))?;

        for (name, view) in st.tensors() {
            match view.dtype() {
                safetensors::Dtype::U8 => {
                    packed_tensors.insert(
                        name.to_string(),
                        (view.data().to_vec(), view.shape().to_vec()),
                    );
                }
                safetensors::Dtype::F32 => {
                    let t = Tensor::from_raw_buffer(
                        view.data(),
                        DType::F32,
                        view.shape(),
                        device,
                    )?;
                    float_tensors.insert(name.to_string(), t);
                }
                safetensors::Dtype::F16 => {
                    let t = Tensor::from_raw_buffer(
                        view.data(),
                        DType::F16,
                        view.shape(),
                        device,
                    )?
                    .to_dtype(DType::F32)?;
                    float_tensors.insert(name.to_string(), t);
                }
                safetensors::Dtype::BF16 => {
                    let t = Tensor::from_raw_buffer(
                        view.data(),
                        DType::BF16,
                        view.shape(),
                        device,
                    )?
                    .to_dtype(DType::F32)?;
                    float_tensors.insert(name.to_string(), t);
                }
                other => {
                    log::warn!("Skipping tensor {name} with unsupported dtype {other:?}");
                }
            }
        }
    }

    let group_size = config.group_size;
    let is_bitnet = config.architecture == Architecture::BitNet;

    // ── Extract global tensors ──
    // embed_tokens and lm_head may be packed ternary or float
    let embed_tokens = if let Some(t) = float_tensors.remove("model.embed_tokens.weight") {
        t
    } else if packed_tensors.contains_key("model.embed_tokens.weight") {
        log::info!("Dequantizing packed embed_tokens...");
        dequant_packed_tensor(
            &packed_tensors,
            &float_tensors,
            "model.embed_tokens.weight",
            group_size,
            is_bitnet,
            device,
        )?
    } else {
        return Err(Error::Loading("missing model.embed_tokens.weight".into()));
    };

    let final_norm = float_tensors
        .remove("model.norm.weight")
        .ok_or_else(|| Error::Loading("missing model.norm.weight".into()))?;

    let lm_head = if let Some(t) = float_tensors.remove("lm_head.weight") {
        t
    } else if packed_tensors.contains_key("lm_head.weight") {
        log::info!("Dequantizing packed lm_head...");
        dequant_packed_tensor(
            &packed_tensors,
            &float_tensors,
            "lm_head.weight",
            group_size,
            is_bitnet,
            device,
        )?
    } else {
        embed_tokens.clone() // tied embeddings fallback
    };

    // ── Build per-layer weights ──
    let mut layers = Vec::with_capacity(config.num_layers);

    for i in 0..config.num_layers {
        let pfx = format!("model.layers.{i}");

        // Helper: load a packed ternary projection
        let load_proj = |proj_key: &str,
                         packed_tensors: &HashMap<String, (Vec<u8>, Vec<usize>)>,
                         float_tensors: &mut HashMap<String, Tensor>|
         -> Result<(Option<PackedProjection>, Option<Tensor>)> {
            let weight_key = format!("{pfx}.{proj_key}.weight");
            // BitNet uses "weight_scale", Qwen uses "weight.scales"
            let scale_key = if is_bitnet {
                format!("{pfx}.{proj_key}.weight_scale")
            } else {
                format!("{pfx}.{proj_key}.weight.scales")
            };

            if let Some((packed_bytes, shape)) = packed_tensors.get(&weight_key) {
                // It's packed ternary
                if shape.len() != 2 {
                    return Err(Error::Shape {
                        expected: "2D packed tensor".into(),
                        actual: format!("{shape:?}"),
                    });
                }

                let (out_features, in_features, packing_format) = if is_bitnet {
                    // 2-bit row-packed: [out/4, in] → out = shape[0]*4, in = shape[1]
                    (shape[0] * 4, shape[1], PackingFormat::TwoBitRows)
                } else {
                    // Base-3 column-packed: [out, in/4] → out = shape[0], in = shape[1]*4
                    (shape[0], shape[1] * 4, PackingFormat::Base3Columns)
                };

                // Get scales
                let scales = if let Some(scales_tensor) = float_tensors.get(&scale_key) {
                    scales_tensor
                        .flatten_all()?
                        .to_vec1()
                        .map_err(Error::Tensor)?
                } else {
                    return Err(Error::Loading(format!("missing scales for {weight_key}")));
                };

                // Get offsets (PT2-LLM asymmetric quantization, optional)
                let offset_key = format!("{pfx}.{proj_key}.weight.offsets");
                let offsets: Option<Vec<f32>> = float_tensors.get(&offset_key).map(|t| {
                    t.flatten_all()
                        .and_then(|t| t.to_vec1())
                        .unwrap_or_default()
                });

                match mode {
                    LoadMode::Packed => Ok((
                        Some(PackedProjection {
                            packed: packed_bytes.clone(),
                            scales,
                            offsets,
                            out_features,
                            in_features,
                            format: packing_format,
                        }),
                        None,
                    )),
                    LoadMode::Dequantized => {
                        let dequant = match packing_format {
                            PackingFormat::Base3Columns => dequantize_weight(
                                packed_bytes,
                                &scales,
                                out_features,
                                in_features,
                                group_size,
                                device,
                            )?,
                            PackingFormat::TwoBitRows => dequantize_weight_2bit(
                                packed_bytes,
                                scales[0],
                                out_features,
                                in_features,
                                device,
                            )?,
                        };
                        Ok((None, Some(dequant)))
                    }
                }
            } else if let Some(tensor) = float_tensors.remove(&weight_key) {
                // Already float (not packed)
                Ok((None, Some(tensor)))
            } else {
                Err(Error::Loading(format!("missing weight: {weight_key}")))
            }
        };

        let (packed_q, q_proj) = load_proj("self_attn.q_proj", &packed_tensors, &mut float_tensors)?;
        let (packed_k, k_proj) = load_proj("self_attn.k_proj", &packed_tensors, &mut float_tensors)?;
        let (packed_v, v_proj) = load_proj("self_attn.v_proj", &packed_tensors, &mut float_tensors)?;
        let (packed_o, o_proj) = load_proj("self_attn.o_proj", &packed_tensors, &mut float_tensors)?;
        let (packed_gate, gate_proj) = load_proj("mlp.gate_proj", &packed_tensors, &mut float_tensors)?;
        let (packed_up, up_proj) = load_proj("mlp.up_proj", &packed_tensors, &mut float_tensors)?;
        let (packed_down, down_proj) = load_proj("mlp.down_proj", &packed_tensors, &mut float_tensors)?;

        let input_ln_key = format!("{pfx}.input_layernorm.weight");
        let post_ln_key = format!("{pfx}.post_attention_layernorm.weight");

        let input_layernorm = float_tensors
            .remove(&input_ln_key)
            .ok_or_else(|| Error::Loading(format!("missing {input_ln_key}")))?;
        let post_attention_layernorm = float_tensors
            .remove(&post_ln_key)
            .ok_or_else(|| Error::Loading(format!("missing {post_ln_key}")))?;

        // Optional Q/K/V biases (Qwen2)
        let q_bias = float_tensors.remove(&format!("{pfx}.self_attn.q_proj.bias"));
        let k_bias = float_tensors.remove(&format!("{pfx}.self_attn.k_proj.bias"));
        let v_bias = float_tensors.remove(&format!("{pfx}.self_attn.v_proj.bias"));

        // Optional BitNet sub-norms
        let attn_sub_norm = float_tensors.remove(&format!("{pfx}.self_attn.attn_sub_norm.weight"))
            .or_else(|| float_tensors.remove(&format!("{pfx}.self_attn.inner_attn_ln.weight")));
        let ffn_sub_norm = float_tensors.remove(&format!("{pfx}.mlp.ffn_sub_norm.weight"))
            .or_else(|| float_tensors.remove(&format!("{pfx}.mlp.ffn_layernorm.weight")));

        layers.push(LayerWeights {
            packed_q,
            packed_k,
            packed_v,
            packed_o,
            packed_gate,
            packed_up,
            packed_down,
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            gate_proj,
            up_proj,
            down_proj,
            input_layernorm,
            post_attention_layernorm,
            q_bias,
            k_bias,
            v_bias,
            attn_sub_norm,
            ffn_sub_norm,
        });

        if (i + 1) % 16 == 0 || i + 1 == config.num_layers {
            log::info!("Loaded layer {}/{}", i + 1, config.num_layers);
        }
    }

    Ok(LoadedWeights {
        embed_tokens,
        layers,
        final_norm,
        lm_head,
    })
}

/// Load model weights from a HuggingFace Hub cache directory.
///
/// Looks for the standard HF cache layout with `model.safetensors.index.json`.
pub fn load_from_hf_cache(
    model_id: &str,
    config: &ModelConfig,
    device: &Device,
    mode: LoadMode,
) -> Result<LoadedWeights> {
    let cache_dir = dirs_next::home_dir()
        .ok_or_else(|| Error::Loading("cannot find home directory".into()))?
        .join(".cache/huggingface/hub")
        .join(format!("models--{}", model_id.replace('/', "--")));

    // Find latest snapshot
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

    load_from_dir(&snapshot, config, device, mode)
}

/// Dequantize a packed ternary tensor from the tensor maps.
fn dequant_packed_tensor(
    packed_tensors: &HashMap<String, (Vec<u8>, Vec<usize>)>,
    float_tensors: &HashMap<String, Tensor>,
    key: &str,
    group_size: usize,
    is_2bit: bool,
    device: &Device,
) -> Result<Tensor> {
    let (packed_bytes, shape) = packed_tensors
        .get(key)
        .ok_or_else(|| Error::Loading(format!("missing packed tensor: {key}")))?;

    if is_2bit {
        let scale_key = format!("{key}_scale");
        let scales_tensor = float_tensors
            .get(&scale_key)
            .ok_or_else(|| Error::Loading(format!("missing scales for {key}")))?;
        let scale: f32 = scales_tensor.flatten_all()?.to_vec1::<f32>().map_err(Error::Tensor)?[0];
        let out_features = shape[0] * 4;
        let in_features = shape[1];
        dequantize_weight_2bit(packed_bytes, scale, out_features, in_features, device)
    } else {
        let scale_key = format!("{key}.scales");
        let scales_tensor = float_tensors
            .get(&scale_key)
            .ok_or_else(|| Error::Loading(format!("missing scales for {key}")))?;
        let scales: Vec<f32> = scales_tensor
            .flatten_all()?
            .to_vec1()
            .map_err(Error::Tensor)?;
        let out_features = shape[0];
        let in_features = shape[1] * 4;
        dequantize_weight(packed_bytes, &scales, out_features, in_features, group_size, device)
    }
}
