//! Transformer forward pass for ternary-quantized models.
//!
//! Implements RMSNorm, RoPE, GQA attention, and SwiGLU MLP using
//! packed ternary matmul for all projection layers.

use candle_core::{Device, Tensor, D};

use crate::config::{HiddenAct, ModelConfig};
use crate::error::{Error, Result};
use crate::kernels::ternary_matmul_fmt;
use crate::kv_cache::KVCache;
use crate::loader::{LayerWeights, LoadedWeights, PackedProjection};

/// Ternary transformer model ready for inference.
pub struct TernaryModel {
    config: ModelConfig,
    weights: LoadedWeights,
    cache: Vec<KVCache>,
    rope_cos: Tensor,
    rope_sin: Tensor,
    device: Device,
}

impl TernaryModel {
    pub fn new(config: ModelConfig, weights: LoadedWeights, device: &Device) -> Result<Self> {
        let head_dim = config.head_dim();
        let max_seq = config.max_position_embeddings;
        let (rope_cos, rope_sin) =
            compute_rope_tables(head_dim, max_seq, config.rope_theta, device)?;
        let cache = (0..config.num_layers).map(|_| KVCache::new()).collect();

        Ok(Self {
            config,
            weights,
            cache,
            rope_cos,
            rope_sin,
            device: device.clone(),
        })
    }

    /// Forward pass: token IDs → logits.
    ///
    /// `tokens` shape: `[batch, seq_len]` (u32)
    /// Returns logits: `[batch, seq_len, vocab_size]` (f32)
    pub fn forward(&mut self, tokens: &Tensor) -> Result<Tensor> {
        let (batch, seq_len) = tokens.dims2().map_err(Error::Tensor)?;

        // Embedding lookup: flatten IDs, index_select rows, reshape back
        let flat_ids = tokens.flatten_all().map_err(Error::Tensor)?;
        let mut x = self
            .weights
            .embed_tokens
            .index_select(&flat_ids, 0)
            .map_err(Error::Tensor)?;
        x = x
            .reshape((batch, seq_len, self.config.hidden_size))
            .map_err(Error::Tensor)?;

        let start_pos = self.cache[0].current_len();

        // Causal mask only needed during prefill (seq_len > 1)
        let mask = if seq_len > 1 {
            Some(create_causal_mask(seq_len, start_pos, &self.device)?)
        } else {
            None
        };

        // Transformer layers
        for i in 0..self.config.num_layers {
            x = forward_layer(
                &x,
                &self.weights.layers[i],
                &mut self.cache[i],
                &self.config,
                &self.rope_cos,
                &self.rope_sin,
                mask.as_ref(),
                start_pos,
                &self.device,
            )?;

            if seq_len > 1 && ((i + 1) % 16 == 0 || i + 1 == self.config.num_layers) {
                log::info!("Prefill layer {}/{}", i + 1, self.config.num_layers);
            }
        }

        // Final RMS norm
        x = rms_norm(&x, &self.weights.final_norm, self.config.rms_norm_eps)?;

        // LM head: [batch*seq, hidden] @ [hidden, vocab] → reshape to [batch, seq, vocab]
        let lm_head_t = self.weights.lm_head.t().map_err(Error::Tensor)?;
        let x_2d = x
            .reshape((batch * seq_len, self.config.hidden_size))
            .map_err(Error::Tensor)?;
        let logits_2d = x_2d.matmul(&lm_head_t).map_err(Error::Tensor)?;
        let vocab_size = logits_2d.dim(1).map_err(Error::Tensor)?;
        let logits = logits_2d
            .reshape((batch, seq_len, vocab_size))
            .map_err(Error::Tensor)?;

        Ok(logits)
    }

    /// Clear KV caches (call between different prompts).
    pub fn reset_cache(&mut self) {
        for c in &mut self.cache {
            c.reset();
        }
    }

    pub fn config(&self) -> &ModelConfig {
        &self.config
    }

    pub fn device(&self) -> &Device {
        &self.device
    }
}

// ─── Layer forward ───────────────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
fn forward_layer(
    x: &Tensor,
    layer: &LayerWeights,
    cache: &mut KVCache,
    config: &ModelConfig,
    rope_cos: &Tensor,
    rope_sin: &Tensor,
    mask: Option<&Tensor>,
    start_pos: usize,
    device: &Device,
) -> Result<Tensor> {
    // Pre-norm → attention → residual
    let normed = rms_norm(x, &layer.input_layernorm, config.rms_norm_eps)?;
    let attn_out =
        attention_block(&normed, layer, cache, config, rope_cos, rope_sin, mask, start_pos, device)?;
    let x = x.add(&attn_out).map_err(Error::Tensor)?;

    // Post-norm → MLP → residual
    let normed = rms_norm(&x, &layer.post_attention_layernorm, config.rms_norm_eps)?;
    let mlp_out = mlp_block(&normed, layer, config, device)?;
    x.add(&mlp_out).map_err(Error::Tensor)
}

// ─── Attention ───────────────────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
fn attention_block(
    x: &Tensor,
    layer: &LayerWeights,
    cache: &mut KVCache,
    config: &ModelConfig,
    rope_cos: &Tensor,
    rope_sin: &Tensor,
    mask: Option<&Tensor>,
    start_pos: usize,
    device: &Device,
) -> Result<Tensor> {
    let (batch, seq_len, _) = x.dims3().map_err(Error::Tensor)?;
    let head_dim = config.head_dim();
    let gs = config.group_size;

    // Q / K / V projections
    let mut q = project(x, &layer.packed_q, &layer.q_proj, gs, device)?;
    let mut k = project(x, &layer.packed_k, &layer.k_proj, gs, device)?;
    let mut v = project(x, &layer.packed_v, &layer.v_proj, gs, device)?;

    // Qwen2 biases
    if let Some(ref b) = layer.q_bias {
        q = q.broadcast_add(b).map_err(Error::Tensor)?;
    }
    if let Some(ref b) = layer.k_bias {
        k = k.broadcast_add(b).map_err(Error::Tensor)?;
    }
    if let Some(ref b) = layer.v_bias {
        v = v.broadcast_add(b).map_err(Error::Tensor)?;
    }

    // Reshape: [B, S, H*D] → [B, S, H, D]
    let q = q
        .reshape((batch, seq_len, config.num_heads, head_dim))
        .map_err(Error::Tensor)?;
    let k = k
        .reshape((batch, seq_len, config.num_kv_heads, head_dim))
        .map_err(Error::Tensor)?;
    let v = v
        .reshape((batch, seq_len, config.num_kv_heads, head_dim))
        .map_err(Error::Tensor)?;

    // RoPE on Q and K
    let (q, k) = apply_rope(&q, &k, rope_cos, rope_sin, start_pos, seq_len)?;

    // Transpose to [B, H, S, D] for batched matmul
    let q = q
        .transpose(1, 2)
        .map_err(Error::Tensor)?
        .contiguous()
        .map_err(Error::Tensor)?;
    let k = k
        .transpose(1, 2)
        .map_err(Error::Tensor)?
        .contiguous()
        .map_err(Error::Tensor)?;
    let v = v
        .transpose(1, 2)
        .map_err(Error::Tensor)?
        .contiguous()
        .map_err(Error::Tensor)?;

    // Append to KV cache → get full K, V over all positions
    let (k, v) = cache.append(&k, &v)?;

    // GQA: repeat KV heads to match Q heads
    let (k, v) = if config.num_kv_heads < config.num_heads {
        let n_rep = config.num_heads / config.num_kv_heads;
        (repeat_kv(&k, n_rep)?, repeat_kv(&v, n_rep)?)
    } else {
        (k, v)
    };

    // Scaled dot-product attention
    let scale = 1.0 / (head_dim as f64).sqrt();
    let attn = q
        .matmul(&k.t().map_err(Error::Tensor)?)
        .map_err(Error::Tensor)?;
    let attn = (attn * scale).map_err(Error::Tensor)?;

    // Causal mask (additive, -inf for masked positions)
    let attn = match mask {
        Some(m) => attn.broadcast_add(m).map_err(Error::Tensor)?,
        None => attn,
    };

    // Softmax over keys
    let attn = candle_nn::ops::softmax(&attn, D::Minus1).map_err(Error::Tensor)?;

    // Weighted sum of values
    let out = attn.matmul(&v).map_err(Error::Tensor)?;

    // [B, H, S, D] → [B, S, H*D]
    let out = out.transpose(1, 2).map_err(Error::Tensor)?;
    let out = out
        .contiguous()
        .map_err(Error::Tensor)?
        .reshape((batch, seq_len, config.num_heads * head_dim))
        .map_err(Error::Tensor)?;

    // BitNet attention sub-norm (before output projection)
    let out = if let Some(ref norm) = layer.attn_sub_norm {
        rms_norm(&out, norm, config.rms_norm_eps)?
    } else {
        out
    };

    // Output projection
    project(&out, &layer.packed_o, &layer.o_proj, gs, device)
}

// ─── MLP (SwiGLU) ────────────────────────────────────────────────────────────

fn mlp_block(
    x: &Tensor,
    layer: &LayerWeights,
    config: &ModelConfig,
    device: &Device,
) -> Result<Tensor> {
    let gs = config.group_size;

    let gate = project(x, &layer.packed_gate, &layer.gate_proj, gs, device)?;

    let gate = match config.hidden_act {
        HiddenAct::Silu => candle_nn::ops::silu(&gate).map_err(Error::Tensor)?,
        HiddenAct::Relu2 => {
            let relu = gate.relu().map_err(Error::Tensor)?;
            relu.sqr().map_err(Error::Tensor)?
        }
    };

    let up = project(x, &layer.packed_up, &layer.up_proj, gs, device)?;

    let hidden = gate.mul(&up).map_err(Error::Tensor)?;

    // BitNet FFN sub-norm (before down projection)
    let hidden = if let Some(ref norm) = layer.ffn_sub_norm {
        rms_norm(&hidden, norm, config.rms_norm_eps)?
    } else {
        hidden
    };

    project(&hidden, &layer.packed_down, &layer.down_proj, gs, device)
}

// ─── Projection dispatch ─────────────────────────────────────────────────────

fn project(
    input: &Tensor,
    packed: &Option<PackedProjection>,
    dequant: &Option<Tensor>,
    group_size: usize,
    device: &Device,
) -> Result<Tensor> {
    if let Some(ref p) = packed {
        let (batch, seq_len, _) = input.dims3().map_err(Error::Tensor)?;
        let flat = input
            .reshape((batch * seq_len, p.in_features))
            .map_err(Error::Tensor)?;
        let out = ternary_matmul_fmt(
            &flat,
            &p.packed,
            &p.scales,
            p.offsets.as_deref(),
            p.out_features,
            p.in_features,
            group_size,
            p.format,
            device,
        )?;
        out.reshape((batch, seq_len, p.out_features))
            .map_err(Error::Tensor)
    } else if let Some(ref w) = dequant {
        let (batch, seq_len, in_dim) = input.dims3().map_err(Error::Tensor)?;
        let flat = input
            .reshape((batch * seq_len, in_dim))
            .map_err(Error::Tensor)?;
        let out = flat
            .matmul(&w.t().map_err(Error::Tensor)?)
            .map_err(Error::Tensor)?;
        let out_dim = out.dim(1).map_err(Error::Tensor)?;
        out.reshape((batch, seq_len, out_dim))
            .map_err(Error::Tensor)
    } else {
        Err(Error::Loading("no weight for projection".into()))
    }
}

// ─── RMSNorm ─────────────────────────────────────────────────────────────────

fn rms_norm(x: &Tensor, weight: &Tensor, eps: f64) -> Result<Tensor> {
    let x_sq = x.sqr().map_err(Error::Tensor)?;
    let mean_sq = x_sq.mean_keepdim(D::Minus1).map_err(Error::Tensor)?;
    let rms = (mean_sq + eps)
        .map_err(Error::Tensor)?
        .sqrt()
        .map_err(Error::Tensor)?;
    let normalized = x.broadcast_div(&rms).map_err(Error::Tensor)?;
    normalized.broadcast_mul(weight).map_err(Error::Tensor)
}

// ─── Rotary Position Embeddings ──────────────────────────────────────────────

fn compute_rope_tables(
    head_dim: usize,
    max_seq: usize,
    theta: f32,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    let half_dim = head_dim / 2;
    let inv_freq: Vec<f32> = (0..half_dim)
        .map(|i| 1.0 / theta.powf(2.0 * i as f32 / head_dim as f32))
        .collect();

    let positions: Vec<f32> = (0..max_seq).map(|p| p as f32).collect();
    let inv_freq_t = Tensor::new(inv_freq.as_slice(), device).map_err(Error::Tensor)?;
    let positions_t = Tensor::new(positions.as_slice(), device).map_err(Error::Tensor)?;

    // Outer product: [max_seq, 1] @ [1, half_dim] → [max_seq, half_dim]
    let angles = positions_t
        .unsqueeze(1)
        .map_err(Error::Tensor)?
        .matmul(&inv_freq_t.unsqueeze(0).map_err(Error::Tensor)?)
        .map_err(Error::Tensor)?;

    // Duplicate to full head_dim: [max_seq, head_dim]
    let angles = Tensor::cat(&[&angles, &angles], 1).map_err(Error::Tensor)?;

    let cos = angles.cos().map_err(Error::Tensor)?;
    let sin = angles.sin().map_err(Error::Tensor)?;

    Ok((cos, sin))
}

fn apply_rope(
    q: &Tensor,
    k: &Tensor,
    cos_table: &Tensor,
    sin_table: &Tensor,
    start_pos: usize,
    seq_len: usize,
) -> Result<(Tensor, Tensor)> {
    // Slice: [seq_len, head_dim]
    let cos = cos_table
        .narrow(0, start_pos, seq_len)
        .map_err(Error::Tensor)?;
    let sin = sin_table
        .narrow(0, start_pos, seq_len)
        .map_err(Error::Tensor)?;

    // Reshape for broadcasting: [1, seq_len, 1, head_dim]
    let cos = cos
        .unsqueeze(0)
        .map_err(Error::Tensor)?
        .unsqueeze(2)
        .map_err(Error::Tensor)?;
    let sin = sin
        .unsqueeze(0)
        .map_err(Error::Tensor)?
        .unsqueeze(2)
        .map_err(Error::Tensor)?;

    // q/k: [batch, seq, heads, head_dim]
    let q_rot = rotate_half(q, &cos, &sin)?;
    let k_rot = rotate_half(k, &cos, &sin)?;

    Ok((q_rot, k_rot))
}

fn rotate_half(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let half = x.dim(D::Minus1).map_err(Error::Tensor)? / 2;
    let x1 = x.narrow(D::Minus1, 0, half).map_err(Error::Tensor)?;
    let x2 = x.narrow(D::Minus1, half, half).map_err(Error::Tensor)?;

    // rotated = [-x2, x1]
    let neg_x2 = x2.neg().map_err(Error::Tensor)?;
    let rotated = Tensor::cat(&[&neg_x2, &x1], D::Minus1).map_err(Error::Tensor)?;

    // x * cos + rotated * sin
    let a = x.broadcast_mul(cos).map_err(Error::Tensor)?;
    let b = rotated.broadcast_mul(sin).map_err(Error::Tensor)?;
    a.add(&b).map_err(Error::Tensor)
}

// ─── Attention helpers ───────────────────────────────────────────────────────

fn create_causal_mask(seq_len: usize, start_pos: usize, device: &Device) -> Result<Tensor> {
    let total_len = start_pos + seq_len;
    let mut mask = vec![0.0f32; seq_len * total_len];
    for i in 0..seq_len {
        for j in 0..total_len {
            if j > i + start_pos {
                mask[i * total_len + j] = f32::NEG_INFINITY;
            }
        }
    }
    // [1, 1, seq_len, total_len] broadcasts over batch and heads
    Tensor::from_vec(mask, (1, 1, seq_len, total_len), device).map_err(Error::Tensor)
}

fn repeat_kv(x: &Tensor, n_rep: usize) -> Result<Tensor> {
    if n_rep == 1 {
        return Ok(x.clone());
    }
    let (b, n_kv, s, d) = x.dims4().map_err(Error::Tensor)?;
    x.unsqueeze(2)
        .map_err(Error::Tensor)?
        .broadcast_as((b, n_kv, n_rep, s, d))
        .map_err(Error::Tensor)?
        .contiguous()
        .map_err(Error::Tensor)?
        .reshape((b, n_kv * n_rep, s, d))
        .map_err(Error::Tensor)
}
