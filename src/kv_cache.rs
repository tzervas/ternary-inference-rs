//! KV cache for autoregressive transformer generation.

use candle_core::Tensor;

use crate::error::{Error, Result};

/// Per-layer KV cache that accumulates key/value tensors across generation steps.
#[derive(Default)]
pub struct KVCache {
    k: Option<Tensor>,
    v: Option<Tensor>,
}

impl KVCache {
    pub fn new() -> Self {
        Self::default()
    }

    /// Append new K/V tensors and return the full cached (K, V).
    ///
    /// Input shapes: [batch, num_kv_heads, new_seq_len, head_dim]
    /// Output shapes: [batch, num_kv_heads, total_seq_len, head_dim]
    pub fn append(&mut self, k_new: &Tensor, v_new: &Tensor) -> Result<(Tensor, Tensor)> {
        let (k, v) = match (&self.k, &self.v) {
            (Some(k_old), Some(v_old)) => {
                let k = Tensor::cat(&[k_old, k_new], 2).map_err(Error::Tensor)?;
                let v = Tensor::cat(&[v_old, v_new], 2).map_err(Error::Tensor)?;
                (k, v)
            }
            _ => (k_new.clone(), v_new.clone()),
        };
        self.k = Some(k.clone());
        self.v = Some(v.clone());
        Ok((k, v))
    }

    /// Number of tokens currently cached.
    pub fn current_len(&self) -> usize {
        self.k
            .as_ref()
            .and_then(|k| k.dim(2).ok())
            .unwrap_or(0)
    }

    /// Clear the cache (e.g., for a new prompt).
    pub fn reset(&mut self) {
        self.k = None;
        self.v = None;
    }
}
