//! CPU ternary matmul using direct packed-weight computation.
//!
//! Avoids full dequantization by computing dot products directly from
//! packed ternary bytes. For each output element:
//!
//! ```text
//! y[i,j] = sum_g (scale[j,g] * sum_k ternary[j, g*gs+k] * x[i, g*gs+k])
//! ```

use candle_core::{Device, Tensor};

use crate::error::{Error, Result};
use crate::unpack::unpack_byte;

/// CPU ternary matrix multiplication without full dequantization.
///
/// Computes output = input @ weight^T where weight is packed ternary.
/// Supports optional per-row offsets for asymmetric quantization (PT2-LLM).
///
/// Without offsets: y[b,j] = sum_g (scale[j,g] * sum_k T[j,g*gs+k] * x[b,g*gs+k])
/// With offsets:    y[b,j] = sum_g (scale[j,g] * sum_k T[j,g*gs+k] * x[b,g*gs+k]) + offset[j] * sum(x[b,:])
///
/// # Arguments
/// * `input` - Float tensor of shape [batch, in_features]
/// * `packed_weight` - Packed bytes [out_features, in_features/4]
/// * `scales` - Per-group scales [out_features, num_groups]
/// * `offsets` - Optional per-row offsets [out_features] (PT2-LLM asymmetric)
/// * `out_features` - Output dimension
/// * `in_features` - Input dimension
/// * `group_size` - Weights per scale group
pub fn ternary_matmul_cpu(
    input: &Tensor,
    packed_weight: &[u8],
    scales: &[f32],
    offsets: Option<&[f32]>,
    out_features: usize,
    in_features: usize,
    group_size: usize,
) -> Result<Tensor> {
    let input_data: Vec<f32> = input
        .flatten_all()?
        .to_vec1()
        .map_err(Error::Tensor)?;

    let batch_size = input.dim(0)?;
    let num_groups = in_features / group_size;
    let packed_cols = in_features / 4;

    let mut output = vec![0.0f32; batch_size * out_features];

    for b in 0..batch_size {
        let x = &input_data[b * in_features..(b + 1) * in_features];

        // Pre-compute sum(x) for offset term (only if offsets present)
        let x_sum = offsets.map(|_| x.iter().sum::<f32>());

        for j in 0..out_features {
            let mut sum = 0.0f32;

            for g in 0..num_groups {
                let scale = scales[j * num_groups + g];
                let col_start = g * group_size;
                let packed_start = j * packed_cols + col_start / 4;

                let mut group_sum = 0.0f32;
                for k_packed in 0..(group_size / 4) {
                    let byte = packed_weight[packed_start + k_packed];
                    let vals = unpack_byte(byte);
                    let col = col_start + k_packed * 4;

                    for (di, &v) in vals.iter().enumerate() {
                        match v {
                            1 => group_sum += x[col + di],
                            -1 => group_sum -= x[col + di],
                            _ => {}
                        }
                    }
                }

                sum += scale * group_sum;
            }

            // Add offset contribution: offset[j] * sum(x)
            if let (Some(offs), Some(xs)) = (offsets, x_sum) {
                sum += offs[j] * xs;
            }

            output[b * out_features + j] = sum;
        }
    }

    Tensor::from_vec(output, (batch_size, out_features), &Device::Cpu)
        .map_err(Error::Tensor)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ternary_matmul_identity_scale() {
        // 1x4 input, 2x4 weight (1 group of 4)
        let input = Tensor::new(&[[1.0f32, 2.0, 3.0, 4.0]], &Device::Cpu).unwrap();

        // Row 0: [1, -1, 0, 1] -> packed: (1+1) + ((-1)+1)*3 + (0+1)*9 + (1+1)*27 = 2 + 0 + 9 + 54 = 65
        // Row 1: [0, 1, 1, -1] -> packed: (0+1) + (1+1)*3 + (1+1)*9 + ((-1)+1)*27 = 1 + 6 + 18 + 0 = 25
        let packed = vec![65u8, 25u8];
        let scales = vec![1.0f32, 1.0]; // identity scales

        let result = ternary_matmul_cpu(&input, &packed, &scales, None, 2, 4, 4).unwrap();
        let data: Vec<f32> = result.to_vec2().unwrap().into_iter().flatten().collect();

        // Row 0 dot: 1*1 + 2*(-1) + 3*0 + 4*1 = 1 - 2 + 0 + 4 = 3
        assert!((data[0] - 3.0).abs() < 1e-6, "got {}", data[0]);
        // Row 1 dot: 1*0 + 2*1 + 3*1 + 4*(-1) = 0 + 2 + 3 - 4 = 1
        assert!((data[1] - 1.0).abs() < 1e-6, "got {}", data[1]);
    }

    #[test]
    fn test_ternary_matmul_with_offsets() {
        // Same setup as above but with per-row offsets
        let input = Tensor::new(&[[1.0f32, 2.0, 3.0, 4.0]], &Device::Cpu).unwrap();
        let packed = vec![65u8, 25u8]; // same weights as above
        let scales = vec![1.0f32, 1.0];
        let offsets = vec![0.5f32, -0.5]; // per-row offsets

        let result = ternary_matmul_cpu(&input, &packed, &scales, Some(&offsets), 2, 4, 4).unwrap();
        let data: Vec<f32> = result.to_vec2().unwrap().into_iter().flatten().collect();

        // sum(x) = 1 + 2 + 3 + 4 = 10
        // Row 0: 3.0 + 0.5 * 10 = 8.0
        assert!((data[0] - 8.0).abs() < 1e-6, "got {}", data[0]);
        // Row 1: 1.0 + (-0.5) * 10 = -4.0
        assert!((data[1] - (-4.0)).abs() < 1e-6, "got {}", data[1]);
    }
}
