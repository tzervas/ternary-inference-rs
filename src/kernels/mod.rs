//! Ternary matmul kernel dispatch.
//!
//! Provides CPU and GPU implementations of matrix multiplication
//! with ternary weights:
//!
//! - **CPU**: Direct packed-weight matmul (add/subtract/skip, no dequant)
//! - **GPU**: LUT-based dequantization on GPU + cuBLAS f16 matmul

pub mod cpu;

#[cfg(feature = "cuda")]
pub mod gpu;

use candle_core::{Device, Tensor};
use crate::error::Result;

use crate::unpack::PackingFormat;

/// Perform ternary matrix multiplication: output = input @ ternary_weight^T * scales
///
/// Dispatches to CPU or GPU kernel based on device.
/// `group_size` is 0 for per-tensor scale (BitNet), >0 for per-group (Qwen2).
pub fn ternary_matmul(
    input: &Tensor,
    packed_weight: &[u8],
    scales: &[f32],
    out_features: usize,
    in_features: usize,
    group_size: usize,
    device: &Device,
) -> Result<Tensor> {
    ternary_matmul_fmt(
        input,
        packed_weight,
        scales,
        out_features,
        in_features,
        group_size,
        PackingFormat::Base3Columns,
        device,
    )
}

/// Perform ternary matmul with explicit packing format.
#[allow(clippy::too_many_arguments)]
pub fn ternary_matmul_fmt(
    input: &Tensor,
    packed_weight: &[u8],
    scales: &[f32],
    out_features: usize,
    in_features: usize,
    group_size: usize,
    format: PackingFormat,
    device: &Device,
) -> Result<Tensor> {
    match device {
        Device::Cpu => match format {
            PackingFormat::Base3Columns => cpu::ternary_matmul_cpu(
                input,
                packed_weight,
                scales,
                out_features,
                in_features,
                group_size,
            ),
            PackingFormat::TwoBitRows => {
                // Dequantize on CPU and use standard matmul
                let dequant = crate::unpack::dequantize_weight_2bit(
                    packed_weight,
                    scales[0],
                    out_features,
                    in_features,
                    &Device::Cpu,
                )?;
                input.matmul(&dequant.t()?).map_err(crate::Error::Tensor)
            }
        },
        #[cfg(feature = "cuda")]
        Device::Cuda(_) => match format {
            PackingFormat::Base3Columns => gpu::gpu_ternary_matmul(
                input,
                packed_weight,
                scales,
                out_features,
                in_features,
                group_size,
                device,
            ),
            PackingFormat::TwoBitRows => gpu::gpu_ternary_matmul_2bit(
                input,
                packed_weight,
                scales[0],
                out_features,
                in_features,
                device,
            ),
        },
        #[allow(unreachable_patterns)]
        _ => {
            // Fallback: dequantize on CPU, transfer, matmul
            let dequant = match format {
                PackingFormat::Base3Columns => crate::unpack::dequantize_weight(
                    packed_weight, scales, out_features, in_features, group_size, device,
                )?,
                PackingFormat::TwoBitRows => crate::unpack::dequantize_weight_2bit(
                    packed_weight, scales[0], out_features, in_features, device,
                )?,
            };
            input.matmul(&dequant.t()?).map_err(crate::Error::Tensor)
        }
    }
}
