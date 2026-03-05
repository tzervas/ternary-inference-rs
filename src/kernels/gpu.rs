//! GPU-accelerated ternary matmul via LUT dequantization + cuBLAS.
//!
//! Strategy: avoid custom GPU kernels by using candle's built-in CUDA ops:
//! 1. Pre-computed lookup table maps each packed byte → 4 ternary values
//! 2. `index_select` does the unpacking entirely on GPU
//! 3. Per-group scales applied via broadcast multiply on GPU
//! 4. cuBLAS handles the actual matrix multiply
//!
//! This avoids the CPU dequantization bottleneck while still using
//! highly optimized cuBLAS GEMM for the heavy compute.

use candle_core::{DType, Device, Tensor};

use crate::error::{Error, Result};
use crate::unpack::unpack_byte;

/// GPU ternary matmul: LUT unpack on GPU → scale → cuBLAS matmul.
///
/// Packed bytes and scales are transferred from CPU per-projection.
/// The input tensor must already be on the target GPU device.
///
/// Peak temporary VRAM per call: ~2× the dequantized weight size.
/// For the largest projection (27648×5120 f16): ~564 MB temporary.
pub fn gpu_ternary_matmul(
    input: &Tensor,
    packed_weight: &[u8],
    scales: &[f32],
    out_features: usize,
    in_features: usize,
    group_size: usize,
    device: &Device,
) -> Result<Tensor> {
    let num_groups = in_features / group_size;

    // LUT: [256, 4] f16 on GPU — maps each byte to 4 ternary values
    let lut = create_unpack_lut(device)?;

    // Convert packed bytes to u32 indices, upload to GPU
    let indices_u32: Vec<u32> = packed_weight.iter().map(|&b| b as u32).collect();
    let indices =
        Tensor::from_vec(indices_u32, packed_weight.len(), device).map_err(Error::Tensor)?;

    // LUT lookup on GPU: [out * packed_cols, 4] f16
    let unpacked = lut.index_select(&indices, 0).map_err(Error::Tensor)?;
    drop(indices); // free GPU memory

    // Reshape to [out, in]: 4 values per byte become 4 consecutive columns
    let ternary = unpacked
        .reshape((out_features, in_features))
        .map_err(Error::Tensor)?;

    // Apply per-group scales on GPU
    // scales: [out, num_groups] → broadcast to [out, in]
    let scales_f16: Vec<half::f16> = scales.iter().map(|&s| half::f16::from_f32(s)).collect();
    let scales_t =
        Tensor::from_vec(scales_f16, (out_features, num_groups), device).map_err(Error::Tensor)?;
    let scales_exp = scales_t
        .unsqueeze(2)
        .map_err(Error::Tensor)?
        .broadcast_as((out_features, num_groups, group_size))
        .map_err(Error::Tensor)?
        .contiguous()
        .map_err(Error::Tensor)?
        .reshape((out_features, in_features))
        .map_err(Error::Tensor)?;
    drop(scales_t);

    let dequant = ternary.mul(&scales_exp).map_err(Error::Tensor)?;
    drop(ternary);
    drop(scales_exp);

    // Cast input to f16 for Tensor Core matmul, then back to f32
    let input_f16 = input.to_dtype(DType::F16).map_err(Error::Tensor)?;
    let result_f16 = input_f16
        .matmul(&dequant.t().map_err(Error::Tensor)?)
        .map_err(Error::Tensor)?;
    drop(dequant);

    result_f16.to_dtype(DType::F32).map_err(Error::Tensor)
}

/// GPU ternary matmul for 2-bit row-packed format (BitNet-2B-4T).
///
/// Format: [out/4, in] where each byte stores 4 output features.
/// Single per-tensor scale factor.
pub fn gpu_ternary_matmul_2bit(
    input: &Tensor,
    packed_weight: &[u8],
    scale: f32,
    out_features: usize,
    in_features: usize,
    device: &Device,
) -> Result<Tensor> {
    let lut = create_unpack_lut_2bit(device)?;

    // Upload packed bytes as u32 indices
    let indices_u32: Vec<u32> = packed_weight.iter().map(|&b| b as u32).collect();
    let packed_rows = out_features / 4;
    let indices = Tensor::from_vec(indices_u32, (packed_rows, in_features), device)
        .map_err(Error::Tensor)?;

    // LUT lookup: [packed_rows * in_features] → [packed_rows * in_features, 4] f16
    let flat_indices = indices.flatten_all().map_err(Error::Tensor)?;
    let unpacked = lut.index_select(&flat_indices, 0).map_err(Error::Tensor)?;
    drop(flat_indices);

    // Reshape to [packed_rows, in_features, 4]
    // Interleaved layout: bit k maps to row (pr + k * packed_rows)
    // Permute to [4, packed_rows, in_features] then reshape to [out_features, in_features]
    let unpacked = unpacked
        .reshape((packed_rows, in_features, 4))
        .map_err(Error::Tensor)?
        .permute((2, 0, 1))
        .map_err(Error::Tensor)?
        .contiguous()
        .map_err(Error::Tensor)?
        .reshape((out_features, in_features))
        .map_err(Error::Tensor)?;

    // Apply per-tensor scale using affine (scale * x + 0)
    let dequant = unpacked
        .affine(scale as f64, 0.0)
        .map_err(Error::Tensor)?;

    // f16 matmul
    let input_f16 = input.to_dtype(DType::F16).map_err(Error::Tensor)?;
    let result_f16 = input_f16
        .matmul(&dequant.t().map_err(Error::Tensor)?)
        .map_err(Error::Tensor)?;
    drop(dequant);

    result_f16.to_dtype(DType::F32).map_err(Error::Tensor)
}

/// Create the byte → ternary lookup table on GPU (2-bit encoding).
///
/// Returns a [256, 4] f16 tensor where row `b` contains the 4 ternary
/// values decoded from the 2-bit packed byte.
fn create_unpack_lut_2bit(device: &Device) -> Result<Tensor> {
    let mut lut = vec![half::f16::from_f32(0.0); 256 * 4];
    for b in 0..=255u8 {
        let vals = crate::unpack::unpack_byte_2bit(b);
        let base = b as usize * 4;
        lut[base] = half::f16::from_f32(vals[0] as f32);
        lut[base + 1] = half::f16::from_f32(vals[1] as f32);
        lut[base + 2] = half::f16::from_f32(vals[2] as f32);
        lut[base + 3] = half::f16::from_f32(vals[3] as f32);
    }
    Tensor::from_vec(lut, (256, 4), device).map_err(Error::Tensor)
}

/// Create the byte → ternary lookup table on GPU.
///
/// Returns a [256, 4] f16 tensor where row `b` contains the 4 ternary
/// values {-1, 0, +1} encoded by byte `b` in base-3 format.
/// Rows 81-255 are invalid and contain zeros.
fn create_unpack_lut(device: &Device) -> Result<Tensor> {
    let mut lut = vec![half::f16::from_f32(0.0); 256 * 4];
    for b in 0..=80u8 {
        let vals = unpack_byte(b);
        let base = b as usize * 4;
        lut[base] = half::f16::from_f32(vals[0] as f32);
        lut[base + 1] = half::f16::from_f32(vals[1] as f32);
        lut[base + 2] = half::f16::from_f32(vals[2] as f32);
        lut[base + 3] = half::f16::from_f32(vals[3] as f32);
    }
    Tensor::from_vec(lut, (256, 4), device).map_err(Error::Tensor)
}
