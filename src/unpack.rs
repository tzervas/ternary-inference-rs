//! Ternary weight unpacking and dequantization.
//!
//! Handles the base-3 packed format where 4 ternary values {-1, 0, +1}
//! are stored per byte using the encoding: v0 + v1*3 + v2*9 + v3*27.

use candle_core::{Device, Tensor};

use crate::error::{Error, Result};

/// Unpack a single byte into 4 ternary values (base-3 encoding).
///
/// Encoding: byte = v0 + v1*3 + v2*9 + v3*27
/// where each vi is mapped {0,1,2} -> {-1,0,+1}
///
/// Used by: tzervas/qwen2.5-coder-32b-bitnet-1.58b
#[inline]
pub fn unpack_byte(byte: u8) -> [i8; 4] {
    let b = byte as i16;
    [
        (b % 3 - 1) as i8,
        ((b / 3) % 3 - 1) as i8,
        ((b / 9) % 3 - 1) as i8,
        ((b / 27) % 3 - 1) as i8,
    ]
}

/// Unpack a single byte into 4 ternary values (2-bit encoding).
///
/// Encoding: byte = v0 + v1*4 + v2*16 + v3*64
/// where each vi occupies 2 bits: {0,1,2} -> {-1,0,+1}
///
/// Used by: microsoft/bitnet-b1.58-2B-4T
#[inline]
pub fn unpack_byte_2bit(byte: u8) -> [i8; 4] {
    [
        (byte & 3) as i8 - 1,
        ((byte >> 2) & 3) as i8 - 1,
        ((byte >> 4) & 3) as i8 - 1,
        ((byte >> 6) & 3) as i8 - 1,
    ]
}

/// Packing format for ternary weights.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PackingFormat {
    /// Base-3: 4 values per byte, packed along columns (in_features).
    /// byte = v0 + v1*3 + v2*9 + v3*27. Max valid byte = 80.
    /// Shape: [out_features, in_features/4]
    Base3Columns,
    /// 2-bit: 4 values per byte, packed along rows (out_features).
    /// byte = v0 + v1*4 + v2*16 + v3*64. Max valid byte = 170.
    /// Shape: [out_features/4, in_features]
    TwoBitRows,
}

/// Unpack a packed uint8 tensor into ternary f32 values (base-3, column-packed).
///
/// Input: uint8 tensor of shape [rows, packed_cols] where packed_cols = cols/4
/// Output: f32 tensor of shape [rows, cols] with values in {-1, 0, +1}
pub fn unpack_ternary(packed: &[u8], rows: usize, cols: usize) -> Result<Vec<f32>> {
    let packed_cols = cols / 4;
    if packed.len() != rows * packed_cols {
        return Err(Error::Shape {
            expected: format!("[{rows}, {packed_cols}] = {} bytes", rows * packed_cols),
            actual: format!("{} bytes", packed.len()),
        });
    }

    let mut output = vec![0.0f32; rows * cols];

    for row in 0..rows {
        for pc in 0..packed_cols {
            let byte = packed[row * packed_cols + pc];
            let vals = unpack_byte(byte);
            let base = row * cols + pc * 4;
            output[base] = vals[0] as f32;
            output[base + 1] = vals[1] as f32;
            output[base + 2] = vals[2] as f32;
            output[base + 3] = vals[3] as f32;
        }
    }

    Ok(output)
}

/// Unpack a packed uint8 tensor into ternary f32 values (2-bit, row-packed, interleaved).
///
/// Input: uint8 tensor of shape [packed_rows, cols] where packed_rows = rows/4
/// Output: f32 tensor of shape [rows, cols] with values in {-1, 0, +1}
///
/// Packing layout (interleaved): byte at (pr, col) stores:
///   - bits 0-1: W[pr, col]
///   - bits 2-3: W[pr + P, col]
///   - bits 4-5: W[pr + 2P, col]
///   - bits 6-7: W[pr + 3P, col]
///
/// where P = packed_rows = rows/4
pub fn unpack_ternary_2bit(packed: &[u8], rows: usize, cols: usize) -> Result<Vec<f32>> {
    let packed_rows = rows / 4;
    if packed.len() != packed_rows * cols {
        return Err(Error::Shape {
            expected: format!("[{packed_rows}, {cols}] = {} bytes", packed_rows * cols),
            actual: format!("{} bytes", packed.len()),
        });
    }

    let mut output = vec![0.0f32; rows * cols];

    for pr in 0..packed_rows {
        for col in 0..cols {
            let byte = packed[pr * cols + col];
            let vals = unpack_byte_2bit(byte);
            // Interleaved: each bit-pair maps to rows separated by packed_rows
            for (k, &v) in vals.iter().enumerate() {
                output[(pr + k * packed_rows) * cols + col] = v as f32;
            }
        }
    }

    Ok(output)
}

/// Dequantize a packed ternary weight tensor with per-group scales (base-3, column-packed).
///
/// # Arguments
/// * `packed` - Raw packed bytes [out_features, in_features/4]
/// * `scales` - Per-group scale factors [out_features, num_groups]
/// * `out_features` - Output dimension
/// * `in_features` - Input dimension
/// * `group_size` - Number of weights per scale group
/// * `device` - Target device for the output tensor
///
/// # Returns
/// Dequantized tensor of shape [out_features, in_features]
pub fn dequantize_weight(
    packed: &[u8],
    scales: &[f32],
    out_features: usize,
    in_features: usize,
    group_size: usize,
    device: &Device,
) -> Result<Tensor> {
    let num_groups = in_features / group_size;

    // Unpack ternary values
    let ternary = unpack_ternary(packed, out_features, in_features)?;

    // Apply per-group scales
    let mut dequantized = vec![0.0f32; out_features * in_features];
    for row in 0..out_features {
        for group in 0..num_groups {
            let scale = scales[row * num_groups + group];
            let base = row * in_features + group * group_size;
            for k in 0..group_size {
                dequantized[base + k] = ternary[base + k] * scale;
            }
        }
    }

    Tensor::from_vec(dequantized, (out_features, in_features), device)
        .map_err(Error::Tensor)
}

/// Dequantize a packed ternary weight with per-tensor scale (2-bit, row-packed).
///
/// Used by microsoft/bitnet-b1.58-2B-4T where each projection has a single scale.
pub fn dequantize_weight_2bit(
    packed: &[u8],
    scale: f32,
    out_features: usize,
    in_features: usize,
    device: &Device,
) -> Result<Tensor> {
    let ternary = unpack_ternary_2bit(packed, out_features, in_features)?;
    let dequantized: Vec<f32> = ternary.iter().map(|&v| v * scale).collect();
    Tensor::from_vec(dequantized, (out_features, in_features), device)
        .map_err(Error::Tensor)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unpack_byte() {
        // Pack: (-1,0,1,0) -> mapped (0,1,2,1) -> 0 + 1*3 + 2*9 + 1*27 = 48
        let vals = unpack_byte(48);
        assert_eq!(vals, [-1, 0, 1, 0]);
    }

    #[test]
    fn test_unpack_byte_all_zeros() {
        // Pack: (0,0,0,0) -> mapped (1,1,1,1) -> 1 + 3 + 9 + 27 = 40
        let vals = unpack_byte(40);
        assert_eq!(vals, [0, 0, 0, 0]);
    }

    #[test]
    fn test_unpack_byte_all_ones() {
        // Pack: (1,1,1,1) -> mapped (2,2,2,2) -> 2 + 6 + 18 + 54 = 80
        let vals = unpack_byte(80);
        assert_eq!(vals, [1, 1, 1, 1]);
    }

    #[test]
    fn test_unpack_byte_all_neg_ones() {
        // Pack: (-1,-1,-1,-1) -> mapped (0,0,0,0) -> 0
        let vals = unpack_byte(0);
        assert_eq!(vals, [-1, -1, -1, -1]);
    }

    #[test]
    fn test_unpack_ternary_small() {
        // 2 rows, 8 cols -> packed is 2 rows, 2 packed_cols
        let packed = vec![40, 80, 0, 48]; // row0: all-zero + all-one, row1: all-neg + mixed
        let result = unpack_ternary(&packed, 2, 8).unwrap();
        assert_eq!(result.len(), 16);
        // Row 0: [0,0,0,0, 1,1,1,1]
        assert_eq!(&result[0..8], &[0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]);
        // Row 1: [-1,-1,-1,-1, -1,0,1,0]
        assert_eq!(
            &result[8..16],
            &[-1.0, -1.0, -1.0, -1.0, -1.0, 0.0, 1.0, 0.0]
        );
    }

    #[test]
    fn test_unpack_byte_2bit() {
        // All zeros: 1 + 1*4 + 1*16 + 1*64 = 85
        let vals = unpack_byte_2bit(85);
        assert_eq!(vals, [0, 0, 0, 0]);
    }

    #[test]
    fn test_unpack_byte_2bit_all_ones() {
        // All +1: 2 + 2*4 + 2*16 + 2*64 = 170
        let vals = unpack_byte_2bit(170);
        assert_eq!(vals, [1, 1, 1, 1]);
    }

    #[test]
    fn test_unpack_byte_2bit_all_neg() {
        // All -1: 0 + 0*4 + 0*16 + 0*64 = 0
        let vals = unpack_byte_2bit(0);
        assert_eq!(vals, [-1, -1, -1, -1]);
    }

    #[test]
    fn test_unpack_byte_2bit_mixed() {
        // (-1, 0, 1, 0): 0 + 1*4 + 2*16 + 1*64 = 100
        let vals = unpack_byte_2bit(100);
        assert_eq!(vals, [-1, 0, 1, 0]);
    }

    #[test]
    fn test_dequantize_weight() {
        // 2x8 weight, group_size=4, so 2 groups per row
        let packed = vec![40, 80, 40, 0]; // row0: zeros+ones, row1: zeros+neg-ones
        let scales = vec![
            2.0, 3.0, // row 0 scales (group0=2.0, group1=3.0)
            0.5, 1.0, // row 1 scales
        ];
        let device = Device::Cpu;
        let result = dequantize_weight(&packed, &scales, 2, 8, 4, &device).unwrap();
        let data: Vec<f32> = result.to_vec2().unwrap().into_iter().flatten().collect();

        // Row 0: [0*2, 0*2, 0*2, 0*2, 1*3, 1*3, 1*3, 1*3] = [0,0,0,0, 3,3,3,3]
        assert_eq!(&data[0..8], &[0.0, 0.0, 0.0, 0.0, 3.0, 3.0, 3.0, 3.0]);
        // Row 1: [0*0.5, 0*0.5, 0*0.5, 0*0.5, -1*1, -1*1, -1*1, -1*1]
        assert_eq!(
            &data[8..16],
            &[0.0, 0.0, 0.0, 0.0, -1.0, -1.0, -1.0, -1.0]
        );
    }
}
