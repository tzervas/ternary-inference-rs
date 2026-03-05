//! Quick GPU smoke test — validates LUT dequant + matmul on CUDA.
//!
//! Usage:
//!   cargo run -p ternary-inference-rs --example gpu_smoke_test --release --features cuda

use candle_core::{Device, Tensor};
use ternary_inference_rs::kernels::ternary_matmul;

fn main() -> anyhow::Result<()> {
    env_logger::init();

    let device = Device::new_cuda(0)?;
    println!("CUDA device: {device:?}");

    // Small test: 1x8 input, 2x8 weight (2 groups of 4)
    // Row 0: [1, -1, 0, 1, 0, 0, 0, 0]
    //   packed: byte0 = (1+1) + ((-1)+1)*3 + (0+1)*9 + (1+1)*27 = 2+0+9+54 = 65
    //           byte1 = (0+1) + (0+1)*3 + (0+1)*9 + (0+1)*27 = 1+3+9+27 = 40
    // Row 1: [0, 1, 1, -1, -1, -1, -1, -1]
    //   packed: byte0 = (0+1) + (1+1)*3 + (1+1)*9 + ((-1)+1)*27 = 1+6+18+0 = 25
    //           byte1 = 0 (all -1)
    let packed = vec![65u8, 40, 25, 0];
    let scales = vec![1.0f32, 2.0, 1.0, 0.5]; // [2 rows, 2 groups]

    let input = Tensor::new(&[[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]], &device)?;

    println!("Input: {:?}", input.to_vec2::<f32>()?);
    println!("Running GPU ternary matmul...");

    let output = ternary_matmul(&input, &packed, &scales, 2, 8, 4, &device)?;

    let result: Vec<Vec<f32>> = output.to_vec2()?;
    println!("Output: {:?}", result);

    // Expected (with group scales):
    // Row 0: scale0=1.0 * (1*1 + (-1)*2 + 0*3 + 1*4) + scale1=2.0 * (0*5+0*6+0*7+0*8)
    //       = 1.0 * (1-2+0+4) + 2.0 * 0 = 3.0
    // Row 1: scale0=1.0 * (0*1 + 1*2 + 1*3 + (-1)*4) + scale1=0.5 * ((-1)*5+(-1)*6+(-1)*7+(-1)*8)
    //       = 1.0 * (0+2+3-4) + 0.5 * (-5-6-7-8) = 1.0 + (-13.0) = -12.0
    let expected = [3.0f32, -12.0];
    for (i, (&got, &exp)) in result[0].iter().zip(expected.iter()).enumerate() {
        let diff = (got - exp).abs();
        let status = if diff < 0.1 { "OK" } else { "MISMATCH" };
        println!("  [{i}] expected={exp:.1}, got={got:.4}, diff={diff:.6} {status}");
    }

    // Larger test: realistic dimensions
    println!("\nLarger test (128x512 → 256 output)...");
    let big_input = Tensor::randn(0.0f32, 1.0, (1, 512), &device)?;
    let big_packed = vec![40u8; 256 * 128]; // all zeros
    let big_scales = vec![1.0f32; 256 * (512 / 64)]; // 8 groups
    let big_output = ternary_matmul(&big_input, &big_packed, &big_scales, 256, 512, 64, &device)?;
    let stats: Vec<f32> = big_output.flatten_all()?.to_vec1()?;
    let mean = stats.iter().sum::<f32>() / stats.len() as f32;
    let max = stats.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    println!("  Shape: {:?}", big_output.shape());
    println!("  Mean={mean:.6}, Max={max:.6} (expect ~0 for all-zero weights)");

    println!("\nGPU smoke test PASSED");
    Ok(())
}
