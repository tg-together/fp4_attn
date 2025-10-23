import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

@triton.jit
def f32_as_u32(x: tl.tensor):
    return x.to(tl.uint32, bitcast=True)

@triton.jit
def u32_as_f32(x: tl.tensor):
    return x.to(tl.float32, bitcast=True)

@triton.jit
def f32_to_f4_pair(x, y):
    SCALE = 1.1754943508222875e-38
    x = tl.clamp(u32_as_f32(x), -6.0, 6.0) * SCALE
    y = tl.clamp(u32_as_f32(y), -6.0, 6.0) * SCALE
    ux = f32_as_u32(x) + (1 << 21)
    uy = f32_as_u32(y) + (1 << 21)
    ux = ((ux >> 22) & 7) | ((ux >> 28) & 8)
    uy = ((uy >> 22) & 7) | ((uy >> 28) & 8)
    return (ux | (uy << 4))

@triton.jit
def f32_to_f4_single(x: tl.tensor):
    SCALE: tl.constexpr = 1.1754943508222875e-38
    x = tl.clamp(x, -6.0, 6.0) * SCALE
    ux = f32_as_u32(x) + (1 << 21)
    ux = ((ux >> 22) & 7) | ((ux >> 28) & 8)
    return ux

@triton.jit
def f4_to_f32_single(x: tl.tensor):
    x = x.to(tl.uint32, bitcast=True)
    ISCALE: tl.constexpr = 8.507059173023462e+37
    ux = x & 15
    ux = ((ux & 7) << 22) | ((ux & 8) << 28)
    return u32_as_f32(ux) * ISCALE

@triton.jit
def add_fp8(x: tl.tensor, y):
    return (x.to(tl.uint8, bitcast=True) + y.to(tl.uint8)).to(tl.float8e4nv, bitcast=True)

@triton.jit
def quantize_to_fp4_single_nosearch(x: tl.tensor, offset: tl.int8, S_BLOCK_SIZE: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    max_abs_x = tl.max(tl.abs(x), axis=-1) * (1.0 / 6.0)
    xscale_f8 = max_abs_x.to(tl.float8e4nv, fp_downcast_rounding="rtne")[:, None]
    xscale_f8 = add_fp8(xscale_f8, offset)
    xscale = xscale_f8.to(tl.float32)
    xscale_inv = tl.where(xscale == 0, 0.0, 1.0 / xscale)
    x = x * xscale_inv
    quantized = f32_to_f4_single(x)
    return quantized, xscale_f8


@triton.autotune(
    configs=[
        triton.Config({"S_BLOCK_SIZE": 16}),
        triton.Config({"S_BLOCK_SIZE": 32}),
        triton.Config({"S_BLOCK_SIZE": 64}),
        triton.Config({"S_BLOCK_SIZE": 128}),
        triton.Config({"S_BLOCK_SIZE": 256}),
        triton.Config({"S_BLOCK_SIZE": 512}),
        triton.Config({"S_BLOCK_SIZE": 1024}),
    ],
    key=["seq_len"],
)
@triton.jit
def quant_kernel(
    input: tl.tensor,
    output: tl.tensor,
    scales_output: tl.tensor,
    stride_bhz_input: tl.int32, stride_seq_input: tl.int32,
    stride_bhz_output: tl.int32, stride_seq_output: tl.int32,
    stride_bhz_output_sf: tl.int32, stride_seq_output_sf: tl.int32,
    seq_len: tl.int32,
    offset_val: tl.int32,
    S_BLOCK_SIZE: tl.constexpr,
    D_BLOCK_SIZE: tl.constexpr,
): 
    off_blk = tl.program_id(0)
    off_s = tl.program_id(1)
    offset = off_blk * stride_bhz_input

    SCALED_D_BLOCK_SIZE: tl.constexpr = D_BLOCK_SIZE

    s_block = tl.arange(0, S_BLOCK_SIZE)[:, None] + off_s * S_BLOCK_SIZE
    d_block = tl.arange(0, D_BLOCK_SIZE)
    mask = s_block < seq_len
    block = s_block * stride_seq_input + d_block

    vals = tl.load(input + offset + block, mask=mask)
    quantized_vals, scales = quantize_to_fp4_single_nosearch(vals, offset_val.to(tl.int8), S_BLOCK_SIZE, D_BLOCK_SIZE)

    q_offset = off_blk * stride_bhz_output 
    sq_block = tl.arange(0, S_BLOCK_SIZE)[:, None] + off_s * S_BLOCK_SIZE
    dq_block = tl.arange(0, SCALED_D_BLOCK_SIZE)
    mask = sq_block < seq_len
    block = sq_block * stride_seq_output + dq_block
    tl.store(output + q_offset + block, quantized_vals, mask=mask)
    
    s_offset = off_blk * stride_bhz_output_sf
    sq_block = tl.arange(0, S_BLOCK_SIZE)[:, None] + off_s * S_BLOCK_SIZE
    mask = sq_block < seq_len
    block = sq_block * stride_seq_output_sf
    tl.store(scales_output + s_offset + block, scales, mask=mask)


def quantize(x: torch.tensor, offset: int = 0):
    b, h, n, d = x.shape
    BLOCK_SIZE = 16
    assert d % BLOCK_SIZE == 0

    x = x.reshape(b * h, n * (d // BLOCK_SIZE), BLOCK_SIZE) 
    rb, rs, rn = x.shape
    scales = torch.empty((rb, rs, 1), dtype=torch.float8_e4m3fn, device=x.device)
    quantized = torch.empty((rb, rs, rn), dtype=torch.uint8, device=x.device)
    
    grid = lambda META: (rb, triton.cdiv(rs, META["S_BLOCK_SIZE"]), 1)
    with torch.cuda.device(x.device.index):
        quant_kernel[grid](
            x, quantized, scales,
            x.stride(0), x.stride(1),
            quantized.stride(0), quantized.stride(1),
            scales.stride(0), scales.stride(1),
            rs,
            offset,
            D_BLOCK_SIZE=BLOCK_SIZE,
        )

    quantized = quantized.reshape(b, h, n, d)
    scales = scales.reshape(b, h, n, d // BLOCK_SIZE)
    return quantized, scales

def uint8_to_fp4(x: torch.tensor):
    # fp4 is stored unpacked as the first 4 bits of the uint8
    # This matches the f4_to_f32_single implementation in Triton
    ISCALE = 8.507059173023462e+37
    
    x = x.to(torch.uint8)
    # Extract fp4 bits (lower 4 bits)
    ux = x & 15
    
    # Reconstruct as uint32 following the Triton implementation:
    # ((ux & 7) << 22) | ((ux & 8) << 28)
    # Bits 0-2 go to bit position 22, bit 3 (sign) goes to position 28
    ux_expanded = ((ux.to(torch.int32) & 7) << 22) | ((ux.to(torch.int32) & 8) << 28)
    
    # Reinterpret as float32 and scale (keep on GPU)
    result = ux_expanded.view(torch.float32) * ISCALE
    
    return result



if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Test a simple example first
    x = torch.randn(1, 1, 1, 16, device=torch.device('cuda:0'), dtype=torch.float32).normal_(mean=0.0, std=1.0)
    quantized, scales = quantize(x, offset=0)
    fp4 = uint8_to_fp4(quantized)
    print("Sample output:", fp4)
    
    # Now create a test with random normal input for different offsets
    # Create random normal input values
    input_vals_reshaped = torch.randn(1, 1, 1, 16, device=torch.device('cuda:0'), dtype=torch.float32).normal_(mean=0.0, std=3.0)
    
    offsets = list(range(-7, 9))  # -7 to 8 inclusive
    input_vals_flat = input_vals_reshaped.flatten().cpu().numpy()
    
    # Collect errors for each offset
    mse_values = []
    mae_values = []
    
    # Create subplots: one for errors, one for MSE comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
    
    # Plot 1: Reconstruction errors (difference from original)
    for offset in offsets:
        quantized, scales = quantize(input_vals_reshaped, offset=offset)
        fp4_vals = uint8_to_fp4(quantized)
        
        # Multiply by scales to get reconstructed values
        scales_expanded = scales.repeat_interleave(16, dim=-1)
        reconstructed_vals = fp4_vals * scales_expanded.to(torch.float32)
        
        # Calculate errors
        reconstructed_vals_flat = reconstructed_vals.flatten().cpu().numpy()
        errors = reconstructed_vals_flat - input_vals_flat
        
        # Calculate metrics
        mse = np.mean(errors ** 2)
        mae = np.mean(np.abs(errors))
        mse_values.append(mse)
        mae_values.append(mae)
        
        # Scatter plot of errors with jitter for visibility
        x_coords = np.full_like(errors, offset) + np.random.uniform(-0.2, 0.2, size=errors.shape)
        ax1.scatter(x_coords, errors, color='darkred', s=40, alpha=0.6)
    
    ax1.set_xlabel('Offset', fontsize=12)
    ax1.set_ylabel('Reconstruction Error', fontsize=12)
    ax1.set_title('Reconstruction Error (Reconstructed - Original)', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='k', linestyle='--', linewidth=1)
    ax1.set_xticks(offsets)
    
    # Plot 2: MSE for each offset
    ax2.plot(offsets, mse_values, 'o-', color='darkblue', linewidth=2, markersize=8, label='MSE')
    ax2.set_xlabel('Offset', fontsize=12)
    ax2.set_ylabel('Mean Squared Error', fontsize=12)
    ax2.set_title('MSE vs Offset (Lower is Better)', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(offsets)
    
    # Highlight the best offset
    best_offset_idx = np.argmin(mse_values)
    best_offset = offsets[best_offset_idx]
    ax2.scatter([best_offset], [mse_values[best_offset_idx]], color='red', s=200, zorder=5, marker='*', label=f'Best: offset={best_offset}')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('fp4_quantization_offsets.png', dpi=150, bbox_inches='tight')
    print(f"Plot saved to fp4_quantization_offsets.png")
    print(f"Best offset: {best_offset} with MSE: {mse_values[best_offset_idx]:.6f}")
    plt.show()