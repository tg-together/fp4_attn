import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
import matplotlib.pyplot as plt
import math

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
def quantize_to_fp4_single_nosearch(x: tl.tensor, S_BLOCK_SIZE: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    max_abs_x = tl.maximum(tl.max(tl.abs(x), axis=-1), 1e-10) * (1.0 / 6.0)
    max_abs_x = tl.clamp(max_abs_x, 1e-10 / 6.0, 448.0)
    xscale_f8 = max_abs_x.to(tl.float8e4nv, fp_downcast_rounding="rtne")[:, None]
    xscale = xscale_f8.to(tl.float32)
    xscale_inv = 1.0 / xscale
    x = x * xscale_inv
    quantized = f32_to_f4_single(x)
    return quantized, xscale_f8

@triton.jit
def dequantize_to_f32_single(x: tl.tensor, xscale: tl.tensor):
    x = f4_to_f32_single(x) * xscale.to(tl.float32)
    return x

@triton.jit
def add_fp8(x: tl.tensor, y):
    return (x.to(tl.uint8, bitcast=True) + y.to(tl.uint8)).to(tl.float8e4nv, bitcast=True)

@triton.jit
def quantize_to_fp4_single_search(x: tl.tensor, S_BLOCK_SIZE: tl.constexpr, BLOCK_SIZE: tl.constexpr, min_range: tl.constexpr, max_range: tl.constexpr):
    max_abs_x = tl.maximum(tl.max(tl.abs(x), axis=-1), 1e-10) * (1.0 / 6.0)
    max_abs_x = tl.clamp(max_abs_x, 1e-10 / 6.0, 448.0)
    xscale_f8 = max_abs_x.to(tl.float8e4nv, fp_downcast_rounding="rtne")[:, None]
    best_loss = tl.full((S_BLOCK_SIZE,), float("inf"), dtype=tl.float32)
    best_quantized = tl.zeros((S_BLOCK_SIZE, BLOCK_SIZE), dtype=tl.uint8)
    best_xscale = tl.zeros((S_BLOCK_SIZE,1), dtype=tl.float8e4nv)
    best_offset = tl.zeros((S_BLOCK_SIZE,1), dtype=tl.int8)
    for offset_val in range(min_range, max_range + 1, 1):
        xscale = (add_fp8(xscale_f8, offset_val)).to(tl.float32)
        xscale_inv = 1.0 / xscale
        quantized = f32_to_f4_single(x * xscale_inv)
        loss = (x - f4_to_f32_single(quantized) * xscale)
        loss = tl.sum(loss * loss, axis=-1)
        better = loss < best_loss
        best_loss = tl.where(better, loss, best_loss)
        best_quantized = tl.where(better[:, None], quantized.to(tl.uint8), best_quantized)
        new_xscale_fp8 = (xscale).to(tl.float8e4nv)
        best_xscale = tl.where(better[:, None], new_xscale_fp8, best_xscale)
        best_offset = tl.where(better[:, None], offset_val.to(tl.int8), best_offset)
    return best_quantized, xscale_f8, best_offset

@triton.autotune(
    configs=[
        triton.Config({"S_BLOCK_SIZE": 16}),
        triton.Config({"S_BLOCK_SIZE": 32}),
        triton.Config({"S_BLOCK_SIZE": 64}),
        triton.Config({"S_BLOCK_SIZE": 128}),
        triton.Config({"S_BLOCK_SIZE": 256}),
        triton.Config({"S_BLOCK_SIZE": 512}),
        triton.Config({"S_BLOCK_SIZE": 1024}),
        triton.Config({"S_BLOCK_SIZE": 2048}),
    ],
    key=["seq_len"],
)
@triton.jit
def quant_kernel(
    input: tl.tensor,
    output: tl.tensor,
    scale_offset_distribution: tl.tensor,
    scales_ptr,
    stride_bhz_input: tl.int32, stride_seq_input: tl.int32,
    stride_bhz_output: tl.int32, stride_seq_output: tl.int32,
    stride_bhz_output_sf: tl.int32, stride_seq_output_sf: tl.int32,
    stride_bhz_output_scales: tl.int32, stride_seq_output_scales: tl.int32,
    seq_len: tl.int32,
    S_BLOCK_SIZE: tl.constexpr,
    D_BLOCK_SIZE: tl.constexpr,
    min_range: tl.constexpr,
    max_range: tl.constexpr,
): 
    off_blk = tl.program_id(0)
    off_s = tl.program_id(1)
    offset = off_blk * stride_bhz_input

    s_block = tl.arange(0, S_BLOCK_SIZE)[:, None] + off_s * S_BLOCK_SIZE
    d_block = tl.arange(0, D_BLOCK_SIZE)
    mask = s_block < seq_len
    block = s_block * stride_seq_input + d_block

    vals = tl.load(input + offset + block, mask=mask, other=0.0)
    quantized_vals, scales, offset = quantize_to_fp4_single_search(vals, S_BLOCK_SIZE, D_BLOCK_SIZE, min_range, max_range)
    quantized_vals = quantized_vals.to(dtype=tl.uint32)
    reconstructed_vals = dequantize_to_f32_single(quantized_vals, scales)

    q_offset = off_blk * stride_bhz_output 
    sq_block = tl.arange(0, S_BLOCK_SIZE)[:, None] + off_s * S_BLOCK_SIZE
    dq_block = tl.arange(0, D_BLOCK_SIZE)
    mask = sq_block < seq_len
    block = sq_block * stride_seq_output + dq_block
    tl.store(output + q_offset + block, reconstructed_vals, mask=mask)

    s_offset = off_blk * stride_bhz_output_sf
    sq_block = tl.arange(0, S_BLOCK_SIZE)[:, None] + off_s * S_BLOCK_SIZE
    mask = sq_block < seq_len
    block = sq_block * stride_seq_output_sf
    tl.store(scale_offset_distribution + s_offset + block, offset, mask=mask)

    s_offset = off_blk * stride_bhz_output_scales
    sq_block = tl.arange(0, S_BLOCK_SIZE)[:, None] + off_s * S_BLOCK_SIZE
    mask = sq_block < seq_len
    block = sq_block * stride_seq_output_scales
    tl.store(scales_ptr+ s_offset + block, scales, mask=mask)


def quantize_single(x: torch.tensor, min_range: int = -7, max_range: int = 8, BLOCK_SIZE: int = 16):
    b, h, n, d = x.shape
    assert d % BLOCK_SIZE == 0

    x = x.reshape(b * h, n * (d // BLOCK_SIZE), BLOCK_SIZE).contiguous()
    rb, rs, rn = x.shape
    output = torch.empty((rb, rs, rn), dtype=torch.float32, device=x.device)
    scale_offset_distribution = torch.empty((rb, rs, 1), dtype=torch.int8, device=x.device)

    scales = torch.empty((rb, rs, 1), dtype=torch.float32, device=x.device)
    
    grid = lambda META: (rb, triton.cdiv(rs, META["S_BLOCK_SIZE"]), 1)
    with torch.cuda.device(x.device.index):
        quant_kernel[grid](
            x, output, scale_offset_distribution, scales,
            x.stride(0), x.stride(1),
            output.stride(0), output.stride(1),
            scale_offset_distribution.stride(0), scale_offset_distribution.stride(1),
            scales.stride(0), scales.stride(1),
            seq_len=rs,
            D_BLOCK_SIZE=BLOCK_SIZE,
            min_range=min_range,
            max_range=max_range,
        )

    reconstructed_vals = output.reshape(b, h, n, d)
    scale_offset_distribution = scale_offset_distribution.reshape(b, h, n, d // BLOCK_SIZE)
    scales = scales.reshape(b, h, n, d // BLOCK_SIZE)
    return reconstructed_vals, scale_offset_distribution, scales

def get_normal_sample(B, H, N, D, mean: float = 0.0, std: float = 1.0):
    return torch.randn(B, H, N, D, device=torch.device('cuda:0'), dtype=torch.float32).normal_(mean=mean, std=std)

def get_log_normal_poisson_sample(B, H, N, D, mean: float = 0.0, std: float = 0.5):
    tau = torch.empty(B, H, N, D, device='cuda:0', dtype=torch.float32).normal_(mean=math.log(max(mean, 1e-6)) - 0.5 * std ** 2, std=std)
    return torch.poisson(torch.exp(tau)) + 1 

def get_outlier_sample(x, threshold=3.0, single_outlier=True):
    max_vals, max_idx = x.max(dim=-1)
    if single_outlier:
        second_max = torch.topk(x, k=2, dim=-1)[0][:, :, :, 1]
        mask = (max_vals > threshold * second_max) & (max_vals > threshold * x.mean(dim=-1))
    else:
        mask = max_vals > threshold * x.mean(dim=-1)
    return x[mask]

def create_outlier_sample(B, H, N, D, mean=0.0, std=1.0, scale_factor=5.0):
    x = torch.randn(B, H, N, D, device='cuda:0', dtype=torch.float32).normal_(mean=mean, std=std)
    max_abs_idx = x.abs().max(dim=-1)[1]
    x[torch.arange(B, device='cuda:0')[:, None, None], torch.arange(H, device='cuda:0')[None, :, None], torch.arange(N, device='cuda:0')[None, None, :], max_abs_idx] *= scale_factor
    return x


if __name__ == "__main__":
    BLOCK_SIZE = 16
    x = get_normal_sample(64, 64, 4096, BLOCK_SIZE, mean=0.0, std=5.0)
    min_range, max_range = -7, 8
    reconstructed_vals, scale_offset_distribution, _ = quantize_single(x, min_range=min_range, max_range=max_range, BLOCK_SIZE=BLOCK_SIZE)
    scale_offset_distribution_uns = scale_offset_distribution.flatten()

    # Count occurrences of each discrete offset value
    histogram = torch.zeros(max_range - min_range + 1, dtype=torch.long, device=x.device)
    for i, val in enumerate(range(min_range, max_range + 1)):
        histogram[i] = (scale_offset_distribution_uns == val).sum()
    probs = histogram.float() / histogram.sum()
    print(probs)
    
    # Plot the histogram
    plt.figure(figsize=(10, 6))
    plt.bar(range(min_range, max_range + 1, 1), probs.cpu().numpy())
    plt.xlabel('Offset Bucket')
    plt.ylabel('Probability')
    plt.title('Scale Offset Distribution Probabilities')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'scale_offset_probs_BLOCK_SIZE_{BLOCK_SIZE}.png', dpi=150)
    print("Probabilities saved to scale_offset_probs.png")
    plt.show()

    min_range, max_range = -7, 8
    scale_factors = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, scale_factor in enumerate(scale_factors):
        x = create_outlier_sample(64, 64, 4096, BLOCK_SIZE, mean=0.0, std=1.0, scale_factor=scale_factor)
        reconstructed_vals, scale_offset_distribution, _ = quantize_single(x, min_range=min_range, max_range=max_range, BLOCK_SIZE=BLOCK_SIZE)
        scale_offset_distribution_uns = scale_offset_distribution.flatten()
        
        histogram = torch.zeros(max_range - min_range + 1, dtype=torch.long, device=x.device)
        for i, val in enumerate(range(min_range, max_range + 1)):
            histogram[i] = (scale_offset_distribution_uns == val).sum()
        probs = histogram.float() / histogram.sum()
        
        axes[idx].bar(range(min_range, max_range + 1, 1), probs.cpu().numpy())
        axes[idx].set_xlabel('Offset Bucket')
        axes[idx].set_ylabel('Probability')
        axes[idx].set_title(f'Scale Factor={scale_factor}')
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'scale_offset_probs_outliers_BLOCK_SIZE_{BLOCK_SIZE}.png', dpi=150)
    print("Outlier histograms saved to scale_offset_probs_outliers.png")
    plt.show()

    # 2D histogram: mantissa bits of original scale vs offset
    x = create_outlier_sample(64, 64, 4096, BLOCK_SIZE, mean=0.0, std=1.0, scale_factor=1.0)
    _, scale_offset_distribution, scales = quantize_single(x, min_range=min_range, max_range=max_range, BLOCK_SIZE=BLOCK_SIZE)
    
    # Extract mantissa bits from FP8 scales (E4M3 format has 3-bit mantissa)
    # First convert float32 back to float8_e4m3fn to get the actual bit pattern
    scales_fp8 = scales.to(torch.float8_e4m3fn)
    scales_uint8 = scales_fp8.view(torch.uint8).flatten()
    mantissa_bits = (scales_uint8 & 0b00000111).long()  # Extract lower 3 bits
    offsets = scale_offset_distribution.flatten().long()
    
    # Create 2D histogram: 8 mantissa values (0-7) x offset range
    num_mantissa_vals = 8
    num_offset_vals = max_range - min_range + 1
    hist_2d = torch.zeros((num_mantissa_vals, num_offset_vals), device='cuda:0')
    
    for i in range(num_mantissa_vals):
        for j in range(num_offset_vals):
            offset_val = min_range + j
            mask = (mantissa_bits == i) & (offsets == offset_val)
            hist_2d[i, j] = mask.sum()
    
    # Normalize to get probabilities
    hist_2d = hist_2d / hist_2d.sum()
    
    plt.figure(figsize=(8, 6))
    plt.imshow(hist_2d.cpu().numpy(), aspect='auto', origin='lower', extent=[min_range, max_range + 1, 0, 8], cmap='viridis')
    plt.colorbar()
    plt.xlabel('offset from vlim scale')
    plt.ylabel('original vlim scale mantissa')
    plt.tight_layout()
    plt.savefig(f'scale_offset_2d_BLOCK_SIZE_{BLOCK_SIZE}.png', dpi=150)
    print("2D histogram saved to scale_offset_2d.png")
    plt.show()

    # 2D histogram grid: mantissa bits vs offset for different outlier scale factors
    scale_factors = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for idx, scale_factor in enumerate(scale_factors):
        x = create_outlier_sample(64, 64, 4096, BLOCK_SIZE, mean=0.0, std=1.0, scale_factor=scale_factor)
        _, scale_offset_distribution, scales = quantize_single(x, min_range=min_range, max_range=max_range, BLOCK_SIZE=BLOCK_SIZE)
        
        # Extract mantissa bits from FP8 scales
        scales_fp8 = scales.to(torch.float8_e4m3fn)
        scales_uint8 = scales_fp8.view(torch.uint8).flatten()
        mantissa_bits = (scales_uint8 & 0b00000111).long()
        offsets = scale_offset_distribution.flatten().long()
        
        # Create 2D histogram
        num_mantissa_vals = 8
        num_offset_vals = max_range - min_range + 1
        hist_2d = torch.zeros((num_mantissa_vals, num_offset_vals), device='cuda:0')
        
        for i in range(num_mantissa_vals):
            for j in range(num_offset_vals):
                offset_val = min_range + j
                mask = (mantissa_bits == i) & (offsets == offset_val)
                hist_2d[i, j] = mask.sum()
        
        # Normalize to get probabilities
        hist_2d = hist_2d / hist_2d.sum()
        
        im = axes[idx].imshow(hist_2d.cpu().numpy(), aspect='auto', origin='lower', 
                              extent=[min_range, max_range + 1, 0, 8], cmap='viridis')
        axes[idx].set_xlabel('offset from vlim scale')
        axes[idx].set_ylabel('original vlim scale mantissa')
        axes[idx].set_title(f'Scale Factor={scale_factor}')
        fig.colorbar(im, ax=axes[idx])
    
    plt.tight_layout()
    plt.savefig(f'scale_offset_2d_outliers_BLOCK_SIZE_{BLOCK_SIZE}.png', dpi=150)
    print(f"2D outlier histograms saved to scale_offset_2d_outliers_BLOCK_SIZE_{BLOCK_SIZE}.png")
    plt.show()

