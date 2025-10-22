import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
import matplotlib.pyplot as plt

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
    best_quantized = tl.zeros((S_BLOCK_SIZE, 16), dtype=tl.uint8)
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
    return best_quantized, best_xscale, best_offset

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
    stride_bhz_input: tl.int32, stride_seq_input: tl.int32,
    stride_bhz_output: tl.int32, stride_seq_output: tl.int32,
    stride_bhz_output_sf: tl.int32, stride_seq_output_sf: tl.int32,
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


def quantize_single(x: torch.tensor, min_range: int = -7, max_range: int = 8):
    b, h, n, d = x.shape
    BLOCK_SIZE = 16
    assert d % BLOCK_SIZE == 0

    x = x.reshape(b * h, n * (d // BLOCK_SIZE), BLOCK_SIZE).contiguous()
    rb, rs, rn = x.shape
    output = torch.empty((rb, rs, rn), dtype=torch.float32, device=x.device)
    scale_offset_distribution = torch.empty((rb, rs, 1), dtype=torch.int8, device=x.device)
    
    grid = lambda META: (rb, triton.cdiv(rs, META["S_BLOCK_SIZE"]), 1)
    with torch.cuda.device(x.device.index):
        quant_kernel[grid](
            x, output, scale_offset_distribution,
            x.stride(0), x.stride(1),
            output.stride(0), output.stride(1),
            scale_offset_distribution.stride(0), scale_offset_distribution.stride(1),
            seq_len=rs,
            D_BLOCK_SIZE=BLOCK_SIZE,
            min_range=min_range,
            max_range=max_range,
        )

    reconstructed_vals = output.reshape(b, h, n, d)
    scale_offset_distribution = scale_offset_distribution.reshape(b, h, n, d // BLOCK_SIZE)
    return reconstructed_vals, scale_offset_distribution

if __name__ == "__main__":
    x = torch.randn(64, 64, 4096, 128, device=torch.device('cuda:0'), dtype=torch.float32).normal_()
    min_range, max_range = -15, 16
    reconstructed_vals, scale_offset_distribution = quantize_single(x, min_range=min_range, max_range=max_range)
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
    plt.savefig('scale_offset_probs.png', dpi=150)
    print("Probabilities saved to scale_offset_probs.png")
    plt.show()

