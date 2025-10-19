import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
import sys
sys.path.append('/workspace/fp4_attn/fast_fp4')
import nvfp4sim

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
def quantize_to_fp4_packed_nosearch(x: tl.tensor, S_BLOCK_SIZE: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    max_abs_x = tl.max(tl.abs(x), axis=-1) * (1.0 / 6.0)
    xscale_f8 = max_abs_x.to(tl.float8e4nv, fp_downcast_rounding="rtne")[:, None]
    xscale = xscale_f8.to(tl.float32)
    xscale_inv = tl.where(xscale == 0, 0.0, 1.0 / xscale)
    x = x * xscale_inv
    # Quantize
    x_reshaped = x.reshape((S_BLOCK_SIZE * 8, 2)).to(tl.uint32, bitcast=True)
    x_scan = tl.associative_scan(x_reshaped, axis=-1, combine_fn=f32_to_f4_pair)
    _, quantized = tl.split(x_scan)
    quantized = quantized.to(tl.uint8).reshape((S_BLOCK_SIZE, 8))
    return quantized, xscale_f8

@triton.jit
def quantize_to_fp4_single_nosearch(x: tl.tensor, S_BLOCK_SIZE: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    max_abs_x = tl.max(tl.abs(x), axis=-1) * (1.0 / 6.0)
    xscale_f8 = max_abs_x.to(tl.float8e4nv, fp_downcast_rounding="rtne")[:, None]
    xscale = xscale_f8.to(tl.float32)
    xscale_inv = tl.where(xscale == 0, 0.0, 1.0 / xscale)
    x = x * xscale_inv
    quantized = f32_to_f4_single(x)
    return quantized, xscale_f8

@triton.jit
def pack_quantized_single(x, y):
    return x | (y << 4)

@triton.jit
def quantize_to_fp4_packed_search(x: tl.tensor, S_BLOCK_SIZE: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    best_quantized, best_xscale = quantize_to_fp4_single_search(x, S_BLOCK_SIZE, BLOCK_SIZE)
    best_quantized = best_quantized.reshape((S_BLOCK_SIZE * 8 , 2))
    best_quantized = tl.associative_scan(best_quantized, axis=-1, combine_fn=pack_quantized_single)
    _, quantized = tl.split(best_quantized)
    quantized = quantized.reshape((S_BLOCK_SIZE, 8))
    return quantized, best_xscale



@triton.jit
def quantize_to_fp4_single_search(x: tl.tensor, S_BLOCK_SIZE: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    max_abs_x = tl.max(tl.abs(x), axis=-1) * (1.0 / 6.0)
    xscale_f8 = max_abs_x.to(tl.float8e4nv, fp_downcast_rounding="rtne")[:, None]

    best_loss = tl.full((S_BLOCK_SIZE,), float("inf"), dtype=tl.float32)
    best_quantized = tl.zeros((S_BLOCK_SIZE, 16), dtype=tl.uint8)
    best_xscale = tl.zeros((S_BLOCK_SIZE,1), dtype=tl.float8e4nv)
    float_val = float(1)
    for offset_val in range(-7, 8, 1):
        offset = (offset_val).to(tl.float32)
        val = tl.full((S_BLOCK_SIZE, 1), 0.0, dtype=tl.float8e4nv)
        xscale = (xscale_f8 + val).to(tl.float32)
        xscale_inv = tl.where(xscale == 0, 0.0, 1.0 / xscale)
        quantized = f32_to_f4_single(x * xscale_inv)
        loss = (x - f4_to_f32_single(quantized) * xscale)
        loss = tl.sum(loss * loss, axis=-1)
        better = loss < best_loss
        best_loss = tl.where(better, loss, best_loss)
        best_quantized = tl.where(better[:, None], quantized.to(tl.uint8), best_quantized)
        new_xscale_fp8 = (xscale).to(tl.float8e4nv)
        best_xscale = tl.where(better[:, None], new_xscale_fp8, best_xscale)
    return best_quantized, best_xscale

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
    scales_output: tl.tensor,
    stride_bhz_input: tl.int32, stride_seq_input: tl.int32,
    stride_bhz_output: tl.int32, stride_seq_output: tl.int32,
    stride_bhz_output_sf: tl.int32, stride_seq_output_sf: tl.int32,
    seq_len: tl.int32,
    S_BLOCK_SIZE: tl.constexpr,
    D_BLOCK_SIZE: tl.constexpr,
    search: tl.constexpr,
    packed: tl.constexpr,
): 
    off_blk = tl.program_id(0)
    off_s = tl.program_id(1)
    offset = off_blk * stride_bhz_input

    if packed:
        SCALED_D_BLOCK_SIZE: tl.constexpr = D_BLOCK_SIZE // 2
    else:
        SCALED_D_BLOCK_SIZE: tl.constexpr = D_BLOCK_SIZE

    s_block = tl.arange(0, S_BLOCK_SIZE)[:, None] + off_s * S_BLOCK_SIZE
    d_block = tl.arange(0, D_BLOCK_SIZE)
    mask = s_block < seq_len
    block = s_block * stride_seq_input + d_block

    vals = tl.load(input + offset + block, mask=mask)
    if packed:
        if search:
            quantized_vals, scales = quantize_to_fp4_packed_search(vals, S_BLOCK_SIZE, D_BLOCK_SIZE)
        else:
            quantized_vals, scales = quantize_to_fp4_packed_nosearch(vals, S_BLOCK_SIZE, D_BLOCK_SIZE)
    else:
        if search:
            quantized_vals, scales = quantize_to_fp4_single_search(vals, S_BLOCK_SIZE, D_BLOCK_SIZE)
        else:
            quantized_vals, scales = quantize_to_fp4_single_nosearch(vals, S_BLOCK_SIZE, D_BLOCK_SIZE)

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


def quantize(x: torch.tensor, packed: bool = True, search: bool = False):
    b, h, n, d = x.shape
    BLOCK_SIZE = 16
    assert d % BLOCK_SIZE == 0

    x = x.reshape(b * h, n * (d // BLOCK_SIZE), BLOCK_SIZE) 
    rb, rs, rn = x.shape
    scales = torch.empty((rb, rs, 1), dtype=torch.float8_e4m3fn, device=x.device)
    if packed:
        quantized = torch.empty((rb, rs, rn // 2), dtype=torch.uint8, device=x.device)
    else:
        quantized = torch.empty((rb, rs, rn), dtype=torch.uint8, device=x.device)
    
    grid = lambda META: (rb, triton.cdiv(rs, META["S_BLOCK_SIZE"]), 1)
    with torch.cuda.device(x.device.index):
        quant_kernel[grid](
            x, quantized, scales,
            x.stride(0), x.stride(1),
            quantized.stride(0), quantized.stride(1),
            scales.stride(0), scales.stride(1),
            seq_len=rs,
            D_BLOCK_SIZE=BLOCK_SIZE,
            search=search,
            packed=packed,
        )

    if packed: 
        quantized = quantized.reshape(b, h, n, d // 2)
        scales = scales.reshape(b, h, n, d // BLOCK_SIZE)
    else:
        quantized = quantized.reshape(b, h, n, d)
        scales = scales.reshape(b, h, n, d // BLOCK_SIZE)
    return quantized, scales

def comparison(x, search=False):
    """Minimal example of single nvfp4 quantization without global scaling factor.
    Follows the exact same reshaping pattern as the quantize kernel."""
    BLOCK_SIZE = 16
    
    # Expect 4D input: (b, h, n, d)
    assert x.dim() == 4, "Input must be 4D: (b, h, n, d)"
    b, h, n, d = x.shape
    assert d % BLOCK_SIZE == 0, f"d must be divisible by {BLOCK_SIZE}"
    
    x = x.to(torch.float32)
    
    # Reshape like the kernel: (b, h, n, d) -> (b*h, n*(d//BLOCK_SIZE), BLOCK_SIZE)
    x = x.reshape(b * h, n * (d // BLOCK_SIZE), BLOCK_SIZE)
    
    # Flatten to (M, BLOCK_SIZE) for nvfp4sim where M = b*h*n*(d//BLOCK_SIZE)
    M = b * h * n * (d // BLOCK_SIZE)
    x_flat = x.reshape(M, BLOCK_SIZE).contiguous()
    
    # Allocate storage for quantized data and scales
    quantized_data = torch.empty(M, dtype=torch.int64, device=x.device)
    scales = torch.empty(M, dtype=torch.float8_e4m3fn, device=x.device)
    
    # Quantize using nvfp4sim
    if search:
        nvfp4sim.f32_to_nvf4(quantized_data, scales, x_flat)
    else:
        nvfp4sim.f32_to_nvf4_nosearch(quantized_data, scales, x_flat)
    
    # Reinterpret int64 as uint8 (8 bytes per int64, matching BLOCK_SIZE//2)
    quantized_data = quantized_data.view(torch.uint8)  # (M,) int64 -> (M*8,) uint8
    
    # Reshape to match kernel intermediate format: (b*h, n*(d//BLOCK_SIZE), BLOCK_SIZE//2)
    quantized_data = quantized_data.view(b * h, n * (d // BLOCK_SIZE), BLOCK_SIZE // 2)
    scales = scales.view(b * h, n * (d // BLOCK_SIZE), 1)
    
    # Final reshape to match kernel output: (b, h, n, d//2) and (b, h, n, d//BLOCK_SIZE)
    quantized_data = quantized_data.reshape(b, h, n, d // 2)
    scales = scales.reshape(b, h, n, d // BLOCK_SIZE)
    
    return quantized_data, scales

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],  # argument names to use as an x-axis for the plot
        x_vals=[128 * i for i in range(2, 100)],  # different possible values for `x_name`
        line_arg='provider',  # argument name whose value corresponds to a different line in the plot
        line_vals=['triton', 'nvfp4sim'],  # possible values for `line_arg``
        line_names=["Triton", "Nvfp4sim"],  # label name for the lines
        styles=[('blue', '-'), ('green', '-')],  # line styles
        ylabel="GB/s",  # label name for the y-axis
        plot_name="quantize-performance",  # name for the plot. Used also as a file name for saving the plot.
        args={'B': 1, 'H': 8, 'D': 128},  # values for function arguments not in `x_names` and `y_name`
    ))
def benchmark(B, H, N, D, provider):
    x = torch.randn(B, H, N, D, device=torch.device('cuda:0'), dtype=torch.float32)
    if provider == 'triton':
        ms = triton.testing.do_bench(lambda: quantize(x, search=False))
    if provider == 'nvfp4sim':
        ms = triton.testing.do_bench(lambda: comparison(x))
    gbps = lambda ms: 2 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms)

def unit_test():
    tests = 1
    n = torch.arange(128, 512, 1)
    dim = [64, 128]
    B, H = 1, 8
    for val in n:
        for d in dim:
            for test in range(tests):
                x = torch.randn(B, H, val, d, device=torch.device('cuda:0'), dtype=torch.float32).normal_()
                quantized_triton, scales_triton = quantize(x)
                quantized_nvfp4, scales_nvfp4 = comparison(x)
                assert torch.all(quantized_triton == quantized_nvfp4), f"{val} {d} {test}"
                assert torch.all(scales_triton == scales_nvfp4), f"{val} {d} {test}"
    print("Unit test passed")


if __name__ == "__main__":
    x = torch.randn(1, 1, 16, 16, device=torch.device('cuda:0'), dtype=torch.float32).normal_()
    
    # # Test Triton quantize function
    # quantized_triton, scales_triton = quantize(x)
    # print("Triton quantize:")
    # print("Quantized shape:", quantized_triton.shape)
    # print("Scales shape:", scales_triton.shape)
    # print("Quantized:", quantized_triton)
    # print("Scales:", scales_triton)
    
    # # Test comparison function (nvfp4sim-based)
    # print("\nComparison (nvfp4sim):")
    # quantized_nvfp4, scales_nvfp4 = comparison(x, search=False)
    # print("Quantized shape:", quantized_nvfp4.shape)
    # print("Scales shape:", scales_nvfp4.shape)
    # print("Quantized:", quantized_nvfp4)
    # print("Scales:", scales_nvfp4)

    # assert torch.all(quantized_triton == quantized_nvfp4)
    # assert torch.all(scales_triton == scales_nvfp4)

    quantized_triton, scales_triton = quantize(x, search=True)
    quantized_nvfp4, scales_nvfp4 = comparison(x, search=True)

    print("Triton quantize with search:")
    print("Quantized shape:", quantized_triton.shape)
    print("Scales shape:", scales_triton.shape)
    print("Quantized:", quantized_triton)
    print("Scales:", scales_triton)

    print("Nvfp4sim quantize with search:")
    print("Quantized shape:", quantized_nvfp4.shape)
    print("Scales shape:", scales_nvfp4.shape)
    print("Quantized:", quantized_nvfp4)
    print("Scales:", scales_nvfp4)

    assert torch.all(scales_triton == scales_nvfp4)
    assert torch.all(quantized_triton == quantized_nvfp4)

    # benchmark.run(show_plots=True, print_data=True, save_path='/workspace/fp4_attn/quant_kernel')
    # unit_test()