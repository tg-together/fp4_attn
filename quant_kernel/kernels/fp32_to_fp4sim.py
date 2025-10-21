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
def quantize_to_fp4_single_search(x: tl.tensor, S_BLOCK_SIZE: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    max_abs_x = tl.maximum(tl.max(tl.abs(x), axis=-1), 1e-10) * (1.0 / 6.0)
    max_abs_x = tl.clamp(max_abs_x, 1e-10 / 6.0, 448.0)
    xscale_f8 = max_abs_x.to(tl.float8e4nv, fp_downcast_rounding="rtne")[:, None]
    best_loss = tl.full((S_BLOCK_SIZE,), float("inf"), dtype=tl.float32)
    best_quantized = tl.zeros((S_BLOCK_SIZE, 16), dtype=tl.uint8)
    best_xscale = tl.zeros((S_BLOCK_SIZE,1), dtype=tl.float8e4nv)
    for offset_val in range(-7, 9, 1):
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
    stride_bhz_input: tl.int32, stride_seq_input: tl.int32,
    stride_bhz_output: tl.int32, stride_seq_output: tl.int32,
    seq_len: tl.int32,
    S_BLOCK_SIZE: tl.constexpr,
    D_BLOCK_SIZE: tl.constexpr,
    search: tl.constexpr,
): 
    off_blk = tl.program_id(0)
    off_s = tl.program_id(1)
    offset = off_blk * stride_bhz_input

    s_block = tl.arange(0, S_BLOCK_SIZE)[:, None] + off_s * S_BLOCK_SIZE
    d_block = tl.arange(0, D_BLOCK_SIZE)
    mask = s_block < seq_len
    block = s_block * stride_seq_input + d_block

    vals = tl.load(input + offset + block, mask=mask, other=0.0)
    if search:
        quantized_vals, scales = quantize_to_fp4_single_search(vals, S_BLOCK_SIZE, D_BLOCK_SIZE)
        quantized_vals = quantized_vals.to(dtype=tl.uint32)
    else:
        quantized_vals, scales = quantize_to_fp4_single_nosearch(vals, S_BLOCK_SIZE, D_BLOCK_SIZE)
    
    reconstructed_vals = dequantize_to_f32_single(quantized_vals, scales)

    q_offset = off_blk * stride_bhz_output 
    sq_block = tl.arange(0, S_BLOCK_SIZE)[:, None] + off_s * S_BLOCK_SIZE
    dq_block = tl.arange(0, D_BLOCK_SIZE)
    mask = sq_block < seq_len
    block = sq_block * stride_seq_output + dq_block
    tl.store(output + q_offset + block, reconstructed_vals, mask=mask)


def quantize_single(x: torch.tensor, search: bool = False):
    b, h, n, d = x.shape
    BLOCK_SIZE = 16
    assert d % BLOCK_SIZE == 0

    x = x.reshape(b * h, n * (d // BLOCK_SIZE), BLOCK_SIZE) 
    rb, rs, rn = x.shape
    output = torch.empty((rb, rs, rn), dtype=torch.float32, device=x.device)
    
    grid = lambda META: (rb, triton.cdiv(rs, META["S_BLOCK_SIZE"]), 1)
    with torch.cuda.device(x.device.index):
        quant_kernel[grid](
            x, output,
            x.stride(0), x.stride(1),
            output.stride(0), output.stride(1),
            seq_len=rs,
            D_BLOCK_SIZE=BLOCK_SIZE,
            search=search,
        )

    reconstructed_vals = output.reshape(b, h, n, d)
    return reconstructed_vals

def comparison(x, search=False):
    """Single nvfp4 quantization without global scaling factor.
    Follows the simulation approach from fp4_quant_utils.py but accepts 4D input.
    Ignores global_sf for now - just does basic quantize/dequantize."""
    BLOCK_SIZE = 16
    
    # Expect 4D input: (b, h, n, d)
    assert x.dim() == 4, "Input must be 4D: (b, h, n, d)"
    b, h, n_orig, d_orig = x.shape
    
    x = x.to(torch.float32)
    
    # Reshape to (b*h, n, d) for easier processing
    x = x.reshape(b * h, n_orig, d_orig)
    
    # Pad columns to be divisible by BLOCK_SIZE
    pad_cols = (BLOCK_SIZE - d_orig % BLOCK_SIZE) % BLOCK_SIZE
    # Pad rows to be even (required by nvfp4sim)
    pad_rows = n_orig % 2
    
    if pad_cols != 0 or pad_rows != 0:
        x = torch.nn.functional.pad(x, (0, pad_cols, 0, pad_rows), value=0.0)
    
    bh, n, d = x.shape
    x = x.contiguous()
    
    # Reshape to (bh * n * (d // BLOCK_SIZE), BLOCK_SIZE) for nvfp4sim
    x = x.view(bh * n * (d // BLOCK_SIZE), BLOCK_SIZE)
    M = x.shape[0]
    
    # Allocate storage for quantized data and scales
    quantized_data = torch.empty(M, dtype=torch.int64, device=x.device)
    scales = torch.empty(M, dtype=torch.float8_e4m3fn, device=x.device)
    
    # Quantize using nvfp4sim
    if search:
        nvfp4sim.f32_to_nvf4(quantized_data, scales, x)
    else:
        nvfp4sim.f32_to_nvf4_nosearch(quantized_data, scales, x)
    
    # Dequantize back to f32
    reconstructed_f32 = torch.empty(M, BLOCK_SIZE, dtype=torch.float32, device=x.device)
    nvfp4sim.nvf4_to_f32(reconstructed_f32, quantized_data, scales)
    
    # Reshape back to (bh, n, d)
    reconstructed_f32 = reconstructed_f32.view(bh, n, d)
    
    # Remove padding
    if n != n_orig or d != d_orig:
        reconstructed_f32 = reconstructed_f32[:, :n_orig, :d_orig]
    
    # Reshape back to (b, h, n, d)
    reconstructed_f32 = reconstructed_f32.reshape(b, h, n_orig, d_orig)
    
    return reconstructed_f32

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],  # argument names to use as an x-axis for the plot
        x_vals=[128 * i for i in range(2, 100)],  # different possible values for `x_name`
        line_arg='provider',  # argument name whose value corresponds to a different line in the plot
        line_vals=['triton', 'nvfp4sim'],  # possible values for `line_arg``
        line_names=["Triton", "Nvfp4sim"],  # label name for the lines
        styles=[('blue', '-'), ('green', '-')],  # line styles
        ylabel="ms",  # label name for the y-axis
        plot_name="quantize-sim-performance-latency",  # name for the plot. Used also as a file name for saving the plot.
        args={'B': 1, 'H': 8, 'D': 128},  # values for function arguments not in `x_names` and `y_name`
    ))
def benchmark(B, H, N, D, provider):
    x = torch.randn(B, H, N, D, device=torch.device('cuda:0'), dtype=torch.float32)
    if provider == 'triton':
        ms = triton.testing.do_bench(lambda: quantize_single(x, search=False))
    if provider == 'nvfp4sim':
        ms = triton.testing.do_bench(lambda: comparison(x, search=False))
    gbps = lambda ms: 2 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
    return ms
    # return gbps(ms)

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],  # argument names to use as an x-axis for the plot
        x_vals=[128 * i for i in range(2, 100)],  # different possible values for `x_name`
        line_arg='provider',  # argument name whose value corresponds to a different line in the plot
        line_vals=['triton', 'nvfp4sim'],  # possible values for `line_arg``
        line_names=["Triton", "Nvfp4sim"],  # label name for the lines
        styles=[('blue', '-'), ('green', '-')],  # line styles
        ylabel="ms",  # label name for the y-axis
        plot_name="quantize-sim-performance-search",  # name for the plot. Used also as a file name for saving the plot.
        args={'B': 1, 'H': 8, 'D': 128},  # values for function arguments not in `x_names` and `y_name`
    ))
def benchmark_search(B, H, N, D, provider):
    x = torch.randn(B, H, N, D, device=torch.device('cuda:0'), dtype=torch.float32)
    if provider == 'triton':
        ms = triton.testing.do_bench(lambda: quantize_single(x, search=True))
    if provider == 'nvfp4sim':
        ms = triton.testing.do_bench(lambda: comparison(x, search=True))
    gbps = lambda ms: 2 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms)

def unit_test(search=False):
    tests = 1
    n = torch.arange(128, 512, 1)
    dim = [64, 128]
    B, H = 1, 8
    for val in n:
        for d in dim:
            for test in range(tests):
                x = torch.randn(B, H, val, d, device=torch.device('cuda:0'), dtype=torch.float32).normal_()
                reconstructed_triton = quantize_single(x, search=search)
                reconstructed_nvfp4 = comparison(x, search=search)
                try:
                    assert torch.all(reconstructed_triton == reconstructed_nvfp4), f"{val} {d} {test} {torch.max(torch.abs(reconstructed_triton - reconstructed_nvfp4)).item()}"
                except AssertionError as e:
                    vals = reconstructed_triton != reconstructed_nvfp4
                    mse_triton = torch.mean((x[vals] - reconstructed_triton[vals]) ** 2)
                    mse_nvfp4 = torch.mean((x[vals] - reconstructed_nvfp4[vals]) ** 2)
                    assert mse_triton <= mse_nvfp4, f"{val} {d} {test} {torch.max(torch.abs(reconstructed_triton - reconstructed_nvfp4)).item()}"
    print("Unit test passed")


if __name__ == "__main__":
    x = torch.randn(1, 8, 163, 128, device=torch.device('cuda:0'), dtype=torch.float32).normal_()
    
    # Test Triton quantize function
    # reconstructed_triton = quantize_single(x, search=False)
    # print("Triton quantize:")
    # print("Reconstructed shape:", reconstructed_triton.shape)
    # print("Reconstructed:", reconstructed_triton)
    
    # # Test comparison function (nvfp4sim-based)
    # print("\nComparison (nvfp4sim):")
    # reconstructed_nvfp4 = comparison(x, search=False)
    # print("Reconstructed shape:", reconstructed_nvfp4.shape)
    # print("Reconstructed:", reconstructed_nvfp4)

    # # Check if results are close
    # print("\nAre results close?", torch.allclose(reconstructed_triton, reconstructed_nvfp4, rtol=1e-5, atol=1e-5))
    # print("Max difference:", torch.max(torch.abs(reconstructed_triton - reconstructed_nvfp4)).item())


    # benchmark.run(show_plots=True, print_data=True, save_path='/workspace/fp4_attn/quant_kernel')
    # unit_test()
    reconstructed_triton = quantize_single(x, search=True)
    print("Triton quantize:")
    print("Reconstructed shape:", reconstructed_triton.shape)
    print("Reconstructed:", reconstructed_triton)
    
    # Test comparison function (nvfp4sim-based)
    print("\nComparison (nvfp4sim):")
    reconstructed_nvfp4 = comparison(x, search=True)
    print("Reconstructed shape:", reconstructed_nvfp4.shape)
    print("Reconstructed:", reconstructed_nvfp4)

    # Check if results are close
    print("\nAre results close?", torch.allclose(reconstructed_triton, reconstructed_nvfp4, rtol=1e-5, atol=1e-5))
    print("Max difference:", torch.max(torch.abs(reconstructed_triton - reconstructed_nvfp4)).item())
    # benchmark_search.run(show_plots=True, print_data=True, save_path='/workspace/fp4_attn/quant_kernel')
    unit_test(search=True)